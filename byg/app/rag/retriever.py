"""检索层：结构化过滤 → 多路召回 → 分层打分 → 融合 → 预算裁剪。

**为什么是这个顺序**（每一环都有硬约束兜着）
--------------------------------------------
1. **结构化过滤先做**：时间窗（``trigger`` 里的 ``time_min between``）是
   **硬语义** —— 3~5 分钟该做的事，10 分钟才做不算数。
   英雄层还必须 ``my_hero`` 精确等于玩家英雄（``hero`` 未知时**只给分路层**，
   否则会把别的英雄的打法捞回来）。
   → 向量排序只能在过滤后的候选里排。
2. **多路召回**：
   * 稠密（语义：模型写的「我在河道跟人打架」≈「河道遭遇战该先占视野」）；
   * 稀疏（字符 bigram BM25：字面重合，中文无需分词器）；
   * 规则命中（query 命中这条打法的关键词/场景标签）。
3. **分层打分**：各路先各自排名，再按来源权重合成分 —— 规则命中最可信。
4. **融合**：Reciprocal Rank Fusion（RRF）。名次比绝对分稳，
   不同路的分数量纲不同，直接加会被某一路带偏。
5. **上下文预算裁剪**：同一模式的多支只留最相关的 n 支
   （**分支不能压平** —— 模型要指认「第几支成立」）。
6. **降级留痕**：召不回、走了兜底、裁掉了多少，都写进 ``degradation``。

借鉴：ALCE（稀疏+稠密两路检索）、同作者 xiagu 项目
（``app/knowledge/retriever.py`` 的 base score + context budget + degradation）。

**换到 Qdrant 之后变在哪**
--------------------------
稠密与稀疏两路**都进 Qdrant 了**（向量检索 + BM25 文本索引），
Python 这边不再把 4 077 个块读进内存算点积。

但**口径没变**：结构化过滤的最终裁决仍在 Python
（:meth:`Retriever._allowed`）—— Qdrant 的 payload filter 只是
**前置缩小范围**，多召回的候选一律再过一遍那道闸。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .embeddings import get_provider, provider_name
from .store import ChunkHit, ModeStore, chunk_doc

#: 各路的来源权重（分层打分用）。
#:
#: 规则命中最可信（query 里明说了场景），稠密次之（语义相近但可能时间不对），
#: 稀疏最弱（字面重合常常是巧合，比如都在说「小龙」）。
SOURCE_WEIGHTS = {
    "rule": 1.1,
    "dense": 1.0,
    "sparse": 0.9,
    "fallback": 0.5,
}

#: RRF 的常数。越大越「只看名次」，越小越「看重头部差距」。
RRF_K = 60.0

#: 召回落池：最终要 k 条，先取 k × expand_factor 条再裁剪。
DEFAULT_EXPAND_FACTOR = 4


@dataclass
class RetrievalResult:
    """一次检索的结果 + **降级留痕**。"""
    hits: List[ChunkHit] = field(default_factory=list)
    #: 走了哪几路（dense / sparse / rule / fallback）。
    routes: List[str] = field(default_factory=list)
    #: 降级说明：召回了多少、裁掉了多少、为什么退化。
    degradation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": [h.to_dict() for h in self.hits],
            "routes": list(self.routes),
            "degradation": dict(self.degradation),
        }


def _superseded_by_hero(doc: Dict[str, Any], hero: str) -> bool:
    """这条通则**有英雄专属版**，而玩家就是那个英雄。

    通则（分路层）规则的护栏写在 ``trigger.none`` 里：
    ``{"fact": "my_hero", "op": "in", "value": [这一批有专属版的英雄]}``。
    判定层（:mod:`app.recognition2.judge`）遇到它会把这条排除 ——
    「你有你自己的版本，这条通则不是给你的」。

    检索层**也要排掉它**，不然它占着召回名额，把**真正该用的**
    英雄专属版挤出 top-k（2026-11-02 熬隐那局实测：
    召回了 7 条发育路通则 → 护栏全排除 → 候选 0 条，
    而敖隐自己的 13 条专属打法一条都没进候选）。

    英雄不知道时**不排** —— 护栏那时也不成立（认不准 ≠ 有专属版）。
    """
    if not hero:
        return False
    payload = doc.get("payload") or {}
    trigger = payload.get("trigger") or {}
    for chk in (trigger.get("none") or []):
        if not isinstance(chk, dict):
            continue
        if str(chk.get("fact") or "") != "my_hero":
            continue
        op = str(chk.get("op") or "")
        if op not in ("==", "in"):
            continue
        value = chk.get("value")
        values = value if isinstance(value, (list, tuple, set)) else [value]
        if hero in [str(v) for v in values]:
            return True
    return False


class Retriever:
    """模式库检索器（向量与 BM25 都在 Qdrant 里）。"""
    def __init__(self, store: Optional[ModeStore] = None,
                 provider: Optional[Any] = None,
                 expand_factor: int = DEFAULT_EXPAND_FACTOR):
        self.store = store or ModeStore()
        self.provider = provider if provider is not None else get_provider()
        self.expand_factor = int(expand_factor)
        self._scan_cache: Optional[Tuple[str, List[Dict[str, Any]]]] = None

    def invalidate(self) -> None:
        """索引重建后要清缓存。"""
        self._scan_cache = None

    # -- 语料 ------------------------------------------------------------

    def _scan(self, *, game: str, version: str, lane: str, hero: str,
              lo: int, hi: int) -> List[Dict[str, Any]]:
        """符合硬约束的检索块（**规则命中那一路**要逐条看）。

        按库指纹缓存 —— 索引重建后指纹变了，自动失效。
        """
        key = "|".join([game, version, lane, hero, str(lo), str(hi)])
        fingerprint = self.store.fingerprint(game=game)
        cached = self._scan_cache
        if cached is not None and cached[0] == key + "#" + fingerprint:
            return cached[1]
        corpus = self.store.scan(game=game, version=version, lane=lane,
                                 hero=hero, lo=lo, hi=hi)
        self._scan_cache = (key + "#" + fingerprint, corpus)
        return corpus

    # -- 过滤 ------------------------------------------------------------

    @staticmethod
    def _allowed(doc: Dict[str, Any], *, game: str, version: str,
                 lane: str, hero: str, lo: int, hi: int) -> bool:
        """**硬约束** —— 结构化过滤，检索的第一道闸。

        规则与 :func:`app.rules.checklist.window_checklist` 一致，
        不许放宽：
        * 作废（retired）与非 active 的不算；
        * 英雄层必须英雄精确相等；**hero 未知时只给分路层**；
        * 时间窗必须相交。
        """
        if doc.get("retired"):
            return False
        if game and doc.get("game") and doc.get("game") != game:
            return False
        if version and doc.get("version") and doc.get("version") != version:
            return False
        if str(doc.get("status") or "active") != "active":
            return False

        layer = str(doc.get("layer") or "lane")
        if layer == "hero":
            if not hero:
                # **英雄不知道，就不给任何英雄层的打法。**
                return False
            if doc.get("hero") != hero:
                return False
        elif layer == "lane":
            if lane and doc.get("lane") and doc.get("lane") != lane:
                return False
            if _superseded_by_hero(doc, hero):
                # **这条通则有「英雄专属版」** —— 玩家就是那个英雄时，
                # 通则那条在判定层会被护栏排除（见 ``judge.py`` 的 ``when_none``）。
                #
                # 这里**在检索层就排掉**，不然它占着召回名额，把**真正该用的**
                # 英雄专属版挤出 top-k：2026-11-02 熬隐那局实测就是
                # 「召回了 7 条发育路通则 → 护栏全排除 → 候选 0 条」，
                # 而敖隐自己的 13 条专属打法一条都没进候选。
                return False
        else:
            # BP 之类其它层：这一轮不参与。
            return False

        d_lo = int(doc.get("window_lo") or 0)
        d_hi = int(doc.get("window_hi") or 0)
        if d_hi < lo or d_lo > hi:
            return False
        return True

    # -- 检索 ------------------------------------------------------------

    def search(self, query: str, *, game: str = "hok", version: str = "",
               lane: str = "", hero: str = "", lo: int = 0, hi: int = 60,
               top_k: int = 8) -> RetrievalResult:
        """检索与这一段相关的打法。

        ``query`` 是**模型自己写的自然语言**（「我在河道跟人打架，
        这会儿该干嘛」），不是关键词列表。

        **版本默认不过滤**（``version=""``）——
        早先这里写死 ``"S45"``，于是用户把模式库换成新版本号、
        重新「载入规则库」之后：判定按新版本走、检索按 S45 走，
        **一条都召不回**，而外面看着像「库里没有适用的打法」。
        检索块每次建库都是**整块重建**的，库里只有当前那一版，
        所以不过滤版本不会把旧版捞回来。
        """
        result = RetrievalResult()
        if not (query or "").strip():
            result.degradation = {"reason": "查询是空的", "candidates": 0}
            return result

        corpus = self._scan(game=game, version=version, lane=lane,
                            hero=hero, lo=lo, hi=hi)
        if not corpus:
            # **三种空得分开说** ——「连不上」「库是空的」「不匹配」
            # 该做的事完全不同：修连接 / 载入规则库 / 换英雄或分路。
            _stats = self.store.stats()
            _err = str(_stats.get("collection_error") or "").strip()
            if _err:
                reason = "打法库连不上（%s）" % _err
            elif not _stats.get("documents"):
                reason = "打法库是空的（还没载入规则库）"
            else:
                reason = "这一段没有适用的打法（时间窗/英雄不匹配）"
            result.degradation = {"reason": reason, "candidates": 0}
            return result
        result.degradation["candidates"] = len(corpus)

        retrieval_k = max(top_k * self.expand_factor, top_k)
        ranked: Dict[str, Dict[str, Any]] = {}
        by_id: Dict[str, Dict[str, Any]] = {
            str(c.get("chunk_id") or ""): c for c in corpus
        }

        def _put(chunk_id: str, route: str, raw_score: float) -> None:
            entry = ranked.setdefault(chunk_id, {"route_scores": {}, "ranks": {}})
            entry["route_scores"][route] = float(raw_score)

        def _keep(chunk_id: str, doc: Dict[str, Any]) -> bool:
            """**Python 侧的那道闸** —— 最终裁决。

            Qdrant 的 payload filter 已经按同一套硬约束筛过一遍，
            但这里**再判一次**：护栏（通则 vs 英雄专属版）Qdrant 表达不了，
            而放宽一点点就会把别的英雄的打法捞回来。
            """
            return self._allowed(doc, game=game, version=version,
                                 lane=lane, hero=hero, lo=lo, hi=hi)

        # **过了那道闸的候选** —— 护栏（通则 vs 英雄专属版）Qdrant 表达不了，
        # 只能在这里排掉。
        allowed = {cid for cid, chunk in by_id.items()
                   if _keep(cid, chunk_doc(chunk))}
        result.degradation["candidates"] = len(allowed)
        if not allowed:
            result.degradation["reason"] = "这一段没有适用的打法（时间窗/英雄不匹配）"
            return result

        # ① 稠密召回（Qdrant 向量检索）
        routes: List[str] = []
        dense_error = ""
        try:
            vec = self.provider.embed_query(query)
        except Exception as exc:                          # noqa: BLE001
            # 以前这里是**静默吞掉**的：查询向量拿不到，稠密路直接不参与，
            # 检索照样返回结果，只是悄悄少了一条路。
            # 现在把原因记下来 —— 它会进 ``degradation``，能被报告看见。
            vec = None
            dense_error = "%s: %s" % (type(exc).__name__, exc)
        if vec is not None:
            try:
                hits = self.store.search_dense(
                    vec, game=game, version=version, lane=lane, hero=hero,
                    lo=lo, hi=hi, limit=retrieval_k)
                routes.append("dense")
                for chunk_id, score, payload in hits:
                    if not chunk_id:
                        continue
                    if not _keep(chunk_id, chunk_doc(payload)):
                        continue
                    _put(chunk_id, "dense", score)
                    by_id.setdefault(chunk_id, {
                        "chunk_id": chunk_id,
                        "mode_id": payload.get("mode_id") or "",
                        "branch_no": payload.get("branch_no") or 0,
                        "text": payload.get("text") or "",
                    })
            except Exception as exc:                      # noqa: BLE001
                dense_error = "%s: %s" % (type(exc).__name__, exc)

        # ② 稀疏召回（Qdrant 的 BM25 文本索引）
        sparse_error = ""
        try:
            hits = self.store.search_sparse(
                query, game=game, version=version, lane=lane, hero=hero,
                lo=lo, hi=hi, limit=retrieval_k)
            routes.append("sparse")
            for chunk_id, score, payload in hits:
                if not chunk_id:
                    continue
                if not _keep(chunk_id, chunk_doc(payload)):
                    continue
                _put(chunk_id, "sparse", score)
                by_id.setdefault(chunk_id, {
                    "chunk_id": chunk_id,
                    "mode_id": payload.get("mode_id") or "",
                    "branch_no": payload.get("branch_no") or 0,
                    "text": payload.get("text") or "",
                })
        except Exception as exc:                          # noqa: BLE001
            # **稀疏路与稠密路分开记** —— 稠密路走通了、只有稀疏路死掉时，
            # 把原因写进 ``dense_error`` 会让报告说「向量检索没走通」，
            # 而向量路其实是通的。
            sparse_error = "%s: %s" % (type(exc).__name__, exc)

        # ③ 规则命中（关键词/场景标签）
        #
        # **只在通过那道闸的候选里找** —— 护栏（通则 vs 英雄专属版）
        # Qdrant 的 payload filter 表达不了，而规则命中这一路是「稳」的一路：
        # 语义相近但时间不对的打法，它能顶上来；护栏排掉的通则，它不能顶上来。
        route_hits = self._rule_hits(query, [by_id[cid] for cid in allowed])
        if route_hits:
            routes.append("rule")
            for chunk_id, score in route_hits.items():
                _put(chunk_id, "rule", score)

        if not ranked:
            result.degradation["reason"] = "没召回到相关内容"
            result.routes = routes
            if dense_error:
                result.degradation["dense_error"] = dense_error
            if sparse_error:
                result.degradation["sparse_error"] = sparse_error
            return result

        # ④ 分层打分 + RRF 融合
        fused = self._fuse(ranked, routes)
        order = sorted(fused.items(), key=lambda kv: kv[1]["score"], reverse=True)
        top = order[:retrieval_k]

        # 原始规则（模型要拿它去对画面）—— 从文档集合里取。
        docs = self.store.documents_by_ids(
            [str(by_id.get(cid, {}).get("mode_id") or "") for cid, _ in top])

        hits_out: List[ChunkHit] = []
        for chunk_id, info in top:
            chunk = by_id.get(chunk_id) or {}
            doc = docs.get(str(chunk.get("mode_id") or "")) or {}
            payload = doc.get("payload") or {}
            hits_out.append(ChunkHit(
                chunk_id=chunk_id,
                mode_id=str(chunk.get("mode_id") or ""),
                branch_no=int(chunk.get("branch_no") or 0),
                text=str(chunk.get("text") or ""),
                score=float(info["score"]),
                source="+".join(sorted(info["route_scores"].keys())),
                title=str(doc.get("title") or chunk.get("title") or ""),
                lane=str(doc.get("lane") or chunk.get("lane") or ""),
                hero=str(doc.get("hero") or chunk.get("hero") or ""),
                window=(int(doc.get("window_lo") or chunk.get("window_lo") or 0),
                        int(doc.get("window_hi") or chunk.get("window_hi") or 0)),
                payload=payload if isinstance(payload, dict) else {},
            ))

        # ⑤ 上下文预算裁剪（同一模式只留最相关的几支）
        selected, dropped = self._budget(hits_out, top_k)
        result.hits = selected
        result.routes = routes
        result.degradation.update({
            "candidates": len(corpus),
            "recalled": len(hits_out),
            "selected": len(selected),
            "dropped_by_budget": dropped,
            "provider": provider_name(self.provider),
        })
        if dropped:
            result.degradation["reason"] = "预算裁掉了 %d 条（分支只留最相关的几支）" % dropped
        if dense_error:
            # 稠密路没参与 —— 写清楚原因（额度/维度不匹配/索引过期）。
            # 少一条路还能用，但用户有权知道为什么。
            result.degradation["dense_error"] = dense_error
        if sparse_error:
            # 稀疏路没参与 —— **与稠密路分开记**（向量路可能好好的）。
            result.degradation["sparse_error"] = sparse_error
        return result

    # -- 融合与裁剪 ------------------------------------------------------

    @staticmethod
    def _fuse(ranked: Dict[str, Dict[str, Any]], routes: Sequence[str]
              ) -> Dict[str, Dict[str, Any]]:
        """分层打分 + RRF 融合。

        先按来源权重把「这一路的原始分」折成**来源内名次分**，
        再用 RRF 把各路的名次合并 —— 量纲不同直接相加会被某一路带偏。
        """
        out: Dict[str, Dict[str, Any]] = {}
        for route in routes:
            scored = [(cid, info["route_scores"][route])
                      for cid, info in ranked.items()
                      if route in info["route_scores"]]
            if not scored:
                continue
            scored.sort(key=lambda kv: kv[1], reverse=True)
            for rank, (cid, raw) in enumerate(scored, start=1):
                weight = SOURCE_WEIGHTS.get(route, 1.0)
                rrf = weight / (RRF_K + rank)
                entry = out.setdefault(cid, {"score": 0.0, "route_scores": {}})
                entry["score"] += rrf
                entry["route_scores"][route] = raw
        return out

    @staticmethod
    def _budget(hits: List[ChunkHit], top_k: int
                ) -> Tuple[List[ChunkHit], int]:
        """上下文预算：同一模式**最多留 2 支**（模式级块 + 最相关的一支）。

        分支不能压平 —— 但也不能把一个节点的 3 支全塞进提示词。
        留「模式级块 + 最相关的那一支」，其余丢掉并计数。
        """
        if top_k <= 0:
            return [], len(hits)
        per_mode: Dict[str, List[ChunkHit]] = {}
        for h in hits:
            per_mode.setdefault(h.mode_id, []).append(h)

        selected: List[ChunkHit] = []
        dropped = 0
        for mode_id, group in per_mode.items():
            group.sort(key=lambda h: (h.branch_no == 0, -h.score))
            keep = group[:2] if len(group) > 2 else group
            dropped += len(group) - len(keep)
            selected.extend(keep)
        selected.sort(key=lambda h: -h.score)
        if len(selected) > top_k:
            dropped += len(selected) - top_k
            selected = selected[:top_k]
        return selected, dropped

    # -- 规则命中 --------------------------------------------------------

    @staticmethod
    def _rule_hits(query: str, corpus: Sequence[Dict[str, Any]]
                   ) -> Dict[str, float]:
        """关键词/场景标签命中。

        模型在 query 里写的场景词（龙坑、河道、对线、回城、支援…）
        命中这条打法的标题或前提 → 给一个**加权分**。

        这一路不是「聪明」，是「稳」：语义相近但时间不对的打法，
        场景词命中能把它顶上来。
        """
        scene_words = [
            "龙", "暴君", "主宰", "先锋", "野怪", "buff", "河道", "草丛",
            "塔", "高地", "水晶", "兵线", "补刀", "对线", "支援", "游走",
            "回城", "撤退", "开团", "团战", "gank", "抓单", "发育",
        ]
        lowered = (query or "").lower()
        hits: Dict[str, float] = {}
        for chunk in corpus:
            chunk_id = str(chunk.get("chunk_id") or "")
            if not chunk_id:
                continue
            hay = " ".join([
                str(chunk.get("title") or ""),
                str(chunk.get("premise") or ""),
                str(chunk.get("action") or ""),
                str(chunk.get("text") or ""),
            ]).lower()
            score = 0.0
            for word in scene_words:
                if word.lower() in lowered and word.lower() in hay:
                    score += 1.0
            if score > 0:
                hits[chunk_id] = score
        return hits


def search_modes(query: str, *, game: str = "hok", version: str = "S45",
                 lane: str = "", hero: str = "", lo: int = 0, hi: int = 60,
                 top_k: int = 8) -> RetrievalResult:
    """便捷入口：建一个检索器查一次。"""
    return Retriever().search(query, game=game, version=version, lane=lane,
                              hero=hero, lo=lo, hi=hi, top_k=top_k)
