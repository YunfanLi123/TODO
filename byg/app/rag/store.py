"""模式库存储层：**Qdrant**（文档 + 检索块 + 向量 + BM25 文本索引）。

和上一版的差别
--------------
上一版（SQLite ``mode_library.db``）里，数据库只是个文件柜：
向量以 ``LargeBinary`` 存着，检索时 4 077 个块**全部读进内存**，
过滤在 Python 里做、相似度也在 Python 里算。

这一版把**检索本身**交给 Qdrant：

* **文档**（一条攻略）→ ``mode_documents`` 集合（纯 payload，不带向量）；
* **检索块**（模式 × 分支）→ ``mode_chunks`` 集合（带稠密向量 + BM25 文本索引）；
* **硬过滤**（时间窗 / 英雄 / 分路 / 作废）→ Qdrant 的 payload filter；
* **稠密召回** → Qdrant 向量检索；
* **稀疏召回** → Qdrant 的 BM25 文本索引（词是我们自己切的，见
  :mod:`app.rag.sparse`）。

**口径不变**：Python 侧的 :meth:`app.rag.retriever.Retriever._allowed`
仍是**最终裁决** —— Qdrant 的过滤只是**前置缩小范围**，
多召回的候选一律再过一遍那道闸。

**真源不变**：文档的原始记录（``payload``）原样存在 payload 里，
「DB → JSON」仍然是可逆的。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import qdrant_client as qc
from .sparse import pretokenized

#: 元信息集合（一个点）：索引的版本、嵌入档、条数。
META_COLLECTION = "mode_index_meta"
#: 元信息点的 id —— **必须是 UUID**（Qdrant 只收 UUID 或无符号整数，
#: 别的格式直接报错，而报错会被吞掉 → 索引看着建好了，却是「没有元信息」）。
META_POINT_ID = "00000000-0000-0000-0000-000000000001"

#: 上一次写元信息失败的原因（空串 = 没失败）。
#:
#: 为什么要有它：写失败如果静默吞掉，**索引看着是建好的**，
#: 但「这批向量是哪种嵌入算的」没人知道 —— 下次检索会拿错档的查询向量
#: 去比库向量，相似度全是噪声，而外面看不出任何异常。
last_meta_error: str = ""


def content_sha256(payload: Dict[str, Any]) -> str:
    """文档内容封条。内容变了索引就得重建。"""
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def branch_text(doc: Dict[str, Any], branch: Optional[Dict[str, Any]],
                branch_no: int) -> str:
    """一条检索块的文本。

    刻意**逐字**用攻略原文：模型要拿画面去对的就是原话，
    我们改写过的转述它对不上（模式库 README 的硬约定）。
    """
    parts: List[str] = []
    title = str(doc.get("title") or "").strip()
    if title:
        parts.append(title)
    if branch is not None:
        if branch_no:
            parts.append("第 %d 支" % branch_no)
        for key in ("premise", "action", "outcome"):
            val = str(branch.get(key) or "").strip()
            if val:
                parts.append(val)
    else:
        for key in ("premise", "action", "outcome"):
            val = str(doc.get(key) or "").strip()
            if val:
                parts.append(val)
    source = doc.get("source") or {}
    quoted = str(source.get("quoted") or "").strip()
    if quoted:
        parts.append("原文：" + quoted)
    return "\n".join(parts)


def _branches_of(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """取分支数组。

    兼容两种形状：规范化后的字典用 ``branches``，
    数据库行读回来时用 ``branches_json``（字符串）。
    """
    raw = doc.get("branches")
    if raw is None and doc.get("branches_json"):
        raw = _loads(doc.get("branches_json"), [])
    if not isinstance(raw, list):
        return []
    return [b for b in raw if isinstance(b, dict)]


def doc_view(doc: Dict[str, Any]) -> Dict[str, Any]:
    """文档 → **检索块上冗余的过滤键**。

    Qdrant 过滤要的是 payload 上的字段，而块与文档是两个集合 ——
    不冗余就得 join，检索时每条都去查一遍文档，慢且没必要。

    这里只放**过滤与判定用得到**的键：游戏/版本/层/分路/英雄/时间窗/
    状态/作废 + 护栏（``trigger``）+ 规则命中那路要的标题与正文。
    """
    payload = doc.get("payload") or {}
    trigger = payload.get("trigger") if isinstance(payload, dict) else None
    return {
        "game": doc.get("game") or "",
        "version": doc.get("version") or "",
        "layer": doc.get("layer") or "lane",
        "lane": doc.get("lane") or "",
        "hero": doc.get("hero") or "",
        "window_lo": int(doc.get("window_lo") or 0),
        "window_hi": int(doc.get("window_hi") or 0),
        "status": doc.get("status") or "active",
        "retired": int(doc.get("retired") or 0),
        "title": doc.get("title") or "",
        "premise": doc.get("premise") or "",
        "action": doc.get("action") or "",
        "trigger": trigger or {},
    }


def make_chunks(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把一条模式切成检索块。

    * 有分支 → 每个分支一块（``branch_no`` = 1..n）；
    * 无分支 → 一块（``branch_no`` = 0）。

    另外额外给**模式级块**（branch_no=0，带全部做法摘要），
    这样「这条模式整体讲什么」也能被检索到。

    每块都带上文档的过滤键（:func:`doc_view`）—— Qdrant 靠它们过滤。
    """
    out: List[Dict[str, Any]] = []
    mode_id = str(doc.get("id") or "")
    sha = content_sha256(doc.get("payload") or doc)
    view = doc_view(doc)

    def _chunk(branch_no: int, text: str) -> Dict[str, Any]:
        return {
            "chunk_id": "%s#b%d" % (mode_id, branch_no),
            "mode_id": mode_id,
            "branch_no": branch_no,
            "text": text,
            "search_text": text,
            "content_sha256": sha,
            # 中文按字符估：粗估够用，只用来做上下文预算裁剪。
            "token_estimate": max(1, len(text) // 2),
            "doc": dict(view),
        }

    branches = _branches_of(doc)
    if branches:
        for i, b in enumerate(branches, start=1):
            out.append(_chunk(i, branch_text(doc, b, i)))
    out.append(_chunk(0, branch_text(doc, None, 0)))
    return out


def normalize_document(rule: Dict[str, Any], *, game: str, version: str,
                       layer: str) -> Dict[str, Any]:
    """把一条原始规则记录整理成文档。

    **原始记录原封不动**地留在 ``payload`` 里 —— 导出 JSON 时原样写回，
    所以「DB → JSON」是可逆的。
    """
    from ..rules.checklist import hero_of, lane_of, window_of

    rid = str(rule.get("id") or "")
    win = window_of(rule) or [0, 0]
    lane = lane_of(rule) or ""
    hero = hero_of(rule) or ""
    return {
        "id": rid,
        "game": game,
        "version": version,
        "layer": layer,
        "lane": lane,
        "hero": hero,
        "window_lo": int(win[0]),
        "window_hi": int(win[1]),
        "title": str(rule.get("title") or ""),
        "category": str(rule.get("category") or ""),
        "severity": str(rule.get("severity") or "medium"),
        "status": str(rule.get("status") or "active"),
        "premise": str(rule.get("premise") or ""),
        "action": str(rule.get("action") or ""),
        "outcome": str(rule.get("outcome") or ""),
        "deviation_hint": str(rule.get("deviation_hint") or ""),
        "branches": [b for b in (rule.get("branches") or [])
                     if isinstance(b, dict)],
        "subject": rule.get("subject") or {},
        "expect": rule.get("expect") or {},
        "node_actions": rule.get("node_actions") or {},
        "source": rule.get("source") or {},
        "payload": rule,
        "content_sha256": content_sha256(rule),
    }


@dataclass
class ChunkHit:
    """检索命中的一条。"""
    chunk_id: str = ""
    mode_id: str = ""
    branch_no: int = 0
    text: str = ""
    #: 命中分（各路融合后的最终分）。
    score: float = 0.0
    #: 这一分是**哪一路**给的（dense / sparse / rule / fallback）。
    source: str = ""
    #: 命中的模式标题（渲染用）。
    title: str = ""
    lane: str = ""
    hero: str = ""
    window: Tuple[int, int] = (0, 0)
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "mode_id": self.mode_id,
            "branch_no": self.branch_no,
            "text": self.text,
            "score": round(self.score, 6),
            "source": self.source,
            "title": self.title,
            "lane": self.lane,
            "hero": self.hero,
            "window": list(self.window),
        }


def _loads(raw: str, default: Any) -> Any:
    try:
        value = json.loads(raw or "")
    except (ValueError, TypeError):
        return default
    return value if value not in (None, "") else default


def _doc_to_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    """payload → 文档字典（形状与旧版 ``_doc_to_dict`` 一致）。"""
    doc = dict(payload or {})
    doc.setdefault("branches", [])
    doc.setdefault("subject", {})
    doc.setdefault("expect", {})
    doc.setdefault("node_actions", {})
    doc.setdefault("source", {})
    doc.setdefault("payload", {})
    return doc


def chunk_doc(payload: Dict[str, Any]) -> Dict[str, Any]:
    """检索块的 payload → **文档形状**（过滤键 + 护栏）。

    检索块上冗余了文档的过滤键（游戏/版本/层/分路/英雄/时间窗/状态/作废），
    这样 Qdrant 能不 join 就过滤。这里把它们拼回文档的形状，
    让 :meth:`app.rag.retriever.Retriever._allowed` 能直接用。
    """
    payload = payload or {}
    doc = {
        "game": payload.get("game") or "",
        "version": payload.get("version") or "",
        "layer": payload.get("layer") or "lane",
        "lane": payload.get("lane") or "",
        "hero": payload.get("hero") or "",
        "window_lo": int(payload.get("window_lo") or 0),
        "window_hi": int(payload.get("window_hi") or 0),
        "status": payload.get("status") or "active",
        "retired": int(payload.get("retired") or 0),
        "title": payload.get("title") or "",
        "premise": payload.get("premise") or "",
        "action": payload.get("action") or "",
        # 护栏（通则 vs 英雄专属版）在 ``trigger.none`` 里。
        "payload": {"trigger": payload.get("trigger") or {}},
    }
    return doc


class ModeStore:
    """模式库的读写（Qdrant）。

    对外接口与上一版保持一致 —— 调用方（检索层、规则层、那些工具脚本）
    不用为了换存储而改写法。
    """

    def __init__(self, client: Any = None):
        self._client = client

    # -- 连接 ------------------------------------------------------------

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = qc.get_client()
        return self._client

    # -- 元信息 ----------------------------------------------------------

    def _read_meta(self) -> Dict[str, Any]:
        """读索引的元信息（一个点）。读不到就是空。"""
        from qdrant_client import models

        try:
            points = self.client.retrieve(
                collection_name=META_COLLECTION,
                ids=[META_POINT_ID],
                with_payload=True,
            )
        except Exception:                                  # noqa: BLE001
            return {}
        if not points:
            return {}
        return dict(points[0].payload or {})

    def _write_meta(self, data: Dict[str, Any]) -> bool:
        """写索引的元信息。**写不成功要说出来** —— 返回 False。"""
        global last_meta_error
        from qdrant_client import models

        try:
            if not qc.collection_exists(self.client, META_COLLECTION):
                # 元信息集合自己也带一个 1 维占位向量 ——
                # Qdrant 的每个点都得带向量（本地模式不支持纯 payload 点）。
                self.client.create_collection(
                    collection_name=META_COLLECTION,
                    vectors_config={
                        "meta": models.VectorParams(
                            size=1, distance=models.Distance.COSINE),
                    })
            self.client.upsert(
                collection_name=META_COLLECTION,
                points=[models.PointStruct(
                    id=META_POINT_ID, vector={"meta": [1.0]}, payload=data)],
            )
            last_meta_error = ""
            return True
        except Exception as exc:                           # noqa: BLE001
            # **不能静默** —— 元信息丢了，索引就说不清自己是哪一档的向量。
            last_meta_error = "%s: %s" % (type(exc).__name__, exc)
            return False

    # -- 写入 ------------------------------------------------------------

    def upsert_documents(self, docs: Sequence[Dict[str, Any]],
                         retired: Optional[Iterable[str]] = None,
                         embeddings: Optional[Dict[str, Any]] = None
                         ) -> int:
        """写入（或覆盖）文档。返回写入条数。

        ``embeddings`` 给的是 ``mode_id → 向量``。文档集合**不参与向量检索**，
        所以向量只是占位（Qdrant 的每个点都得带一个）——
        没给就用零向量。
        """
        import numpy as np
        from qdrant_client import models

        retired_set = {str(r) for r in (retired or [])}
        embeddings = embeddings or {}
        if not qc.collection_exists(self.client, qc.DOCS_COLLECTION):
            qc.ensure_collections(self.client, self._dim_or_default())
        points = []
        for doc in docs:
            rid = str(doc.get("id") or "")
            if not rid:
                continue
            payload = dict(doc)
            payload["retired"] = 1 if rid in retired_set else 0
            payload["payload"] = doc.get("payload") or {}
            # 文档集合用 1 维占位向量（不参与向量检索）。
            vector = {qc.DENSE_VECTOR: [0.0] * qc.DOCS_DIM}
            points.append(models.PointStruct(
                id=qc.point_id("doc:" + rid), vector=vector, payload=payload))
        if not points:
            return 0
        for start in range(0, len(points), 256):
            self.client.upsert(collection_name=qc.DOCS_COLLECTION,
                               points=points[start:start + 256])
        self._sync_retired_to_chunks(docs, retired_set)
        return len(points)

    def _sync_retired_to_chunks(self, docs: Sequence[Dict[str, Any]],
                                retired_set: set) -> None:
        """**文档退役了，它的检索块也必须退役。**

        为什么不能不管：块上冗余了 ``retired`` 过滤键（Qdrant 靠它过滤），
        文档改了状态而块没改 —— 检索照样把这条打法捞回来，
        **拿退休打法纠正玩家**，而报告上看不出任何异常。

        两次 ``set_payload`` 搞定（不用一条条改）。
        """
        from qdrant_client import models

        if not docs or not qc.collection_exists(self.client, qc.CHUNKS_COLLECTION):
            return
        live = [str(d.get("id") or "") for d in docs
                if str(d.get("id") or "") not in retired_set]
        dead = [str(d.get("id") or "") for d in docs
                if str(d.get("id") or "") in retired_set]
        try:
            for ids, flag in ((live, 0), (dead, 1)):
                if not ids:
                    continue
                # **用 MatchAny，不要用 1 680 个条件的 should 列表** ——
                # 本地模式是逐点逐条件比对的，那样的列表要几百万次比较，
                # 实测卡住好几分钟。MatchAny 是一条条件里放一组值。
                cond = models.FieldCondition(
                    key="mode_id", match=models.MatchAny(any=list(ids)))
                self.client.set_payload(
                    collection_name=qc.CHUNKS_COLLECTION,
                    payload={"retired": flag},
                    points=models.FilterSelector(filter=models.Filter(
                        must=[cond])),
                )
        except Exception:                                  # noqa: BLE001
            pass

    def upsert_chunks(self, chunks: Sequence[Dict[str, Any]],
                      embeddings: Optional[Dict[str, Any]] = None,
                      embedding_model: str = "",
                      embedding_dim: int = 0) -> int:
        """写入检索块。``embeddings`` 给的是 ``chunk_id → 向量``。"""
        import numpy as np
        from qdrant_client import models

        embeddings = embeddings or {}
        dim = int(embedding_dim or 0)
        for vec in list(embeddings.values())[:1]:
            dim = max(dim, int(np.asarray(vec).shape[-1]))
        # **写之前先校准集合的维度** —— 维度对不上时 ``ensure_collections``
        # 会删掉重建（旧向量已经没法用了）。
        # 这一步不能省：文档先入库时还不知道向量维度，集合会被建成 1 维，
        # 随后拿 256/1536 维的向量往里写就是「维度不符」。
        if dim:
            qc.ensure_collections(self.client, dim)
        elif not qc.collection_exists(self.client, qc.CHUNKS_COLLECTION):
            qc.ensure_collections(self.client, self._dim_or_default())
        points = []
        for chunk in chunks:
            cid = str(chunk.get("chunk_id") or "")
            if not cid:
                continue
            doc = chunk.get("doc") or {}
            vec = embeddings.get(cid)
            if vec is not None:
                # 命名向量空间：必须写成 ``{"dense": [...]}``，
                # 传裸 list 会被拒（"Unnamed vectors are not allowed…"）。
                vector = {qc.DENSE_VECTOR:
                          np.asarray(vec, dtype="float32").tolist()}
            else:
                # **没算出来的向量也要写点** —— Qdrant 的每个点都得带向量。
                # 零向量 = 「这一块没有向量」，检索时靠 ``embedded`` 标记认出来。
                dim = qc.dense_dim(self.client) or self._dim_or_default()
                vector = {qc.DENSE_VECTOR: [0.0] * int(dim)}
            payload = {
                "chunk_id": cid,
                "mode_id": str(chunk.get("mode_id") or ""),
                "branch_no": int(chunk.get("branch_no") or 0),
                "text": str(chunk.get("text") or ""),
                # **预分词文本** —— BM25 查的就是它（见 app.rag.sparse）。
                "search_text": pretokenized(
                    str(chunk.get("search_text") or chunk.get("text") or "")),
                "token_estimate": int(chunk.get("token_estimate") or 0),
                "content_sha256": str(chunk.get("content_sha256") or ""),
                "embedded": 1 if vector is not None else 0,
                # 过滤键：冗余在块上，Qdrant 才能不 join 就过滤。
                "game": str(doc.get("game") or ""),
                "version": str(doc.get("version") or ""),
                "layer": str(doc.get("layer") or "lane"),
                "lane": str(doc.get("lane") or ""),
                "hero": str(doc.get("hero") or ""),
                "window_lo": int(doc.get("window_lo") or 0),
                "window_hi": int(doc.get("window_hi") or 0),
                "status": str(doc.get("status") or "active"),
                "retired": int(doc.get("retired") or 0),
                # 规则命中那一路要用的文本。
                "title": str(doc.get("title") or ""),
                "premise": str(doc.get("premise") or ""),
                "action": str(doc.get("action") or ""),
                # 判定用的护栏（通则 vs 英雄专属版）。
                "trigger": doc.get("trigger") or {},
            }
            points.append(models.PointStruct(
                id=qc.point_id("chunk:" + cid), vector=vector, payload=payload))
        if not points:
            return 0
        for start in range(0, len(points), 256):
            self.client.upsert(collection_name=qc.CHUNKS_COLLECTION,
                               points=points[start:start + 256])
        return len(points)

    def clear_chunks(self, dim: int = 0) -> None:
        """清空检索块（重建前用）。文档不动。

        **删之前先记住维度** —— 集合删掉之后就查不到维度了，
        重建时只能拿 1 维占位，随后写入 1536 维的向量就是「维度不符」。
        """
        if not qc.collection_exists(self.client, qc.CHUNKS_COLLECTION):
            return
        keep = int(dim or 0) or qc.dense_dim(self.client) or 1
        try:
            self.client.delete_collection(qc.CHUNKS_COLLECTION)
        except Exception:                                  # noqa: BLE001
            pass
        qc.ensure_collections(self.client, keep)

    def _dim_or_default(self) -> int:
        dim = qc.dense_dim(self.client)
        return dim if dim else 1

    # -- 读取 ------------------------------------------------------------

    def documents(self, *, layer: Optional[str] = None,
                  game: Optional[str] = None,
                  version: Optional[str] = None,
                  include_retired: bool = False
                  ) -> List[Dict[str, Any]]:
        """按过滤条件读文档。"""
        from qdrant_client import models

        if not qc.collection_exists(self.client, qc.DOCS_COLLECTION):
            return []
        must: List[Any] = []
        if layer:
            must.append(models.FieldCondition(
                key="layer", match=models.MatchValue(value=layer)))
        if game:
            must.append(models.FieldCondition(
                key="game", match=models.MatchValue(value=game)))
        if version:
            must.append(models.FieldCondition(
                key="version", match=models.MatchValue(value=version)))
        if not include_retired:
            must.append(models.FieldCondition(
                key="retired", match=models.MatchValue(value=0)))
        out: List[Dict[str, Any]] = []
        for batch in self._scroll(qc.DOCS_COLLECTION, must, limit=100000):
            for payload in batch:
                out.append(_doc_to_dict(payload))
        return out

    def rules(self, game: str = "", layer: str = "",
              include_retired: bool = False) -> List[Dict[str, Any]]:
        """以**规则**的形状读出来（payload 原样）。

        规则库（:func:`app.rules.density._seed_rules`）吃的就是这种形状，
        所以 Qdrant 里的内容能**直接顶替** JSON 文件 —— 判定口径一字不变。
        """
        docs = self.documents(layer=layer or None, game=game or None,
                               include_retired=include_retired)
        out: List[Dict[str, Any]] = []
        for doc in docs:
            payload = doc.get("payload") or {}
            if isinstance(payload, dict) and payload:
                out.append(payload)
        return out

    def chunk_corpus(self, *, game: str = "", layer: str = "",
                     include_retired: bool = False) -> List[Dict[str, Any]]:
        """检索块语料（**不含向量** —— 向量在 Qdrant 里，不用搬回 Python）。"""
        from qdrant_client import models

        if not qc.collection_exists(self.client, qc.CHUNKS_COLLECTION):
            return []
        must: List[Any] = []
        if game:
            must.append(models.FieldCondition(
                key="game", match=models.MatchValue(value=game)))
        if layer:
            must.append(models.FieldCondition(
                key="layer", match=models.MatchValue(value=layer)))
        if not include_retired:
            must.append(models.FieldCondition(
                key="retired", match=models.MatchValue(value=0)))
        out: List[Dict[str, Any]] = []
        for batch in self._scroll(qc.CHUNKS_COLLECTION, must, limit=100000):
            out.extend(batch)
        return out

    def _scroll(self, collection: str, must: Sequence[Any],
                limit: int = 100000) -> Iterable[List[Dict[str, Any]]]:
        """滚动取 payload（分批）。"""
        from qdrant_client import models

        flt = models.Filter(must=list(must)) if must else None
        offset = None
        got = 0
        while got < limit:
            points, offset = self.client.scroll(
                collection_name=collection, scroll_filter=flt,
                limit=min(512, limit - got), offset=offset,
                with_payload=True, with_vectors=False,
            )
            if not points:
                break
            batch = [dict(p.payload or {}) for p in points]
            got += len(batch)
            yield batch
            if offset is None:
                break

    # -- 检索 ------------------------------------------------------------

    def build_filter(self, *, game: str = "", version: str = "", lane: str = "",
                     hero: str = "", lo: int = 0, hi: int = 60) -> Any:
        """硬约束过滤器（payload filter）。

        口径与 :meth:`app.rag.retriever.Retriever._allowed` 一致：
        * 作废 / 非 active 的不算；
        * 英雄层必须英雄精确相等；**英雄未知时只给分路层**；
        * 分路层必须分路相等；
        * 时间窗必须相交。
        """
        from qdrant_client import models

        must: List[Any] = [
            models.FieldCondition(key="retired",
                                  match=models.MatchValue(value=0)),
            models.FieldCondition(key="status",
                                  match=models.MatchValue(value="active")),
        ]
        if game:
            must.append(models.FieldCondition(
                key="game", match=models.MatchValue(value=game)))
        if version:
            must.append(models.FieldCondition(
                key="version", match=models.MatchValue(value=version)))

        lane_conds: List[Any] = [
            models.FieldCondition(key="layer",
                                  match=models.MatchValue(value="lane"))]
        if lane:
            # 分路为空也算匹配 —— 与 Python 侧的口径一致。
            lane_conds.append(models.Filter(should=[
                models.FieldCondition(key="lane",
                                      match=models.MatchValue(value=lane)),
                models.FieldCondition(key="lane",
                                      match=models.MatchValue(value="")),
            ]))
        if hero:
            must.append(models.Filter(should=[
                models.Filter(must=lane_conds),
                models.Filter(must=[
                    models.FieldCondition(key="layer",
                                          match=models.MatchValue(value="hero")),
                    models.FieldCondition(key="hero",
                                          match=models.MatchValue(value=hero)),
                ]),
            ]))
        else:
            must.append(models.Filter(must=lane_conds))

        # 时间窗相交：块的 [lo, hi] 与查询的 [lo, hi] 有交集。
        must.append(models.FieldCondition(
            key="window_hi", range=models.Range(gte=int(lo))))
        must.append(models.FieldCondition(
            key="window_lo", range=models.Range(lte=int(hi))))
        return models.Filter(must=must)

    def search_dense(self, vector: Sequence[float], *, game: str = "",
                     version: str = "", lane: str = "", hero: str = "",
                     lo: int = 0, hi: int = 60, limit: int = 32
                     ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """稠密召回。返回 ``(chunk_id, 分, payload)``。"""
        from qdrant_client import models

        if not qc.collection_exists(self.client, qc.CHUNKS_COLLECTION):
            return []
        flt = self.build_filter(game=game, version=version, lane=lane,
                                hero=hero, lo=lo, hi=hi)
        try:
            res = self.client.query_points(
                collection_name=qc.CHUNKS_COLLECTION,
                query=list(vector), using=qc.DENSE_VECTOR,
                query_filter=flt, limit=int(limit), with_payload=True,
            )
        except Exception as exc:                           # noqa: BLE001
            raise RuntimeError("Qdrant 稠密检索失败：%s: %s"
                               % (type(exc).__name__, exc)) from exc
        out = []
        for point in res.points:
            payload = dict(point.payload or {})
            out.append((str(payload.get("chunk_id") or ""),
                        float(point.score), payload))
        return out

    def search_sparse(self, query: str, *, game: str = "", version: str = "",
                      lane: str = "", hero: str = "", lo: int = 0, hi: int = 60,
                      limit: int = 32) -> List[Tuple[str, float, Dict[str, Any]]]:
        """稀疏召回（Qdrant 的 BM25 文本索引）。

        传进来的 ``query`` 是**原始自然语言**；这里先切成词再交给 Qdrant，
        它用 BM25（含 IDF）算分。
        """
        from qdrant_client import models

        if not qc.collection_exists(self.client, qc.CHUNKS_COLLECTION):
            return []
        from .sparse import query_text

        flt = self.build_filter(game=game, version=version, lane=lane,
                                hero=hero, lo=lo, hi=hi)
        try:
            res = self.client.query_points(
                collection_name=qc.CHUNKS_COLLECTION,
                query=models.TextQuery(text=query_text(query)),
                using=qc.TEXT_FIELD,
                query_filter=flt, limit=int(limit), with_payload=True,
            )
        except Exception as exc:                           # noqa: BLE001
            raise RuntimeError("Qdrant 稀疏检索失败：%s: %s"
                               % (type(exc).__name__, exc)) from exc
        out = []
        for point in res.points:
            payload = dict(point.payload or {})
            out.append((str(payload.get("chunk_id") or ""),
                        float(point.score), payload))
        return out

    def scan(self, *, game: str = "", version: str = "", lane: str = "",
             hero: str = "", lo: int = 0, hi: int = 60
             ) -> List[Dict[str, Any]]:
        """按硬约束扫一遍语料（规则命中那一路要用）。"""
        if not qc.collection_exists(self.client, qc.CHUNKS_COLLECTION):
            return []
        flt = self.build_filter(game=game, version=version, lane=lane,
                                hero=hero, lo=lo, hi=hi)
        out: List[Dict[str, Any]] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=qc.CHUNKS_COLLECTION, scroll_filter=flt,
                limit=1024, offset=offset, with_payload=True,
                with_vectors=False,
            )
            if not points:
                break
            out.extend(dict(p.payload or {}) for p in points)
            if offset is None:
                break
        return out

    def documents_by_ids(self, mode_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """按模式 id 取文档（拿到原始规则）。"""
        from qdrant_client import models

        ids = [str(m) for m in mode_ids if str(m)]
        if not ids:
            return {}
        if not qc.collection_exists(self.client, qc.DOCS_COLLECTION):
            return {}
        point_ids = [qc.point_id("doc:" + m) for m in ids]
        try:
            points = self.client.retrieve(
                collection_name=qc.DOCS_COLLECTION, ids=point_ids,
                with_payload=True,
            )
        except Exception:                                  # noqa: BLE001
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for point in points:
            payload = dict(point.payload or {})
            rid = str(payload.get("id") or "")
            if rid:
                out[rid] = payload
        return out

    # -- 状态 ------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """库的状态（条数、层、版本、嵌入档）。"""
        from qdrant_client import models

        out: Dict[str, Any] = {
            "documents": 0, "retired": 0, "chunks": 0, "with_embedding": 0,
            "layers": [], "games": [], "versions": [],
            "embedding_models": [], "embedding_dims": [],
            # **连不上时的原因**（空串 = 连得上）。
            # 「库是空的」与「库连不上」要分开说 —— 该做的事完全不同
            # （一个补资料，一个修连接）。
            "collection_error": "",
        }
        _exists, _err = qc.collection_state(self.client, qc.DOCS_COLLECTION)
        if _err:
            out["collection_error"] = _err
        if not _exists:
            return out
        try:
            out["documents"] = int(
                self.client.count(qc.DOCS_COLLECTION, exact=True).count)
            out["retired"] = int(self.client.count(
                qc.DOCS_COLLECTION, exact=True,
                count_filter=models.Filter(must=[
                    models.FieldCondition(key="retired",
                                          match=models.MatchValue(value=1))]),
            ).count)
            layers, games, versions = set(), set(), set()
            for batch in self._scroll(qc.DOCS_COLLECTION, [], limit=100000):
                for payload in batch:
                    if payload.get("layer"):
                        layers.add(str(payload["layer"]))
                    if payload.get("game"):
                        games.add(str(payload["game"]))
                    if payload.get("version"):
                        versions.add(str(payload["version"]))
            out["layers"] = sorted(layers)
            out["games"] = sorted(games)
            out["versions"] = sorted(versions)
        except Exception:                                  # noqa: BLE001
            pass
        if qc.collection_exists(self.client, qc.CHUNKS_COLLECTION):
            try:
                out["chunks"] = int(
                    self.client.count(qc.CHUNKS_COLLECTION, exact=True).count)
                out["with_embedding"] = int(self.client.count(
                    qc.CHUNKS_COLLECTION, exact=True,
                    count_filter=models.Filter(must=[
                        models.FieldCondition(key="embedded",
                                              match=models.MatchValue(value=1))]),
                ).count)
            except Exception:                              # noqa: BLE001
                pass
            dim = qc.dense_dim(self.client)
            if dim:
                out["embedding_dims"] = [int(dim)]
        meta = self._read_meta()
        if meta.get("embedding_model"):
            out["embedding_models"] = [str(meta["embedding_model"])]
        return out

    def fingerprint(self, game: str = "", layer: str = "") -> str:
        """库的轻量指纹（条数 + 索引版本）。

        用来判断进程内缓存要不要失效 —— 跨进程也能发现
        「另一个进程刚刚重建了索引」。
        """
        meta = self._read_meta()
        docs = int(meta.get("doc_count") or 0)
        chunks = int(meta.get("chunk_count") or 0)
        version = str(meta.get("index_version") or "")
        if not meta:
            try:
                docs = int(self.client.count(
                    qc.DOCS_COLLECTION, exact=True).count)
            except Exception:                              # noqa: BLE001
                docs = 0
        return "%d|%d|%s" % (docs, chunks, version)

    def embedding_backend(self) -> str:
        """库里的向量是**哪种嵌入**算的（轻量查询）。

        为什么单独一个方法：``stats()`` 会把全部块读一遍 ——
        「索引是不是过期了」这个问题每次检索都要问一次，不能用重查询。
        """
        meta = self._read_meta()
        model = str(meta.get("embedding_model") or "")
        dim = int(meta.get("embedding_dim") or 0)
        if not model:
            dim = dim or qc.dense_dim(self.client)
        if not model:
            return ""
        return "%s-%s" % (model, "x".join(str(d) for d in [dim])) if dim else model

    def index_meta(self) -> Dict[str, Any]:
        """索引的元信息（嵌入档、维度、条数、版本）。"""
        return self._read_meta()

    def set_index_meta(self, data: Dict[str, Any]) -> bool:
        """写索引元信息。写不成功返回 False（原因见 ``last_meta_error``）。"""
        return self._write_meta(data)
