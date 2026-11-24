"""**候选检索 skill**：让模型自己决定「这一局该对照哪几条攻略」。

这一步替代了什么
----------------
旧口径（:func:`app.recognition2.judge.screen_candidates`）把整库
（1 680 条）按三个字段筛一遍：**作用域**（分路）、**英雄**、**时间窗**。
筛出来的是「这一局**可能**有关的全部规则」—— 不管这一局到底发生了什么。

于是模型被塞了一堆与画面无关的尺子：它要逐条回答「玩家做没做到」，
而玩家那几段压根没那回事。报告上就会出现「这一条你该做没做」，
指认的却是对局里根本没出现过的时刻。

**筛选本身不该由代码写死** —— 该由**看过这一局的人**决定。
所以现在这一步交给模型：

1. 模型看完整局录像，**自己写下**「我想查什么」（自然语言 + 时刻）；
2. 这些请求走 **RAG**（:mod:`app.rag.retriever`）去模式库里检索，
   检索层只保留**数据级不变量**（退役 / 非 active / 版本 / 英雄 / 分路 / 时间窗）；
3. 检索回来的条目 → 候选 → 模型看录像片段逐条裁决。

**代码不再决定「这一局该查什么」**，只保留不变量与兜底。

和 RAG 的关系
------------
这一步就是 :mod:`app.harness.tools` 里那个 ``search_modes`` 工具
（模式库检索），只不过**由模型自己发起**、而不是由代码替模型决定查什么。
工具层的老规矩照旧：**永不抛异常** —— 查不到就回一句话，
检索是补充，不是前提（见 :func:`app.harness.tools.tool_search_modes`）。

兜底要说清楚
------------
RAG 库建不起来、或者模型一个字都没查的时候，退回旧口径
（按作用域/英雄/时间窗整条筛）。但**必须在留痕里写明这次走的是哪条路**——
说谎的诊断比没有诊断更坏。
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..harness import tools as htools
from .beat import Beat, Timeline
from .judge import (Candidate, MatchFacts, RuleJudgement, dominant_position,
                    screen_candidates, screen_one)


# ---------------------------------------------------------------- 常量

#: 一次检索最多带回几条（**模式**数，不是块数）。
DEFAULT_TOP_K = 5

#: 一局最多用几个检索请求。
DEFAULT_MAX_REQUESTS = 8

#: 模型一次最多写几个检索请求（写多了就截断 —— 提示词里也这么要求）。
MAX_MODEL_REQUESTS = 12

#: 候选最多留几条（再多的按检索顺序截断，**并写进留痕**）。
DEFAULT_MAX_CANDIDATES = 18

#: 一次检索最多带回几条（模型说「100 条」也不行 —— 判不完）。
MAX_TOP_K = 10

#: 检索请求的来源。
SOURCE_MODEL = "model"
SOURCE_NARRATIVE = "narrative"


# ---------------------------------------------------------------- 请求


@dataclass
class RetrievalRequest:
    """模型自己写的**检索请求**。

    ``query`` 是模型**自己的话**（「我在河道跟人打架，这时候该占龙坑吗」）；
    ``at`` 是它说的**原视频第几秒**（可空 —— 没有就查整局）；
    ``k`` 是它要几条（0 = 用默认）。
    """

    query: str = ""
    at: Optional[float] = None
    k: int = 0
    #: 请求从哪来：``model``（模型自己写的）/ ``narrative``
    #: （模型没写请求时，用它自己讲的事件叙述当查询）。
    source: str = SOURCE_MODEL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "at": (round(float(self.at), 2) if self.at is not None else None),
            "k": int(self.k or 0),
            "source": self.source,
        }


def parse_requests(raw: Any, *, scale: float = 1.0, offset: float = 0.0
                   ) -> List[RetrievalRequest]:
    """把模型回答里的 ``queries`` 解析成检索请求。

    ``scale`` / ``offset`` 用来把模型报的秒换算回**原视频**秒
    （重编码过的片段与原片差一个比例 —— 与事件时刻同一套换算）。

    解析不出来就返回空列表 —— **不猜**。空列表会让调用方去走
    「用模型自己的叙述当查询」那条路，而不是**替模型编一个**。
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    if isinstance(raw, dict):
        # 也接受整个回答对象（``{"queries": [...]}``）——
        # 调用方少一次解包，少一种「传错形状」的坑。
        raw = raw.get("queries")
    if not isinstance(raw, list):
        return []
    out: List[RetrievalRequest] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query") or item.get("about") or "").strip()
        if not query:
            continue
        try:
            k = int(item.get("k") or 0)
        except (TypeError, ValueError):
            k = 0
        at: Optional[float] = None
        raw_at = item.get("at", item.get("t"))
        if raw_at is not None:
            try:
                at = float(raw_at)
            except (TypeError, ValueError):
                at = None
            if at is not None:
                at = at * float(scale or 1.0) + float(offset or 0.0)
        out.append(RetrievalRequest(query=query, at=at, k=k,
                                    source=SOURCE_MODEL))
    return out


def requests_from_beats(beats: Sequence[Beat], limit: int = DEFAULT_MAX_REQUESTS
                        ) -> List[RetrievalRequest]:
    """用模型**自己讲的事件叙述**当查询。

    什么时候走这条路：模型写了检索请求之外，它讲的事件本身就是
    「这一局发生了什么」——拿它去检索，检索到的就是**它关心**的那些打法。

    仍然只取**有叙述**的段（没讲清楚的段检索不出东西）。
    """
    out: List[RetrievalRequest] = []
    for beat in beats or []:
        text = str(getattr(beat, "narrative", "") or "").strip()
        if not text:
            continue
        at = getattr(beat, "video_time", None)
        out.append(RetrievalRequest(
            query=text,
            at=(float(at) if at is not None else None),
            source=SOURCE_NARRATIVE))
        if len(out) >= max(1, int(limit or 0)):
            break
    return out


# ---------------------------------------------------------------- 配置与结果


@dataclass
class SelectorConfig:
    """候选检索 skill 的参数。"""

    game: str = "hok"
    #: 一次检索带回几条。
    top_k: int = DEFAULT_TOP_K
    #: 一局最多用几个检索请求。
    max_requests: int = DEFAULT_MAX_REQUESTS
    #: 候选最多留几条。
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    #: 分路 / 英雄 —— 交给检索层做**硬过滤**（不变量）。
    lane: str = ""
    hero: str = ""
    #: 允许退回旧口径吗（只在语义那条路彻底走不通时）。
    allow_fallback: bool = True
    #: **前提（``when``）交给模型判**（2026-11-14 用户要求：
    #: 局势与规则的理解都归模型）。打开后，代码只把「哪几项前提对不上」
    #: 记成线索，适不适用由模型裁决；关掉就退回「前提不成立 → 不适用」。
    model_premise: bool = True


@dataclass
class SelectionResult:
    """选出来的候选 + **留痕**。"""

    candidates: List[Candidate] = field(default_factory=list)
    #: 被判成「不适用」的（检索回来了，但这一局不适用他）。
    screened: List[RuleJudgement] = field(default_factory=list)
    #: 实际发出的检索请求。
    requests: List[RetrievalRequest] = field(default_factory=list)
    #: 每个请求召回了什么（报告与排查用）。
    hits: List[Dict[str, Any]] = field(default_factory=list)
    #: 走的是哪条路：``model``（模型自己找）/ ``fallback``（退回顾则筛）。
    mode: str = "model"
    notes: List[str] = field(default_factory=list)
    degradation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "candidates": len(self.candidates),
            "screened": len(self.screened),
            "requests": [r.to_dict() for r in self.requests],
            "hits": list(self.hits),
            "notes": list(self.notes),
            "degradation": dict(self.degradation),
        }


# ---------------------------------------------------------------- 检索上下文


class _ModeSearchContext:
    """``search_modes`` 工具的上下文（工具契约：``search_modes(query, k)``）。

    把工具接到 RAG 上：检索走 :class:`app.rag.context.ModeSearchService`，
    条目形状与窗口清单**完全同形**（:func:`app.rules.checklist.entry_from_rule`）。

    每个请求的**时间窗**由调用方给（模型说的是「哪一刻」）——
    检索层据此只保留窗口与那一刻相交的打法（硬语义：10 分钟才该做的事，
    3 分钟做不算数）。
    """

    def __init__(self, service: Any, *, lane: str = "", hero: str = "",
                 lo: int = 0, hi: int = 60, top_k: int = DEFAULT_TOP_K):
        self._service = service
        self.lane = lane
        self.hero = hero
        self.lo = int(lo)
        self.hi = int(hi)
        self.top_k = int(top_k)

    def search_modes(self, query: str, k: int = 0) -> Dict[str, Any]:
        """查打法。返回 ``{"entries": [...], "degradation": {...}}``。"""
        top_k = int(k or 0) or self.top_k
        # **模型说几条就几条**？不行 —— 它说「100 条」也能说。
        # 一次检索带回太多，提示词会被撑爆，模型也判不完。
        top_k = max(1, min(int(top_k), MAX_TOP_K))
        result = self._service.search(query, lane=self.lane, hero=self.hero,
                                      lo=self.lo, hi=self.hi, top_k=top_k)
        entries = self._service.entries(result)
        return {
            "entries": entries,
            "degradation": dict(result.degradation or {}),
        }


# ---------------------------------------------------------------- 时间窗


def _video_offset(timeline: Timeline) -> Optional[float]:
    """原视频秒 → 局内秒 的偏移（取中位数）。

    Beat 上有 ``video_time`` 与 ``game_time``，两者一减就是偏移。
    取中位数：单个段的边界是切出来的，估出来会有几秒误差。
    """
    diffs: List[float] = []
    for beat in timeline.beats or []:
        game = getattr(beat, "game_time", None)
        video = getattr(beat, "video_time", None)
        if game is None or video is None:
            continue
        if float(game) < 0:
            continue
        diffs.append(float(video) - float(game))
    if not diffs:
        return None
    return float(statistics.median(diffs))


def _window_for(request: RetrievalRequest, offset: Optional[float],
                span: Tuple[float, float]) -> Tuple[int, int]:
    """这个请求对应局内的哪几分钟（闭区间，分钟）。

    * 请求带时刻 → 取那一分钟（向下取整到分、向上取整到分，
      让「2:00~4:00」这种跨两分钟的窗口整个落进来）；
    * 不带时刻 → 整局。
    """
    lo_min = int(math.floor(max(0.0, span[0]) / 60.0))
    hi_min = int(math.ceil(max(0.0, span[1]) / 60.0))
    if request.at is None or offset is None:
        return (lo_min, max(hi_min, lo_min))
    game_second = float(request.at) - float(offset)
    if game_second < 0:
        game_second = 0.0
    minute = game_second / 60.0
    lo = int(math.floor(minute))
    hi = int(math.ceil(minute))
    if hi < lo:
        hi = lo
    return (lo, hi)


# ---------------------------------------------------------------- 主流程


def _in_game(timeline: Timeline) -> List[Beat]:
    return [b for b in (timeline.beats or [])
            if b.game_time_start >= 0 and b.game_time_end >= 0]


def _fallback(timeline: Timeline, patterns: Sequence[Any], facts: MatchFacts,
              reason: str, notes: List[str]
              ) -> Tuple[List[Candidate], List[RuleJudgement]]:
    """退回旧口径（按作用域/英雄/时间窗整条筛）。

    只在语义那条路**彻底走不通**时用（库建不起来、模型没给任何可查的情形）。
    走这条路必须在留痕里说清楚 —— 否则报告上「13 条要检验」看起来
    跟模型自己找出来的没有区别。
    """
    candidates, screened = screen_candidates(timeline, patterns, facts)
    notes.append("候选规则退回按「作用域/英雄/时间窗」整条筛的口径（%s）。" % reason)
    return candidates, screened


def requests_from_situation(deaths: Sequence[Dict[str, Any]],
                            operations: Sequence[Dict[str, Any]] = (),
                            limit: int = 4) -> List[RetrievalRequest]:
    """**从这一局真实发生的事**补检索请求（2026-11-14 用户要求）。

    原话：「候选检索应该完全依赖 2 和 4、5，因为判定前提肯定是需要根据
    真实情况触发的」。

    光靠看整局时写的检索请求，会漏掉一类规则：**只有真出了事才想得起来查**
    的打法（「残血还追人」→ 该不该追、「单人带线」→ 带线边界）。
    所以这里把已经复盘出来的事实直接当查询：

    * 每次死亡：``<位置> <死因>``；
    * 判过有问题的操作：那一段的 ``what``。

    这些查询的 ``at`` 就是事发时刻，检索层会顺手把时间窗收紧。
    """
    out: List[RetrievalRequest] = []
    for item in deaths or []:
        if not isinstance(item, dict):
            continue
        where = str(item.get("where") or "").strip()
        cause = str(item.get("cause") or "").strip()
        query = ("%s %s" % (where, cause)).strip()
        if not query:
            continue
        at = None
        try:
            at = float(item.get("video_start") or 0.0) or None
        except (TypeError, ValueError):
            at = None
        out.append(RetrievalRequest(query=query, at=at, k=4,
                                    source="situation"))
        if len(out) >= limit:
            return out
    for item in operations or []:
        if not isinstance(item, dict) or str(item.get("verdict")) != "problem":
            continue
        what = str(item.get("what") or "").strip()
        if not what:
            continue
        out.append(RetrievalRequest(query=what[:60], k=4, source="situation"))
        if len(out) >= limit:
            break
    return out


def select_candidates(timeline: Timeline,
                      requests: Optional[Sequence[RetrievalRequest]] = None,
                      facts: Optional[MatchFacts] = None,
                      patterns: Optional[Sequence[Any]] = None,
                      config: Optional[SelectorConfig] = None,
                      situation_requests: Optional[Sequence[RetrievalRequest]] = None,
                      ) -> SelectionResult:
    """选出「这一局要检验的候选规则」。

    流程
    ----
    1. 模型写的检索请求（没有就用它自己讲的事件叙述）；
    2. 每个请求走 ``search_modes`` 工具 → RAG 检索；
    3. 检索回来的规则 id → 模式库里的那条 → 候选（带检验区间与画面段）；
    4. 语义那条路走不通 → 退回旧口径，并**写明**。

    ``patterns`` 是模式库（:func:`app.rules.patterns.load_patterns` 的产物）。
    检索层给的只是**规则 id**，规则的原文、适用条件、时间窗都从
    ``patterns`` 里取 —— **同一份真源**，不另编一套。
    """
    cfg = config or SelectorConfig()
    facts = facts or MatchFacts()
    patterns = list(patterns or [])
    out = SelectionResult()

    in_game = _in_game(timeline)
    if not in_game:
        # **不是「退回整条筛」** —— 是一局都没得判。
        # 这两种情况在报告上必须分开说：退回整条筛 = 候选可能带无关的规则；
        # 一局都没得判 = 根本没得判。
        out.mode = "none"
        out.notes.append("这一局没有带局内时刻的段 —— 判不了任何规则。")
        return out
    game_span = (min(b.game_time_start for b in in_game),
                 max(b.game_time_end for b in in_game))

    # -- 1. 请求：模型自己写的优先 --
    reqs = list(requests or [])
    # **把「按局势补的请求」并进来**（2026-11-14）：死亡/问题操作先排前面 ——
    # 它们是这一局真出过事的地方，比模型凭记忆写的请求更该被检索到。
    for extra in (situation_requests or []):
        if extra.query and all(extra.query != r.query for r in reqs):
            reqs.insert(0, extra)
    if not reqs:
        reqs = requests_from_beats(in_game, cfg.max_requests)
        if reqs:
            out.notes.append(
                "模型没写检索请求 —— 改用它自己讲的事件叙述当查询（%d 条）。"
                % len(reqs))
    reqs = reqs[:max(1, int(cfg.max_requests or 0))]
    out.requests = list(reqs)
    if not reqs:
        if not cfg.allow_fallback:
            out.mode = "model"
            out.notes.append("模型没讲出任何可查的情形 —— 没有候选，没有裁决。")
            return out
        candidates, screened = _fallback(
            timeline, patterns, facts, "模型没讲出任何可查的情形", out.notes)
        out.mode = "fallback"
        out.candidates = candidates
        out.screened = screened
        return out

    # -- 2. 检索 --
    from ..rag.context import ModeSearchService

    service = ModeSearchService(game=cfg.game)
    offset = _video_offset(timeline)
    lane = cfg.lane or (facts.known_position() or "")
    if not lane:
        # **对局级分路读不到时，用逐段读数兜底**（与判定层同一口径）。
        # 不兜底的话，检索层会把**所有分路**的打法都放进来 ——
        # 打野的尺子拿到中路玩家手里，指认的却是对局里没出现过的时刻。
        lane = dominant_position(in_game) or ""
    hero = cfg.hero or (facts.known_hero() or "")

    order: List[str] = []
    seen: set = set()
    for request in reqs:
        lo, hi = _window_for(request, offset, game_span)
        ctx = _ModeSearchContext(service, lane=lane, hero=hero, lo=lo, hi=hi,
                                 top_k=cfg.top_k)
        call = htools.ToolCall(name="search_modes", query=request.query,
                               k=int(request.k or 0))
        result = htools.dispatch(call, ctx)
        payload = result.data if isinstance(result.data, dict) else {}
        entries = payload.get("entries") or []
        ids: List[str] = []
        for entry in entries:
            rid = str((entry or {}).get("id") or "").strip()
            if not rid or rid in seen:
                continue
            seen.add(rid)
            ids.append(rid)
            order.append(rid)
        out.hits.append({
            "query": request.query,
            "at": request.at,
            "source": request.source,
            "window": [lo, hi],
            "ids": ids,
            "ok": bool(result.ok),
            "note": result.note or "",
            "error": result.error or "",
            "degradation": dict(payload.get("degradation") or {}),
        })
        # **向量那路没走通** —— 检索能力降级了，得说出来。
        #
        # 稠密路死掉时检索只剩字面匹配（BM25 + 场景词）：说得不一样但
        # 意思相同的那条打法就找不到了。这个原因在检索层的 ``degradation``
        # 里，但**没人看得见** —— 冒泡成留痕，报告上才说得清。
        dense_error = str((payload.get("degradation") or {}).get("dense_error")
                          or "")
        if dense_error and dense_error not in str(
                out.degradation.get("dense_error") or ""):
            out.degradation["dense_error"] = dense_error
            out.notes.append("这次找攻略只按字面匹配（向量检索没走通：%s）。"
                             % dense_error)
        # **稀疏路失败单独说** —— 稠密路（向量）可能好好的，
        # 混进 ``dense_error`` 会让报告说「向量检索没走通」。
        sparse_error = str((payload.get("degradation") or {}).get("sparse_error")
                            or "")
        if sparse_error and sparse_error not in str(
                out.degradation.get("sparse_error") or ""):
            out.degradation["sparse_error"] = sparse_error
            out.notes.append("这次找攻略的字面匹配那路没走通（%s）。"
                             % sparse_error)
        if not result.ok:
            out.notes.append("检索「%s」失败：%s" % (request.query, result.error))
        elif not entries:
            reason = str((payload.get("degradation") or {}).get("reason") or "")
            out.notes.append("检索「%s」没召回到相关打法%s。"
                             % (request.query, ("（%s）" % reason) if reason else ""))

    if not order:
        if not cfg.allow_fallback:
            out.mode = "model"
            out.notes.append("模型提的检索请求都没召回到打法 —— 没有候选，没有裁决。")
            return out
        candidates, screened = _fallback(
            timeline, patterns, facts,
            "模型提的 %d 个检索请求都没召回到打法" % len(reqs), out.notes)
        out.mode = "fallback"
        out.candidates = candidates
        out.screened = screened
        return out

    # -- 3. 检索回来的 id → 候选 --
    by_id = {str(getattr(p, "id", "") or ""): p for p in patterns}
    candidates: List[Candidate] = []
    screened: List[RuleJudgement] = []
    for rid in order:
        pattern = by_id.get(rid)
        if pattern is None:
            out.notes.append("检索到的 %s 在模式库里找不到对应条目（跳过）。" % rid)
            continue
        candidate, dropped = screen_one(pattern, facts, in_game, game_span,
                                    model_premise=config.model_premise)
        if candidate is None:
            if dropped is not None:
                screened.append(dropped)
            continue
        candidates.append(candidate)

    # -- 3b. 检索回来的**一条都对不上** → 退回整条筛，并写明 --
    #
    # 2026-11-13 实测（``hok-video-8178dc031675``）：模型 6 个请求召回了
    # 8 条公孙离（发育路）的打法，可这一局画面上玩家在中路 ——
    # 8 条全被筛掉 → 候选 0 条 → 报告整页空着，只剩一句「没对上」。
    #
    # 这说明**检索这条路的答案不适用**（检索按英雄/分路召回，而这两样
    # 只要有一个读错，召回来的就全用不上）。这时宁可退回「按分路/时间窗
    # 整条筛」，也不要交一张空表 —— 但要**写明走的是哪条路**，
    # 因为整条筛里可能有这一局压根没发生的事（见 ``_fallback``）。
    if not candidates and screened:
        if not cfg.allow_fallback:
            out.mode = "model"
            out.notes.append("检索回来的打法一条都对不上这一局 —— 没有候选，没有裁决。")
            return out
        candidates, screened = _fallback(
            timeline, patterns, facts,
            "检索回来的 %d 条打法一条都对不上这一局" % len(order), out.notes)
        if not candidates:
            out.mode = "fallback"
            out.candidates = []
            out.screened = screened
            out.notes.append("退回整条筛之后也没有对得上的 —— 这一局没有可判的条目。")
            return out
        # **走的是整条筛这条路** —— 别在下面被改回 ``model``：
        # 报告层靠这个字段决定要不要写「这些条目不是模型自己找的」
        # （口径变了不说，读者会把「整条筛」当成「模型挑出来的」）。
        out.mode = "fallback"
        out.degradation = {
            "requests": len(reqs),
            "retrieved": len(order),
            "candidates": len(candidates),
            "not_applicable": len(screened),
            "fallback_reason": "检索回来的打法一条都对不上这一局",
        }
        out.candidates = list(candidates[:max(1, int(cfg.max_candidates or 0))])
        out.screened = screened
        return out

    limit = max(1, int(cfg.max_candidates or 0))
    if len(candidates) > limit:
        out.notes.append("检索回来 %d 条，按检索顺序截到 %d 条（模型写请求的顺序"
                         "就是它的优先级）。" % (len(candidates), limit))
        candidates = candidates[:limit]

    out.candidates = candidates
    out.screened = screened
    out.mode = "model"
    out.degradation = {
        "requests": len(reqs),
        "retrieved": len(order),
        "candidates": len(candidates),
        "not_applicable": len(screened),
    }
    out.notes.append("候选规则由模型自己提的检索请求找出来（%d 个请求 → %d 条打法）。"
                     % (len(reqs), len(candidates)))
    return out
