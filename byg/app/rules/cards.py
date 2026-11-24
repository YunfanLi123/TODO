# -*- coding: utf-8 -*-
"""**规则卡** —— 模式库给模型看的那一面（2026-11-13）。

为什么要有这一层
---------------
库里的数据有两代形状：v1 的 ``dict``（``app/seed/hok_modes.json`` 那种，
带 ``branches[]``）和 v2 的 ``Pattern``（``app/core/dsl2.py``）。
两代字段名还不一样（``lane`` / ``applies_to.lane``、``deviation_hint`` / ``advice``），
于是**每一处要「把规则讲给模型听」的地方都各拼一次字符串**：

* 裁决层 ``Candidate.rule_text()`` 拼的是标题/前提/要求/结果/建议；
* 综述层 ``guide_digest()`` 拼的是「标题 —— 要求」一行一条。

同一份规则在两个地方长成两个样子，模型读到的口径就不一致；更要命的是
**分支（``branches``）根本没进提示词** —— 而英雄层攻略的价值恰恰在分支：
「敌方红开就这样、蓝开就那样、前期弱就别接战」。只给第一条，
模型只能拿一句不适用本局的要求去判，然后判「你没做到」。

这一层做三件事：

1. **统一形状**：不管是 v1 dict 还是 v2 Pattern，出来都是同一张卡；
2. **带上分支**：``when → do → because`` 三件套，按条给；
3. **带上「看什么」**：新增 ``watch`` —— 判这条要看画面上的哪个东西。
   老数据没有这一栏就空着（不编）；以后从视频文案里抽出来时填进去。

这张卡既给人读（``to_block`` 的排版就是给模型读的话），也能落库
（``to_dict``），所以它同时是**提示词素材**和**数据集格式**。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

#: 分路码 → 人话。
_LANE_LABELS = {
    "jungle": "打野", "mid": "中路", "side": "对抗路",
    "carry": "发育路", "roam": "游走", "": "通用",
}

#: 一条卡里最多带几个分支（再多就是拿钱请模型读废纸）。
MAX_BRANCHES = 4

#: 给模型读时的单行宽度（超过就截断，保证一份提示词里每条不超过几行）。
MAX_LINE = 120


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _clip(text: str, limit: int = MAX_LINE) -> str:
    text = _text(text)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip("，。、； ") + "…"


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """对象或字典都取得到（v1 是 dict，v2 是 dataclass）。"""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _minutes(obj: Any) -> List[int]:
    """这条规则管的是第几分钟（``[lo, hi]``）。取不到就是空。

    两代形状的字段名不一样：v2 是 ``node.from_min/to_min``，
    v1 是 ``trigger.all`` 里那条 ``time_min between [lo, hi]``。
    """
    node = _get(obj, "node")
    lo = _get(node, "from_min") if node is not None else None
    hi = _get(node, "to_min") if node is not None else None
    if lo is None or hi is None:
        trig = _get(obj, "trigger") or {}
        for cond in (_get(trig, "all") or []):
            if str(_get(cond, "fact") or "") == "time_min":
                value = _get(cond, "value") or []
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    lo, hi = value[0], value[1]
                    break
    try:
        return [int(round(float(lo))), int(round(float(hi)))]
    except (TypeError, ValueError):
        return []


#: 老库里没有 ``lane`` 字段 —— 从 ``category``（「打野·前期」）或
#: ``trigger`` 里的 ``position`` 条件反推，**推不出来就是空**（不猜）。
_CATEGORY_LANES = (
    ("打野", "jungle"), ("中路", "mid"), ("对抗路", "side"), ("边路", "side"),
    ("发育路", "carry"), ("游走", "roam"),
)


def _lane_of(obj: Any) -> str:
    """这条规则属于哪条分路。三种来源，按可靠程度排。"""
    applies = _get(obj, "applies_to")
    lane = _get(obj, "lane") or (_get(applies, "lane") if applies else "") or ""
    if lane:
        return str(lane)
    # 2) trigger 里的 position 条件（最准 —— 那是筛选真正用的字段）
    trig = _get(obj, "trigger") or {}
    for cond in (_get(trig, "all") or []):
        if str(_get(cond, "fact") or "") == "position":
            value = str(_get(cond, "value") or "")
            if value:
                return value
    # 3) category 里的第一个词（老库只有这个）
    category = _text(_get(obj, "category"))
    for word, code in _CATEGORY_LANES:
        if category.startswith(word) or word in category[:6]:
            return code
    return ""


def _heroes(obj: Any) -> List[str]:
    applies = _get(obj, "applies_to")
    heroes = _get(applies, "heroes") if applies is not None else ()
    return [str(h) for h in (heroes or ()) if h]


def _branches(obj: Any) -> List[Dict[str, str]]:
    """分支：``[{when, do, because}]``。v1 有 ``branches[]``，v2 没有。"""
    out: List[Dict[str, str]] = []
    for item in (_get(obj, "branches") or []):
        if not isinstance(item, dict):
            continue
        when = _text(item.get("premise"))
        do = _text(item.get("action"))
        because = _text(item.get("outcome"))
        if not (when or do):
            continue
        out.append({"when": when, "do": do, "because": because})
        if len(out) >= MAX_BRANCHES:
            break
    return out


def rule_card(obj: Any, *, watch: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """把一条规则（v1 dict / v2 Pattern）压成**同一张卡**。

    ``watch`` 是「判这条要看画面上的什么」。没显式给就从**原文**里抽
    （:mod:`app.rules.watch_hints`）—— 抽不到就空着，**不编**。
    """
    lane = _lane_of(obj)
    minutes = _minutes(obj)
    source = _get(obj, "source") or {}
    if watch is None:
        watch = _watch_from_text(
            "。".join([_text(_get(source, "quoted")),
                       _text(_get(obj, "premise")),
                       _text(_get(obj, "action"))]))
    card: Dict[str, Any] = {
        "id": str(_get(obj, "id") or ""),
        "title": _text(_get(obj, "title")),
        "lane": lane,
        "lane_label": _LANE_LABELS.get(lane, lane or "通用"),
        "minutes": minutes,
        "heroes": _heroes(obj),
        "situation": _text(_get(obj, "premise")),
        "should": _text(_get(obj, "action")),
        "why": _text(_get(obj, "outcome")),
        "advice": _text(_get(obj, "deviation_hint") or _get(obj, "advice")),
        #: 这一步为什么有用（因果链，见 app.rules.mechanisms）。
        "why_chain": _text(_get(obj, "why_chain")),
        "branches": _branches(obj),
        "watch": [str(w) for w in (watch or []) if str(w).strip()],
        "source": {
            "guide": _text(_get(source, "guide")),
            "quoted": _text(_get(source, "quoted")),
            "url": _text(_get(source, "url")),
            "locator": _text(_get(source, "locator")),
        },
    }
    return card


def _watch_from_text(text: str) -> List[str]:
    """从原文里抽「看什么」。抽不到就是空列表（不编）。"""
    text = (text or "").strip("。 ")
    if not text:
        return []
    try:
        from . import watch_hints
    except Exception:                                  # noqa: BLE001
        return []
    return watch_hints.extract_watch(text)


def card_block(card: Dict[str, Any], *, compact: bool = False) -> str:
    """把卡排成给模型读的一段话。

    ``compact=True`` 用于「要点清单」（综述层要一次看几十条）：
    只留 `路 + 分钟 · 情形 → 该做`，一行一条。

    ⚠️ **有分支时不重复顶层字段**（2026-11-13）：v1 的
    ``premise``/``action`` 本来就是各分支拼起来的一长串，再跟分支列表
    一起打出来，一条规则能占掉半页提示词，模型反而抓不住重点。
    """
    title = card.get("title") or "（无标题）"
    lane = card.get("lane_label") or "通用"
    minutes = card.get("minutes") or []
    window = ("%d~%d 分钟" % (minutes[0], minutes[1])) if len(minutes) == 2 else ""
    head = "%s%s" % (lane, ("·" + window) if window else "")
    branches = card.get("branches") or []

    if compact:
        line = "%s｜%s" % (head, title)
        if branches:
            line += "｜%s → %s" % (_clip(branches[0].get("when") or "", 30),
                                   _clip(branches[0].get("do") or "", 50))
        elif card.get("situation") and card.get("should"):
            line += "｜%s → %s" % (_clip(card["situation"], 30),
                                   _clip(card["should"], 50))
        return _clip(line, 180)

    lines: List[str] = ["[打法 %s] %s" % (head, title)]
    heroes = card.get("heroes") or []
    if heroes:
        lines.append("  适用：%s" % "、".join(heroes[:6]))
    if branches:
        for branch in branches[:MAX_BRANCHES]:
            when = _clip(branch.get("when") or "（任意）", 40)
            do = _clip(branch.get("do") or "（未写）", 60)
            lines.append("  · 若 %s → %s" % (when, do))
            because = _clip(branch.get("because") or "", 40)
            if because:
                lines.append("    因为：%s" % because)
    else:
        if card.get("situation"):
            lines.append("  情形：%s" % _clip(card["situation"]))
        if card.get("should"):
            lines.append("  该做：%s" % _clip(card["should"]))
        if card.get("why"):
            lines.append("  为什么：%s" % _clip(card["why"]))
    if card.get("why_chain"):
        lines.append("  为什么有用：%s" % _clip(card["why_chain"], 140))
    if card.get("watch"):
        lines.append("  看什么：%s" % _clip("；".join(card["watch"]), 100))
    quoted = (card.get("source") or {}).get("quoted") or ""
    if quoted and not branches:
        lines.append("  出处原文：「%s」" % _clip(quoted, 60))
    return "\n".join(lines)


def cards_of(patterns: Iterable[Any], *, lane: str = "",
             limit: int = 0) -> List[Dict[str, Any]]:
    """一批规则 → 一批卡（可只留某条分路）。"""
    out: List[Dict[str, Any]] = []
    for pattern in patterns or []:
        card = rule_card(pattern)
        if lane and card.get("lane") and card["lane"] != lane:
            continue
        out.append(card)
        if limit and len(out) >= limit:
            break
    return out
