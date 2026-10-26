"""报告数据组装：把内核的结构化输出翻译成人能读的话。

这一层存在的理由很实际：内核只产出 ``Condition(fact="health", op=">=", value=50)``
这样的结构，报告要说的是「血量 ≥ 50 时应停手」。所有「怎么把结构说成人话」
的活集中在这里，模板只负责排版。
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from ..adapters.base import GameAdapter, get_adapter
from ..core.attribution import Gap, analyze_match
from ..core.context import MatchContext
from ..core.curve import backtest_version_change, build_curve, diff_versions
from ..core.dsl import Rule
from ..core.lane_analysis import (
    VERDICT_NEVER,
    VERDICT_OCCASIONAL,
    VERDICT_SYSTEMIC,
)
from ..core.matcher import STATUS_DEVIATED

from . import organize

#: 事实 -> 「事实名 + 操作符 + 阈值」的中文表达模板
_OP_TEXT = {
    "==": "为",
    "!=": "不为",
    ">": "高于",
    ">=": "不低于",
    "<": "低于",
    "<=": "不高于",
    "in": "属于",
    "not_in": "不属于",
    "between": "介于",
}


def describe_condition(adapter: GameAdapter, cond: Dict[str, Any]) -> str:
    """把单个条件渲染成中文短语，如「血量 不低于 50」。"""
    fact = adapter.describe_fact(cond.get("fact", ""))
    op = cond.get("op", "==")
    value = cond.get("value")
    op_text = _OP_TEXT.get(op, op)

    if op in ("in", "not_in") and isinstance(value, (list, tuple)):
        rendered = "、".join(adapter.describe_action(v) for v in value)
    elif op == "between" and isinstance(value, (list, tuple)) and len(value) == 2:
        rendered = f"{value[0]} 到 {value[1]}"
    else:
        rendered = adapter.describe_action(value) if op == "==" else str(value)

    return f"{fact} {op_text} {rendered}"


def describe_expected(adapter: GameAdapter, expected: Dict[str, Any]) -> str:
    return describe_condition(adapter, expected)


def describe_actual(adapter: GameAdapter, actual: Dict[str, Any]) -> str:
    """玩家实际做了什么。

    期望那一类一个都没做时，**要说清楚他当时在干什么**。
    早先这里只回一句「该做而没做（此回合无相关操作）」，可玩家明明做了
    别的（去抓人而不是刷野），报告却说他什么都没做 —— 「怎么改」那一条
    于是给不出方向，玩家只能自己猜哪里不对。
    """
    labels = []
    for a in actual.get("actions") or []:
        lbl = a.get("label") or adapter.describe_action(a.get("value"))
        labels.append(lbl)
    if labels:
        return "、".join(labels)

    others = []
    for a in actual.get("all_actions") or []:
        lbl = a.get("label") or adapter.describe_action(a.get("value"))
        if lbl and lbl not in others:
            others.append(lbl)
    if others:
        return "该做而没做（" + _now(adapter) + "做的是：" + "、".join(others) + "）"
    return "该做而没做（" + _now(adapter) + "无相关操作）"


# --------------------------------------------------------------------------
# 报告正文：三件事
#
# 报告里每一条偏离只讲三件事 ——
#   实际：客观描述这一回合做了什么
#   该做：按模式应执行的动作
#   为什么：结合这一局成立的前提，说明为什么该那么做
#
# 为什么不写成「原因：原文前提：…；原文后果：…」：
# 那是**引文**，不是判断。读者要看到的是「这一局第 8 分钟经济落后，
# 所以该只刷安全野区」，而不是「攻略的前提是落后，后果是翻盘」。
# 引文另有去处 —— 规则库页与审核工作台。
# --------------------------------------------------------------------------

#: 触发条件里不再写进「为什么」的事实 —— 分路在标题上，时间在时间节点上，
#: 英雄在分类标签上，再写一遍就是重复。
_PREMISE_SKIP_FACTS = {"position", "time_min", "phase", "my_hero"}

#: 动作事实的中文名，用于「这一回合做的是……」的句子。
_ACTION_FACT_TEXT = {
    "objective_action": "地图目标",
    "positioning_action": "站位",
    "teamfight_action": "团战处理",
}


def _unit(adapter) -> str:
    """这一局的操作单位。

    自走棋真的有「一回合」，王者荣耀没有 —— 一局王者是**连续的一整场**，
    切出来的只能是**时间片**。报告里说「这一回合」会让玩家去找回合，
    而王者里根本没这个东西。
    """
    return getattr(adapter, "segment_unit", None) or "回合"


def _now(adapter) -> str:
    """指代当前这一段的说法。"""
    return getattr(adapter, "segment_now", None) or "这一回合"


def describe_actual_sentence(
    adapter: GameAdapter, actual: Dict[str, Any]
) -> str:
    """实际做了什么 —— 一句主谓宾齐全的客观陈述。

    只描述动作，不评价对错：「做错了」是判定结论，不是描述。
    """
    fact = actual.get("fact") or ""
    labels = []
    for a in actual.get("actions") or []:
        lbl = a.get("label") or adapter.describe_action(a.get("value"))
        if lbl and lbl not in labels:
            labels.append(lbl)
    if labels:
        return f"{_now(adapter)}的操作是{'、'.join(labels)}。"

    others = []
    for a in actual.get("all_actions") or []:
        lbl = a.get("label") or adapter.describe_action(a.get("value"))
        if lbl and lbl not in others:
            others.append(lbl)
    if others:
        return f"{_now(adapter)}没有做{_ACTION_FACT_TEXT.get(fact, '该类操作')}，做的是{'、'.join(others)}。"
    return f"{_now(adapter)}没有记录到操作。"


def describe_expected_sentence(
    adapter: GameAdapter, expected: Dict[str, Any]
) -> str:
    """按模式应执行的动作 —— 一句话说明这条模式要什么。"""
    fact = expected.get("fact") or ""
    op = expected.get("op", "in")
    values = expected.get("value")
    if isinstance(values, (list, tuple)):
        labels = [adapter.describe_action(v) for v in values]
        rendered = "或".join(labels) if len(labels) > 1 else (labels[0] if labels else "")
    else:
        rendered = adapter.describe_action(values)
    if not rendered:
        return "这条模式没有给出具体动作。"
    if op in ("in", "not_in"):
        return f"这条模式要求{rendered}。"
    return f"这条模式要求{rendered}。"


def premise_facts(
    adapter: GameAdapter, evidence: List[Dict[str, Any]]
) -> List[str]:
    """这一局成立的前提，写成客观事实。

    只取**成立且有取值**的条件，并且跳过分路和时间 ——
    那两样已经在标题和时间节点上了。
    """
    out: List[str] = []
    for e in evidence or []:
        if e.get("group") not in (None, "all"):
            continue
        if not e.get("present") or not e.get("result"):
            continue
        fact = e.get("fact") or ""
        if fact in _PREMISE_SKIP_FACTS:
            continue
        actual = e.get("actual")
        if actual is None:
            continue
        out.append(f"{adapter.describe_fact(fact)} {adapter.describe_value(fact, actual)}")
    return out


#: 「为什么」里优先带的局面事实。
#:
#: 为什么需要这个兜底
#: ------------------
#: 攻略的模式**不带数值触发条件**了（「己方处于落后」这种前提画面里读不出来，
#: 硬写成一个 ``gold_diff < 0`` 的条件，规则在视频上就永远触发不了）。
#: 于是触发证据里没有经济差、击杀差这些事实，「为什么」就只剩一句引文 ——
#: 跟任何一局都成立，玩家读不出跟自己有没有关系。
#:
#: 所以证据里没有的时候，从这一回合的局面快照里取：差值类的事实最能说明
#: 「当时是什么局面」。
_FALLBACK_FACTS = ("gold_diff", "kill_diff", "tower_diff", "matchup",
                   "has_ult_ready", "in_combat", "level", "gold")


def situation_facts(
    adapter: GameAdapter, situation: Dict[str, Any], limit: int = 2
) -> List[str]:
    """从回合的局面快照里取几条能说明「当时什么局面」的事实。"""
    out: List[str] = []
    for name in _FALLBACK_FACTS:
        if name not in (situation or {}):
            continue
        value = situation.get(name)
        if value is None:
            continue
        text = adapter.describe_value(name, value)
        if not str(text).strip():
            continue
        out.append(f"{adapter.describe_fact(name)} {text}")
        if len(out) >= limit:
            break
    return out


def describe_why(
    adapter: GameAdapter,
    premise: str,
    hint: str,
    facts: List[str],
) -> str:
    """为什么该这么做 —— 客观前提 + 这一局的实际取值 + 做法与理由。

    写法上有三条硬要求：

    1. **主谓宾齐全**，每句都能独立读懂。
    2. **只陈述**，不下判断。「应该」只出现在模式的指令里，
       那是模式要求，不是作者在评价玩家。
    3. **带上这一局的实际取值**（如「经济差 -300」）——
       没有它，这段话跟任何一局都成立，读者无法判断跟自己有没有关系。

    正文用映射表里手写的那段散文（``hint``），
    不用原文的「动作 + 后果」拼接 —— 后者在多分支合并的规则上
    会拼成一张决策表（三个动作用分号连着），读起来像规格书而不像人话；
    而且 85 条里有 52 条的后果已经写进散文里了，再补一遍会重复，
    遇到「放弃河蟹，拿到 1 分钟河蟹」这种合并规则甚至会自相矛盾。
    """
    parts: List[str] = []

    premise = (premise or "").strip()
    # 攻略原文用「A + B」表示两个条件同时成立。
    # 印到报告里要写成「A，B」—— 加号是提纲符号，不是句子。
    premise = premise.replace(" + ", "，")

    body = (hint or "").strip().rstrip("。")

    # 正文自己就带着条件（「己方处于落后时只刷…」），
    # 前提再写一遍就重复了 —— 这时候前面只留这一局的实际取值。
    clauses = [c for c in premise.split("，") if len(c) >= 4]
    if clauses and any(body.startswith(c) for c in clauses):
        premise = ""

    if facts:
        head = _now(adapter) + "、".join(facts)
        # 前提是多支的时候不拼进去 —— 那是一张分支清单，不是句子。
        if premise and "；" not in premise:
            head += f"，{premise}"
        parts.append(head + "。")

    if body:
        parts.append(body + "。")

    if not parts:
        return "这条模式没有写明前提。"
    return "".join(parts)


def describe_situation(adapter: GameAdapter, situation: Dict[str, Any]) -> str:
    """当时局面的一句话摘要。

    两条硬规矩：

    1. **没填的取值不占位。** 空值意味着「玩家没填」，
       印出来会变成「… · 我方英雄 · 对位优劣 …」这种后面空着的标签，
       读起来像内容被截断了，而不是「这一项没有」。
    2. **取值要说人话。** 分路要写「打野」不是 ``jungle``，
       对位优劣要写「不确定」不是 ``unknown`` —— 这些事实不是动作，
       走 ``describe_action`` 只会原样吐出内部代码。
    """
    if not situation:
        return "局面数据缺失"
    parts = []
    for fact, value in situation.items():
        if value is None:
            continue
        text = adapter.describe_value(fact, value)
        if not str(text).strip():
            continue
        parts.append(f"{adapter.describe_fact(fact)} {text}")
    return " · ".join(parts)


def describe_gap(
    adapter: GameAdapter,
    gap: Gap,
    evidence: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """一条偏离的呈现：实际 / 该做 / 为什么，外加准确的时间节点。"""
    d = gap.to_dict()
    d["expected_text"] = describe_expected_sentence(adapter, gap.expected)
    d["actual_text"] = describe_actual_sentence(adapter, gap.actual)
    d["why_text"] = describe_why(
        adapter,
        gap.premise,
        gap.hint,
        # 触发证据里没有局面事实时（模式不带数值触发条件了），
        # 退回这一回合的局面快照 —— 「为什么」要带这一局的实际取值。
        premise_facts(adapter, evidence or [])
        or situation_facts(adapter, gap.situation),
    )
    d["time_label"] = gap.round_label
    d["headline"] = gap.headline(_unit(adapter))
    return d


def _describe_evidence(adapter: GameAdapter, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for e in evidence:
        out.append(
            {
                "fact_label": adapter.describe_fact(e.get("fact", "")),
                "op": e.get("op"),
                "threshold": e.get("threshold"),
                "actual": e.get("actual"),
                "present": e.get("present", True),
                "result": e.get("result"),
                "group": e.get("group"),
            }
        )
    return out


def _recognition_notes(ctx: MatchContext) -> Dict[str, Any]:
    """从原始数据里取出视频识别的诊断信息。

    识别质量必须跟着结论一起走 —— 玩家有权知道「血量 58」是读出来的还是
    猜出来的。内核不做这件事（它只认事实是否存在），所以由报告层从
    ``raw.recognition`` 里取。
    """
    rec = (ctx.raw or {}).get("recognition")
    if not rec:
        return {}
    degradation = rec.get("degradation") or {}
    return {
        "recognizer_id": rec.get("recognizer_id", ""),
        "recognizer_version": rec.get("recognizer_version", ""),
        # **哪一个模型读出来的**。换一版模型结论就可能不同 ——
        # 读者得知道这局是哪套得出的，否则没法判断可信度。
        "model": rec.get("model", ""),
        "provider": rec.get("provider", ""),
        "stats": rec.get("stats", {}),
        "usable_rounds": degradation.get("usable_rounds", 0),
        "total_rounds": degradation.get("total_rounds", 0),
        "dropped_rounds": degradation.get("dropped_rounds", 0),
        "low_confidence_facts": degradation.get("low_confidence_facts", []),
        "rejected_facts": degradation.get("rejected_facts", []),
        "dropped_actions": degradation.get("dropped_actions", []),
        "warnings": degradation.get("warnings", []),
        # 原视频 id。报告里的「看这一段」靠它拼视频地址 ——
        # 手工录入的对局这里是空，界面上就不显示那个入口。
        "video_id": rec.get("video_id", ""),
    }


def _low_confidence_facts_of(ctx: MatchContext, gaps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把低置信度事实与「用到它的差距」关联起来。

    只列一张孤立的事实清单没用 —— 玩家需要知道的是「哪个结论是建立在这条
    不可靠的识别之上的」。
    """
    notes = _recognition_notes(ctx)
    flagged = {
        (item.get("round_label"), item.get("fact"))
        for item in notes.get("low_confidence_facts", [])
    }
    if not flagged:
        return []

    out: List[Dict[str, Any]] = []
    for gap in gaps:
        # gap 的 situation 是触发时的局面快照，键即事实名
        hits = [
            item
            for item in notes["low_confidence_facts"]
            if item.get("round_label") == gap.get("round_label")
            and item.get("fact") in gap.get("situation", {})
        ]
        if hits:
            out.append(
                {
                    "round_label": gap.get("round_label"),
                    "rule_id": gap.get("rule_id"),
                    "rule_title": gap.get("rule_title"),
                    "facts": hits,
                }
            )
    return out


def _mode_actual(adapter: GameAdapter, verdicts: List[Dict[str, Any]]) -> str:
    """这条模式涉及的那几个回合里，玩家实际在做哪几件事。

    取**所有**相关回合（做到的和没做到的都算），而不是只看偏离的 ——
    只看偏离回合的话，做对了的模式会得到一句「没有记录到操作」，
    那显然不对：那一回合他做了，而且做对了。
    """
    counts: Dict[str, int] = {}
    for v in verdicts:
        actual = v.get("actual") or {}
        pool = actual.get("actions") or actual.get("all_actions") or []
        for a in pool:
            code = a.get("value") or a.get("label") or ""
            if not code:
                continue
            name = a.get("label") or adapter.describe_action(code)
            counts[name] = counts.get(name, 0) + 1
    if not counts:
        return "这些" + _unit(adapter) + "没有记录到操作。"
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:2]
    return "这些" + _unit(adapter) + "的操作是" + "、".join(n for n, _ in top) + "。"


def _lane_text(
    adapter: GameAdapter,
    lanes: Dict[str, Any],
    verdicts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """给分路结论补上「时间节点 + 实际 / 该做 / 为什么」。

    分路结论是**按模式聚合**的（一条模式可能横跨好几个回合），
    所以实际动作取这些回合里出现最多的那一两项，
    而前提取值取第一个偏离回合的 —— 同一时间窗内局面差别不大，
    取哪一个都不算编造，但必须写清楚是「这一回合」。
    """
    by_rule: Dict[str, List[Dict[str, Any]]] = {}
    for v in verdicts:
        by_rule.setdefault(v.get("rule_id") or "", []).append(v)

    def one(finding: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(finding)
        rule_id = out.get("mode_id") or ""
        vs = by_rule.get(rule_id, [])
        deviated = [v for v in vs if v.get("status") == "deviated"] or vs

        facts: List[str] = []
        if deviated:
            facts = (premise_facts(adapter, deviated[0].get("evidence") or [])
                     or situation_facts(adapter, deviated[0].get("context") or {}))

        out["actual_text"] = _mode_actual(adapter, vs)

        should = (out.get("should_do") or "").strip()
        out["expected_text"] = (
            f"这条模式要求{should}。" if should else "这条模式没有给出具体动作。"
        )

        out["why_text"] = describe_why(
            adapter,
            out.get("premise") or "",
            out.get("how_to_fix") or "",
            facts,
        )
        out["time_label"] = "、".join(out.get("round_labels") or [])
        return out

    for group in lanes.get("lanes", []):
        group["findings"] = [one(f) for f in group.get("findings", [])]
    for c in lanes.get("cells", []):
        c["finding"] = one(c["finding"])
    lanes["findings"] = [one(f) for f in lanes.get("findings", [])]
    return lanes


def _with_media(lanes: Dict[str, Any], video_id: str) -> Dict[str, Any]:
    """给分路结论补上原视频的定位信息。

    没有 video_id（手工录入的对局）就**原样返回**并清空 moments ——
    此时报告不该出现一个点开是 404 的播放入口，那比没有更糟：
    它会让「结论指向了一段不存在的画面」。
    """
    if not video_id:
        for group in lanes.get("lanes", []):
            for f in group.get("findings", []):
                f["moments"] = []
        for c in lanes.get("cells", []):
            c["finding"]["moments"] = []
        lanes["video_id"] = ""
        lanes["video_url"] = ""
        return lanes

    def decorate(finding: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(finding)
        moments = []
        for m in out.get("moments") or []:
            img = m.get("image") or ""
            moments.append({
                **m,
                "image_url": (
                    f"/api/videos/{video_id}/frames/{img}" if img else ""
                ),
            })
        out["moments"] = moments
        return out

    for group in lanes.get("lanes", []):
        group["findings"] = [decorate(f) for f in group.get("findings", [])]
    for c in lanes.get("cells", []):
        c["finding"] = decorate(c["finding"])
    lanes["findings"] = [decorate(f) for f in lanes.get("findings", [])]
    lanes["video_id"] = video_id
    lanes["video_url"] = f"/api/videos/{video_id}/file"
    return lanes


# --------------------------------------------------------------------------
# 操作模式分析：分路结论 + 逐回合差距，合成一份
# --------------------------------------------------------------------------

#: 「原视频被建议部分片段」播放多长（秒）。
#:
#: 十秒足够看出一个操作，也短到不会让人以为要看完一整局。
CLIP_SECONDS = 10

#: 攻略原文里的「状态【…】」是给下一个节点做引用输入的内部标记，
#: 报告里读不懂也不该出现，渲染前去掉。
_STATE_TAG_RE = re.compile(r"[，,；;]?\s*状态【[^】]*】")

#: 会出现在报告里的偏离结论。只有这三种算「有问题」。
_PROBLEM_VERDICTS = (VERDICT_SYSTEMIC, VERDICT_NEVER, VERDICT_OCCASIONAL)


def _round_minutes(ctx: MatchContext) -> Dict[str, int]:
    """回合标签 -> 局内分钟。

    取不到时间的回合直接不在这张表里 —— 报告里的时间区间宁可少一个端点，
    也不能拿别的回合的时间来凑。
    """
    out: Dict[str, int] = {}
    for r in ctx.rounds:
        value = r.facts.get("time_min")
        if value is None:
            continue
        try:
            out[r.label] = int(value)
        except (TypeError, ValueError):
            continue
    return out


def _time_range_text(minutes: List[Optional[int]]) -> str:
    """对局时间区间 —— 「第 18 分钟」或「第 18 ~ 20 分钟」。"""
    real = [m for m in minutes if m is not None]
    if not real:
        return ""
    lo, hi = min(real), max(real)
    if lo == hi:
        return f"第 {lo} 分钟"
    return f"第 {lo} ~ {hi} 分钟"


def _segment_actual_text(adapter: GameAdapter, deviated: List[Any],
                         absent: Optional[List[str]] = None) -> str:
    """这一段实际做了什么 —— 报告里的「原视频操作的文字描述」。

    只描述动作，不评价对错 —— 对错是判定结论，写进描述里就成了替玩家
    下判断。「做不到位」那层意思交给「建议操作描述」那一部分。
    """
    counts: Dict[str, int] = {}
    for v in deviated:
        actual = v.actual or {}
        pool = actual.get("actions") or actual.get("all_actions") or []
        for a in pool:
            code = a.get("value") or ""
            if not code:
                continue
            name = a.get("label") or adapter.describe_action(code)
            counts[name] = counts.get(name, 0) + 1
    if not counts:
        names = []
        for key in absent or []:
            value = str(key).split(":")[-1]
            if value:
                names.append(adapter.describe_action(value))
        if names:
            return "这一段画面里没看到" + "、".join(dict.fromkeys(names)) + "。"
        return "这一段没有记录到操作。"
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
    return "这一段做的是" + "、".join(name for name, _ in top) + "。"


def _advice_text(finding: Dict[str, Any]) -> str:
    """根据模式库建议操作的描述 —— 该做什么，写到具体步骤。

    用的是模式库里的 ``deviation_hint``：分路层是手写的人话，
    英雄层是攻略原文的具体操作。两边都是「怎么做」，不是「做什么类别」——
    只说「该刷野」等于没说，玩家要的是「先蓝后红、33 秒进野区、
    手里捏着惩」这种能照着做的句子。

    英雄层的建议是**条件式**的（「敌方打野红开时：己方蓝开反向开局……」），
    一条分支一句。挤成一整段的话，几种情形会糊在一起，
    玩家读不出哪条适用 —— 条件是分开的关键，不是标点。

    前面不再加「应该」：模式库里的话本身就有「敌方红开就己方蓝开」
    「需要绕到敌方一塔后就绕后断对方兵线」这种主语，
    硬加「应该」会变成「应该需要绕到…」这种别扭句子。
    这一栏的标题已经说明了它是建议。
    """
    hint = (finding.get("how_to_fix") or "").strip()
    if hint:
        return hint if hint.endswith(("。", "！", "？")) else hint + "。"
    should = (finding.get("should_do") or "").strip()
    if should:
        return f"这条模式要求{should}。"
    return "这条模式没有写明该怎么做。"


def _split_clauses(text: str) -> List[str]:
    """把一段话拆成一句一句 —— 报告里一行一句，读起来才不挤。

    条件式建议尤其需要：几句「什么情况下：做什么」排在一起，
    玩家要能一眼看出哪句对应哪个条件。
    """
    out: List[str] = []
    for part in re.split(r"(?<=。)", text or ""):
        part = part.strip()
        if part:
            out.append(part)
    return out



def _reason_text(
    adapter: GameAdapter,
    finding: Dict[str, Any],
    deviated: List[Any],
) -> str:
    """建议理由 —— 为什么这一段该那么打。

    两段话拼起来：

    1. **这一局的实际情况**。取自触发证据里的实际取值（经济差多少、
       大招有没有就绪…）。没有它这段话跟任何一局都成立，
       玩家没法判断跟自己有没有关系。
    2. **这么做的收益**。取自模式库的 ``outcome``，去掉攻略内部用的
       「状态【…】」标记 —— 那是给下一个节点做引用输入的，
       印在报告里读者读不懂。
    """
    parts: List[str] = []

    facts: List[str] = []
    if deviated:
        facts = premise_facts(adapter, deviated[0].evidence or [])
    if facts:
        parts.append("这一段" + "、".join(facts) + "。")

    outcome = _STATE_TAG_RE.sub("", finding.get("outcome") or "").strip()
    outcome = outcome.replace("；", "，").rstrip("。")
    if outcome:
        parts.append(f"照这个打法，{outcome}。")

    if not parts:
        return "这条模式没有写明理由。"
    return "".join(parts)


def _round_reviews(ctx: MatchContext) -> List[Dict[str, Any]]:
    """取出**模型对攻略清单的裁决**（带回合坐标）。

    裁决存在每个回合的 ``review`` 里 —— 它是**窗级**的结论
    （模型对这一窗的十几帧做过一次综合判断），落在窗对应的回合上。

    从 :attr:`MatchContext.rounds` 取而不是从 ``raw`` 取：
    落库时存的是规范化后的上下文（``ctx.to_dict()``），
    回合记录才是那条路上唯一稳定的地方。
    """
    out: List[Dict[str, Any]] = []
    for rnd in ctx.rounds:
        items = getattr(rnd, "review", None) or []
        if not items:
            continue
        out.append({
            "label": rnd.label_override or rnd.label,
            "time_index": rnd.time_index,
            "video_time": rnd.video_time,
            "image": rnd.image,
            "items": items,
        })
    return out


def _review_entries(
    adapter: GameAdapter,
    ctx: MatchContext,
    video_id: str,
    video_url: str,
) -> List[Dict[str, Any]]:
    """模型裁决 -> 报告条目。

    这一层是报告正文**主要**的来源。

    为什么不是只靠规则引擎
    ----------------------
    攻略的分支前提（「塔的早期护盾已消失」「暴君倒计时进入最后一段」
    「对方有人阵亡未复活」）**画面里读不出来**。规则引擎只能判那些
    前提能从数值或枚举判出来的模式 —— 王者 S45 分路层 84 条里只剩 9 条，
    报告于是几乎是空的。

    模型能看着画面判断这些前提：**看得见成立**才纠正，看不见就说
    「看不出来」。这正是「只指出改正之后一定会进步的模式」的做法。

    每一条都自带：模型看到的那几帧（片段指向它们）、玩家在做什么、
    该做什么（攻略原文）、为什么（前提 + 后果）。
    """
    entries: List[Dict[str, Any]] = []
    seen: set = set()

    #: 玩家打哪条线。分路已知时，**别的线**的模式一条都不进报告 ——
    #: 分路没读到时清单是全线的，模型可能对着别的线的打法说话。
    player_lane = str((ctx.player or {}).get("position") or "").strip()

    for rnd in _round_reviews(ctx):
        time_range = rnd.get("label") or ""
        for item in rnd.get("items") or []:
            verdict = item.get("verdict")
            if verdict not in ("deviation", "never"):
                continue
            kind = item.get("kind") or "mode"
            mode_id = item.get("mode_id") or ""
            if kind == "mode":
                if not mode_id or mode_id in seen:
                    continue
                if player_lane and item.get("lane") and item["lane"] != player_lane:
                    continue
                seen.add(mode_id)
            elif kind != "node":
                continue

            # 片段：**模型看到那件事的那一帧**。文字描述说的就是这一帧，
            # 播放器放的也是这一帧 —— 两者必须同一时刻。
            moments = []
            evidence = []
            seen_evidence: set = set()
            for m in item.get("moments") or []:
                seconds = m.get("t") or 0
                img = m.get("image") or ""
                moments.append({
                    "label": time_range,
                    "time": seconds,
                    "time_label": "%d:%02d" % (int(seconds) // 60, int(seconds) % 60),
                    "image_url": (f"/api/videos/{video_id}/frames/{img}"
                                  if video_id and img else ""),
                })
                # 同一句话挂在好几帧上时只留一次 ——
                # 「看到了什么」重复两遍，玩家只会觉得是程序凑的。
                why = (m.get("why") or "").strip()
                if why and why in seen_evidence:
                    continue
                if why:
                    seen_evidence.add(why)
                evidence.append({
                    "time": seconds,
                    "time_label": "%d:%02d" % (int(seconds) // 60, int(seconds) % 60),
                    "why": why,
                    "kind": "",
                    "value": "",
                })
            if not moments:
                continue

            did = (item.get("did") or "").strip()
            actual_text = f"这一段做的是{did}。" if did else "这一段的操作见下面的片段。"

            if kind == "node":
                # 整段没做这个时段该做的任何一件事 —— 建议是「挑一条做」，
                # 不是「该做某一条」。攻略在这一段给的是**菜单**，
                # 按当时的局面选哪条都行，所以不能钉死一条去指责。
                options = item.get("options") or []
                titles = [o.get("title") for o in options if o.get("title")]
                advice = ("这一段攻略给的是：" + "；".join(titles) +
                          "。按当时的局面挑一条做。") if titles else ""
                reason = (item.get("why") or "").strip()
                if reason and not reason.endswith(("。", "！", "？")):
                    reason += "。"
                title = "这一段没做这个时段该做的任何一件事"
                category = item.get("lane_label") or ""
                verdict_label = "整段没做"
            else:
                # 建议：**这一局适用的那一支**，不是整张菜单。
                #
                # 玩家拿到「敌方红开就己方蓝开，敌方蓝开就己方红开」这样
                # 一张菜单，配上「敌方蓝开，我方红开」这个理由，读完只知道
                # 自己被判了偏离，却不知道自己错在哪一支 —— 于是报告不敢信。
                advice = (item.get("branch_action") or "").strip()
                if not advice:
                    advice = (item.get("advice") or "").strip()
                if not advice:
                    advice = (item.get("action") or "").strip()
                if advice and not advice.endswith(("。", "！", "？")):
                    advice += "。"
                reason = (item.get("why") or "").strip()
                # 「适用的是第几支」写进理由 —— 菜单与结论要对得上号。
                try:
                    branch_no = int(item.get("branch_no") or 0)
                except (TypeError, ValueError):
                    branch_no = 0
                branch_premise = (item.get("branch_premise") or "").strip()
                if branch_no and branch_premise:
                    head = f"这一局适用的是第 {branch_no} 支：{branch_premise}。"
                    reason = (head + reason).strip() if reason else head
                outcome = _STATE_TAG_RE.sub(
                    "", item.get("branch_outcome") or item.get("outcome") or ""
                ).strip()
                # 原文里相邻分支的后果常写「同上」（指上一支）——
                # 单拎一支出来读就断了。这种时候用整条模式的后果。
                if outcome.startswith("同上"):
                    outcome = _STATE_TAG_RE.sub(
                        "", item.get("outcome") or "").strip()
                outcome = outcome.replace("；", "，").rstrip("。")
                if outcome:
                    # 模型写的依据与攻略的后果是两句 —— 拼在一起要断句，
                    # 否则读起来是一长串没有标点的话。
                    if reason and not reason.endswith(("。", "！", "？")):
                        reason += "。"
                    reason = (reason + f"照这个打法，{outcome}。").strip()
                title = item.get("title") or ""
                category = item.get("lane_label") or ""
                verdict_label = "有偏离"

            entries.append({
                "mode_id": mode_id,
                "category": category,
                "title": title,
                "lane_label": item.get("lane_label") or "",
                "phase_label": "",
                "verdict_label": verdict_label,
                "time_range": time_range,
                "clip": {
                    "video_url": video_url,
                    "moments": moments,
                    "seconds": CLIP_SECONDS,
                },
                "actual_text": actual_text,
                "evidence": evidence,
                "absent_kinds": [],
                "advice_text": advice,
                "advice_lines": _split_clauses(advice),
                "reason_text": reason or "这条结论的依据见上面的片段。",
                # 模型自评的置信度（不是识别置信度 —— 识别置信度是另一个口径）。
                "low_confidence": float(item.get("confidence") or 1.0) < 0.5,
                "confidence": float(item.get("confidence") or 1.0),
            })
    return entries


def build_mode_analysis(
    adapter: GameAdapter,
    ctx: MatchContext,
    lanes: Dict[str, Any],
    verdicts: List[Any],
    rules: Optional[Sequence[Rule]] = None,
    #: **能拿来纠正玩家**的模式 id 集合。
    #:
    #: ``None`` 表示**不筛** —— 玩家自己录的对局，事实是他给的，
    #: 按规则库字面判即可。用户原话是「**视频**做的没错与数据库不一致的
    #: 不用纠正」，那道闸管的是模型看画面猜出来的那种（视频来源的对局）。
    corrective: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """把分路结论与逐回合差距合成一份「操作模式分析」。

    每一条四部分：

    1. **对局时间区间** —— 这一条覆盖本局哪一段时间；
    2. **原视频被建议部分片段** —— 出问题那几秒的画面；
    3. **原视频操作的文字描述** —— 这一段实际做了什么；
    4. **根据模式库建议操作的描述 + 建议理由** —— 该做什么、为什么。

    只收**有偏离**的条目。做对了的条目没有片段也没有改法，
    四部分凑不齐 —— 留着只会让报告变成一张记分表，
    而这页要回答的是「我该怎么改」。

    两个来源都要收，缺一不可：

    - **分路结论**（``lanes['findings']``）—— 按分路 × 阶段收拢的运营结论；
    - **逐回合差距** —— 那些**没有分路结论**的规则。

    第二份来源不是补丁。分路结论只覆盖「能归到某一条分路」的模式
    （``position`` 条件恰好一个值），自走棋那种按阶段编号的规则、
    以及「全位置通用」的模式都归不进去 —— 它们的偏离如果没人收，
    报告就会对着一局有偏离的对局说「没有需要改的地方」。
    """
    by_rule: Dict[str, List[Any]] = {}
    for v in verdicts:
        by_rule.setdefault(v.rule_id, []).append(v)

    rule_by_id: Dict[str, Rule] = {r.id: r for r in (rules or [])}

    minutes_of = _round_minutes(ctx)
    video_url = lanes.get("video_url") or ""
    video_id = lanes.get("video_id") or ""

    #: 回合标签 -> 回合本身。逐回合那一路要自己找片段位置 ——
    #: 分路结论那边已经带着 moments 了，这一路没有。
    rounds_by_label: Dict[str, Any] = {r.label: r for r in ctx.rounds}

    #: (回合标签, 事实名) -> 该事实的置信度。
    #:
    #: 识别不确定这件事必须看得见 —— 一份掺了猜测却看不出哪些是猜测的
    #: 报告，比不给报告更糟。但它不该再单独占一节（报告正文只有
    #: 「操作模式分析」一节），所以标记直接挂在用它的条目上。
    #:
    #: 键带上**事实名**，是因为「这一回合有个低置信度的事实」不等于
    #: 「这条结论建立在一个猜测之上」：同一回合里所有规则都沾上标记，
    #: 标记就等于没有标记 —— 玩家看到的每一条都写着「置信度不足」，
    #: 反而看不出哪一条真的该打问号。
    flagged: Dict[tuple, float] = {}
    for item in _recognition_notes(ctx).get("low_confidence_facts") or []:
        label = item.get("round_label") or ""
        fact = item.get("fact") or ""
        try:
            conf = float(item.get("confidence") or 0)
        except (TypeError, ValueError):
            conf = 0.0
        key = (label, fact)
        flagged[key] = min(conf, flagged[key]) if key in flagged else conf

    def _weak_of(deviated: List[Any]) -> List[float]:
        """这批偏离里，哪些结论用到了低置信度的事实。

        判据是「这条规则**用到**了这个事实」—— 拿证据里的事实名
        去对，不是「这个回合有过低置信度的事实」。
        """
        weak: List[float] = []
        for v in deviated:
            for e in v.evidence or []:
                conf = flagged.get((v.round_label, e.get("fact") or ""))
                if conf is not None:
                    weak.append(conf)
        return weak

    entries: List[Dict[str, Any]] = []
    covered: set = set()

    # **模型裁决优先**。
    #
    # 规则引擎判的是「前提能从数值或枚举判出来」的那几条；模型判的是
    # 「前提能不能在画面里被看见」。后者覆盖面大得多，而且每一句都
    # 带着它据以判断的那几帧 —— 文字与片段天然对得上号。
    #
    # 同一件事两条都判了，以模型的为准：它看过画面，知道前提成不成立。
    review_entries = _review_entries(adapter, ctx, video_id, video_url)
    entries.extend(review_entries)
    covered.update(e["mode_id"] for e in review_entries if e.get("mode_id"))

    for finding in lanes.get("findings") or []:
        mode_id = finding.get("mode_id") or ""
        # 有分路结论的模式以分路结论为准 —— 逐回合那层只是它的明细，
        # 同一件事不该在报告里出现两条。
        covered.add(mode_id)
        if mode_id in {e.get("mode_id") for e in review_entries}:
            continue
        if finding.get("verdict") not in _PROBLEM_VERDICTS:
            continue
        if rules and corrective is not None and mode_id not in corrective:
            # 前提读不出来的模式：这一局他做的可能就是对的，不纠正。
            continue
        vs = by_rule.get(mode_id, [])
        deviated = [v for v in vs if v.status == STATUS_DEVIATED]
        if not deviated:
            continue

        moments = []
        for m in finding.get("moments") or []:
            seconds = m.get("time") or 0
            moments.append({
                "label": m.get("label") or "",
                "time": seconds,
                "time_label": "%d:%02d" % (int(seconds) // 60, int(seconds) % 60),
                "image_url": m.get("image_url") or "",
            })

        weak: List[float] = _weak_of(deviated)
        advice = _advice_text(finding)

        # 证据：这一段里探测器在画面上看到了什么（第几秒）。
        # 没有它，玩家只能看到一句「没做到」，没法回看原画面确认。
        evidence = []
        for m in (finding.get("evidence") or [])[:6]:
            seconds = m.get("t") or 0
            evidence.append({
                "time": seconds,
                "time_label": "%d:%02d" % (int(seconds) // 60, int(seconds) % 60),
                "why": m.get("why") or "",
                "kind": m.get("kind") or "",
                "value": m.get("value") or "",
            })

        entries.append({
            "mode_id": mode_id,
            "category": finding.get("category") or "",
            "title": finding.get("title") or "",
            "lane_label": finding.get("lane_label") or "",
            "phase_label": finding.get("phase_label") or "",
            "verdict_label": finding.get("verdict_label") or "",
            # 1. 对局时间区间
            # 窗口标签优先（「12:00 之后」「6:00 ~ 10:00」）——
            # 规则管的是一整段，只写「第 12 分钟」会把范围说成时点。
            "time_range": _round_range_text(
                [minutes_of.get(v.round_label) for v in deviated],
                [v.round_label for v in deviated if v.round_label],
                _unit(adapter),
            ),
            # 2. 原视频被建议部分片段
            "clip": {
                "video_url": video_url,
                "moments": moments,
                "seconds": CLIP_SECONDS,
            },
            # 3. 原视频操作的文字描述
            "actual_text": _segment_actual_text(
                adapter, deviated, finding.get("absent_kinds")),
            # 画面里看到什么（证据时刻）
            "evidence": evidence,
            # 明确没做的动作类 —— 报告里要写「画面里没看到」，
            # 而不是「没记录到操作」（后者听着像玩家没勾）。
            "absent_kinds": list(finding.get("absent_kinds") or []),
            # 4. 根据模式库建议操作的描述 + 建议理由
            "advice_text": advice,
            # 建议按句拆开：条件式建议挤成一段，条件与做法的对应关系就糊了。
            "advice_lines": _split_clauses(advice),
            "reason_text": _reason_text(adapter, finding, deviated),
            # 这条结论是不是建立在低置信度的识别之上
            "low_confidence": bool(weak),
            "confidence": min(weak) if weak else 1.0,
        })

    # 逐回合差距：没有分路结论的规则，单独成条。
    for mode_id, vs in by_rule.items():
        if mode_id in covered:
            continue
        deviated = [v for v in vs if v.status == STATUS_DEVIATED]
        if not deviated:
            continue
        rule = rule_by_id.get(mode_id)
        if rule is None or rule.status != "active":
            continue
        if rules and corrective is not None and mode_id not in corrective:
            continue

        labels = [v.round_label for v in deviated if v.round_label]
        moments = []
        for v in deviated:
            if not v.round_label:
                continue
            r = rounds_by_label.get(v.round_label)
            if r is None or r.video_time <= 0:
                continue
            moments.append({
                "label": v.round_label,
                "time": r.video_time,
                "time_label": "%d:%02d" % (int(r.video_time) // 60,
                                          int(r.video_time) % 60),
                "image_url": (
                    f"/api/videos/{video_id}/frames/{r.image}"
                    if video_id and r.image else ""
                ),
            })

        weak: List[float] = _weak_of(deviated)

        finding = _rule_as_finding(rule, adapter)
        advice = _advice_text(finding)
        entries.append({
            "mode_id": mode_id,
            "category": rule.category or "",
            "title": rule.title or "",
            "lane_label": "",
            "phase_label": "",
            "verdict_label": "有偏离",
            "time_range": _round_range_text(
                [minutes_of.get(v.round_label) for v in deviated], labels,
                unit=_unit(adapter),
            ),
            "clip": {
                "video_url": video_url,
                "moments": moments,
                "seconds": CLIP_SECONDS,
            },
            "actual_text": _segment_actual_text(
                adapter, deviated,
                [k for v in deviated if v.round_label
                 for k in (rounds_by_label.get(v.round_label).probe_absent
                           if rounds_by_label.get(v.round_label) else [])]),
            "evidence": [],
            "absent_kinds": [],
            "advice_text": advice,
            "advice_lines": _split_clauses(advice),
            "reason_text": _reason_text(adapter, finding, deviated),
            "low_confidence": bool(weak),
            "confidence": min(weak) if weak else 1.0,
        })

    # 时间区间从早到晚排 —— 玩家是照着时间轴改自己的操作的。
    entries.sort(key=lambda e: _range_start(e.get("time_range") or ""))
    return entries


def _rule_as_finding(rule: Rule, adapter: GameAdapter) -> Dict[str, Any]:
    """把一条规则摆成分路结论那种形状，好让两处共用同一套话术。

    没有分路结论可依的规则也值得用同样口径讲：建议怎么打（``deviation_hint``）、
    要求做到什么（``expect_value``）、照这么打的结果（``outcome``）。
    """
    values = rule.expect_value
    values = values if isinstance(values, list) else [values]
    names = [adapter.action_labels.get(str(v), str(v)) for v in values if v is not None]
    should = ""
    if len(names) == 1:
        should = names[0]
    elif len(names) == 2:
        should = f"{names[0]}或{names[1]}"
    elif names:
        should = "、".join(names[:-1]) + f"或{names[-1]}"
    return {
        "title": rule.title,
        "how_to_fix": rule.deviation_hint,
        "should_do": should,
        "outcome": rule.outcome,
        "premise": rule.premise,
        "action": rule.action,
    }


def _round_range_text(minutes: List[Optional[int]], labels: List[str],
                      unit: str = "回合") -> str:
    """对局时间区间 —— 有时间坐标用时间，没有就用回合坐标。

    自走棋按阶段编号（「4-1」），没有分钟；照着时间轴改操作的玩家
    看的其实是「第几个回合」，把回合坐标换成时间是编造。
    """
    labs = [str(x) for x in labels if x]
    # **窗口标签优先**（"6:00 ~ 10:00"）。
    #
    # 分析单元是时间窗，规则管的是「6 ~ 9 分钟」这一段；
    # 只写「第 6 分钟」会让人以为只管第 6 分钟 —— 那是把范围说成时点。
    # 口径与标题一致：**区间就是范围，时点就是准确时刻**。
    timed = [x for x in labs if ":" in x]
    if timed:
        # 窗口标签本身就是范围（「0:00 ~ 2:00」）。若其它标签都落在某一个
        # 范围里，就用那个范围 —— 否则会拼成「0:00 ~ 0:00 ~ 2:00」。
        ranges = [x for x in timed if " ~ " in x]
        if ranges:
            widest = max(ranges, key=len)
            lo, hi = widest.split(" ~ ", 1)
            if all(x == widest or (lo <= x <= hi) for x in timed):
                return widest
        first, last = timed[0], timed[-1]
        return first if first == last else f"{first} ~ {last}"
    real = [m for m in minutes if m is not None]
    if real:
        return _time_range_text(real)
    if not labs:
        return ""
    first, last = labs[0], labs[-1]
    if first.startswith("第"):
        # 标签本身已经是「第 12 分钟」这种写法，别再套一层。
        return first if first == last else f"{first} ~ {last}"
    if first == last:
        return f"第 {first} {unit}"
    return f"第 {first} ~ {last} {unit}"


def _range_start(text: str) -> int:
    """从「第 18 ~ 20 分钟」或「6:00 ~ 10:00」里取出起始分钟。

    取不到就排最后。
    """
    m = re.search(r"(\d+)\s*:\s*\d+", text or "")
    if m:
        return int(m.group(1))
    m = re.search(r"第\s*(\d+)", text or "")
    return int(m.group(1)) if m else 10 ** 6


def _corrective_ids(rules: Optional[Sequence[Rule]]) -> set:
    """**能拿来纠正玩家**的模式 id 集合。

    攻略的取舍由读不出来的前提决定（「塔的护盾没了」「对方减员」
    「谁先到龙坑」）。前提在画面里读不出来时，玩家做的是**对的** ——
    原文说「该推塔」，可当时塔还有护盾、敌方打野就在我方半区，
    他退塔下才是对的。拿这种前提去纠正人，就是在冤枉。

    所以这类模式**仍然在模式库里**（攻略内容照常可查），只是不进报告的
    「要改的地方」。

    而**模型裁决**的条目不受这条限制 —— 模型是看着画面判断的，
    它说前提成立就是成立（前提成立与否它看得见）。
    """
    out = set()
    for rule in (rules or []):
        if getattr(rule, "corrective", True):
            out.add(rule.id)
    return out


def _filter_corrective(lanes: Dict[str, Any], corrective: Optional[set]) -> Dict[str, Any]:
    """把**不能拿来纠正**的模式从分路结论里摘掉。

    报告与 JSON 要说同一件事：不可纠正的偏离不进报告，也不留在
    分路结论里 —— 留着就会出现「JSON 里有 8 处差距，报告上只有 2 条」
    这种对不上的情况。

    ``corrective`` 为 None 表示**不筛**（见 :func:`build_match_report`）。
    """
    if corrective is None:
        return lanes

    def keep(f: Dict[str, Any]) -> bool:
        mode_id = f.get("mode_id") or ""
        return not mode_id or mode_id in corrective

    out = dict(lanes)
    out["findings"] = [f for f in (lanes.get("findings") or []) if keep(f)]
    groups = []
    for group in lanes.get("lanes") or []:
        g = dict(group)
        g["findings"] = [f for f in (group.get("findings") or []) if keep(f)]
        g["problem_count"] = sum(
            1 for f in g["findings"]
            if f.get("verdict") in _PROBLEM_VERDICTS)
        g["mode_count"] = len(g["findings"])
        groups.append(g)
    out["lanes"] = groups
    cells = []
    for cell in lanes.get("cells") or []:
        c = dict(cell)
        if not keep(c.get("finding") or {}):
            continue
        cells.append(c)
    out["cells"] = cells
    out["problem_count"] = sum(g["problem_count"] for g in groups)
    return out


def build_match_report(
    rules: Sequence[Rule],
    ctx: MatchContext,
    curve: Optional[Dict[str, Any]] = None,
    version_note: str = "",
) -> Dict[str, Any]:
    """组装单局报告。"""
    adapter = get_adapter(ctx.game)
    analysis = analyze_match(rules, ctx)

    #: **能拿来纠正玩家**的模式 id。报告只呈现这些（见 :func:`_corrective_ids`）。
    #
    #: **只对视频来源的对局生效**。用户自己录的对局（demo、手工录入）
    #: 事实是玩家给的，判定按规则库的字面来 —— 用户原话是「**视频**做的
    #: 没错与数据库不一致的不用纠正」，那道闸管的是「模型看画面猜的」。
    corrective = (_corrective_ids(rules)
                 if (ctx.annotation_source or "").lower() == "derived"
                 else None)

    gaps = []
    for gap in analysis["gaps"]:
        # **不能拿来纠正的偏离不进报告**。
        #
        # 攻略的取舍由读不出来的前提决定 —— 原文说「该推塔」，可当时
        # 塔还有护盾、敌方打野就在我方半区，他退塔下才是对的。
        # 拿这种前提去纠正人就是在冤枉（用户原话：「视频做的没错与
        # 数据库不一致的不用纠正」）。
        if rules and corrective is not None and gap.rule_id not in corrective:
            continue
        # 证据在 Verdict 上，先找出来：写「为什么」要用它 ——
        # 成立的前提得带上这一局的实际取值，那取值就在证据里。
        evidence = []
        for v in analysis["verdicts"]:
            if (
                v.rule_id == gap.rule_id
                and v.round_label == gap.round_label
                and v.status == "deviated"
            ):
                evidence = v.evidence
                break
        rendered = describe_gap(adapter, gap, evidence)
        rendered["evidence"] = _describe_evidence(adapter, evidence)
        gaps.append(rendered)

    recognition = _recognition_notes(ctx)

    undetermined = []
    for v in analysis["undetermined"]:
        exp_fact = (v.expected or {}).get("fact") or ""
        missing = {e["fact"] for e in v.evidence if not e.get("present")}
        if exp_fact:
            missing.add(exp_fact)
        undetermined.append(
            {
                "rule_id": v.rule_id,
                "rule_title": v.rule_title,
                "round_label": v.round_label,
                # 要指名缺的是**哪个**事实。早先只从触发证据里挑「没采到」的，
                # 而触发条件往往都成立（时间、分路都在），挑出来是空表 ——
                # 报告于是只说一句「无法判定」，玩家不知道该补填什么。
                # 真正缺的是**判定动作那一类**（玩家没标这一帧做了什么）。
                "reason": (
                    f"缺少「{adapter.describe_fact(exp_fact)}」，无法判定"
                    if exp_fact else "关键事实未采集到，无法判定"
                ),
                "missing_facts": sorted(missing),
                # 中文名。原始码是给程序看的，页面上要给人看。
                "missing_fact_labels": [
                    adapter.describe_fact(f) for f in sorted(missing)
                ],
            }
        )

    by_category = [
        {
            "category": cat,
            "triggered": b["triggered"],
            "satisfied": b["satisfied"],
            "deviated": b["deviated"],
            "compliance_rate": b["compliance_rate"],
        }
        for cat, b in sorted(
            analysis["by_category"].items(),
            key=lambda kv: (kv[1]["compliance_rate"] is None, kv[1]["compliance_rate"] or 0),
        )
    ]

    # 判不了的原因**汇总**：哪一类动作读不出来，挡掉了多少条规则。
    #
    # 逐条的「无法判定」写在 undetermined 里，但玩家看到的是一堆条目，
    # 不知道**整体上**缺什么。汇总一句就够：「补刀、开团这两类读不出来，
    # 所以 645 条规则判不了」。
    blocked: Dict[str, int] = {}
    for v in analysis["undetermined"]:
        fact = (v.expected or {}).get("fact") or ""
        if fact:
            blocked[fact] = blocked.get(fact, 0) + 1
    unreadable = [
        {"fact": f, "label": adapter.describe_fact(f), "blocked": n}
        for f, n in sorted(blocked.items(), key=lambda kv: -kv[1])
    ]

    # 给每条分路结论补上「去哪看原画面」。
    #
    # 内核只带出回合在视频里的秒数与截图文件名 —— 它不认识 HTTP，也不该认识。
    # URL 在这里按 video_id 拼：截图和视频是两个地址，快照轻、点开才拉视频。
    video_id = recognition.get("video_id", "")
    lanes = _with_media(analysis["lanes"], video_id)
    lanes = _lane_text(
        adapter, lanes,
        [v.to_dict() for v in analysis["verdicts"]],
    )
    # 分路结论同样只留**能拿来纠正**的 —— 报告与 JSON 要说同一件事。
    lanes = _filter_corrective(lanes, corrective)

    # 分路结论与逐回合差距合成一份「操作模式分析」。
    # 两条链路是同一件事的两个视角：分路说「哪一段做错了」，
    # 逐回合说「哪一回合错在哪」。分开摆着，玩家要自己对两次，
    # 而这页只要能照着改。
    mode_analysis = build_mode_analysis(
        adapter, ctx, lanes, list(analysis["verdicts"]), rules, corrective
    )

    return {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "game": ctx.game,
            "game_display": adapter.display_name,
            "rules_version": ctx.game_version,
            "report_title": f"{adapter.display_name} · 操作模式分析",
            # 报告类型：模板据此决定渲染哪些区块。三类报告的语义不同，
            # 混在一起会出现「版本报告里有本局没有触发任何偏离」这种
            # 既无意义又误导的话。
            "report_kind": "match",
            # 动作标注的来源：player = 玩家自己勾的；derived = 从画面
            # 反推的（验证链路）。报告据此说明「实际在做什么」是谁说的 ——
            # 两者可信度不同，混着看会替玩家编话。
            "annotation_source": ctx.annotation_source or "player",
        },
        "player": ctx.player or {},
        "outcome": ctx.outcome or {},
        "stats": analysis["stats"].to_dict(),
        "round_count": analysis["round_count"],
        "gaps": gaps,
        # 分路 × 阶段的运营结论 —— 报告的主视图。
        # gaps 是逐回合的明细，适合复查；lanes 是按分路收拢后的结论，
        # 回答的是「我这条分路哪一段做错了、应该怎么做」。
        "lanes": lanes,
        # 报告正文真正渲染的那一份：分路 + 逐回合合成，每条四部分。
        "mode_analysis": mode_analysis,
        "by_category": by_category,
        "undetermined": undetermined,
        # 判不了的原因汇总：哪一类动作读不出来，挡掉了多少条规则。
        "unreadable_kinds": unreadable,
        "curve": curve or {},
        "version_note": version_note,
        # 以下三项仅视频来源的对局非空，手工录入 / API 导入时为空
        "recognition": recognition,
        "low_confidence_facts": _low_confidence_facts_of(ctx, gaps),
        # 事实中文名：报告里要展示「血量 58（45%）」而不是
        # 「health 58（45%）」。随报告下发，模板无需反查适配器。
        "fact_labels": dict(adapter.fact_labels),
    }


# --------------------------------------------------------------------------
# BP 复盘
# --------------------------------------------------------------------------

#: 局面里不展示的事实：坐标类与动作本身（页面另有位置说）。
#: 段位与预选冲突也不展示 —— 页面不再问、也不再提，
#: 拿一个页面上没有的事实去解释结论，读者只会更困惑。
_BP_HIDDEN_FACTS = (
    "bp_hero", "bp_step", "bp_action", "bp_phase",
)


def _bp_situation_text(
    adapter: GameAdapter,
    situation: Dict[str, Any],
    keep: Optional[set] = None,
) -> str:
    """把 BP 局面事实拼成一句中文。

    ``keep`` 给定时只留**这条规则用到的那几个事实** ——
    把整回合的事实都倒出来，读者要自己找哪条与结论有关；
    只留规则用到的那几个，一眼就明白「它凭什么这么判」。
    """
    parts: List[str] = []
    for key, value in (situation or {}).items():
        if key in _BP_HIDDEN_FACTS or value is None:
            continue
        if keep is not None and key not in keep:
            continue
        if isinstance(value, bool):
            parts.append(f"{adapter.describe_fact(key)}：{'是' if value else '否'}")
        elif isinstance(value, (int, float)):
            parts.append(f"{adapter.describe_fact(key)} {value}")
        else:
            parts.append(
                f"{adapter.describe_fact(key)} {adapter.describe_value(key, value)}"
            )
    return "、".join(parts)


def build_bp_report(rules: Sequence[Rule], ctx: MatchContext) -> Dict[str, Any]:
    """BP 复盘的数据组装。

    BP 在**比赛之前**，所以这里的坐标不是「第几分钟」而是
    「哪个环节」；也没有原视频片段（BP 不在局内录像里）。

    每条四部分，跟局内报告同一个骨架：

    1. **BP 环节** —— 说的是哪一个决策环节（禁用 / 一抢 / 二抢…）；
    2. **我做了什么** —— 动了哪个英雄、属于哪一类操作；
    3. **应该怎么做** —— 模式库里的原文建议；
    4. **为什么** —— 当时的局面 + 照这么打的结果。
    """
    adapter = get_adapter(ctx.game)
    analysis = analyze_match(rules, ctx)
    rounds_by_label = {r.label: r for r in ctx.rounds}
    # 规则用到的那些事实 —— 「为什么」只说这些，不把整个局面倒出来。
    used_facts = {r.id: r.trigger.referenced_facts() for r in rules}

    entries: List[Dict[str, Any]] = []
    for gap in analysis["gaps"]:
        rnd = rounds_by_label.get(gap.round_label)
        action_label = ""
        hero = ""
        step_index = 0.0
        if rnd is not None:
            for act in rnd.actions:
                if act.kind == "bp_action":
                    action_label = act.label or adapter.describe_action(act.value)
            hero = str(rnd.facts.get("bp_hero") or "")
            step_index = rnd.time_index

        move = f"{action_label}：{hero}" if hero else action_label
        situation_text = _bp_situation_text(
            adapter, gap.situation, used_facts.get(gap.rule_id)
        )
        outcome = _STATE_TAG_RE.sub("", gap.outcome or "").strip().rstrip("。")

        reason_parts: List[str] = []
        if situation_text:
            reason_parts.append(f"这一步的局面是{situation_text}。")
        if outcome:
            reason_parts.append(f"照这个打法，{outcome}。")
        if not reason_parts:
            reason_parts.append("这条模式写明了该怎么做，但没写明为什么。")

        entries.append({
            "rule_id": gap.rule_id,
            "title": gap.rule_title,
            "category": gap.category,
            "severity": gap.severity,
            # 1. BP 环节
            "step_label": gap.round_label,
            "step_index": step_index,
            # 2. 我做了什么
            "my_move": move,
            "hero": hero,
            "action_label": action_label,
            # 3. 应该怎么做
            "advice_text": gap.hint,
            "advice_lines": _split_clauses(gap.hint),
            # 4. 为什么
            "reason_text": "".join(reason_parts),
            "situation_text": situation_text,
            "premise": gap.premise,
            "outcome": outcome,
        })

    # 按 BP 的决策先后排 —— 玩家是照着 draft 的先后顺序复盘的。
    # 不能按中文环节名排：「一抢」和「三抢」的字序不等于 BP 的先后。
    entries.sort(key=lambda e: e.get("step_index") or 0)

    my_heroes: List[str] = []
    opp_heroes: List[str] = []
    for decision in (ctx.raw.get("bp") or {}).get("decisions") or []:
        hero = str(decision.get("hero") or "").strip()
        if not hero:
            continue
        if decision.get("side") == "my":
            if decision.get("phase") not in ("ban_1",):
                my_heroes.append(hero)
        else:
            if decision.get("phase") not in ("ban_1",):
                opp_heroes.append(hero)

    return {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "game": ctx.game,
            "game_display": adapter.display_name,
            "report_title": f"{adapter.display_name} · BP 复盘",
            "report_kind": "bp",
            "annotation_source": ctx.annotation_source or "player",
            # 「这是哪一局」用人能认的名字（视频文件名 · 分路 · 胜负），
            # 不摆内部 id。
            "match_name": organize.match_display_name(ctx.raw or {}, adapter),
        },
        "entries": entries,
        "summary": {
            "steps": len(ctx.rounds),
            "my_heroes": my_heroes,
            "opp_heroes": opp_heroes,
        },
        "stats": analysis["stats"].to_dict(),
        "fact_labels": dict(adapter.fact_labels),
    }


def build_growth_report(
    analyses: Sequence[Dict[str, Any]],
    game: str,
    rules_version: str,
) -> Dict[str, Any]:
    """组装跨对局的成长曲线报告。"""
    adapter = get_adapter(game)
    curve = build_curve(analyses)

    weakest = [
        {
            "rule_id": rc["rule_id"],
            "rule_title": rc["rule_title"],
            "category": rc["category"],
            "latest_rate": rc["latest_rate"],
            "trend": rc["trend"],
            "checks": rc["checks"],
        }
        for rc in curve["weakest_rules"]
    ]

    return {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "game": game,
            "game_display": adapter.display_name,
            "rules_version": rules_version,
            "report_title": f"{adapter.display_name} · 执行度成长曲线",
            "report_kind": "growth",
        },
        "player": {},
        "outcome": {},
        "stats": {},
        "gaps": [],
        "lanes": {"lane": "", "lanes": [], "cells": [], "findings": [],
                  "problem_count": 0},
        "mode_analysis": [],
        "by_category": [],
        "undetermined": [],
        "curve": curve,
        "weakest_rules": weakest,
        "recognition": {},
        "low_confidence_facts": [],
        "version_note": "",
    }


def build_version_report(
    game: str,
    old_rules: Sequence[Rule],
    new_rules: Sequence[Rule],
    contexts: Sequence,
    deprecations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """组装版本切换报告：差异 + 历史回测。

    ``contexts`` 是历史对局上下文。回测的价值全在「同一批数据两套规则各判一次」，
    所以这里传的是上下文而不是分析结果 —— 调用方负责从原始数据重建它。
    """
    adapter = get_adapter(game)
    diff = diff_versions(old_rules, new_rules)
    backtest = backtest_version_change(old_rules, new_rules, contexts)

    summary = _summarize_backtest(backtest)

    return {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "game": game,
            "game_display": adapter.display_name,
            "rules_version": diff["new_version"],
            "report_title": f"{adapter.display_name} · 版本适配报告",
            "report_kind": "version",
        },
        "player": {},
        "outcome": {},
        "stats": {},
        "gaps": [],
        "lanes": {"lane": "", "lanes": [], "cells": [], "findings": [],
                  "problem_count": 0},
        "mode_analysis": [],
        "by_category": [],
        "undetermined": [],
        "curve": {},
        "version_diff": diff,
        "backtest": backtest,
        "backtest_summary": summary,
        "deprecations": deprecations or [],
        "recognition": {},
        "low_confidence_facts": [],
        "version_note": (
            f"规则库 {diff['old_version']} → {diff['new_version']}："
            f"新增 {len(diff['added'])} 条，"
            f"失效 {len(diff['deprecated'])} 条，"
            f"变更 {len(diff['changed'])} 条"
        ),
    }


def _summarize_backtest(backtest: List[Dict[str, Any]]) -> Dict[str, Any]:
    """回测结论概览：玩家最先该看哪一条。"""
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "需要更严格": [],
        "新增约束": [],
        "可以放松": [],
        "基本一致": [],
        "数据不足": [],
    }
    for row in backtest:
        buckets.setdefault(row["verdict"], []).append(row)

    # 「最该先看」的排序：先看新增约束和更严格，再看可以放松，最后是保持不变
    priority = ["需要更严格", "新增约束", "可以放松", "基本一致", "数据不足"]
    todo = buckets.get("需要更严格", []) + buckets.get("新增约束", [])

    return {
        "total": len(backtest),
        "measured": len(backtest) - len(buckets.get("数据不足", [])),
        "stricter": buckets.get("需要更严格", []),
        "newly_binding": buckets.get("新增约束", []),
        "relaxable": buckets.get("可以放松", []),
        "unchanged": buckets.get("基本一致", []),
        "insufficient": buckets.get("数据不足", []),
        "priority_order": priority,
        "action_first": todo,
        "action_count": len(todo),
    }
