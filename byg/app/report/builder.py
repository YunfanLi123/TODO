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
from ..core.diagnose import diagnose
from ..core.dsl import Rule
from ..core.lane_analysis import (
    VERDICT_NEVER,
    VERDICT_OCCASIONAL,
    VERDICT_SYSTEMIC,
)
from ..core.matcher import STATUS_DEVIATED

from . import organize
from .narrative import NarrativeCard, split_clauses

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

#: 王者的局势阶段交界分钟（与适配器 PHASE_WINDOWS 的 0-4 / 5-10 / 11+ 对齐）。
#: 落在这几个分钟上的规则是阶段转换的决策时刻，时间区间显示时点明它的语义。
PHASE_TRANSITION_MINUTES = {4, 10}


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

    王者荣耀的一局是**连续的一整场**，切出来的只能是**时间片**。
    报告里说「这一回合」会让玩家去找回合，而王者里根本没这个东西。
    """
    return getattr(adapter, "segment_unit", None) or "时间片"


def _now(adapter) -> str:
    """指代当前这一段的说法。"""
    return getattr(adapter, "segment_now", None) or "这一段"


def _phrase_with_places(label: str, places: Sequence[str]) -> str:
    """动作名 + 一组位置，拼成一句能读的话。

    位置在**前**（在己方蓝区刷野补经济）还是**后**（转线支援（从野区转到中路）），
    看位置本身怎么开头 —— 「在…」放前面顺，「从…转到…」放后面才通顺。
    """
    uniq = list(dict.fromkeys(p for p in places if p))
    if not uniq:
        return label
    where = "、".join(uniq)
    if where.startswith("在"):
        return f"{where}{label}"
    return f"{label}（{where}）"


def _action_phrase(adapter: GameAdapter, action: Dict[str, Any]) -> str:
    """动作名 + 位置，拼成一句**能读**的话。

    为什么要带位置
    --------------
    「刷野补经济」不带位置，报告里就是一句没有指向的话 ——
    玩家看了也不知道自己是在己方蓝区刷野还是跑到敌方红区去刷。
    位置是我们自己**从轨迹算出来的**（不是模型说的），算不出就留空，
    那句话就还是原来的样子 —— 不给一个编出来的位置。
    """
    code = action.get("value") or ""
    label = action.get("label") or adapter.describe_action(code)
    where = str(action.get("where") or "").strip()
    return _phrase_with_places(label, [where] if where else [])


def describe_actual_sentence(
    adapter: GameAdapter, actual: Dict[str, Any]
) -> str:
    """实际做了什么 —— 一句主谓宾齐全的客观陈述。

    只描述动作，不评价对错：「做错了」是判定结论，不是描述。
    """
    fact = actual.get("fact") or ""
    labels = []
    for a in actual.get("actions") or []:
        lbl = _action_phrase(adapter, a)
        if lbl and lbl not in labels:
            labels.append(lbl)
    if labels:
        return f"{_now(adapter)}的操作是{'、'.join(labels)}。"

    others = []
    for a in actual.get("all_actions") or []:
        lbl = _action_phrase(adapter, a)
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


def _shown_outcome(outcome: Any) -> Dict[str, Any]:
    """只让**真的读到了**的胜负上屏。

    为什么必须筛
    ------------
    页面模板是冻结的，它这么写::

        {% if outcome.get('result') %}
          <span>结果 <b>{{ '失败' if outcome.get('result') == 'lose' else '胜利' }}</b></span>
        {% endif %}

    也就是**任何非空值都会被当成「胜利」**。2026-11-02 实测（熬隐那局）：
    读胜负那一步的模型调用返回了空，识别层如实记下 ``"unknown"``，
    报告顶上却直接写着「胜利」—— 而那一局的结算画面明明是失败。

    **不知道就说不知道**：读不到胜负就别给结果那一行。
    少一个信息项好过给一个假结论。
    """
    if not isinstance(outcome, dict):
        return {}
    out = dict(outcome)
    result = str(out.get("result") or "").strip().lower()
    out["result"] = result if result in ("win", "lose") else ""
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
        # **候选是哪来的**：模型自己检索的，还是退回顾则筛的。
        "selection": rec.get("selection") or {},
        # **英雄没认准**（模型说的和技能按钮认的对不上）——
        # 英雄层的打法这次一条都没判。
        "hero_conflict": rec.get("hero_conflict") or {},
        # **英雄是按哪条读数定的** + **哪些读数被否掉了、为什么**
        # （2026-11-03：被禁用 / 不在我方名单里的读数一律作废）。
        "hero_source": rec.get("hero_source") or "",
        "hero_voided": rec.get("hero_voided") or [],
        # **只是提醒、不作废**的读数（BP 记录是模型看头像认的时）。
        "hero_soft": rec.get("hero_soft") or [],
        # **这份 BP 记录是哪来的**（``model`` = 模型看头像认的）。
        # 不取出来，报告上那句「BP 里的名字是猜的」就是死代码（2026-11-13）。
        "bp_source": rec.get("bp_source") or "",
        # **这一局怎么「看」的**：整局视频还是抽帧。两者粗细差很多，
        # 报告要说出来（2026-11-13）。
        "watch_mode": rec.get("watch_mode") or "",
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
    labels: Dict[str, str] = {}
    places: Dict[str, List[str]] = {}
    for v in verdicts:
        actual = v.get("actual") or {}
        pool = actual.get("actions") or actual.get("all_actions") or []
        for a in pool:
            code = a.get("value") or a.get("label") or ""
            if not code:
                continue
            counts[code] = counts.get(code, 0) + 1
            labels.setdefault(code, a.get("label")
                              or adapter.describe_action(code))
            where = str(a.get("where") or "").strip()
            if where:
                places.setdefault(code, []).append(where)
    if not counts:
        return "这些" + _unit(adapter) + "没有记录到操作。"
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:2]
    names = [_phrase_with_places(labels.get(code) or adapter.describe_action(code),
                                 places.get(code) or [])
             for code, _ in top]
    return "这些" + _unit(adapter) + "的操作是" + "、".join(names) + "。"


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


def _actual_actions(adapter: GameAdapter, deviated: List[Any]):
    """这一批偏离回合里，玩家**实际做过的动作**。

    返回 ``(次数, 动作名, 出现位置)`` 三张表。

    抽出来是因为有两处要用：模板写法（:func:`_segment_actual_text`）
    与证据卡（喂给文字模型的素材）—— 两边必须看同一份动作清单，
    不然模型写出来的和报告里写的会对不上。
    """
    counts: Dict[str, int] = {}
    labels: Dict[str, str] = {}
    places: Dict[str, List[str]] = {}
    for v in deviated:
        actual = v.actual or {}
        pool = actual.get("actions") or actual.get("all_actions") or []
        for a in pool:
            code = a.get("value") or ""
            if not code:
                continue
            counts[code] = counts.get(code, 0) + 1
            labels.setdefault(code, a.get("label")
                              or adapter.describe_action(code))
            where = str(a.get("where") or "").strip()
            if where:
                places.setdefault(code, []).append(where)
    return counts, labels, places


def _segment_actual_text(adapter: GameAdapter, deviated: List[Any],
                         absent: Optional[List[str]] = None) -> str:
    """这一段实际做了什么 —— 报告里的「原视频操作的文字描述」。

    只描述动作，不评价对错 —— 对错是判定结论，写进描述里就成了替玩家
    下判断。「做不到位」那层意思交给「建议操作描述」那一部分。

    **这是模板兜底**：文字模型写出来的版本更贴这一局（见
    :mod:`app.report.narrative`），写不出来时才轮到这段话。
    """
    counts, labels, places = _actual_actions(adapter, deviated)
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
    names = [_phrase_with_places(labels.get(code) or adapter.describe_action(code),
                                 places.get(code) or [])
             for code, _ in top]
    return "这一段做的是" + "、".join(names) + "。"



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
        # 用建议而不是命令的口吻：这是给玩家参考的打法，不是硬性对错。
        return f"这种局面通常可以考虑{should}，具体仍要看当时的阵容与局势。"
    return "这一条攻略没有写出更具体的做法，可以结合实际局面灵活判断。"


def _split_clauses(text: str) -> List[str]:
    """把一段话拆成一句一句 —— 报告里一行一句，读起来才不挤。

    条件式建议尤其需要：几句「什么情况下：做什么」排在一起，
    玩家要能一眼看出哪句对应哪个条件。

    实现在 :mod:`app.report.narrative`（叙述层也要用它拆模型写的话，
    两边必须用同一套拆法，否则模板腔和模型腔的行文会不一样）。
    """
    return split_clauses(text)




def _reason_text(
    adapter: GameAdapter,
    finding: Dict[str, Any],
    deviated: List[Any],
) -> str:
    """建议理由 —— 玩家差在哪一步。

    **对照式**：「攻略要 X，你做的是 Y」。

    为什么不再用「照这个打法，{outcome}」
    ------------------------------------
    那是攻略原文写给「照着做的人」看的结果，跟玩家这一段实际做的事
    **没有任何对照关系** —— 玩家读到的是一句正确的废话，
    不知道自己错在哪一步，于是整页都像在敷衍。

    对照式的好处是它**可核对**：X 与 Y 都是这一局的事，
    玩家能一眼看出差在哪。

    两段话拼起来：

    1. **这一局的实际情况**。取自触发证据里的实际取值（经济差多少、
       大招有没有就绪…）。没有它这段话跟任何一局都成立，
       玩家没法判断跟自己有没有关系。
    2. **差在哪一步**：攻略要什么 vs 玩家实际做了什么。
    """
    parts: List[str] = []

    facts: List[str] = []
    if deviated:
        facts = premise_facts(adapter, deviated[0].evidence or [])
    if facts:
        parts.append("这一段" + "、".join(facts) + "。")

    expected = (finding.get("should_do") or "").strip()
    actual = (finding.get("actually_doing") or "").strip()
    if expected and actual:
        parts.append(f"攻略这一段要的是{expected}，你做的是{actual}。")
    elif expected:
        parts.append(f"攻略这一段要的是{expected}。")

    if not parts:
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
            # 这一窗记录到的操作。喂给文字模型的证据卡要用它 ——
            # 「我方动作」那段话得站在这一窗真实做过的动作上。
            "actions": [
                {
                    "label": str(a.label or ""),
                    "where": str(a.where or ""),
                    "value": str(a.value or ""),
                }
                for a in (getattr(rnd, "actions", None) or [])
            ],
        })
    return out


def _known_lane(ctx: MatchContext, adapter: GameAdapter) -> str:
    """玩家这一局的分路（规范码）。**没读到就是空串**（不是 ``"unknown"``）。

    为什么不能直接用 ``player["position"]``
    --------------------------------------
    那个值读不到时是 ``"unknown"`` —— 而 ``"unknown"`` 是**真值**，
    ``item["lane"] != player_lane`` 于是把**每一条**带分路的裁决都丢掉，
    报告成了空表，页面却写「这一局没有发现偏离操作模式的地方」——
    **假成功**。这里的口径是「分路没读到就别按分路过滤」。
    """
    raw = str((ctx.player or {}).get("position") or "").strip()
    if not raw or raw == "unknown":
        return ""
    try:
        value = adapter.normalize_position(raw)
    except Exception:                                   # noqa: BLE001
        value = raw
    value = str(value or "").strip()
    return "" if value in ("", "unknown") else value


def _review_entries(
    adapter: GameAdapter,
    ctx: MatchContext,
    video_id: str,
    video_url: str,
    stats: Optional[Dict[str, Any]] = None,
    narrator: Any = None,
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

    **接地闸**
    --------
    一条结论要进报告正文，必须同时满足三样：

    1. **有画面依据** —— 至少一帧真的看到过（``moments`` 非空）；
    2. **指认了攻略里的具体一条** —— 不是「这几种打法挑一个」，
       而是「这一局适用的是第几支、该做什么」；
    3. **理由是对照式的** —— 「攻略要 X，你做的是 Y」。

    早先缺第 2 条时，报告里出现的就是七条一模一样的
    「这一段暂时没看到推荐打法」+ 一整张条件菜单：读起来像在敷衍，
    而且与视频无关 —— 那正是用户说的「胡编」。
    宁可少几条，也不要给没有出处的结论。
    """
    entries: List[Dict[str, Any]] = []
    seen: set = set()
    #: 丢弃原因（写进诊断，说明白为什么报告里少了东西）。
    dropped = {"no_ground": 0, "no_advice": 0}
    if stats is not None:
        stats.setdefault("dropped", dropped)
    else:
        stats = {"dropped": dropped}

    #: 玩家打哪条线。分路已知时，**别的线**的模式一条都不进报告 ——
    #: 分路没读到时清单是全线的，模型可能对着别的线的打法说话。
    player_lane = _known_lane(ctx, adapter)

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
                # 没有画面依据 → 不进正文。
                #
                # 只对**视频来源**的对局生效：玩家手工录入的对局
                # 事实是他自己给的，报告并不描述画面，不需要这一关
                # （用户原话：「视频做的没错与数据库不一致的不用纠正」，
                # 同一个道理）。
                if (ctx.annotation_source or "").lower() == "derived":
                    dropped["no_ground"] += 1
                    continue

            did = (item.get("did") or "").strip()
            actual_text = f"这一段做的是{did}。" if did else "这一段的操作见下面的片段。"

            #: 这一局适用的是第几支、这一支的前提 —— 两条分支各填各的。
            branch_no = 0
            guide_premise = ""

            if kind == "node":
                # 整段没做这个时段该做的任何一件事 —— 建议必须是
                # **这一段该做的那一条**，不是一整张菜单。
                #
                # 模型被要求指认清单里的具体一条（``should``），
                # 指不出来就不给这条结论 —— 「你什么都没做对」而说不出
                # 「该做什么」，对玩家没有任何用。
                advice = (item.get("should_action") or "").strip()
                if not advice:
                    dropped["no_advice"] += 1
                    continue
                if advice and not advice.endswith(("。", "！", "？")):
                    advice += "。"
                reason = (item.get("why") or "").strip()
                if reason and not reason.endswith(("。", "！", "？")):
                    reason += "。"
                should_premise = (item.get("should_premise") or "").strip()
                guide_premise = should_premise
                if should_premise:
                    head = f"这一段该做的是：{should_premise}。"
                    reason = (head + reason).strip() if reason else head
                title = "这一段没按攻略做"
                category = item.get("lane_label") or ""
                verdict_label = "这一段没按攻略做"
            else:
                # 建议：**这一局适用的那一支**，不是整张菜单。
                #
                # 玩家拿到「敌方红开就己方蓝开，敌方蓝开就己方红开」这样
                # 一张菜单，配上「敌方蓝开，我方红开」这个理由，读完只知道
                # 自己被判了偏离，却不知道自己错在哪一支 —— 于是报告不敢信。
                #
                # **模型没指认是哪一支，就不给这条结论**（`branch_no == 0`）。
                # 这种情况说明模型自己也没把握适用哪支 —— 没把握的结论
                # 进报告，就是拿一张菜单敷衍人。
                try:
                    branch_no = int(item.get("branch_no") or 0)
                except (TypeError, ValueError):
                    branch_no = 0
                advice = (item.get("branch_action") or "").strip()
                if not advice and branch_no:
                    advice = (item.get("action") or "").strip()
                if not advice:
                    dropped["no_advice"] += 1
                    continue
                if advice and not advice.endswith(("。", "！", "？")):
                    advice += "。"
                reason = (item.get("why") or "").strip()
                # 「适用的是第几支」写进理由 —— 菜单与结论要对得上号。
                branch_premise = (item.get("branch_premise") or "").strip()
                guide_premise = branch_premise
                if branch_no and branch_premise:
                    head = f"这一局适用的是第 {branch_no} 支：{branch_premise}。"
                    reason = (head + reason).strip() if reason else head
                # **不再追加「照这个打法，{outcome}」** ——
                # 那是攻略原文写给「照着做的人」看的结果，与玩家这一段
                # 做的事没有对照关系。用户明确说过不要这种写法：
                # 理由要**对照着说**（「攻略要 X，你做的是 Y」），
                # 模型写的 why 已经是这个形状。
                title = item.get("title") or ""
                category = item.get("lane_label") or ""
                verdict_label = "这里可以调整"

            entry = {
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
            }

            # **让文字模型把这两段话写出来**（写不出来就留着模板那套）。
            # 证据卡里只装已经采到的东西：模型写的话要靠它接地。
            if narrator is not None:
                card = NarrativeCard(
                    entry_id=mode_id or str(item.get("should_id") or ""),
                    hero=str((ctx.player or {}).get("hero") or ""),
                    lane_label=item.get("lane_label") or "",
                    time_range=time_range,
                    did=did,
                    actual_actions=[
                        (a.get("label")
                         or adapter.describe_action(a.get("value") or ""))
                        for a in (rnd.get("actions") or [])
                        if a.get("label") or a.get("value")
                    ],
                    actual_places=[
                        str(a.get("where")) for a in (rnd.get("actions") or [])
                        if a.get("where")
                    ],
                    evidence=[m.get("why") for m in evidence if m.get("why")],
                    # **不给整张菜单** —— ``item["action"]`` 是这个节点的
                    # 全部做法（可能含好几支），而 ``advice`` 才是
                    # 「这一局适用的那一支」。把菜单一起喂进去，
                    # 模型会把好几支揉成一段，玩家读不出哪条适用。
                    expected="",
                    guide_premise=guide_premise,
                    guide_action=advice,
                    guide_outcome=(item.get("branch_outcome")
                                   or item.get("outcome") or ""),
                    actual_doing=did,
                    reason=item.get("why") or "",
                    branch_no=branch_no,
                )
                narrator.apply(entry, card)

            entries.append(entry)
    return entries


def _death_entries(ctx: MatchContext, video_url: str = "") -> List[Dict[str, Any]]:
    """失败局的**死亡倒推复盘** —— 每次死亡一条。

    **绕过规则库**：这一层自己走 —— 规则库那套判不了死因
    （种子模式里写着「『有没有被切』读不到」），也不该由它来挑
    这一局该看哪一条打法。

    死因取**模型看到的画面**（且要核过画面依据），
    「下次怎么做」直接检索攻略语料。

    **异常不吞** —— 调用方要写进说明位（静默失败必须堵住）。
    """
    outcome = ctx.outcome or {}
    if str(outcome.get("result") or "").strip().lower() != "lose":
        return []
    rec = (ctx.raw or {}).get("recognition") or {}
    deaths = rec.get("deaths") or []
    if not deaths:
        return []
    from ..recognition2.death_review import build_entries, collect

    moments = collect(deaths, ctx.rounds)
    return build_entries(moments, video_url=video_url)


def _hp_entries(ctx: MatchContext, video_url: str = "") -> List[Dict[str, Any]]:
    """**残血时段** —— 血量掉到阈值以下的那些片段，每段一条。

    血量是**量出来的**（像素法读血条，模型拉近复核过），不是模型
    叙述的 —— 所以它和死亡复盘一样：**绕过规则库判定**，
    自己走这一层。

    条目只留三样：时间区间 / 该做什么（模式库原文）/ 为什么有用。

    **异常不吞** —— 调用方要写进说明位（静默失败必须堵住）。
    """
    rec = (ctx.raw or {}).get("recognition") or {}
    hp = rec.get("hp") or {}
    episodes = hp.get("episodes") or []
    if not episodes:
        return []
    from ..recognition2.hp_bar import build_entries

    low = hp.get("low_pct")
    low_pct = float(low) if low is not None else 35.0
    return build_entries(episodes, low_pct=low_pct, video_url=video_url)


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
    #: **诊断计数**（可选传出）：丢了多少条、为什么丢。
    #:
    #: 空报告必须能解释自己 —— 用户看到一张空表，第一反应是
    #: 「系统根本没看」而不是「我没被判定」。把丢弃原因带出去，
    #: 空报告时就能说一句实话。
    notes: Optional[Dict[str, Any]] = None,
    #: **叙述器**（可选）：把「我方动作」「建议动作」两段话交给文字模型写。
    #:
    #: 传 None = 全用模板写法（测试与离线重建都走这条）。
    #: 传了就用模型写的，但模型写不出来 / 写得不接地时**逐段退回模板** ——
    #: 叙述是锦上添花，报告不能被它拖住。
    narrator: Any = None,
) -> List[Dict[str, Any]]:

    """把分路结论与逐回合差距合成一份「操作模式分析」。

    每一条三部分：

    1. **对局时间区间** —— 这一条覆盖本局哪一段时间；
    2. **原视频操作的文字描述** —— 这一段实际做了什么；
    3. **根据模式库建议操作的描述 + 建议理由** —— 该做什么、为什么。

    只收**有偏离**的条目。做对了的条目没有改法，
    三部分凑不齐 —— 留着只会让报告变成一张记分表，
    而这页要回答的是「我该怎么改」。

    两个来源都要收，缺一不可：

    - **分路结论**（``lanes['findings']``）—— 按分路 × 阶段收拢的运营结论；
    - **逐时间片差距** —— 那些**没有分路结论**的规则。

    第二份来源不是补丁。分路结论只覆盖「能归到某一条分路」的模式
    （``position`` 条件恰好一个值），按阶段编号定位的规则、
    以及「全位置通用」的模式都归不进去 —— 它们的偏离如果没人收，
    报告就会对着一局有偏离的对局说「没有需要改的地方」。
    """
    by_rule: Dict[str, List[Any]] = {}
    for v in verdicts:
        by_rule.setdefault(v.rule_id, []).append(v)

    rule_by_id: Dict[str, Rule] = {r.id: r for r in (rules or [])}

    #: 当前对照的版本 —— 模式库里的最新版本。
    current_version = _current_version(rules)

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
    review_entries = _review_entries(adapter, ctx, video_id, video_url,
                                     stats=notes, narrator=narrator)
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
        if not moments:
            # **没有画面依据 → 不进正文**（只对视频来源的对局生效）。
            #
            # 规则引擎判出来的偏离，文字描述说的是「画面里看到了什么」。
            # 看不到画面却要写一段描述，那段描述只能靠攻略原文凑 ——
            # 于是报告里出现「照这个打法，X」这种与画面无关的话，
            # 也就是用户说的「胡编」。
            #
            # 玩家手工录入的对局（事实是他自己给的）不适用这一关：
            # 那份报告本来就不描述画面。
            #
            # 宁可少一条，也不要给一条说不出出处的结论。
            if (ctx.annotation_source or "").lower() == "derived":
                if notes is not None:
                    dropped = notes.setdefault("dropped", {})
                    dropped["no_ground"] = int(dropped.get("no_ground") or 0) + 1
                continue

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

        # 时间区间先算出来：条目与证据卡都要用（键要对得上）。
        time_range = _round_range_text(
            [minutes_of.get(v.round_label) for v in deviated],
            [v.round_label for v in deviated if v.round_label],
            _unit(adapter),
        )

        entry = {
            "mode_id": mode_id,
            "category": finding.get("category") or "",
            "title": finding.get("title") or "",
            "lane_label": finding.get("lane_label") or "",
            "phase_label": finding.get("phase_label") or "",
            "verdict_label": finding.get("verdict_label") or "",
            # 1. 对局时间区间
            # 窗口标签优先（「12:00 之后」「6:00 ~ 10:00」）——
            # 规则管的是一整段，只写「第 12 分钟」会把范围说成时点。
            "time_range": time_range,
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
            # 这条模式可能已经过时 —— 提醒一句，别让玩家照着旧攻略去改
            "freshness_note": _freshness_note(
                rule_by_id.get(mode_id), current_version),
            # 这条结论是不是建立在低置信度的识别之上
            "low_confidence": bool(weak),
            "confidence": min(weak) if weak else 1.0,
        }

        # **让文字模型把这两段话写出来**（写不出来就留着模板那套）。
        #
        # 这一条路径原先只有模板腔：「这一段做的是转线支援、回撤拉扯。」
        # 加上逐字照搬的 `deviation_hint` —— 同一段话贴在十局不同的对局上
        # 都不违和，因为它根本不在描述这一局。
        if narrator is not None:
            _, action_labels, action_places = _actual_actions(adapter, deviated)
            # **这一段的局面**（经济差多少、大招有没有就绪…）——
            # 动作名只说了「做了什么」，说了不说明当时怎么样；
            # 有了局面，叙述才写得出具体的话，而不是把动作名念一遍。
            facts = premise_facts(adapter, deviated[0].evidence or []) \
                if deviated else []
            card = NarrativeCard(
                entry_id=mode_id,
                hero=str((ctx.player or {}).get("hero") or ""),
                lane_label=finding.get("lane_label") or "",
                phase_label=finding.get("phase_label") or "",
                time_range=time_range,
                facts=facts,
                actual_actions=[
                    action_labels.get(code) or adapter.describe_action(code)
                    for code in list(action_labels)[:3]
                ],
                actual_places=[
                    where for code in list(action_labels)[:3]
                    for where in (action_places.get(code) or [])
                ],
                evidence=[m.get("why") for m in evidence if m.get("why")],
                absent=[adapter.describe_action(str(k).split(":")[-1])
                        for k in (finding.get("absent_kinds") or [])],
                expected=finding.get("should_do") or "",
                guide_premise=finding.get("premise") or "",
                guide_action=(finding.get("how_to_fix")
                              or finding.get("action") or ""),
                guide_outcome=finding.get("outcome") or "",
                actual_doing=finding.get("actually_doing") or "",
                reason=finding.get("why") or "",
            )
            narrator.apply(entry, card)

        entries.append(entry)

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
            "verdict_label": "这里可以调整",
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
            # 这条模式可能已经过时 —— 提醒一句，别让玩家照着旧攻略去改
            "freshness_note": _freshness_note(rule, current_version),
            "low_confidence": bool(weak),
            "confidence": min(weak) if weak else 1.0,
        })

    # 时间区间从早到晚排 —— 玩家是照着时间轴改自己的操作的。
    entries.sort(key=lambda e: _range_start(e.get("time_range") or ""))

    # **模型写的描述不上前端**（用户拍板）。
    #
    # 判定层内部照样用这些画面依据做**接地闸**（没有依据就不给建议）——
    # 只是**页面不再展示**：模型对这一局的白描读起来像人机，
    # 而这页要回答的是「我该怎么改」。
    #
    # 条目只留：**时间区间 / 该做什么（模式库原文）/ 为什么有用（理由）**。
    for entry in entries:
        entry["actual_text"] = ""
        entry["evidence"] = []
    return entries


def _current_version(rules: Optional[Sequence[Any]]) -> str:
    """模式库的当前版本（库里最新的那个）。

    时效判定要拿它跟规则的 ``version`` 比 ——
    规则写于 S45、库里已经有 S46，那这条规则就**可能**过时了。
    """
    versions = [str(getattr(r, "version", "") or "")
                for r in (rules or [])]
    versions = [v for v in versions if v]
    return max(versions) if versions else ""


def _freshness_note(rule: Any, current_version: str) -> str:
    """这条模式可能已经过时 —— 提醒一句。

    借鉴「峡谷智囊」的时效治理（见 :mod:`app.rules.freshness`）：
    **没有证据不等于没问题** —— 有更新的版本却没记录到改动时，
    也要在报告里说一声「无法确认是否仍然适用」，而不是默默按旧打法给建议。
    """
    if rule is None:
        return ""
    try:
        from ..rules import freshness as fr

        verdict = fr.judge(rule, current_version=current_version)
        return fr.note_for(verdict)
    except Exception:                                   # noqa: BLE001
        # 时效判定**绝不能**让报告出不来 —— 判不出来就不提醒。
        return ""


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
                      unit: str = "时间片") -> str:
    """对局时间区间 —— 有时间坐标用时间，没有就用时间片坐标。

    王者荣耀的一局是连续的一整场，切出来的是时间片；把它说成「第几回合」
    会让玩家去找回合，而王者里根本没这个东西。
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
        if first == last:
            # 单点时点（如 4:00 / 10:00）本质是阶段转换的决策时刻，规则其实
            # 管这一分钟附近。只印一个「4:00」显得又单薄又绝对，补成「前后」，
            # 并在 4:00 / 10:00 这两个阶段交界点点明语义，不夸大成一段时间。
            m = re.match(r"^(\d+):(\d+)$", first)
            if m and m.group(2) == "00":
                minute = int(m.group(1))
                if minute in PHASE_TRANSITION_MINUTES:
                    return f"{first} 前后（阶段转换点）"
                return f"{first} 前后"
            return first
        return f"{first} ~ {last}"
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


def build_coverage(ctx: MatchContext, verdicts: Sequence[Any],
                   adapter: GameAdapter) -> Dict[str, Any]:
    """分析覆盖小结：按时间窗交代「这一段看到了什么、判了几条」。

    报告主体只列偏离，玩家无从分辨某条规则是「做对了 / 没做到 /
    这一刻不适用 / 这段根本没采到画面」。这里从两个来源拼出每段的账：

    - 回合（时间窗）：这段采没采到、看到几个动作 —— 来自 ``ctx.rounds``；
    - 判定结果：这段触发了几条、满足 / 偏离 / 无法判定各几条 —— 来自匹配器。

    「不适用」（subject 不匹配或做了节点认可的别的事）匹配器直接返回 None，
    不在判定里出现 —— 它体现在「触发的规则数 < 这段相关规则数」上，
    这里不强行估算，避免再编一个数。只如实摆能确证的四件事。
    """
    # 窗口标签 -> 回合（同一标签可能有多个观测点，动作数取合计）。
    window_order: List[str] = []
    seen: set = set()
    acts_by_window: Dict[str, List[str]] = {}
    start_minute: Dict[str, int] = {}
    for rnd in ctx.iter_rounds():
        label = str(rnd.label or "").strip()
        if not label:
            continue
        if label not in seen:
            seen.add(label)
            window_order.append(label)
            try:
                start_minute[label] = int(rnd.time_index or 0)
            except (TypeError, ValueError):
                start_minute[label] = _range_start(label)
        bucket = acts_by_window.setdefault(label, [])
        # **模型原话优先**。
        #
        # 老链路这一列填的是**动作枚举**（``actions``）—— 那是识别器
        # 认出来的码，认不出就显示「这段没识别到明确操作」。那句话最容易被
        # 读成「这段没分析」，而实际只是「认不出这个码」。
        #
        # v2 给的是**模型自己写的句子**（``narrative``）：看到了就有话说，
        # 看不清就写「看不清」—— 两者说的是同一件事的两面，不会互相冒充。
        narrative = str(getattr(rnd, "narrative", "") or "").strip()
        if narrative:
            for part in narrative.split("；"):
                part = part.strip()
                if part and part not in bucket:
                    bucket.append(part)
            continue
        for a in getattr(rnd, "actions", []) or []:
            name = adapter.describe_action(a.value) if a.value else ""
            if name and name not in bucket:
                bucket.append(name)

    # 窗口标签 -> 判定计数。
    cnt: Dict[str, Dict[str, int]] = {}
    for v in verdicts:
        label = str(getattr(v, "round_label", "") or "").strip()
        if not label:
            continue
        cell = cnt.setdefault(label, {"checked": 0, "satisfied": 0,
                                      "deviated": 0, "undetermined": 0})
        cell["checked"] += 1
        st = getattr(v, "status", "")
        if st in ("satisfied", "deviated", "undetermined"):
            cell[st] += 1

    windows: List[Dict[str, Any]] = []
    for label in window_order:
        c = cnt.get(label, {"checked": 0, "satisfied": 0,
                            "deviated": 0, "undetermined": 0})
        acts = acts_by_window.get(label, [])
        # **这一列不能是一整墙** —— 一格塞十句（每句二十来字），
        # 页面上一行三百多字，读者根本扫不下去（2026-11-12 实测）。
        # 只留前几句当摘要；完整清单在 ``actions`` 里，报告 JSON 里还在。
        summary = "、".join(acts[:3])
        if len(acts) > 3:
            summary += "…（这一段共 %d 条操作）" % len(acts)
        windows.append({
            "label": label,
            "actions": acts,
            "action_text": summary or "这段没识别到明确操作",
            "checked": c["checked"],
            "satisfied": c["satisfied"],
            "deviated": c["deviated"],
            "undetermined": c["undetermined"],
            # 采到了画面但没有任何规则可判 —— 通常是动作落在 subject 适用集外，
            # 或这段没有对应模式。和「没采到画面」要区分开。
            "sampled": bool(acts) or label in cnt,
        })

    total_checked = sum(w["checked"] for w in windows)
    total_dev = sum(w["deviated"] for w in windows)
    total_sat = sum(w["satisfied"] for w in windows)
    return {
        "windows": windows,
        "window_count": len(windows),
        "checked": total_checked,
        "satisfied": total_sat,
        "deviated": total_dev,
    }



def _load_patterns(ctx: MatchContext) -> Optional[List[Any]]:
    """读 v2 操作模式库。

    读不到就用 v1 规则引擎 —— 内核换了但结论的**形状**不变，
    所以退回是安全的，不会让页面崩、不会让数字对不上。

    模式库与真源（v1 规则库）对不上时也退回：宁可用旧引擎，
    也不要拿一份和规则库不一致的模式去判玩家。
    """
    try:
        from ..rules.patterns import load_patterns
        return load_patterns(ctx.game, ctx.game_version)
    except Exception:                                       # noqa: BLE001
        return None


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


def _stats_of(analysis: Dict[str, Any], harness: Any) -> Dict[str, Any]:
    """这一局的执行度统计 —— **按判定来源取**。

    视频来源（``derived``）且模型给了裁决 → 用模型的（:mod:`app.harness`）；
    否则用规则引擎的。

    两者的字段形状一致，所以页面不用改。
    数字旁边的 ``source`` 字段要说清是谁算的 ——
    读者得知道这个执行度是**模型看画面判的**，还是规则套出来的。
    """
    if harness is not None and getattr(harness, "verdicts", None):
        try:
            return harness.stats.to_dict()
        except Exception:                               # noqa: BLE001
            pass
    return analysis["stats"].to_dict()


#: 交叉验证里「两边结论相反」的两种组合。
#:
#: 只有这两种算**分歧**：
#:
#: * 模型说「做到了」，按规则条件直接比对说「没做到」；
#: * 模型说「没做到」，直接比对说「做到了」。
#:
#: 别的组合**不算**分歧 ——
#: 「代码判不了、模型判得了」是正常的（模型能读画面语义，
#: 代码只能比对动作标签），把它算成分歧会制造一堆假警报。
_CROSS_CHECK_OPPOSITE = {
    ("satisfied", "deviated"),
    ("deviated", "satisfied"),
}


def _cross_check(harness: Any) -> Dict[str, Any]:
    """**模型裁决**与**纯代码比对**的对账结果。

    判定权在模型（用户拍板的口径），代码这边只做交叉验证。
    两边结论相反时，报告要把它写出来 —— 藏起来等于让玩家自己猜哪边对。

    返回 ``{"checked": n, "diverged": n, "items": [...]}``。
    """
    out: Dict[str, Any] = {"checked": 0, "diverged": 0, "items": []}
    if harness is None:
        return out
    for v in (getattr(harness, "verdicts", None) or []):
        code = str(getattr(v, "code_status", "") or "")
        if not code:
            continue
        out["checked"] += 1
        model = str(getattr(v, "status", "") or "")
        if (model, code) not in _CROSS_CHECK_OPPOSITE:
            continue
        out["diverged"] += 1
        out["items"].append({
            "rule_id": str(getattr(v, "rule_id", "") or ""),
            "rule_title": str(getattr(v, "rule_title", "") or ""),
            "round_label": str(getattr(v, "round_label", "") or ""),
            "model_status": model,
            "code_status": code,
            "why": str(getattr(v, "why", "") or ""),
            "did": str(getattr(v, "did", "") or ""),
        })
    return out


def _cross_check_note(harness: Any) -> str:
    """交叉验证分歧的说明文字（写进 ``version_note``）。

    为什么用文字而不是只给个数字：数字只说「有分歧」，没说**分歧在哪**。
    读者要能自己核对，才知道该信哪边 —— 所以要给出规则名和两边各说了什么。
    """
    cc = _cross_check(harness)
    n = int(cc.get("diverged") or 0)
    if not n:
        return ""
    model_sat = sum(1 for it in cc["items"]
                    if it["model_status"] == "satisfied")
    model_dev = n - model_sat
    text = ("另有 %d 条规则，模型的裁决与按规则条件的直接比对结论相反" % n)
    if model_sat:
        text += "（%d 条模型判为做到、比对判为没做到" % model_sat
    if model_dev:
        text += ("，%d 条模型判为没做到、比对判为做到"
                 % model_dev if model_sat else
                 "（%d 条模型判为没做到、比对判为做到" % model_dev)
    text += "）"
    text += "；报告一律以模型的裁决为准"
    names = [str(it.get("rule_title") or "") for it in cc["items"]
             if str(it.get("rule_title") or "")]
    if names:
        text += "，涉及：" + "、".join(names[:3])
        if len(names) > 3:
            text += " 等 %d 条" % len(names)
    text += "。"
    return text


#: **落库的裁决**只留这几项 —— 成长报告用得上的就是这些。
#:
#: 为什么不全存：裁决里带着模型写的整段话、证据帧清单、检验区间，
#: 一局几百条，全存进快照会让库迅速变大，而这些字段成长报告一条都不用。
#:
#: 留的：曲线的四个维度（id / 标题 / 分类 / 状态）+
#: 模型原话（玩家做了什么 / 为什么 / 建议）—— 最后三项是
#: 「最需要改进的规则」那条说明位要引用的。
_VERDICT_FIELDS = (
    "rule_id", "rule_title", "category", "severity", "status",
    "round_label", "did", "why", "advice",
)

#: 裁决的两个来源。
#:
#: ``model`` = 视频来源的对局，**模型看画面逐条裁决**；
#: ``rule``  = 玩家录入的对局，规则引擎按事实比对。
#:
#: 为什么必须记下来：成长报告跨对局，两种来源可能混在一起 ——
#: 说明位要说清「这些点是谁算的」，不能一律说「模型裁决」，
#: 那是在骗读者。
_VERDICT_SOURCE_MODEL = "model"
_VERDICT_SOURCE_RULE = "rule"


def _verdicts_for_curve(harness: Any,
                        analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """这一局的**逐条裁决** —— 成长报告的逐规则维度靠它。

    视频来源优先用**模型裁决**（带模型原话：玩家做了什么 / 为什么 /
    建议）；玩家录入的对局没有模型裁决，退回规则引擎的裁决。

    为什么以前拿不到：成长报告的曲线是从**落库的快照**重建的，
    而快照里从来没存过逐条裁决 —— 曲线只有点数，没有单规则维度，
    「最需要改进的规则」永远是空的。
    """
    if harness is not None:
        out = list(getattr(harness, "verdicts", None) or [])
        if out:
            return [_slim_verdict(v, _VERDICT_SOURCE_MODEL) for v in out]
    return [
        _slim_verdict(v, _VERDICT_SOURCE_RULE)
        for v in (analysis.get("verdicts") or [])
        if hasattr(v, "to_dict")
    ]


def _slim_verdict(v: Any, source: str) -> Dict[str, Any]:
    """把一条裁决压成**落库要留的那几项**，并记下它是谁算的。"""
    data = v.to_dict() if hasattr(v, "to_dict") else dict(v)
    out = {k: data.get(k, "") for k in _VERDICT_FIELDS}
    out["source"] = source
    return out


def _field(v: Any, key: str, default: Any = "") -> Any:
    """读裁决字段，兼容 dict 与对象两种形态。"""
    if isinstance(v, dict):
        return v.get(key, default)
    return getattr(v, key, default)


def _growth_note(analyses: Sequence[Dict[str, Any]],
                 weakest: Sequence[Dict[str, Any]]) -> str:
    """成长报告的说明位 —— 讲清**口径**，并把最该改的规则用模型原话写出来。

    为什么写在这里：页面模板是冻结的，而「最需要改进的规则」那一栏
    只渲染标题与数字。要让玩家看到**模型自己怎么说**，就只能放进
    已经在渲染的说明位（``version_note``）—— 模板一个字都不用动。

    口径要写清楚：**谁算的点**。
    视频来源的对局是模型逐条裁决，玩家录入的对局是规则引擎比对 ——
    两种混在一批对局里时，说「都来自模型裁决」是在骗读者。
    """
    sources = set()
    for a in analyses:
        for v in (a.get("verdicts") or []):
            src = str(_field(v, "source", "") or "")
            if src:
                sources.add(src)
    if sources == {_VERDICT_SOURCE_MODEL}:
        parts = ["曲线与下面的优先级都来自模型逐条裁决"
                 "（判定归模型，规则条件比对只作交叉验证）。"]
    elif sources == {_VERDICT_SOURCE_RULE}:
        parts = ["曲线与下面的优先级来自规则引擎的逐条比对"
                 "（这批对局是手工录入的，没有画面可看）。"]
    elif sources:
        parts = ["曲线与下面的优先级按各局自己的口径统计"
                 "（模型裁决 / 规则引擎比对）。"]
    else:
        parts = ["曲线与下面的优先级按逐条裁决统计"
                 "（判定归模型，规则条件比对只作交叉验证）。"]

    lines: List[str] = []
    for rc in list(weakest)[:3]:
        rid = str(rc.get("rule_id") or "")
        title = str(rc.get("rule_title") or rid or "（规则标题缺失）")
        why = ""
        advice = ""
        for a in reversed(list(analyses)):
            for v in (a.get("verdicts") or []):
                if str(_field(v, "rule_id", "") or "") != rid:
                    continue
                why = str(_field(v, "why", "") or "")
                advice = str(_field(v, "advice", "") or "")
                if why or advice:
                    break
            if why or advice:
                break
        if not (why or advice):
            continue
        piece = title
        if why:
            piece += "：模型说「%s」" % why[:120]
        if advice:
            piece += "，建议「%s」" % advice[:120]
        lines.append(piece)
    if lines:
        parts.append("最该先改的：" + "；".join(lines) + "。")
    return "".join(parts)


#: 英雄这个读数**是谁说的** —— 页面据此告诉读者「这个英雄请核对」。
#:
#: 为什么不能一律写「模型读的」：``bp`` 是从 BP 记录推的（最弱的一路），
#: ``config`` 是玩家自己填的（不用核对）。混成一句话就等于骗读者。
_HERO_SOURCE_LABELS = {
    "clip": "从选人界面读的，请核对",
    "bp": "从 BP 记录推的，请核对",
    "watch": "看完整局认的，请核对",
    "icon": "从技能按钮认的，请核对",
    "局内技能按钮": "从技能按钮认的，请核对",
    "config": "你填的",
}


def _position_label(code: str) -> str:
    """分路码 -> 人话（拿不到映射就原样返回，别让读者看码）。"""
    try:
        from ..recognition2.lane_evidence import POSITION_LABELS

        return POSITION_LABELS.get(code, code)
    except Exception:                                   # noqa: BLE001
        return code


def _empty_reason(mode_analysis: Sequence[Any], stats: Dict[str, Any]) -> str:
    """正文空的时候说清**是哪一种空**。

    「一条都没偏离」与「压根没得比」在页面上**长得一样**（都是空的），
    意思完全相反：前者是「做对了」，后者是「没分析」。
    2026-11-12 实测：最新那一局候选 0 条，正文却写
    「这一局没有发现偏离操作模式的地方」—— 读者会以为这一局没问题。

    判据是**裁决数**：一条都没裁决过（``decided == 0``）= 没得比；
    裁决过但都没偏离 = 真的没问题，保留模板原话。
    """
    if mode_analysis:
        return ""
    if int(stats.get("decided") or 0) <= 0:
        return ("这一局没有可对照的攻略条目，所以没有结论 —— "
                "不是「做对了」，是压根没有能拿来比的东西。")
    return ""


def _player_view(ctx: MatchContext) -> Dict[str, Any]:
    """报告里的「玩家」这一块 —— 英雄、分路，以及**这个读数是谁说的**。

    模板读的是顶层 ``player``（``views.render_report`` 负责传进来），
    之前一直没传 —— 英雄那一行在页面上**从来没有出现过**。

    **手工录入的对局**没有 ``hero_source``（那是识别链才有的）——
    那时英雄是**玩家自己填的**，标一句「你填的」比不显示强。
    """
    out = dict(ctx.player or {})
    src = str(out.get("hero_source") or "")
    label = _HERO_SOURCE_LABELS.get(src, "")
    if not label and str(out.get("hero") or "").strip():
        label = "你填的"
    if label:
        out["hero_source_label"] = label
    return out


def _hero_lane_mismatch_note(player: Dict[str, Any]) -> str:
    """英雄的官方分路与这一局判出来的分路**对不上**时，说清楚。

    为什么必须在报告上说（2026-11-13 实测，``hok-video-8178dc031675``）
    ------------------------------------------------------------------
    报告顶着「英雄 公孙离（从 BP 记录推的）· 分路 中路」，正文整页空着，
    说明位只说「一条可对照的攻略都没进候选」，外加一句「规则库里有
    「公孙离」的攻略 12 条（发育路 12 条）」。
    读者要自己把这两句拼起来才看得出问题 —— 而拼起来就是：
    **这个英雄根本不在中路打**，所以英雄层的打法一条也套不上。

    真值是敖隐（发育路英雄打中路），也就是说**英雄读数本身认错了**。
    这一句就是把「对不上」这个事实摆到读者面前：
    要么他核对后改英雄，要么他知道为什么英雄层一条都没判。
    """
    hero = str(player.get("hero") or "").strip()
    position = str(player.get("position") or "").strip()
    if not hero or not position or position == "unknown":
        return ""
    try:
        from ..adapters.honor_of_kings import official_lanes_of

        lanes = [x for x in official_lanes_of(hero) if x]
    except Exception:                                   # noqa: BLE001
        return ""
    if not lanes or position in lanes:
        return ""
    lanes_txt = "、".join(_position_label(x) for x in lanes)
    return ("英雄「%s」的官方分路是「%s」，而这一局判出来的是「%s」—— "
            "这两样对不上：英雄层的打法都是按「%s」写的，这一局一条也没对照上。"
            "（英雄读数可能是错的，请核对；改对之后英雄层的打法才会参与判定。）"
            % (hero, lanes_txt, _position_label(position), lanes_txt))


def _dedupe_advice(entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """同一条建议在一页里只写一遍。

    为什么（2026-11-13 实测）：残血时段一段一条，而检索到的**是同一条攻略**——
    页面上连着三格写着「团战打输、己方还有人没复活时，就后撤清线……」，
    读者会以为是三条不同的建议。这里只把重复的收起来：
    第一条保留原文，后面的写「处理方式同上一条」，理由照旧各自保留。
    """
    out: List[Dict[str, Any]] = []
    seen: Dict[str, int] = {}
    for item in entries or []:
        if not isinstance(item, dict):
            continue
        entry = dict(item)
        advice = str(entry.get("advice_text") or "").strip()
        if advice:
            if advice in seen:
                entry["advice_text"] = "处理方式同上一条。"
                entry["advice_lines"] = []
            else:
                seen[advice] = len(out)
        out.append(entry)
    return out


def _wrong_points(ctx: MatchContext) -> List[Dict[str, str]]:
    """**正文第一部分「错在哪」**（2026-11-13 用户定的三段式）。

    来源：模型以教官身份做的整局复盘（``raw["coach"]["match"]``）。
    **不看规则命中** —— 它讲的是这一局真实发生的事。

    模型没写成时**不空着**：拿逐条明细里已经成立的事实顶上去
    （有害死亡 + 有问题的操作），每条一句话。宁可粗，也不能让第一部分空着 ——
    空着等于告诉读者「这一局没问题」。
    """
    coach = (ctx.raw or {}).get("coach") or {}
    match = coach.get("match") or {}
    items = [p for p in (match.get("wrong_points") or [])
             if isinstance(p, dict)]
    out: List[Dict[str, str]] = []
    for item in items[:6]:
        what = str(item.get("what") or "").strip()
        why = str(item.get("why") or "").strip()
        if not (what or why):
            continue
        out.append({"at": str(item.get("at") or "").strip(),
                    "what": what or why, "why": why})
    if out:
        return out
    # 兜底：从明细里能确认的事实收拢。
    for entry in _death_insight_entries(ctx):
        if entry.get("verdict_kind") == "ok":
            continue
        out.append({"at": str(entry.get("time_range") or "").replace("局内 ", ""),
                    "what": str(entry.get("title") or ""),
                    "why": "这一次死亡没换到东西，见下面的明细。"})
    for entry in _insight_entries(ctx):
        out.append({"at": str(entry.get("time_range") or ""),
                    "what": str(entry.get("title") or ""),
                    "why": str(entry.get("reason_text") or "")[:80]})
    return out[:6]


def _coach_analysis(ctx: MatchContext) -> List[Dict[str, str]]:
    """**正文第二部分「教官分析」** —— 模型以教练口吻展开（不看规则命中）。"""
    coach = (ctx.raw or {}).get("coach") or {}
    match = coach.get("match") or {}
    out: List[Dict[str, str]] = []
    # **两段式**（2026-11-14 用户定的）：第二段是「该怎么做」——
    # 模型参照打法要点 + 触发前提的对局描述，用自己的话给出。
    for item in (match.get("advice") or [])[:6]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        how = str(item.get("how") or "").strip()
        basis = str(item.get("basis") or "").strip()
        if not (title or how):
            continue
        out.append({"title": title or how[:20], "detail": "",
                    "fix": how, "basis": basis})
    if out:
        return out
    # 老数据（只有 analysis）照旧显示，别让旧对局空掉。
    for item in (match.get("analysis") or [])[:6]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        detail = str(item.get("detail") or "").strip()
        fix = str(item.get("fix") or "").strip()
        if not (title or detail):
            continue
        out.append({"title": title or detail[:20], "detail": detail,
                    "fix": fix})
    return out


def _death_insight_entries(ctx: MatchContext) -> List[Dict[str, Any]]:
    """**每次死亡一条**：这波赚还是亏、换到了什么、下次怎么做（2026-11-13）。

    它在报告里取代了旧的那一层（``_death_entries``：只写「怎么死的」+
    从攻略库检索一句建议）。区别就是用户要的那句：

    > 针对局内死亡，模型可以参考模式库的基础上，并自己搜索攻略，
    > 给出死亡分析，分析对局内有益还是有害。

    * ``trade`` 决定条目长什么样：有益 = 绿（``cell-v-ok``）、
      有害 = 红、看不清 = 灰；
    * 每条都带「换到了什么」（``what``）与理由（``why``），
      以及模型自己查回来的攻略原文（``guides``）。
    """
    coach = (ctx.raw or {}).get("coach") or {}
    deaths = [d for d in (coach.get("deaths") or []) if isinstance(d, dict)]
    if not deaths:
        return []
    try:
        from ..recognition2.coach import TRADE_LABELS
    except Exception:                                   # noqa: BLE001
        TRADE_LABELS = {}
    kind_of = {
        "beneficial": "ok",
        "harmful": "problem",
        "even": "occasional",
        "unknown": "undetermined",
    }
    ordered = sorted(deaths, key=lambda d: float(d.get("video_start") or 0.0))
    total = len(ordered)
    out: List[Dict[str, Any]] = []
    for index, item in enumerate(ordered, 1):
        trade = str(item.get("trade") or "unknown")
        clock = str(item.get("clock") or "").strip()
        where = str(item.get("where") or "").strip()
        cause = str(item.get("cause") or "").strip()
        label = TRADE_LABELS.get(trade, "")
        head = "第 %d 次死亡（这局共 %d 次）" % (index, total)
        if clock:
            head += " · %s" % clock
        if where:
            head += " · %s" % where
        if label:
            head += " · %s" % label
        reason: List[str] = []
        what = str(item.get("what") or "").strip()
        why = str(item.get("why") or "").strip()
        if what:
            reason.append("换到了什么：%s" % what)
        if why:
            reason.append("依据：%s" % why)
        if cause:
            reason.append("死因：%s" % cause)
        guides = [g for g in (item.get("guides") or []) if isinstance(g, dict)]
        if guides:
            g = guides[0]
            text = str(g.get("title") or "").strip()
            action = str(g.get("action") or "").strip()
            if text or action:
                reason.append("参照打法：%s%s"
                              % (text, ("｜%s" % action[:80]) if action else ""))
        advice = str(item.get("advice") or "").strip()
        if not advice and trade == "beneficial":
            advice = "这次换到了东西，这种取舍可以继续。"
        out.append({
            "kind": "death",
            "mode_id": "",
            "category": "死亡复盘",
            "lane_label": "",
            "phase_label": "",
            "title": head,
            "verdict_label": label or "看不出来值不值",
            "verdict_kind": kind_of.get(trade, "undetermined"),
            "time_range": ("局内 %s" % clock) if clock else "",
            "clip": {"video_url": "", "moments": [], "seconds": 0},
            "actual_text": "",
            "evidence": [],
            "absent_kinds": [],
            "advice_text": advice or "这一条没给出具体做法，见理由。",
            "advice_lines": _split_clauses(advice) if advice else [],
            "reason_text": "；".join(reason) or "（模型没写理由）",
            "low_confidence": False,
            "confidence": 1.0,
        })
    return out


def _insight_entries(ctx: MatchContext) -> List[Dict[str, Any]]:
    """**模型按这一局写的复盘建议**（攻略没覆盖、但画面里确实有问题的地方）。

    为什么要有它（用户原话，2026-11-13）
    ------------------------------------
    「报告层为什么不能根据对局真实情况，产出有价值分析，仅仅把模式库作为参考」。
    在那之前，正文里只有两类东西：

    * 模式库**判出来的偏离** —— 规则没覆盖到的地方，一条都没有；
    * 血线 / 死亡条目 —— 它们的建议也是从模式库检索来的，
      检索不到就写「这个节点没查到可对照的攻略」。

    于是报告读起来像一张「没查到可对照的攻略」的清单：这一局真实发生了什么、
    哪里该改、下次怎么做，反而没被讲。而模型**看过这一局**，
    它在综述那一步本来就写了「攻略之外的问题」（``findings``）——
    只是那一段从来没落库、也没上屏（2026-11-13 补上）。

    每条都走报告正文那套字段位，**模板一行都不用改**。
    """
    overview = (ctx.raw or {}).get("overview") or {}
    if not isinstance(overview, dict):
        return []
    # **逐段操作复盘优先**（用户 2026-11-13 的要求：每一段都要判「有没有问题」）。
    # 老对局没有 ``operations``，回落到 ``findings``（只挑最要紧的那几条）。
    operations = [o for o in (overview.get("operations") or [])
                  if isinstance(o, dict)]
    out: List[Dict[str, Any]] = []
    if operations:
        for item in operations:
            verdict = str(item.get("verdict") or "").strip().lower()
            if verdict != "problem":
                continue        # 「没问题」「看不清」不进调整清单
            what = str(item.get("what") or "").strip()
            why = str(item.get("why") or "").strip()
            advice = str(item.get("advice") or "").strip()
            at = str(item.get("at") or "").strip()
            guide = str(item.get("guide") or "").strip()
            if not (what or why):
                continue
            reason_parts: List[str] = []
            if why:
                reason_parts.append(why)
            if guide:
                reason_parts.append("参照打法：%s" % guide)
            reason_parts.append("这一条是按你这一局的画面推的，模式库只是参考。")
            out.append({
                "kind": "insight",
                "mode_id": "",
                "category": "操作复盘",
                "lane_label": "",
                "phase_label": "",
                "title": what or why[:40],
                "verdict_label": "这里可以调整",
                "verdict_kind": "problem",
                "time_range": at,
                "clip": {"video_url": "", "moments": [], "seconds": 0},
                "actual_text": "",
                "evidence": [],
                "absent_kinds": [],
                "advice_text": advice or "这一条没给出具体做法，见理由。",
                "advice_lines": _split_clauses(advice) if advice else [],
                "reason_text": "；".join(reason_parts),
                "low_confidence": False,
                "confidence": 1.0,
            })
        if out:
            return out

    for item in (overview.get("findings") or []):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        detail = str(item.get("detail") or "").strip()
        advice = str(item.get("advice") or "").strip()
        evidence = str(item.get("evidence") or "").strip()
        at = str(item.get("at") or "").strip()
        if not (title or detail):
            continue
        reason_parts: List[str] = []
        if detail:
            reason_parts.append(detail)
        if evidence:
            reason_parts.append("画面依据：%s" % evidence)
        # **说清这条是从哪来的** —— 它可能不在模式库里。
        # 不说清楚，读者会把「按这一局推的」当成「攻略说的」。
        reason_parts.append("这一条是按你这一局的画面推的，模式库只是参考。")
        out.append({
            "kind": "insight",
            "mode_id": "",
            "category": "复盘建议",
            "lane_label": "",
            "phase_label": "",
            "title": title or detail[:40],
            "verdict_label": "这里可以调整",
            "time_range": at,
            "clip": {"video_url": "", "moments": [], "seconds": 0},
            "actual_text": "",
            "evidence": [],
            "absent_kinds": [],
            "advice_text": advice or "这一条没给出具体做法，见理由。",
            "advice_lines": _split_clauses(advice) if advice else [],
            "reason_text": "；".join(reason_parts),
            "low_confidence": False,
            "confidence": 1.0,
        })
    return out


def _narrative_death_note(ctx: MatchContext) -> str:
    """模型叙述里写的阵亡，和死亡清单**对不上**时，把这件事说出来。

    为什么要说（2026-11-13 实测，``hok-video-8178dc031675``）
    ----------------------------------------------------------
    那一局的叙述里有 **6 个时段**写着玩家阵亡/倒地（0~2、2~4、4~6、6~8、
    12~14、14~16 分钟），而死亡清单只记了 2 次。对着录像核过：
    真值是 **3 次**（顶栏右上角的死亡计数 0→1→2→3），
    叙述前两分钟写的「完成一换一后倒地」**根本没发生**
    （00:47 与 02:41 两帧上那个计数都是 0）。

    也就是说：**那一列（「这一段画面里识别到的操作」）是模型自己讲的话，
    没有逐条核对过**。它写得像事实，读者就会当事实读。
    这里不替模型遮掩，也不假装清单就是对的 —— 两边对不上就说对不上，
    并告诉读者该拿什么去核（录像）。

    判据用**时段**而不是句子数：同一句话里「倒地」出现两次、
    或者下一句接着写「从倒地状态恢复」，都只算同一个时段的一件事。
    """
    rounds = list(ctx.iter_rounds())
    if not rounds:
        return ""
    try:
        from ..recognition2.death_review import looks_like_death
    except Exception:                                   # noqa: BLE001
        return ""
    claimed: List[str] = []
    for rnd in rounds:
        text = str(getattr(rnd, "narrative", "") or "")
        if not text:
            continue
        if not any(looks_like_death(part)
                   for part in re.split(r"[；。]", text)):
            continue
        label = str(getattr(rnd, "label", "") or "")
        if label and label not in claimed:
            claimed.append(label)
    if not claimed:
        return ""

    recognition = (ctx.raw or {}).get("recognition") or {}
    recorded = len(recognition.get("deaths") or [])
    reported = int(recognition.get("deaths_total") or 0) or recorded
    if claimed and reported and len(claimed) <= reported:
        # 叙述里写的时段不比清单多 —— 说明没多编，不必打扰读者。
        return ""
    return ("模型叙述里有 %d 个时段写到玩家阵亡（%s），而死亡清单只记了 %d 次 —— "
            "两边对不上（叙述是模型自己讲的话，没有逐条核对；清单也可能漏）。"
            "看这一列时请对着录像核。"
            % (len(claimed), "、".join(claimed), reported or recorded))


def _library_hint(rules: Sequence[Rule], hero: str, position: str) -> str:
    """候选为 0 的时候，告诉读者**规则库里到底有什么**。

    为什么必须写（2026-11-12 实测）：那一局读成「桑启 · 中路」，
    而规则库里「桑启」的攻略只有 9 条、**分路全写着「游走」** ——
    两边对不上 → 候选 0 条 → 报告整页是空的。
    只说「没对上」，读者不知道该改英雄还是该改分路；
    把库里那几条摆出来，才知道该往哪边改。

    只在**读到了英雄或分路**时才有话说；两样都没有就别硬凑。
    """
    hero = str(hero or "").strip()
    position = str(position or "").strip()
    if not hero and not position:
        return ""

    hero_lanes: Dict[str, int] = {}
    hero_n = 0
    pos_heroes: Dict[str, int] = {}
    pos_n = 0
    for rule in rules or []:
        conds = list(getattr(getattr(rule, "trigger", None), "all", []) or [])
        r_hero = ""
        r_lane = ""
        for cond in conds:
            fact = str(getattr(cond, "fact", "") or "")
            if fact == "my_hero" and not r_hero:
                r_hero = str(getattr(cond, "value", "") or "")
            elif fact == "position" and not r_lane:
                r_lane = str(getattr(cond, "value", "") or "")
        if hero and r_hero == hero:
            hero_n += 1
            if r_lane:
                hero_lanes[r_lane] = hero_lanes.get(r_lane, 0) + 1
        if position and r_lane == position:
            pos_n += 1
            if r_hero:
                pos_heroes[r_hero] = pos_heroes.get(r_hero, 0) + 1

    parts: List[str] = []
    if hero and hero_n:
        lanes_txt = "、".join(
            "%s %d 条" % (_position_label(lane), n)
            for lane, n in sorted(hero_lanes.items(), key=lambda kv: -kv[1]))
        parts.append("规则库里有「%s」的攻略 %d 条（%s）" % (hero, hero_n, lanes_txt))
    if position and pos_n:
        heroes_txt = "、".join(
            "%s %d 条" % (name, n)
            for name, n in sorted(pos_heroes.items(), key=lambda kv: -kv[1])[:6])
        parts.append("「%s」分路下的攻略 %d 条（%s）"
                     % (_position_label(position), pos_n, heroes_txt))
    if not parts:
        return ""
    return "；".join(parts) + "。"


def build_match_report(
    rules: Sequence[Rule],
    ctx: MatchContext,
    curve: Optional[Dict[str, Any]] = None,
    version_note: str = "",
    #: **叙述器**（可选）：把正文里最该是人话的两段交给文字模型写。
    #:
    #: None = 全用模板写法。传了就用模型写的 —— 但**写不出来就退回模板**，
    #: 报告永远照出（见 :mod:`app.report.narrative`）。
    narrator: Any = None,
    #: 新链路的 300 字分析报告（描述 + 规则 → 三部分）。非空时覆盖正文。
    analysis_report: str = "",
) -> Dict[str, Any]:
    """组装单局报告。"""
    adapter = get_adapter(ctx.game)
    #: v2 操作模式库 —— **只给诊断层用**（说明「为什么这一条没判」），
    #: 不参与判定。视频路径的判定归模型。
    patterns = _load_patterns(ctx)

    # -- 判定走哪条路 ------------------------------------------------------
    #
    # **视频来源的对局（``derived``）：判定权归模型。**
    #
    # 老内核（:func:`app.core.attribution.analyze_match`）是
    # 「规则引擎拿事实比对」：它判不了「前提能不能在画面里被看见」
    # 这类**语义**问题，于是要么不判、要么猜。
    #
    # 模型逐条裁决过清单（前提成立没、哪一支成立、玩家做的是什么），
    # 那是**带画面依据**的判断 —— 执行度应当由它统计。
    #
    # 玩家手工录入的对局没有画面、没有模型，仍走老内核
    # （事实是玩家给的，规则字面就是唯一依据）。
    #
    # 所以这里**按来源分流**：视频路径**不调用**老内核的判定
    # （阶段 5 的验收就是「视频路径上老内核调用点归零」），
    # 报告要的形状由 :mod:`app.harness.bridge` 转出来。
    is_video = (ctx.annotation_source or "").lower() == "derived"
    harness = None
    #: **模型裁决拿不到**（视频路径上 harness 建不起来）——
    #: 退回老内核时要说出来，否则这份报告与「玩家录入的对局」长得一样，
    #: 读者以为模型判过。
    harness_missing = False
    if is_video:
        try:
            from ..harness import analyze_from_model

            # **把「这一局本该检验的条目总数」传进去** ——
            # 不传的话分母就是「已检验到的那些」，覆盖率恒为 100%，
            # 哪怕模型只裁决了 1 条、清单里有 84 条。
            #
            # 分母取**候选条数**（模型被要求判的那些）——
            # 不是整库规则数：整库有 1680 条，而这一局只该判其中几条。
            _sel = ((ctx.raw or {}).get("recognition") or {}).get("selection") or {}
            _cand_ids = [x for x in (_sel.get("candidate_ids") or []) if x]
            # 老对局只存了**个数**，没存是哪几条 —— 那就用个数。
            _cand_n = len(_cand_ids) or int(_sel.get("candidates") or 0)
            harness = analyze_from_model(
                ctx, active_rules=_cand_n or None)
        except Exception:                               # noqa: BLE001
            harness = None
        if harness is not None:
            from ..harness.bridge import analysis_from_model

            analysis = analysis_from_model(rules, ctx, harness)
        else:
            # 模型这头完全不可用（连裁决都拿不到）——
            # 回退老内核，但**在报告上说明**（下面 version_note 会写）。
            harness_missing = True
            patterns = _load_patterns(ctx)
            analysis = analyze_match(rules, ctx, patterns=patterns)
    else:
        patterns = _load_patterns(ctx)
        analysis = analyze_match(rules, ctx, patterns=patterns)

    #: **能拿来纠正玩家**的模式 id 集合（None = 不筛）。
    #:
    #: 视频路径上是 **None（不筛）**：结论已经由**模型裁决**给出，
    #: 而模型裁决**过了接地闸**（前提看得见成立才纠正、帧号要真实、
    #: 玩家做过节点认可的事就不纠正 —— 见 :mod:`app.collection.vlm_review`）。
    #: 再按规则的**静态**可纠正性筛一遍，等于用「读不出来的前提」
    #: 去否决「模型看着画面确认了的前提」—— 那会把刚补上的结论又筛掉。
    #:
    #: 玩家录入路径本来就是 None（事实是玩家给的，规则字面就是唯一依据）。
    corrective = None

    gaps = []
    #: **诊断计数**：这一局丢了多少条结论、因为什么丢。
    report_notes: Dict[str, Any] = {}
    #: **这一局**算出来的可纠正性（v2 引擎）。它比静态的规则库判断更准：
    #: 分支前提被模型指认了，那条模式这一局就能纠正。
    runtime_corrective = {
        v.rule_id: v.corrective
        for v in analysis["verdicts"]
        if getattr(v, "corrective", None) is not None
    }
    for gap in analysis["gaps"]:
        # **不能拿来纠正的偏离不进报告**。
        #
        # 攻略的取舍由读不出来的前提决定 —— 原文说「该推塔」，可当时
        # 塔还有护盾、敌方打野就在我方半区，他退塔下才是对的。
        # 拿这种前提去纠正人就是在冤枉（用户原话：「视频做的没错与
        # 数据库不一致的不用纠正」）。
        if rules and corrective is not None:
            flag = runtime_corrective.get(gap.rule_id)
            if flag is not None:
                # v2 引擎给了这一局的结论 —— 以它为准。
                if not flag:
                    continue
            elif gap.rule_id not in corrective:
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

    # **前提诊断**：为什么这些规则一条都没触发。
    #
    # 没有它，「0 条结论」是个哑巴 —— 用户不知道是缺分路、缺英雄，
    # 还是自己打的不是这条路。2026-10-29 那次真实回归就是踩在这上面：
    # 分路被塞了「野区」，归一化失败 → 84 条规则全部不触发，
    # 而报告上什么都看不出来。
    diagnosis = None
    if patterns:
        from ..core.diagnose import diagnose_patterns
        diagnosis = diagnose_patterns(
            patterns,
            ctx,
            adapter,
            triggered_pattern_ids={v.rule_id for v in analysis["verdicts"]},
        )
    if diagnosis is None:
        diagnosis = diagnose(
            rules,
            ctx,
            adapter,
            triggered_rule_ids={v.rule_id for v in analysis["verdicts"]},
        )

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
        adapter, ctx, lanes, list(analysis["verdicts"]), rules, corrective,
        notes=report_notes, narrator=narrator,
    )

    # **失败局的倒推复盘** —— 每次死亡一条，放在正文**最前面**。
    #
    # 为什么放最前：用户要的是「从水晶爆炸往开局推」——
    # 离结局越近的死亡越靠前。
    #
    # **绕过规则库**：这一层自己走（规则库那套判不了死因）。
    #
    # **不许静默吞掉异常** —— 这一整组条目消失，报告却什么都不说，
    # 读者只会以为「这一局没有死亡/没有残血」。
    section_note = ""
    #: **每次死亡一条**（模型判了「赚还是亏」的那种）—— 有它就不再用旧的那套。
    coach_deaths = _death_insight_entries(ctx)
    try:
        death_entries = [] if coach_deaths else _death_entries(
            ctx, str(lanes.get("video_url") or ""))
    except Exception as exc:                           # noqa: BLE001
        death_entries = []
        section_note = "死亡复盘这一节没能生成（%s）。" % exc
    # **残血时段** —— 血量掉到阈值以下的片段，每段一条。
    #
    # 放在死亡复盘**后面**：死亡是败因（用户要的是从结局往回推），
    # 残血是风险点（这段差点出事）—— 顺序按「先说败因」排。
    try:
        hp_entries = _hp_entries(ctx, str(lanes.get("video_url") or ""))
    except Exception as exc:                           # noqa: BLE001
        hp_entries = []
        section_note = (section_note + "血线这一节没能生成（%s）。" % exc
                        if section_note else
                        "血线这一节没能生成（%s）。" % exc)
    death_note = ""
    # **血线条目不上屏了**（2026-11-14 用户拍板）。
    #
    # 原话：「这些不知所以然的血线描述是什么」—— 那几张卡只有
    # 「血量最低 22%」+「这一段没做攻略对照」/「处理方式同上一条」，
    # 玩家读完不知道发生了什么、也不知道该改什么。它是**风险点的原始读数**，
    # 不是结论；讲血量风险是教官分析那一段的事，不该在正文里刷六张卡。
    #
    # 数据仍然留着（``report["hp"]``）：成长曲线与以后的风险模型还要用它。
    if hp_entries:
        # 数量写进说明位（``report_notes`` 是给正文用的结构化留痕，
        # 这里借 ``hp_note`` 一个键带上；说明位在下面统一拼文案）。
        report_notes["hp_note"] = (
            "这一局有 %d 段血量低于阈值（原始读数留在数据里，不单独出条目）。"
            % len(hp_entries))
        hp_entries = []
    # **模型按这一局写的「复盘建议」** —— 放在**最前面**。
    #
    # 为什么要有这一节（用户原话，2026-11-13）：
    # 「报告层为什么不能根据对局真实情况，产出有价值分析，仅仅把模式库
    #  作为参考」。在那之前，正文里只有两类东西：
    #   * 模式库**判出来的偏离**（规则没覆盖到的地方一条都没有）；
    #   * 血线/死亡条目（它们的建议还是从模式库检索来的，
    #     检索不到就写「这个节点没查到可对照的攻略」）。
    # 于是玩家看到的是一堆「没查到可对照的攻略」——模式库成了唯一真源，
    # 而这一局**真实发生的事**反而没被讲。
    #
    # 这些条目来自 :mod:`app.recognition2.observer` 那一步：它在看完这一局
    # 之后写「攻略没覆盖、但画面里确实有问题」的地方（``findings``），
    # 每条带「下次怎么做」。**模式库只是参考**，不是结论的来源。
    insights = _insight_entries(ctx)
    death_note = ""
    #: 顺序：**死亡复盘（每次，判了赚亏）→ 操作复盘（有问题的段）→
    #: 残血时段 → 模式判定**。越靠前越「说的是你这一局」。
    _tail = _dedupe_advice(list(death_entries or []) + list(hp_entries or []))
    if coach_deaths or insights or _tail:
        mode_analysis = coach_deaths + insights + _tail + mode_analysis
    # **三段式的第三部分只要明细**（2026-11-13 用户拍板）：
    # 「不要再考虑命中规则的事情」—— 规则判出来的偏离（kind 为空的那批）
    # 不再进这个板块；它们仍然留在 ``report["gaps"]`` 里给成长曲线用。
    #
    # ⚠️ **只在确实有三段内容时才过滤**：老对局 / 模型没跑成的时候，
    # 正文里本来就只剩规则条目 —— 一并滤掉的话，报告会变成
    # 「这一局没有发现偏离」，那是把「没算」说成「没做错」。
    wrong_points_list = _wrong_points(ctx)
    coach_analysis_list = _coach_analysis(ctx)
    if coach_deaths or wrong_points_list or coach_analysis_list:
        _detail_kinds = ("death", "insight", "hp")
        mode_analysis = [e for e in mode_analysis
                         if str(e.get("kind") or "") in _detail_kinds]
    # **死亡次数要说准** —— 三个数各管一段：
    # 模型报了几次（真实总数）、落库几次（截断过）、写进复盘几次（有画面依据）。
    # 拿其中任一个当「这局共 N 次」都会说错。
    _rec = (ctx.raw or {}).get("recognition") or {}
    n_stored = len(_rec.get("deaths") or [])
    n_reported = int(_rec.get("deaths_total") or 0) or n_stored
    #: 写进复盘的次数 = 旧的接底闸通过的 + 模型判过赚亏的。
    n_grounded = len(death_entries) + len(coach_deaths)
    _death_parts: List[str] = []
    if n_reported > n_stored:
        _death_parts.append("模型报了 %d 次死亡，这次只记了时间上最后 %d 次"
                            % (n_reported, n_stored))
    if n_stored and n_grounded < n_stored:
        _death_parts.append(
            "记下来的 %d 次里有 %d 次在画面里没读到死亡依据，没有写进复盘"
            % (n_stored, n_stored - n_grounded))
    if _death_parts:
        death_note = "；".join(_death_parts) + "。"

    # **空正文必须说清是哪一种空** ——
    #
    # 「一条都没偏离」与「压根没得比」在页面上**长得一样**（都是空的），
    # 意思完全相反：前者是「做对了」，后者是「没分析」。
    # 2026-11-12 实测：最新那一局候选 0 条，正文却写
    # 「这一局没有发现偏离操作模式的地方」—— 读者会以为这一局没问题。
    stats_view = _stats_of(analysis, harness)
    empty_reason = _empty_reason(mode_analysis, stats_view)

    # 分析覆盖小结：每个时间窗看到了什么、判了几条（满足/偏离/无法判定）。
    # 让「报告只列偏离」之外，玩家能核对系统确实分析过每一段、没漏。
    coverage = build_coverage(ctx, list(analysis["verdicts"]), adapter)

    # **空报告必须能解释自己。**
    #
    # 「0 条结论」如果不给理由，用户看到的就是一张空表 —— 而空报告最常见
    # 的原因不是「打得没问题」，是**前提没采到或对不上**（2026-10-29 那次
    # 真实回归：分路被塞了「野区」，归一化失败 → 全部规则不触发，
    # 而报告上什么都看不出来）。
    #
    # 页面模板是**冻结**的（前端不改），所以这句话放进页面**已经在渲染**的
    # 说明位 ``version_note`` —— 模板一个字都不用动，页面上照样看得见。
    # 只在**确实没有结论**时补：有结论的报告不该再堆一段解释性文字
    # （用户明确说过：报告只回答「我该怎么改」）。
    #
    # ``mode_analysis`` 是**条目列表**（模板里 ``report.get('mode_analysis') or []``），
    # 空列表 = 这一局没找到任何需要调整的地方。
    if not version_note and not (mode_analysis or []):
        # 先说**丢了什么、为什么丢** —— 这比「信息不足」具体得多。
        dropped = (report_notes or {}).get("dropped") or {}
        parts: List[str] = []
        no_ground = int(dropped.get("no_ground") or 0)
        no_advice = int(dropped.get("no_advice") or 0)
        if no_ground:
            parts.append(
                "有 %d 条结论没有对应的画面依据，没有写进报告" % no_ground)
        if no_advice:
            parts.append(
                "有 %d 条结论没能指认攻略里的具体做法（模型也说不清"
                "这一局适用哪一支），没有写进报告" % no_advice)
        if parts:
            version_note = "；".join(parts) + "。"
        try:
            note = diagnosis.note(adapter)
        except Exception:                                   # noqa: BLE001
            note = ""
        if note:
            version_note = (version_note + note) if version_note else note

    # **模型判了但没进结论的条目，要写明挡了几条、为什么**（用户拍板的口径：
    # 维持严闸 —— 不写进结论区；但不许静默吞掉）。
    #
    # 为什么必须在报告上写：模型判「看不出来」的条目，
    # 在执行度里是「无法判定」，在结论区里是「不存在」。
    # 两边都不说的话，玩家看到的执行度会比他以为的高，
    # 而报告看上去像是「全都看过了」。
    if harness is not None:
        dropped_h = getattr(harness, "dropped", None) or {}
        n_unclear = int(dropped_h.get("unclear") or 0)
        n_branch = int(dropped_h.get("no_branch") or 0)
        bits: List[str] = []
        if n_unclear:
            bits.append("%d 条模型说画面里看不出来（没判对错）" % n_unclear)
        if n_branch:
            bits.append("%d 条模型没能指认是第几支成立" % n_branch)
        if bits:
            text = ("另有 " + "、".join(bits) + "，未纳入执行度统计。")
            version_note = (version_note + text) if version_note else text

        # **检索口径** —— 模型自己查了打法库的话，报告要说清
        # 「查了几次、召回几条、用上几条」。
        #
        # 为什么必须写：RAG 之后结论条数可能变多。
        # 不说清是「检索补齐了清单」还是「模型判得更细」，
        # 下次不知道该改检索还是改提示词 —— 归因错了就会一直白改。
        retrieval = getattr(harness, "retrieval", None) or {}
        n_searched = int(retrieval.get("searched") or 0)
        if n_searched:
            n_recalled = int(retrieval.get("recalled") or 0)
            n_used = int(retrieval.get("used") or 0)
            text = ("模型在 %d 段里自己查了打法库，召回 %d 条打法，其中 %d 条进了结论。"
                    % (n_searched, n_recalled, n_used))
            version_note = (version_note + text) if version_note else text

        # **交叉验证分歧** —— 模型裁决与纯代码比对结论相反的条目。
        #
        # 为什么必须写：判定权交给了模型，但模型也会看错。
        # 有一条独立的口径摆在旁边（规则条件直接比对），
        # 两边说法相反时读者才有机会自己核对。
        #
        # 口径：**以模型的裁决为准**（那是用户拍板的），
        # 代码这边只做对账，不参与判定、不进执行度分母。
        note = _cross_check_note(harness)
        if note:
            version_note = (version_note + note) if version_note else note

    # **模型没调通** —— 这一局等于没分析，必须说在报告上。
    #
    # 2026-10-31 实测：模型接口 Token Plan 配额用尽，所有窗都 429，
    # 报告却显示「0 条结论」。空表最容易被读成「打得没问题」——
    # 实际是根本没分析。宁可写清楚「没分析」，也不能装作分析完了。
    model_fail = [
        w for w in (recognition.get("warnings") or [])
        if isinstance(w, str) and "模型没调通" in w
    ]
    if model_fail:
        reason = ""
        for w in model_fail:
            # 原因在**第一对**括号里 ——
            # 老写法「模型没调通，这一窗读不出来（原因）」
            # 与新写法「模型没调通，70 段读不出来（原因）」都认。
            # （用第一对而不是固定标记：新旧两种留痕的量词不一样。）
            start = w.find("（")
            end = w.rfind("）")
            if 0 <= start < end:
                reason = w[start + 1:end]
                break
        usable = int(recognition.get("usable_rounds") or 0)
        failed = 0
        for w in model_fail:
            m = re.search(r"(\d+)\s*段读不出来", w)
            if m:
                failed = int(m.group(1))
                break
        if usable and failed:
            text = "模型接口调不通，有 %d 段没读出来" % failed
        elif usable:
            text = "模型接口调不通，有几段没读出来"
        else:
            text = "模型接口调不通，这一局没有生成画面结论"
        if reason:
            text += "（%s）" % reason
        text += "。"
        version_note = (version_note + text) if version_note else text

    # **筛子把候选全筛掉了** —— 这一局等于没裁决，必须说在报告上。
    #
    # 2026-11-02 实测（熬隐那局）：规则库里的 1680 条没有一条落在
    # 这一局的适用范围内 → 候选 0 → 模型一次都没裁决 → 报告正文空表，
    # 而**说明位也是空的** —— 用户看到的是一张空表加一堆叙述，
    # 完全看不出「为什么没有结论」。
    #
    # 原因其实**记下来了**（识别层留的痕在 ``recognition.warnings`` 里），
    # 只是没人把它写出来。空表必须能解释自己。
    screened_note = [
        w for w in (recognition.get("warnings") or [])
        if isinstance(w, str) and ("没有要检验的规则" in w
                                   or "没有这一局要检验的条目" in w)
    ]
    # **闸门**：这句话只在**筛子真的一条都没筛出**时才能写。
    # 有裁决就说明筛子**筛出了东西** —— 这时再说「没有要检验的规则」
    # 是**说谎**（哪怕那些裁决后来被别的闸挡掉了，挡掉的原因
    # 说明位里已经写了）。
    screened_nothing = (
        not (mode_analysis or [])
        and not (getattr(harness, "verdicts", None) or [])
    )
    if screened_note and screened_nothing:
        text = ("这一局没有要检验的规则 —— 规则库里的条目没有一条落在"
                "这一局的适用范围内（分路 / 英雄 / 时间窗都没对上），"
                "所以模型没有裁决任何一条。")
        #: **是哪一步没对上** —— 扫描层已经数好了（``筛掉明细：…``），
        #: 原样带上去：读者不用猜该改分路、该改英雄还是该改时间轴。
        detail = [w for w in (recognition.get("warnings") or [])
                  if isinstance(w, str) and w.startswith("筛掉明细：")]
        if detail:
            text += " " + detail[0] + "。"
        version_note = (version_note + text) if version_note else text

    # **候选不是模型自己找的** —— 攻略条目是「整条筛」出来的。
    #
    # 为什么必须写：这一局的**口径变了**。
    # 模型自己找的 = 「它看着画面，认为这几段值得对照攻略」；
    # 退回整条筛的 = 「凡是分路/英雄/时间窗对得上的都拿来判了」——
    # 里面很可能有这一局压根没发生的事。
    #
    # 读者有权知道这一点：不然他会把「模型挑出来的偏离」当成
    # 「这一局该做而没做的全部清单」。
    selection = recognition.get("selection") or {}
    if str(selection.get("mode") or "") == "fallback":
        text = ("这次的攻略条目不是模型自己找的（检索没走通，退回按分路/英雄/"
                "时间窗整条筛），里面可能有这一局压根没发生的事。")
        version_note = (version_note + text) if version_note else text
    # **检索能力降级** —— 稠密路没走通时，检索只剩字面匹配：
    # 说得不一样但意思相同的那条打法就找不到了。
    # 这会改变「这一局该对照哪几条」，所以要在报告上说明。
    degraded = str(selection.get("degraded") or "")
    if degraded:
        text = "这次找攻略只按字面匹配（向量检索没走通：%s）。" % degraded
        version_note = (version_note + text) if version_note else text

    # **英雄没认准** —— 英雄层的打法这次一条都没判。
    #
    # 2026-11-02 实测（熬隐那局）：模型看整局录像说玩家玩「不知火舞」，
    # 认技能按钮说「弈星」，两个对不上。真身是敖隐 —— 英雄层一旦按
    # 「不知火舞」去筛，报告就会拿**别的英雄的打法**指责玩家，
    # 而且玩家一看就知道「我根本不是这个英雄」。
    #
    # 这属于**口径变了**：这次只判了分路层的打法。必须写出来。
    #
    # 两种情形都要说：
    # * **两个来源对不上**（结构化字段里有）；
    # * **两个来源都没认准**（识别层留了痕，但结构化字段是空的）。
    #
    # 后一种 2026-11-02 实测踩到过：认技能按钮的把握不够（0.58 < 0.60），
    # 模型也没说出英雄名 → 英雄层一条没判，而报告页**一个字都没提** ——
    # 读者只会以为这就是全查过了。
    conflict = recognition.get("hero_conflict") or {}
    hero_note = ""
    decision = str(conflict.get("decision") or "") if isinstance(conflict, dict) else ""
    voided = recognition.get("hero_voided") or []
    voided_text = ""
    if isinstance(voided, list):
        parts = []
        for v in voided:
            if not isinstance(v, dict):
                continue
            name = str(v.get("hero") or "")
            why = str(v.get("why") or "")
            if name:
                parts.append("%s（%s）" % (name, why or "不合"))
        if parts:
            voided_text = "否掉了 %s" % "、".join(parts)
    if isinstance(conflict, dict) and conflict and decision == "clip":
        # **以「模型读选人界面那段画面上的字」为准**（2026-11-03 新口径）。
        hero_note = ("这一局的英雄是按**选人界面那段画面上印着的名字**定的"
                     "（模型读到的是 %s）"
                     % (conflict.get("clip") or "（空）"))
        if conflict.get("model") and conflict.get("model") != conflict.get("clip"):
            hero_note += ("；模型看完整局说的是 %s" % conflict.get("model"))
        if voided_text:
            hero_note += "；另外%s" % voided_text
        hero_note += "。"
    elif isinstance(conflict, dict) and conflict and decision == "bp":
        # **以 BP 记录里我方最后选的那个为准** —— 同样是画面上印着的字。
        hero_note = ("这一局的英雄是按**选人记录里我方最后选的那个**定的"
                     "（%s）" % (conflict.get("bp") or "（空）"))
        if conflict.get("model") and conflict.get("model") != conflict.get("bp"):
            hero_note += ("；模型看完整局说的是 %s" % conflict.get("model"))
        if voided_text:
            hero_note += "；另外%s" % voided_text
        hero_note += "。"
    elif isinstance(conflict, dict) and conflict and decision == "watch":
        # **以模型看完整局的读数为准**（2026-11-03 新口径下的最后一条路）。
        hero_note = ("这一局的英雄以模型看完整局的读数为准（模型说 %s）"
                     % (conflict.get("model") or "（空）"))
        if voided_text:
            hero_note += "；另外%s" % voided_text
        hero_note += "。"
    elif isinstance(conflict, dict) and conflict and decision == "button_model":
        # **以「专门认技能按钮」那一问的读数为准**（2026-11-03 老口径）。
        #
        # 这一局的情况是：看完整局的模型说「马超」，而专门认技能按钮那一问
        # 说「敖隐」（按钮图标匹配分也支持它）。英雄层是按**敖隐**判的，
        # 读者有权知道这一点 —— 尤其当他自己一眼就看出不是马超的时候。
        hero_note = ("这一局的英雄以「专门认技能按钮」那一问的读数为准"
                     "（看完整局说 %s、专门认技能按钮说 %s）—— "
                     "那一问看的是技能图标特写，按钮图标匹配分也支持它。"
                     % (conflict.get("model") or "（空）",
                        conflict.get("button_model") or "（空）"))
    elif isinstance(conflict, dict) and conflict and decision == "model":
        # **以模型看完整局的读数为准**（用户 2026-11-03 拍板）。
        # 技能按钮那个读数取样点锚在「对局起点」上，起点抖 ±5 秒第一名就换人。
        hero_note = ("这一局的英雄以模型看完整局的读数为准"
                     "（模型说 %s、技能按钮认成 %s）—— "
                     "技能按钮那个取样点会因为对局起点差几秒就换一批帧。"
                     % (conflict.get("model") or "（空）",
                        conflict.get("icon") or "（空）"))
    elif isinstance(conflict, dict) and conflict:
        # **老对局**（2026-11-02 当时的做法）：两个来源对不上 → 两个都不采信。
        hero_note = ("这一局的英雄没认准（模型说 %s、技能按钮认成 %s）—— "
                     "英雄层的打法这次一条都没判，只判了分路层的。"
                     % (conflict.get("model") or "（空）",
                        conflict.get("icon") or "（空）"))
    elif voided_text:
        # **所有读数都被否掉了** —— 说清楚为什么，别让读者以为是「查过了没问题」。
        hero_note = "这一局的英雄没认准 —— %s；英雄层的打法这次一条都没判。" % voided_text
    else:
        # 结构化字段没有，就从**识别层留的痕**里找 ——
        # 留痕在 ``recognition.warnings``，出口只有 ``version_note`` 一个。
        for w in (recognition.get("warnings") or []):
            if not isinstance(w, str):
                continue
            if ("认英雄这一步两个来源对不上" in w
                    or "认英雄这一步几个读数对不上" in w
                    or "认英雄这一步有个读数被否掉" in w
                    or "没认准这一局的英雄" in w):
                hero_note = w
                break
    if hero_note:
        version_note = (version_note + hero_note) if version_note else hero_note

    # **胜负没读到** —— 报告顶上因此没有结果栏，得说清楚为什么。
    #
    # 2026-11-02 实测：读胜负那一步的模型调用返回了空，识别层记的是
    # ``"unknown"``。页面模板是 ``{'失败' if result == 'lose' else '胜利'}`` ——
    # 任何非空值都会被当成「胜利」，那一局明明是输的。
    # 现在模板不动，改为**读不到就不给结果栏**，并在这里说明为什么没有。
    for w in (recognition.get("warnings") or []):
        if isinstance(w, str) and w.startswith("这一局的胜负没读到"):
            version_note = (version_note + w) if version_note else w
            break

    # **分路判错了（或者判法变了）** —— 分路是整份报告的前提，
    # 前提错了结论就全错，读者必须知道。
    #
    # 2026-11-07 实测：那一局玩家走的是**上路**，报告却写成「中路」——
    # 模型顶层那个分路字段孤证独大，错了没人拦。现在模型**逐段**标路、
    # 按众数定分路，跟顶层读数对不上时以画面为准，并把这件事留痕
    # （识别层写进 warnings，前缀「分路：」）。
    for w in (recognition.get("warnings") or []):
        if isinstance(w, str) and w.startswith("分路："):
            version_note = (version_note + w) if version_note else w
            break

    # **血量这一项的留痕**（前缀「血量：」）——
    #
    # 「没读到血量」与「这一局没有残血」是**两回事**：前者是压根没测，
    # 后者是测了没发现。不说清楚，读者会把前者读成后者 ——
    # 以为「血线没问题」，其实这一项根本没结论。
    for w in (recognition.get("warnings") or []):
        if isinstance(w, str) and w.startswith("血量："):
            version_note = (version_note + w) if version_note else w
            break

    # **覆盖表整列是「—」** —— 读者会以为「没看」或者「看不出来」。
    #
    # 实际可能是第三种：画面读到了（每一段都有动作），但**一条可对照的
    # 攻略都没进候选** —— 那三列自然是「—」。2026-11-07 实测就是这样：
    # 覆盖表九行全是「—」，而真实原因是「候选 0 条」。
    #
    # 「没看到画面」与「看到了但没有可比对的打法」不是一回事，
    # 读者按前者理解就会以为系统没干活。
    if not int(coverage.get("checked") or 0):
        n_sampled = sum(1 for w in (coverage.get("windows") or [])
                        if isinstance(w, dict) and w.get("sampled"))
        if n_sampled:
            # **不要 Markdown** —— 说明位是**纯文本**，页面原样输出，
            # 星号会变成「**一条…**」摆在读者眼前（2026-11-12 实测）。
            text = ("%d 段画面都读到了，但这一局一条可对照的攻略都没进候选，"
                    "所以「做对 / 需调整 / 判不了」全是「—」" % n_sampled)
            version_note = (version_note + text + "。") if version_note else text + "。"
            # **库里到底有什么** —— 读者要靠这句知道该改英雄还是该改分路。
            _player = ctx.player or {}
            hint = _library_hint(rules,
                                 str(_player.get("hero") or ""),
                                 str(_player.get("position") or ""))
            if hint:
                version_note = (version_note + " " + hint) if version_note else hint

    # **检索本身调不通** —— 这跟「库里没有」要分开说。
    #
    # 2026-11-07 实测：这一局每个检索请求的 error 被丢在了写库那一步，
    # 报告只能写「0 条打法」，看着像库里没东西 —— 读者会去补资料，
    # 而真正该修的是接口。所以把失败原因摆出来。
    for hit in ((recognition.get("selection") or {}).get("hits") or []):
        if not isinstance(hit, dict):
            continue
        err = str(hit.get("error") or "").strip()
        if not err:
            continue
        text = "检索攻略时有一次调用没调通（%s）—— 这一局的候选可能不全。" % err
        version_note = (version_note + text) if version_note else text
        break

    # **打法库连不上** —— 与「库里没有」分开说。
    #
    # 这两件事该做的事完全相反：连不上要修连接，库里没有要补资料。
    # 检索层以前把两者都说成「库是空的」，那句话还会喂给模型 ——
    # 模型于是建议用户去补攻略，方向完全错。
    for hit in ((recognition.get("selection") or {}).get("hits") or []):
        if not isinstance(hit, dict):
            continue
        reason = str((hit.get("degradation") or {}).get("reason") or "").strip()
        if not reason.startswith("打法库连不上"):
            continue
        text = ("%s —— 这一局的候选可能不全；该做的是修连接，不是补攻略。"
                % reason)
        version_note = (version_note + text) if version_note else text
        break

    # **有死亡却没复盘** —— 说清楚，别让读者以为是漏了。
    if death_note:
        version_note = (version_note + death_note) if version_note else death_note

    # **叙述写的阵亡次数与死亡清单对不上** —— 那一列是模型自己讲的话，
    # 不核就说，读者会整段当成事实读（见 :func:`_narrative_death_note`）。
    narrative_note = _narrative_death_note(ctx)
    if narrative_note:
        version_note = (version_note + narrative_note) if version_note else narrative_note

    # **英雄的官方分路与这一局判出来的分路对不上** —— 英雄层的打法
    # 一条也套不上，读者得知道为什么（见 :func:`_hero_lane_mismatch_note`）。
    mismatch_note = _hero_lane_mismatch_note(dict(ctx.player or {}))
    if mismatch_note:
        version_note = (version_note + mismatch_note) if version_note else mismatch_note

    # **英雄层的攻略不覆盖这局的打法**（前缀「英雄：」）——
    # 那不是「没认准」，而是「英雄是按 A 路写的、你这局打 B 路」，
    # 这时报告只按分路通则判，得让读者知道为什么英雄层的条目一条都没有。
    for w in (recognition.get("warnings") or []):
        if isinstance(w, str) and w.startswith("英雄："):
            version_note = (version_note + w) if version_note else w
            break

    # **死亡次数对不上** —— 模型数的 vs 顶栏 K/D/A 推出来的（2026-11-13）。
    #
    # 这一句必须有：死亡复盘是按模型那份写的，而「顶栏」是画面上印着的数字。
    # 对不上时不替读者选边，把两个数都摆出来，让他对着录像核 ——
    # 指错一次死亡的代价，比漏一次大得多。
    check = (ctx.raw or {}).get("kda_check") or {}
    if isinstance(check, dict) and check and not check.get("ok"):
        model_n = int(check.get("model_count") or 0)
        hud_n = int(check.get("hud_count") or 0)
        text = ("死亡次数对不上：模型报了 %d 次，顶栏 K/D/A 推出来 %d 次"
                "（对上 %d 次）—— 死亡复盘是按模型那份写的，"
                "对不上的地方请对着录像核。" % (model_n, hud_n,
                                                int(check.get("matched") or 0)))
        missed = check.get("missed_by_model") or []
        if missed:
            text += "顶栏显示还有 %d 次没被复盘。" % len(missed)
        extra = check.get("not_on_hud") or []
        if extra:
            text += "另有 %d 次在顶栏对不上（可能是识别错，也可能是读取时被遮挡）。" % len(extra)
        version_note = (version_note + text) if version_note else text

    # **BP 记录是模型看头像认的** —— 那一段画面上没印英雄名。
    # 报告里显示「谁选了谁」时，读者会把它当记录看；得说清它是猜的。
    if str(recognition.get("bp_source") or "") == "model":
        text = ("BP 记录里那些英雄名是模型看选人界面的头像认的"
                "（那一段画面上只印昵称、分路与「预选」，不印英雄名）—— "
                "名字请核对。")
        version_note = (version_note + text) if version_note else text

    # **这一局是抽帧看的** —— 与整局视频不是一回事，粗细差很多。
    # 不说清楚，读者会把「两次采样之间没看到的事」当成「没发生」。
    if str(recognition.get("watch_mode") or "") == "frames":
        text = ("这一局的画面是抽帧看的（模型收不了整局视频：按画面变化抽帧、"
                "每 5 分钟一批）—— 两次采样之间发生的事看不到，"
                "所以叙述比整局视频粗，死亡与动作也可能漏。")
        version_note = (version_note + text) if version_note else text

    # **模型裁决没拿到**（视频路径上 harness 建不起来）——
    # 报告是按规则条件直接比对得出的，不说清楚，读者以为模型判过。
    if harness_missing:
        text = ("这次没拿到模型裁决，报告里的结论是按规则条件直接比对得出的。"
                "模型看的画面结论这次一条都没参与。")
        version_note = (version_note + text) if version_note else text

    # **判定口径降级了** —— 动作类清单取不到时，v2 的动作类判定项会按标量算，
    # 「做对了」可能判成「偏离」。这种降级不能只存在心里。
    for _deg in (analysis.get("degradation") or []):
        _text = str(_deg or "").strip()
        if not _text:
            continue
        version_note = (version_note + _text) if version_note else _text

    # **某一节没生成出来** —— 必须在说明位说一句，
    # 否则读者只会以为「这一局没有死亡 / 没有残血」。
    if section_note:
        version_note = (version_note + section_note) if version_note else section_note
    # **血线条目不上屏之后，数量还是要交代一句**（2026-11-14）。
    _hp_note = str((report_notes or {}).get("hp_note") or "").strip()
    if _hp_note:
        version_note = (version_note + _hp_note) if version_note else _hp_note

    # **叙述没写成** —— 也要说。
    #
    # 叙述层写不出来时正文退回模板写法，报告**照样是对的**，
    # 但用户看到的是「和以前一样死板」，会以为功能没生效。
    # 这里说清楚「没写成 + 为什么」，比让人猜强。
    # 只在**一条都没写成**时说 —— 写成了就不该再堆解释（报告只回答
    # 「我该怎么改」）。
    if narrator is not None:
        calls = int(getattr(narrator, "calls", 0) or 0)
        # 只看**正文**（我方动作 / 建议）有没有换成模型写的 ——
        # 「只有建议理由换了模型写法」的话，报告页上看着还是老样子，
        # 那就该说出来，而不是当成「叙述写成了」。
        written = int(getattr(narrator, "written_body", 0) or 0)
        if calls and not written:
            reasons = [n for n in (getattr(narrator, "notes", None) or []) if n]
            text = "正文没能改成模型写的叙述，仍用模板写法"
            text += ("（%s）" % reasons[0]) if reasons else ""
            text += "。"
            version_note = (version_note + text) if version_note else text

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
        "player": _player_view(ctx),
        # **胜负只上屏真的读到的那个**（读不到就不给结果栏，别写「胜利」）。
        "outcome": _shown_outcome(ctx.outcome),
        # **执行度按判定来源取**：视频来源的对局由模型裁决统计，
        # 玩家录入的对局才用规则引擎的数字。
        "stats": stats_view,
        # **正文空的时候是哪一种空** —— 「没偏离」和「没得比」长得一样，
        # 意思相反。空字符串 = 保留模板原来的「没发现偏离」。
        "mode_analysis_empty_reason": empty_reason,
        # **harness 口径**：这次判定是谁做的、挡了几条、为什么。
        # 报告 JSON 里始终带着，页面读的是 ``version_note``。
        "harness": (harness.to_dict() if harness is not None else None),
        # **交叉验证**：模型裁决与纯代码比对的对账结果。
        # 判定以模型为准，这里只是把「两边说法相反」的条目摊开。
        "cross_check": _cross_check(harness),
        # **逐条裁决** —— 落库后供成长报告重建逐规则曲线。
        "verdicts": _verdicts_for_curve(harness, analysis),
        "round_count": analysis["round_count"],
        "gaps": gaps,
        # 分路 × 阶段的运营结论 —— 报告的主视图。
        # gaps 是逐回合的明细，适合复查；lanes 是按分路收拢后的结论，
        # 回答的是「我这条分路哪一段做错了、应该怎么做」。
        "lanes": lanes,
        # 报告正文真正渲染的那一份：分路 + 逐回合合成，每条四部分。
        "mode_analysis": mode_analysis,
        # **三段式的第一、二部分**（2026-11-13 用户定的）：
        #   ① wrong_points  这一局错在哪（总结）
        #   ② coach_analysis 模型当教官给的分析
        #   ③ mode_analysis  逐条明细（死亡复盘 / 操作复盘 / 血线）
        # ①② **不看规则命中** —— 讲的是这一局真实发生的事。
        "wrong_points": wrong_points_list,
        "coach_analysis": coach_analysis_list,
        # 分析覆盖小结：逐时间窗的采样动作与判定计数（满足/偏离/无法判定）。
        "coverage": coverage,
        "by_category": by_category,
        "undetermined": undetermined,
        # 判不了的原因汇总：哪一类动作读不出来，挡掉了多少条规则。
        "unreadable_kinds": unreadable,
        # 前提诊断：为什么这些规则没触发。空报告时它就是唯一的解释。
        "diagnosis": diagnosis.to_dict(),
        "curve": curve or {},
        "version_note": version_note,
        # 以下三项仅视频来源的对局非空，手工录入 / API 导入时为空
        "recognition": recognition,
        "low_confidence_facts": _low_confidence_facts_of(ctx, gaps),
        # 事实中文名：报告里要展示「血量 58（45%）」而不是
        # 「health 58（45%）」。随报告下发，模板无需反查适配器。
        "fact_labels": dict(adapter.fact_labels),
        # 新链路：描述 + 规则 → 300 字三部分报告（非空时模板用它当正文）。
        "analysis_report": analysis_report,
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

    # **空正文要说清是哪一种空** ——
    #
    # 「每个环节都合规」与「压根没得比」在页面上**长得一样**（都是空的），
    # 意思完全相反：前者是「做对了」，后者是「没录 / 没得比」。
    # 之前 BP 复盘没有说明位，空页只能写死那句「都符合操作模式库里的做法」——
    # 那是在**没得比**的时候撒谎（2026-11 复核：录了空 decisions 的一局，
    # 页面写「每个环节的操作都符合操作模式库里的做法」）。
    _stats = analysis["stats"].to_dict()
    # **键名是 ``checked_rules``**（``ExecutionStats.to_dict`` 里的那个），
    # 不是 ``checked`` —— 写错的话这个数恒为 0，
    # 「没得比」那句话就会盖住「都合规」那一种。
    _checked = int(_stats.get("checked_rules") or 0)
    empty_reason = ""
    if not ctx.rounds:
        empty_reason = ("这一局没录到你的 BP 决策 —— 没有可比的环节，"
                        "所以没能给出复盘。")
    elif not entries and _checked <= 0:
        empty_reason = ("这一局的 BP 环节没有一条能对上操作模式库里的条目 —— "
                        "没有可比的东西，所以没能给出复盘。")

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
        # **空正文的理由** —— 页面据此区分「都合规」与「没得比」。
        "empty_reason": empty_reason,
        "summary": {
            "steps": len(ctx.rounds),
            "my_heroes": my_heroes,
            "opp_heroes": opp_heroes,
        },
        "stats": _stats,
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
        # **说明位**：讲清口径 + 把最该改的规则用模型原话写出来。
        "version_note": _growth_note(analyses, weakest),
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
