"""分路 × 阶段的操作模式评估。

为什么要在逐回合判定之上再做一层
--------------------------------
逐回合判定回答的是「第 7 分钟这一下该不该控龙」。但攻略讲的是**运营模式**：
「发育路中期一塔之后必须转线」不是某一秒的动作，而是**一整段都在做的事**。

同一段里偏离 5 次，逐回合看是 5 个独立失误；按阶段看是「你这一整段都在
守线补兵」—— 一个系统性问题，能给的改法也只有一条。把前者原样摊给用户，
他会去修 5 个细节；后者才是运营模式真正想指的东西。

所以这一层只做**聚合**：
    逐回合判定  ->  按 (分路, 阶段) 收拢  ->  「哪里操作有问题 / 应该怎么做」

它不改判定，只决定**怎么讲**。判定仍然逐回合可复查 —— 聚合只是把同一段
里的结论合上，具体落在哪一回合由 :attr:`ModeFinding.round_labels` 带着。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .context import MatchContext
from .dsl import EXPECT_SCOPE_AT_LEAST_ONCE, Rule
from .matcher import (
    STATUS_DEVIATED,
    STATUS_SATISFIED,
    STATUS_UNDETERMINED,
    Verdict,
)

#: 分路的呈现顺序 —— 从适配器取，报告、标注页、筛选条共用同一份。
#: 打野 → 中路 → 对抗路 → 发育路 → 游走，读起来就是一路打完一路的顺序。
def _adapter_of(game: str):
    """取适配器；没注册就返回 None。

    内核必须**游戏无关** —— 测试用一个没进注册表的虚构游戏跑全流程，
    这里抛 KeyError 就等于把内核绑死在了具体游戏上。所以查不到就退化成
    最小可用的那套：没有中文名就显示编码本身，没有分路顺序就按编码排。
    """
    from ..adapters.base import get_adapter

    try:
        return get_adapter(game)
    except KeyError:
        return None


def _position_order() -> List[str]:
    ad = _adapter_of("hok")
    return list(ad.position_order) if ad else []


def _phase_labels() -> Dict[str, str]:
    ad = _adapter_of("hok")
    return dict(ad.phase_labels) if ad else dict(PHASE_LABELS_FALLBACK)


#: 适配器里没有阶段表时的兜底叫法。内核不该因为不认识某款游戏就说不出话。
PHASE_LABELS_FALLBACK: Dict[str, str] = {
    "early": "前期",
    "mid": "中期",
    "late": "后期",
}

#: 阶段顺序：前期 → 中期 → 后期 → 决胜期。
PHASE_ORDER: List[str] = ["early", "mid", "late"]

#: 低于这个达成率，且偏离次数 >= 2，就判为「整段做反」而不是偶发失误。
#:
#: 阈值是拍的，但它有明确用途：把「偶发一次没转线」和「一整段都在守线」
#: 分开。后者才是运营模式要指的东西，也是用户真正要改的那一条。
SYSTEMIC_RATE = 0.6

#: 结论取值。
VERDICT_OK = "ok"                  # 这段做对了
VERDICT_OCCASIONAL = "occasional"  # 大体对，偶发偏离
VERDICT_SYSTEMIC = "systemic"      # 整段做反 —— 系统性问题
VERDICT_NEVER = "never"            # 这段一次都没做该做的
VERDICT_UNDETERMINED = "undetermined"  # 动作没标，判不了

VERDICT_LABELS: Dict[str, str] = {
    VERDICT_OK: "这一段做得不错",
    VERDICT_OCCASIONAL: "偶尔可以调整",
    VERDICT_SYSTEMIC: "这一段可以换个思路",
    VERDICT_NEVER: "这一段暂时没看到推荐打法",
    VERDICT_UNDETERMINED: "信息不足，暂不评价",
}


@dataclass
class ModeFinding:
    """一条操作模式在某一局里的表现。"""

    mode_id: str
    title: str
    lane: str
    lane_label: str
    phase: str
    phase_label: str
    category: str
    severity: str
    #: 「这一局你应该做什么」
    should_do: str
    #: 「你实际在做什么」—— 从偏离回合的实际动作里统计出来的
    actually_doing: str
    verdict: str
    verdict_label: str
    #: 「怎么改」
    how_to_fix: str
    #: 「为什么」，来自攻略原文
    why: str
    #: 规则原文的前提 / 动作 / 后果。报告写「为什么」时用这三个字段，
    #: 不用 ``why`` —— 那是引文（带「原文前提：」这种前缀），
    #: 放到报告里读者看不到任何关于这一局的信息。
    premise: str = ""
    action: str = ""
    outcome: str = ""
    rounds_triggered: int = 0
    rounds_satisfied: int = 0
    rounds_deviated: int = 0
    round_labels: List[str] = field(default_factory=list)
    #: 出问题的那几个回合在原视频里的位置与截图。
    #:
    #: 只有视频来源的对局才有内容（``video_time > 0``）；手工录入的对局这里是空的，
    #: 界面上就不显示「看这一段」。内核不认识视频 —— 它只把坐标带出来，
    #: URL 由报告层按 video_id 拼。
    moments: List[Dict[str, Any]] = field(default_factory=list)
    #: 探测器找到的**证据时刻**（这一段里画面上出现过什么）。
    #:
    #: 光给「做对了/没做对」玩家没法确认 —— 报告要能指出
    #: 「12:34 那几秒在小地图上看到你在龙坑」。
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    #: 明确**没做**的动作类（信号可读、画面里没找到）。
    absent_kinds: List[str] = field(default_factory=list)
    source: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)

    @property
    def cell(self) -> str:
        return f"{self.lane_label}·{self.phase_label}"

    @property
    def is_problem(self) -> bool:
        return self.verdict in (VERDICT_SYSTEMIC, VERDICT_NEVER, VERDICT_OCCASIONAL)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode_id": self.mode_id,
            "title": self.title,
            "lane": self.lane,
            "lane_label": self.lane_label,
            "phase": self.phase,
            "phase_label": self.phase_label,
            "category": self.category,
            "severity": self.severity,
            "should_do": self.should_do,
            "actually_doing": self.actually_doing,
            "verdict": self.verdict,
            "verdict_label": self.verdict_label,
            "how_to_fix": self.how_to_fix,
            "why": self.why,
            "premise": self.premise,
            "action": self.action,
            "outcome": self.outcome,
            "rounds_triggered": self.rounds_triggered,
            "rounds_satisfied": self.rounds_satisfied,
            "rounds_deviated": self.rounds_deviated,
            "round_labels": self.round_labels,
            "moments": self.moments,
            "evidence": self.evidence,
            "absent_kinds": self.absent_kinds,
            "source": self.source,
            "tags": self.tags,
        }


def _lane_labels(game: str) -> Dict[str, str]:
    ad = _adapter_of(game)
    return dict(ad.position_labels) if ad else {}


def mode_cell(rule: Rule) -> Optional[Tuple[str, Optional[List[int]]]]:
    """从一条模式里读出「属于哪条分路」和「哪个时间窗」。

    刻意从**触发条件**读而不是解析 ``category`` 字符串 —— 分类名是给人看的，
    改了措辞数据就跟着漂；而 ``position == jungle`` 是数据本身。

    返回 ``(分路, 时间窗)``；没有 position 条件的规则返回 None，
    因为按分路评估的前提就是它属于某条分路。
    """
    lane: Optional[str] = None
    window: Optional[List[int]] = None
    for cond in rule.trigger.all:
        if cond.fact == "position" and cond.op in ("==", "in"):
            values = cond.value if isinstance(cond.value, list) else [cond.value]
            real = [v for v in values if v != "unknown"]
            # 写了全部分路（如「全位置」的模式）不属于任何单一分路，
            # 按分路评估时不该挂在某一路名下 —— 返回 None 让调用方跳过。
            if len(real) != 1:
                return None
            lane = real[0]
        elif cond.fact == "time_min" and cond.op == "between":
            if isinstance(cond.value, (list, tuple)) and len(cond.value) == 2:
                window = [cond.value[0], cond.value[1]]
    if lane is None:
        return None
    return lane, window


def phase_of_window(window: Optional[List[int]]) -> str:
    """时间窗 -> 局势阶段。

    公开出来是因为展示层也要用同一份口径：模式库把一条模式归到
    「前期 / 中期 / 后期」哪一档，判的是同一件事。口径分成两处的话，
    库里出现「窗口 4-10」这种跨档的写法时，两边会各说各话。
    """
    if not window:
        return "early"
    lo = window[0]
    if lo < 5:
        return "early"
    if lo < 11:
        return "mid"
    return "late"


def _describe_expected(rule: Rule, action_labels: Dict[str, str]) -> str:
    """把「期望动作」翻成人话 —— 报告里「应该怎么做」那一栏。"""
    values = rule.expect_value
    values = values if isinstance(values, list) else [values]
    names = [action_labels.get(v, v) for v in values]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]}或{names[1]}"
    return "、".join(names[:-1]) + f"或{names[-1]}"


def _describe_actual(
    deviated: List[Verdict], action_labels: Dict[str, str], limit: int = 2
) -> str:
    """统计偏离回合里玩家实际在做啥，取最常见的一两项。

    这一栏是「哪里操作有问题」的**事实部分**：不能只说「偏离了」，
    得说清楚他当时在干什么 —— 否则用户不知道该改哪个动作。

    先看该模式期望的那一类动作；那一类一个都没做时，退回看这一回合
    做的**所有**事。早先只看期望那一类，于是「去抓人而不是刷野」这种
    偏离会写成「（未标）」—— 明明做了很多事，报告却说成什么都没做，
    「怎么改」那一条也就给不出方向。
    """
    counts: Dict[str, int] = {}
    for v in deviated:
        same_kind = v.actual.get("actions") or []
        pool = same_kind or (v.actual.get("all_actions") or [])
        for a in pool:
            code = a.get("value") or a.get("label") or ""
            if not code:
                continue
            name = a.get("label") or action_labels.get(code, code)
            counts[name] = counts.get(name, 0) + 1
    if not counts:
        return ""
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:limit]
    return "、".join(f"{name}" for name, _ in top)


def build_findings(
    rules: List[Rule],
    ctx: MatchContext,
    verdicts: List[Verdict],
    action_labels: Dict[str, str],
) -> List[ModeFinding]:
    """把逐回合判定收成「分路 × 阶段」的结论。"""
    lane_labels = _lane_labels(ctx.game)
    by_rule: Dict[str, List[Verdict]] = {}
    for v in verdicts:
        by_rule.setdefault(v.rule_id, []).append(v)

    findings: List[ModeFinding] = []
    phase_labels = _phase_labels()
    #: 回合标签 -> 该回合在原视频里的位置。只收录视频来源的对局，
    #: 手工录入的对局查不到就是空 dict，报告据此不显示播放入口。
    moments_by_label: Dict[str, Dict[str, Any]] = {
        r.label: {"time": r.video_time, "image": r.image}
        for r in ctx.rounds
        if r.video_time > 0
    }
    #: 回合标签 -> 那一段的探测器证据。
    #:
    #: 报告里「这段在 12:34 出现开龙迹象」靠它 —— 玩家得能回看原画面确认，
    #: 光给一句「没做到」没有用。
    evidence_by_label: Dict[str, List[Dict[str, Any]]] = {
        r.label: list(r.probe_moments or [])
        for r in ctx.rounds
        if r.probe_moments
    }
    absent_by_label: Dict[str, List[str]] = {
        r.label: list(r.probe_absent or [])
        for r in ctx.rounds
        if r.probe_absent
    }
    for rule in rules:
        if rule.status != "active":
            continue
        cell = mode_cell(rule)
        if cell is None:
            continue
        lane, window = cell
        phase = phase_of_window(window)

        vs = by_rule.get(rule.id, [])
        satisfied = [v for v in vs if v.status == STATUS_SATISFIED]
        # 这条模式期望的动作**取值**（证据按它过滤）。
        _ev_values = rule.expect_value
        _ev_values = (_ev_values if isinstance(_ev_values, list)
                      else [_ev_values])
        deviated = [v for v in vs if v.status == STATUS_DEVIATED]
        undetermined = [v for v in vs if v.status == STATUS_UNDETERMINED]
        decided = len(satisfied) + len(deviated)

        if rule.expect_scope == EXPECT_SCOPE_AT_LEAST_ONCE:
            # 这一段里做到过就算做到了。
            #
            # 攻略写的是「中期必须转线」「能拿就尽量拿」这类**一次性要求**，
            # 不是「每一分钟都得转一次」。按帧算达成率的话，
            # 玩家 5 分钟里转了 1 次线、其余 4 分钟在正常打，
            # 达成率 20% → 判成「整段做反」—— 冤枉人。
            if not vs:
                # 这条规则整局都没进过（连适用的时刻都没有）——
                # 不进报告。与逐帧口径下的处理一致。
                continue
            if satisfied:
                verdict = VERDICT_OK
                deviated = []          # 那几帧不构成独立偏差
            elif deviated:
                verdict = VERDICT_NEVER
            else:
                verdict = VERDICT_UNDETERMINED
        elif decided == 0:
            # 整段都没触发：这套时间窗里玩家压根没进入这条模式适用的局面
            # —— 或者动作一个都没标。前者是「不适用」，不进报告；
            # 后者要说清楚，否则用户会以为系统没看见。
            if not undetermined:
                continue
            verdict = VERDICT_UNDETERMINED
        else:
            rate = len(satisfied) / decided
            if len(deviated) == 0:
                verdict = VERDICT_OK
            elif rate == 0:
                verdict = VERDICT_NEVER
            elif rate < SYSTEMIC_RATE and len(deviated) >= 2:
                verdict = VERDICT_SYSTEMIC
            else:
                verdict = VERDICT_OCCASIONAL

        # 列出哪些回合。一次性口径下只列**做到过**的那几帧 ——
        # 列一堆「没做到」的回合会让人以为那是偏差。
        listed = satisfied if (rule.expect_scope == EXPECT_SCOPE_AT_LEAST_ONCE
                               and satisfied) else vs

        # 这一段里探测器看到的证据（哪几秒、画面上看到了什么）。
        #
        # **只留与这条模式期望的那件事相关的** ——
        # 不然一条「该带线」的模式会显示出「在泉水停留」这种
        # 属于别的动作的证据，读起来莫名其妙。
        #
        #: 过滤按**取值**（不只是动作种类）：「带线」与「分推」都是
        # objective_action，只看种类会串。
        _evidence = [
            dict(m, label=v.round_label)
            for v in vs if v.round_label
            for m in (evidence_by_label.get(v.round_label) or [])
            if not rule.expect_fact or m.get("kind") == rule.expect_fact
            if not _ev_values or m.get("value") in _ev_values
        ][:6]

        # 出问题的片段：**优先用模型看到那件事的那一帧**。
        #
        # 为什么：报告的「原视频操作的文字描述」说的是模型在某一帧
        # 看到的事（那一帧就在 ``_evidence`` 里），而片段播放器原来放的是
        # 回合的代表性切片 —— 两者**不是同一时刻**。玩家点开一看
        # 画面和文字说的不是一回事，整条结论就跟着塌了。
        #
        # 没有证据帧时（模型没给出帧号）才退回回合的位置。
        _moments = [
            {
                "label": m.get("label") or "",
                "time": m.get("t") or 0,
                "image": m.get("image") or "",
            }
            for m in _evidence
            if m.get("t")
        ][:6]
        if not _moments:
            _moments = [
                {"label": v.round_label,
                 "time": moments_by_label[v.round_label]["time"],
                 "image": moments_by_label[v.round_label]["image"]}
                for v in deviated
                if v.round_label in moments_by_label
            ]

        findings.append(ModeFinding(
            mode_id=rule.id,
            title=rule.title,
            lane=lane,
            lane_label=lane_labels.get(lane, lane),
            phase=phase,
            phase_label=phase_labels.get(phase, phase),
            category=rule.category,
            severity=rule.severity,
            should_do=_describe_expected(rule, action_labels),
            actually_doing=_describe_actual(deviated, action_labels),
            verdict=verdict,
            verdict_label=VERDICT_LABELS[verdict],
            how_to_fix=rule.deviation_hint,
            why=rule.rationale,
            premise=rule.premise,
            action=rule.action,
            outcome=rule.outcome,
            rounds_triggered=len(vs),
            rounds_satisfied=len(satisfied),
            rounds_deviated=len(deviated),
            round_labels=[v.round_label for v in listed if v.round_label],
            # 出问题的片段：模型看到那件事的那一帧（见上面）。
            moments=_moments,
            # 这一段里探测器看到的证据（哪几秒、画面上看到了什么）。
            evidence=_evidence,
            # 明确没做的动作类（信号可读、画面里没找到）。
            absent_kinds=sorted({
                k for v in vs if v.round_label
                for k in (absent_by_label.get(v.round_label) or [])
            }),
            source=rule.source.to_dict() if rule.source else None,
            tags=list(rule.tags),
        ))

    return findings


def _rank(f: ModeFinding) -> Tuple:
    """问题排前面，且「整段做反 / 整段没做」排在「偶有偏离」前面。"""
    problem_rank = {
        VERDICT_SYSTEMIC: 0, VERDICT_NEVER: 1,
        VERDICT_OCCASIONAL: 2, VERDICT_UNDETERMINED: 3, VERDICT_OK: 4,
    }
    sev_rank = {"high": 0, "medium": 1, "low": 2}
    order = _position_order()
    return (
        f.is_problem is False,
        problem_rank.get(f.verdict, 9),
        sev_rank.get(f.severity, 3),
        order.index(f.lane) if f.lane in order else 9,
        PHASE_ORDER.index(f.phase) if f.phase in PHASE_ORDER else 9,
    )


def analyze_lanes(
    rules: List[Rule],
    ctx: MatchContext,
    verdicts: List[Verdict],
) -> Dict[str, Any]:
    """按分路给出这一局的运营结论。

    返回 ``findings``（全部模式，按问题优先排序）、``lanes``（按分路分组）
    与 ``cells``（分路 × 阶段网格，供页面渲染成表格）。
    """
    ad = _adapter_of(ctx.game)
    action_labels = dict(ad.action_labels) if ad else {}
    findings = build_findings(rules, ctx, verdicts, action_labels)
    findings.sort(key=_rank)

    lanes: List[Dict[str, Any]] = []
    cells: List[Dict[str, Any]] = []
    for lane in _position_order():
        mine = [f for f in findings if f.lane == lane]
        if not mine:
            continue
        problems = [f for f in mine if f.is_problem]
        lanes.append({
            "lane": lane,
            "lane_label": mine[0].lane_label,
            "mode_count": len(mine),
            "problem_count": len(problems),
            "findings": [f.to_dict() for f in mine],
        })
        for phase in PHASE_ORDER:
            for f in mine:
                if f.phase == phase:
                    cells.append({
                        "lane": lane,
                        "lane_label": f.lane_label,
                        "phase": phase,
                        "phase_label": f.phase_label,
                        "cell": f.cell,
                        "finding": f.to_dict(),
                    })

    return {
        "lane": ctx.player.get("position", ""),
        "lanes": lanes,
        "cells": cells,
        "findings": [f.to_dict() for f in findings],
        "problem_count": sum(1 for f in findings if f.is_problem),
    }
