"""差距归因：把判定结果变成「差在哪一局、哪一回合、哪一步」。

立项设计里最关键的产品主张：
    现有工具说「你经济崩了」。
    我们说「第 4-1 回合你剩 58 血、连胜 4 连，按规则应该停手抢经济，
          你却直接追三导致经济断档」。

所以一条归因记录必须同时携带：触发时的局面事实、规则要求的动作、
玩家实际的动作、以及这处偏离本身的严重程度 —— 缺一不可复查。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .context import MatchContext, RoundRecord
from .dsl import EXPECT_SCOPE_AT_LEAST_ONCE, Rule
from .matcher import (
    STATUS_DEVIATED,
    STATUS_SATISFIED,
    STATUS_UNDETERMINED,
    Verdict,
    match_rules,
)

#: 低于此置信度的判定，报告里必须显式标记为「低置信度」。
LOW_CONFIDENCE_THRESHOLD = 0.75


@dataclass
class Gap:
    """一处可定位的差距。"""

    rule_id: str
    rule_title: str
    category: str
    severity: str
    round_label: str
    #: 触发条件成立时的局面快照 —— 报告里「你剩 58 血、连胜 4 连」的来源
    situation: Dict[str, Any]
    expected: Dict[str, Any]
    actual: Dict[str, Any]
    rationale: str
    hint: str
    #: 规则原文的前提 / 动作 / 后果（见 :class:`app.core.dsl.Rule`）。
    premise: str = ""
    action: str = ""
    outcome: str = ""
    source: Optional[Dict[str, Any]] = None
    #: 参与判定的事实中是否存在低置信度识别结果
    low_confidence: bool = False
    confidence: float = 1.0
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_title": self.rule_title,
            "category": self.category,
            "severity": self.severity,
            "round_label": self.round_label,
            "situation": self.situation,
            "expected": self.expected,
            "actual": self.actual,
            "rationale": self.rationale,
            "hint": self.hint,
            "premise": self.premise,
            "action": self.action,
            "outcome": self.outcome,
            "source": self.source,
            "low_confidence": self.low_confidence,
            "confidence": round(self.confidence, 3),
            "tags": self.tags,
        }

    def headline(self, unit: str = "回合") -> str:
        """一句话结论 —— 报告的时间线列表直接用它。

        ``unit`` 是这个游戏的操作单位：自走棋是「回合」，
        王者荣耀没有回合（它是一局连续的一整场），叫「时间片」。
        """
        return f"{self.round_label} {unit}｜{self.rule_title}"


def _situation_from_evidence(evidence: List[Dict[str, Any]],
                              context: Optional[Dict[str, Any]] = None
                              ) -> Dict[str, Any]:
    """把触发证据压成「局面事实」快照。

    触发证据只包含规则**用到的**那几个事实，而报告要回答的是
    「当时是什么局面」—— 所以这里再叠上这一帧的全部标量事实。
    两边同名时以触发证据为准（那是规则实际拿到的取值）。
    """
    snapshot = dict(context or {})
    snapshot.update({
        e["fact"]: e["actual"]
        for e in evidence
        if e.get("present") and e.get("actual") is not None
    })
    return snapshot


def _verdict_confidence(v: Verdict) -> float:
    confs = [
        a.get("confidence", 1.0)
        for a in v.actual.get("actions", [])
    ]
    return min(confs) if confs else 1.0


def to_gaps(
    verdicts: List[Verdict],
    skip_rules: Optional[Set[str]] = None,
    once_rules: Optional[Set[str]] = None,
) -> List[Gap]:
    """把偏离判定转成归因记录。

    ``skip_rules`` 里的规则不产出偏差记录。
    用来处理「一次性口径」的规则：整段只要做到过一次就算做到了，
    那几帧「没做到」就不该再单独列成偏差 ——
    否则报告一边说「做对了」，一边挂着 5 条偏差，自相矛盾。

    ``once_rules``（一次性口径的规则）**每条只出一条记录**：
    同一分钟里切了三片、这三片都没做到 —— 那是**一件事**
    （「这一段没做到」），不是三件。

    这条对王者特别要紧：切片是**按规则密度**切的（密的地方多切几刀），
    密的地方正是最容易重复判到同一条规则的地方。按帧记，
    同一个错会被放大成好几条，报告看着像玩家连犯了好几次。
    """
    gaps: List[Gap] = []
    skip = skip_rules or set()
    once = once_rules or set()
    seen_once: Set[str] = set()

    for v in verdicts:
        if v.status != STATUS_DEVIATED:
            continue
        if v.rule_id in skip:
            continue
        if v.rule_id in once:
            if v.rule_id in seen_once:
                continue
            seen_once.add(v.rule_id)
        conf = _verdict_confidence(v)
        gaps.append(
            Gap(
                rule_id=v.rule_id,
                rule_title=v.rule_title,
                category=v.category,
                severity=v.severity,
                round_label=v.round_label or "未知",
                situation=_situation_from_evidence(v.evidence, v.context),
                expected=v.expected,
                actual=v.actual,
                rationale=v.rationale,
                hint=v.deviation_hint,
                premise=v.premise,
                action=v.action,
                outcome=v.outcome,
                source=v.source,
                low_confidence=conf < LOW_CONFIDENCE_THRESHOLD,
                confidence=conf,
                tags=list(v.tags),
            )
        )

    return gaps


@dataclass
class ExecutionStats:
    """一局对局的规则执行度统计。"""

    triggered: int = 0
    satisfied: int = 0
    deviated: int = 0
    undetermined: int = 0
    active_rules: int = 0
    #: 本局被真正检验过的规则 id。
    #: 必须与 ``triggered`` 区分开：triggered 数的是「回合」，
    #: 而 coverage 要表达的是「规则」，两者单位不同，不能相除。
    checked_rule_ids: Set[str] = field(default_factory=set)

    @property
    def decided(self) -> int:
        """可判定总数 —— 无法判定的必须从分母里剔除，否则会虚高。"""
        return self.satisfied + self.deviated

    @property
    def compliance_rate(self) -> Optional[float]:
        """执行度。没有任何规则被触发时返回 None —— 不该编造一个数字。"""
        if self.decided == 0:
            return None
        return self.satisfied / self.decided

    @property
    def coverage(self) -> float:
        """触发覆盖率：规则库里有多少条规则在这局中真的被检验过。

        分母是规则条数，分子是被检验过的规则条数，取值恒在 0~1。
        它回答的是「这局数据够不够得出可信结论」—— 覆盖率低意味着大量
        规则根本没机会触发，此时执行度的参考价值有限。
        """
        if self.active_rules == 0:
            return 0.0
        return min(1.0, len(self.checked_rule_ids) / self.active_rules)

    def by_category(
        self,
        verdicts: List[Verdict],
        once_rules: Optional[Set[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """按分类统计。

        ``once_rules`` 里是「一次性口径」的规则 id：它们整段只算一次
        （做到过就算做到），不能按帧计数 —— 否则同一条规则会在
        「分路 × 阶段」里显示「做对了」，在执行度里却显示「80% 偏离」。
        """
        once = once_rules or set()
        buckets: Dict[str, Dict[str, Any]] = {}
        once_state: Dict[Tuple[str, str], str] = {}

        for v in verdicts:
            b = buckets.setdefault(
                v.category, {"satisfied": 0, "deviated": 0, "triggered": 0}
            )
            b["triggered"] += 1
            if v.status not in (STATUS_SATISFIED, STATUS_DEVIATED):
                continue
            if v.rule_id in once:
                key = (v.category, v.rule_id)
                prev = once_state.get(key)
                if prev is None:
                    once_state[key] = v.status
                elif prev == STATUS_DEVIATED and v.status == STATUS_SATISFIED:
                    once_state[key] = STATUS_SATISFIED
                continue
            b[v.status] += 1

        for (category, _rule_id), status in once_state.items():
            buckets[category][status] += 1

        for b in buckets.values():
            decided = b["satisfied"] + b["deviated"]
            b["compliance_rate"] = (
                (b["satisfied"] / decided) if decided else None
            )
        return buckets

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggered": self.triggered,
            "satisfied": self.satisfied,
            "deviated": self.deviated,
            "undetermined": self.undetermined,
            "decided": self.decided,
            "active_rules": self.active_rules,
            "compliance_rate": (
                round(self.compliance_rate, 4)
                if self.compliance_rate is not None
                else None
            ),
            "coverage": round(self.coverage, 4),
            "checked_rules": len(self.checked_rule_ids),
        }


def analyze_match(
    rules: List[Rule],
    ctx: MatchContext,
    only_rules: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """完整分析一局对局。

    返回可直接喂给报告层的结构：判定明细 + 归因列表 + 执行度统计。
    """
    active = [r for r in rules if r.status == "active"]
    if only_rules:
        keep = set(only_rules)
        active = [r for r in active if r.id in keep]

    ctx.precompute()
    verdicts = match_rules(active, ctx, only_rules=only_rules)

    #: 「一次性口径」的规则：整段做到过一次就算做到。
    #:
    #: 这类规则不能按帧计数 —— 攻略说的是「中期这一段必须转线」，
    #: 不是「每一帧都得转」。按帧算，玩家 5 分钟里转了 1 次线、
    #: 其余 4 分钟在正常打，达成率 20% → 执行度显示 20%，
    #: 而分路结论说「做对了」。两个数字互相打脸，用户没法信任何一个。
    once_rules = {
        r.id for r in active if r.expect_scope == EXPECT_SCOPE_AT_LEAST_ONCE
    }

    stats = ExecutionStats(active_rules=len(active))
    #: 一次性规则每条只记一个状态：做到过就是 satisfied。
    once_state: Dict[str, str] = {}
    for v in verdicts:
        if v.status == STATUS_UNDETERMINED:
            stats.undetermined += 1
            stats.triggered += 1
            stats.checked_rule_ids.add(v.rule_id)
            continue
        if v.status not in (STATUS_SATISFIED, STATUS_DEVIATED):
            continue
        stats.triggered += 1
        # 无论判定为何，只要触发过就算「这条规则本局被检验过」
        stats.checked_rule_ids.add(v.rule_id)
        if v.rule_id in once_rules:
            prev = once_state.get(v.rule_id)
            if prev is None:
                once_state[v.rule_id] = v.status
            elif prev == STATUS_DEVIATED and v.status == STATUS_SATISFIED:
                # 做到过就算做到 —— 后面的「没做到」不再抵消它
                once_state[v.rule_id] = STATUS_SATISFIED
            continue
        if v.status == STATUS_SATISFIED:
            stats.satisfied += 1
        else:
            stats.deviated += 1

    for status in once_state.values():
        if status == STATUS_SATISFIED:
            stats.satisfied += 1
        else:
            stats.deviated += 1

    #: 一次性规则里已经做到过的那些，不再单独产出偏差记录。
    #:
    #: 否则报告一边说「这一段做对了」，一边挂着 5 条「该做而没做」——
    #: 同一件事两种说法。用户会以为是系统有 bug，而实际上系统是自洽的：
    #: 做到过就是做到过，那几帧只是没做那件事而已。
    skip_rules = {
        rule_id for rule_id, status in once_state.items()
        if status == STATUS_SATISFIED
    }
    gaps = to_gaps(verdicts, skip_rules=skip_rules, once_rules=once_rules)

    #: 分路 × 阶段的运营结论。报告主视图用它 ——
    #: 逐回合判定回答「这一下该不该控龙」，而用户要修的是
    #: 「整段都在做反」这件事，粒度不同。
    from .lane_analysis import analyze_lanes

    lanes = analyze_lanes(active, ctx, verdicts)

    return {
        "match_id": ctx.match_id,
        "game": ctx.game,
        "game_version": ctx.game_version,
        "player": ctx.player,
        "outcome": ctx.outcome,
        "round_count": len(ctx.rounds),
        "stats": stats,
        "gaps": gaps,
        "lanes": lanes,
        "verdicts": verdicts,
        "by_category": stats.by_category(verdicts, once_rules=once_rules),
        "undetermined": [v for v in verdicts if v.status == STATUS_UNDETERMINED],
    }
