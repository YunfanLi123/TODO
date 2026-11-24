"""v2 判定结果 → v1 判定结果 的桥接层。

为什么要桥接
------------
内核换成了 v2（:mod:`app.core.dsl2`），但报告层、归因层、页面都认
v1 的 :class:`~app.core.matcher.Verdict` 结构。**前端不能改**，
所以契约必须保持不变 —— 换引擎不换契约。

这一层只做**形状转换**，不做任何判定：
所有「算不算偏离」「能不能纠正」的结论都来自 v2 引擎，
桥接层只是把它摊平成 v1 认得的字段。

字段对应里有一处刻意的取舍：v1 的 ``round_label`` 指向**一个**回合，
而 v2 判定的是**一个节点**（可能横跨多个回合）。桥接层取节点里
第一个适用回合的标签作为定位点（报告里的「看这一段」靠它），
完整的回合列表放进 ``context["rounds"]``，不丢信息。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .dsl2 import CHECK_OK, Pattern, PatternVerdict, match_patterns
from .matcher import Verdict

#: v2 的「豁免」在 v1 里没有对应状态 ——
#: v1 遇到「玩家做了节点认可的另一件事」是直接不产出判定（返回 None）。
#: 桥接层保持一致：豁免的判定不进来。
_DROPPED_STATUSES = {"exempt"}


def _evidence(outcomes) -> List[Dict[str, Any]]:
    """v2 的判定项结果 → v1 的证据条目。"""
    out: List[Dict[str, Any]] = []
    for o in outcomes:
        out.append({
            "fact": o.check.fact,
            "op": o.check.op,
            "threshold": o.check.value,
            "actual": o.actual,
            "present": o.present,
            "result": o.status == CHECK_OK,
            "group": "all",
            # v2 额外带的字段：证据等级与说明。
            "provenance": o.provenance,
            "detail": o.detail,
            "count": o.count,
        })
    return out


def _expected(pattern: Pattern) -> Dict[str, Any]:
    if not pattern.expect:
        return {}
    first = pattern.expect[0]
    return {"fact": first.fact, "op": first.op, "value": first.value}


def _actual(pattern: Pattern, verdict: PatternVerdict) -> Dict[str, Any]:
    kind = pattern.expect[0].fact if pattern.expect else ""
    values = verdict.performed.get(kind, [])
    actions = [
        r for r in verdict.performed_records if r.get("kind") == kind
    ]
    return {
        "fact": kind,
        "actions": actions,
        "values": values,
        "all_actions": list(verdict.performed_records),
    }


def _context(pattern: Pattern, verdict: PatternVerdict) -> Dict[str, Any]:
    """这一帧的局面快照 —— 标量事实，不含动作事实（与 v1 同口径）。"""
    action_kinds = {r.get("kind") for r in verdict.performed_records}
    ctx: Dict[str, Any] = {}
    if verdict.rounds:
        first = verdict.rounds[0]
        ctx = {
            k: v for k, v in (first.get("facts") or {}).items()
            if k not in action_kinds and v is not None
        }
    ctx["rounds"] = [dict(r) for r in verdict.rounds]
    ctx["node_label"] = pattern.node.label
    ctx["node_from_min"] = pattern.node.from_min
    ctx["node_to_min"] = pattern.node.to_min
    ctx["pattern_score"] = round(verdict.score(), 4)
    if verdict.branch_no:
        ctx["branch_no"] = verdict.branch_no
    if verdict.branch_detail:
        # 分支级明细：这一局成立的是哪一支，玩家做的是不是那一支。
        ctx["branch_detail"] = dict(verdict.branch_detail)
    ctx["corrective_reason"] = verdict.corrective_reason
    return ctx


def verdict_from_pattern(
    pattern: Pattern,
    verdict: PatternVerdict,
) -> Optional[Verdict]:
    """一条 v2 判定 → 一条 v1 判定。豁免的返回 ``None``。"""
    if verdict.status in _DROPPED_STATUSES:
        return None
    rounds = verdict.rounds or []
    first = rounds[0] if rounds else {}
    return Verdict(
        rule_id=pattern.id,
        rule_title=pattern.title,
        category=pattern.category,
        severity=pattern.severity,
        status=verdict.status,
        round_label=first.get("label") or pattern.node.label,
        round_stage=int(first.get("stage") or 0),
        round_phase=int(first.get("phase") or 0),
        round_time=float(first.get("time_index") or 0.0),
        evidence=_evidence(verdict.when_outcomes),
        expected=_expected(pattern),
        actual=_actual(pattern, verdict),
        context=_context(pattern, verdict),
        rationale=pattern.rationale,
        deviation_hint=pattern.deviation_hint,
        premise=pattern.premise,
        action=pattern.action,
        outcome=pattern.outcome,
        source=dict(pattern.source) if pattern.source else None,
        tags=list(pattern.tags),
        # v2 的可纠正性是**这一局**算出来的（分支前提被指认了就能纠正），
        # 与 v1 的静态布尔不同 —— 报告闸门优先用它。
        corrective=verdict.corrective,
    )


def patterns_to_verdicts(
    patterns: Sequence[Pattern],
    ctx,
    *,
    action_facts: Optional[set] = None,
    only_patterns: Optional[List[str]] = None,
) -> List[Verdict]:
    """跑 v2 引擎并把结果摊平成 v1 判定列表。

    形状与 :func:`app.core.matcher.match_rules` 一致，
    所以报告层不需要知道内核换过。
    """
    keep = set(only_patterns) if only_patterns else None
    picked = [
        p for p in patterns
        if p.status == "active" and (keep is None or p.id in keep)
    ]
    raw = match_patterns(picked, ctx, action_facts=action_facts or set())
    by_id = {p.id: p for p in picked}
    out: List[Verdict] = []
    for verdict in raw:
        pattern = by_id.get(verdict.pattern_id)
        if pattern is None:
            continue
        bridged = verdict_from_pattern(pattern, verdict)
        if bridged is not None:
            out.append(bridged)
    return out


def pattern_scopes(patterns: Sequence[Pattern]) -> Dict[str, str]:
    """模式 id -> 满足口径。归因层用它判断「一次性口径」的规则。"""
    return {p.id: p.scope for p in patterns}
