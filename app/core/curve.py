"""成长曲线与版本适配。

立项设计里两个长期有效性机制：

1. 成长曲线记录的不只是胜率，而是**规则执行度随时间的变化** ——
   这戳破玩家最容易自我欺骗的地方：段位涨了，但打的还是老套路。

2. 版本适配：游戏一更新攻略就更新（天然的好处），但上版本的部分建议
   到新版本已不成立。所以规则库按版本独立存储，版本切换时标记失效条目，
   并用玩家历史对局回测 —— 明确告诉玩家这个版本哪些老习惯该改了。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .context import MatchContext
from .dsl import Rule


# --------------------------------------------------------------------------
# 成长曲线
# --------------------------------------------------------------------------


@dataclass
class CurvePoint:
    """曲线上的一个点 = 一局对局。"""

    match_id: str
    played_at: str            # ISO 日期
    game: str
    game_version: str
    compliance_rate: Optional[float]
    triggered: int
    satisfied: int
    deviated: int
    outcome: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "played_at": self.played_at,
            "game": self.game,
            "game_version": self.game_version,
            "compliance_rate": self.compliance_rate,
            "triggered": self.triggered,
            "satisfied": self.satisfied,
            "deviated": self.deviated,
            "outcome": self.outcome,
        }


@dataclass
class RuleCompliance:
    """单条规则跨对局的执行情况。"""

    rule_id: str
    rule_title: str
    category: str
    points: List[Optional[float]] = field(default_factory=list)
    checks: int = 0

    @property
    def trend(self) -> str:
        """趋势判断：改进了 / 退步了 / 持平。"""
        vals = [v for v in self.points if v is not None]
        if len(vals) < 2:
            return "数据不足"
        delta = vals[-1] - vals[0]
        if delta > 0.1:
            return "改进了"
        if delta < -0.1:
            return "退步了"
        return "持平"

    @property
    def latest_rate(self) -> Optional[float]:
        vals = [v for v in self.points if v is not None]
        return vals[-1] if vals else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_title": self.rule_title,
            "category": self.category,
            "points": self.points,
            "checks": self.checks,
            "latest_rate": self.latest_rate,
            "trend": self.trend,
        }


def _extract_counts(analysis: Dict[str, Any]) -> tuple:
    """从一份分析结果里取出执行度计数。

    兼容两种输入形态 —— :func:`~app.core.attribution.analyze_match` 的实时
    输出（执行度嵌在 ``stats`` 子对象里）和数据库快照的扁平结构
    （执行度在顶层）。两种形态在系统里都会出现，这里统一收口，
    免得调用方因为数据来源不同而漏字段。
    """
    stats = analysis.get("stats")
    source = stats if isinstance(stats, dict) and stats else analysis
    return (
        source.get("compliance_rate"),
        int(source.get("triggered") or 0),
        int(source.get("satisfied") or 0),
        int(source.get("deviated") or 0),
    )


def _vfield(v: Any, key: str, default: Any = "") -> Any:
    """读取判定结果的字段，兼容 Verdict 对象与其 dict 形态。

    曲线构建的输入有两个来源：内核实时产出的 :class:`Verdict` 对象，
    以及数据库快照重建出的 dict（``Verdict.to_dict()``）。两边字段相同
    但类型不同，硬用属性访问会在快照路径上炸掉。
    """
    if isinstance(v, dict):
        return v.get(key, default)
    return getattr(v, key, default)


def build_curve(
    analyses: Sequence[Dict[str, Any]],
    min_rate: float = 0.0,
    max_rate: float = 1.0,
) -> Dict[str, Any]:
    """由多局分析结果构建成长曲线。

    ``analyses`` 需按时间正序排列。缺少执行度的对局（没有任何规则被触发）
    会作为 ``None`` 点保留在曲线上 —— 空洞就是信息，不该被插值抹平。
    """
    points: List[CurvePoint] = []
    for a in analyses:
        rate, triggered, satisfied, deviated = _extract_counts(a)
        points.append(
            CurvePoint(
                match_id=a.get("match_id", ""),
                played_at=a.get("played_at", ""),
                game=a.get("game", ""),
                game_version=a.get("game_version", ""),
                compliance_rate=rate,
                triggered=triggered,
                satisfied=satisfied,
                deviated=deviated,
                outcome=a.get("outcome", {}) or {},
            )
        )

    valid = [p.compliance_rate for p in points if p.compliance_rate is not None]

    # 单规则维度的执行度
    per_rule: Dict[str, List[Optional[float]]] = {}
    rule_meta: Dict[str, Dict[str, str]] = {}
    for a in analyses:
        for v in a.get("verdicts", []):
            rid = _vfield(v, "rule_id")
            if not rid:
                continue
            per_rule.setdefault(rid, [])
            rule_meta.setdefault(
                rid,
                {
                    "rule_title": _vfield(v, "rule_title"),
                    "category": _vfield(v, "category"),
                },
            )
            # 只统计该局真正触发过的规则；未触发记为 None（保持时间轴对齐）
            status = _vfield(v, "status")
            per_rule[rid].append(
                1.0 if status == "satisfied" else 0.0
            )

    rules_compliance = [
        RuleCompliance(
            rule_id=rid,
            rule_title=meta["rule_title"],
            category=meta["category"],
            points=vals,
            checks=len([v for v in vals if v is not None]),
        ).to_dict()
        for rid, vals in per_rule.items()
        for meta in [rule_meta[rid]]
    ]
    # 只保留被检验过至少两次的规则，避免用单点噪声说「退步了」
    rules_compliance = [rc for rc in rules_compliance if rc["checks"] >= 2]
    rules_compliance.sort(key=lambda rc: (rc["latest_rate"] or 0.0, rc["rule_id"]))

    # 最需要改的：执行率最低且有足够样本的规则
    weakest = rules_compliance[:5]

    # 仍在退步的规则 —— 玩家最容易自我欺骗的地方
    regressing = [rc for rc in rules_compliance if rc["trend"] == "退步了"]

    return {
        "points": [p.to_dict() for p in points],
        "axis": {"min": min_rate, "max": max_rate},
        "summary": {
            "match_count": len(points),
            "measured_count": len(valid),
            "latest_rate": valid[-1] if valid else None,
            "first_rate": valid[0] if valid else None,
            "delta": (
                round(valid[-1] - valid[0], 4) if len(valid) >= 2 else None
            ),
        },
        "rules": rules_compliance,
        "weakest_rules": weakest,
        "regressing_rules": regressing,
    }


# --------------------------------------------------------------------------
# 版本适配
# --------------------------------------------------------------------------


def _rule_fingerprint(rule: Rule) -> Dict[str, Any]:
    """规则的可比内容，**不含版本号**。

    跨版本比较时 version 必然不同，把它算进差异会让规则库里**每一条**
    都显示为「已变更」，把真正的阈值调整淹没在噪声里 —— 这是个很容易
    漏掉的错误，因为本系统其他地方的 to_dict() 都需要带 version。
    """
    data = rule.to_dict()
    data.pop("version", None)
    return data


def diff_versions(
    old_rules: Sequence[Rule],
    new_rules: Sequence[Rule],
) -> Dict[str, Any]:
    """比较两个版本的规则库差异。"""
    old_by_id = {r.id: r for r in old_rules}
    new_by_id = {r.id: r for r in new_rules}

    changed: List[Dict[str, Any]] = []
    for rid in sorted(set(old_by_id) & set(new_by_id)):
        o, n = old_by_id[rid], new_by_id[rid]
        old_fp = _rule_fingerprint(o)
        new_fp = _rule_fingerprint(n)
        diffs: List[str] = []
        if o.title != n.title:
            diffs.append("标题变更")
        if o.status != n.status:
            diffs.append(f"状态 {o.status} → {n.status}")
        if o.severity != n.severity:
            diffs.append(f"严重度 {o.severity} → {n.severity}")
        if old_fp["trigger"] != new_fp["trigger"]:
            diffs.append("触发条件变更")
        if old_fp["expect"] != new_fp["expect"]:
            diffs.append("期望动作变更")
        if diffs:
            changed.append(
                {
                    "rule_id": rid,
                    "rule_title": n.title,
                    "changes": diffs,
                    "old_status": o.status,
                    "new_status": n.status,
                }
            )

    added = [
        {"rule_id": r.id, "rule_title": r.title, "category": r.category}
        for r in new_rules
        if r.id not in old_by_id
    ]
    removed = [
        {"rule_id": r.id, "rule_title": r.title, "category": r.category}
        for r in old_rules
        if r.id not in new_by_id
    ]

    return {
        "old_version": old_rules[0].version if old_rules else "",
        "new_version": new_rules[0].version if new_rules else "",
        "added": added,
        "removed": removed,
        "changed": changed,
        "deprecated": [
            c for c in changed
            if c["new_status"] == "deprecated"
        ],
    }


def backtest_version_change(
    old_rules: Sequence[Rule],
    new_rules: Sequence[Rule],
    contexts: Sequence[MatchContext],
    target_rule_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """版本切换回测：用玩家历史对局检验新版本规则是否仍成立。

    对每条发生变更的规则，把**同一批历史对局**分别用旧规则和新规则各判一次，
    比较两边的偏离率：

      - 新版明显更宽松 → 玩家原来的「老习惯」在新版本已经不亏，提示可以放松
      - 新版更严格       → 这是该版本下最该改的一条
      - 基本一致         → 保持

    ``contexts`` 是历史对局上下文，由调用方从持久化的原始数据重建
    （见 :meth:`MatchContext.from_dict`）。本函数刻意只接受上下文而不碰数据库，
    保持 core 层游戏无关且无副作用 —— 同一段回测逻辑对三款游戏通用。

    注意统计口径：**每一局里该规则触发的每一个回合各记一次**。
    只看首个触发回合会让「一局反复犯同一个错」被记成一次，
    而版本回测恰恰要衡量这个习惯有多顽固。
    """
    from .matcher import STATUS_DEVIATED, evaluate_round

    old_by_id = {r.id: r for r in old_rules}
    new_by_id = {r.id: r for r in new_rules}

    ids = target_rule_ids or sorted(set(old_by_id) & set(new_by_id))

    out: List[Dict[str, Any]] = []
    for rid in ids:
        o, n = old_by_id.get(rid), new_by_id.get(rid)
        if not o or not n:
            continue
        if _rule_fingerprint(o) == _rule_fingerprint(n):
            continue

        old_dev = new_dev = old_chk = new_chk = 0
        old_examples: List[Dict[str, str]] = []
        new_examples: List[Dict[str, str]] = []

        for ctx in contexts:
            old_hits = _count_rule_hits(o, ctx)
            new_hits = _count_rule_hits(n, ctx)

            old_chk += old_hits["checked"]
            old_dev += old_hits["deviated"]
            new_chk += new_hits["checked"]
            new_dev += new_hits["deviated"]

            if old_hits["deviated"] and len(old_examples) < 3:
                old_examples.append(old_hits["first_deviation"])
            if new_hits["deviated"] and len(new_examples) < 3:
                new_examples.append(new_hits["first_deviation"])

        old_rate = (old_dev / old_chk) if old_chk else None
        new_rate = (new_dev / new_chk) if new_chk else None

        verdict = "数据不足"
        if old_chk >= 2 and new_chk == 0:
            # 新版本下这条规则在玩家的所有历史局里一次都没触发。
            # 这是最强的「可以放松」信号，不是样本不足 —— 说明玩家
            # 原来那条老习惯在新版本里已经彻底不适用了。
            verdict = "可以放松"
        elif old_chk == 0 and new_chk >= 1:
            # 旧版本这条规则对玩家一次都没触发过，新版本却开始覆盖。
            # 这不是「无法对比」而是「新增了约束」：版本更新后这条规则
            # 开始管到你了，而你在所有历史局里都没达标。
            verdict = "新增约束"
        elif old_chk >= 2 and new_chk >= 2:
            if old_rate > 0.5 and new_rate <= 0.5:
                verdict = "可以放松"
            elif new_rate > old_rate + 0.05:
                verdict = "需要更严格"
            else:
                verdict = "基本一致"

        out.append(
            {
                "rule_id": rid,
                "rule_title": n.title,
                "category": n.category,
                "old_version": o.version,
                "new_version": n.version,
                "old_deviated": old_dev,
                "old_checked": old_chk,
                "old_rate": round(old_rate, 4) if old_rate is not None else None,
                "new_deviated": new_dev,
                "new_checked": new_chk,
                "new_rate": round(new_rate, 4) if new_rate is not None else None,
                "rate_delta": (
                    round(new_rate - old_rate, 4)
                    if (new_rate is not None and old_rate is not None)
                    else None
                ),
                "verdict": verdict,
                "message": _backtest_message(n, verdict, new_chk, new_rate),
                "old_examples": old_examples,
                "new_examples": new_examples,
            }
        )

    # 影响最大的排前面：偏离率上升最多的最该先看
    out.sort(key=lambda r: (r["rate_delta"] is None, -(r["rate_delta"] or 0.0)))
    return out


def _count_rule_hits(rule: Rule, ctx: "MatchContext") -> Dict[str, Any]:
    """统计一条规则在**一局**里触发多少次、偏离多少次。

    这是回测的原子操作：同一局内该规则触发的每个回合都独立计数。
    """
    from .matcher import STATUS_DEVIATED, evaluate_round

    checked = deviated = 0
    first_deviation: Dict[str, str] = {}

    for rnd in ctx.iter_rounds():
        verdict = evaluate_round(rule, ctx, rnd)
        if verdict is None:
            continue
        checked += 1
        if verdict.status == STATUS_DEVIATED:
            deviated += 1
            if not first_deviation:
                first_deviation = {
                    "match_id": ctx.match_id,
                    "round_label": verdict.round_label or "",
                    "expected": _render(verdict.expected),
                    "actual": _render(verdict.actual),
                }

    return {
        "checked": checked,
        "deviated": deviated,
        "first_deviation": first_deviation,
    }


def _render(spec: Dict[str, Any]) -> str:
    """把判定条件/动作渲染成简短可读串，用于回测证据。"""
    if not spec:
        return ""
    if "values" in spec:
        return "、".join(str(v) for v in spec["values"]) or "（无操作）"
    value = spec.get("value")
    if isinstance(value, (list, tuple)):
        return "、".join(str(v) for v in value)
    return str(value if value is not None else "")


def _backtest_message(
    rule: Rule, verdict: str, new_checked: int = 0, new_rate: Optional[float] = None
) -> str:
    """把回测结论翻译成玩家读得懂的一句话。"""
    if verdict == "可以放松":
        if new_checked == 0:
            return (
                f"「{rule.title}」在新版本已彻底不适用 —— "
                f"你的历史对局在新规则下一次都没触发它，这条老习惯可以放下了。"
            )
        return f"「{rule.title}」在新版本已不再是硬约束，按旧习惯打也不算错。"
    if verdict == "新增约束":
        pct = f"，偏离率 {new_rate:.0%}" if new_rate is not None else ""
        return (
            f"「{rule.title}」是本版本新增生效的规则 —— "
            f"你的历史对局在这 {new_checked} 个回合里一次都没达标{pct}。"
            f"这是升级后最该先看的一条。"
        )
    if verdict == "需要更严格":
        return f"「{rule.title}」在新版本更重要了，这是你当前最该改的一条。"
    if verdict == "基本一致":
        return f"「{rule.title}」新旧版本结论一致，继续保持。"
    return f"「{rule.title}」历史样本不足，暂不下结论。"
