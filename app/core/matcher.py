"""规则匹配：把规则库套到一局对局的每个回合上。

这一层是「逐回合对账」的执行者，也是全系统最需要可复查的部分 ——
每一条判定都保留触发证据（阈值 / 实际取值 / 每个子条件的结果），
让报告里的任何一句话都能顺着这些证据复查回原始数据。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .context import ActionRecord, MatchContext, RoundRecord
from .dsl import Rule


@dataclass
class Verdict:
    """一条规则在一局对局上的判定结果。

    生命周期：
      not_triggered  触发条件未成立 —— 这局没有踩这个坑
      satisfied      触发且实际动作符合规范
      deviated       触发且实际动作偏离规范 —— 这是要写进报告的差距
      undetermined   触发条件成立，但关键事实缺失，无法判定
    """

    rule_id: str
    rule_title: str
    category: str
    severity: str
    status: str                       # 上面四种之一
    round_label: Optional[str] = None
    #: 回合的数值坐标。排序用它们，**不**解析 round_label ——
    #: 标签是给人看的（可能是「第 12 分钟」这种非数字形式），
    #: 用字符串解析排序既脆弱又会在非数字标签上静默降级。
    round_stage: int = 0
    round_phase: int = 0
    #: 同一时间片内的连续时间（分钟）。默认 0，自走棋无需设置。
    round_time: float = 0.0
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    expected: Dict[str, Any] = field(default_factory=dict)
    actual: Dict[str, Any] = field(default_factory=dict)
    #: 这一帧的局面快照（不依赖触发条件）。
    #:
    #: 触发条件里只有规则**用到的**那几个事实，而报告要回答的是
    #: 「当时到底什么局面」——「第 4 分钟你落后 320 经济，还在刷野」
    #: 这句话里的 320 就不是任何一条规则的触发条件。
    #: 缺了它，玩家拿到一条偏离只能自己去翻原始数据。
    context: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    deviation_hint: str = ""
    #: 规则原文的前提 / 动作 / 后果。报告写「为什么」时要用，
    #: 不能拿 ``rationale`` 顶替 —— 那是引文，不是判断。
    premise: str = ""
    action: str = ""
    outcome: str = ""
    source: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)

    @property
    def is_deviation(self) -> bool:
        return self.status == "deviated"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_title": self.rule_title,
            "category": self.category,
            "severity": self.severity,
            "status": self.status,
            "round_label": self.round_label,
            "round_stage": self.round_stage,
            "round_phase": self.round_phase,
            "evidence": self.evidence,
            "expected": self.expected,
            "actual": self.actual,
            "context": self.context,
            "rationale": self.rationale,
            "deviation_hint": self.deviation_hint,
            "premise": self.premise,
            "action": self.action,
            "outcome": self.outcome,
            "source": self.source,
            "tags": self.tags,
        }


#: 判定结论的四种取值。
STATUS_NOT_TRIGGERED = "not_triggered"
STATUS_SATISFIED = "satisfied"
STATUS_DEVIATED = "deviated"
STATUS_UNDETERMINED = "undetermined"


def _describe_expected(rule: Rule) -> Dict[str, Any]:
    return {
        "fact": rule.expect_fact,
        "op": rule.expect_op,
        "value": rule.expect_value,
    }


def _round_context(rnd: RoundRecord) -> Dict[str, Any]:
    """这一帧的局面快照 —— 标量事实，不含动作事实。

    动作事实由适配器镜像进来（一次动作一个值），放进快照会让人误以为
    「当时只做了这一件事」—— 实际那一帧可能同时做了好几件。
    """
    action_kinds = {a.kind for a in rnd.actions}
    return {
        k: v for k, v in rnd.facts.items()
        if k not in action_kinds and v is not None and k not in rnd.missing_facts
    }


def evaluate_round(
    rule: Rule,
    ctx: MatchContext,
    rnd: RoundRecord,
) -> Optional[Verdict]:
    """在单个回合上判定一条规则。触发条件不成立时返回 ``None``。"""
    # 注意：**这里不按 ``rule.corrective`` 过滤**。
    #
    # ``corrective`` 说的是「这一条能不能拿来纠正玩家」（前提读不出来的
    # 那些不纠正），那是**报告层**的决定 —— 分析层照实判，报告层只呈现
    # 能拿来改的。装在判定层的后果是：整个引擎被掐哑，报告空得连
    # 「哪里做错了」都说不出来，而模式库页与统计也跟着失真。

    resolve = ctx.resolve_in_round(rnd)
    trig = rule.trigger.evaluate(resolve)

    if not trig.fired:
        return None

    # 适用时机：这一帧玩家根本没有参与这类动作。
    #
    # 「不适用」和「没做到」是两件事 ——
    # 攻略说的是「在做 A 还是做 B 的时候该选 A」，
    # 玩家这一帧在推线补兵，跟这条规则没关系，
    # 判成偏离就是冤枉人。返回 None，让它不进这一帧的分母。
    #
    # **只有这一帧真的标了动作才做这个过滤**。一帧什么都没标有两种可能：
    # 「玩家这一帧确实什么都没做」（该判偏离）和「玩家没标 / 看不清」
    # （该判无法判定）。两者在动作列表上都长一样 —— 都是空的 ——
    # 所以不能在这里替玩家做判断，交给下面的「动作事实缺失」分支去分。
    if (
        rule.subject is not None
        and rnd.actions
        and not any(
            rule.subject.matches(a.kind, a.value) for a in rnd.actions
        )
    ):
        return None

    base = dict(
        rule_id=rule.id,
        rule_title=rule.title,
        category=rule.category,
        severity=rule.severity,
        round_label=rnd.label,
        round_stage=rnd.stage,
        round_phase=rnd.phase,
        round_time=rnd.time_index,
        evidence=trig.evidence,
        expected=_describe_expected(rule),
        context=_round_context(rnd),
        rationale=rule.rationale,
        deviation_hint=rule.deviation_hint,
        premise=rule.premise,
        action=rule.action,
        outcome=rule.outcome,
        source=rule.source.to_dict() if rule.source else None,
        tags=list(rule.tags),
    )

    # 触发条件成立，但判定所需的动作事实缺失 —— 不猜，显式标为无法判定
    exp_cond = rule.expect_condition()
    if exp_cond.fact in rnd.missing_facts:
        return Verdict(status=STATUS_UNDETERMINED, actual={}, **base)

    performed: List[ActionRecord] = [
        a for a in rnd.actions if a.kind == exp_cond.fact
    ]
    actual = {
        "fact": exp_cond.fact,
        "actions": [a.to_dict() for a in performed],
        "values": [a.value for a in performed],
        # 该回合做的**所有**事，不只是期望那一类。
        #
        # 报告的「你实际在做啥」要靠它。只看期望那一类的话，玩家做了
        # 别的事（去抓人而不是刷野）时，报告会写成「（未标）」——
        # 明明他做了很多事，报告却只说他「什么都没做」，
        # 于是「怎么改」那一条给不出方向。
        "all_actions": [a.to_dict() for a in rnd.actions],
    }

    if not performed:
        # 该回合完全没做这一类动作 —— 等价于偏离（该做而没做）
        return Verdict(
            status=STATUS_DEVIATED,
            actual=actual,
            **base,
        )

    # **同一时段里的其它合法打法**（见 :attr:`app.core.dsl.Rule.node_actions`）。
    #
    # 攻略在一个时段给好几套打法，各自的前提画面里读不出来。
    # 玩家做了其中**另一件**时，他这会儿是在按攻略做事，
    # 只是跟规则库写的那一条不一样 —— **不纠正他**。
    #
    # 这一条是「零人工」能不能不冤枉人的关键：没有它，
    # 「4:00 该抢龙」会判掉 4:00 正在刷野的玩家（龙还没刷呢）。
    if rule.node_actions:
        expect_values = set(exp_cond.value or [])
        alternatives = [
            a for a in performed
            if a.value in rule.node_actions and a.value not in expect_values
        ]
        if alternatives and not any(
            exp_cond.evaluate(a.value, present=True) for a in performed
        ):
            # 他做了这个时段攻略认可的另一件事 —— 这条规则不适用。
            return None

    # 判定「这一帧有没有做该做的事」：**任一**同类动作符合就算符合。
    #
    # 为什么不是只看最后一个动作
    # --------------------------
    # MOBA 的一帧里本来就会有多件同类的事 —— 「刷野 + 清视野精灵」正是
    # 攻略让打野前期做的两件事（见 HOK-MODE-JGL-E）。只看最后一个动作时，
    # 玩家在第 2 分钟勾了「刷野补经济 + 清视野精灵」，而赵云那条
    # （HOK-HERO-ZYC-M，期望「刷野补经济」）拿到的是最后一个动作
    # 「清视野精灵」，于是判成偏离 —— 玩家明明刷了野，却被告知没做到。
    #
    # 早先「同一回合做不了两件互斥的事」这个前提是自走棋留下的：
    # 那里一个回合只有一个经济动作。MOBA 不成立。
    #
    # 触发条件那一侧仍然只看镜像出来的最后一个动作（适配器把动作镜像进
    # facts 就是给 trigger 用的）。所以库里**不能**在同一个格子里
    # 既用动作事实写 none 护栏、又有另一条模式期望同一个动作事实 ——
    # 那样会出现「这一帧既算做到了、又触发了互斥的那条」。
    # tests/test_mode_library.py 里的检查盯着这一条。
    ok = any(exp_cond.evaluate(a.value, present=True) for a in performed)
    return Verdict(
        status=STATUS_SATISFIED if ok else STATUS_DEVIATED,
        actual=actual,
        **base,
    )


def match_rules(
    rules: List[Rule],
    ctx: MatchContext,
    only_rules: Optional[List[str]] = None,
) -> List[Verdict]:
    """对整局做规则匹配，返回所有触发过的判定结果。

    结果按「回合先后 → 严重程度」排序，让报告天然按对局时间线展开。
    """
    wanted = set(only_rules) if only_rules else None
    out: List[Verdict] = []

    for rnd in ctx.iter_rounds():
        for rule in rules:
            if rule.status != "active":
                continue
            if wanted is not None and rule.id not in wanted:
                continue
            verdict = evaluate_round(rule, ctx, rnd)
            if verdict is not None:
                out.append(verdict)

    sev_rank = {"high": 0, "medium": 1, "low": 2}

    def sort_key(v: Verdict):
        # 用数值坐标排序，不解析展示标签 —— 标签可能是「第 12 分钟」。
        # round_time 放在严重度之前：同一时间片内必须按真实先后排列，
        # 否则报告时间线会被严重度打散，读者看到的不是一局棋的推进过程。
        return (
            (v.round_stage, v.round_phase, v.round_time),
            sev_rank.get(v.severity, 3),
            v.rule_id,
        )

    out.sort(key=sort_key)
    return out
