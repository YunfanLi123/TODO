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
    #: 「判不了」的原因码，报告据此告诉玩家缺的是**哪一类**信息。
    #:
    #: 形如 ``premise:ward_vision``（前提要读视野）、``confidence``
    #: （动作识别置信度不足）、``missing_fact:positioning_action``。
    #: 早先报告只说「缺少站位」，可站位这一类本来是读得出来的 ——
    #: 真正缺的是「草丛里有没有人」，玩家照着补也补不到点上。
    undetermined_reason: str = ""
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
            "undetermined_reason": self.undetermined_reason,
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

#: 动作识别的最低置信度。低于它当作「没读到」，不当作证据。
#:
#: 为什么需要这条：模型在一帧里会同时给出好几个候选动作，0.49~0.57 的那些
#: 本质是在摇摆（实测「刷野补经济 0.504」被拿去判了一次偏离）。
#: 拿硬币当证据会凭空造出「需调整」，而报告的可信度全押在这上面。
#:
#: 0.55 而不是 0.6：0.6 会连 0.574 的「开团先手」一起否决，把一条已经
#: 读到的正向结论压成「判不了」—— 证据在，只是弱一点。0.5 以下才是纯噪音。
MIN_ACTION_CONFIDENCE = 0.55

#: 前提里出现这些词，说明规则要验证一个**画面读不出**的状态。
#:
#: 例：「对方中单进入我方塔前」（要读对手位置）、「河道草丛有敌方头像」
#: （要读视野）、「敌方先打出控制并命中己方前排」（要读控制）、
#: 「我方一人未复活」（要读复活计时）。这些在识别结果里要么永远是
#: ``unknown``，要么根本不存在。
#:
#: 命中之后**只把「偏离」降级成「判不了」**，不动「做对」：
#: 前提没证实却判人家「做错了」，是最容易被玩家当场推翻的那类结论。
_PREMISE_UNREADABLE = (
    ("复活", "respawn"),
    ("控制", "cc"),
    ("被控", "cc"),
    ("草丛", "ward_vision"),
    ("视野", "ward_vision"),
)

_UNREADABLE_VALUES = (None, "", "unknown")

#: 前提里**能被画面直接否证**的说法。
#:
#: 这一类比「读不出来」更进一步：画面明确给出了相反的事实。
#: 例：前提写「团战结束」，而这一帧 ``in_combat`` 是 True —— 前提根本不成立，
#: 规则不该在这一帧适用。继续判只会得到一条凭空生出的「偏离」或「判不了」，
#: 两个都是噪音。
#:
#: 只在事实**读得到**（布尔值）时才校验；读不到就交给「读不出来」那条路。
_PREMISE_CONTRADICTS = (
    ("团战结束", "in_combat", False),
    ("战斗结束", "in_combat", False),
    ("大招可用", "has_ult_ready", True),
    ("技能保留", "has_ult_ready", True),
)


def _has_laning_window(rule: Rule) -> bool:
    """这条规则是否要求「对线期（从 0 分钟起）」的做法。

    看图说话：窗口起点是 0 分钟的规则，讲的是开局对线。
    """
    for cond in rule.trigger.all:
        if cond.fact != "time_min" or cond.op != "between":
            continue
        value = cond.value
        if isinstance(value, (list, tuple)) and value and value[0] == 0:
            return True
    return False


def _premise_contradicted(rule: Rule, rnd: RoundRecord) -> bool:
    """前提被这一帧的画面直接否证时返回 True。"""
    text = rule.premise or ""
    for word, fact, expected in _PREMISE_CONTRADICTS:
        if word not in text:
            continue
        value = rnd.facts.get(fact)
        if isinstance(value, bool) and value is not expected:
            return True
    return False


def _premise_unreadable(rule: Rule, rnd: RoundRecord) -> Optional[str]:
    """规则的前提在这一帧的画面里读不出来时，返回读不出的那个事实。

    只对**偏离**做降级：前提没被证实，就不能说玩家「做错了」；
    但已经做对的仍然算做对（前提读不出来不等于做得不对）。

    **合并过的规则不降级**。同一节点里各分支的前提是「或」关系
    （「龙坑有我方视野」/「龙坑无我方视野但视野精灵在附近」），合并后的
    前提文本里自然会出现「视野」这种词 —— 但规则本来就不靠单一分支的
    前提成立，读不出其中一支并不影响判定，降级反而会抹掉真实的失误。
    """
    if len(rule.branches) > 1:
        return None

    text = rule.premise or ""
    for word, fact in _PREMISE_UNREADABLE:
        if word not in text:
            continue
        if fact == "ward_vision":
            # 视野是动作类事实：要么这一帧标了「做视野」，要么读了事实值。
            if any(a.value == "ward_vision" for a in rnd.actions):
                continue
            if rnd.facts.get("ward_vision") not in _UNREADABLE_VALUES:
                continue
            return fact
        if fact in rnd.missing_facts:
            return fact
        if rnd.facts.get(fact) in _UNREADABLE_VALUES:
            return fact
    return None


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

    # 识别能力边界内的规则不判（见 :attr:`app.core.dsl.Rule.unjudgeable`）。
    # 它们期望的动作这一套识别链路产不出来，硬判只会每局挂一条「判不了」。
    if rule.unjudgeable:
        return None

    resolve = ctx.resolve_in_round(rnd)
    trig = rule.trigger.evaluate(resolve)

    if not trig.fired:
        return None

    # 前提被画面否证 —— 这一帧规则根本不适用。
    if _premise_contradicted(rule, rnd):
        return None

    # 0:00 那一帧不套对线期的规则。
    #
    # 开局帧玩家还在泉水/刚回城（画面里标的是「回城补给」，没有补刀），
    # 兵线根本还没到。拿「对线期该补兵/该压线」的规则去判这一帧，
    # 得到的必然是「没做到」—— 这是结构性的误判，不是玩家的问题。
    if (
        rnd.time_index <= 0
        and _has_laning_window(rule)
        and any(a.value == "recall" for a in rnd.actions)
        and not any(a.value == "last_hit" for a in rnd.actions)
    ):
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
        return Verdict(
            status=STATUS_UNDETERMINED,
            actual={},
            undetermined_reason=f"missing_fact:{exp_cond.fact}",
            **base,
        )

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

    def conclude(status: str, reason: str = "") -> Verdict:
        """收尾：前提读不出来的「偏离」降级为「判不了」。"""
        if status == STATUS_DEVIATED:
            unreadable = _premise_unreadable(rule, rnd)
            if unreadable:
                status = STATUS_UNDETERMINED
                reason = f"premise:{unreadable}"
        return Verdict(
            status=status,
            actual=actual,
            undetermined_reason=reason if status == STATUS_UNDETERMINED else "",
            **base,
        )

    # 置信度不够的动作不算证据。
    #
    # 模型在同一帧里经常同时给出几个候选动作，0.5 上下那些本质是在摇摆
    # （实测「刷野补经济 0.504」被拿去判过一次偏离）。拿摇摆当证据，
    # 报告就会凭空多出一条「需调整」，而它的可信度全押在这上面。
    confident = [a for a in performed if a.confidence >= MIN_ACTION_CONFIDENCE]
    if performed and not confident:
        return conclude(STATUS_UNDETERMINED, "confidence")

    if not confident:
        # 这一帧完全没做这一类动作 —— 等价于偏离（该做而没做）。
        # 但前提若读不出来，就只是「判不了」。
        return conclude(STATUS_DEVIATED)

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
            a for a in confident
            if a.value in rule.node_actions and a.value not in expect_values
        ]
        if alternatives and not any(
            exp_cond.evaluate(a.value, present=True) for a in confident
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
    ok = any(exp_cond.evaluate(a.value, present=True) for a in confident)
    return conclude(STATUS_SATISFIED if ok else STATUS_DEVIATED)


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
