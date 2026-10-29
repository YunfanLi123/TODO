"""规则 DSL：把攻略文本约束成机器可判定的结构化规则。

设计约束（来自立项设计）：
- 一条规则 = 触发条件（Trigger） + 期望动作（Expectation） + 溯源信息（Source）
- 触发条件必须只引用「事实（fact）」，事实是游戏无关的原子量，
  由各游戏适配器负责把原始对局数据翻译成事实。
- 主观判断（「这波该不该打」）无法用事实表达，因此无法通过准入校验。
  这是「可判定性筛选」在类型系统层面的强制实现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------
# 操作符
# --------------------------------------------------------------------------

#: 全部操作符。每个操作符都必须是「可判定」的 —— 给定事实取值即可得到确定的布尔结果。
OPERATORS = {
    "==",
    "!=",
    ">",
    ">=",
    "<",
    "<=",
    "in",
    "not_in",
    "between",
    "exists",
}

#: 需要数值型取值的操作符。
NUMERIC_OPS = {">", ">=", "<", "<=", "between"}

SEVERITIES = {"high", "medium", "low"}

RULE_STATUS = {"active", "deprecated"}


class RuleValidationError(ValueError):
    """规则不满足可判定性准入标准。"""


# --------------------------------------------------------------------------
# 条件
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    """单个原子判定条件。

    ``fact`` 必须是适配器注册过的事实名；``op`` 必须属于 :data:`OPERATORS`。
    """

    fact: str
    op: str
    value: Any = None

    def validate(self, known_facts: Optional[set] = None) -> None:
        if not self.fact or not isinstance(self.fact, str):
            raise RuleValidationError("fact 必须是非空字符串")
        if self.op not in OPERATORS:
            raise RuleValidationError(f"不支持的操作符 {self.op!r}，可用：{sorted(OPERATORS)}")
        if known_facts is not None and self.fact not in known_facts:
            raise RuleValidationError(
                f"事实 {self.fact!r} 未在适配器中注册，规则不可判定"
            )
        if self.op in {"in", "not_in"}:
            if not isinstance(self.value, (list, tuple, set)) or not self.value:
                raise RuleValidationError(f"{self.op} 需要非空集合作为取值")
        if self.op == "between":
            if (
                not isinstance(self.value, (list, tuple))
                or len(self.value) != 2
            ):
                raise RuleValidationError("between 需要 [下界, 上界] 两个值")
            if self.value[0] > self.value[1]:
                raise RuleValidationError("between 的下界不能大于上界")
        if self.op in NUMERIC_OPS:
            lo = self.value[0] if self.op == "between" else self.value
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in [lo]):
                raise RuleValidationError(f"{self.op} 需要数值型取值，实际得到 {self.value!r}")

    # -- 求值 ---------------------------------------------------------------

    def evaluate(self, actual: Any, present: bool = True) -> bool:
        """对一次事实取值求值。

        ``present=False`` 表示该事实在这局里根本没有被采集到，
        这类取值不参与判定（返回 False），并在归因阶段标记为「数据缺失」。
        """
        if self.op == "exists":
            return present
        if not present or actual is None:
            return False
        # ``unknown`` 是“没有读到”，不是一个可拿来通过 ``!=`` / ``not_in``
        # 条件的真实局面。否则“对位不处于劣势”会在对位根本未知时成立，
        # 随后把规则硬套到视频上；只有规则显式要求 unknown 时才允许命中。
        if actual == "unknown":
            if self.op == "==":
                return self.value == "unknown"
            if self.op == "in":
                return "unknown" in self.value
            return False

        try:
            if self.op == "==":
                return actual == self.value
            if self.op == "!=":
                return actual != self.value
            if self.op == ">":
                return actual > self.value
            if self.op == ">=":
                return actual >= self.value
            if self.op == "<":
                return actual < self.value
            if self.op == "<=":
                return actual <= self.value
            if self.op == "in":
                return actual in self.value
            if self.op == "not_in":
                return actual not in self.value
            if self.op == "between":
                lo, hi = self.value[0], self.value[1]
                return lo <= actual <= hi
        except TypeError:
            # 事实取值类型与阈值不符 —— 视为不成立，并保留原始数据供复查。
            return False
        raise RuleValidationError(f"不支持的操作符 {self.op!r}")


@dataclass(frozen=True)
class Trigger:
    """触发条件。

    ``all`` 全部成立、``any`` 任一成立、``none`` 全部不成立时触发。
    三者同时为空表示「每回合都触发」，用于全局性规则。
    """

    all: List[Condition] = field(default_factory=list)
    any: List[Condition] = field(default_factory=list)
    none: List[Condition] = field(default_factory=list)

    def validate(self, known_facts: Optional[set] = None) -> None:
        if not self.all and not self.any and not self.none:
            # 允许「无触发条件」，但要求规则显式声明 scope=global
            pass
        for group in (self.all, self.any, self.none):
            for cond in group:
                cond.validate(known_facts)

    def is_global(self) -> bool:
        return not (self.all or self.any or self.none)

    def referenced_facts(self) -> set:
        return {c.fact for c in (self.all + self.any + self.none)}

    def evaluate(self, resolve) -> "TriggerResult":
        """``resolve(fact) -> (value, present)``，由 MatchContext 提供。"""
        evidence: List[Dict[str, Any]] = []

        all_ok = True
        for cond in self.all:
            value, present = resolve(cond.fact)
            ok = cond.evaluate(value, present)
            all_ok = all_ok and ok
            evidence.append(
                {
                    "fact": cond.fact,
                    "op": cond.op,
                    "threshold": cond.value,
                    "actual": value,
                    "present": present,
                    "result": ok,
                    "group": "all",
                }
            )
            if not ok:
                # all 组一旦不成立即可短路，减少无谓求值
                return TriggerResult(False, evidence)

        # ``any`` 不是可选的装饰条件：只要规则写了它，就必须至少有
        # 一项成立。这里过去从 True 开始再做 ``or``，会让每一个含
        # any 的规则无条件触发，进而把不相干的打法套进报告。
        any_ok = False
        any_seen = False
        for cond in self.any:
            value, present = resolve(cond.fact)
            ok = cond.evaluate(value, present)
            any_seen = True
            any_ok = any_ok or ok
            evidence.append(
                {
                    "fact": cond.fact,
                    "op": cond.op,
                    "threshold": cond.value,
                    "actual": value,
                    "present": present,
                    "result": ok,
                    "group": "any",
                }
            )

        none_ok = True
        for cond in self.none:
            value, present = resolve(cond.fact)
            violated = cond.evaluate(value, present)
            none_ok = none_ok and (not violated)
            evidence.append(
                {
                    "fact": cond.fact,
                    "op": cond.op,
                    "threshold": cond.value,
                    "actual": value,
                    "present": present,
                    "result": not violated,
                    "group": "none",
                }
            )
            if not none_ok:
                return TriggerResult(False, evidence)

        fired = all_ok and none_ok and (any_ok if any_seen else True)
        return TriggerResult(fired, evidence)


@dataclass
class TriggerResult:
    fired: bool
    evidence: List[Dict[str, Any]]


# --------------------------------------------------------------------------
# 规则
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Subject:
    """适用时机 —— 这条规则在**哪些动作上**才适用。

    为什么需要它
    ------------
    攻略说的往往不是「这段时间每分钟都得做 X」，而是
    「**在做 A 还是做 B 的时候，该选 A**」：

        清完自己野区升 4 级 > 任何一次 gank
        控龙优先，而不是追人头

    这两条的前提都是「**那一刻玩家正在做某个目标类动作**」——
    在刷野 / 抓人 / 打龙这些动作里挑，而不是每一分钟都做一遍。

    没有它的时候，触发条件只剩时间窗，于是窗口内**每一帧都在判**：
    玩家 5 分钟里只有 1 帧打了龙，其余 4 帧在推线补兵，
    那 4 帧全被判成偏离 —— 玩家明明控了龙，却被说成「整段做反」。
    这是逐帧判定最系统的一类误判。

    ``values`` 为空表示「这一类动作做过就算适用」。
    """

    fact: str
    values: List[str] = field(default_factory=list)

    def matches(self, kind: str, value: Any) -> bool:
        if kind != self.fact:
            return False
        if not self.values:
            return True
        return value in self.values

    def validate(self, known_facts: Optional[set] = None) -> None:
        if not self.fact or not isinstance(self.fact, str):
            raise RuleValidationError("subject.fact 必须是非空字符串")
        if known_facts is not None and self.fact not in known_facts:
            raise RuleValidationError(
                f"适用时机引用了未注册事实 {self.fact!r}，规则不可判定"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {"fact": self.fact, "values": list(self.values)}

    @staticmethod
    def from_dict(data: Optional[Dict[str, Any]]) -> Optional["Subject"]:
        if not data:
            return None
        return Subject(
            fact=str(data.get("fact") or ""),
            values=[str(v) for v in (data.get("values") or [])],
        )


#: 满足口径 —— 判定「做到了」是按帧算，还是按整段算。
#:
#: ``per_frame``      每一个适用的时刻都该做期望动作（默认）
#: ``at_least_once``  这一段时间里至少做过一次就算整段做到了
EXPECT_SCOPE_PER_FRAME = "per_frame"
EXPECT_SCOPE_AT_LEAST_ONCE = "at_least_once"
EXPECT_SCOPES = {EXPECT_SCOPE_PER_FRAME, EXPECT_SCOPE_AT_LEAST_ONCE}


@dataclass(frozen=True)
class Source:
    """规则溯源。规则库只存结构化规则，但必须能指回原始攻略位置。"""

    guide: str
    locator: str = ""          # 章节 / 段落定位
    url: str = ""              # 公开数据源链接
    quoted: str = ""           # 原文引用
    collected_at: str = ""     # ISO 日期

    def to_dict(self) -> Dict[str, Any]:
        return {
            "guide": self.guide,
            "locator": self.locator,
            "url": self.url,
            "quoted": self.quoted,
            "collected_at": self.collected_at,
        }


@dataclass
class Rule:
    """一条可判定的攻略规则。

    ``id`` 形如 ``TFT-ECON-001``，在 ``(game, version)`` 内唯一且稳定。
    """

    id: str
    game: str
    version: str
    title: str
    category: str
    trigger: Trigger
    expect_fact: str
    expect_op: str
    expect_value: Any
    #: 适用时机（可选）。不填就是老行为：触发条件成立就判。
    #: 填了就只在「玩家做了这些动作」的回合上判 —— 见 :class:`Subject`。
    subject: Optional[Subject] = None
    #: 满足口径。见 :data:`EXPECT_SCOPES`。
    expect_scope: str = EXPECT_SCOPE_PER_FRAME
    severity: str = "medium"
    status: str = "active"
    source: Optional[Source] = None
    rationale: str = ""        # 为什么这样判定（中文，写进报告）
    deviation_hint: str = ""   # 偏离时给玩家的具体建议
    #: 攻略原文里的前提、动作、后果，三段分开存。
    #:
    #: 为什么单列三个字段而不是塞进 ``rationale``：
    #: ``rationale`` 是**引文**（规则库页要写「原文前提：…；原文后果：…」），
    #: 而报告要的是**判断**——结合这一局的实际前提，说明为什么该那么做。
    #: 混在一处的话，报告要么照抄引文（读者看不到关于这局的任何信息），
    #: 要么就得从引文里反向抠句子。
    premise: str = ""          # 前提（原文）
    action: str = ""           # 动作（原文）
    outcome: str = ""          # 后果（原文）
    tags: List[str] = field(default_factory=list)
    #: **这一条能不能用来纠正玩家**。
    #:
    #: 什么时候不能
    #: ------------
    #: 攻略的分支前提往往读不出来（「塔的早期护盾已消失」「对方减员」
    #: 「暴君倒计时进入最后一段」「谁先到龙坑」）。而**取舍**恰恰由这些
    #: 前提决定：原文说「塔的护盾没了、敌方打野不在我方半区 → 带兵线推塔」，
    #: 前提不成立时玩家**退塔下才是对的**。照着规则库的字面去纠正他，
    #: 就是在说「你应该推塔」—— 可当时根本不具备推塔条件。
    #:
    #: 什么时候能
    #: ------------
    #: 这一条不取决于读不出来的前提：
    #: - 触发条件里已经有**画面读得出**的事实（大招就绪、是否在团战）；或者
    #: - 同一个时段里攻略给的几套打法**都是同一类**（都在野区做事），
    #:   前提只决定「哪种打法」，不决定「该不该做这类事」。
    #:
    #: 不能纠正的规则**仍然在模式库里**（攻略内容照常可查），
    #: 只是不会拿它去指责玩家。
    corrective: bool = True
    #: **现有识别能力下判不出来**。
    #:
    #: 有些模式的动作本身要靠画面去探草/放眼（``ward_vision``），而识别链路
    #: 几乎不产出这个动作 —— 期望的动作永远没出现，规则每局都挂在
    #: 「判不了」上，报告看起来像出了问题，实际是能力边界。
    #:
    #: 标了它的规则**不参与判定**，也不计入判不了统计；攻略内容照常在
    #: 模式库里可查。识别能力补上之后把标记去掉即可。
    unjudgeable: bool = False
    #: **同一时段（节点）里攻略认可的全部打法**。
    #:
    #: 为什么需要它
    #: ------------
    #: 攻略在一个时段里往往给**好几套**打法，各自有前提：
    #:
    #:     4:00  塔的护盾没了 + 敌方打野不在我方半区 → 带兵线推塔
    #:     4:00  暴君倒计时进最后一段 + 龙坑有我方视野 → 集合占位龙坑
    #:
    #: 这些前提（塔的护盾、倒计时、视野）**画面里读不出来**。
    #: 于是系统只知道「玩家这会儿在刷野」——而刷野既不是「推塔」也不是
    #: 「占龙坑」，照着规则库的字面判就是一条**冤枉**：
    #: 玩家在龙还没刷的时候刷野，被说「该抢龙」。
    #:
    #: 所以每个规则都带着**这个节点全部合法打法**：
    #: 玩家做了其中任何一件，说明他这会儿在按攻略做事，
    #: **就不纠正他** —— 他做得没错，只是跟规则库写的那一条不一样。
    #:
    #: 只有当玩家**整段一次都没做过**节点里的任何一件事时，
    #: 才有话说：「这段你一件事都没做」—— 那才是改了一定有进步的纠正。
    node_actions: List[str] = field(default_factory=list)
    #: **分支**（原文一支一条）。
    #:
    #: ``premise`` / ``action`` 是把同一节点里的好几支用「；」拼起来的一句
    #: 大菜单 —— 玩家读到的是一张菜单（「敌方红开就己方蓝开，敌方蓝开就
    #: 己方红开」），读不出**这一局适用哪一支**。于是「我做的正是菜单里
    #: 那一支，却说我做错了」—— 报告从此不敢信。
    #:
    #: 分支拆开之后，模型只要指认第几支成立，报告就能给出**那一支**的
    #: 具体做法，而不是整张菜单。每一项是
    #: ``{"premise": ..., "action": ..., "outcome": ...}``。
    branches: List[Dict[str, Any]] = field(default_factory=list)

    # -- 校验 ---------------------------------------------------------------

    def validate(self, known_facts: Optional[set] = None) -> None:
        if not self.id or "-" not in self.id:
            raise RuleValidationError("规则 id 必须形如 PREFIX-CATEGORY-NNN")
        if not self.game:
            raise RuleValidationError("规则必须绑定游戏")
        if not self.version:
            raise RuleValidationError("规则必须绑定游戏版本")
        if not self.title:
            raise RuleValidationError("规则必须有标题")
        if self.severity not in SEVERITIES:
            raise RuleValidationError(f"severity 必须是 {sorted(SEVERITIES)} 之一")
        if self.status not in RULE_STATUS:
            raise RuleValidationError(f"status 必须是 {sorted(RULE_STATUS)} 之一")
        if self.expect_op not in OPERATORS:
            raise RuleValidationError(f"expect_op 不支持 {self.expect_op!r}")
        if self.expect_scope not in EXPECT_SCOPES:
            raise RuleValidationError(
                f"expect_scope 必须是 {sorted(EXPECT_SCOPES)} 之一，"
                f"实际得到 {self.expect_scope!r}"
            )
        if self.subject is not None:
            self.subject.validate(known_facts)
        if known_facts is not None:
            if not self.trigger.is_global():
                self.trigger.validate(known_facts)
            if self.expect_fact not in known_facts:
                raise RuleValidationError(
                    f"期望动作引用了未注册事实 {self.expect_fact!r}，规则不可判定"
                )
        Condition(self.expect_fact, self.expect_op, self.expect_value).validate(
            known_facts if known_facts is not None else None
        )

    def expect_condition(self) -> Condition:
        return Condition(self.expect_fact, self.expect_op, self.expect_value)

    def referenced_facts(self) -> set:
        facts = set(self.trigger.referenced_facts())
        facts.add(self.expect_fact)
        return facts

    # -- 序列化 -------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        def cond_list(items):
            return [
                {"fact": c.fact, "op": c.op, "value": c.value}
                for c in items
            ]

        return {
            "id": self.id,
            "game": self.game,
            "version": self.version,
            "title": self.title,
            "category": self.category,
            "trigger": {
                "all": cond_list(self.trigger.all),
                "any": cond_list(self.trigger.any),
                "none": cond_list(self.trigger.none),
            },
            "expect": {
                "fact": self.expect_fact,
                "op": self.expect_op,
                "value": self.expect_value,
            },
            "subject": self.subject.to_dict() if self.subject else None,
            "expect_scope": self.expect_scope,
            "severity": self.severity,
            "status": self.status,
            "rationale": self.rationale,
            "deviation_hint": self.deviation_hint,
            "premise": self.premise,
            "action": self.action,
            "outcome": self.outcome,
            "tags": list(self.tags),
            "node_actions": list(self.node_actions),
            "corrective": self.corrective,
            # 只有标了的规则才写这个键 —— 种子文件里绝大多数规则没有它，
            # 无条件写 False 会让序列化往返对不上（与 ``branches`` 同理）。
            **({"unjudgeable": True} if self.unjudgeable else {}),
            # 分支**为空时不写这个键** —— 英雄层的种子没有分支，
            # 写了就会与种子文件对不上（序列化往返要无损）。
            **({"branches": [dict(b) for b in self.branches]}
               if self.branches else {}),
            "source": (
                {
                    "guide": self.source.guide,
                    "locator": self.source.locator,
                    "url": self.source.url,
                    "quoted": self.source.quoted,
                    "collected_at": self.source.collected_at,
                }
                if self.source
                else None
            ),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Rule":
        trg = data.get("trigger", {}) or {}

        def mk(items):
            return [
                Condition(c["fact"], c["op"], c.get("value"))
                for c in (items or [])
            ]

        exp = data.get("expect", {}) or {}
        src = data.get("source")
        return Rule(
            id=data["id"],
            game=data["game"],
            version=data["version"],
            title=data["title"],
            category=data.get("category", "uncategorized"),
            trigger=Trigger(
                all=mk(trg.get("all")),
                any=mk(trg.get("any")),
                none=mk(trg.get("none")),
            ),
            expect_fact=exp["fact"],
            expect_op=exp["op"],
            expect_value=exp.get("value"),
            subject=Subject.from_dict(data.get("subject")),
            expect_scope=data.get("expect_scope") or EXPECT_SCOPE_PER_FRAME,
            severity=data.get("severity", "medium"),
            status=data.get("status", "active"),
            rationale=data.get("rationale", ""),
            deviation_hint=data.get("deviation_hint", ""),
            premise=data.get("premise", ""),
            action=data.get("action", ""),
            outcome=data.get("outcome", ""),
            tags=list(data.get("tags", []) or []),
            node_actions=list(data.get("node_actions", []) or []),
            branches=[dict(b) for b in (data.get("branches") or [])
                      if isinstance(b, dict)],
            corrective=bool(data.get("corrective", True)),
            unjudgeable=bool(data.get("unjudgeable", False)),
            source=Source(**src) if src else None,
        )


# --------------------------------------------------------------------------
# 准入筛选
# --------------------------------------------------------------------------


def check_admission(
    rule: Rule,
    known_facts: set,
    subjective_markers: Optional[List[str]] = None,
    action_facts: Optional[set] = None,
) -> List[str]:
    """可判定性准入检查。返回拒绝原因列表，空列表代表通过。

    这是立项设计中「规则宁可少也不能注水」的强制执行点。

    ``action_facts`` 是该游戏里「能承载玩家实际操作」的事实集合。
    传入后，expect 字段必须挂在其中之一 —— 否则这条规则会在每一局每一回合
    都判成「该做而没做」，产生恒为偏离的假信号。见 GameAdapter 上的说明。
    """
    reasons: List[str] = []

    if rule.trigger.is_global() and not rule.expect_fact:
        reasons.append("规则既无触发条件也无判定动作，无法判定")

    try:
        rule.validate(known_facts)
    except RuleValidationError as exc:
        reasons.append(str(exc))

    if rule.trigger.is_global():
        reasons.append("缺少触发条件：全局规则无法定位到具体回合")

    if action_facts is not None and rule.expect_fact not in action_facts:
        reasons.append(
            f"期望动作 {rule.expect_fact!r} 不是可记录玩家操作的事实"
            f"（可用：{sorted(action_facts)}），该规则会恒判为偏离"
        )

    # 主观判断词表 —— 这些词说明规则描述的是判断而非可观测事实。
    # 词表是强制防线，不能因为调用方没传就失效。
    markers = (
        DEFAULT_SUBJECTIVE_MARKERS
        if subjective_markers is None
        else subjective_markers
    )
    blob = " ".join(
        [rule.title, rule.rationale, rule.deviation_hint, rule.category]
    )
    for marker in markers:
        if marker in blob:
            reasons.append(f"疑似主观判断（含「{marker}」），不具备可判定性")
            break

    if not rule.source or not (rule.source.guide or rule.source.url):
        reasons.append("缺少溯源信息：规则必须能指回原始攻略")

    # 去重并保序
    seen = set()
    unique = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return unique


#: 默认主观判断词表（中文），用于自动拒收攻略里的定性描述。
DEFAULT_SUBJECTIVE_MARKERS = [
    "该不该",
    "看情况",
    "尽量",
    "酌情",
    "灵活",
    "感觉",
    "我觉得",
    "一般情况下",
    "尽量避免",
    "看局势",
]
