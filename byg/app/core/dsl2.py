"""规则 DSL v2：把攻略的「一个时点该做什么」表达成可判定、可审计的结构。

为什么要有 v2
------------
v1（:mod:`app.core.dsl`）把一条规则压成「触发条件 + 一个期望动作」，
用它承载 MOBA 攻略时有三个结构性问题，改不掉就没法把 84 条模式迁过来：

1. **时点是散的**。时间写在触发条件里（``time_min between [4,4]``），
   于是「这个节点是 4 分钟」这件事既在触发条件里、又在标题里、
   又在分类里，改一次要改三处，还容易改漏。
   v2 把时点提成第一等的 :class:`Node`，触发条件里不再出现时间。

2. **期望只有一个值**。「刷野 ≥4 次」「开局 90 秒内到 4 级」「先清精灵再推线」
   这些攻略里最具体的话，全都退化成一个 ``expect: {fact, op, value}``。
   v1 的 ``expect_scope``（``at_least_once`` / ``per_frame``）**在匹配器里
   根本没被实现** —— 字段是摆设，语义全是空的。
   v2 把期望改成 :class:`Check` 列表，每项带计数区间、时间窗、权重。

3. **「能不能纠正玩家」是静态布尔**。v1 里 84 条有 75 条 ``corrective=False``，
   理由是「攻略的分支前提读不出来」（敌方打野红不红、龙坑有没有视野）。
   但这个判断是**入库时定死的**：后来有人把「敌方开野方向」读出来了，
   那条规则还是纠正不了。
   v2 把它改成 :class:`Corrective`，判定**推迟到运行时**：
   分支前提一旦被指认（模型指认了第几支，或用户填了），
   规则就自动变成可纠正的。

贯穿全模块的一条纪律
--------------------
**识别 → 匹配必须确定性、可复现、可审计。**
LLM 只负责「指认哪一支前提成立」和「给解释」，不负责「判玩家做没做」。
所以这里的每一个判定都是：给定事实取值 → 得到确定的布尔与证据。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# 证据等级
# --------------------------------------------------------------------------

#: 画面直接读出来的（模型看帧、探针量到的）。
PROV_OBSERVED = "observed"
#: 由别的别的**事实**推出来的实推出来的（分路由英雄推、阶段由时间推）。
PROV_INFERRED = "inferred"
#: 用户手填的（对局名里填的分路、英雄、对位优劣）。
PROV_DECLARED = "declared"
#: 不限来源。
PROV_ANY = "any"

PROVENANCE_CLASSES = {PROV_OBSERVED, PROV_INFERRED, PROV_DECLARED, PROV_ANY}

#: 各证据等级的中文名，报告与诊断要用。
PROVENANCE_LABELS: Dict[str, str] = {
    PROV_OBSERVED: "画面读到",
    PROV_INFERRED: "推断",
    PROV_DECLARED: "你填的",
    PROV_ANY: "不限来源",
}

# --------------------------------------------------------------------------
# 操作符与口径
# --------------------------------------------------------------------------

OPERATORS = {
    "==", "!=", ">", ">=", "<", "<=", "in", "not_in", "between", "exists",
}

NUMERIC_OPS = {">", ">=", "<", "<=", "between"}

SEVERITIES = {"high", "medium", "low"}
RULE_STATUS = {"active", "deprecated"}

#: 满足口径 —— 「做到了」按什么算。
#:
#: ``at_least_once``  节点窗口内至少做过一次就算整段做到了（默认）
#: ``per_frame``       节点窗口内**每一个**适用的回合都该做
SCOPE_AT_LEAST_ONCE = "at_least_once"
SCOPE_PER_FRAME = "per_frame"
SCOPES = {SCOPE_AT_LEAST_ONCE, SCOPE_PER_FRAME}

#: 判定项的状态。
CHECK_OK = "ok"
CHECK_MISS = "miss"
CHECK_UNKNOWN = "unknown"


class PatternError(ValueError):
    """模式不满足可判定性准入标准。"""


# --------------------------------------------------------------------------
# 判定项
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    """一个可判定项。

    「可判定」= 给定这一局的事实取值，结论唯一确定。

    两种用法：

    * **局面类检查**：``fact`` 是标量事实（``time_min`` / ``position`` /
      ``matchup`` …），按 :attr:`op` 比较 :attr:`value`。
    * **动作类检查**：``fact`` 是动作事实（``objective_action`` /
      ``positioning_action`` / ``teamfight_action``），此时

      - :attr:`min_count` / :attr:`max_count` 给出次数区间；
      - :attr:`window` 给出时间窗（**局内绝对分钟**）；
      - 没写次数就是「至少一次」。

    :attr:`require` 是**证据等级要求**。写成 :data:`PROV_OBSERVED` 表示
    「这个结论必须由画面直接读出来才算数」—— 推断值不足以据此指责玩家。
    这是从参考项目里学到的最重要的一条：仿真器给的精确坐标与我们
    压缩录屏里的噪声推测，可信度根本不是一回事，不能混在一起打分。
    """

    fact: str
    op: str = "=="
    value: Any = None
    #: 动作类检查：至少做几次（None = 不限下界）。
    min_count: Optional[int] = None
    #: 动作类检查：最多做几次（None = 不限上界）。
    max_count: Optional[int] = None
    #: 时间窗（局内绝对分钟，二元组）。None = 用节点窗口。
    window: Optional[Tuple[float, float]] = None
    #: 证据等级要求。
    require: str = PROV_ANY
    #: 权重（同一节点多项判定时的相对重要性）。
    weight: float = 1.0
    #: 人类可读名字（报告里用）。
    label: str = ""

    # -- 校验 ---------------------------------------------------------------

    def validate(self, known_facts: Optional[set] = None,
                 action_facts: Optional[set] = None) -> None:
        if not self.fact or not isinstance(self.fact, str):
            raise PatternError("fact 必须是非空字符串")
        if self.op not in OPERATORS:
            raise PatternError(f"不支持的操作符 {self.op!r}，可用：{sorted(OPERATORS)}")
        if self.require not in PROVENANCE_CLASSES:
            raise PatternError(f"require 必须是 {sorted(PROVENANCE_CLASSES)} 之一")
        if known_facts is not None and self.fact not in known_facts:
            raise PatternError(f"事实 {self.fact!r} 未在适配器中注册，规则不可判定")
        if action_facts is not None and self.fact in action_facts:
            if self.op not in {"in", "not_in", "==", "!="}:
                raise PatternError(
                    f"动作事实 {self.fact!r} 只支持 in / not_in / == / != 比较"
                )
            if self.op in {"in", "not_in"}:
                if not isinstance(self.value, (list, tuple, set)) or not self.value:
                    raise PatternError(f"{self.op} 需要非空集合作为取值")
        elif self.op in {"in", "not_in"}:
            if not isinstance(self.value, (list, tuple, set)) or not self.value:
                raise PatternError(f"{self.op} 需要非空集合作为取值")
        if self.op == "between":
            if not isinstance(self.value, (list, tuple)) or len(self.value) != 2:
                raise PatternError("between 需要 [下界, 上界] 两个值")
            if self.value[0] > self.value[1]:
                raise PatternError("between 的下界不能大于上界")
        if self.op in NUMERIC_OPS and self.op != "between":
            if not isinstance(self.value, (int, float)) or isinstance(self.value, bool):
                raise PatternError(f"{self.op} 需要数值型取值，实际得到 {self.value!r}")
        if self.min_count is not None and self.min_count < 0:
            raise PatternError("min_count 不能为负")
        if (self.min_count is not None and self.max_count is not None
                and self.min_count > self.max_count):
            raise PatternError("min_count 不能大于 max_count")
        if self.window is not None:
            if not isinstance(self.window, (list, tuple)) or len(self.window) != 2:
                raise PatternError("window 需要 [起, 止] 两个值")
            if self.window[0] > self.window[1]:
                raise PatternError("window 的起点不能大于终点")

    # -- 求值 ---------------------------------------------------------------

    def matches_value(self, actual: Any) -> bool:
        """单个取值是否满足本判定项。

        ``["*"]`` 是**通配哨兵**（2026-11-13）：分路通则要给「玩家选了自己
        的英雄」让路，护栏里就得挡住全部英雄 —— 但把一百多个名字抄进每条
        规则，种子库一半是同一份名单，diff 也没法看。所以允许写
        ``value = ["*"]``，含义是「任何有值的英雄都命中」。
        """
        if self.op == "exists":
            return actual is not None
        if actual is None:
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
                return actual in self.value or self._wildcard(actual)
            if self.op == "not_in":
                return actual not in self.value and not self._wildcard(actual)
            if self.op == "between":
                lo, hi = self.value[0], self.value[1]
                return lo <= actual <= hi
        except TypeError:
            # 取值类型与阈值不符 —— 视为不成立，并保留原始数据供复查。
            return False
        raise PatternError(f"不支持的操作符 {self.op!r}")

    def _wildcard(self, actual: Any = None) -> bool:
        """值表里是不是**只有**通配哨兵 ``"*"``，且这次真的有值可比。

        空串/None **不算命中** —— 护栏的语义是「玩家选了自己的英雄就挡住
        这条通则」；没认出英雄时通则必须照常生效，否则它永远被挡住。
        """
        try:
            if list(self.value) != ["*"]:
                return False
        except TypeError:
            return False
        return actual is not None and str(actual).strip() != ""

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"fact": self.fact, "op": self.op, "value": self.value}
        if self.min_count is not None:
            out["min_count"] = self.min_count
        if self.max_count is not None:
            out["max_count"] = self.max_count
        if self.window is not None:
            out["window"] = list(self.window)
        if self.require != PROV_ANY:
            out["require"] = self.require
        if self.weight != 1.0:
            out["weight"] = self.weight
        if self.label:
            out["label"] = self.label
        return out

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Check":
        window = data.get("window")
        return Check(
            fact=str(data.get("fact") or ""),
            op=str(data.get("op") or "=="),
            value=data.get("value"),
            min_count=data.get("min_count"),
            max_count=data.get("max_count"),
            window=(float(window[0]), float(window[1])) if window else None,
            require=str(data.get("require") or PROV_ANY),
            weight=float(data.get("weight") or 1.0),
            label=str(data.get("label") or ""),
        )


# --------------------------------------------------------------------------
# 时点
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Node:
    """攻略里的一个**时点**。

    ``from_min`` / ``to_min`` 是局内分钟（含端点）。
    窗口是闭区间：``[4, 4]`` 表示第 4 分钟那一分钟。
    """

    label: str
    from_min: float
    to_min: float

    def contains(self, minute: Any) -> bool:
        try:
            m = float(minute)
        except (TypeError, ValueError):
            return False
        return self.from_min <= m <= self.to_min

    def validate(self) -> None:
        if self.from_min > self.to_min:
            raise PatternError("node 的起点不能大于终点")
        if self.from_min < 0:
            raise PatternError("node 的起点不能为负")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "from_min": self.from_min,
            "to_min": self.to_min,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Node":
        return Node(
            label=str(data.get("label") or ""),
            from_min=float(data.get("from_min") or 0),
            to_min=float(data.get("to_min") if data.get("to_min") is not None else 0),
        )


# --------------------------------------------------------------------------
# 适用范围
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AppliesTo:
    """这条模式对谁成立。

    ``heroes`` 为空表示「该分路的任何英雄都行」；
    非空时是**白名单**（英雄专属模式）。
    ``requires`` 是**必须存在**的事实 —— 缺了就无法定位，
    要在诊断里点名，而不是让规则静默不触发。
    """

    lane: str = ""
    heroes: Tuple[str, ...] = ()
    requires: Tuple[str, ...] = ()

    def validate(self, known_facts: Optional[set] = None) -> None:
        if known_facts is not None:
            for fact in self.requires:
                if fact not in known_facts:
                    raise PatternError(
                        f"适用范围要求事实 {fact!r}，但适配器没有注册它"
                    )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.lane:
            out["lane"] = self.lane
        if self.heroes:
            out["heroes"] = list(self.heroes)
        if self.requires:
            out["requires"] = list(self.requires)
        return out

    @staticmethod
    def from_dict(data: Optional[Dict[str, Any]]) -> "AppliesTo":
        if not data:
            return AppliesTo()
        return AppliesTo(
            lane=str(data.get("lane") or ""),
            heroes=tuple(str(h) for h in (data.get("heroes") or [])),
            requires=tuple(str(r) for r in (data.get("requires") or [])),
        )


# --------------------------------------------------------------------------
# 可纠正性
# --------------------------------------------------------------------------

#: 永远可纠正（前提不依赖读不出来的信息）。
CORRECTIVE_ALWAYS = "always"
#: 只有分支前提被指认后才可纠正。
CORRECTIVE_ON_BRANCH = "on_branch"
#: 永远不纠正。
CORRECTIVE_NEVER = "never"

CORRECTIVE_MODES = {CORRECTIVE_ALWAYS, CORRECTIVE_ON_BRANCH, CORRECTIVE_NEVER}


@dataclass(frozen=True)
class Corrective:
    """这条模式能不能拿来纠正玩家 —— **运行时决定**，不是入库时定死。

    ``mode``：

    * :data:`CORRECTIVE_ALWAYS` —— 不依赖分支前提，任何时候都能纠正。
    * :data:`CORRECTIVE_ON_BRANCH` —— 攻略给了好几支，前提读不出来就不纠正；
      一旦模型指认了第几支成立（或用户填了），立刻可纠正。
    * :data:`CORRECTIVE_NEVER` —— 明确不用于纠正（只作模式库展示）。

    ``branches`` 是原文的分支列表（每项 ``premise``/``action``/``outcome``）。
    ``branch_facts`` 是「哪一支成立」由哪些事实判定 —— 由适配器或采集层填。
    """

    mode: str = CORRECTIVE_ALWAYS
    branches: Tuple[Dict[str, Any], ...] = ()
    reason: str = ""

    def validate(self) -> None:
        if self.mode not in CORRECTIVE_MODES:
            raise PatternError(
                f"corrective.mode 必须是 {sorted(CORRECTIVE_MODES)} 之一，"
                f"实际得到 {self.mode!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"mode": self.mode}
        if self.branches:
            out["branches"] = [dict(b) for b in self.branches]
        if self.reason:
            out["reason"] = self.reason
        return out

    @staticmethod
    def from_dict(data: Optional[Dict[str, Any]]) -> "Corrective":
        if not data:
            return Corrective()
        return Corrective(
            mode=str(data.get("mode") or CORRECTIVE_ALWAYS),
            branches=tuple(
                dict(b) for b in (data.get("branches") or []) if isinstance(b, dict)
            ),
            reason=str(data.get("reason") or ""),
        )


# --------------------------------------------------------------------------
# 模式
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Subject:
    """适用时机 —— 这条模式在**哪些动作上**才适用。

    为什么需要它
    ------------
    攻略说的往往不是「这段时间每分钟都得做 X」，而是
    「**在做 A 还是做 B 的时候，该选 A**」::

        清完自己野区升 4 级 > 任何一次 gank
        控龙优先，而不是追人头

    这两条的前提都是「**那一刻玩家正在做某个目标类动作**」。
    没有它，模式就会在玩家做别的事情的回合上也判 —— 玩家在推线补兵，
    却被说「该抢龙」，那是在冤枉人。

    ``values`` 为空表示「这一类动作做过就算适用」。
    """

    fact: str
    values: Tuple[str, ...] = ()

    def matches(self, kind: str, value: Any) -> bool:
        if kind != self.fact:
            return False
        if not self.values:
            return True
        return value in self.values

    def validate(self, known_facts: Optional[set] = None) -> None:
        if not self.fact or not isinstance(self.fact, str):
            raise PatternError("subject.fact 必须是非空字符串")
        if known_facts is not None and self.fact not in known_facts:
            raise PatternError(
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
            values=tuple(str(v) for v in (data.get("values") or [])),
        )


@dataclass
class Pattern:
    """一条可判定的操作模式（v2）。

    结构：

    * :attr:`node` —— **时点**。攻略在什么时刻谈这件事。
    * :attr:`applies_to` —— **对谁**成立（分路 / 英雄 / 必需事实）。
    * :attr:`when` —— **前提**（局面事实）。全成立才算适用。
    * :attr:`expect` —— **期望**（一组判定项）。这是「该做什么」。
    * :attr:`accept` —— **本节点认可的全部打法**。玩家做了其中另一件，
      说明他这会儿在按攻略做事，只是跟本条不一样 → 不纠正他。
    * :attr:`corrective` —— **能不能纠正**（运行时决定）。
    """

    id: str
    game: str
    version: str
    title: str
    category: str = ""
    lane: str = ""
    phase: str = ""
    node: Node = field(default_factory=lambda: Node("", 0, 0))
    applies_to: AppliesTo = field(default_factory=AppliesTo)
    #: 适用时机（可选）。不填就是老行为：前提成立就判。
    subject: Optional[Subject] = None
    when: List[Check] = field(default_factory=list)
    #: 「任一成立」（可空）。
    when_any: List[Check] = field(default_factory=list)
    #: 「全部不成立」（可空）。用来写护栏（「这一帧不能回城」）。
    when_none: List[Check] = field(default_factory=list)
    expect: List[Check] = field(default_factory=list)
    accept: List[str] = field(default_factory=list)
    scope: str = SCOPE_AT_LEAST_ONCE
    severity: str = "medium"
    status: str = "active"
    corrective: Corrective = field(default_factory=Corrective)
    source: Optional[Dict[str, Any]] = None
    rationale: str = ""
    deviation_hint: str = ""
    premise: str = ""
    action: str = ""
    outcome: str = ""
    tags: List[str] = field(default_factory=list)

    # -- 校验 ---------------------------------------------------------------

    def validate(self, known_facts: Optional[set] = None,
                 action_facts: Optional[set] = None) -> None:
        if not self.id:
            raise PatternError("模式必须有 id")
        if not self.game:
            raise PatternError("模式必须绑定游戏")
        if not self.version:
            raise PatternError("模式必须绑定游戏版本")
        if not self.title:
            raise PatternError("模式必须有标题")
        if self.severity not in SEVERITIES:
            raise PatternError(f"severity 必须是 {sorted(SEVERITIES)} 之一")
        if self.status not in RULE_STATUS:
            raise PatternError(f"status 必须是 {sorted(RULE_STATUS)} 之一")
        if self.scope not in SCOPES:
            raise PatternError(f"scope 必须是 {sorted(SCOPES)} 之一")
        self.node.validate()
        self.applies_to.validate(known_facts)
        self.corrective.validate()
        if self.subject is not None:
            self.subject.validate(known_facts)
        for chk in (list(self.when) + list(self.when_any) + list(self.when_none)
                    + list(self.expect)):
            chk.validate(known_facts, action_facts)
        if not self.expect:
            raise PatternError("模式必须至少有一项期望")
        if known_facts is not None and not self.when:
            raise PatternError("模式必须有前提条件：没有前提的模式无法定位到具体局面")

    def referenced_facts(self) -> set:
        facts = {c.fact for c in self.when}
        facts |= {c.fact for c in self.when_any}
        facts |= {c.fact for c in self.when_none}
        facts |= {c.fact for c in self.expect}
        facts |= set(self.applies_to.requires)
        return facts

    # -- 序列化 -------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "id": self.id,
            "game": self.game,
            "version": self.version,
            "title": self.title,
            "category": self.category,
            "lane": self.lane,
            "phase": self.phase,
            "node": self.node.to_dict(),
            "applies_to": self.applies_to.to_dict(),
            "subject": self.subject.to_dict() if self.subject else None,
            "when": [c.to_dict() for c in self.when],
            "when_any": [c.to_dict() for c in self.when_any],
            "when_none": [c.to_dict() for c in self.when_none],
            "expect": [c.to_dict() for c in self.expect],
            "accept": list(self.accept),
            "scope": self.scope,
            "severity": self.severity,
            "status": self.status,
            "corrective": self.corrective.to_dict(),
            "source": dict(self.source) if self.source else None,
            "rationale": self.rationale,
            "deviation_hint": self.deviation_hint,
            "premise": self.premise,
            "action": self.action,
            "outcome": self.outcome,
            "tags": list(self.tags),
        }
        return out

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Pattern":
        return Pattern(
            id=str(data.get("id") or ""),
            game=str(data.get("game") or ""),
            version=str(data.get("version") or ""),
            title=str(data.get("title") or ""),
            category=str(data.get("category") or ""),
            lane=str(data.get("lane") or ""),
            phase=str(data.get("phase") or ""),
            node=Node.from_dict(data.get("node") or {}),
            applies_to=AppliesTo.from_dict(data.get("applies_to")),
            subject=Subject.from_dict(data.get("subject")),
            when=[Check.from_dict(c) for c in (data.get("when") or [])],
            when_any=[Check.from_dict(c) for c in (data.get("when_any") or [])],
            when_none=[Check.from_dict(c) for c in (data.get("when_none") or [])],
            expect=[Check.from_dict(c) for c in (data.get("expect") or [])],
            accept=[str(a) for a in (data.get("accept") or [])],
            scope=str(data.get("scope") or SCOPE_AT_LEAST_ONCE),
            severity=str(data.get("severity") or "medium"),
            status=str(data.get("status") or "active"),
            corrective=Corrective.from_dict(data.get("corrective")),
            source=dict(data["source"]) if data.get("source") else None,
            rationale=str(data.get("rationale") or ""),
            deviation_hint=str(data.get("deviation_hint") or ""),
            premise=str(data.get("premise") or ""),
            action=str(data.get("action") or ""),
            outcome=str(data.get("outcome") or ""),
            tags=[str(t) for t in (data.get("tags") or [])],
        )


# --------------------------------------------------------------------------
# 判定结果
# --------------------------------------------------------------------------


@dataclass
class CheckOutcome:
    """一项判定的结果与证据。"""

    check: Check
    status: str                    # ok / miss / unknown
    count: int = 0
    actual: Any = None
    present: bool = True
    provenance: str = PROV_ANY
    detail: str = ""

    @property
    def is_ok(self) -> bool:
        return self.status == CHECK_OK

    @property
    def is_unknown(self) -> bool:
        return self.status == CHECK_UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact": self.check.fact,
            "op": self.check.op,
            "threshold": self.check.value,
            "min_count": self.check.min_count,
            "max_count": self.check.max_count,
            "window": list(self.check.window) if self.check.window else None,
            "require": self.check.require,
            "weight": self.check.weight,
            "status": self.status,
            "count": self.count,
            "actual": self.actual,
            "present": self.present,
            "provenance": self.provenance,
            "provenance_label": PROVENANCE_LABELS.get(self.provenance, self.provenance),
            "detail": self.detail,
        }


@dataclass
class PatternVerdict:
    """一条模式在一局上的判定结果。"""

    pattern_id: str
    title: str
    lane: str
    phase: str
    node_label: str
    #: 与 v1 对齐的四种结论之一，外加 ``exempt``（玩家做了本节点认可的另一件事）。
    status: str
    #: 节点的起始分钟。排序用（标签是「第 12 分钟」这种字符串，解析不了）。
    node_from_min: float = 0.0
    #: 前提判定结果。
    when_outcomes: List[CheckOutcome] = field(default_factory=list)
    #: 期望判定结果。
    expect_outcomes: List[CheckOutcome] = field(default_factory=list)
    #: 适用到的回合标签。
    round_labels: List[str] = field(default_factory=list)
    #: 这一局实际做了的事（全部动作，按类型分组）。
    performed: Dict[str, List[str]] = field(default_factory=dict)
    #: 这一局实际做了的事（完整记录，供报告层展开）。
    performed_records: List[Dict[str, Any]] = field(default_factory=list)
    #: 节点窗口内适用到的回合（完整快照）。
    rounds: List[Dict[str, Any]] = field(default_factory=list)
    #: 这一局指认了第几支（0 = 没指认）。
    branch_no: int = 0
    #: **分支级明细**：这一局成立的是哪一支，玩家做的是不是那一支。
    #:
    #: 为什么模式级的结论不够
    #: ----------------------
    #: 模式级判的是「这一段里玩家做没做攻略要求的那类动作」。
    #: 但攻略在一个节点给的是**好几支**，每支的做法不同。
    #: 玩家做了「刷野」—— 满足模式级判定，可这一局成立的是
    #: 「敌方打野蓝开」那一支（该红开反向开局），
    #: 他做的其实是另一支的事。模式级看不见这个差别，
    #: 玩家看到的建议也就还是那张菜单。
    #:
    #: 指认了分支之后就能回答：**这一支该做什么，你做的是不是它。**
    branch_detail: Dict[str, Any] = field(default_factory=dict)
    #: 这一局能不能拿来纠正玩家（运行时结论）。
    corrective: bool = False
    #: 纠正不了的原因（中文）。
    corrective_reason: str = ""
    rationale: str = ""
    deviation_hint: str = ""
    premise: str = ""
    action: str = ""
    outcome: str = ""
    source: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)
    severity: str = "medium"

    @property
    def is_deviation(self) -> bool:
        return self.status == "deviated"

    def score(self) -> float:
        """节点完成度：达成项的权重占比（未知的项不计入分母）。"""
        total = sum(c.check.weight for c in self.expect_outcomes)
        if total <= 0:
            return 0.0
        got = sum(
            c.check.weight for c in self.expect_outcomes if c.is_ok
        )
        return got / total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "title": self.title,
            "lane": self.lane,
            "phase": self.phase,
            "node_label": self.node_label,
            "node_from_min": self.node_from_min,
            "status": self.status,
            "when": [o.to_dict() for o in self.when_outcomes],
            "expect": [o.to_dict() for o in self.expect_outcomes],
            "round_labels": list(self.round_labels),
            "performed": {k: list(v) for k, v in self.performed.items()},
            "performed_records": [dict(r) for r in self.performed_records],
            "rounds": [dict(r) for r in self.rounds],
            "branch_no": self.branch_no,
            "branch_detail": dict(self.branch_detail),
            "corrective": self.corrective,
            "corrective_reason": self.corrective_reason,
            "rationale": self.rationale,
            "deviation_hint": self.deviation_hint,
            "premise": self.premise,
            "action": self.action,
            "outcome": self.outcome,
            "source": dict(self.source) if self.source else None,
            "tags": list(self.tags),
            "severity": self.severity,
            "score": round(self.score(), 4),
        }


#: 与 v1 一致的四种结论。
STATUS_NOT_TRIGGERED = "not_triggered"
STATUS_SATISFIED = "satisfied"
STATUS_DEVIATED = "deviated"
STATUS_UNDETERMINED = "undetermined"
#: 玩家做了本节点认可的另一件事 —— 不是偏离。
STATUS_EXEMPT = "exempt"

# --------------------------------------------------------------------------
# 分支级判定
# --------------------------------------------------------------------------

#: 玩家做的正是这一局成立的那一支。
BRANCH_ON_BRANCH = "on_branch"
#: 玩家做的是**别的**分支的事（做的是攻略认可的，只是不是这一支）。
BRANCH_OTHER = "other_branch"
#: 这一支要求的动作没做。
BRANCH_MISS = "miss"

BRANCH_STATUS_LABELS: Dict[str, str] = {
    BRANCH_ON_BRANCH: "就是这一支",
    BRANCH_OTHER: "做的是别的分支那件事",
    BRANCH_MISS: "这一支要求的没做",
}


def branch_values(branch: Dict[str, Any]) -> Tuple[str, ...]:
    """一条分支要求玩家做的**动作取值**。

    只有英雄层的分支带这个 —— 分路层的分支是散文，
    没有可机读的动作取值（它的 ``accept`` 走另一套机制）。
    """
    return tuple(str(v) for v in (branch.get("values") or []))



# --------------------------------------------------------------------------
# 判定
# --------------------------------------------------------------------------


def _fact_source(rnd, fact: str) -> str:
    """取该回合里这个事实的证据等级。

    取不到就当 :data:`PROV_ANY` —— 旧数据没有来源标记，
    不该因为「没有来源信息」把结论降级成无法判定。
    """
    sources = getattr(rnd, "fact_sources", None) or {}
    return str(sources.get(fact) or PROV_ANY)


def _prov_satisfies(require: str, actual_prov: str) -> bool:
    if require == PROV_ANY or actual_prov == PROV_ANY:
        return True
    if require == PROV_OBSERVED:
        return actual_prov == PROV_OBSERVED
    if require == PROV_INFERRED:
        return actual_prov in (PROV_INFERRED, PROV_OBSERVED)
    if require == PROV_DECLARED:
        return actual_prov in (PROV_DECLARED, PROV_OBSERVED)
    return True


def _round_minute(rnd) -> Optional[float]:
    facts = getattr(rnd, "facts", None) or {}
    for key in ("time_min", "minute"):
        if key in facts:
            try:
                return float(facts[key])
            except (TypeError, ValueError):
                return None
    idx = getattr(rnd, "time_index", None)
    if idx is not None:
        try:
            return float(idx)
        except (TypeError, ValueError):
            return None
    return None


def _branch_no_for(rnd, pattern_id: str) -> int:
    """这一回合模型指认了本模式的第几支。

    采集层的评审结果里带着 ``mode_id`` 与 ``branch``（1 起），
    指认了就说明分支前提成立 → 规则可纠正。
    """
    for item in (getattr(rnd, "review", None) or []):
        if not isinstance(item, dict):
            continue
        if item.get("mode_id") != pattern_id:
            continue
        try:
            no = int(item.get("branch_no") or 0)
        except (TypeError, ValueError):
            no = 0
        if no > 0:
            return no
    return 0


def _performances(rnd) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for act in (getattr(rnd, "actions", None) or []):
        out.setdefault(act.kind, []).append(str(act.value))
    return out


def evaluate_pattern_rounds(
    pattern: Pattern,
    ctx,
    *,
    action_facts: Optional[set] = None,
) -> List[PatternVerdict]:
    """在节点窗口内**逐回合**判定一条模式。

    为什么是逐回合
    --------------
    报告的核心主张是「结论必须指向具体回合」——
    「第 8 分钟你在经济落后 300 的时候还在分推」这句话里，
    时间点和那一回合的实际取值都是必需的。

    而节点级聚合会把整个窗口压成一条判定：窗口里第 6 分钟做到了、
    第 8 分钟没做到，压成一条就只剩「做到了」——
    **恰恰把做错的那一分钟抹掉了**。

    所以：窗口负责圈定「这段时间」，判定仍然逐回合出。
    窗口级的口径（至少一次 / 逐回合 / 计数）由 :attr:`Pattern.scope`
    与判定项上的 ``min_count`` 表达，它们聚合的是**结论**而不是回合。

    带计数约束的判定项（``min_count > 1`` 或设了 ``max_count``）
    是**节点级**的 —— 它们问的是「这段一共做了几次」，
    逐回合问没有意义。这类判定项会单独产出一条节点级判定。
    """
    action_facts = action_facts or set()

    if pattern.applies_to.heroes:
        hero = str((ctx.player or {}).get("hero") or "").strip()
        if not hero:
            # **不知道在玩谁 → 这条英雄专属模式无法定位。**
            #
            # 以前这里是「英雄对不上就跳过」，英雄为空时**不跳过** ——
            # 于是「赵云 2:00 该控河道」会被判到任何一个没填英雄的对局上。
            # 英雄专属模式的全部内容都挂在「你在玩赵云」这个前提上，
            # 前提不成立就没有模式可言。
            return []
        if hero not in pattern.applies_to.heroes:
            return []

    rounds: List[Tuple[Any, float]] = []
    for rnd in ctx.iter_rounds():
        minute = _round_minute(rnd)
        if minute is None:
            continue
        if pattern.node.contains(minute):
            rounds.append((rnd, minute))
    if not rounds:
        return []

    # 前提：逐回合求值。局面类事实（分路 / 英雄 / 对位）是对局级的，
    # 会广播到每个回合，所以任何一个回合成立就等于整窗成立。
    applicable: List[Tuple[Any, float, List[CheckOutcome]]] = []
    for rnd, minute in rounds:
        # **适用时机**：这一帧玩家根本没有参与这类动作 → 这条模式不适用。
        #
        # 「不适用」和「没做到」是两件事 —— 攻略说的是
        # 「在做 A 还是做 B 的时候该选 A」，玩家这一帧在推线补兵，
        # 跟这条模式没关系，判成偏离就是冤枉人。
        #
        # **只有这一帧真的标了动作才做这个过滤**：一帧什么都没标有两种可能
        # （「确实什么都没做」/「没标、看不清」），交给下面的
        # 「动作事实缺失」分支去分，这里不替玩家做判断。
        if (pattern.subject is not None
                and getattr(rnd, "actions", None)
                and not any(
                    pattern.subject.matches(a.kind, a.value)
                    for a in rnd.actions
                )):
            continue
        outcomes, ok = _evaluate_when(pattern, rnd, ctx)
        if ok:
            applicable.append((rnd, minute, outcomes))
    if not applicable:
        return []

    simple_checks = [
        c for c in pattern.expect
        if (c.min_count or 0) <= 1 and c.max_count is None
    ]
    count_checks = [c for c in pattern.expect if c not in simple_checks]

    out: List[PatternVerdict] = []
    if simple_checks:
        for rnd, minute, when_outcomes in applicable:
            verdict = _verdict_for_round(
                pattern, rnd, minute, when_outcomes, simple_checks,
                ctx, action_facts,
            )
            if verdict is not None:
                out.append(verdict)

    if count_checks:
        node_verdict = _verdict_for_node(
            pattern, applicable, count_checks, ctx, action_facts,
        )
        if node_verdict is not None:
            out.append(node_verdict)

    return out


def _branch_detail(
    pattern: Pattern,
    branch_no: int,
    performed: Dict[str, List[str]],
) -> Dict[str, Any]:
    """这一局成立的是哪一支，玩家做的是不是那一支。

    只在**分支带动作取值**时才有结论 ——
    分路层的分支是散文（「敌方打野为红开」），没有可机读的取值可比，
    那就不猜。

    三种结论：

    * ``on_branch``     —— 做的正是这一支要求的动作；
    * ``other_branch``  —— 做的是**别的**分支那件事（做的是攻略认可的，
      只是不是这一支）。这跟「没做到」是两回事，不能混为一谈；
    * ``miss``          —— 这一支要求的动作一件都没做。
    """
    branches = pattern.corrective.branches
    if not branch_no or not (1 <= branch_no <= len(branches)):
        return {}
    branch = branches[branch_no - 1]
    want = set(branch_values(branch))
    if not want:
        return {}

    all_values: set = set()
    for b in branches:
        all_values |= set(branch_values(b))

    done: set = set()
    for values in (performed or {}).values():
        for v in values:
            done.add(str(v))

    matched = sorted(done & want)
    other = sorted((done & all_values) - want)
    if matched:
        status = BRANCH_ON_BRANCH
    elif other:
        status = BRANCH_OTHER
    else:
        status = BRANCH_MISS

    return {
        "branch_no": branch_no,
        "premise": str(branch.get("premise") or ""),
        "action": str(branch.get("action") or ""),
        "outcome": str(branch.get("outcome") or ""),
        "values": sorted(want),
        "matched_values": matched,
        "other_branch_values": other,
        "status": status,
        "label": BRANCH_STATUS_LABELS.get(status, status),
    }


def _verdict_for_round(
    pattern: Pattern,
    rnd,
    minute: float,
    when_outcomes: List[CheckOutcome],
    checks: List[Check],
    ctx,
    action_facts: set,
) -> Optional[PatternVerdict]:
    """一个回合一条判定。"""
    expect_outcomes = _evaluate_expect_in_round(
        pattern, checks, rnd, ctx, action_facts
    )

    performed = _performances(rnd)
    performed_records = [
        a.to_dict() for a in (getattr(rnd, "actions", None) or [])
    ]
    round_snapshots = [_round_snapshot(rnd)]

    # **期望动作已经做到了，就不能豁免** ——
    #
    # 豁免的口径是「玩家做的是这个节点攻略认可的**另一件**事」，
    # 前提是**期望那件事他没做**。他要是两件都做了（期望的那件也做了），
    # 那就是**做对了**，不是「做了另一件事所以不算」。
    #
    # v1 引擎一直是这样（见 :func:`app.core.matcher`）：
    # 豁免要求「期望动作没做到」。
    #
    # 缺这道闸的后果很安静：种子 84 条模式（都有 accept）里，
    # 玩家在同一回合**既做了期望动作、又做了另一个合法动作**时，
    # 整条判定被判成 exempt → 做对了既不计入 satisfied、
    # 也不进报告（bridge 把豁免判定整条丢掉）。
    _achieved = any(o.status == CHECK_OK for o in expect_outcomes)
    accepted_hit = (None if _achieved
                    else _accepted_alternative(pattern, checks, performed))
    branch_no = _branch_no_for(rnd, pattern.id)
    corrective, corrective_reason = _resolve_corrective(pattern, branch_no)
    branch_detail = _branch_detail(pattern, branch_no, performed)

    base = _base_verdict(pattern, rnd, minute, when_outcomes, round_snapshots)

    if accepted_hit:
        return PatternVerdict(
            status=STATUS_EXEMPT,
            expect_outcomes=expect_outcomes,
            performed=performed,
            performed_records=performed_records,
            rounds=round_snapshots,
            branch_no=branch_no,
            branch_detail=branch_detail,
            corrective=False,
            corrective_reason=(
                f"玩家做了本节点攻略认可的另一件事（{accepted_hit}），不纠正"
            ),
            **base,
        )

    unknown = [o for o in expect_outcomes if o.is_unknown]
    missed = [o for o in expect_outcomes if o.status == CHECK_MISS]
    if missed:
        status = STATUS_DEVIATED
    elif unknown:
        status = STATUS_UNDETERMINED
    else:
        status = STATUS_SATISFIED

    return PatternVerdict(
        status=status,
        expect_outcomes=expect_outcomes,
        performed=performed,
        performed_records=performed_records,
        rounds=round_snapshots,
        branch_no=branch_no,
        branch_detail=branch_detail,
        corrective=corrective and status == STATUS_DEVIATED,
        corrective_reason=corrective_reason,
        **base,
    )


def _verdict_for_node(
    pattern: Pattern,
    applicable: List[Tuple[Any, float, List[CheckOutcome]]],
    checks: List[Check],
    ctx,
    action_facts: set,
) -> Optional[PatternVerdict]:
    """节点级判定（带计数约束的判定项）。"""
    expect_outcomes = _evaluate_expect(
        pattern, checks, applicable, ctx, action_facts
    )

    performed: Dict[str, List[str]] = {}
    performed_records: List[Dict[str, Any]] = []
    round_snapshots: List[Dict[str, Any]] = []
    for rnd, _minute, _when in applicable:
        for kind, values in _performances(rnd).items():
            performed.setdefault(kind, []).extend(values)
        for act in (getattr(rnd, "actions", None) or []):
            performed_records.append(act.to_dict())
        round_snapshots.append(_round_snapshot(rnd))

    first_rnd, first_minute, when_outcomes = applicable[0]
    base = _base_verdict(
        pattern, first_rnd, first_minute, when_outcomes, round_snapshots
    )
    branch_no = 0
    for rnd, _minute, _when in applicable:
        branch_no = _branch_no_for(rnd, pattern.id)
        if branch_no:
            break
    corrective, corrective_reason = _resolve_corrective(pattern, branch_no)
    branch_detail = _branch_detail(pattern, branch_no, performed)

    # **期望动作已经做到了，就不能豁免**（与上面那条同一个口径）。
    _achieved = any(o.status == CHECK_OK for o in expect_outcomes)
    accepted_hit = (None if _achieved
                    else _accepted_alternative(pattern, checks, performed))
    if accepted_hit:
        return PatternVerdict(
            status=STATUS_EXEMPT,
            expect_outcomes=expect_outcomes,
            performed=performed,
            performed_records=performed_records,
            rounds=round_snapshots,
            branch_no=branch_no,
            branch_detail=branch_detail,
            corrective=False,
            corrective_reason=(
                f"玩家做了本节点攻略认可的另一件事（{accepted_hit}），不纠正"
            ),
            **base,
        )

    unknown = [o for o in expect_outcomes if o.is_unknown]
    missed = [o for o in expect_outcomes if o.status == CHECK_MISS]
    if missed:
        status = STATUS_DEVIATED
    elif unknown:
        status = STATUS_UNDETERMINED
    else:
        status = STATUS_SATISFIED

    return PatternVerdict(
        status=status,
        expect_outcomes=expect_outcomes,
        performed=performed,
        performed_records=performed_records,
        rounds=round_snapshots,
        branch_no=branch_no,
        branch_detail=branch_detail,
        corrective=corrective and status == STATUS_DEVIATED,
        corrective_reason=corrective_reason,
        **base,
    )


def _base_verdict(pattern, rnd, minute, when_outcomes, round_snapshots):
    return dict(
        pattern_id=pattern.id,
        title=pattern.title,
        lane=pattern.lane,
        phase=pattern.phase,
        node_label=pattern.node.label,
        node_from_min=pattern.node.from_min,
        when_outcomes=when_outcomes,
        round_labels=[rnd.label],
        rationale=pattern.rationale,
        deviation_hint=pattern.deviation_hint,
        premise=pattern.premise,
        action=pattern.action,
        outcome=pattern.outcome,
        source=dict(pattern.source) if pattern.source else None,
        tags=list(pattern.tags),
        severity=pattern.severity,
    )


def _round_snapshot(rnd) -> Dict[str, Any]:
    facts = getattr(rnd, "facts", None) or {}
    missing = getattr(rnd, "missing_facts", None) or set()
    return {
        "label": rnd.label,
        "time_index": float(getattr(rnd, "time_index", 0.0) or 0.0),
        "stage": int(getattr(rnd, "stage", 0) or 0),
        "phase": int(getattr(rnd, "phase", 0) or 0),
        "facts": {
            k: v for k, v in facts.items()
            if v is not None and k not in missing
        },
    }


def _accepted_alternative(pattern, checks, performed) -> Optional[str]:
    """玩家做了本节点攻略认可的**另一件**事吗？

    只在**期望所看的那几类动作**里找 —— 拿别的动作类别的值去豁免是错的
    （「清视野精灵」被认可，不代表「抓人」也算这条节点的正解）。
    """
    if not pattern.accept:
        return None
    expect_values = set()
    expect_kinds = set()
    for chk in checks:
        expect_kinds.add(chk.fact)
        if isinstance(chk.value, (list, tuple, set)):
            expect_values |= {str(v) for v in chk.value}
        elif chk.value is not None:
            expect_values.add(str(chk.value))
    for kind, values in performed.items():
        if kind not in expect_kinds:
            continue
        for v in values:
            if v in pattern.accept and v not in expect_values:
                return v
    return None


def evaluate_pattern(
    pattern: Pattern,
    ctx,
    *,
    action_facts: Optional[set] = None,
) -> Optional[PatternVerdict]:
    """判定一条模式，返回**第一条**判定（节点级判定优先）。

    报告层用的是 :func:`evaluate_pattern_rounds`（逐回合）；
    这个函数给「只想知道这条模式整体什么结论」的场景用 ——
    比如诊断、覆盖率统计。
    """
    verdicts = evaluate_pattern_rounds(
        pattern, ctx, action_facts=action_facts or set()
    )
    if not verdicts:
        return None
    for v in verdicts:
        if len(v.rounds) > 1:
            return v            # 节点级判定排前面
    return verdicts[0]


def _evaluate_expect_in_round(pattern, checks, rnd, ctx, action_facts):
    """单个回合上的期望判定。"""
    outcomes: List[CheckOutcome] = []
    for chk in checks:
        if chk.fact in action_facts:
            acts = [
                a for a in (getattr(rnd, "actions", None) or [])
                if a.kind == chk.fact
            ]
            if not acts:
                missing = chk.fact in (getattr(rnd, "missing_facts", None) or set())
                outcomes.append(CheckOutcome(
                    check=chk,
                    status=CHECK_UNKNOWN if missing else CHECK_MISS,
                    count=0, actual=[], present=not missing,
                    detail="这一帧没读到这类动作" if missing
                           else "这一帧没做这类动作",
                ))
                continue
            hits = [a for a in acts if chk.matches_value(a.value)]
            need = chk.min_count if chk.min_count is not None else 1
            ok = len(hits) >= need
            outcomes.append(CheckOutcome(
                check=chk,
                status=CHECK_OK if ok else CHECK_MISS,
                count=len(hits),
                actual=[str(a.value) for a in hits],
                present=True,
                detail=f"这一帧做了 {len(hits)} 次（要求 ≥{need}）",
            ))
        else:
            value, present = ctx.resolve_in_round(rnd)(chk.fact)
            prov = _fact_source(rnd, chk.fact)
            if not present:
                outcomes.append(CheckOutcome(
                    check=chk, status=CHECK_UNKNOWN, present=False,
                    actual=None, provenance=prov,
                    detail="这一局没采到这个事实",
                ))
                continue
            if not _prov_satisfies(chk.require, prov):
                outcomes.append(CheckOutcome(
                    check=chk, status=CHECK_UNKNOWN, present=True,
                    actual=value, provenance=prov,
                    detail=(
                        f"这个结论要求「{PROVENANCE_LABELS.get(chk.require, chk.require)}」"
                        f"的证据，实际是「{PROVENANCE_LABELS.get(prov, prov)}」"
                    ),
                ))
                continue
            hit = chk.matches_value(value)
            outcomes.append(CheckOutcome(
                check=chk, status=CHECK_OK if hit else CHECK_MISS,
                present=True, actual=value, provenance=prov,
            ))
    return outcomes


def _evaluate_when(pattern: Pattern, rnd, ctx):
    """在一回合上求前提。返回 (各项结果, 是否全成立)。

    三组语义与 v1 一致：

    * ``when``      —— 全部成立
    * ``when_any``  —— 任一成立（一个都没成立则不触发）
    * ``when_none`` —— 全部**不**成立（护栏；事实缺失时不判违反）
    """
    outcomes: List[CheckOutcome] = []
    ok = True
    resolve = ctx.resolve_in_round(rnd)

    def _scalar(chk: Check) -> CheckOutcome:
        value, present = resolve(chk.fact)
        prov = _fact_source(rnd, chk.fact)
        if not present:
            return CheckOutcome(
                check=chk, status=CHECK_UNKNOWN, present=False,
                actual=None, provenance=prov,
                detail="这一局没采到这个事实",
            )
        if not _prov_satisfies(chk.require, prov):
            return CheckOutcome(
                check=chk, status=CHECK_UNKNOWN, present=True,
                actual=value, provenance=prov,
                detail=(
                    f"这个结论要求「{PROVENANCE_LABELS.get(chk.require, chk.require)}」"
                    f"的证据，实际是「{PROVENANCE_LABELS.get(prov, prov)}」"
                ),
            )
        hit = chk.matches_value(value)
        return CheckOutcome(
            check=chk, status=CHECK_OK if hit else CHECK_MISS,
            present=True, actual=value, provenance=prov,
        )

    for chk in pattern.when:
        out = _scalar(chk)
        outcomes.append(out)
        ok = ok and out.status == CHECK_OK

    if pattern.when_any:
        any_outcomes = [_scalar(chk) for chk in pattern.when_any]
        outcomes.extend(any_outcomes)
        if not any(o.status == CHECK_OK for o in any_outcomes):
            ok = False

    for chk in pattern.when_none:
        out = _scalar(chk)
        # 护栏：**条件成立**才是违反；读不到就不判违反（与 v1 一致）。
        #
        # 「这一帧没出现被禁止的事」是护栏**通过**，
        # 不是「这个条件没满足」—— 两者在判据上正好相反。
        violated = (out.status == CHECK_OK)
        outcomes.append(CheckOutcome(
            check=chk,
            status=CHECK_MISS if violated else CHECK_OK,
            present=out.present,
            actual=out.actual,
            provenance=out.provenance,
            detail=("护栏被触发：这一帧不该出现这个动作" if violated
                    else "护栏未触发"),
        ))
        ok = ok and not violated

    return outcomes, ok


def _evaluate_expect(pattern: Pattern, checks, applicable, ctx, action_facts: set):
    """在节点窗口内求期望（节点级）。返回各项结果。"""
    outcomes: List[CheckOutcome] = []
    for chk in checks:
        if chk.fact in action_facts:
            outcomes.append(_evaluate_action_expect(pattern, chk, applicable))
        else:
            outcomes.append(_evaluate_scalar_expect(pattern, chk, applicable, ctx))
    return outcomes


def _evaluate_action_expect(pattern: Pattern, chk: Check, applicable):
    """节点级判定项：统计窗口内一共做了几次符合的动作。

    这是**节点级**的（配合 ``min_count`` / ``max_count`` 使用）——
    逐回合问「这段一共做了几次」没有意义。

    口径说明：``at_least_once`` 与 ``per_frame`` 的差别现在体现在
    **判定粒度**上（逐回合 vs 汇总成一条），由 :func:`match_patterns`
    与归因层处理；这里只负责数次数。
    """
    lo, hi = chk.window or (pattern.node.from_min, pattern.node.to_min)
    hits = []
    total = 0
    rounds_in_window = 0
    missing = False
    any_actions = False

    for entry in applicable:
        rnd, minute = entry[0], entry[1]
        if not (lo <= minute <= hi):
            continue
        rounds_in_window += 1
        acts = [a for a in (getattr(rnd, "actions", None) or []) if a.kind == chk.fact]
        if getattr(rnd, "actions", None):
            any_actions = True
        if not acts:
            if chk.fact in (getattr(rnd, "missing_facts", None) or set()):
                missing = True
            continue
        total += len(acts)
        for a in acts:
            if chk.matches_value(a.value):
                hits.append(a)

    count = len(hits)
    need = chk.min_count if chk.min_count is not None else 1

    if count == 0 and total == 0:
        if not any_actions and not missing:
            return CheckOutcome(
                check=chk, status=CHECK_UNKNOWN, count=0, actual=[],
                present=False, detail="这一段没有记录到任何动作",
            )
        return CheckOutcome(
            check=chk,
            status=CHECK_UNKNOWN if missing else CHECK_MISS,
            count=0, actual=[], present=not missing,
            detail="这一段没有记录到这类动作" if not missing
                   else "这一段没读到这类动作",
        )
    ok = count >= need
    if chk.max_count is not None and count > chk.max_count:
        ok = False
    return CheckOutcome(
        check=chk,
        status=CHECK_OK if ok else CHECK_MISS,
        count=count,
        actual=[str(a.value) for a in hits],
        present=True,
        detail=f"窗口内做了 {count} 次（要求 ≥{need}）",
    )


def _evaluate_scalar_expect(pattern: Pattern, chk: Check, applicable, ctx):
    """局面类判定项：节点窗口内任一回合满足即算达成。"""
    values = []
    prov = PROV_ANY
    unknown = False
    for entry in applicable:
        rnd = entry[0]
        value, present = ctx.resolve_in_round(rnd)(chk.fact)
        if not present:
            unknown = True
            continue
        prov = _fact_source(rnd, chk.fact)
        values.append(value)
        if chk.matches_value(value):
            return CheckOutcome(
                check=chk, status=CHECK_OK, count=1, actual=value,
                present=True, provenance=prov,
                detail="节点窗口内满足",
            )
    if not values:
        return CheckOutcome(
            check=chk, status=CHECK_UNKNOWN, count=0, actual=None,
            present=False, provenance=prov,
            detail="这一局没采到这个事实",
        )
    if not _prov_satisfies(chk.require, prov):
        return CheckOutcome(
            check=chk, status=CHECK_UNKNOWN, count=0,
            actual=values[-1], present=True, provenance=prov,
            detail=(
                f"这个结论要求「{PROVENANCE_LABELS.get(chk.require, chk.require)}」"
                f"的证据，实际是「{PROVENANCE_LABELS.get(prov, prov)}」"
            ),
        )
    return CheckOutcome(
        check=chk, status=CHECK_MISS, count=0, actual=values[-1],
        present=True, provenance=prov,
        detail="节点窗口内不满足",
    )


def _resolve_corrective(pattern: Pattern, branch_no: int):
    """运行时决定这条模式能不能拿来纠正玩家。"""
    mode = pattern.corrective.mode
    if mode == CORRECTIVE_NEVER:
        return False, pattern.corrective.reason or "这条模式不用于纠正玩家"
    if mode == CORRECTIVE_ALWAYS:
        # 恒可纠正。指认了分支就说明得说清「按哪一支纠正」——
        # 原因写进报告，玩家才知道建议是针对哪一支给的。
        if branch_no > 0:
            return True, f"已指认第 {branch_no} 支，建议按这一支给"
        return True, ""
    # CORRECTIVE_ON_BRANCH
    if not pattern.corrective.branches:
        # 没有分支却要等分支 —— 视为可纠正，否则这条永远哑掉。
        return True, ""
    if branch_no > 0:
        return True, f"已指认第 {branch_no} 支，前提成立"
    return False, (
        "攻略在这一节点给了多套打法，前提读不出来，"
        "无法确认该按哪一支纠正"
    )


# --------------------------------------------------------------------------
# 整局判定
# --------------------------------------------------------------------------


def match_patterns(
    patterns: Sequence[Pattern],
    ctx,
    *,
    action_facts: Optional[set] = None,
) -> List[PatternVerdict]:
    """对整局做模式匹配，返回所有判定过的模式。

    结果按「节点时间 → 严重程度 → 模式 id」排序。
    """
    action_facts = action_facts or set()
    out: List[PatternVerdict] = []
    for pat in patterns:
        if pat.status != "active":
            continue
        out.extend(
            evaluate_pattern_rounds(pat, ctx, action_facts=action_facts)
        )

    sev_rank = {"high": 0, "medium": 1, "low": 2}

    def sort_key(v: PatternVerdict):
        # 用节点的起始分钟排序 —— 标签是「第 12 分钟」这种字符串，
        # 拿它排序会把时间线打乱。
        return (
            v.node_from_min,
            sev_rank.get(v.severity, 3),
            v.pattern_id,
        )

    out.sort(key=sort_key)
    return out
