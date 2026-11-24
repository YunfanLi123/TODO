"""前提诊断：回答「为什么这一局一条结论都没有」。

问题从哪来
----------
2026-10-29 的真实 API 回归里，一局跑完是::

    判定单元=3 | 事实=14 | 动作=5 | 触发规则=0 | 结论条数=0

模型**读出了画面**（动作有 ``farm_jungle`` / ``ward_vision`` / ``push_wave``，
位置有「河道」「野区」），但一条规则都没触发。原因是触发前提
``position == "jungle"`` 里的 ``position`` 被塞了「野区」——
分路归一化认不出来 → ``unknown`` → 前提不成立。

这类故障的共同点是**静默**：
没有报错、没有异常，报告只是「空的」，
而「空」有四种完全不同的原因（读不到画面 / 读错了 / 规则没匹配 / 匹配了但被过滤）。
不诊断就只能靠猜。

这个模块做什么
--------------
对每条**没有触发**的规则，说出它卡在哪个前提上：

- ``missing``    —— 这个事实在这一局压根没采到（分路没填、英雄没认出来）；
- ``mismatched`` —— 采到了，但取值与规则要求**矛盾**（赵云是打野，分路却读成中路）。

两种原因对用户的意义完全不同：
``missing`` 是「你还没填」，``mismatched`` 是「你填错了」。
所以必须分开，不能只说「没匹配上」。

什么**不算**卡点
----------------
这一节是 2026-10-30 补的 —— 踩过坑才补上的。

「这条模式不是给你的」与「这条模式坏了」是两回事。混在一起，
诊断就会说谎，而**说谎的诊断比没有诊断更坏**：

* **别的分路的通则 / 别的英雄的模式** —— 它们本来就不适用，
  不该报成卡点。2026-10-29 那次真实回归就吃过这个亏：
  打野玩家看到「分路识别成 jungle，71 条模式因此没生效」——
  那 71 条是**中路**的通则，本就不该触发。
* **让路护栏命中**（``none`` / ``when_none``）—— 规则库在说
  「这句话不是对你说的」，那是设计，不是故障。
* **时刻对不上**（对位不是劣势、大招没就绪）—— 这条模式这一刻
  本来就不适用，是**正常**的，不是故障。

于是真正要报给用户的只剩两件事：

* **事实没采到** → 去补（``missing``）；
* **读到的值与英雄定位矛盾** → 去修（``mismatched``，且只限身份类事实）。

身份类事实只有两个：``position``（分路）与 ``my_hero``（我方英雄）。
它们决定的是「这条是给谁看的」；其余事实决定的是「这一刻适不适用」。

矛盾怎么判
----------
拿「玩家这个英雄按官方定位表该打哪条路」当**独立证据**：

* 读到的分路与英雄定位**一致** → 那些别的分路的模式就是**不是给你的**；
* 读到的分路与英雄定位**矛盾** → 是读错了，该让用户去改；
* 英雄是摇摆位 / 库里没有 / 没认出来 → 说不准，**不指责**。

设计约束
--------
- **只诊断，不判定**。这个模块不改任何判定结果，只解释结果。
- **不猜**。前提不成立就说不成立，不给「可能是因为」。
- **计数而非罗列**。规则库有上千条，逐条列出来没法看；
  按「卡在哪个前提上」聚合成几行，才是能读的信息。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .context import MatchContext
from .dsl import Rule

#: 诊断结果的两种原因。
#:
#: ``missing``    —— 前提事实这一局压根没采到。
#: ``mismatched`` —— 采到了，但取值与规则要求对不上。
REASON_MISSING = "missing"
REASON_MISMATCHED = "mismatched"

#: **身份类事实**：这条模式是「给谁看的」，不是「什么时候适用」。
#:
#: 分路与英雄决定的是「这条模式是不是你的」；其余事实决定的是
#: 「这一刻适不适用」。两者的失败含义**完全不同**：
#:
#: * 身份类取值对不上 → 这条**是给别人写的**
#:   （中路玩家的打野通则），**不是卡点**，更不该告诉用户「你去改」；
#: * 其余事实取值对不上 → 这一刻确实不适用，是**正常**的。
IDENTITY_FACTS = frozenset({"position", "my_hero"})

#: **时间窗类事实**：它们是「什么时候适用」，不是「缺了什么」。
#:
#: v1 把时间窗写在触发条件里（``time_min between [0, 1]``），
#: v2 把它提成了节点。诊断不该因为「第 3 分钟不在这条模式的时间窗里」
#: 就说「局内分钟识别错了」—— 那是这条模式管别的时段去了。
WINDOW_FACTS = frozenset({"time_min"})


@dataclass
class Blocker:
    """一个卡住规则集的前提。"""

    #: 事实名（内核里的规范名）
    fact: str
    #: 事实的中文名（报告里用）
    fact_label: str = ""
    #: 原因
    reason: str = REASON_MISSING
    #: 这一局实际取到的值（``missing`` 时是空串）
    actual: Any = None
    #: 规则要求的值（取若干条规则的并集）
    expected: List[Any] = field(default_factory=list)
    #: 被这条前提卡住的规则条数
    rule_count: int = 0
    #: 被卡住的规则 id（前若干条，便于定位）。
    #: **只是样本** —— 完整的条数看 ``rule_count``。
    rule_ids: List[str] = field(default_factory=list)
    #: 这条卡点是不是「用户该动手修」的。
    #:
    #: ``missing`` 一律是（去补）。
    #: ``mismatched`` 只有**身份类事实**才是 —— 那是读到的值和英雄
    #: 定位互相矛盾，用户改一下就能解锁一批模式。
    #: 其余事实的「对不上」是**这一刻不适用**，说出来只是噪音。
    actionable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact": self.fact,
            "fact_label": self.fact_label or self.fact,
            "reason": self.reason,
            "actual": self.actual,
            "expected": list(self.expected),
            "rule_count": self.rule_count,
            "rule_ids": list(self.rule_ids),
            "actionable": self.actionable,
        }


@dataclass
class Diagnosis:
    """一局对局的「为什么没结论」诊断。"""

    #: 卡住规则集的前提
    blockers: List[Blocker] = field(default_factory=list)
    #: 规则库里完全没碰过的规则条数（前提全成立但判定为空的情况不在这里）
    untouched_rules: int = 0
    #: 诊断覆盖的规则总数
    total_rules: int = 0
    #: 被卡住的规则**总数**（去重）。
    #:
    #: 不能靠 ``blockers[*].rule_ids`` 求和 —— 那是被截断的样本
    #: （每条只留前几个 id 便于定位），数它会得到一个偏小的假数字。
    blocked_rules: int = 0

    @property
    def blocked_rule_count(self) -> int:
        """被前提卡住的规则条数（去重）。"""
        return self.blocked_rules

    @property
    def is_empty_reason(self) -> bool:
        """这一局「没结论」是因为前提缺失/对不上，而不是规则本身没问题。"""
        return bool(self.blockers)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blockers": [b.to_dict() for b in self.blockers],
            "blocked_rule_count": self.blocked_rule_count,
            "untouched_rules": self.untouched_rules,
            "total_rules": self.total_rules,
            "is_empty_reason": self.is_empty_reason,
        }

    def note(self, adapter: Any = None, max_items: int = 3) -> str:
        """一句话说明「为什么这一局没结论」。

        为什么要有这句人话
        ----------------
        诊断的结构化数据在 JSON 里，但**页面上不能只有一张空表** ——
        用户看到「0 条结论」时最需要知道的是「缺什么、怎么补」。
        空报告最常见的原因不是「打得没问题」，是**前提没采到或对不上**。

        两类卡点说得不一样，混成一句就废了：

        * ``missing``    —— 「我方英雄没认出来」→ 该**去补**；
        * ``mismatched`` —— 「分路读成中路，但赵云是打野」→ 该**去修**。

        只说**该动手修的**（``actionable``）：时刻对不上不是故障，
        摆到页面上只会让人以为自己漏了事。

        卡片与模板
        ----------
        页面模板是**冻结**的（前端不改），所以这句话放进页面**已经在渲染**的
        说明位（``version_note``），而不是新加一个区块 ——
        结论照样看得见，模板一个字都不用动。
        """
        actionable = [b for b in self.blockers if b.actionable]
        if not actionable:
            return ""
        parts: List[str] = []
        for b in actionable[:max_items]:
            if not b.rule_count:
                continue
            label = b.fact_label or b.fact
            if b.reason == REASON_MISSING:
                parts.append(f"{label}没采到，{b.rule_count} 条模式因此没生效")
            else:
                actual = b.actual
                parts.append(
                    f"{label}识别成「{actual}」，与模式库要求的对不上，"
                    f"{b.rule_count} 条模式因此没生效"
                )
        if not parts:
            return ""
        text = "；".join(parts)
        if len(actionable) > max_items:
            text += f"；另有 {len(actionable) - max_items} 项卡点见诊断"
        return text + "。"

    def missing_only_note(self, max_items: int = 3) -> str:
        """只说「没采到」的那些卡点。

        什么时候用这个
        --------------
        「对不上」有一类**不是问题**：玩家的分路是 ``mid``，那 300 条打野模式
        自然不适用 —— 它们不是被卡住，是**不属于这个玩家**。
        把这类算成卡点，报告上就会说「分路识别成 mid，300 条模式因此没生效」，
        吓人而且是错的。

        所以给页面用的那句话默认**只说没采到的**（那一定是要去补的）。
        「对不上」只有在能确定它真的是问题时才说。
        """
        return Diagnosis(
            blockers=[
                b for b in self.blockers
                if b.reason == REASON_MISSING and b.actionable
            ],
        ).note(max_items=max_items)


# --------------------------------------------------------------------------
# 前提判定：按操作符判，不按取值集合硬碰
# --------------------------------------------------------------------------


def _premise_holds(cond: Any, value: Any, present: bool) -> bool:
    """这个前提**成立**吗？

    用条件自己的求值方法（v1 ``Condition.evaluate`` / v2 ``Check.matches_value``），
    不自己拿取值集合硬碰 —— 硬碰会把否定条件判反：

    * ``my_hero not_in [甲, 乙]`` 在 ``my_hero == 甲`` 时**不成立**；
    * ``matchup != [一串值]`` 在 ``matchup == 一串值`` 时**成立**。

    早先的 ``_intersects`` 只认「取值在不在集合里」，
    于是 ``!=`` / ``not_in`` 全判反了 —— 而 ``none`` 护栏里恰好有这两种。
    """
    evaluate = getattr(cond, "evaluate", None)
    if callable(evaluate):
        return bool(evaluate(value, present))
    matches = getattr(cond, "matches_value", None)
    if callable(matches):
        return bool(matches(value))
    return _intersects(value, _expected_values(cond))


def _expected_values(cond: Any) -> List[Any]:
    """一个条件要求的取值集合。"""
    value = getattr(cond, "value", None)
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _fact_is_blank(value: Any, present: bool) -> bool:
    """这个事实这一局**没采到**吗？

    ``unknown`` 与**空串**都算「没填 / 认不出」——
    它们不是一个具体取值，不能当成「你打的是别的路」。
    """
    return (not present) or value is None or value == "unknown" or value == ""


def _intersects(actual: Any, expected: List[Any]) -> bool:
    """实际取值与期望取值集合是否**有交集**。

    只在没有条件自己的求值方法时兜底用 —— 主路径是
    :func:`_premise_holds`，它认得每种操作符。
    """
    if not expected:
        return True
    if actual in expected:
        return True
    return False


def _describe_fact(adapter: Any, fact: str) -> str:
    try:
        return adapter.describe_fact(fact)
    except Exception:                                  # noqa: BLE001
        return fact


# --------------------------------------------------------------------------
# 诊断的统一视图
# --------------------------------------------------------------------------


@dataclass
class _Item:
    """诊断用的「一条规则 / 一条模式」的统一视图。

    v1 的 ``Rule`` 与 v2 的 ``Pattern`` 结构不同，但诊断要问的问题一样：
    「这条卡在哪个前提上」。所以先把两边压成同一种形状，再走同一套判定 ——
    两套口径各写一遍，迟早会漂。
    """

    #: 规则 / 模式 id
    id: str
    #: 前提条件（v1 的 ``Condition`` / v2 的 ``Check``）
    premises: List[Any] = field(default_factory=list)
    #: 让路护栏（v1 的 ``trigger.none`` / v2 的 ``when_none``）
    guard: List[Any] = field(default_factory=list)
    #: **必需事实**（v2 的 ``applies_to.requires``）：只要它在就行，
    #: 缺了就点名 —— 英雄就属于这一类。
    requires: Tuple[str, ...] = ()
    #: 英雄白名单（v2 的 ``applies_to.heroes``）：空 = 该分路的任何英雄都行。
    heroes: Tuple[str, ...] = ()


def _collect_fact_values(
    items: List[_Item],
    ctx: MatchContext,
) -> Dict[str, Tuple[Any, bool]]:
    """事实 -> 这一局的取值。

    取**第一个读到该事实的回合**，不是第一个回合。
    早先取「第一回合的快照」，于是「第 3 分钟才采到的事实」
    被判成「整局没采到」—— 诊断会说谎，而报告上只是一句
    「对位优劣没采到」，用户去补一个自己明明填过的东西。
    """
    wanted: Set[str] = set()
    for item in items:
        for cond in list(item.premises) + list(item.guard):
            fact = getattr(cond, "fact", "")
            if fact:
                wanted.add(fact)
        wanted.update(item.requires)
        if item.heroes:
            wanted.add("my_hero")
    wanted.discard("")
    # ``my_hero`` 始终要取：它可能是**前提之外**的独立证据 ——
    # 「你玩的赵云是打野，但分路读成了中路」这个矛盾，只有拿英雄定位
    # 表去比对才看得出来。没有它，读错分路会被当成「你打的是别的路」。
    wanted.add("my_hero")

    fact_values: Dict[str, Tuple[Any, bool]] = {}
    for rnd in ctx.iter_rounds():
        if len(fact_values) == len(wanted):
            break
        resolve = ctx.resolve_in_round(rnd)
        for fact in list(wanted):
            if fact in fact_values:
                continue
            value, present = resolve(fact)
            if present:
                fact_values[fact] = (value, present)
    for fact in wanted:
        fact_values.setdefault(fact, (None, False))
    return fact_values


def _guardrail_excluded(
    guard_conds: List[Any],
    fact_values: Dict[str, Tuple[Any, bool]],
) -> bool:
    """让路护栏命中了吗？（``none`` / ``when_none`` 条件成立 = 这条按设计不适用于你。）

    护栏命中**不是故障** —— 它是规则库在说「这句话不是对你说的」。
    所以这类规则不进卡点统计。

    读不到事实时**不判违反** —— 与引擎一致（引擎里护栏未知按通过算）。
    """
    for cond in guard_conds:
        value, present = fact_values.get(getattr(cond, "fact", ""), (None, False))
        if _fact_is_blank(value, present):
            continue
        if _premise_holds(cond, value, present):
            return True
    return False


def _hero_not_for_this_player(
    item: _Item,
    fact_values: Dict[str, Tuple[Any, bool]],
) -> bool:
    """这条是不是**别的英雄**的模式？

    英雄认出来了、且不是白名单里的那个 → 是别人的模式，不该算卡点。
    英雄**没认出来** → 说不准是不是给你的，留给 ``requires`` 去报
    「我方英雄没采到」（那是要去补的）。
    """
    if not item.heroes:
        return False
    value, present = fact_values.get("my_hero", (None, False))
    if _fact_is_blank(value, present):
        return False
    return str(value) not in item.heroes


def _hero_lane(
    fact_values: Dict[str, Tuple[Any, bool]],
    adapter: Any,
) -> str:
    """玩家这个英雄按官方定位表该打哪条路（推不出就空串）。"""
    value, present = fact_values.get("my_hero", (None, False))
    if _fact_is_blank(value, present):
        return ""
    infer = getattr(adapter, "infer_position_from_hero", None)
    if not callable(infer):
        return ""
    try:
        return str(infer(str(value)) or "").strip()
    except Exception:                                  # noqa: BLE001
        return ""


def _identity_kind(
    premises: List[Any],
    fact_values: Dict[str, Tuple[Any, bool]],
    adapter: Any,
) -> str:
    """身份类前提判成哪一种。

    返回：

    * ``none``         —— 身份前提都成立（或没采到，交给 missing 去报）；
    * ``other_player`` —— 这条是**给别人写的**（别人分路 / 别人英雄）；
    * ``contradiction`` —— 读到的值与英雄定位**矛盾**（读错了，要去修）。

    「给别人写的」优先于「矛盾」：一条给赵云写的模式，
    哪怕分路读错了，也不该算到玩家头上 —— 先看它是不是你的。
    """
    kinds: List[str] = []
    for cond in premises:
        fact = getattr(cond, "fact", "")
        if fact not in IDENTITY_FACTS:
            continue
        value, present = fact_values.get(fact, (None, False))
        if _fact_is_blank(value, present):
            continue        # 身份没采到 → 说不准是不是给你写的 → 当卡点（要去补）
        if _premise_holds(cond, value, present):
            continue        # 成立
        if fact == "position":
            lane = _hero_lane(fact_values, adapter)
            if lane and str(value) != lane:
                # 读到的分路与英雄定位矛盾 → 是读错了。
                kinds.append("contradiction")
                continue
        kinds.append("other_player")
    if "other_player" in kinds:
        return "other_player"
    if "contradiction" in kinds:
        return "contradiction"
    return "none"


def _diagnose_items(
    items: List[_Item],
    ctx: MatchContext,
    adapter: Any,
    max_rule_ids_per_blocker: int,
) -> Diagnosis:
    """诊断的核心：v1 规则与 v2 模式走同一套判定。"""
    fact_values = _collect_fact_values(items, ctx)

    # 「不是给这个玩家的」不进卡点统计 ——
    # 别人分路 / 别人英雄的模式，本来就不该触发，那不是故障。
    scoped: List[_Item] = []
    for item in items:
        if _guardrail_excluded(item.guard, fact_values):
            continue
        if _hero_not_for_this_player(item, fact_values):
            continue
        if _identity_kind(item.premises, fact_values, adapter) == "other_player":
            continue
        scoped.append(item)

    # 聚合：fact -> (缺失的规则 / 对不上的规则 / 期望取值 / 实际取值)
    agg: Dict[str, Dict[str, Any]] = {}

    def slot_for(fact: str) -> Dict[str, Any]:
        return agg.setdefault(fact, {
            "missing": [], "mismatched": [],
            "expected": [], "actual": None,
        })

    for item in scoped:
        for cond in item.premises:
            fact = getattr(cond, "fact", "")
            if not fact or fact in WINDOW_FACTS:
                continue        # 时间窗是「什么时候适用」，不是「缺了什么」
            value, present = fact_values.get(fact, (None, False))
            if _fact_is_blank(value, present):
                reason = REASON_MISSING
            elif _premise_holds(cond, value, present):
                continue        # 这个前提是成立的，不是卡点
            else:
                reason = REASON_MISMATCHED
            slot = slot_for(fact)
            slot[reason].append(item.id)
            if reason == REASON_MISMATCHED:
                slot["actual"] = value
            for v in _expected_values(cond):
                if v not in slot["expected"]:
                    slot["expected"].append(v)

        for fact in item.requires:
            value, present = fact_values.get(fact, (None, False))
            if not _fact_is_blank(value, present):
                continue        # 在 → 不是卡点
            slot_for(fact)["missing"].append(item.id)

    blockers: List[Blocker] = []
    blocked_ids: Set[str] = set()
    for fact, slot in agg.items():
        missing_ids = list(dict.fromkeys(slot["missing"]))
        mismatched_ids = list(dict.fromkeys(slot["mismatched"]))
        if not missing_ids and not mismatched_ids:
            continue
        # 同一个事实既有缺失又有对不上时，以「缺失」为准 ——
        # 缺失是更根本的原因（补上它才谈得上别的）。
        reason = REASON_MISSING if missing_ids else REASON_MISMATCHED
        ids = missing_ids + mismatched_ids
        blocked_ids.update(ids)
        blockers.append(
            Blocker(
                fact=fact,
                fact_label=_describe_fact(adapter, fact),
                reason=reason,
                actual=slot["actual"] if reason == REASON_MISMATCHED else None,
                expected=list(slot["expected"]),
                rule_count=len(ids),
                rule_ids=ids[:max_rule_ids_per_blocker],
                # 时刻对不上不是故障 —— 只有身份类的「对不上」才该让人去改。
                actionable=(reason == REASON_MISSING or fact in IDENTITY_FACTS),
            )
        )

    # 卡住规则最多的排前面 —— 那是最该先解决的那个。
    blockers.sort(key=lambda b: (-b.rule_count, b.fact))
    blocked_total = len(blocked_ids)
    return Diagnosis(
        blockers=blockers,
        untouched_rules=len(scoped) - blocked_total,
        total_rules=len(scoped),
        blocked_rules=blocked_total,
    )


# --------------------------------------------------------------------------
# 对外接口
# --------------------------------------------------------------------------


def diagnose(
    rules: List[Rule],
    ctx: MatchContext,
    adapter: Any,
    triggered_rule_ids: Optional[set] = None,
    max_rule_ids_per_blocker: int = 6,
) -> Diagnosis:
    """诊断「为什么这些规则没触发」（v1 规则库口径）。

    只看**没触发**的规则（``triggered_rule_ids`` 给出已触发的那些）。
    对每条没触发的规则，逐个前提检查：

    - 前提事实在这一局**没采到** → ``missing``；
    - 采到了但取值**与规则要求矛盾** → ``mismatched``。

    同一个前提被多条规则卡住时合并成一条 ``Blocker``，计数累加 ——
    「分路没填」卡住了 1680 条规则，说一次比说 1680 次有用。

    实现说明
    --------
    这里**不重跑一遍触发求值**（那会把不成立的规则也算一遍，浪费且容易与
    判定层算出不同的结论）。而是用一个更保守的办法：
    只检查那些**值明显不满足**的前提 —— 即事实存在、且它的取值
    与规则要求**完全不相交**。事实缺失的情况靠
    ``ctx`` 的 ``missing_facts`` 判定。

    这样得到的诊断是**充分条件**（卡住了就一定没触发），
    不是穷举（可能漏报「条件其实成立但因为别的原因没触发」）。
    宁可少说，不可说错。
    """
    fired = set(triggered_rule_ids or ())
    items: List[_Item] = []
    for rule in rules:
        if getattr(rule, "status", "active") != "active":
            continue
        if rule.id in fired:
            continue
        items.append(_Item(
            id=rule.id,
            premises=list(rule.trigger.all) + list(rule.trigger.any),
            guard=list(rule.trigger.none),
        ))
    return _diagnose_items(items, ctx, adapter, max_rule_ids_per_blocker)


def diagnose_patterns(
    patterns: List[Any],
    ctx: MatchContext,
    adapter: Any,
    triggered_pattern_ids: Optional[set] = None,
    max_rule_ids_per_blocker: int = 6,
) -> Diagnosis:
    """诊断「为什么这些**模式**没触发」（v2 模式库口径）。

    与 :func:`diagnose` 的区别只在**前提从哪来**：

    - v1 的前提在 ``rule.trigger`` 里；
    - v2 的前提在 ``pattern.when`` / ``when_any`` 里，
      另有一类**必需事实**在 ``pattern.applies_to.requires`` ——
      英雄就属于这一类：它不是「筛选条件」，而是「不知道你在玩谁就无从判断」。

    这两类都要诊断。否则会出现一种很难解释的情况：
    规则一条都不触发，而诊断说「没有卡点」—— 因为卡点藏在 requires 里。

    判定口径与 v1 **完全一致**（走同一个 :func:`_diagnose_items`）：
    别的分路 / 别的英雄的模式不算卡点，护栏命中不算卡点，
    时刻对不上不算卡点。
    """
    fired = set(triggered_pattern_ids or ())
    items: List[_Item] = []
    for pat in patterns:
        if getattr(pat, "status", "active") != "active":
            continue
        if pat.id in fired:
            continue
        items.append(_Item(
            id=pat.id,
            premises=list(pat.when) + list(pat.when_any),
            guard=list(pat.when_none),
            requires=tuple(getattr(pat.applies_to, "requires", ()) or ()),
            heroes=tuple(getattr(pat.applies_to, "heroes", ()) or ()),
        ))
    return _diagnose_items(items, ctx, adapter, max_rule_ids_per_blocker)
