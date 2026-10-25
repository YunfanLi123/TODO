"""游戏适配器基类。

立项设计的分层意义：「新增一款游戏只需要写规则层和分析映射，不动内核」。
本文件定义了新增一款游戏需要实现的最小接口。

新增游戏时只需三步：
  1. 继承 :class:`GameAdapter`，声明 ``game_id`` 和 ``FACT_NAMES``
  2. 实现 :meth:`GameAdapter.normalize`，把原始对局数据翻译成
     :class:`~app.core.context.MatchContext`（回合 + 事实 + 动作）
  3. 在 :data:`REGISTRY` 注册，并提供该游戏的种子规则库

分析内核（matcher / attribution / curve）完全不需要任何改动。
"""

from __future__ import annotations

import abc
from typing import Any, Callable, Dict, List, Optional

from ..core.context import ActionRecord, MatchContext, RoundRecord


#: 通用事实名 —— 三款游戏都能表达。
#: 规则库应优先复用这些事实，让同一条运营逻辑能在不同游戏间迁移。
#:
#: 注意这里只列「各适配器实际注册了的事实」。曾经存在两个声明了却没有
#: 任何适配器实现的条目（decision_point / discipline），它们会误导后续
#: 适配器作者以为系统已经支持 —— 不可用的事实比不存在的事实更危险，
#: 因为规则会顺利通过校验，然后在运行时永远取不到值。
COMMON_FACTS = {
    "stage",             # 阶段序号（自走棋）或时间片编号（MOBA）
    "is_decision_point", # 关键决策时刻标记
    "pressure",          # 局面压力等级：low / mid / high
}


class GameAdapter(abc.ABC):
    """一款游戏的适配器。"""

    #: 稳定标识，全系统唯一（写入数据库、作为规则前缀）
    game_id: str = ""
    #: 中文展示名
    display_name: str = ""
    #: 本游戏可用的全部事实名。规则只能引用这里注册过的事实 ——
    #: 这是「规则必须可判定」的类型系统强制点。
    fact_names: List[str] = list(COMMON_FACTS)
    #: 本游戏中「能承载玩家实际操作」的事实名。
    #:
    #: 这是一个容易踩坑的区分。规则的 expect 字段判定的是「玩家实际做了
    #: 什么动作」，而 matcher 对 expect 的求值方式是**在该回合的动作列表里
    #: 查找同类动作**。如果把 expect 挂在纯观测事实上（例如 pressure），
    #: 那个事实永远不会作为动作出现，于是每局每回合都判成「该做而没做」，
    #: 规则恒为偏离且毫无信号。准入筛选会强制这条约束。
    action_fact_names: List[str] = []
    #: **BP（选人禁用）阶段**能承载玩家实际操作的事实名。
    #:
    #: BP 在比赛之前，问的是「这一手该禁谁、该选谁」——
    #: 它既没有局内分钟，也没有刷野开团那套动作。塞进 ``action_fact_names``
    #: 的话，局内标注页会多出一栏「BP 动作」，玩家在局内根本用不上。
    #: 所以单列一组：准入校验认两组（``known_action_facts``），
    #: 局内标注页只认第一组。
    bp_action_fact_names: List[str] = []
    #: 事实标签的中文名，报告里展示用
    fact_labels: Dict[str, str] = {}
    #: 动作值的中文名
    action_labels: Dict[str, str] = {}
    #: 动作事实 -> 该类动作的取值 -> 中文名。
    #:
    #: 必须**按动作事实分组**，不能只有一张平表。标注页要为每一类动作
    #: 单独列出可选取值，平表做不到 —— 早先把共享词表（自走棋时代的
    #: take_baron / push_tower / adjust…）和平表一起塞进去，于是王者的
    #: 标注页上出现了「开大龙」「调整站位」这种**这个游戏根本没有的动作**，
    #: 还显示成英文原始码，玩家根本不知道勾的是啥。
    #:
    #: 留空表示「这个游戏没给出分组表」，调用方退回共享词表。
    action_values_by_fact: Dict[str, Dict[str, str]] = {}
    #: 标注页要**问玩家**的数字型事实。
    #:
    #: 刻意不用「所有带数字的事实」—— 那样会把派生的、对局级的、
    #: 字符串型的事实也塞进表单：
    #:
    #:   局势阶段  由局内时间推导，玩家填不了也不该填
    #:   分路      在标注页顶部选了，全局生效
    #:   我方英雄 / 对位优劣   同样是顶部选的那两项
    #:   局势压力 / 关键决策时刻   由经济差与时间推导
    #:
    #: 它们不是「不问就没事」，是**问了会给错答案**：数字输入框里
    #: 填「前期」是填不进去的，填成数字又会被适配器覆盖掉 ——
    #: 玩家以为自己改了什么，其实没有。
    annotate_numeric_facts: List[str] = []
    #: 标注页要**问玩家**的布尔型事实（渲染成勾选框，不是数字框）。
    #:
    #: 为什么单列一份：有些事实是「是 / 否」，用数字框问等于让玩家
    #: 在「1 / 0 / 留空」之间猜 —— 填错就等于没填。
    #: 典型例子是王者规则里的「我方核心大招是否就绪」：
    #: 它是「控制链就绪」「开团条件齐」这类分支的唯一可观察近似，
    #: 问不到它，那几条模式就永远不触发。
    annotate_bool_facts: List[str] = []
    #: 分路的规范取值（打野 / 中路 / …）。只有 MOBA 这类有角色分工的游戏
    #: 需要填；自走棋留空，标注页也就不会出现分路下拉。
    #:
    #: 它存在的理由是分路必须由用户在录对局时选 —— 视频里看不出自己在打什么
    #: 位置。规则一旦带 position 条件而这局没填，规则会全部不触发，报告是空的。
    position_values: List[str] = []
    #: 分路取值的中文名
    position_labels: Dict[str, str] = {}
    #: 分路的展示顺序（不含 unknown）。报告按它展开，别处不要另排一套。
    position_order: List[str] = []
    #: 局势阶段的中文名。MOBA 有「前中后期」这种时间分片，自走棋没有。
    phase_labels: Dict[str, str] = {}
    #: 阶段 -> 时间区间（分钟）。MOBA 有，自走棋留空。
    #:
    #: 采集层按它决定「每个阶段抽几张帧、抽在第几分钟」。抽帧落点必须与
    #: 规则库判定用的阶段边界**完全一致** —— 否则会出现「这一帧标着中期，
    #: 内核却按前期判」的错位。那种错在报告里完全看不出来：结论只是少了一条。
    phase_time_windows: Dict[str, List[int]] = {}
    #: 这个游戏的**操作单位**怎么叫。
    #:
    #: 自走棋真的有「一回合」；王者荣耀的一局是连续的一整场，没有回合，
    #: 切出来的只是**时间片**。报告里说「这一回合」会让玩家去找回合。
    segment_unit: str = "回合"
    #: 非动作事实的取值中文名：``事实 -> 取值 -> 中文名``。
    #:
    #: 为什么动作之外也要一张表：报告里「当时局面」是把一批事实拼成一句话，
    #: 而 ``describe_action`` 只认动作词表 —— 分路、对位优劣、局势阶段这些
    #: 走到那儿就原样打印内部代码，报告上就会出现「分路 jungle」
    #: 「对位优劣 unknown」这种半吊子英文，用户看不懂。
    fact_value_labels: Dict[str, Dict[str, str]] = {}
    #: 回合时间线的展示单位
    round_unit: str = "回合"
    #: 该游戏当前使用的默认规则库版本。上传 / 攻略页面据此预填 ——
    #: 版本填错会直接得到空报告，而那种错误很难一眼看出来。
    default_version: str = ""

    def __init__(self) -> None:
        self._resolvers: Dict[str, Callable] = {}
        self._register_fact_resolvers()

    def normalize_position(self, raw: Any) -> str:
        """把玩家填的分路规整成规范值。

        基类**不猜**：认不出来就原样返回，由调用方决定怎么显示。
        有分路概念的游戏（王者荣耀）会覆写它，把「打野位」「野核」这类
        写法统一到 ``jungle``。

        页面要用它给对局起名 —— 库里存的可能是「发育路」也可能是
        ``carry``，两种写法不归一的话，同一条分路会被拆成两栏。
        """
        return str(raw or "")

    # -- 子类实现 -----------------------------------------------------------

    @abc.abstractmethod
    def normalize(self, raw: Dict[str, Any]) -> MatchContext:
        """把原始对局数据翻译成内核可消费的对局上下文。"""

    def _register_fact_resolvers(self) -> None:
        """注册派生事实的计算逻辑。

        默认不需要：适配器在 :meth:`normalize` 里直接把事实算好写进
        ``RoundRecord.facts``。只有需要「按需跨原始数据计算」的复杂事实
        才在这里注册解析器。
        """

    # -- 对外接口 -----------------------------------------------------------

    def build_context(self, raw: Dict[str, Any]) -> MatchContext:
        ctx = self.normalize(raw)
        ctx.game = self.game_id
        ctx.fact_resolvers = self._resolvers
        ctx.precompute()
        return ctx

    def known_facts(self) -> set:
        return set(self.fact_names)

    def known_action_facts(self) -> set:
        """能承载玩家实际操作的全部事实（局内 + BP）。

        准入校验用这份：BP 规则的 expect 挂在 ``bp_action`` 上，
        少了它，BP 规则会整批被拒。
        局内标注页只认 :attr:`action_fact_names`。
        """
        return set(self.action_fact_names) | set(self.bp_action_fact_names)

    def describe_fact(self, fact: str) -> str:
        return self.fact_labels.get(fact, fact)

    def describe_action(self, value: Any) -> str:
        return self.action_labels.get(str(value), str(value))

    def describe_value(self, fact: str, value: Any) -> str:
        """把一个事实的取值说成人话。

        动作事实走动作词表；非动作事实走 ``fact_value_labels``；
        两张表都没有就原样返回 —— 但**不返回空串**：
        空取值是「没填」，由调用方决定要不要印。

        布尔值说成「是 / 否」：报告是中文的，句子里蹦出 ``True``
        就像译文没翻干净 —— 而这条事实往往正是一句话的关键
        （「关键决策回合 是」比「关键决策回合 True」读得懂）。
        """
        if isinstance(value, bool):
            return "是" if value else "否"
        if fact in self.known_action_facts():
            return self.action_labels.get(str(value), str(value))
        labels = self.fact_value_labels.get(fact) or {}
        return labels.get(str(value), str(value))

    def validate_round(self, rnd: RoundRecord) -> List[str]:
        """校验回合数据质量，返回问题列表。"""
        problems: List[str] = []
        for action in rnd.actions:
            if action.kind not in self.fact_names:
                problems.append(
                    f"{rnd.label}: 动作类型 {action.kind!r} 未在本游戏注册"
                )
        unknown = set(rnd.facts) - set(self.fact_names)
        if unknown:
            problems.append(
                f"{rnd.label}: 事实 {sorted(unknown)} 未在本游戏注册"
            )
        return problems


#: 适配器注册表。新增游戏在这里登记即可。
REGISTRY: Dict[str, GameAdapter] = {}


def register(adapter: GameAdapter) -> GameAdapter:
    REGISTRY[adapter.game_id] = adapter
    return adapter


def get_adapter(game_id: str) -> GameAdapter:
    if game_id not in REGISTRY:
        raise KeyError(
            f"未注册的游戏 {game_id!r}，已注册：{sorted(REGISTRY)}"
        )
    return REGISTRY[game_id]


def registered_games() -> List[Dict[str, str]]:
    return [
        {
            "game_id": a.game_id,
            "display_name": a.display_name,
            "fact_count": len(a.fact_names),
        }
        for a in REGISTRY.values()
    ]
