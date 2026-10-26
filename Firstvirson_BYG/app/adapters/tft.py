"""云顶之弈（Teamfight Tactics）适配器。

分层约定：内核（matcher / attribution）不认识金币和连胜，它只认
:class:`~app.core.context.RoundRecord` 里的「事实字典」。把云顶一局对局翻译成
事实字典，是本文件唯一的职责。

关于共用逻辑的存放位置：云顶与金铲铲的**解析流程完全一致**，差异只在
:data:`AUTOCHESS_FACTS` 之外的几个数字口径（存钱上限、人口里程碑、连胜奖励
阈值）。这些口径由 :class:`AutoChessProfile` 承载，所以共用函数放在本文件里由
金铲铲导入 —— 两款游戏共用同一段流程，反而更容易在改一处时发现另一处没跟上。
注意：本文件导出的 :data:`TFT_ADAPTER` 与金铲铲导出的适配器是两个独立实例，
事实集也各自独立，不存在共用同一个对象的可能。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

from ..core.context import ActionRecord, MatchContext, RoundRecord
from .base import GameAdapter, register

# --------------------------------------------------------------------------
# 自走棋共用的事实集与口径
# --------------------------------------------------------------------------

#: 自走棋可判定事实集。规则的 trigger / expect 只能引用这里的名字 ——
#: 这是「规则必须完全可判定」在类型系统层面的强制点。
#: 与云顶共享同一套事实名，但两款游戏的 fact_names 是各自独立的列表对象。
AUTOCHESS_FACT_NAMES: List[str] = [
    "stage",                 # 阶段序号，如 4（对应 4-1）
    "phase",                 # 小阶段，如 1（对应 4-1）
    "health",                # 剩余生命值
    "level",                 # 人口等级
    "gold",                  # 当前金币
    "income",                # 本回合可支配收入（含利息与连胜/连败奖励）
    "streak",                # 连胜(+) / 连败(-) 场次
    "econ_action",           # 动作事实：经济运营选择
    "positioning_action",    # 动作事实：站位选择
    "item_count",            # 主 C 身上的装备件数
    "has_main_carry",        # 主 C 是否成型
    "is_decision_point",     # 是否关键决策回合
    "pressure",              # 派生事实：局面压力 low/mid/high
    "round_tempo",           # 派生事实：人口节奏 fast/normal/slow
]

#: 事实的中文名。报告层用它把「gold: 12」写成「金币：12」。
AUTOCHESS_FACT_LABELS: Dict[str, str] = {
    "stage": "阶段",
    "phase": "小阶段",
    "health": "剩余生命值",
    "level": "人口等级",
    "gold": "金币",
    "income": "本回合收入",
    "streak": "连胜/连败场次",
    "econ_action": "经济操作",
    "positioning_action": "站位操作",
    "item_count": "主C装备件数",
    "has_main_carry": "主C成型",
    "is_decision_point": "关键决策回合",
    "pressure": "局面压力",
    "round_tempo": "人口节奏",
}

#: 动作取值的中文名。取值集合必须被规则库当作闭集使用 —— 新增取值时
#: 同步扩充，否则老规则会静默漏判。
#:
#: 按动作事实分组：标注页要为每一类动作单独列出可选取值。
AUTOCHESS_ACTION_VALUES: Dict[str, Dict[str, str]] = {
    "econ_action": {
        "save": "存钱吃利息",
        "level_up": "升人口",
        "roll": "搜牌锁血",
        "upgrade_carry": "强化主C",
        "all_in": "梭哈追三星",
        "sell_core": "卖核心补经济",
    },
    "positioning_action": {
        "shift_lineup": "调整站位",
        "protect_carry": "主C缩角保护",
        "counter_lineup": "针对敌方对位",
        "hold_lineup": "保持站位",
    },
}

#: 扁平版：取值 -> 中文名。规则与报告按取值查名字。
AUTOCHESS_ACTION_LABELS: Dict[str, str] = {
    value: label
    for group in AUTOCHESS_ACTION_VALUES.values()
    for value, label in group.items()
}


@dataclass(frozen=True)
class AutoChessProfile:
    """两款自走棋共用的「同一回合该长什么样」的口径。

    把差异显式收敛成可枚举的几个数字，而不是散落在两套分支里：规则库的
    阈值差异必须能追溯到这里，改一个数字就等于改了一条规则的判定依据。
    """

    game_id: str
    display_name: str
    default_version: str
    #: 存钱上限。超过这个数不再吃满利息，再存就是浪费回合。
    gold_cap: int
    #: stage -> 该阶段应该达到的人口。人口低于它记 slow，高于它记 fast。
    level_milestones: Dict[int, int] = field(default_factory=dict)
    #: 连胜/连败达到该场次时收入翻倍（+2 而非 +1）。
    streak_bonus_threshold: int = 5
    #: 血量低于 health_low 记为高压局面，低于 health_mid 记为中压。
    health_low: int = 30
    health_mid: int = 60
    round_unit: str = "回合"


TFT_PROFILE = AutoChessProfile(
    game_id="tft",
    display_name="云顶之弈",
    default_version="13.1",
    gold_cap=50,
    # 云顶的标准人口里程碑：2 阶段 6 人口，4-1 提到 8 人口，6-1 提 9 人口
    level_milestones={1: 4, 2: 6, 3: 6, 4: 8, 5: 8, 6: 9},
    streak_bonus_threshold=5,
    health_low=30,
    health_mid=60,
)


# --------------------------------------------------------------------------
# 取值归一
# --------------------------------------------------------------------------


def _as_int(value: Any) -> Optional[int]:
    """把事实取值收敛成数值。

    采集层（视频识别 / 手工录入）会给到 ``"58"``、``58.0``、``None`` 等各种
    形态。宁可判成「没采到」也不能把字符串塞进数值比较里 —— 那会在
    :class:`~app.core.dsl.Condition` 里抛 TypeError 被吞成 False，
    结论就会变成一条静默的错误归因。
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.lstrip("-").isdigit():
            return int(text)
    return None


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "1", "y"}:
            return True
        if text in {"false", "no", "0", "n"}:
            return False
    return None


def _put_derived(
    facts: Dict[str, Any],
    missing: set,
    name: str,
    inputs: tuple,
    compute: Callable[[], Any],
) -> None:
    """写入一个派生事实，原料不齐就不写。

    原料事实缺失时把派生事实同样标进 ``missing_facts``，这样
    :func:`~app.core.matcher.evaluate_round` 会把该回合判成 ``undetermined``，
    而不是拿半截数据编出一个确定的结论。
    """
    if name in missing:
        return
    if name in facts:
        return
    for key in inputs:
        if key in missing or facts.get(key) is None:
            missing.add(name)
            return
    facts[name] = compute()


# --------------------------------------------------------------------------
# 派生事实
# --------------------------------------------------------------------------


def derive_income(gold: int, streak: int, profile: AutoChessProfile) -> int:
    """本回合收入 = 基础 5 + 利息 + 连胜/连败奖励。

    利息按 ``gold // 10`` 封顶到 ``gold_cap // 10`` —— 超过吃满利息的门槛后
    再存金币不产生额外收入，这是所有「该停手了」结论的计算基础。
    """
    interest = min(profile.gold_cap // 10, max(0, gold) // 10)
    bonus = 0
    chain = abs(streak)
    if chain >= profile.streak_bonus_threshold:
        bonus = 2
    elif chain >= 2:
        bonus = 1
    return 5 + interest + bonus


def derive_round_tempo(level: int, stage: int, profile: AutoChessProfile) -> str:
    """人口节奏：按当前阶段的里程碑判断是否超前 / 标准 / 落后。

    不引入「应该玩什么阵容」这类判断 —— 节奏只看人口这一个可观测量。
    """
    target = profile.level_milestones.get(stage)
    if target is None:
        return "unknown"
    if level >= target + 1:
        return "fast"
    if level >= target:
        return "normal"
    return "slow"


def derive_pressure(
    health: int,
    streak: int,
    profile: AutoChessProfile,
) -> str:
    """局面压力：血量决定基调，连败会把压力向上顶一级。

    只用血量与连胜/连败 —— 这两个是采集层最稳定拿得到的事实，
    任何需要「局面观感」的输入都会让规则退化成主观判断。
    """
    if health <= profile.health_low:
        level = "high"
    elif health <= profile.health_mid:
        level = "mid"
    else:
        level = "low"
    if streak <= -3 and level == "low":
        return "mid"
    return level


def derive_is_decision_point(phase: int, health: int, profile: AutoChessProfile) -> bool:
    """关键决策回合。

    攻略体系把「阶段第一回合」和「血量进入危险线」定义为必须做决定的两个点：
    前者人口 / 装备 / 站位三项决策集中在这一回合，后者不容许再试错。
    """
    if phase == 1:
        return True
    return health <= profile.health_low


# --------------------------------------------------------------------------
# 回合解析
# --------------------------------------------------------------------------


def _build_action(
    payload: Mapping[str, Any],
    action_labels: Mapping[str, str],
) -> ActionRecord:
    """把原始动作翻译成 :class:`ActionRecord`。

    ``confidence`` 必须原样带过去：报告层靠它给判定打「低置信度」标记，
    在这里抹平成 1.0 等于把不确定性藏起来 —— 那正是这个产品要暴露的东西。
    """
    kind = str(payload.get("kind") or "").strip()
    value = payload.get("value")
    label = str(payload.get("label") or "").strip()
    if not label:
        label = action_labels.get(str(value), str(value))
    confidence = _as_float(payload.get("confidence"), default=1.0)
    ts = _as_float(payload.get("ts"), default=0.0)
    return ActionRecord(
        kind=kind,
        value=value,
        label=label,
        ts=ts,
        # 置信度裁剪到 [0, 1]：越界值会让 low_confidence 判定失去意义
        confidence=min(1.0, max(0.0, confidence)),
        # 来源：原始数据里没写就当玩家自己录的（自走棋是手工录入的）。
        source=str(payload.get("source") or "player"),
    )


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def normalize_autochess_round(
    raw_round: Mapping[str, Any],
    profile: AutoChessProfile,
    action_labels: Mapping[str, str],
    known_facts: set,
) -> RoundRecord:
    """把一个原始回合翻译成 :class:`RoundRecord`。两款自走棋共用。"""
    facts: Dict[str, Any] = dict(raw_round.get("facts") or {})
    missing: set = set(raw_round.get("missing_facts") or [])

    stage = _as_int(raw_round.get("stage"))
    if stage is None:
        stage = 0
    phase = _as_int(raw_round.get("phase"))
    if phase is None:
        phase = 1

    # stage / phase 是回合自身的坐标，必然已知，直接作为事实登记，
    # 这样规则可以用统一的方式对它们下条件。
    facts["stage"] = stage
    facts["phase"] = phase

    health = _as_int(facts.get("health"))
    level = _as_int(facts.get("level"))
    gold = _as_int(facts.get("gold"))
    streak = _as_int(facts.get("streak"))

    _put_derived(
        facts, missing, "income", ("gold", "streak"),
        lambda: derive_income(gold, streak, profile),
    )
    _put_derived(
        facts, missing, "round_tempo", ("level",),
        lambda: derive_round_tempo(level, stage, profile),
    )
    _put_derived(
        facts, missing, "pressure", ("health", "streak"),
        lambda: derive_pressure(health, streak, profile),
    )
    _put_derived(
        facts, missing, "is_decision_point", (),
        lambda: derive_is_decision_point(phase, health, profile)
        if health is not None
        else phase == 1,
    )

    actions = [
        _build_action(a, action_labels) for a in (raw_round.get("actions") or [])
    ]

    # 动作事实镜像进 facts：trigger 也需要对动作事实下条件（例如
    # 「本回合不能做 XXX」写在 none 组里）。只镜像已注册的事实，
    # 未注册的动作类型留给 validate_round 报错，不污染事实字典。
    for act in actions:
        if not act.kind or act.kind not in known_facts or act.kind in missing:
            continue
        # 取回合内最后一次 —— 动作事实镜像出来是给 **trigger** 用的
        # （例如「本回合不能回城」写在 none 组里），触发条件看的是这一个值。
        # 期望动作那一侧的口径不同：那里是「这一回合做过符合要求的动作就算
        # 符合」，见 app/core/matcher.py。
        facts[act.kind] = act.value

    return RoundRecord(
        stage=stage,
        phase=phase,
        facts=facts,
        actions=actions,
        missing_facts=missing,
    )


def build_autochess_context(
    raw: Mapping[str, Any],
    profile: AutoChessProfile,
    fact_names: List[str],
    action_labels: Mapping[str, str],
) -> MatchContext:
    """自走棋原始数据 → :class:`MatchContext`。两款自走棋共用。"""
    known_facts = set(fact_names)
    rounds = [
        normalize_autochess_round(r, profile, action_labels, known_facts)
        for r in (raw.get("rounds") or [])
    ]
    return MatchContext(
        match_id=str(raw.get("match_id") or ""),
        game=profile.game_id,
        game_version=str(raw.get("game_version") or profile.default_version),
        rounds=rounds,
        raw=dict(raw),
        player=dict(raw.get("player") or {}),
        outcome=dict(raw.get("outcome") or {}),
    )


# --------------------------------------------------------------------------
# 适配器
# --------------------------------------------------------------------------


class TFTAdapter(GameAdapter):
    """云顶之弈。"""

    game_id = "tft"
    # profile 里已经带着这一版的版本号，直接暴露出来，
    # 免得上传 / 攻略页还要各维护一份版本默认值。
    default_version = TFT_PROFILE.default_version
    display_name = "云顶之弈"
    #: 独立列表对象 —— 与金铲铲不共享同一个 list
    fact_names = list(AUTOCHESS_FACT_NAMES)
    #: 只有这两类事实承载玩家的实际操作，规则的 expect 必须挂在其中之一
    action_fact_names = ["econ_action", "positioning_action"]
    fact_labels = dict(AUTOCHESS_FACT_LABELS)
    action_labels = dict(AUTOCHESS_ACTION_LABELS)
    action_values_by_fact = {
        k: dict(v) for k, v in AUTOCHESS_ACTION_VALUES.items()
    }
    #: 标注页要问玩家的数字型事实：**只问原始量，派生量不问**。
    #:
    #: 本回合收入（income）、人口节奏（round_tempo）、局面压力（pressure）、
    #: 是否关键回合（is_decision_point）全部由下面这五个原始量派生 ——
    #: 问了派生量，填进去的值会被适配器覆盖，等于让玩家白填。
    #:
    #: stage / phase 是回合自身的坐标，由回合边界给出，不问。
    annotate_numeric_facts = [
        "health",     # 剩余生命值
        "level",      # 人口等级
        "gold",       # 当前金币
        "streak",     # 连胜(+) / 连败(-) 场次
        "item_count", # 主 C 身上的装备件数
    ]
    round_unit = "回合"
    profile = TFT_PROFILE

    def normalize(self, raw: Dict[str, Any]) -> MatchContext:
        return build_autochess_context(
            raw,
            self.profile,
            self.fact_names,
            self.action_labels,
        )


#: 模块级句柄：注册表与测试都从这里拿同一个实例
TFT_ADAPTER = register(TFTAdapter())
