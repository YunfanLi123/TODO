"""王者荣耀（Honor of Kings）适配器。

为什么这套事实集和自走棋不能合并：自走棋是**离散回合经济**（每回合一次
买人/升人口决策，金币与人口是可数的离散量，节奏看「阶段」）；MOBA 是
**连续资源差**（金币差、击杀差、防御塔差是随时间连续变化的对团队势，
决策看「分钟」与「地图目标刷新」）。两者的可比量只有
``is_decision_point`` 和 ``pressure``，其余事实名语义完全不同 ——
强行统一只会得到一堆「看起来一样、阈值完全不同」的伪公共事实。

回合坐标约定：``stage`` 取**时间片编号**（每 5 分钟一片），``phase`` 取
该片对应的局势阶段编码（1 前期 / 2 中期 / 3 后期 / 4 决胜期）。
精确到分钟的判断用 ``time_min`` 事实，它同时会出现在报告的局面快照里。
"""

from __future__ import annotations

import json
import os

from typing import Any, Dict, List, Mapping, Optional

from ..core.context import ActionRecord, MatchContext, RoundRecord
from .base import GameAdapter, register

#: 每个时间片的分钟数。时间片编号是 RoundRecord.stage 的来源。
SLICE_MINUTES = 5

#: MOBA 可判定事实集
MOBA_FACT_NAMES: List[str] = [
    "time_min",              # 局内分钟（精确时间）
    "phase",                 # 局势阶段：early / mid / late
    "position",              # 分路：jungle / mid / roam / side / carry / support
    "gold",                  # 本方经济
    "gold_diff",             # 经济差（本方 - 敌方）
    "level",                 # 平均等级
    "tower_diff",            # 防御塔差（本方 - 敌方）
    "kill_diff",             # 击杀差（本方 - 敌方）
    "objective_action",      # 动作事实：地图目标 / 带线选择
    "positioning_action",    # 动作事实：站位 / 视野 / 转线选择
    "teamfight_action",      # 动作事实：团战处理方式
    "has_ult_ready",         # 核心大招是否就绪
    "in_combat",             # 此刻是否在团战里
    "cc",                    # 画面是否确认出现控制链（仅多帧证据时写入）
    "is_decision_point",     # 是否关键决策时刻
    "pressure",              # 派生事实：局势压力 low/mid/high
    "my_hero",               # 自己玩的英雄（分路层之下再细分一层）
    "matchup",               # 对位优劣：advantage / disadvantage / unknown
]

# --------------------------------------------------------------------------
# BP（选人禁用）阶段
#
# BP 在**比赛之前**，没有局内分钟、没有经济差 —— 它问的是
# 「这一步该禁谁、该选谁」。所以它是独立的一套事实与动作，
# 判定也独立（见 ``build_bp_context``）。
#
# 事实的来源有两处：玩家在 BP 记录里勾的（阵容缺件），
# 以及官方英雄定位表算的（输出位、刺客数、摇摆位）。
# --------------------------------------------------------------------------

#: BP 阶段的事实。
BP_FACT_NAMES: List[str] = [
    "bp_phase",                  # 决策环节：禁用 / 一抢 / 二抢 / 三抢 / 末选 / 收尾
    "bp_step",                   # 我方第几次决策
    "bp_hero",                   # 这一步动的英雄
    "opp_picks_before",          # 对方已暴露几个 pick
    "my_picks_before",           # 己方已选几个
    "my_output_count",           # 己方已选里的输出位数
    "my_missing_frontline",      # 己方缺前排承伤
    "my_missing_engage",         # 己方缺开团点
    "my_missing_protect",        # 己方缺反手保护
    "my_missing_damage",         # 己方缺输出
    "opp_missing_frontline",     # 对方缺前排
    "my_other_has_tank_or_support",  # 己方其余位置还有坦克或辅助
    "my_other_all_output",       # 己方其余位置全是输出位
    "my_all_output",             # 己方全是输出位
    "opp_all_output",            # 对方全是输出位
    "my_assassin_count",         # 己方刺客数
    "opp_assassin_count",        # 对方刺客数
    "my_has_flex",               # 己方有摇摆位
    "bp_action",                 # 动作事实：这一步属于哪一类 BP 操作
]

BP_FACT_LABELS: Dict[str, str] = {
    "bp_phase": "决策环节",
    "bp_step": "第几次决策",
    "bp_hero": "这一步的英雄",
    "opp_picks_before": "对方已暴露 pick 数",
    "my_picks_before": "己方已选数",
    "my_output_count": "己方已选里的输出位数",
    "my_missing_frontline": "己方缺前排承伤",
    "my_missing_engage": "己方缺开团点",
    "my_missing_protect": "己方缺反手保护",
    "my_missing_damage": "己方缺输出",
    "opp_missing_frontline": "对方缺前排",
    "my_other_has_tank_or_support": "己方其余位置还有坦克或辅助",
    "my_other_all_output": "己方其余位置全是输出位",
    "my_all_output": "己方全是输出位",
    "opp_all_output": "对方全是输出位",
    "my_assassin_count": "己方刺客数",
    "opp_assassin_count": "对方刺客数",
    "my_has_flex": "己方有摇摆位",
    "bp_action": "BP 操作",
}

#: BP 的动作类别。玩家在 BP 记录里勾的是「这一步属于哪一类」，
#: 不是「我选了个什么英雄」—— 跟局内标注页勾「刷野补经济」是一个路子。
BP_ACTION_LABELS: Dict[str, str] = {
    # 禁用
    "ban_opp_core": "针对对方核心",
    "ban_break_comp": "拆对方体系",
    "ban_press_strong": "压先手必抢",
    "ban_idol": "压绝活",
    "ban_cover_gap": "堵自己的短板",
    "ban_engage": "封对面的开团位",
    "ban_break_heal": "破续航",
    "ban_tank": "禁掉坦边",
    # 选人
    "pick_frontline": "补前排承伤",
    "pick_engage": "补开团点",
    "pick_protect": "补反手保护",
    "pick_damage": "补持续输出与爆发",
    "pick_counter": "针对对方核心",
    "pick_flex": "摇摆位",
    "pick_core_output": "抢核心输出",
    "pick_tempo": "抢前期节奏",
    "pick_defensive": "防守型",
    "pick_ally_main": "给队友的本命",
    "pick_heal_cut": "拿能破续航的输出位",
    "pick_anti_control": "拿解控",
    # 规划与检查
    # plan_* 只留给旧记录显示：「进入 BP」那一环撤了，
    # 已经没有规则再要求这两类动作，新记录里也不该出现。
    "plan_comp": "先定体系（这套阵容靠什么赢）",
    "plan_lane": "先定分路意向",
    "check_comp": "核对阵容四要素",
}

#: BP 的决策环节 -> 中文名。
#:
#: 「进入 BP」那一环**不在这里**：它的三条规则全部依赖段位与预选，
#: 而 BP 记录页与复盘页都不再问这两项 —— 录了也不会有结论，所以不收了。
BP_PHASE_LABELS: Dict[str, str] = {
    "ban_1": "禁用（阶段一）",
    "pick_1": "一抢",
    "pick_2": "二抢",
    "pick_3": "三抢",
    "pick_late_1": "阶段二一抢",
    "pick_last": "阶段二末选",
    "final_check": "阵容完整性检查",
}

#: 旧记录里可能出现、如今已经撤下的环节。
#:
#: 只为「显示得出来」而留：老记录里录过「进入 BP」这一步，
#: 坐标不能变成一串裸编码（``bp_start``）。
BP_PHASE_LEGACY_LABELS: Dict[str, str] = {
    "bp_start": "进入 BP",
}

#: 选人环节（其余是禁用与检查）。
BP_PICK_PHASES = ("pick_1", "pick_2", "pick_3", "pick_late_1", "pick_last")

MOBA_FACT_LABELS: Dict[str, str] = {
    "time_min": "局内时间",
    "phase": "局势阶段",
    "gold": "本方经济",
    "gold_diff": "经济差",
    "level": "平均等级",
    "tower_diff": "防御塔差",
    "kill_diff": "击杀差",
    "objective_action": "目标处理",
    "positioning_action": "站位处理",
    "teamfight_action": "团战处理",
    "has_ult_ready": "大招就绪",
    "in_combat": "是否在团战",
    "cc": "控制事件",
    "is_decision_point": "关键决策时刻",
    "pressure": "局势压力",
    "position": "分路",
    "my_hero": "我方英雄",
    "matchup": "对位优劣",
}

#: S45 梯度榜上的 T0 / T0.5 英雄，按位置归类。
#:
#: 用途是让用户**选**而不是手打 —— 英雄名手打会出「哪叱」「那吒」这种同音错字，
#: 而英雄名拼错的后果是**静默失效**：那条针对该英雄的模式永远不触发，
#: 而界面上看起来就是「这条英雄专属的建议没出现」。
#:
#: 这份是**兜底清单**。真正的下拉清单在 ``app/seed/hok_heroes.json`` ——
#: 那份由英雄级攻略生成，覆盖攻略里出现的全部英雄（123 个）。
#: 兜底只在读不到那份文件时用（比如老包升级前的老数据目录）。
HEROES_BY_LANE_FALLBACK: Dict[str, List[str]] = {
    "side": ["哪吒", "关羽", "司空震"],
    "carry": ["敖隐", "百里守约", "元射"],
    "mid": ["海月", "王维", "大乔"],
    "jungle": ["赵云", "镜", "马超"],
    "roam": ["盾山", "鲁班大师", "元辅"],
}


def _load_heroes_by_lane() -> Dict[str, List[str]]:
    """读英雄级攻略生成的英雄清单。

    清单是**生成产物**（由 ``tools/build_hok_hero_modes.py`` 从攻略原文推出），
    适配器只读不写 —— 名单跟着攻略走，手工维护一份只会和攻略对不上。
    """
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "seed", "hok_heroes.json",
    )
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {k: list(v) for k, v in HEROES_BY_LANE_FALLBACK.items()}
    lanes = data.get("lanes") or {}
    out: Dict[str, List[str]] = {}
    for code, entry in lanes.items():
        heroes = list((entry or {}).get("heroes") or [])
        if heroes:
            out[code] = heroes
    return out or {k: list(v) for k, v in HEROES_BY_LANE_FALLBACK.items()}


HEROES_BY_LANE: Dict[str, List[str]] = _load_heroes_by_lane()


def _load_hero_tags() -> Dict[str, Dict]:
    """官方英雄属性表（定位 / 推荐分路）。

    来自 ``app/seed/hok_hero_tags.json`` —— 由 ``tools/build_hok_hero_tags.py``
    从官方英雄资料生成。读不到就退化成空表：那时 BP 规则里
    「对方全是输出位」这类事实**取不到**，规则不触发，
    而不是拿默认英雄硬判。
    """
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "seed", "hok_hero_tags.json",
    )
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


#: 英雄 -> 定位/分路标记。模块级加载一次。
HERO_TAGS: Dict[str, Dict] = _load_hero_tags()

#: 定位里的输出位（射手/法师/刺客）——「阵容有没有输出」数的就是它。
_OUTPUT_ROLES = ("射手", "法师", "刺客")
#: 定位里的前排（坦克/战士）——「阵容有没有前排」数的就是它。
_FRONTIER_ROLES = ("坦克", "战士")


#: 对位优劣。``unknown`` 是「没填」—— 它对应**不判断**，不是「五五开」。
#: 对位优劣是玩家才知道的事，系统从画面里看不出来，所以只能问。
MATCHUP_VALUES: List[str] = ["advantage", "disadvantage", "unknown"]

MATCHUP_LABELS: Dict[str, str] = {
    "advantage": "优势对位",
    "disadvantage": "劣势对位",
    "unknown": "不确定",
}

#: 动作事实 -> 该类动作的取值 -> 中文名。
#:
#: 必须**按动作事实分组**：标注页要为每一类动作单独列出可选取值，
#: 平表做不到。早先只有一张平表，于是王者标注页上混进了共享词表里
#: 自走棋时代的取值（take_baron / push_tower / adjust…），显示成
#: 英文原始码 —— 玩家根本不知道勾的是啥，而勾了也不对应任何一条王者模式。
MOBA_ACTION_VALUES: Dict[str, Dict[str, str]] = {
    # objective_action
    "objective_action": {
        "contest_dragon": "开小龙团",
        "take_dragon": "拿下小龙",
        "fight_baron": "开大龙团",
        "split_push": "分推牵制",
        "siege": "拆高地塔",
        "recall": "回城补给",
        "farm_jungle": "刷野补经济",
        "invade_jungle": "反野入侵",
        "defend_inner": "回防高地",
        # 下面两个是 S45 视野精灵改版与发育路补刀带来的操作，原词表里没有。
        # 不加的话，「顺风清掉视野精灵」和「补刀优先于推线」这两条最具体的
        # 分路模式就只能退化成「刷野 / 推线」，攻略的结论就丢了。
        "clear_vision_sprite": "清视野精灵",
        "last_hit": "补刀",
        # 河道之灵（河蟹）。S45 决策树里「中路河道之灵优先于常规野怪」
        # 是三条分路（打野 / 中路 / 发育路）前期第一次资源取舍的依据，
        # 没有这个取值就只能退化成「打龙」，那会把「1 分钟河蟹」说成「暴君」。
        "take_river_spirit": "控河道之灵",
    },
    # positioning_action
    "positioning_action": {
        "push_wave": "推线压塔",
        "pull_back": "回撤拉扯",
        "ward_vision": "游走做视野",
        "hold_position": "卡位蹲伏",
        "rotate_support": "转线支援",
        "cover_carry": "保护射手位",
    },
    # teamfight_action
    "teamfight_action": {
        "engage": "开团先手",
        "focus_target": "集火秒人",
        "protect_carry": "保护C位输出",
        "retreat": "撤退脱离",
        "kite": "拉扯消耗",
        "split_fight": "分散拉扯",
    },
}

#: 扁平版：取值 -> 中文名。规则与报告按取值查名字。
MOBA_ACTION_LABELS: Dict[str, str] = {
    value: label
    for group in MOBA_ACTION_VALUES.values()
    for value, label in group.items()
}

#: 分路 / 职责。**五条**。
#:
#: 「前中后期不同位置该做什么」这类攻略，说的是**角色分工**而不是地图坐标 ——
#: 打野前期刷野、中期抓单、后期开龙；中单前期清线、中期支援。
#: 地图站位那类信息由 ``positioning_action`` 承载，两者不要混。
#:
#: 曾经把游走和辅助拆成两个值，那是自走棋时代留下的习惯。MOBA 里
#: 「游走（辅助）」是同一个职责 —— 信息位 + 开团位，拆开会让同一条
#: 攻略里的同一句话被分到两个格子，而它们要的其实是同一批动作。
#:
#: 一局里角色基本固定，所以 position 是**对局级**事实，会广播到每个回合。
POSITION_VALUES: List[str] = [
    "jungle",      # 打野
    "mid",         # 中路
    "side",        # 对抗路
    "carry",       # 发育路 / 射手
    "roam",        # 游走（辅助）
    "unknown",     # 没填 —— 保留「不知道」，而不是默认成某个位置
]

POSITION_LABELS: Dict[str, str] = {
    "jungle": "打野",
    "mid": "中路",
    "side": "对抗路",
    "carry": "发育路",
    "roam": "游走",
    "unknown": "未指定",
}

#: 攻略里常见的别称 -> 规范值。抽取和录入都走这张表。
POSITION_ALIASES: Dict[str, str] = {
    "打野": "jungle", "野核": "jungle", "刺客": "jungle",
    # 模型描述打野时更常说「野区」—— 那是地图区域，但在这句问「玩家主要
    # 打哪条路」的话里，它指的就是打野这个分工。
    "野区": "jungle", "野怪区": "jungle",
    "中单": "mid", "中路": "mid", "法师": "mid",
    "对抗路": "side", "边路": "side", "战士": "side", "对抗": "side",
    "射手": "carry", "c位": "carry", "输出": "carry", "发育路": "carry",
    # 游走与辅助是同一个位置，别名指向同一个规范值。
    # 写错的表现很隐蔽：填「辅助」归不到任何位置，规则会全部不触发。
    "游走": "roam", "游走位": "roam", "辅助": "roam", "辅助游走": "roam",
    "坦克": "roam", "前排": "roam",
}

POSITION_CODES: Dict[str, int] = {v: i + 1 for i, v in enumerate(POSITION_VALUES)}

#: 分路的**展示顺序**。报告、标注页、筛选条都按这个顺序展开 ——
#: 打野 → 中路 → 对抗路 → 发育路 → 游走，读起来就是一路打完一路的顺序。
#: 由 :data:`POSITION_VALUES` 派生，所以增删分路不会漏改这里。
POSITION_ORDER: List[str] = [v for v in POSITION_VALUES if v != "unknown"]

#: 阶段的中文名。报告与标注页都用它，不要各自另起一套叫法。
PHASE_LABELS: Dict[str, str] = {
    "early": "前期",
    "mid": "中期",
    "late": "后期",
}

#: 局势压力的中文名。``unknown`` 是「没采到差值」——
#: 它表示**算不出**，不是「局势平」。
PRESSURE_LABELS: Dict[str, str] = {
    "low": "低压",
    "mid": "中压",
    "high": "高压",
    "unknown": "未判定",
}


def normalize_position(raw: Any) -> str:
    """把玩家填的分路规整成规范值。

    填「打野」「打野位」「野核」都归到 ``jungle``；认不出来就返回
    ``unknown``，**不猜**。猜错位置会让整局每一条带位置的规则都判错，
    而那种错误在报告里看不出来 —— 它只是安静地一条都不触发。
    """
    if not raw:
        return "unknown"
    s = str(raw).strip()
    if s in POSITION_CODES:
        return s
    if s in POSITION_LABELS:
        return s
    return POSITION_ALIASES.get(s, "unknown")


#: 局势阶段 -> (编码, 事实取值)。stage 用编码，facts["phase"] 用取值，
#: 两者同时可读：报告里既能看到「3-2」，也能看到「后期」。
PHASE_CODES: Dict[str, int] = {
    "early": 1,
    "mid": 2,
    "late": 3,
}

#: 攻略里的阶段词 -> 局势取值。
#:
#: 攻略几乎不写分钟数，只写「前期」「中期」「开局」「6 分钟后」这类词。
#: 没有这张表，抽出来的规则只有 action 没有时间，等于一条永远生效的
#: 全局规则 —— 那是噪音，不是知识。
#:
#: 「决胜期 / 大龙期 / 末期」归到后期而不是单列一档：攻略的时间轴只有
#: 三段，而这三个词指的都是「10 分钟以后」的那一段。硬拆成第四段，
#: 只会让库里的模式落在一个攻略没有单独讨论过的档里。
PHASE_ALIASES: Dict[str, str] = {
    "前期": "early", "开局": "early", "前期发育": "early", "刚开局": "early",
    "中期": "mid", "中期游走": "mid", "中期团战": "mid",
    "后期": "late", "后期团战": "late",
    "决胜期": "late", "大后期": "late", "大龙期": "late", "末期": "late",
    "两分钟前": "early", "五分钟前": "early",
}

#: 阶段 -> 时间区间（分钟）。
#:
#: **三个**，不再是四个。边界取自 S45 分路运营报告的通用决策速查表：
#: 0-4 分钟 / 4-10 分钟 / 10 分钟以后。攻略怎么分，库就怎么分 ——
#: 多切一刀不会让结论更准，只会让「这一段属于哪一档」变得要看说明书。
#: 早先多出来的「决胜期」就是这么来的，而 17 条模式里没有一条起点落在 25 分钟之后。
#:
#: **区间之间不许重叠**。``between`` 是闭区间，早先按 [0,4] / [4,10] 写，
#: 第 4 分钟会同时落进「前期要刷野」和「中期要控龙」，两条必定有一条判偏离，
#: 而玩家其实什么都没做错。所以 [0,4] / [5,10] / [11,60] 不重不漏。
PHASE_TIME_WINDOWS: Dict[str, List[int]] = {
    "early": [0, 4],
    "mid": [5, 10],
    "late": [11, 60],
}


def _as_int(value: Any) -> Optional[int]:
    """把事实取值收敛成数值；无法解析返回 None 表示「没采到」。"""
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
    compute,
) -> None:
    """原料齐全才写派生事实，否则标记缺失 —— 宁可判不了，不给半截结论。"""
    if name in missing or name in facts:
        return
    for key in inputs:
        if key in missing or facts.get(key) is None:
            missing.add(name)
            return
    facts[name] = compute()


def phase_of(time_min: int) -> str:
    """按分钟映射局势阶段。

    边界与 :data:`PHASE_TIME_WINDOWS` 必须**逐一对齐** ——
    0-4 前期 / 5-10 中期 / 11 分钟以后后期。
    对不齐就会出现「facts 说中期、窗口却落在前期」的规则库，
    而那种错在报告里完全看不出来。
    """
    if time_min < 5:
        return "early"
    if time_min < 11:
        return "mid"
    return "late"


def derive_pressure(
    gold_diff: int,
    kill_diff: int,
    tower_diff: int,
) -> str:
    """局势压力由三项差值投票决定。

    三项各计 ±1：单项落后只到中压，两项以上落后才是高压。
    差值全缺时返回 ``unknown``，规则里的 ``pressure == "high"`` 因此不会
    把「没采到数据」误当成「局势很差」。
    """
    values = (gold_diff, kill_diff, tower_diff)
    if all(v is None for v in values):
        return "unknown"

    score = 0
    if gold_diff is not None:
        score += 1 if gold_diff <= -2000 else (-1 if gold_diff >= 2000 else 0)
    if kill_diff is not None:
        score += 1 if kill_diff <= -5 else (-1 if kill_diff >= 5 else 0)
    if tower_diff is not None:
        score += 1 if tower_diff <= -2 else (-1 if tower_diff >= 2 else 0)

    if score >= 2:
        return "high"
    if score >= 0:
        return "mid"
    return "low"


def _num_or_zero(facts: Dict[str, Any], missing: set, name: str) -> int:
    """取事实里的数值；缺失或为空时按 0 处理。

    「缺失」和「0」在派生计算里意义不同：缺失意味着**不知道**，
    而派生公式对「不知道」只能当中性值用。

    早先这里只判了 ``missing`` 集合，没判值本身是 None，于是
    **视频链路直接抛 TypeError**：用户不填经济差（默认就不填），
    ``facts.get("gold_diff")`` 返回 None，被原样传进派生函数，
    ``None <= -5000`` 就炸了。表现是上传扫描都正常，提交标注时报 500。
    """
    if name in missing:
        return 0
    value = _as_int(facts.get(name))
    return 0 if value is None else value


def derive_is_decision_point(
    time_min: int,
    gold_diff: int,
    kill_diff: int,
) -> bool:
    """关键决策时刻 = 时间片切换的第一分钟，或局势已经崩到需要立刻回应的程度。

    时间片切换点之所以是关键决策时刻，是因为野区刷新、龙坑计时、
    大招循环都以此为节拍；把判定锚在「整 5 分钟」上而不是「感觉该动了」，
    是让规则可判定的核心。

    ``gold_diff`` / ``kill_diff`` 允许为 None（没填就是不知道），
    此时按「局势不崩」处理 —— 不知道不等于崩。
    """
    if time_min % SLICE_MINUTES == 0:
        return True
    if gold_diff is not None and gold_diff <= -5000:
        return True
    if kill_diff is not None and kill_diff <= -8:
        return True
    return False


class HonorOfKingsAdapter(GameAdapter):
    """王者荣耀。"""

    game_id = "hok"
    #: 与 app/seed/hok_modes.json 的版本一致。填错的后果很隐蔽：
    #: 上传时按这个版本去找规则库，找不到就整局判不出任何结论。
    default_version = "S45"
    display_name = "王者荣耀"
    fact_names = list(MOBA_FACT_NAMES) + list(BP_FACT_NAMES)
    #: MOBA 的可操作事实是三类，期望动作必须挂在其中之一
    action_fact_names = [
        "objective_action",
        "positioning_action",
        "teamfight_action",
    ]
    #: BP 阶段的动作事实单列一组 —— 它在局内标注页用不上。
    bp_action_fact_names = ["bp_action"]
    #: BP 决策环节的中文名。模式库页的 BP 那一节按它分组。
    bp_phase_labels = dict(BP_PHASE_LABELS)
    fact_labels = {**MOBA_FACT_LABELS, **BP_FACT_LABELS}
    action_labels = {**MOBA_ACTION_LABELS, **BP_ACTION_LABELS}
    #: 非动作事实的取值中文名。报告的「当时局面」要把这些事实拼成一句话，
    #: 没有这张表的话页面上会出现「分路 jungle」「对位优劣 unknown」。
    fact_value_labels = {
        "position": dict(POSITION_LABELS),
        "matchup": dict(MATCHUP_LABELS),
        "phase": dict(PHASE_LABELS),
        "pressure": dict(PRESSURE_LABELS),
        "has_ult_ready": {"True": "是", "False": "否"},
        "in_combat": {"True": "是", "False": "否"},
        "is_decision_point": {"True": "是", "False": "否"},
    }
    action_values_by_fact = {
        k: dict(v) for k, v in MOBA_ACTION_VALUES.items()
    }
    action_values_by_fact["bp_action"] = dict(BP_ACTION_LABELS)
    #: 阶段 -> 时间区间。采集层按它抽帧，判定层按它判阶段，两边必须同源 ——
    #: 各自写一份就会在某次改窗口时只改一处，然后出现「帧标中期、判成前期」。
    phase_time_windows = dict(PHASE_TIME_WINDOWS)
    #: HUD 布局：采集层用它判**录屏方向**。
    #:
    #: 手机录屏不一定带正确的旋转元数据（实测遇到过写反的：
    #: 按元数据转过来，画面是上下颠倒的），而界面只显示「没读到时钟」，
    #: 看不出真正的原因是画面转反了。所以方向改用画面布局判 ——
    #: 这个游戏的 HUD 布局是固定的：小地图在左上、技能按钮在右下、
    #: 摇杆在左下（一个很大的光滑圆）。
    #:
    #: 这个判据与分辨率、屏幕比例、清晰度都无关，换机型换录屏软件都成立。
    hud_layout = {
        "minimap": "上左",
        "skills": "下右",
        "joystick": "下左",
    }
    #: 报告里的操作单位。
    #:
    #: 自走棋真的有「一回合」，王者荣耀没有 —— 一局王者是**连续的一整场**，
    #: 切出来的只能是**时间片**。报告里说「这一回合」会让玩家去找回合，
    #: 而王者里根本没这个东西。
    #:
    #: 判定层的「回合」是自走棋沿用下来的历史包袱，报告这一层
    #: 按游戏说话：王者叫「时间片」「这一段」。
    segment_unit = "时间片"
    segment_now = "这一段"    #: 标注页要问玩家的数字型事实：**局内时间、经济差、击杀差**。
    #:
    #: 这份清单是**从规则库倒推**出来的，不是拍脑袋定的。
    #: S45 决策树那套模式里，触发条件引用的事实只有 time_min /
    #: position / matchup / my_hero / gold_diff / kill_diff / has_ult_ready：
    #:
    #:   position / matchup / my_hero  标注页顶部一次选完，对局级
    #:   time_min                      时钟读出来并预填
    #:   gold_diff / kill_diff         真正要玩家手填的两项（都是记分板上的数）
    #:   has_ult_ready                 是 / 否，见 annotate_bool_facts
    #:
    #: 早先这里是「所有带标签的非动作事实」，于是每张卡渲染 13 个数字
    #: 输入框，玩家面对 13 个框、真正该填的只有 1 个。而且多出来的那些
    #: 不是「问了也没事」，是**问了会给错答案** —— 局势阶段、分路、
    #: 英雄、对位优劣都由顶部选择或派生逻辑决定，在数字框里填是填不进去的，
    #: 玩家以为自己改了什么，其实被适配器覆盖掉了。
    #:
    #: 少问比多问好，但**不能少问到规则要用的那一项** ——
    #: 少问一条，引用它的模式就永远不触发，而报告上只是「没结论」，
    #: 看不出是哪里错了。所以有一条测试盯着「规则库引用的每个数值事实
    #: 都必须出现在这份清单里（或由顶部选择 / 时钟自动提供）」。
    annotate_numeric_facts = ["time_min", "gold_diff", "kill_diff"]
    #: 标注页要问玩家的**布尔**事实，渲染成勾选框。
    #:
    #: 「我方核心大招是否就绪」是攻略里「控制链就绪」「开团条件齐」
    #: 这两类分支唯一能被观察到的近似 —— 问不到它，那几条模式就永远
    #: 不触发，而报告上只是「没结论」。
    annotate_bool_facts = ["has_ult_ready"]
    #: 标注页据此渲染分路下拉。留着 unknown 在列表里是有意的：
    #: 「我这局没填分路」是一个合法选项，它对应「不评判」，而不是「随便算一个」。
    position_values = list(POSITION_VALUES)
    position_labels = dict(POSITION_LABELS)
    position_order = list(POSITION_ORDER)
    phase_labels = dict(PHASE_LABELS)
    #: 英雄按分路分组，标注页据此把英雄下拉按位置联动
    heroes_by_lane = {k: list(v) for k, v in HEROES_BY_LANE.items()}
    matchup_values = list(MATCHUP_VALUES)
    matchup_labels = dict(MATCHUP_LABELS)
    round_unit = "分钟"

    def normalize_position(self, raw: Any) -> str:
        """分路写法归一。页面给对局起名时要用它。

        库里存的可能是「打野」也可能是 ``jungle``，不归一的话同一条分路
        会被拆成两栏，对局列表也就分不干净。
        """
        return normalize_position(raw)

    def normalize(self, raw: Dict[str, Any]) -> MatchContext:
        known_facts = set(self.fact_names)
        rounds: List[RoundRecord] = []

        for raw_round in (raw.get("rounds") or []):
            facts: Dict[str, Any] = dict(raw_round.get("facts") or {})
            missing: set = set(raw_round.get("missing_facts") or [])

            time_min = _as_int(raw_round.get("time_min"))
            if time_min is None:
                time_min = _as_int(facts.get("time_min"))
            if time_min is None:
                # 没有时间就定位不到回合，整局分析对该片段无意义
                time_min = 0
                missing.add("time_min")
            facts["time_min"] = time_min

            stage = time_min // SLICE_MINUTES
            phase_code = PHASE_CODES[phase_of(time_min)]
            # 只登记 phase 事实，不登记 stage：MOBA 事实集里没有 stage
            # （自走棋才有阶段序号），回合坐标由 RoundRecord.stage 承担，
            # 规则要定位时间就精确到 time_min，不依赖粗粒度的时间片编号。
            facts["phase"] = phase_of(time_min)

            # position 是**对局级**事实：分路在一局里基本固定，所以广播到
            # 每一个回合。规则要写「前期打野该做什么」，时间与位置两个维度
            # 必须同时取得到 —— 只有时间没有位置，表达不出分工，
            # 而分工恰恰是 MOBA 攻略最核心的那部分。
            player = raw.get("player") or {}
            facts["position"] = normalize_position(
                player.get("position") or raw.get("position")
            )

            # 英雄与对位优劣同样是对局级的：一场对局英雄不会变，
            # 而它们是英雄层模式能不能生效的前提。不填就是空串 / unknown ——
            # 英雄层模式自然一条都不触发，而不是拿某个默认英雄硬判。
            facts["my_hero"] = str(
                player.get("hero") or raw.get("my_hero") or ""
            ).strip()
            matchup = str(
                player.get("matchup") or raw.get("matchup") or ""
            ).strip().lower()
            facts["matchup"] = (
                matchup if matchup in MATCHUP_VALUES else "unknown"
            )

            gold_diff = _as_int(facts.get("gold_diff"))
            kill_diff = _as_int(facts.get("kill_diff"))
            tower_diff = _as_int(facts.get("tower_diff"))

            _put_derived(
                facts, missing, "pressure",
                ("gold_diff", "kill_diff", "tower_diff"),
                lambda: derive_pressure(gold_diff, kill_diff, tower_diff),
            )
            _put_derived(
                facts, missing, "is_decision_point", ("time_min",),
                lambda: derive_is_decision_point(
                    time_min,
                    _num_or_zero(facts, missing, "gold_diff"),
                    _num_or_zero(facts, missing, "kill_diff"),
                ),
            )

            actions: List[ActionRecord] = []
            for payload in (raw_round.get("actions") or []):
                kind = str(payload.get("kind") or "").strip()
                value = payload.get("value")
                label = str(payload.get("label") or "").strip() or self.action_labels.get(
                    str(value), str(value)
                )
                confidence = payload.get("confidence", 1.0)
                try:
                    confidence = float(confidence)
                except (TypeError, ValueError):
                    confidence = 1.0
                actions.append(
                    ActionRecord(
                        kind=kind,
                        value=value,
                        label=label,
                        ts=float(payload.get("ts") or 0.0),
                        # 置信度是报告里「低置信度」标记的唯一依据，不能丢
                        confidence=min(1.0, max(0.0, confidence)),
                        # 来源：**画面里看出来的**还是**玩家自己勾的** ——
                        # 报告要标出来，否则用户会以为那是玩家说的。
                        source=str(payload.get("source") or "player"),
                    )
                )

            # 动作镜像进 facts，让 trigger 也能对动作事实下条件
            # （例如「本回合不能回城」写在 none 组里）。
            # 同一个动作事实出现多次时保留最后一次 —— 触发条件看的是这一个值。
            # 期望动作那一侧的口径不同：那里是「这一帧做过符合要求的动作就算
            # 符合」，见 app/core/matcher.py。
            for act in actions:
                if not act.kind or act.kind not in known_facts or act.kind in missing:
                    continue
                facts[act.kind] = act.value

            rounds.append(
                RoundRecord(
                    stage=stage,
                    phase=phase_code,
                    facts=facts,
                    actions=actions,
                    missing_facts=missing,
                    # MOBA 没有「阶段」概念。标签直接给分钟数，
                    # 否则报告里会出现「第 2-2 回合」这种玩家读不懂的坐标 ——
                    # 系统的价值在于把结论指到玩家能定位的地方，
                    # 指到一个看不懂的坐标等于没指。
                    #
                    # 视频来源的标签是**窗口标签**（「6:00 ~ 10:00」「12:00 之后」）：
                    # 分析单元是时间窗，规则管的是一整段，
                    # 只写「第 6 分钟」会把范围说成时点。
                    label_override=str(
                        raw_round.get("label_override")
                        or f"第 {time_min} 分钟"
                    ),
                    # 同一 5 分钟时间片内可能有好几个观测点，带上精确分钟
                    # 才能让报告按真实先后排列而不是按严重度跳来跳去
                    time_index=float(time_min),
                    # 原视频里的秒数与抽帧文件。视频来源的对局才有；
                    # 手工录入的对局这里是 0，报告就不显示「看这一段」。
                    video_time=float(raw_round.get("video_time") or 0.0),
                    # 探测器找到的证据时刻：报告里「这段在 12:34 出现开龙迹象」
                    # 靠它 —— 玩家要能回看原画面确认，光给结论不够。
                    probe_moments=list(raw_round.get("probe_moments") or []),
                    # 明确没做的动作类：报告里写「画面里没看到」，
                    # 而不是「没记录到操作」（后者听着像玩家没勾）。
                    probe_absent=list(raw_round.get("probe_absent") or []),
                    # 模型对操作模式清单的裁决。报告正文主要靠它 ——
                    # 逐回合判定只覆盖前提能从数值判出来的那几条。
                    review=list(raw_round.get("review") or []),
                    image=raw_round.get("image", "") or "",
                )
            )

        return MatchContext(
            match_id=str(raw.get("match_id") or ""),
            game=self.game_id,
            game_version=str(raw.get("game_version") or self.default_version),
            rounds=rounds,
            raw=dict(raw),
            player=dict(raw.get("player") or {}),
            outcome=dict(raw.get("outcome") or {}),
            annotation_source=str(raw.get("annotation_source") or "player"),
        )

    # -- BP（选人禁用）阶段 --------------------------------------------------

    def build_bp_context(self, raw: Dict[str, Any]) -> MatchContext:
        """把一条对局的 **BP 记录**翻译成可判定的上下文。

        BP 记录长这样（``payload['bp']``）::

            {
              "my_missing_engage": true,
              "opp_missing_frontline": true,
              "decisions": [
                {"side": "my",  "phase": "ban_1",  "action": "ban_break_comp", "hero": "海月"},
                {"side": "opp", "phase": "ban_1",  "action": "ban_opp_core",   "hero": "甄姬"},
                {"side": "my",  "phase": "pick_1", "action": "pick_flex",      "hero": "澜"}
              ]
            }

        只有 ``side == 'my'`` 的决策进判定（那是玩家自己做的选择）；
        对方的决策只提供**局面**：对方已暴露几个 pick、对方阵容什么结构。

        事实的来源有讲究：

        - **玩家勾的**（阵容缺件）—— 优先，信玩家的；
        - **官方定位表算的**（输出位、刺客数、摇摆位）—— 兜底；
        - **两者都拿不到** —— 该事实**不登记**（而不是填个默认值）。
          没登记的事实让规则**不触发**，并在报告里标成「数据缺失」——
          这跟判成偏离是两件事：前者是「不知道」，后者是「知道且做错了」。
        """
        bp = raw.get("bp") or {}
        decisions = list(bp.get("decisions") or [])

        # 己方与对方最终阵容（选人环节的英雄）。
        my_heroes = [
            str(d.get("hero") or "").strip()
            for d in decisions
            if d.get("side") == "my"
            and d.get("phase") in BP_PICK_PHASES
            and str(d.get("hero") or "").strip()
        ]
        opp_heroes = [
            str(d.get("hero") or "").strip()
            for d in decisions
            if d.get("side") == "opp"
            and d.get("phase") in BP_PICK_PHASES
            and str(d.get("hero") or "").strip()
        ]

        my_tags = [HERO_TAGS.get(h) for h in my_heroes]
        opp_tags = [HERO_TAGS.get(h) for h in opp_heroes]
        my_final_tags = [t for t in my_tags if t]
        opp_final_tags = [t for t in opp_tags if t]

        def _count(tags, key: str) -> int:
            return sum(1 for t in tags if t and t.get(key))

        rounds: List[RoundRecord] = []
        my_picks_so_far: List[str] = []
        opp_picks_so_far: List[str] = []
        my_step = 0

        for decision in decisions:
            if decision.get("side") != "my":
                if decision.get("phase") in BP_PICK_PHASES:
                    hero = str(decision.get("hero") or "").strip()
                    if hero:
                        opp_picks_so_far.append(hero)
                continue

            my_step += 1
            phase = str(decision.get("phase") or "").strip()
            action = str(decision.get("action") or "").strip()
            hero = str(decision.get("hero") or "").strip()

            facts: Dict[str, Any] = {}
            missing: set = set()

            facts["bp_phase"] = phase
            facts["bp_step"] = my_step
            facts["bp_hero"] = hero
            # 段位档与预选冲突这两条事实**已经不存在了**：
            # BP 记录页与复盘页都不再问它们，判定层也不登记 ——
            # 拿玩家看不见的东西下结论，只会更困惑。
            # 依赖它们的 3 条规则已从规则库删除（见 tools/hok_bp_spec.py）。

            facts["opp_picks_before"] = len(opp_picks_so_far)
            facts["my_picks_before"] = len(my_picks_so_far)

            # 己方已选里的输出位数 —— 官方定位表算。
            so_far_tags = [HERO_TAGS.get(h) for h in my_picks_so_far]
            facts["my_output_count"] = _count(so_far_tags, "output")

            # 局面事实：玩家勾的优先，官方表能算的兜底，都没有就不登记。
            self._put_bp_missing(
                facts, missing, "my_missing_frontline", bp.get("my_missing_frontline"),
                lambda: not _count(so_far_tags, "frontline"),
                need_comp=bool(my_picks_so_far),
            )
            self._put_bp_missing(
                facts, missing, "my_missing_damage", bp.get("my_missing_damage"),
                lambda: not _count(so_far_tags, "output"),
                need_comp=bool(my_picks_so_far),
            )
            self._put_bp_missing(
                facts, missing, "my_missing_engage", bp.get("my_missing_engage"),
                None,
            )
            self._put_bp_missing(
                facts, missing, "my_missing_protect", bp.get("my_missing_protect"),
                None,
            )
            self._put_bp_missing(
                facts, missing, "opp_missing_frontline", bp.get("opp_missing_frontline"),
                lambda: not _count(opp_final_tags, "frontline"),
                need_comp=bool(opp_picks_so_far),
            )

            # 「己方其余位置」= 最终阵容里除这一步之外的部分。
            others = [t for h, t in zip(my_heroes, my_final_tags)
                      if h != hero] if hero else list(my_final_tags)
            facts["my_other_has_tank_or_support"] = bool(
                others
                and any(
                    t.get("tank") or t.get("support")
                    for t in others
                    if t
                )
            )
            facts["my_other_all_output"] = bool(
                others and all(t.get("output") for t in others if t)
            )
            facts["my_has_flex"] = bool(
                any(t.get("flexible") for t in my_final_tags if t)
            )
            facts["my_all_output"] = facts["my_output_count"] >= 5
            facts["opp_all_output"] = _count(opp_final_tags, "output") >= 5
            facts["my_assassin_count"] = _count(my_final_tags, "assassin")
            facts["opp_assassin_count"] = _count(opp_final_tags, "assassin")

            actions: List[ActionRecord] = []
            if action:
                actions.append(
                    ActionRecord(
                        kind="bp_action",
                        value=action,
                        label=BP_ACTION_LABELS.get(action, action),
                        ts=float(my_step),
                        confidence=1.0,
                    )
                )
            else:
                missing.add("bp_action")

            rounds.append(
                RoundRecord(
                    stage=0,
                    phase=0,
                    facts=facts,
                    actions=actions,
                    missing_facts=missing,
                    # BP 的坐标是「哪个环节」，不是局内分钟。
                    # 不写「第几手」—— 「手」是自走棋的单位，BP 里没有这个说法，
                    # 环节名（一抢 / 二抢 / 三抢）自己就把先后说清楚了。
                    label_override=(BP_PHASE_LABELS.get(phase)
                                    or BP_PHASE_LEGACY_LABELS.get(phase, phase)),
                    # 按决策先后排序 —— 报告里 BP 条目要按 draft 顺序讲。
                    time_index=float(my_step),
                )
            )

            if phase in BP_PICK_PHASES and hero:
                my_picks_so_far.append(hero)

        ctx = MatchContext(
            match_id=str(raw.get("match_id") or ""),
            game=self.game_id,
            game_version=str(raw.get("game_version") or self.default_version),
            rounds=rounds,
            raw=dict(raw),
            player=dict(raw.get("player") or {}),
            outcome=dict(raw.get("outcome") or {}),
            annotation_source=str(raw.get("annotation_source") or "player"),
        )
        ctx.fact_resolvers = self._resolvers
        ctx.precompute()
        return ctx

    @staticmethod
    def _put_bp_missing(
        facts: Dict[str, Any],
        missing: set,
        name: str,
        player_value: Any,
        derive,
        need_comp: bool = True,
    ) -> None:
        """登记一个 BP 局面事实：玩家勾的优先，官方表能算的兜底，都没有就不登记。"""
        if player_value is not None:
            facts[name] = bool(player_value)
            return
        if derive is None or not need_comp:
            missing.add(name)
            return
        facts[name] = bool(derive())


HOK_ADAPTER = register(HonorOfKingsAdapter())
