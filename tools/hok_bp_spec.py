"""BP 阶段的事实集、动作类别与分支映射表。

为什么 BP 要单独一套
--------------------
局内规则问的是「第 7 分钟这一下该不该控龙」；
BP 规则问的是「这一手该禁谁、该选谁」。
两者的**时间坐标不同**（局内是分钟，BP 是决策环节）、
**事实不同**（局内是经济差血量，BP 是阵容缺件）、
**动作不同**（局内是刷野开团，BP 是禁用与选人），
硬塞进同一层，最后会得到一批永远不触发的规则。

事实的两种来源
--------------
BP 的局面事实有两个来源，**混着用**：

1. **玩家勾选**（``my_missing_engage`` 这类）——
   开团点、反手保护这些，官方资料里没有，靠玩家的 BP 记录勾。
2. **官方定位表算**（``my_all_output``、``my_assassin_count``、``my_has_flex``）——
   定位与推荐分路是客观数据，官方英雄资料里有，机器可算。

**玩家勾选优先**：玩家比系统更清楚自己阵容缺什么。
两处冲突时以玩家为准 —— 他就在 BP 现场。

期望动作怎么写
--------------
一条 BP 规则的期望是「**这一手属于哪一类才算做对**」。
攻略里的「不要 X」不写成单独的禁令，而是**从可接受集合里去掉 X**：
DSL 的 ``none`` 组是**适用性护栏**（这条规则在这一帧适不适用），
不是禁令 —— 拿它表达禁令会正好写反（写了「别抢核心输出」，
规则反而在抢了核心输出时不适用）。

判不了的分支怎么办
----------------
``SKIPPED`` 里逐条写明理由。这跟英雄层的 ``hok_hero_skipped.json``
是同一套规矩：**每条分支要么变成规则，要么写清楚为什么进不去**。
需要「版本 Ban 率」「选手熟练度」「对手意图」这类事实的分支，
系统没有这些事实，硬编一张表就是替玩家下判断 —— 判错了玩家还找不到原因。
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 事实与动作的词表是**运行时契约**（适配器的事），
# 生成器只负责「哪条分支对应哪条规则」。两边各写一份迟早会漂 ——
# 漂了的表现是规则引用的动作不在词表里，整批被准入拦下。
from app.adapters.honor_of_kings import (  # noqa: E402
    BP_ACTION_LABELS,
    BP_FACT_LABELS,
    BP_PHASE_LABELS,
)

#: 重新导出，方便别的模块只引这一处。
BP_ACTIONS: Dict[str, str] = dict(BP_ACTION_LABELS)
BP_FACTS: Dict[str, Tuple[str, str]] = {
    name: (label, "见适配器 BP_FACT_LABELS 与本文件的事实说明")
    for name, label in BP_FACT_LABELS.items()
}
PHASE_LABELS: Dict[str, str] = dict(BP_PHASE_LABELS)

#: 禁用类动作（BP 录入界面分栏用）。
BAN_ACTIONS: List[str] = [k for k in BP_ACTIONS if k.startswith("ban_")]
PICK_ACTIONS: List[str] = [k for k in BP_ACTIONS if k.startswith("pick_")]
PLAN_ACTIONS: List[str] = [k for k in BP_ACTIONS if k.startswith(("plan_", "check_"))]

#: 每一个禁用动作都可接受时，说明「这一手禁得没问题」。
_ALL_BANS_EXCEPT_TANK = [
    "ban_engage", "ban_break_comp", "ban_press_strong", "ban_opp_core",
    "ban_cover_gap", "ban_idol", "ban_break_heal",
]

#: 「针对性选人」这一类里都算做对的做法。
_COUNTER_PICKS = [
    "pick_counter", "pick_heal_cut", "pick_anti_control", "pick_damage",
    "pick_engage",
]

# --------------------------------------------------------------------------
# 分支 -> 规则
# --------------------------------------------------------------------------

#: 每条规则：键是 (环节编码, 分支序号)。
#:
#: 同一个键可以出**多条**规则 —— 一条分支里写了两个要求时
#: （比如「别把坦边都禁了」+「留一个 ban 位给开团辅助」），
#: 两个要求的判定口径不同（逐手 vs 整个环节），合成一条会两边都不准。
RULES: List[Dict[str, Any]] = [
    # 「进入 BP」环节原有三条分支（高端局先定体系 / 预选冲突先确认分路 /
    # 低分段一楼不锁脆皮），它们依赖「段位」与「预选冲突」这两条事实。
    # 2026-10-22：BP 记录页与复盘页都不再问段位与预选，判定层也不再登记
    # 这两条事实 —— 留着它们就是三条永远判不出结果的规则。
    # 已移到下面的 SKIPPED，理由逐条写明。
    # 这一环整条撤下后，记录页不再有「进入 BP」这个环节。
    # ---- 禁用（阶段一）----
    {
        "key": ("ban_1", 3),
        "variant": "A",
        "title": "己方缺前排时，别把坦边禁了",
        "trigger": [["bp_phase", "==", "ban_1"], ["my_missing_frontline", "==", True]],
        "expect": list(_ALL_BANS_EXCEPT_TANK),
        "scope": "per_frame",
        "severity": "high",
    },
    {
        "key": ("ban_1", 3),
        "variant": "B",
        "title": "己方缺前排时，留一个 ban 位给对面的开团位",
        "trigger": [["bp_phase", "==", "ban_1"], ["my_missing_frontline", "==", True]],
        "expect": ["ban_engage"],
        "scope": "at_least_once",
        "severity": "medium",
        "note": "攻略说「留一个 ban 位」——是分配，不是每一手都这么禁，"
                "所以按「做到过就算」的口径判。",
    },
    # ---- 一抢 ----
    {
        "key": ("pick_1", 1),
        "title": "一抢不锁核心输出，先拿摇摆位",
        "trigger": [["bp_phase", "==", "pick_1"], ["my_output_count", "==", 0]],
        "expect": ["pick_flex", "pick_engage", "pick_anti_control",
                   "pick_ally_main", "pick_frontline", "pick_tempo",
                   "pick_defensive"],
        "scope": "per_frame",
        "severity": "medium",
    },
    {
        "key": ("pick_1", 2),
        "title": "已有控制型坦克或辅助，一抢直接锁开团位",
        "trigger": [["bp_phase", "==", "pick_1"],
                    ["my_other_has_tank_or_support", "==", True]],
        "expect": ["pick_engage", "pick_anti_control"],
        "scope": "per_frame",
        "severity": "medium",
        "note": "「己方已有」按**除这一手之外的阵容**判定：标准 BP 顺序下"
                "一抢时己方还没选人，按当前已选数判这条前提永远不成立。",
    },
    # ---- 二抢 ----
    {
        "key": ("pick_2", 1),
        "title": "对方露出两个 pick，二抢就该针对",
        "trigger": [["bp_phase", "==", "pick_2"], ["opp_picks_before", "==", 2]],
        "expect": list(_COUNTER_PICKS),
        "scope": "per_frame",
        "severity": "high",
    },
    # ---- 三抢 ----
    {
        "key": ("pick_3", 1),
        "title": "还缺开团点，三抢补上",
        "trigger": [["bp_phase", "==", "pick_3"], ["my_missing_engage", "==", True]],
        "expect": ["pick_engage"],
        "scope": "per_frame",
        "severity": "high",
    },
    {
        "key": ("pick_3", 2),
        "title": "还缺前排承伤，三抢补坦",
        "trigger": [["bp_phase", "==", "pick_3"], ["my_missing_frontline", "==", True]],
        "expect": ["pick_frontline"],
        "scope": "per_frame",
        "severity": "high",
    },
    {
        "key": ("pick_3", 3),
        "title": "还缺远程消耗，三抢补法核",
        "trigger": [["bp_phase", "==", "pick_3"], ["my_missing_damage", "==", True]],
        "expect": ["pick_damage"],
        "scope": "per_frame",
        "severity": "medium",
    },
    # ---- 阶段二一抢 ----
    {
        "key": ("pick_late_1", 1),
        "title": "对方露出三个 pick，这是最有效的针对窗口",
        "trigger": [["bp_phase", "==", "pick_late_1"],
                    ["opp_picks_before", "==", 3]],
        "expect": list(_COUNTER_PICKS),
        "scope": "per_frame",
        "severity": "high",
    },
    {
        "key": ("pick_late_1", 2),
        "title": "对方缺前排，拿个能扛的",
        "trigger": [["bp_phase", "==", "pick_late_1"],
                    ["opp_missing_frontline", "==", True]],
        "expect": ["pick_frontline", "pick_defensive"],
        "scope": "per_frame",
        "severity": "medium",
    },
    # ---- 阶段二末选 ----
    {
        "key": ("pick_last", 1),
        "title": "还缺一个位置，末选先补缺口不看对方",
        "trigger": [["bp_phase", "==", "pick_last"],
                    ["my_picks_before", "==", 4]],
        "expect": ["pick_frontline", "pick_engage", "pick_protect", "pick_damage"],
        "scope": "per_frame",
        "severity": "high",
    },
    {
        "key": ("pick_last", 2),
        "title": "阵容已齐，末选拿终极反手",
        "trigger": [["bp_phase", "==", "pick_last"],
                    ["my_picks_before", ">=", 5]],
        "expect": ["pick_counter", "pick_anti_control", "pick_heal_cut"],
        "scope": "per_frame",
        "severity": "medium",
    },
    {
        "key": ("pick_last", 4),
        "title": "有摇摆位就留到最后一手亮",
        "trigger": [["bp_phase", "==", "pick_last"], ["my_has_flex", "==", True]],
        "expect": ["pick_flex"],
        "scope": "per_frame",
        "severity": "low",
        "note": "「对面还没识别」指的是对方没摸清你的分路；系统能判的是"
                "「己方确实有摇摆位」，另一半按语境取末选环节。",
    },
    # ---- 阵容完整性检查 ----
    {
        "key": ("final_check", 1),
        "title": "收尾逐条核对阵容四要素",
        "trigger": [["bp_phase", "==", "final_check"],
                    ["my_picks_before", ">=", 5]],
        "expect": ["check_comp"],
        "scope": "per_frame",
        "severity": "medium",
    },
    {
        "key": ("final_check", 2),
        "title": "全队五个输出位，得换掉一个",
        "trigger": [["bp_phase", "==", "final_check"], ["my_all_output", "==", True]],
        "expect": ["check_comp"],
        "scope": "at_least_once",
        "severity": "high",
        "note": "「全队五个输出位」是收尾检查的**发现**，改法是改选一个坦克或战士；"
                "系统判的是「有没有做收尾检查」，具体换谁由玩家定。",
    },
    {
        "key": ("final_check", 5),
        "title": "双刺客阵容，前期要反野、中期执行 131",
        "trigger": [["bp_phase", "==", "final_check"],
                    ["my_assassin_count", "==", 2]],
        "expect": ["check_comp"],
        "scope": "at_least_once",
        "severity": "medium",
        "note": "攻略给的是局内打法（2 分钟反野、131 战术），BP 阶段能判的只有"
                "「你选了双刺客这件事有没有被检查过」，所以期望动作是收尾检查。",
    },
]

# --------------------------------------------------------------------------
# 分支 -> 不入库（写明理由）
# --------------------------------------------------------------------------

#: 键是 (环节编码, 分支序号)；值是**为什么这条进不去**。
#:
#: 理由要说清缺的是哪个事实 —— 「判不了」是结论，「缺什么」才是可执行的结论。
SKIPPED: Dict[Tuple[str, int], str] = {
    ("bp_start", 1): "前提要「双方均为钻石以上排位或巅峰赛」—— 需要段位事实。"
                     "BP 记录页与复盘页都不再问段位，判定层也不再登记这条事实："
                     "留着就是一条永远判不出结果的规则。"
                     "「进入 BP」这一环因此整个撤下 —— 记录页不再有这一环。",
    ("bp_start", 2): "前提要「己方无人预选，或预选互相冲突」—— 需要预选冲突事实。"
                     "记录页不再问预选，这条事实也不再登记。"
                     "「进入 BP」这一环因此整个撤下 —— 记录页不再有这一环。",
    ("bp_start", 3): "前提要「低分段（钻石以下）」—— 需要段位事实。"
                     "记录页不再问段位，这条事实也不再登记。"
                     "「进入 BP」这一环因此整个撤下 —— 记录页不再有这一环。",
    ("ban_1", 1): "前提要「版本非 Ban 必选」的先手英雄 —— 需要版本 Ban 率数据，"
                  "系统里没有这个事实；硬编一张表就是替版本下判断。",
    ("ban_1", 2): "前提要判断对方三人的预选「指向同一套体系」—— 需要英雄到体系"
                  "（冲脸/消耗/拉扯/单带/后期）的归属表，攻略只给了体系定义，"
                  "没给英雄归属。",
    ("ban_1", 4): "前提要判断敌方阵容是不是「回复型」—— 回复型是玩法描述，"
                  "官方英雄资料里没有这个属性。",
    ("ban_1", 5): "前提要看对方某个 ID 的常用英雄与省标国标 —— 需要对手账号"
                  "与熟练度数据，系统完全没有。",
    ("pick_1", 3): "前提要判断「对方突进刺客数量多」—— 刺客定位官方表有，"
                   "但「突进」要位移与进场方式，官方资料里没有。",
    ("pick_1", 4): "前提要看「自己只有一个人会玩某个英雄且熟练度最高」—— "
                   "需要选手熟练度数据，系统没有。",
    ("pick_2", 2): "前提要判断「无位移站桩射手」—— 射手定位有，位移属性官方"
                  "资料里没有。",
    ("pick_2", 3): "前提「依赖技能连招的英雄」是玩法描述，判不了。",
    ("pick_2", 4): "前提要判断对方是不是「回复型阵容」—— 判不了。",
    ("pick_2", 5): "前提要判断对方是不是「消耗型阵容」—— 判不了。",
    ("pick_3", 4): "前提要判断「对面控制链很足」—— 需要控制型强度的统计口径。",
    ("pick_3", 5): "前提要判断「对方已经成体系且拆不动」—— 需要体系归属表。",
    ("pick_late_1", 3): "前提要判断「对方是后期阵容且已成型」—— 后期阵容是"
                        "发育形态描述，判不了。",
    ("pick_late_1", 4): "前提要判断「对方是前期入侵阵容」—— 入侵型要具体玩法"
                        "归类，判不了。",
    ("pick_last", 3): "前提要判断「对方五个人都是硬控阵容」—— 硬控强度没有"
                      "可查的客观口径。",
    ("pick_last", 5): "前提是「对方最后一手准备反制自己」—— 对手意图，"
                      "系统无从知道。",
    ("final_check", 3): "前提要判断「全队五个近战或五个远程」—— 近战/远程"
                        "属性官方资料里没有。",
    ("final_check", 4): "前提要判断「只有一个惩戒携带者且是脆皮」—— 脆皮与"
                        "携带者都要额外数据表。",
}
