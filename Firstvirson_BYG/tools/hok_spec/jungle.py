"""打野分路的映射表。

时间轴：0:00 开局 → 0:30-1:00 第一分钟 → 2:00 → 4:00 → 6:00 → 10:00 → 12:00 之后。
"""

from __future__ import annotations

LANE = "jungle"

RULES = [
    # ================================================== 0:00 开局
    {
        "node": "0:00",
        "key": "敌方打野为红开（已知或从预选推测）",
        "merge": [
            "敌方打野为蓝开",
            "己方打野前期强度弱于对方",
            "对方已进我方 BUFF 区且血量高于我方惩击可收割线",
        ],
        "title": "打野开局：反向开局刷野，1:50 前到 4 级",
        "expect": {"fact": "objective_action",
                   "values": ["farm_jungle", "invade_jungle"]},
        "subject": {"fact": "objective_action",
                    "values": ["farm_jungle", "invade_jungle", "clear_vision_sprite"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "敌方红开就己方蓝开（蓝 BUFF→蜥蜴→狼→红 BUFF→小鸟），"
                "敌方蓝开就己方红开（红 BUFF→小鸟→猪→蓝 BUFF），1:15 至 1:50 到 4 级；"
                "33 秒进野区、进场手里捏着惩。野区已被对方先占就不抢 BUFF，"
                "改入侵对方同侧野区对换；我方前期弱于对方时不接战，直接换到另一半野区。",
        "why": "这四支的前提（对方开哪个野色、我方强弱、BUFF 有没有被占）都只能从选人预选"
               "推断，对局数据里没有，所以合成一条：这一分钟打野在野区做事就算数，"
               "但只清了个视野精灵不算刷野。",
        "tags": ["打野", "前期", "开局"],
    },

    # ================================================== 2:00
    {
        "node": "2:00",
        "key": "【野区均势】+ 暴君已被我方击杀",
        "merge": [
            "【野区均势，无龙 buff】+ 对方正在打暴君",
            "【野区劣势】",
        ],
        "title": "打野两分钟：均势去反野，劣势守自家野区",
        "expect": {"fact": "objective_action",
                   "values": ["invade_jungle", "farm_jungle"]},
        "subject": {"fact": "objective_action",
                    "values": ["farm_jungle", "invade_jungle", "contest_dragon",
                               "take_dragon"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "暴君已被我方击杀就入侵对方野区，或帮线上推塔；对方正在打暴君就绕过龙坑"
                "进对方野区；野区劣势就守自家野区，等敌方分散后单抓，等级追平前不进 5v5。",
        "why": "【野区均势】【野区劣势】是攻略内部状态，对局数据里没有，"
               "所以三支合并：反野或守自家野区刷野都算做到。",
        "tags": ["打野", "前期", "反野"],
    },

    # ================================================== 4:00
    {
        "node": "4:00",
        "key": "塔的早期护盾已消失 + 敌方打野不在我方半区",
        "title": "打野四分钟：带线推塔，顺路做敌方半区视野",
        "expect": {"fact": "positioning_action",
                   "values": ["push_wave", "ward_vision"]},
        "subject": {"fact": "positioning_action",
                    "values": ["push_wave", "ward_vision", "pull_back",
                               "hold_position"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "带兵线推塔，推完顺路在敌方半区做视野。推塔顺序：边路一塔 → 中路一塔与"
                "边路二塔 → 中路二塔与边路二塔 → 高地。",
        "why": "原文的前提是塔的护盾状态和敌方打野位置，数据里都没有；"
               "这条只判「该推塔时有没有推塔、有没有顺手做视野」。",
        "tags": ["打野", "前期", "推塔"],
    },
    {
        "node": "4:00",
        "key": "暴君倒计时进入最后一段 + 龙坑有我方视野",
        "merge": ["暴君倒计时进入最后一段 + 龙坑无我方视野 + 视野精灵在龙坑附近"],
        "title": "打野四分钟：龙坑最后一刻交惩，视野有无都去抢",
        "expect": {"fact": "objective_action",
                   "values": ["contest_dragon", "take_dragon"]},
        "subject": {"fact": "objective_action",
                    "values": ["contest_dragon", "take_dragon", "fight_baron",
                               "farm_jungle", "invade_jungle"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "龙坑有我方视野就集合占位、惩击留到龙最后一刻再交；没有视野就抢时间差，"
                "在对方视野盲区的窗口里打完。",
        "why": "有视野 / 无视野两支的差别只在细节，期望的动作一样（打龙），"
               "所以合成一条。",
        "tags": ["打野", "前期", "抢龙"],
    },
    {
        "node": "4:00",
        "key": "团战中对方先打出控制并命中我方前排",
        "title": "打野四分钟：对方先交控制就不进团，等第二轮",
        "expect": {"fact": "teamfight_action",
                   "values": ["retreat", "kite"]},
        "subject": {"fact": "teamfight_action",
                    "values": ["engage", "focus_target", "protect_carry",
                               "split_fight", "retreat", "kite"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "团战里对方先打出控制并命中我方前排时，不进入，等对方控制交完再打第二轮。",
        "why": "「对方控制交没交」数据里读不到，所以判在参与团战的回合上："
               "这时该做的是不接团。",
        "tags": ["打野", "前期", "团战"],
    },

    # ================================================== 6:00
    {
        "node": "6:00",
        "key": "双方五人都在场 + 敌方有人阵亡未复活",
        "title": "打野六分钟：敌方减员就主动开团",
        "expect": {"fact": "teamfight_action", "values": ["engage"]},
        "subject": {"fact": "teamfight_action",
                    "values": ["engage", "focus_target", "protect_carry",
                               "split_fight", "retreat", "kite"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "敌方有人阵亡未复活就趁人数缺失主动开团，随后控龙推塔，"
                "以多打少让对方减员期间接不了团。",
        "why": "原文这一支的前提（谁减员、谁领先）在画面里读不出来，所以不拿它当纠正依据：只有当玩家**整段一次都没做过**这个时段攻略给的任何一件事时，这条才判偏离。",
        "tags": ["打野", "中期", "开团"],
    },
    {
        "node": "6:00",
        "key": "双方五人都在场 + 无人阵亡",
        "title": "打野六分钟：人齐不先开团，用龙和塔逼对方动",
        "expect": {"fact": "objective_action",
                   "values": ["contest_dragon", "take_dragon"]},
        "subject": {"fact": "objective_action",
                    "values": ["contest_dragon", "take_dragon", "fight_baron",
                               "farm_jungle", "split_push", "siege"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "双方五人都在场又无人阵亡时，不主动开团，以龙逼团、以塔逼团，"
                "迫使对方为争资源移动阵型后再进。",
        "why": "原文这一支的前提（谁减员、谁领先）在画面里读不出来，所以不拿它当纠正依据：只有当玩家**整段一次都没做过**这个时段攻略给的任何一件事时，这条才判偏离。"
               "同节点「落后就只刷安全野区」那条是同一件事的另一种打法，"
               "玩家做了其中任一件，这一条都不说他错。",
        "tags": ["打野", "中期", "逼团"],
    },
    {
        "node": "6:00",
        "key": "己方有人阵亡未复活",
        "title": "打野六分钟：我方减员就不接团，清线控龙",
        "expect": {"fact": "teamfight_action", "values": ["retreat"]},
        "subject": {"fact": "teamfight_action",
                    "values": ["engage", "focus_target", "protect_carry",
                               "split_fight", "retreat", "kite"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "己方有人阵亡未复活时不接团，清线、控龙、缩小活动范围，"
                "避免二次减员，等队友复活。",
        "why": "原文这一支的前提（谁减员、谁领先）在画面里读不出来，所以不拿它当纠正依据：只有当玩家**整段一次都没做过**这个时段攻略给的任何一件事时，这条才判偏离。"
               "同节点「敌方减员就主动开团」那条说的是同一场团战的另一种打法，"
               "玩家做了其中任一件，这一条都不说他错。",
        "tags": ["打野", "中期", "避战"],
    },
    {
        "node": "6:00",
        "key": "己方处于落后",
        "title": "打野六分钟：落后就只刷安全野区",
        "expect": {"fact": "objective_action", "values": ["farm_jungle"]},
        "subject": {"fact": "objective_action",
                    "values": ["farm_jungle", "invade_jungle", "contest_dragon",
                               "take_dragon", "split_push"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "己方处于落后时只刷己方安全野区，gank 己方优势路而不是劣势路，"
                "放弃争不到的龙，用近侧两座塔换远侧资源，等对方失误翻盘。",
        "why": "原文这一支的前提（谁减员、谁领先）在画面里读不出来，所以不拿它当纠正依据：只有当玩家**整段一次都没做过**这个时段攻略给的任何一件事时，这条才判偏离。"
               "同节点「人齐就不先开团、以龙逼团」那条是同一时段的另一种打法。",
        "tags": ["打野", "中期", "逆风"],
    },

    # ================================================== 10:00
    {
        "node": "10:00",
        "key": "我方先到龙坑",
        "merge": ["敌方先到龙坑"],
        "title": "打野十分钟：先到开龙，被占就转推边路塔",
        "expect": {"fact": "objective_action",
                   "values": ["contest_dragon", "take_dragon", "fight_baron",
                              "split_push", "siege"]},
        "subject": {"fact": "objective_action",
                    "values": ["contest_dragon", "take_dragon", "fight_baron",
                               "split_push", "siege", "farm_jungle", "invade_jungle"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "我方先到龙坑就开龙（拿到主宰先锋，拆边路二塔）；敌方先到就放弃龙，"
                "转推边路塔，用塔经济换时间。",
        "why": "「谁先到龙坑」数据里读不到，两支的期望动作都是「打资源或拆塔」，"
               "所以合成一条。",
        "tags": ["打野", "中期", "主宰"],
    },
    {
        "node": "10:00",
        "key": "双方同时抵达、均未开龙",
        "title": "打野十分钟：龙坑外埋伏，让游走探草",
        "expect": {"fact": "positioning_action",
                   "values": ["hold_position", "ward_vision"]},
        "subject": {"fact": "positioning_action",
                    "values": ["hold_position", "ward_vision", "pull_back",
                               "push_wave", "rotate_support"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "双方同时抵达、均未开龙时，由打野在龙坑外埋伏，制造要开龙的假象，"
                "让游走探草确认对方后排位置；对方进坑后被包，或拿到视野差后再决定开或撤。",
        "tags": ["打野", "中期", "埋伏"],
    },

    # ================================================== 12:00 之后
    {
        "node": "12:00",
        "key": "【有主宰】",
        "merge": ["对方在另一侧打龙", "双方无主宰且己方五人齐"],
        "title": "打野后期：有主宰带线拆塔，没主宰就开龙",
        "expect": {"fact": "objective_action",
                   "values": ["split_push", "siege", "contest_dragon", "take_dragon",
                              "fight_baron"]},
        "subject": {"fact": "objective_action",
                    "values": ["split_push", "siege", "contest_dragon", "take_dragon",
                               "fight_baron", "farm_jungle", "invade_jungle"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "有主宰就带线拆塔推进敌方高地；双方无主宰且己方五人齐就开龙，"
                "惩击留到最后一刻，拿到后立刻转推；对方在另一侧打龙就不去争，"
                "去推塔或打另一个资源，用地图压力换掉这次龙。",
        "why": "主宰归属（我方 / 对方 / 都没有）数据里读不到，三支的期望动作都是"
               "「打资源或拆塔」，所以合成一条。",
        "tags": ["打野", "后期", "推进"],
    },
    {
        "node": "12:00",
        "key": "对方有主宰",
        "title": "打野后期：对方有主宰就守高地、优先清龙兵线",
        "expect": {"fact": "positioning_action", "values": ["hold_position"]},
        # 适用时机只圈「防守类」动作。
        #
        # 这一支的前提是「对方有主宰」，数据里读不到；同节点另一支是
        # 「我方有主宰 → 带线拆塔」。前提分不清时，两条规则若都对推线
        # 的那一帧生效，就会一边说「该拆塔」一边说「该守塔」。
        # 把这条的适用时机收窄到防守动作上，推线那一帧就只由「拆塔」那条判。
        "subject": {"fact": "positioning_action",
                    "values": ["hold_position", "pull_back"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "对方有主宰就守高地塔，优先清龙兵线，拖到队友复活。",
        "why": "「对方有主宰」数据里读不到，这条按「后期该守还是该推」来判："
               "在守高地（卡位）就算守住；撤退不算守。",
        "tags": ["打野", "后期", "防守"],
    },
]

#: 明确不入库的分支及理由。
SKIPPED = [
    {
        "node": "0:30-1:00",
        "key": "中单与游走已到位",
        "reason": "这一分钟的核心决策对象是「河道之灵」（河蟹），"
                  "S36（2024-06）起已从中路移除、对抗路换成空间之灵；"
                  "当前版本第 1 分钟没有任何河道目标（空间之灵 2:00、暴君/主宰 4:00），"
                  "原文这一节的决策在 S45 已经不成立，所以整条不入库。",
    },
    {
        "node": "0:30-1:00",
        "key": "中单或游走未到位",
        "reason": "同上：这一支的落点也是「放弃河蟹、回野区刷野」，"
                  "而「放弃一个不存在的目标」没有可判定的含义，整条不入库。",
    },
    {
        "node": "4:00",
        "key": "全场决策",
        "reason": "内容是实时指挥与喊话（报敌方打野位置、报倒计时、喊开团），"
                  "属于沟通而不是玩家在地图上的操作，标注页没有对应动作，系统判不了。",
    },
]
