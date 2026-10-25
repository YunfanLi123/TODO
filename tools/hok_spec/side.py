"""对抗路分路的映射表。

时间轴：0:00 开局 → 2:00 → 4:00 → 6:00 → 10:00 → 12:00 之后。

对抗路原文里有两个贯穿全文的状态：【对线优势】/【对线劣势】，
它们对应系统的 ``matchup``（advantage / disadvantage）——
这是本表最主要的事实近似。同格里的规则用 ``matchup`` 三分
（advantage / disadvantage / unknown）互斥开，避免玩家被两条同时指控。
"""

from __future__ import annotations

LANE = "side"

RULES = [
    # ================================================== 0:00 开局
    {
        "node": "0:00",
        "key": "1 级换血后对方血量低于我方且其位移未交",
        "extra": [{"fact": "matchup", "op": "==", "value": "advantage"}],
        "title": "对抗路开局：换血占优就抢线压线",
        "expect": {"fact": "positioning_action",
                   "values": ["push_wave"]},
        "subject": {"fact": "positioning_action",
                    "values": ["push_wave", "pull_back", "hold_position",
                               "ward_vision", "rotate_support"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "1 级换血占优就抢线建立线权，2 级继续压线，只求把对方位移逼出来，"
                "不追求换血收益。",
        "why": "这一支的结果是【对线优势】，用 对位优劣 == 优势 近似；"
               "「逼位移」交的是哪个技能判不了，写在建议里。",
        "tags": ["对抗路", "前期", "抢线"],
    },
    {
        "node": "0:00",
        "key": "1 级换血后对方血量高于我方",
        "extra": [{"fact": "matchup", "op": "==", "value": "disadvantage"}],
        "title": "对抗路开局：换血吃亏就塔下补刀",
        "expect": {"fact": "objective_action",
                   "values": ["last_hit"]},
        "subject": {"fact": "objective_action",
                    "values": ["last_hit", "farm_jungle", "recall",
                               "take_river_spirit", "contest_dragon",
                               "take_dragon"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "1 级换血吃亏就退回塔下补刀，把兵线卡在塔前等打野，不贪战换血。",
        "why": "这一支的结果是【对线劣势】，用 对位优劣 == 劣势 近似；"
               "「等打野」是配合，判不了，这条只判有没有在塔下稳住补刀。",
        "tags": ["对抗路", "前期", "补刀"],
    },
    {
        "node": "0:00",
        "key": "对方一级埋伏在我方兵线侧草丛",
        "guards": [{"fact": "matchup", "op": "!=", "value": "disadvantage"}],
        "title": "对抗路开局：对方一级蹲草就试探草丛或走线上，不进草赌",
        "expect": {"fact": "objective_action",
                   "values": ["last_hit"]},
        "subject": {"fact": "objective_action",
                    "values": ["last_hit", "farm_jungle", "recall",
                               "take_river_spirit", "contest_dragon",
                               "take_dragon"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "对方一级可能蹲在我方兵线侧草丛：用技能试探草丛，"
                "或者直接走线上补刀，别进草丛赌，交出一血就成劣势开局。",
        "why": "「草丛里有没有埋伏」数据里读不到，「用技能试探」也没有对应的动作取值，"
               "所以这条只判走线上的那一半：有没有继续在线上补刀、没交头。"
               "对位劣势时由「塔下补刀」那条覆盖，这里排除劣势，"
               "免得同一格两条规则重复指控补刀。",
        "tags": ["对抗路", "前期", "防埋伏"],
    },

    # ================================================== 2:00
    {
        "node": "2:00",
        "key": "【对线优势】",
        "extra": [{"fact": "matchup", "op": "==", "value": "advantage"}],
        "title": "对抗路两分钟：优势推线后往河道蟹或传送阵走",
        "expect": {"fact": "positioning_action",
                   "values": ["push_wave", "rotate_support"]},
        "subject": {"fact": "positioning_action",
                    "values": ["push_wave", "pull_back", "hold_position",
                               "ward_vision", "rotate_support"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "抢完线把兵线推到对方塔下；走河道前先确认这条线兵线已过河、"
                "己方塔前没有兵，再往河道蟹或传送阵方向走。",
        "why": "【对线优势】用 对位优劣 == 优势 近似；"
               "「兵线已过河、塔前无兵」是判不了的细节，写在建议里。",
        "tags": ["对抗路", "前期", "推线"],
    },
    {
        "node": "2:00",
        "key": "【对线劣势】",
        "extra": [{"fact": "matchup", "op": "==", "value": "disadvantage"}],
        "title": "对抗路两分钟：劣势塔下清线并做视野",
        "expect": {"fact": "positioning_action",
                   "values": ["ward_vision"]},
        "subject": {"fact": "positioning_action",
                    "values": ["ward_vision", "push_wave", "pull_back",
                               "hold_position", "rotate_support"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "劣势就在塔下清线，兵线别推过前，再去己方野区做视野，防止对方抓单。",
        "why": "【对线劣势】用 对位优劣 == 劣势 近似。"
               "同格「优势推线转线」那条要求 对位优劣 == 优势，"
               "两条互斥，不会同时指控同一个玩家。",
        "tags": ["对抗路", "前期", "视野"],
    },
    {
        "node": "2:00",
        "key": "敌方三人以上出现在我方防御塔范围内",
        "title": "对抗路两分钟：三人压塔就退塔下不交位移",
        "expect": {"fact": "teamfight_action",
                   "values": ["retreat"]},
        "subject": {"fact": "teamfight_action",
                    "values": ["engage", "focus_target", "protect_carry",
                               "retreat", "kite", "split_fight"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "敌方三人以上压到我方防御塔范围，就退塔下保住人头，不交位移，"
                "等队友到位。",
        "why": "「敌方来了几个人、在不在塔下」数据里读不到，"
               "所以判在参与团战的回合上：这时候该做的是脱离，别拿人头换技能。",
        "tags": ["对抗路", "前期", "避战"],
    },
    {
        "node": "2:00",
        "key": "需要绕到敌方一塔后",
        "title": "对抗路两分钟：需要绕后就绕后断对方兵线",
        "expect": {"fact": "objective_action",
                   "values": ["last_hit"]},
        "subject": {"fact": "objective_action",
                    "values": ["last_hit", "farm_jungle", "recall",
                               "take_river_spirit", "contest_dragon",
                               "take_dragon", "split_push"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "需要绕到敌方一塔后就绕后断对方兵线，多出 10 到 20 秒支援时间，"
                "限制对方在守塔与支援之间抉择。",
        "why": "「需不需要绕后」是攻略给的时机条件，数据里读不到，"
               "所以这条只判绕后时有没有断线补刀。",
        "tags": ["对抗路", "前期", "绕后"],
    },

    # ================================================== 4:00
    {
        "node": "4:00",
        "key": "4 级且大招在手 + 敌方打野不在我方半区",
        # 原文这一支明写「**大招在手**」—— 大招亮不亮在画面里读得出，
        # 不依赖任何数值。「4 级」是数值（读不出），「敌方打野位置」读不出，
        # 都不拿来当纠正依据。
        "extra": [
            {"fact": "matchup", "op": "==", "value": "advantage"},
            {"fact": "has_ult_ready", "op": "==", "value": True},
        ],
        "title": "对抗路四分钟：优势压线逼出对方位移",
        "expect": {"fact": "positioning_action",
                   "values": ["push_wave"]},
        "subject": {"fact": "positioning_action",
                    "values": ["push_wave", "pull_back", "hold_position",
                               "ward_vision", "rotate_support"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "4 级且大招在手、对方打野不在我方半区时压线消耗，"
                "只求逼出对方位移与闪现，不追求换血收益。",
        "why": "原文这一支的前提是「4 级且大招在手」加「敌方打野位置」，"
               "后者读不到；对位优势（对位优劣 == 优势）是最接近的近似，"
               "判的是有没有压线逼位移。",
        "tags": ["对抗路", "前期", "压线"],
    },
    {
        "node": "4:00",
        "key": "敌方打野在我方半区",
        "merge": ["想补经济去中路 / 中路有人且正在赶回"],
        "extra": [{"fact": "matchup", "op": "==", "value": "disadvantage"}],
        "title": "对抗路四分钟：劣势收缩，不压线也不单走中路",
        "expect": {"fact": "positioning_action",
                   "values": ["pull_back", "hold_position"]},
        "subject": {"fact": "positioning_action",
                    "values": ["pull_back", "hold_position", "push_wave",
                               "ward_vision", "rotate_support"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "对方打野在我方半区就回塔下、放弃压线，保住血量与技能；"
                "想补经济去中路时，中路还有人正赶回来就别一个人去吃线，留在原地等队友。",
        "why": "这两支的共同点是「不去」：敌方打野在我方半区就收缩，"
               "中路有人赶回就不单走中路；可接受动作有交集（都是留在塔下卡位不追），"
               "所以合成一条：对位劣势（对位优劣 == 劣势）时判，"
               "回撤或卡位都算做到。",
        "tags": ["对抗路", "前期", "收缩"],
    },
    {
        "node": "4:00",
        "key": "传送阵已激活，图标出现在小地图 + 队友在另一侧",
        "extra": [{"fact": "matchup", "op": "==", "value": "unknown"}],
        "title": "对抗路四分钟：传送阵激活就交传送支援",
        "expect": {"fact": "positioning_action",
                   "values": ["rotate_support"]},
        "subject": {"fact": "positioning_action",
                    "values": ["rotate_support", "push_wave", "pull_back",
                               "hold_position", "ward_vision"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "传送阵已激活、小地图出现图标、队友在另一侧，就交传送支援过去形成夹击。",
        "why": "「传送阵有没有激活」只能靠小地图图标识别，数据里读不到，"
               "「队友在哪一侧」也读不到，所以这条只判有没有转线支援；"
               "为不和同格另两条（优势压线 / 劣势收缩）互相指控，"
               "限定在对位不确定时判。",
        "tags": ["对抗路", "前期", "传送支援"],
    },
    {
        "node": "4:00",
        "key": "想补经济去中路 / 中路阵亡或赶不回来",
        "title": "对抗路四分钟：中路空了就转去中路吃线",
        "expect": {"fact": "objective_action",
                   "values": ["last_hit"]},
        "subject": {"fact": "objective_action",
                    "values": ["last_hit", "farm_jungle", "recall",
                               "take_river_spirit", "contest_dragon",
                               "take_dragon", "split_push"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "中路阵亡或赶不回来，就去中路吃线补上中路的空档。",
        "why": "「中路是不是空着」数据里读不到，所以这条只判有没有补线补刀；"
               "「吃线」在词表里落到补刀，转线动作写在建议里。",
        "tags": ["对抗路", "前期", "补经济"],
    },
    {
        "node": "4:00",
        "key": "对方关键控制未交 + 己方队友未全员就位",
        "title": "对抗路四分钟：控制没交队友没齐就不开团",
        "expect": {"fact": "teamfight_action",
                   "values": ["retreat", "kite"]},
        "subject": {"fact": "teamfight_action",
                    "values": ["engage", "focus_target", "protect_carry",
                               "retreat", "kite", "split_fight"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "对方关键控制没交、我方队友没到齐，就不开团，"
                "等控制交完、队友到位再进。",
        "why": "「控制交没交、队友到没到齐」数据里读不到，"
               "所以判在参与团战的回合上：这时候该做的是不进团，往后撤或拉扯。",
        "tags": ["对抗路", "前期", "不开团"],
    },

    # ================================================== 6:00
    {
        "node": "6:00",
        "key": "敌方五人位置集中 + 己方打野位置可确认",
        "title": "对抗路六分钟：敌方集中就参团并回防射手",
        "expect": {"fact": "teamfight_action",
                   "values": ["engage", "protect_carry"]},
        "subject": {"fact": "teamfight_action",
                    "values": ["engage", "focus_target", "protect_carry",
                               "retreat", "kite", "split_fight"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "敌方五人位置集中、我方打野位置能确认，就参团，等对方控制交完再进场；"
                "坦边开团后第一动作是回防保护己方射手。",
        "why": "「敌方五人位置」「我方打野位置」都读不到，"
               "所以判在参与团战的回合上：这一支要的就是参团开团、并把射手护住。",
        "tags": ["对抗路", "中期", "参团"],
    },
    {
        "node": "6:00",
        "key": "敌方五人位置集中 + 己方打野位置无法确认",
        "merge": ["敌方分散在各线"],
        "title": "对抗路六分钟：打不团就继续带线牵制",
        "expect": {"fact": "objective_action",
                   "values": ["split_push"]},
        "subject": {"fact": "objective_action",
                    "values": ["split_push", "siege", "farm_jungle", "recall",
                               "last_hit", "take_river_spirit",
                               "contest_dragon", "take_dragon"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "敌方五人集中而我方打野位置确认不了，就不参团，继续带线；"
                "敌方分散在各线时同样带线牵制，逼对方回防。",
        "why": "这两支的期望动作都是「去带线」：敌方集中时用带线换掉被以多打少，"
               "敌方分散时用带线逼回防。可接受动作相同，所以合成一条，"
               "分推牵制就算做到。",
        "tags": ["对抗路", "中期", "带线"],
    },
    {
        "node": "6:00",
        "key": "带线时敌方 2 至 3 人出现在我方兵线前方",
        "title": "对抗路六分钟：兵线前有敌人就交位移撤",
        "expect": {"fact": "positioning_action",
                   "values": ["pull_back"]},
        "subject": {"fact": "positioning_action",
                    "values": ["pull_back", "push_wave", "hold_position",
                               "ward_vision", "rotate_support"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "带线时敌方 2 至 3 人出现在我方兵线前方，交位移撤离，"
                "放弃这条线，等级不落后。",
        "why": "「来了几个人」读不到，所以判在有站位动作的回合上："
               "这时候该做的是回撤，别硬守这条线；交哪个位移判不了，写在建议里。",
        "tags": ["对抗路", "中期", "撤离"],
    },

    # ================================================== 10:00
    {
        "node": "10:00",
        "key": "敌方五人集中在龙坑",
        "merge": ["敌方分散"],
        "title": "对抗路十分钟：敌方抱团就带线拆二塔",
        "expect": {"fact": "objective_action",
                   "values": ["split_push", "siege"]},
        "subject": {"fact": "objective_action",
                    "values": ["split_push", "siege", "push_wave",
                               "contest_dragon", "take_dragon", "fight_baron",
                               "farm_jungle", "invade_jungle"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "敌方五人集中在龙坑就不进龙坑，带线拆对方二塔或高地外塔；"
                "敌方分散时同样带线牵制，优先边路二塔，逼对方回防。",
        "why": "两支的期望动作都是「带线 + 拆塔」：敌方抱团时用塔换龙，"
               "敌方分散时用带线逼回防，所以合成一条，分推牵制或拆塔都算做到。",
        "tags": ["对抗路", "中期", "带线拆塔"],
    },
    {
        "node": "10:00",
        "key": "自己被对方切",
        "title": "对抗路十分钟：被切就交位移撤，别阵亡",
        "expect": {"fact": "teamfight_action",
                   "values": ["retreat"]},
        "subject": {"fact": "teamfight_action",
                    "values": ["engage", "focus_target", "protect_carry",
                               "retreat", "kite", "split_fight"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "被对方切就交位移撤，团战交给队友，不阵亡、保留技能 CD。",
        "why": "「有没有被切」读不到，所以判在参与团战的回合上："
               "这时候该做的是脱离战场，技能留着下一波。",
        "tags": ["对抗路", "中期", "保命"],
    },

    # ================================================== 12:00 之后
    {
        "node": "12:00",
        "key": "视野精灵仍在边线附近",
        "title": "对抗路后期：边线有视野精灵就先清掉",
        "expect": {"fact": "objective_action",
                   "values": ["clear_vision_sprite"]},
        "subject": {"fact": "objective_action",
                    "values": ["clear_vision_sprite", "split_push", "siege",
                               "recall", "last_hit", "farm_jungle",
                               "take_river_spirit"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "边线附近还有视野精灵，就先清掉或把它驱离，再推线，"
                "别在被照出之后被包。",
        "why": "「精灵还在不在边线」读不到，所以判在有目标处理的回合上；"
               "这条只判有没有清视野精灵，顺序上应当先清再推。",
        "tags": ["对抗路", "后期", "视野精灵"],
    },
    {
        "node": "12:00",
        "key": "视野精灵已清除 + 己方主力在对面 + 己方需要完整人数",
        "title": "对抗路后期：人不够就回队伍集合",
        "expect": {"fact": "positioning_action",
                   "values": ["rotate_support"]},
        "subject": {"fact": "positioning_action",
                    "values": ["rotate_support", "push_wave", "pull_back",
                               "hold_position", "ward_vision"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "视野精灵已清除、己方主力在对面、己方需要完整人数时，"
                "就不深推孤线，回队伍集合，保留人数优势。",
        "why": "原文这一支的前提（谁减员、谁领先）在画面里读不出来，所以不拿它当纠正依据：只有当玩家**整段一次都没做过**这个时段攻略给的任何一件事时，这条才判偏离。"
               "「精灵是否已清除」「主力在哪一侧」「人数够不够」都读不到，写在建议里。"
               "同节点「守得住就深推」那条是同一时段的另一种打法 —— "
               "回队伍集合与深推最远边线，玩家做了任一件都不说他错。",
        "tags": ["对抗路", "后期", "集合"],
    },
    {
        "node": "12:00",
        "key": "视野精灵已清除 + 己方主力在对面 + 己方已能稳守高地",
        "title": "对抗路后期：守得住就深推最远边线",
        "expect": {"fact": "positioning_action",
                   "values": ["push_wave"]},
        "subject": {"fact": "positioning_action",
                    "values": ["push_wave", "rotate_support", "pull_back",
                               "hold_position", "ward_vision"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "视野精灵已清除、己方主力在对面、己方已能稳守高地，"
                "就深推最远边线，一路推到对方高地换塔的经济。",
        "why": "原文这一支的前提（谁减员、谁领先）在画面里读不出来，所以不拿它当纠正依据：只有当玩家**整段一次都没做过**这个时段攻略给的任何一件事时，这条才判偏离。"
               "「守不守得住高地」读不到，写在建议里。"
               "同节点「人不够就回队伍集合」那条是同一时段的另一种打法。",
        "tags": ["对抗路", "后期", "深推"],
    },
    {
        "node": "12:00",
        "key": "己方需要回防",
        "title": "对抗路后期：回防时留在队伍中间护后排",
        "expect": {"fact": "teamfight_action",
                   "values": ["protect_carry"]},
        "subject": {"fact": "teamfight_action",
                    "values": ["engage", "focus_target", "protect_carry",
                               "retreat", "kite", "split_fight"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "己方需要回防时，留在队伍中间威慑对方刺客，不主动开团，保住己方后排。",
        "why": "「己方需不需要回防」读不到，所以判在参与团战的回合上："
               "这时候该做的是护住后排，不是自己先开团。",
        "tags": ["对抗路", "后期", "回防"],
    },
]

#: 明确不入库的分支及理由。
SKIPPED = [
    {
        "node": "4:00",
        "key": "想补经济去中路",
        "reason": "这是嵌套分支的父节点，本身没有动作，只把「想补经济去中路」这个场景"
                  "分成两个子分支；判定由两个子分支各自的规则承担"
                  "（中路有人赶回 → 不单走，中路空了 → 去吃线）。",
    },
]
