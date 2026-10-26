"""发育路分路的映射表。

时间轴：0:00 开局对线 → 2:00 河道与被抓 → 4:00 转线 → 6:00 团战取舍 → 10:00 打龙 → 12:00 之后。

发育路的判据和别的分路不一样：这条路的攻略大半落在**对线细节**上
（补刀、消耗完就退、被抓交位移、塔破没破转不转线），
所以这里的 expect 多数挂在「补刀 / 站位」两类动作事实上。
"""

from __future__ import annotations

LANE = "carry"

RULES = [
    # ================================================== 0:00 开局
    {
        "node": "0:00",
        "key": "对方射手在塔前补刀且其位移未交",
        "title": "发育路开局：站在兵后消耗并补刀",
        "expect": {"fact": "objective_action", "values": ["last_hit"]},
        "subject": {"fact": "objective_action",
                    "values": ["last_hit", "push_wave", "farm_jungle", "recall"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "对方射手在塔前补刀、位移还没交时，站在自己小兵身后用普攻消耗，"
                "顺便补掉这波兵，消耗完立刻退回，不追着换血。",
        "why": "「对方射手是否在补刀」「位移交没交」数据里读不到，所以这条判在发育路的"
               "补刀操作上：消耗和补刀在同一次操作里完成，补上没补上这一波兵就是判据；"
               "普攻消耗本身没有对应的动作词表，判不了的部分写在提示里。",
        "tags": ["发育路", "前期", "补刀"],
    },
    {
        "node": "0:00",
        "key": "对方射手前压出塔",
        "title": "发育路开局：对方前压就退回塔下",
        "expect": {"fact": "positioning_action", "values": ["pull_back"]},
        "subject": {"fact": "positioning_action",
                    "values": ["pull_back", "hold_position", "push_wave",
                               "rotate_support"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "对方射手前压出塔时立刻后撤到塔下，用塔下位置补刀，不上前换血，避免被开。",
        "why": "「对方射手是否前压出塔」数据里读不到，这条判在发育路的站位操作上："
               "退回塔下、放弃对拼就算做到（塔下补刀属于目标处理，由同节点的补刀规则判）。",
        "tags": ["发育路", "前期", "对线"],
    },
    {
        "node": "0:00",
        "key": "敌方刺客两名以上且已进入我方半区",
        "title": "发育路开局：敌方刺客进场就放弃对拼",
        "expect": {"fact": "positioning_action",
                   "values": ["pull_back", "hold_position"]},
        "subject": {"fact": "positioning_action",
                    "values": ["pull_back", "hold_position", "push_wave",
                               "cover_carry"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "敌方两名以上刺客进到我方半区就放弃对拼，不上前拼血、不抢人头；"
                "第一件装备改出抗性鞋或物抗，不做输出装。",
        "why": "这一支的落点是出装（抗性鞋 / 物抗），系统判不了装备，所以只判能观测的部分："
               "不上前对拼、退回或卡住自己的位置；出装这条写进提示里，不写成期望动作。",
        "tags": ["发育路", "前期", "防刺客"],
    },

    # ================================================== 2:00
    {
        "node": "2:00",
        "key": "河道路口有视野",
        "merge": ["河道路口无视野"],
        "title": "发育路两分钟：有视野控河蟹，没视野塔下补刀",
        "expect": {"fact": "objective_action",
                   "values": ["take_river_spirit", "last_hit"]},
        "subject": {"fact": "objective_action",
                    "values": ["take_river_spirit", "last_hit", "farm_jungle",
                               "contest_dragon", "take_dragon"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "河道路口有视野就和游走一起控河道蟹；没有视野就别进河道，退回塔下补刀。"
                "按你当时的视野选一条做，不要空进河道。",
        "why": "「河道路口有没有视野」数据里读不到，两支的共同底线是「不在没视野的河道里送死」，"
               "所以合成一条：控到河道之灵或塔下补刀都算做到。",
        "tags": ["发育路", "前期", "河道"],
    },
    {
        "node": "2:00",
        "key": "被抓 / 雾行千山未使用",
        "merge": ["被抓 / 雾行千山已使用"],
        "title": "发育路两分钟：被抓先交位移脱离战场",
        "expect": {"fact": "teamfight_action", "values": ["retreat"]},
        "subject": {"fact": "teamfight_action",
                    "values": ["retreat", "kite", "focus_target", "engage",
                               "split_fight"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "被抓时优先交位移脱身：雾行千山还没用就穿墙撤离，位移期闪避伤害、落地雾区减速；"
                "已经用过了就交位移或闪现，残血回城别硬撑（雾行千山 100 秒冷却内更要先撤）。",
        "why": "两个子分支的动作差别只在用哪个位移，期望动作都是「脱离战场」，所以合成一条；"
               "残血回城属于目标处理，同一条提示里写明。",
        "tags": ["发育路", "前期", "被抓"],
    },

    # ================================================== 4:00
    {
        "node": "4:00",
        "key": "我方下路外塔已破 + 敌方打野不在下路",
        "merge": ["己方经济被压", "转线途中敌方打野出现在附近"],
        "title": "发育路四分钟：转线中路补经济，遇敌先做视野",
        "expect": {"fact": "positioning_action",
                   "values": ["rotate_support", "push_wave", "ward_vision"]},
        "subject": {"fact": "positioning_action",
                    "values": ["rotate_support", "push_wave", "ward_vision",
                               "pull_back", "hold_position"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "下路外塔已破、敌方打野不在下路时转线到中路吃线磨塔，顺路收己方红区小野；"
                "经济被压时下路塔没破也去中路吃线补经济；"
                "转线途中发现敌方打野在附近就暂停转线，先做视野再走。",
        "why": "「下路外塔破没破」「敌方打野在哪」「经济被压」数据里都读不到，"
               "所以三支按可接受动作集合合并：转线支援、推线压塔、停下做视野都算做到。",
        "tags": ["发育路", "前期", "转线"],
    },
    {
        "node": "4:00",
        "key": "附近有团战且自己已就位",
        "title": "发育路四分钟：附近有团战就过去参团",
        "expect": {"fact": "teamfight_action", "values": ["engage", "focus_target"]},
        "subject": {"fact": "teamfight_action",
                    "values": ["engage", "focus_target", "kite", "split_fight",
                               "protect_carry", "retreat"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "附近有团战且自己已经就位就过去参团、跟上输出，不要站在远处等。",
        "why": "「附近有没有团战」「自己是否已就位」数据里读不到，所以这条判在参加团战的回合上："
               "进场并打输出就算做到。",
        "tags": ["发育路", "前期", "参团"],
    },
    {
        "node": "4:00",
        "key": "我方下路外塔未破",
        "title": "发育路四分钟：塔没破就继续推线",
        "expect": {"fact": "positioning_action", "values": ["push_wave"]},
        "subject": {"fact": "positioning_action",
                    "values": ["push_wave", "hold_position", "pull_back",
                               "ward_vision"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "我方下路外塔还没破就继续推线压塔，留一个身位的退路，为转线做准备。",
        "why": "「我方下路外塔未破」这个前提数据里读不到（塔差只统计总数），"
               "所以这条判在发育路的推线操作上：塔没破时把兵线推上去就算做到。",
        "tags": ["发育路", "前期", "推线"],
    },

    # ================================================== 6:00
    {
        "node": "6:00",
        "key": "团战中己方前排已与对方前排接触",
        "title": "发育路六分钟：等控制交完再集火主C",
        "expect": {"fact": "teamfight_action", "values": ["focus_target"]},
        "subject": {"fact": "teamfight_action",
                    "values": ["focus_target", "engage", "kite", "split_fight",
                               "retreat", "protect_carry"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "团战里己方前排已经接上对方前排，就等对方突进和控制交完再输出，"
                "集火对方主输出点，不分摊伤害到坦克。",
        "why": "「前排有没有接触」「对方控制交没交」数据里读不到，所以这条判在团战回合上："
               "等控制交完再集火对方主输出点就算做到。",
        "tags": ["发育路", "中期", "团战输出"],
    },
    {
        "node": "6:00",
        "key": "团战中己方前排不在场",
        "title": "发育路六分钟：前排不在场就带线推塔",
        "expect": {"fact": "positioning_action", "values": ["push_wave"]},
        "subject": {"fact": "positioning_action",
                    "values": ["push_wave", "hold_position", "pull_back",
                               "rotate_support"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "团战里己方前排不在场就不进团，改成带线推塔，把节奏换成塔的经济。",
        "why": "「己方前排是否在场」数据里读不到。这一支与同节点的集火规则是两种情形"
               "（一个进团输出、一个不进团带线），用不同的动作事实分开判，不会互相指控。",
        "tags": ["发育路", "中期", "带线"],
    },
    {
        "node": "6:00",
        "key": "本局己方已阵亡 2 人次以上",
        "title": "发育路六分钟：己方崩了就缩塔刷野补经济",
        "expect": {"fact": "objective_action", "values": ["farm_jungle", "split_push"]},
        "subject": {"fact": "objective_action",
                    "values": ["farm_jungle", "split_push", "last_hit", "recall",
                               "invade_jungle"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "本局己方已经阵亡两人次以上就缩塔，减少参团，专心带线刷野补经济，"
                "等对方失误再动。",
        "why": "原文这一支的前提（谁减员、谁领先）在画面里读不出来，所以不拿它当纠正依据：只有当玩家**整段一次都没做过**这个时段攻略给的任何一件事时，这条才判偏离。"
               "这一支只判客观资源的处理（刷野 / 带线牵制），"
               "进不进团由同节点前两条判。",
        "tags": ["发育路", "中期", "止损"],
    },

    # ================================================== 10:00
    {
        "node": "10:00",
        "key": "龙坑有我方视野",
        "merge": ["龙坑无我方视野"],
        "title": "发育路十分钟：打龙前先占龙坑视野",
        "expect": {"fact": "objective_action",
                   "values": ["contest_dragon", "take_dragon"]},
        "subject": {"fact": "objective_action",
                    "values": ["contest_dragon", "take_dragon", "fight_baron",
                               "farm_jungle"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "龙坑有我方视野就集合打龙，打龙前先占龙坑视野、等前排探完再跟输出；"
                "没有视野就等视野，或抢时间差在对方视野盲区里打完。",
        "why": "「龙坑有没有我方视野」数据里读不到，两支的期望动作都是打龙，所以合成一条："
               "跟输出把龙拿下就算做到；等视野与抢时间差的差别写在提示里。",
        "tags": ["发育路", "中期", "打龙"],
    },
    {
        "node": "10:00",
        "key": "敌方刺客已在我方后排附近",
        "title": "发育路十分钟：刺客进后排就闪现拉开",
        "expect": {"fact": "positioning_action", "values": ["pull_back"]},
        "subject": {"fact": "positioning_action",
                    "values": ["pull_back", "hold_position", "cover_carry",
                               "rotate_support"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "敌方刺客已经摸到我方后排附近就闪现拉开距离，等辅助到位再输出，"
                "不要站在原地对拼。",
        "why": "「敌方刺客的位置」数据里读不到，这条判在发育路的站位操作上："
               "拉开距离就算做到；等辅助到位再输出这句写在提示里。",
        "tags": ["发育路", "中期", "保命"],
    },

    # ================================================== 12:00 之后
    {
        "node": "12:00",
        "key": "己方拿到风暴龙王",
        "title": "发育路后期：有风暴龙王就直推水晶",
        "expect": {"fact": "objective_action", "values": ["siege", "split_push"]},
        "subject": {"fact": "objective_action",
                    "values": ["siege", "split_push", "contest_dragon",
                               "take_dragon", "fight_baron"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "己方拿到风暴龙王就直接推进，一路带线拆高地塔直推水晶，终结比赛。",
        "why": "「谁拿到风暴龙王」数据里读不到，这条与同节点的守高地规则用不同的动作事实分开判"
               "（一个拆塔推进、一个卡位防守），所以不会同时指控玩家。",
        "tags": ["发育路", "后期", "推进"],
    },
    {
        "node": "12:00",
        "key": "对方拿到风暴龙王",
        "title": "发育路后期：对方有龙王就守高地",
        "expect": {"fact": "positioning_action", "values": ["hold_position"]},
        "subject": {"fact": "positioning_action",
                    "values": ["hold_position", "pull_back", "ward_vision",
                               "cover_carry"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "对方拿到风暴龙王就守高地塔，优先清掉龙兵线，拖时间等对方失误。",
        "why": "「对方有没有风暴龙王」数据里读不到，这条判在防守时的站位上："
               "卡住高地位置就算守住。",
        "tags": ["发育路", "后期", "防守"],
    },
    {
        "node": "12:00",
        "key": "龙未开",
        "title": "发育路后期：龙没开就占龙坑视野等刷新",
        "expect": {"fact": "positioning_action",
                   "values": ["ward_vision", "hold_position"]},
        "subject": {"fact": "positioning_action",
                    "values": ["ward_vision", "hold_position", "cover_carry",
                               "pull_back"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "龙还没开就在龙坑附近占视野等刷新，视野拿到后再推进。",
        "why": "「龙开没开」数据里读不到，这条判在龙坑附近的站位与视野上：占住坑位、"
               "把视野做好就算做到；取值与同节点的守高地规则有交集，所以不会互相指控。",
        "tags": ["发育路", "后期", "龙坑视野"],
    },
]

#: 明确不入库的分支及理由。
SKIPPED = [
    {
        "node": "2:00",
        "key": "被抓",
        "reason": "这一行只是嵌套分支的父节点，原文只给了「被抓」这个前提、没有动作"
                  "（动作都写在两个子分支上），系统按叶子分支判定，所以父节点本身不入库。",
    },
]
