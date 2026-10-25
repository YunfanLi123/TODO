"""中路分路的映射表。

时间轴：0:00 开局 → 2:00 首条暴君 → 4:00 线权取舍 → 6:00 中野联动 →
10:00 主宰团 → 12:00 之后风暴龙王。
"""

from __future__ import annotations

LANE = "mid"

RULES = [
    # ================================================== 0:00 开局
    {
        "node": "0:00",
        "key": "1 级能先清完第一波线",
        "merge": [
            "清线慢半拍",
            "被对方二级抓中 / 血量健康",
        ],
        "title": "中路开局：推线到对方塔下，占住河道视野",
        "expect": {"fact": "positioning_action",
                   "values": ["push_wave", "hold_position", "ward_vision", "pull_back"]},
        "subject": {"fact": "positioning_action",
                    "values": ["push_wave", "hold_position", "ward_vision", "pull_back",
                               "rotate_support", "cover_carry"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "1 级先普攻后技能抢线，在兵线进对方塔前清完，再移动到能观察河道草丛的位置，"
                "不要退回塔下；清兵只用小额基础技能，控制与爆发留着。清线慢半拍就放弃这波线权，"
                "直接去河道占位。被对方二级抓中但血量健康，就交位移往己方塔下退，保住经验和技能。",
        "why": "三支的前提（1 级能否清完线、清线快慢、是否被二级抓中）系统都读不到，"
               "可接受的动作合并成一条：推线到对方塔下 / 去河道占位 / 被抓后退回塔下，"
               "都算按攻略处理了第一波线。血量与线权这类细节判不了的部分写在提示里。",
        "tags": ["中路", "前期", "抢线"],
    },
    {
        "node": "0:00",
        "key": "被对方二级抓中 / 血量低于一半",
        "title": "中路开局：被抓残血就直接回城保住发育",
        "expect": {"fact": "objective_action", "values": ["recall"]},
        "subject": {"fact": "objective_action",
                    "values": ["recall", "farm_jungle", "clear_vision_sprite", "last_hit",
                               "take_river_spirit", "defend_inner"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "被对方二级抓中、血量掉到一半以下就直接回城，掉一层经济换后续发育，"
                "别留在原地等第二次抓。",
        "why": "原文这一支的前提（谁减员、谁领先）在画面里读不出来，所以不拿它当纠正依据：只有当玩家**整段一次都没做过**这个时段攻略给的任何一件事时，这条才判偏离。"
               "这一支说的是「被抓残血就回城」，而「被抓」「残血」都读不到 —— "
               "所以只有当玩家整段一次都没做过这个时段攻略给的任何一件事时，"
               "才说他该回城。",
        "tags": ["中路", "前期", "被抓"],
    },

    # ================================================== 2:00 首条暴君
    {
        "node": "2:00",
        "key": "河道路口有视野",
        "merge": ["河道路口无视野"],
        "title": "中路两分钟：河道有视野跟打野打龙，没视野先补视野",
        "expect": {"fact": "objective_action",
                   "values": ["contest_dragon", "take_dragon", "clear_vision_sprite"]},
        "subject": {"fact": "objective_action",
                    "values": ["contest_dragon", "take_dragon", "clear_vision_sprite",
                               "farm_jungle", "invade_jungle", "take_river_spirit",
                               "last_hit"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "河道路口有视野就与打野集合打首条暴君，首条暴君固定在发育路与对抗路之间的河道右侧；"
                "河道路口没视野就不进河道，先清线并把河道路口视野补上，下一次暴君再集合。",
        "why": "河道有没有视野系统读不到，两支的可接受动作并成一条：打龙或补河道视野都算"
               "按攻略处理了这条龙。没有视野硬进河道这一层判不了，写在提示里。",
        "tags": ["中路", "前期", "暴君"],
    },
    {
        "node": "2:00",
        "key": "对方已在龙坑内打暴君",
        "title": "中路两分钟：对方已在打龙就转压边路塔",
        "expect": {"fact": "positioning_action", "values": ["push_wave"]},
        "subject": {"fact": "positioning_action",
                    "values": ["push_wave", "rotate_support", "pull_back",
                               "hold_position", "ward_vision"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "对方已经在龙坑里打暴君，就放弃这条龙，转去压对方边路外塔，用塔换龙。",
        "why": "「对方已在龙坑内打暴君」这个前提系统读不到（龙坑视野没有数据），"
               "这条只判有没有用塔换龙；什么时候转、转哪条边路判不了。",
        "tags": ["中路", "前期", "换龙"],
    },

    # ================================================== 4:00 线权取舍
    {
        "node": "4:00",
        "key": "已抢到线，且清完后无新兵线刷新",
        "merge": ["已抢到线，但清完后兵线又刷新"],
        "extra": [{"fact": "matchup", "op": "!=", "value": "disadvantage"}],
        "title": "中路四分钟：线权在手就转线支援或继续清线",
        "expect": {"fact": "positioning_action", "values": ["rotate_support", "push_wave"]},
        "subject": {"fact": "positioning_action",
                    "values": ["rotate_support", "push_wave", "pull_back", "hold_position",
                               "ward_vision", "cover_carry"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "清完线没有新兵线刷新，就立刻去支援，支援方向优先劣势路止损、其次优势路扩优；"
                "新兵线又刷新了就留在线上清掉，别因为支援把线权空出来。",
        "why": "两支的差别是「清完后有没有新兵线刷新」，系统读不到，可接受动作并成一条："
               "转线支援或继续推线都算。「已抢到线」用对位优劣近似（不是劣势对位才判这条），"
               "与「没抢到线」那条互斥开，不会同时指控两种打法。",
        "tags": ["中路", "前期", "线权"],
    },
    {
        "node": "4:00",
        "key": "没抢到线",
        "extra": [{"fact": "matchup", "op": "==", "value": "disadvantage"}],
        "title": "中路四分钟：没线权就跟住对方中路等救场",
        "expect": {"fact": "positioning_action", "values": ["hold_position"]},
        "subject": {"fact": "positioning_action",
                    "values": ["hold_position", "pull_back", "push_wave", "ward_vision",
                               "rotate_support", "cover_carry"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "没抢到线就清完线跟在对方中路后面移动，不进无视野河道，"
                "等对方打过来时能第一时间到场救场。",
        "why": "「没抢到线」用对位优劣近似（劣势对位才判这条），与「已抢到线」那条互斥开，"
               "避免同一格里两条规则同时指控。",
        "tags": ["中路", "前期", "保命"],
    },
    {
        "node": "4:00",
        "key": "敌方打野最后出现位置在另一半地图",
        "merge": [
            "敌方打野最后出现位置在我方半区",
            "敌方打野位置未知",
        ],
        "title": "中路四分钟：按敌方打野位置决定压塔还是做视野",
        "expect": {"fact": "positioning_action",
                   "values": ["push_wave", "ward_vision", "hold_position"]},
        "subject": {"fact": "positioning_action",
                    "values": ["push_wave", "ward_vision", "hold_position", "pull_back",
                               "rotate_support", "cover_carry"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "敌方打野最后出现位置在另一半地图，就主动压对方中一塔，压缩他的转线路径；"
                "敌方打野在我方半区，就清线并把河道路口视野做出来，不压塔；"
                "敌方打野位置未知，就清完线回塔下或转去己方野区做视野，先保住自己。",
        "why": "敌方打野的位置系统读不到，三支的可接受动作并成一条：压塔 / 做视野 / 回塔下"
               "都算按攻略应对了这条信息。敌方打野在哪一侧决定做哪件事，判不了的部分写在提示里。",
        "tags": ["中路", "前期", "视野"],
    },

    # ================================================== 6:00 中野联动
    {
        "node": "6:00",
        "key": "己方兵线已推进至对方塔下",
        "merge": ["己方兵线还在自家塔前"],
        "title": "中路六分钟：跟打野游走，或塔下清线做视野反蹲",
        "expect": {"fact": "positioning_action",
                   "values": ["rotate_support", "ward_vision", "hold_position"]},
        "subject": {"fact": "positioning_action",
                    "values": ["rotate_support", "ward_vision", "hold_position",
                               "push_wave", "pull_back", "cover_carry"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "己方兵线已经推进到对方塔下，就清完线跟打野游走，形成中野联动；"
                "兵线还在自家塔前，就在塔下清线、去己方野区做视野，帮打野反蹲，防止被对方抓单。",
        "why": "兵线在对方塔下还是自家塔前系统读不到，两支的可接受动作并成一条："
               "转线跟打野或做视野反蹲都算。",
        "tags": ["中路", "中期", "联动"],
    },
    {
        "node": "6:00",
        "key": "团战中对方先打出控制并命中己方前排",
        "title": "中路六分钟：对方先交控制就退到前排身后",
        "expect": {"fact": "teamfight_action", "values": ["retreat"]},
        "subject": {"fact": "teamfight_action",
                    "values": ["engage", "focus_target", "protect_carry", "retreat",
                               "kite", "split_fight"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "团战里对方先打出控制并命中己方前排，就后撤到己方前排身后，"
                "等对方控制交完再进。",
        "why": "「对方先交控制」系统读不到，这条判在参与团战的回合上：这时该做的是不接团。",
        "tags": ["中路", "中期", "团战"],
    },

    # ================================================== 10:00 主宰团
    {
        "node": "10:00",
        "key": "我方先到龙坑",
        "merge": ["敌方先到龙坑"],
        "title": "中路十分钟：先到开龙，被占就转压边路塔",
        "expect": {"fact": "objective_action",
                   "values": ["contest_dragon", "take_dragon", "split_push", "siege"]},
        "subject": {"fact": "objective_action",
                    "values": ["contest_dragon", "take_dragon", "fight_baron", "split_push",
                               "siege", "farm_jungle", "invade_jungle"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "我方先到龙坑就开龙，拿到暗影暴君与暗影主宰增益，主宰先锋替换兵线；"
                "敌方先到就放弃这条龙，转去压边路塔，用塔换龙。",
        "why": "「谁先到龙坑」系统读不到，两支的可接受动作并成一条：打龙或转线拆塔"
               "都算按攻略处理了这条龙。",
        "tags": ["中路", "中期", "主宰"],
    },
    {
        "node": "10:00",
        "key": "团战即将开始",
        "title": "中路十分钟：团战前站远，保持输出环境",
        "expect": {"fact": "positioning_action", "values": ["pull_back", "hold_position"]},
        "subject": {"fact": "positioning_action",
                    "values": ["pull_back", "hold_position", "push_wave", "rotate_support",
                               "ward_vision", "cover_carry"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "团战即将开始时，站到与敌方突进距离大于「一个突进位移加一个闪现」的位置，"
                "并在队伍侧后方，让对方刺客多交一个位移才够得着你，输出环境才留得住。",
        "why": "具体站位距离系统读不到，这条只判有没有拉开距离、保留输出环境（回撤或卡位）；"
               "距离差多少写在提示里。",
        "tags": ["中路", "中期", "站位"],
    },
    {
        "node": "10:00",
        "key": "对方控制与爆发技能已交 + 己方全员就位",
        # 原文的动作是「**开大**分割战场」—— 己方大招就绪是它的前提。
        # 大招亮不亮在画面里**读得出**，不依赖任何数值。
        "extra": [{"fact": "has_ult_ready", "op": "==", "value": True}],
        "title": "中路十分钟：对方控制交了就开大分割战场",
        "expect": {"fact": "teamfight_action", "values": ["engage", "focus_target"]},
        "subject": {"fact": "teamfight_action",
                    "values": ["engage", "focus_target", "protect_carry", "retreat",
                               "kite", "split_fight"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "对方控制与爆发技能已经交掉、己方全员就位时，就开大分割战场，"
                "优先框住敌方主输出点，打成以多打少。",
        "why": "「对方控制交没交」「己方是否全员就位」系统都读不到，"
               "这条判在团战回合上：该抢的就是这一波开团。",
        "tags": ["中路", "中期", "开团"],
    },
    {
        "node": "10:00",
        "key": "团战打输，己方有人未复活",
        "title": "中路十分钟：团战打输就后撤清线止损",
        "expect": {"fact": "positioning_action", "values": ["pull_back", "push_wave"]},
        "subject": {"fact": "positioning_action",
                    "values": ["pull_back", "push_wave", "hold_position", "rotate_support",
                               "ward_vision", "cover_carry"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "团战打输、己方还有人没复活时，就后撤清线，不补位置，止损并保留技能。",
        "why": "原文这一支的前提（谁减员、谁领先）在画面里读不出来，所以不拿它当纠正依据：只有当玩家**整段一次都没做过**这个时段攻略给的任何一件事时，这条才判偏离。"
               "「团战打输、己方有人未复活」这个前提读不到，"
               "所以拿它纠正人就是冤枉。",
        "tags": ["中路", "中期", "止损"],
    },

    # ================================================== 12:00 之后 风暴龙王
    {
        "node": "12:00",
        "key": "对方已在龙坑开风暴龙王",
        "title": "中路后期：对方开龙王就守高地清龙兵线",
        "expect": {"fact": "positioning_action", "values": ["hold_position", "push_wave"]},
        "subject": {"fact": "positioning_action",
                    "values": ["hold_position", "push_wave", "pull_back", "rotate_support",
                               "ward_vision", "cover_carry"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "对方已经在龙坑开风暴龙王，就守高地塔，优先清掉龙兵线，拖到对方失误。",
        "why": "原文这一支的前提（谁减员、谁领先）在画面里读不出来，所以不拿它当纠正依据：只有当玩家**整段一次都没做过**这个时段攻略给的任何一件事时，这条才判偏离。"
               "「拿到龙王就推进」是同一时段的另一种打法，玩家做了任一件都不说他错 —— "
               "守与推不会同时指控。",
        "tags": ["中路", "后期", "防守"],
    },
    {
        "node": "12:00",
        "key": "己方拿到风暴龙王",
        "title": "中路后期：拿到龙王就直接推水晶",
        "expect": {"fact": "objective_action", "values": ["split_push", "siege"]},
        "subject": {"fact": "objective_action",
                    "values": ["split_push", "siege", "contest_dragon", "take_dragon",
                               "fight_baron", "farm_jungle"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "己方拿到风暴龙王，就一路推进直推水晶，终结比赛。",
        "why": "原文这一支的前提（谁减员、谁领先）在画面里读不出来，所以不拿它当纠正依据：只有当玩家**整段一次都没做过**这个时段攻略给的任何一件事时，这条才判偏离。"
               "「对方开龙王就守高地」是同一时段的另一种打法，玩家做了任一件都不说他错。",
        "tags": ["中路", "后期", "推进"],
    },
    {
        "node": "12:00",
        "key": "龙未开",
        "title": "中路后期：龙没开就占龙坑视野等刷新",
        "expect": {"fact": "positioning_action", "values": ["ward_vision", "hold_position"]},
        "subject": {"fact": "positioning_action",
                    "values": ["ward_vision", "hold_position", "push_wave", "pull_back",
                               "rotate_support", "cover_carry"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "龙还没开就占住龙坑视野等刷新，拿到后再推进。",
        "why": "「龙未开」是局势状态，系统读不到（己方有没有龙 buff 没有数据），"
               "这条只判有没有去占龙坑视野、等刷新。",
        "tags": ["中路", "后期", "龙坑"],
    },
    {
        "node": "12:00",
        "key": "己方需要完整人数",
        "title": "中路后期：需要人数就集合，不单独进带",
        "expect": {"fact": "positioning_action", "values": ["hold_position", "rotate_support"]},
        "subject": {"fact": "positioning_action",
                    "values": ["hold_position", "rotate_support", "push_wave", "pull_back",
                               "ward_vision", "cover_carry"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "己方需要完整人数时就集合起来，不单独进带，避免被单抓。",
        "why": "「己方需要完整人数」系统读不到，这条只判有没有抱团集合；"
               "够不够人、缺的是谁，判不了的部分写在提示里。",
        "tags": ["中路", "后期", "抱团"],
    },
]

#: 明确不入库的分支及理由。
SKIPPED = [
    {
        "node": "0:00",
        "key": "被对方二级抓中",
        "reason": "这一支只是前提分组（父级），本身没有动作也没有后果，"
                  "具体动作在「血量健康」「血量低于一半」两个子分支里，已分别归宿。",
    },
]
