"""游走分路的映射表。

时间轴：0:00 开局 → 2:00 → 4:00 → 6:00 → 10:00 → 12:00。

游走位（辅助）干的活就两类：把视野做出来、把团战的先后手抢下来。
攻略里大量前提是「对方控制交没交」「C 位有没有被切入」「龙坑有没有视野」，
这些在数据里都读不到，所以能按「可接受动作集合」合并的分支一律合并，
剩下的用可观察的标量事实（我方大招是否就绪）把互斥的两支隔开。
"""

from __future__ import annotations

LANE = "roam"

RULES = [
    # ================================================== 0:00 开局
    {
        "node": "0:00",
        "key": "己方中单需要抢线",
        "merge": ["中单能自己清线"],
        "title": "游走开局：跟中单抢线或跟打野做视野",
        "expect": {"fact": "positioning_action",
                   "values": ["push_wave", "ward_vision"]},
        "subject": {"fact": "positioning_action",
                    "values": ["push_wave", "pull_back", "ward_vision",
                               "rotate_support", "hold_position"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "四级前与中单一起抢线，抢线前先确认河道草丛没有敌方单位，"
                "抢完线再和打野一起抓人、控龙；中单能自己清线就跟着打野进野区做视野，"
                "或去对方野区入口探敌方打野路线。",
        "why": "「中单要不要一起抢线」「中单能不能自己清线」对局数据里读不到，"
               "但两支可接受的动作有交集：都是前期跟着人走并把视野做出来，"
               "所以合成一条，期望取并集。",
        "tags": ["游走", "前期", "视野"],
    },
    {
        "node": "0:00",
        "key": "射手被压到塔下",
        "title": "游走开局：射手被压塔下就回发育路",
        "expect": {"fact": "positioning_action",
                   "values": ["ward_vision", "cover_carry"]},
        "subject": {"fact": "positioning_action",
                    "values": ["ward_vision", "cover_carry", "pull_back",
                               "hold_position"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "射手被压到塔下时回发育路做视野，让射手能安心补刀；"
                "已经站到射手身边就守在射手位上保护他。",
        "why": "「射手被压到塔下」在数据里读不到，这条按「有没有回发育路、"
               "有没有护住射手」判：回发育路做视野对应做视野，"
               "守在射手位上对应保护射手位。",
        "tags": ["游走", "前期", "保护射手"],
    },
    {
        "node": "0:00",
        "key": "敌方进入我方野区",
        "title": "游走开局：敌方进野区就做视野逼退",
        "expect": {"fact": "positioning_action",
                   "values": ["ward_vision", "hold_position"]},
        "subject": {"fact": "positioning_action",
                    "values": ["ward_vision", "hold_position", "pull_back",
                               "rotate_support"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "敌方进入我方野区时立刻过去做视野逼退，必要时交控制，"
                "别让对方带着 BUFF 走；也可以在野区入口卡位守死对方。",
        "why": "「必要时交控制」在动作词表里没有对应取值（只有开团先手），"
               "所以这条只判有没有过去做视野或卡位逼退。"
               "同一节点还有「用信号报出敌方英雄位置与状态」那一支，"
               "那是沟通动作，标注页没有可勾的动作，判不了，理由见未入库清单。",
        "tags": ["游走", "前期", "逼退"],
    },

    # ================================================== 2:00
    {
        "node": "2:00",
        "key": "龙坑有我方视野",
        "merge": ["龙坑无我方视野", "己方打野在打资源且敌方来抢"],
        "title": "游走两分钟：龙坑占位探草，资源旁埋伏",
        "expect": {"fact": "positioning_action",
                   "values": ["hold_position", "ward_vision"]},
        "subject": {"fact": "positioning_action",
                    "values": ["hold_position", "ward_vision", "pull_back",
                               "rotate_support", "push_wave"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "龙坑有我方视野就占龙坑后方草丛；没有视野就先探龙坑或绕后，"
                "拿到视野再动；己方打野在打资源而对方来抢，就在资源旁草丛埋伏，"
                "露视野让对方进草被控住再反打。",
        "why": "「龙坑有没有我方视野」「对方来没来抢资源」都是攻略内部状态，"
               "数据里读不到；三支要求的动作同属站位一类（占位 + 拿视野），"
               "所以合成一条，期望取并集。",
        "tags": ["游走", "前期", "龙坑"],
    },
    {
        "node": "2:00",
        "key": "对方已在龙坑内打暴君",
        "title": "游走两分钟：对方打暴君就集合龙坑",
        "expect": {"fact": "objective_action",
                   "values": ["contest_dragon"]},
        "subject": {"fact": "objective_action",
                    "values": ["contest_dragon", "take_dragon", "fight_baron",
                               "split_push", "siege", "farm_jungle"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "对方已经在龙坑内打暴君时立刻集合到龙坑形成夹击，"
                "别让对方把龙安稳打完。",
        "why": "「对方已在龙坑内打暴君」在数据里读不到，这条按"
               "「该来抢龙的时候有没有来」判：集合龙坑对应开小龙团。"
               "期望只留开团先手，是因为原文这一支的动作就是集合起来夹击，"
               "不包括拿完龙再去拆塔。",
        "tags": ["游走", "前期", "抢龙"],
    },

    # ================================================== 4:00
    {
        "node": "4:00",
        "key": "对方开团与控制技能在本场团战尚未出现",
        "merge": ["己方 C 位被对方切入"],
        "title": "游走四分钟：不接团，做视野或回防C位",
        "expect": {"fact": "positioning_action",
                   "values": ["ward_vision", "cover_carry"]},
        "subject": {"fact": "positioning_action",
                    "values": ["ward_vision", "cover_carry", "pull_back",
                               "hold_position", "rotate_support"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "对方这场还没打出开团和控制时不要先开团，改为做视野，等对方先交控制；"
                "己方 C 位被对方切入时强制回防，放弃开团，先把 C 位救回来。",
        "why": "这两支的触发前提（对方控制交没交、C 位有没有被切入）数据里都读不到，"
               "但共同点是这一回合不接团，动作上都是做视野或者回护 C 位，"
               "所以合成一条，期望取并集。同节点的「开团接控」那一支用的是"
               "团战处理这个动作事实，不会和这条互相指控。",
        "tags": ["游走", "前期", "不开团"],
    },
    {
        "node": "4:00",
        "key": "对方开团与控制技能本场已交 + 己方控制链就绪 + 己方前排已站位",
        # 「己方控制链就绪」→ **大招就绪**（游走位的大招就是控制链的起点）。
        #
        # 这是原文给的、画面里**读得出**的前提：不依赖任何数值。
        # 与同节点「对方还没开团就不开团」那条用同一个事实的两边互斥开。
        "extra": [{"fact": "has_ult_ready", "op": "==", "value": True}],
        "title": "游走四分钟：对方控制已交就开团接控",
        "expect": {"fact": "teamfight_action",
                   "values": ["engage"]},
        "subject": {"fact": "teamfight_action",
                    "values": ["engage", "focus_target", "protect_carry",
                               "split_fight", "retreat", "kite"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "对方这场已经把开团和控制交出来、我方控制链就绪、前排已站位时，"
                "主动开团接控，打出人数优势。",
        "why": "三个前提（对方控制已交、我方控制链就绪、前排已站位）数据里都读不到，"
               "所以这条只判「该开团的时候有没有开团」：这一回合玩家在打团就算适用。"
               "适用时机圈的是全部团战处理动作，比期望宽。",
        "tags": ["游走", "前期", "开团"],
    },

    # ================================================== 6:00
    {
        "node": "6:00",
        "key": "己方控制链全部就绪",
        "extra": [{"fact": "has_ult_ready", "op": "==", "value": True}],
        "title": "游走六分钟：控制链齐就主动开团",
        "expect": {"fact": "teamfight_action",
                   "values": ["engage"]},
        "subject": {"fact": "teamfight_action",
                    "values": ["engage", "focus_target", "protect_carry",
                               "split_fight", "retreat", "kite"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "我方控制链全部就绪时主动开团，抢先手优势，"
                "开团后跟上集火，不要站着看。",
        "why": "「控制链全部就绪」用「我方核心大招是否就绪」近似 —— "
               "游走位的大招通常就是控制链的起点。这条与下一条（控制链缺一环）"
               "用同一个事实的两边互斥开，玩家不会同时被两条指控。",
        "tags": ["游走", "中期", "开团"],
    },
    {
        "node": "6:00",
        "key": "己方控制链有一环未就绪",
        "extra": [{"fact": "has_ult_ready", "op": "==", "value": False}],
        "title": "游走六分钟：控制链缺一环就不先手",
        "expect": {"fact": "teamfight_action",
                   "values": ["kite", "retreat"]},
        "subject": {"fact": "teamfight_action",
                    "values": ["engage", "focus_target", "protect_carry",
                               "split_fight", "retreat", "kite"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "控制链有一环没就绪时不要先手：跟在我方前排后面推进，"
                "用拉扯消耗等对方先交控制；站不住就撤退脱离，别把人头送出去。",
        "why": "「控制链有一环未就绪」用「我方核心大招是否就绪」近似，与上一条互斥。"
               "「跟己方前排推进，等对方交控制」在词表里最接近的是拉扯消耗与撤退脱离"
               "（不接团、不先手），推进本身没有可勾的取值，写进提示。",
        "tags": ["游走", "中期", "不先手"],
    },
    {
        "node": "6:00",
        "key": "破盾型辅助的盾已被打碎",
        "title": "游走六分钟：盾碎了就用大招撞阵",
        "expect": {"fact": "teamfight_action",
                   "values": ["engage", "retreat"]},
        "subject": {"fact": "teamfight_action",
                    "values": ["engage", "focus_target", "protect_carry",
                               "split_fight", "retreat", "kite"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "盾被打碎以后不要再举盾，改用大招撞击敌方阵型，"
                "或者退回后排身后，别在无盾状态被集火。",
        "why": "「举盾」在动作词表里没有对应取值，所以这条只判盾碎之后的两个替代动作："
               "开团先手与撤退脱离。「破盾型辅助」是英雄类型，数据里只有我方英雄能近似，"
               "这条按「盾碎了之后怎么处理」判。",
        "tags": ["游走", "中期", "破盾"],
    },
    {
        "node": "6:00",
        "key": "中期需要布控",
        "merge": ["己方射手为抗压与进攻兼具型"],
        "title": "游走六分钟：按龙坑、河道、野区入口、边路草丛的顺序布控；"
                 "射手抗压且能进攻时可跟中路支援",
        "expect": {"fact": "positioning_action",
                   "values": ["ward_vision", "rotate_support"]},
        "subject": {"fact": "positioning_action",
                    "values": ["ward_vision", "rotate_support", "hold_position",
                               "pull_back", "cover_carry", "push_wave"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "中期按龙坑、河道、敌方打野必经的野区入口、边路草丛的顺序布控，"
                "视野优先覆盖下一条资源的争夺点；己方射手是抗压与进攻兼具型时，"
                "可以选择跟中路一起支援打开局势。",
        "why": "「中期需要布控」是攻略内部状态，「己方射手是抗压与进攻兼具型」"
               "要看对面英雄，两件事数据里都读不到；两支要求的动作同属站位一类"
               "（做视野 / 转线支援），所以合成一条，期望取并集。"
               "原文说这一支「可跟可不跟」，所以转线支援只是可接受动作之一。",
        "tags": ["游走", "中期", "布控"],
    },

    # ================================================== 10:00
    {
        "node": "10:00",
        "key": "我方先到龙坑",
        "merge": ["敌方先到龙坑"],
        "title": "游走十分钟：先到开龙，被占就绕后推塔",
        "expect": {"fact": "objective_action",
                   "values": ["contest_dragon", "take_dragon",
                              "split_push", "siege"]},
        "subject": {"fact": "objective_action",
                    "values": ["contest_dragon", "take_dragon", "fight_baron",
                               "split_push", "siege", "farm_jungle",
                               "invade_jungle"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "我方先到龙坑就占位并开龙；敌方先到就绕后，或者转去推塔，"
                "用塔换资源，不硬接这波龙。",
        "why": "「谁先到龙坑」数据里读不到，两支的期望动作同属「打资源或换资源」，"
               "所以合成一条，期望取并集。绕后本身没有对应的动作取值，"
               "和推塔一起写进提示。",
        "tags": ["游走", "中期", "主宰"],
    },
    {
        "node": "10:00",
        "key": "己方后排被对方切入",
        "title": "游走十分钟：后排被切入先救回不强换",
        "expect": {"fact": "teamfight_action",
                   "values": ["protect_carry", "retreat"]},
        "subject": {"fact": "teamfight_action",
                    "values": ["engage", "focus_target", "protect_carry",
                               "split_fight", "retreat", "kite"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "己方后排被对方切入时优先救回，放弃开团，不跟对方强换人头，"
                "保住己方输出。",
        "why": "「己方后排被对方切入」在数据里读不到，这条按"
               "「有没有回护 C 位、有没有为了保人而脱离」判："
               "保护 C 位对应保护输出，放弃强换对应撤退脱离。",
        "tags": ["游走", "中期", "救回"],
    },

    # ================================================== 12:00 之后
    {
        "node": "12:00",
        "key": "己方开团条件齐",
        "extra": [{"fact": "has_ult_ready", "op": "==", "value": True}],
        "title": "游走后期：条件齐就主动开团终结比赛",
        "expect": {"fact": "teamfight_action",
                   "values": ["engage"]},
        "subject": {"fact": "teamfight_action",
                    "values": ["engage", "focus_target", "protect_carry",
                               "split_fight", "retreat", "kite"]},
        "severity": "high",
        "scope": "at_least_once",
        "hint": "后期开团条件齐就主动开团，直接终结比赛。",
        "why": "「开团条件齐」用「我方核心大招是否就绪」近似，"
               "与下一条（开团条件不齐）用同一个事实的两边互斥开，"
               "玩家不会同时被两条指控。",
        "tags": ["游走", "后期", "开团"],
    },
    {
        "node": "12:00",
        "key": "己方开团条件不齐",
        "extra": [{"fact": "has_ult_ready", "op": "==", "value": False}],
        "title": "游走后期：开团条件不齐就跟在我方前排后面推进",
        "expect": {"fact": "teamfight_action",
                   "values": ["kite", "retreat"]},
        "subject": {"fact": "teamfight_action",
                    "values": ["engage", "focus_target", "protect_carry",
                               "split_fight", "retreat", "kite"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "后期开团条件不齐就跟在我方前排后面推进，用拉扯消耗等条件齐了再开团，"
                "稳住别乱冲。",
        "why": "「开团条件不齐」用「我方核心大招是否就绪」近似，与上一条互斥；"
               "「跟己方前排推进」在词表里最接近的是拉扯消耗与撤退脱离，"
               "推进本身没有可勾的取值，写进提示。",
        "tags": ["游走", "后期", "稳扎稳打"],
    },
    {
        "node": "12:00",
        "key": "己方射手为进攻型且对线能抗压",
        "title": "游走后期：跟进攻型射手别让他被抓崩",
        "expect": {"fact": "positioning_action",
                   "values": ["cover_carry"]},
        "subject": {"fact": "positioning_action",
                    "values": ["cover_carry", "ward_vision", "hold_position",
                               "pull_back", "rotate_support"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "己方射手是进攻型且对线能抗压时跟住射手，保他不被抓崩一路。",
        "why": "「射手是不是进攻型、对线抗不抗压」要看对面英雄，数据里读不到，"
               "这条按「有没有跟住并护住射手位」判。",
        "tags": ["游走", "后期", "跟射手"],
    },
    {
        "node": "12:00",
        "key": "对方已在龙坑开风暴龙王",
        "title": "游走后期：对方开风暴龙王就转拆塔",
        "expect": {"fact": "objective_action",
                   "values": ["siege"]},
        "subject": {"fact": "objective_action",
                    "values": ["siege", "split_push", "contest_dragon",
                               "take_dragon", "fight_baron", "farm_jungle"]},
        "severity": "medium",
        "scope": "at_least_once",
        "hint": "对方已经在龙坑开风暴龙王时别去接，转去拆高地外塔，用塔换时间。",
        "why": "「对方是否已经开风暴龙王」在数据里读不到，这条按"
               "「后期该守还是该拆」判：转去拆高地外塔对应拆高地塔。",
        "tags": ["游走", "后期", "拆塔"],
    },
]

#: 明确不入库的分支及理由。
SKIPPED = [
    {
        "node": "0:00",
        "key": "全时段",
        "reason": "这一支的动作是「用信号报出敌方英雄位置与状态，信号优先于打字」，"
                  "属于沟通而不是玩家在地图上的操作；标注页只让勾选地图上的动作，"
                  "没有「发信号」这个动作事实，系统判不了。"
                  "报点这件事写进了同节点其它规则的 why 作为补充说明。",
    },
]
