"""英雄级攻略的动作文本 -> 系统已注册的动作类别。

为什么要有这张表
----------------
英雄级攻略里写的是**带技能名的具体操作**：

    「三技能附魔突进抢龙，落地挥斩后接一技能」
    「一技能一判定推开兵线，二判定压上去清线，不越过小兵身位」
    「转为后置保护，站在己方 C 位与敌方突进之间」

而规则库里能判定的只有 :data:`app.adapters.honor_of_kings.MOBA_ACTION_VALUES`
里那 26 个**动作类别**。把前者映射到后者，就是这张表干的活。

映射为什么不能靠关键词硬匹配
--------------------------
攻略里同一个词在不同句子里角色不同：

    「带主宰先锋拆边路二塔」   —— 主宰是修饰语，动词是「拆塔」
    「控暗影主宰，惩击留到最后一刻」 —— 动词是「控主宰」

所以这张表做三件事：

1. **先切小句**：以「，；、」切分，一小句一个动作意图；
2. **否定不算动作**：「不进龙坑」里的「龙坑」是被否定的动作，
   不该进期望值 —— 期望值只放玩家**该做**的事；
3. **一小句取一个动作族，取同族里优先级最高的那个词**：
   动词优先于名词，修饰语（主宰/暴君）不单独成动作。

匹配不上的小句不是错误，是**判不了**：攻略写的是沟通、装备、
惩击时机这类系统记不下来的事，这类分支由生成器标成
「不可判定，不入库」并写明理由 —— 宁可少，也不能编。

与分路层的口径一致
------------------
分类用的词表就是分路层那套（``HOK-MODE-*`` 那 85 条用的同一份
``MOBA_ACTION_VALUES``），所以同一件事在两层里的说法一致：
攻略说「控暗影暴君」，分路层那条规则也写「开小龙团 / 拿下小龙」。
两层说法不一致的话，玩家会看到互相打架的结论。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# 动作类别
# --------------------------------------------------------------------------

#: 适用时机（subject）的取值集合，按动作族划分。
#:
#: subject 圈的是「这条规则在玩家**做了哪一类操作**时才适用」，
#: 必须比 expect 宽 —— 宽到和 expect 一样就等于没判。
#: 分路层那 85 条用的也是这套分组，这里保持一致。
SUBJ_DRAGON = ["contest_dragon", "take_dragon", "fight_baron"]
SUBJ_JUNGLE = ["farm_jungle", "invade_jungle", "clear_vision_sprite"]
SUBJ_RIVER = ["take_river_spirit", "farm_jungle", "invade_jungle",
              "contest_dragon", "take_dragon"]
SUBJ_LAST_HIT = ["last_hit", "farm_jungle", "recall", "take_river_spirit",
                 "contest_dragon", "take_dragon", "split_push"]
SUBJ_PUSH = ["split_push", "siege", "farm_jungle", "invade_jungle",
             "contest_dragon", "take_dragon", "fight_baron", "recall", "last_hit"]
SUBJ_DEFEND = ["defend_inner", "recall", "last_hit", "farm_jungle",
               "take_river_spirit", "split_push"]
SUBJ_RECALL = ["recall", "farm_jungle", "clear_vision_sprite", "last_hit",
               "take_river_spirit", "defend_inner"]
SUBJ_VISION = ["clear_vision_sprite", "split_push", "siege", "recall",
               "last_hit", "farm_jungle", "take_river_spirit"]
SUBJ_POSITION = ["push_wave", "pull_back", "ward_vision", "hold_position",
                 "rotate_support", "cover_carry"]
SUBJ_TEAMFIGHT = ["engage", "focus_target", "protect_carry", "retreat",
                  "kite", "split_fight"]


@dataclass(frozen=True)
class Category:
    """一个动作类别：系统认得的一个取值 + 它的适用时机 + 严重度。"""

    name: str
    fact: str
    value: str
    severity: str = "medium"
    subject: Tuple[str, ...] = ()
    label: str = ""


CATEGORIES: Dict[str, Category] = {
    # -- 目标处理（objective_action） ------------------------------------
    "take_river_spirit": Category(
        "take_river_spirit", "objective_action", "take_river_spirit",
        "medium", tuple(SUBJ_RIVER), "控河道之灵"),
    "contest_dragon": Category(
        "contest_dragon", "objective_action", "contest_dragon",
        "high", tuple(SUBJ_DRAGON), "开小龙团"),
    "take_dragon": Category(
        "take_dragon", "objective_action", "take_dragon",
        "high", tuple(SUBJ_DRAGON), "拿下小龙"),
    "fight_baron": Category(
        "fight_baron", "objective_action", "fight_baron",
        "high", tuple(SUBJ_DRAGON), "开大龙团"),
    "farm_jungle": Category(
        "farm_jungle", "objective_action", "farm_jungle",
        "medium", tuple(SUBJ_JUNGLE), "刷野补经济"),
    "invade_jungle": Category(
        "invade_jungle", "objective_action", "invade_jungle",
        "medium", tuple(SUBJ_JUNGLE), "反野入侵"),
    "clear_vision_sprite": Category(
        "clear_vision_sprite", "objective_action", "clear_vision_sprite",
        "low", tuple(SUBJ_VISION), "清视野精灵"),
    "last_hit": Category(
        "last_hit", "objective_action", "last_hit",
        "medium", tuple(SUBJ_LAST_HIT), "补刀"),
    "split_push": Category(
        "split_push", "objective_action", "split_push",
        "high", tuple(SUBJ_PUSH), "分推牵制"),
    "siege": Category(
        "siege", "objective_action", "siege",
        "high", tuple(SUBJ_PUSH), "拆高地塔"),
    "defend_inner": Category(
        "defend_inner", "objective_action", "defend_inner",
        "high", tuple(SUBJ_DEFEND), "回防高地"),
    "recall": Category(
        "recall", "objective_action", "recall",
        "medium", tuple(SUBJ_RECALL), "回城补给"),
    # -- 站位处理（positioning_action） ----------------------------------
    "push_wave": Category(
        "push_wave", "positioning_action", "push_wave",
        "medium", tuple(SUBJ_POSITION), "推线压塔"),
    "pull_back": Category(
        "pull_back", "positioning_action", "pull_back",
        "medium", tuple(SUBJ_POSITION), "回撤拉扯"),
    "ward_vision": Category(
        "ward_vision", "positioning_action", "ward_vision",
        "medium", tuple(SUBJ_POSITION), "游走做视野"),
    "hold_position": Category(
        "hold_position", "positioning_action", "hold_position",
        "medium", tuple(SUBJ_POSITION), "卡位蹲伏"),
    "rotate_support": Category(
        "rotate_support", "positioning_action", "rotate_support",
        "high", tuple(SUBJ_POSITION), "转线支援"),
    "cover_carry": Category(
        "cover_carry", "positioning_action", "cover_carry",
        "high", tuple(SUBJ_POSITION), "保护射手位"),
    # -- 团战处理（teamfight_action） ------------------------------------
    "engage": Category(
        "engage", "teamfight_action", "engage",
        "high", tuple(SUBJ_TEAMFIGHT), "开团先手"),
    "focus_target": Category(
        "focus_target", "teamfight_action", "focus_target",
        "high", tuple(SUBJ_TEAMFIGHT), "集火秒人"),
    "protect_carry": Category(
        "protect_carry", "teamfight_action", "protect_carry",
        "high", tuple(SUBJ_TEAMFIGHT), "保护C位输出"),
    "retreat": Category(
        "retreat", "teamfight_action", "retreat",
        "medium", tuple(SUBJ_TEAMFIGHT), "撤退脱离"),
    "kite": Category(
        "kite", "teamfight_action", "kite",
        "medium", tuple(SUBJ_TEAMFIGHT), "拉扯消耗"),
    "split_fight": Category(
        "split_fight", "teamfight_action", "split_fight",
        "medium", tuple(SUBJ_TEAMFIGHT), "分散拉扯"),
}


# --------------------------------------------------------------------------
# 关键词表
# --------------------------------------------------------------------------

#: 关键词 -> 类别名。**顺序即优先级**：靠前的是更具体的动作，
#: 靠后的是笼统的词。先命中先用。
#:
#: 写复合词（「控主宰」「开暴君」）而不是裸名词，是因为裸名词
#: 经常只是修饰语：「带主宰先锋拆边路二塔」里主宰不是要玩家去打龙。
KEYWORDS: List[Tuple[str, str]] = [
    # -- 河道之灵 -------------------------------------------------------
    ("河道之灵", "take_river_spirit"),
    ("中路河道", "take_river_spirit"),
    ("河灵", "take_river_spirit"),
    ("河蟹", "take_river_spirit"),
    ("蟹坑", "take_river_spirit"),
    ("控河", "take_river_spirit"),

    # -- 惩击 / 大龙 / 小龙 --------------------------------------------
    ("惩击留到暴君", "take_dragon"),
    ("惩击留在龙", "take_dragon"),
    ("惩击留到最后一刻", "take_dragon"),
    ("惩击留在最后一刻", "take_dragon"),
    ("惩击留到最后", "take_dragon"),
    ("惩击留到", "take_dragon"),
    ("抢暴君", "contest_dragon"),
    ("抢下暴君", "take_dragon"),
    ("拿下暴君", "take_dragon"),
    ("拿暴君", "take_dragon"),
    ("收掉暴君", "take_dragon"),
    ("打掉暴君", "take_dragon"),
    ("开暴君", "contest_dragon"),
    ("开暗影暴君", "contest_dragon"),
    ("打暴君", "contest_dragon"),
    ("控暴君", "contest_dragon"),
    ("暴君增益", "take_dragon"),
    ("龙 buff", "take_dragon"),
    ("龙buff", "take_dragon"),
    ("有龙", "take_dragon"),
    ("占龙坑", "contest_dragon"),
    ("占坑", "contest_dragon"),
    ("集合占坑", "contest_dragon"),
    ("龙坑集合", "contest_dragon"),
    ("蹲暴君", "contest_dragon"),
    ("暴君倒计时", "contest_dragon"),
    ("暴君刷新", "contest_dragon"),
    ("开暗影主宰", "fight_baron"),
    ("控暗影主宰", "fight_baron"),
    ("开主宰", "fight_baron"),
    ("控主宰", "fight_baron"),
    ("打主宰", "fight_baron"),
    ("抢主宰", "fight_baron"),
    ("拿主宰", "fight_baron"),
    ("风暴龙王", "fight_baron"),
    ("开龙王", "fight_baron"),
    ("控龙王", "fight_baron"),
    ("打龙王", "fight_baron"),
    ("龙王刷新", "fight_baron"),
    ("大龙", "fight_baron"),
    # 「与打野一起打发育路与对抗路之间的暴君」—— 动词和名词之间隔着字，
    # 复合词匹配不到，所以补一条「之间的暴君」。
    ("之间的暴君", "take_dragon"),

    # -- 视野 -----------------------------------------------------------
    ("插眼", "ward_vision"),
    ("放眼", "ward_vision"),
    ("补一次眼", "ward_vision"),
    ("插一个眼", "ward_vision"),
    ("做视野", "ward_vision"),
    ("铺视野", "ward_vision"),
    ("探草", "ward_vision"),
    ("试探草丛", "ward_vision"),
    ("草丛试探", "ward_vision"),
    ("草丛口", "ward_vision"),
    ("草丛边缘", "ward_vision"),
    ("视野精灵", "clear_vision_sprite"),

    # -- 野区 -----------------------------------------------------------
    ("反向开局", "farm_jungle"),
    ("进野区", "farm_jungle"),
    ("刷满", "farm_jungle"),
    ("抢野怪", "farm_jungle"),
    ("补经济", "farm_jungle"),
    ("积攒经济", "farm_jungle"),
    ("收野怪", "farm_jungle"),
    ("刷野", "farm_jungle"),
    ("清野", "farm_jungle"),
    ("刷到 4 级", "farm_jungle"),
    ("清到 4 级", "farm_jungle"),
    ("打野怪", "farm_jungle"),
    ("收野", "farm_jungle"),
    ("自家野区", "farm_jungle"),
    ("己方野区", "farm_jungle"),
    ("安全野区", "farm_jungle"),
    ("反野", "invade_jungle"),
    ("入侵", "invade_jungle"),
    ("换野区", "invade_jungle"),
    ("另一半野区", "invade_jungle"),
    ("对方野区", "invade_jungle"),
    ("对方蓝区", "invade_jungle"),
    ("控资源", "invade_jungle"),
    ("gank", "engage"),
    ("抓单", "engage"),

    # -- 兵线 / 推塔 -----------------------------------------------------
    ("拆塔", "siege"),
    ("二塔", "siege"),
    ("拆二塔", "siege"),
    ("拆外塔", "siege"),
    ("拆高地", "siege"),
    ("拆水晶", "siege"),
    ("推水晶", "siege"),
    ("压高地", "siege"),
    ("带线", "split_push"),
    ("分推", "split_push"),
    ("牵制", "split_push"),
    ("转推", "split_push"),
    ("推塔", "push_wave"),
    ("压塔", "push_wave"),
    ("推线", "push_wave"),
    ("抢线", "push_wave"),
    ("推过河", "push_wave"),
    ("兵线推过河", "push_wave"),
    ("带兵线", "push_wave"),
    ("带兵推", "push_wave"),
    ("跟线", "push_wave"),
    ("推进", "push_wave"),
    ("压上", "push_wave"),
    ("补刀", "last_hit"),
    ("尾刀", "last_hit"),
    ("收兵", "last_hit"),
    ("收线", "last_hit"),
    ("清线", "last_hit"),
    ("清兵", "last_hit"),
    ("兵线", "last_hit"),
    ("断兵线", "last_hit"),

    # -- 防守 / 回城 -----------------------------------------------------
    ("守高地", "defend_inner"),
    ("守龙兵线", "defend_inner"),
    ("清龙兵线", "defend_inner"),
    ("回防守高地", "defend_inner"),
    ("守内塔", "defend_inner"),
    ("回城", "recall"),
    ("补给", "recall"),
    ("买装备", "recall"),

    # -- 支援 / 保护 -----------------------------------------------------
    ("支援", "rotate_support"),
    ("转线", "rotate_support"),
    ("游走", "rotate_support"),
    ("集结", "rotate_support"),
    ("集合", "rotate_support"),
    ("赶到", "rotate_support"),
    ("保护己方 C 位", "protect_carry"),
    ("保护射手", "cover_carry"),
    ("护后排", "cover_carry"),
    ("保护 C 位", "protect_carry"),
    ("保护C位", "protect_carry"),
    ("护住 C 位", "protect_carry"),

    # -- 站位 / 后撤 -----------------------------------------------------
    ("不第一个进场", "hold_position"),
    ("落在己方坦克之后", "hold_position"),
    ("等己方前排", "hold_position"),
    ("等对方控制", "hold_position"),
    ("当前排", "hold_position"),
    ("顶在阵型前", "hold_position"),
    ("站在己方前排身后", "hold_position"),
    ("跟在己方前排后面", "hold_position"),
    ("对方位移交出后不再压上", "pull_back"),
    ("不再压上", "pull_back"),
    ("不追", "pull_back"),
    ("不回头", "pull_back"),
    ("后撤", "pull_back"),
    ("退回", "pull_back"),
    ("回撤", "pull_back"),
    ("退到", "pull_back"),
    ("缩到", "hold_position"),
    ("缩塔", "hold_position"),
    ("龟缩", "hold_position"),
    ("蹲", "hold_position"),
    ("埋伏", "hold_position"),
    ("卡位", "hold_position"),
    ("站位", "hold_position"),
    ("守塔下", "hold_position"),
    ("塔下待机", "hold_position"),
    ("后置保护", "hold_position"),

    # -- 团战：开团 / 集火 ----------------------------------------------
    ("开团", "engage"),
    ("先手", "engage"),
    ("起手", "engage"),
    ("进团", "engage"),
    ("进场", "engage"),
    ("切入", "engage"),
    ("突进", "engage"),
    ("贴脸", "engage"),
    ("贴身", "engage"),
    ("对拼", "engage"),
    ("换血", "engage"),
    ("接战", "engage"),
    ("接一套", "engage"),
    ("打满一套", "engage"),
    ("控住对方", "engage"),
    ("开大", "engage"),
    ("放大", "engage"),
    ("绕后", "engage"),
    ("绕侧后", "engage"),
    ("绕到侧翼", "engage"),
    ("绕到侧后", "engage"),
    ("侧翼", "engage"),
    ("包抄", "engage"),
    ("集火", "focus_target"),
    ("秒人", "focus_target"),
    ("落点选对方 C 位", "focus_target"),
    ("切后排", "focus_target"),
    ("贴后排", "focus_target"),
    ("锁定敌方后排", "focus_target"),
    ("打对方 C 位", "focus_target"),
    ("对方后排", "focus_target"),
    ("己方后排", "focus_target"),
    ("后排输出", "focus_target"),
    ("输出对方", "focus_target"),
    ("压血线", "focus_target"),
    ("持续输出", "focus_target"),
    ("远程普攻", "focus_target"),

    # -- 团战：撤退 / 拉扯 ----------------------------------------------
    ("撤退", "retreat"),
    ("脱离", "retreat"),
    ("退出战场", "retreat"),
    ("撤出", "retreat"),
    ("拉开距离", "retreat"),
    ("分散", "split_fight"),
    ("分开打", "split_fight"),
    ("拉扯消耗", "kite"),
    ("拉扯", "kite"),
    ("消耗", "kite"),
    ("放风筝", "kite"),
    ("不单独进团", "split_fight"),

    # -- 通用打法（英雄技能各不相同，但意图是这几类） --------------------
    # 「脱离被控范围后重进」= 被控后挣出来再进场，是开团不是撤退。
    # 光看「脱离」会翻成 retreat —— 与原文「不退出前排位置」正好相反。
    ("重进", "engage"),
    ("直接进", "engage"),
    ("开打", "engage"),
    ("贴身", "engage"),
    ("贴近", "engage"),
    ("定胜负", "engage"),
    ("一套爆发", "engage"),
    ("连招", "engage"),
    ("接大招", "engage"),
    ("打一套", "engage"),
    ("打第二轮", "engage"),
    ("逼对方", "engage"),
    ("聚怪", "engage"),
    ("收掉", "engage"),
    ("收后排", "focus_target"),
    ("压血", "focus_target"),
    ("拉开", "retreat"),
    ("退开", "pull_back"),
    ("顶住", "hold_position"),
    ("清中线", "last_hit"),
]


#: 否定词。出现在关键词**前面**（同一小句内）就说明这个动作**不该做**，
#: 不能进期望值。
NEGATIONS = (
    "不进", "不交", "不打", "不追", "不贪", "不回", "不硬", "不接",
    "不参", "不跟", "不刷", "不插", "不探", "不占", "不推", "不拆",
    "不控", "不抢", "不拿", "不守", "不撤", "不切", "不绕", "不压",
    "不换", "不补", "不打", "别", "勿", "不用", "不要", "无需",
    "放弃", "避免", "拒绝", "不为", "不",
)


def _negated(clause: str, start: int) -> bool:
    """关键词 ``clause[start:]`` 之前是否被否定。"""
    window = clause[max(0, start - 4):start]
    return any(neg in window for neg in NEGATIONS)


def _split_clauses(text: str) -> List[str]:
    """把一行动作切成小句。"""
    parts = re.split(r"[，；;、]", text)
    return [p.strip() for p in parts if p.strip()]


def _clause_categories(clause: str, fact: Optional[str] = None) -> List[str]:
    """一小句 -> 命中的类别名（按优先级从高到低）。

    ``fact`` 给定时只取该动作族的命中：规则里的期望动作只能挂在
    一个事实下，先定族再挑词，跨族的词才不会互相顶掉。
    """
    hits: List[Tuple[int, str]] = []
    for rank, (kw, cat) in enumerate(KEYWORDS):
        if fact is not None and CATEGORIES[cat].fact != fact:
            continue
        idx = clause.find(kw)
        if idx < 0:
            continue
        if _negated(clause, idx):
            continue
        if kw == "脱离" and any(
            cc in clause[idx: idx + 6] for cc in ("被控", "控制")
        ):
            # 「脱离被控范围后重进」是被控后挣出来**再进场**，不是撤退 ——
            # 这条小句的性质由后面的「重进」（engage）决定。
            continue
        hits.append((rank, cat))
    hits.sort()
    out: List[str] = []
    for _, cat in hits:
        if cat not in out:
            out.append(cat)
    return out


#: 裸怪物名词 + 动作动词 -> 怪物类别。
#:
#: 攻略常写成「与打野一起打发育路与对抗路之间的暴君」——
#: 动词和名词之间隔着字，复合词匹配不上，但意图很清楚：打暴君。
#: 这里补这一条，避免把「暴君」当修饰语丢掉整条动作。
_MONSTER_VERBS = ("打", "开", "控", "抢", "拿", "收", "惩击", "打掉", "拿下")


def _monster_by_verb(clause: str) -> Optional[str]:
    if not any(v in clause for v in _MONSTER_VERBS):
        return None
    if "主宰" in clause or "大龙" in clause or "龙王" in clause:
        return "fight_baron"
    if "暴君" in clause or "小龙" in clause or "龙坑" in clause:
        return "contest_dragon"
    return None


def classify_line(text: str) -> List[str]:
    """一行动作文本 -> 命中的类别名（去重，保持出现顺序）。"""
    out: List[str] = []
    for clause in _split_clauses(text):
        for cat in _clause_categories(clause):
            if cat not in out:
                out.append(cat)
    if not out:
        # 一个复合词都没命中，但小句里可能有裸怪物名词 + 动词
        for clause in _split_clauses(text):
            cat = _monster_by_verb(clause)
            if cat and cat not in out:
                out.append(cat)
    return _bind_smite(out, text)


def _bind_smite(cats: List[str], text: str) -> List[str]:
    """把「惩击」绑到这一行真正说的那只野怪上。

    「控暗影主宰，惩击留到最后一刻」说的是主宰，「惩击留到暴君
    最后一刻」说的是暴君 —— 惩击本身不指明打谁，靠同行的怪物名
    才知道。绑错就会把「打主宰」说成「打小龙」。
    """
    if not any(c in ("take_dragon", "fight_baron") for c in cats):
        return cats
    if "惩击" not in text:
        return cats
    if any(w in text for w in ("主宰", "大龙", "龙王")):
        return [("fight_baron" if c == "take_dragon" else c) for c in cats]
    if any(w in text for w in ("暴君", "小龙", "龙坑")):
        return [("contest_dragon" if c == "fight_baron" else c) for c in cats]
    return cats


def classify(text: str) -> List[Category]:
    """一行动作文本 -> 类别列表。

    只返回**同一个动作族**的类别：规则里的期望动作只能挂在
    一个事实下（``objective_action`` / ``positioning_action`` /
    ``teamfight_action`` 三选一），混着给就没法判了。
    """
    cats = [CATEGORIES[c] for c in classify_line(text) if c in CATEGORIES]
    if not cats:
        return []
    order = {"objective_action": 0, "positioning_action": 1, "teamfight_action": 2}
    best_fact = min((order[c.fact] for c in cats))
    return [c for c in cats if order[c.fact] == best_fact]


#: 前提里出现这些词，说明这一段的主题是打架 / 拿资源 / 守线。
#:
#: 前提决定「这段该做什么」，动作写的是「具体怎么做」——
#: 两者族别不一致时（前提说打团、动作说清线），以前提为准：
#: 「团战中对方先打出控制」这个前提下的「不进场，等对方控制交完」
#: 是团战里的后手，不是「这五分钟该清线」。
_FACT_HINTS: List[Tuple[str, str, int]] = [
    ("teamfight_action", ("团战", "开团", "团", "决战", "高地团", "龙坑团",
                          "团灭", "打架", "交战"), 3),
    ("teamfight_action", ("伏击", "对位", "换血",
                          "单杀", "刺客", "突进", "开打", "位移", "窗口", "越塔", "反打",
                          "抓人", "抓单"), 2),
    ("objective_action", ("龙", "暴君", "主宰", "资源", "河蟹", "河道之灵",
                          "野区", "buff", "BUFF"), 2),
    ("positioning_action", ("兵线", "推线", "塔", "线权", "草丛", "视野",
                            "河道", "站位", "视野精灵"), 2),
]


def fact_order(premise: str) -> List[str]:
    """按前提给出动作族的优先顺序。"""
    score = {"objective_action": 0, "positioning_action": 0, "teamfight_action": 0}
    for fact, words, weight in _FACT_HINTS:
        if any(w in premise for w in words):
            score[fact] += weight
    order = ["objective_action", "positioning_action", "teamfight_action"]
    return sorted(order, key=lambda f: (-score[f], order.index(f)))


def classify_branch(premise: str, actions: Sequence[str]) -> List[Category]:
    """一条分支（前提 + 若干动作）-> 期望动作的类别。

    动作族按**前提**决定：前提说的是「这一段在干什么」，
    动作族应当与之相符 —— 「团战中对方先打出控制」这个前提下的
    「等对方控制交完后切入」是团战里的后手，不是「这五分钟该清线」。
    前提定下的族一个都没命中时，才按动作自己的族退回去。
    """
    lines = list(actions)
    order = fact_order(premise)
    for fact in order:
        cats: List[Category] = []
        for line in lines:
            for name in _line_categories(line, fact):
                cat = CATEGORIES[name]
                if cat not in cats:
                    cats.append(cat)
        if cats:
            return cats
    return []


def _line_categories(line: str, fact: str) -> List[str]:
    """一行里属于某个动作族的类别名。

    先按小句切分；每小句取该族里优先级最高的命中；
    沟通类小句（「通知边路集结」「让游走探草」）跳过 ——
    那是给别人下指令，不是玩家自己的操作。
    """
    out: List[str] = []
    for clause in _split_clauses(line):
        if _is_communication(clause):
            continue
        hits = _clause_categories(clause, fact)
        if not hits:
            continue
        for cat in hits:
            if cat not in out:
                out.append(cat)
    return _bind_smite(out, line)


#: 沟通类小句的标记。这些内容系统记不下来（不是玩家的地图操作），
#: 也不该进期望动作。
_COMMUNICATION = (
    "通知", "让游走", "让边路", "让打野", "让中单", "让队友", "让射手",
    "报点", "报敌方", "喊", "告诉", "信号", "打字",
)


def _is_communication(clause: str) -> bool:
    return any(w in clause for w in _COMMUNICATION)


def subject_for(categories: Sequence[Category]) -> List[str]:
    """由类别推出适用时机（subject）取值。"""
    out: List[str] = []
    for c in categories:
        for v in c.subject:
            if v not in out:
                out.append(v)
    return out
