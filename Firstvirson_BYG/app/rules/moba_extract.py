"""MOBA 攻略抽取：把「前期打野该控龙」变成可判定的规则。

为什么需要单独一套
------------------
现有的 :mod:`app.rules.extractor` 是围绕自走棋写的：认「4-1」这种阶段坐标、
认「金币/人口」这类数值事实。而 MOBA 攻略长这样：

    前期打野以刷野为主，尽快到四级抓人
    中期游走位要跟着打野节奏入侵野区
    后期视野很重要，开龙前先把河道点亮

三种信号它一个都不认：**阶段**是「前期/中期/后期」而不是「4-1」，
**位置**是「打野/游走」，动作词也全是 MOBA 的（控龙、抓单、点亮河道）。
硬套自走棋词表的结果只有一个：抽出零条规则，或者更糟 —— 抽出
一条没有阶段限制的全局规则，看起来像那么回事，其实每局都在触发。

抽取口径：宁缺毋滥
------------------
一条 MOBA 规则至少要凑齐**位置 + 阶段 + 动作**才算数。缺任何一样就丢掉。

  · 只有动作没有位置  → 那是常识，不是规则（「记得补兵」）
  · 只有动作没有阶段  → 每局都触发，等于没有门槛
  · 只有位置没有动作  → 描述而非判定

宁可漏掉十句，也不放一条「看着对但没约束力」的规则进库 ——
规则库的价值在于每一条都能被对局数据验证。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..core.dsl import Condition, Rule, Source, Trigger

# --------------------------------------------------------------------------
# 词表
# --------------------------------------------------------------------------

#: 分路词 -> 规范值。和适配器共用同一套映射，避免两边认知不一致。
POSITION_WORDS: Dict[str, str] = {
    "打野": "jungle", "野核": "jungle", "刺客": "jungle",
    "中单": "mid", "中路": "mid", "法师": "mid",
    "游走": "roam", "游走位": "roam",
    "对抗路": "side", "边路": "side", "对抗": "side", "战士": "side",
    "射手": "carry", "c位": "carry", "发育路": "carry",
    "辅助": "support", "坦克": "support", "前排": "support",
}

#: 阶段词 -> 规范值
#:
#: 「决胜期 / 大龙期 / 末期」归到后期：攻略的时间轴只有三段，
#: 而这三个词指的都是「10 分钟以后」那一段。硬拆成第四档，
#: 抽出来的模式会落在一个攻略没单独讨论过的区间里。
PHASE_WORDS: Dict[str, str] = {
    "前期": "early", "开局": "early", "刚开局": "early", "前期发育": "early",
    "中期": "mid", "中期游走": "mid",
    "后期": "late", "大后期": "late",
    "决胜期": "late", "大龙期": "late", "末期": "late",
}

#: 阶段 -> 默认时间窗（分钟）。攻略只写「中期」，要能判定就得落成时间区间。
#:
#: 直接复用适配器的 PHASE_TIME_WINDOWS：抽取出来的规则和操作模式库
#: 必须用**同一套**时间窗。两处各写一份，迟早会漂，而漂了之后
#: 抽出来的规则永远对不上同一阶段的既有模式，而没人看得出来。
from ..adapters.honor_of_kings import PHASE_TIME_WINDOWS as PHASE_WINDOWS

#: MOBA 动作词 -> (动作事实, 取值)
#:
#: 只收录**能从画面或选择上判定**的动作。「意识好」「注意节奏」这类
#: 无法判定，不进表 —— 收了也只会变成噪音。
ACTION_WORDS: Dict[str, Tuple[str, str]] = {
    # 打野 / 刷野
    "刷野": ("objective_action", "farm_jungle"),
    "刷野怪": ("objective_action", "farm_jungle"),
    "gank": ("teamfight_action", "engage"),
    "抓人": ("teamfight_action", "engage"),
    "抓单": ("teamfight_action", "engage"),
    "入侵": ("objective_action", "invade_jungle"),
    "反野": ("objective_action", "invade_jungle"),
    # 地图目标
    "控龙": ("objective_action", "contest_dragon"),
    "开龙": ("objective_action", "contest_dragon"),
    "拿龙": ("objective_action", "take_dragon"),
    "开大龙": ("objective_action", "fight_baron"),
    "大龙团": ("objective_action", "fight_baron"),
    # 带线 / 推塔
    "推塔": ("objective_action", "push_wave"),
    "推线": ("objective_action", "push_wave"),
    "带线": ("objective_action", "push_wave"),
    "清线": ("objective_action", "push_wave"),
    "拆塔": ("objective_action", "siege"),
    "拆高地": ("objective_action", "siege"),
    "分推": ("objective_action", "split_push"),
    # 视野 / 游走
    "做视野": ("positioning_action", "ward_vision"),
    "点亮": ("positioning_action", "ward_vision"),
    "插眼": ("positioning_action", "ward_vision"),
    "支援": ("positioning_action", "rotate_support"),
    "转线": ("positioning_action", "rotate_support"),
    "绕后": ("positioning_action", "rotate_support"),
    "蹲": ("positioning_action", "hold_position"),
    "卡位": ("positioning_action", "hold_position"),
    # 团战
    "开团": ("teamfight_action", "engage"),
    "先手": ("teamfight_action", "engage"),
    "集火": ("teamfight_action", "focus_target"),
    "保护c位": ("teamfight_action", "protect_carry"),
    "保护射手": ("teamfight_action", "protect_carry"),
    "撤退": ("teamfight_action", "retreat"),
    "拉扯": ("teamfight_action", "kite"),
    # 其他
    "回城": ("objective_action", "recall"),
    "补给": ("objective_action", "recall"),
}

#: 主观词 —— 命中就不抽。
#:
#: 判据是「模糊」而不是「语气」。``应该`` 不在这里：它是**规范表述**，
#: 「前期打野应该刷野」是攻略里最标准的句式，也是最该被抽成规则的一句。
#: 早先把「应该」列进来，结果整套攻略一句都抽不出来。
#:
#: 真正要拦的是那些没有可执行边界的话：尽量、注意、意识、看情况…
SUBJECTIVE = [
    "尽量", "注意", "意识", "感觉", "看情况", "酌情", "灵活",
    "一般来说", "一般来说", "比较多", "比较重要", "有优势", "合理",
    "比较合适", "不一定", "看自己", "自行判断",
]

#: 数值词表：这些事实能从局面里读出来，可以作为附加条件
NUMERIC_FACT_WORDS: Dict[str, List[str]] = {
    "gold_diff": ["经济差", "经济领先", "经济落后"],
    "tower_diff": ["防御塔差", "塔差", "推塔数"],
    "kill_diff": ["击杀差", "人头差"],
    "level": ["平均等级"],
}


@dataclass
class MobaCandidate:
    """一条 MOBA 规则候选。"""

    title: str
    position: str
    phase: str
    action_fact: str
    action_value: str
    game: str = "hok"
    version: str = ""
    severity: str = "medium"
    rationale: str = ""
    deviation_hint: str = ""
    extra_conditions: List[Condition] = field(default_factory=list)
    source_quote: str = ""
    source_guide: str = ""
    source_url: str = ""
    confidence: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_rule(self, rule_id: str) -> Rule:
        conds: List[Condition] = [
            Condition("position", "==", self.position),
            Condition(
                "time_min", "between",
                list(PHASE_WINDOWS[self.phase]),
            ),
        ] + self.extra_conditions
        return Rule(
            id=rule_id,
            game=self.game,
            version=self.version,
            title=self.title,
            category=f"{POSITION_CN.get(self.position, self.position)}·{PHASE_CN.get(self.phase, self.phase)}",
            trigger=Trigger(all=conds),
            expect_fact=self.action_fact,
            expect_op="in",
            expect_value=[self.action_value],
            severity=self.severity,
            rationale=self.rationale,
            deviation_hint=self.deviation_hint,
            tags=[POSITION_CN.get(self.position, self.position),
                  PHASE_CN.get(self.phase, self.phase)],
            source=Source(
                guide=self.source_guide or "攻略抽取",
                url=self.source_url,
                quoted=self.source_quote,
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "position": self.position,
            "position_label": POSITION_CN.get(self.position, self.position),
            "phase": self.phase,
            "phase_label": PHASE_CN.get(self.phase, self.phase),
            "time_window": PHASE_WINDOWS[self.phase],
            "action_fact": self.action_fact,
            "action_value": self.action_value,
            "extra_conditions": [
                {"fact": c.fact, "op": c.op, "value": c.value}
                for c in self.extra_conditions
            ],
            "severity": self.severity,
            "confidence": round(self.confidence, 2),
            "notes": self.notes,
            "source_quote": self.source_quote,
            "trigger": {
                "all": [
                    {"fact": "position", "op": "==", "value": self.position},
                    {"fact": "time_min", "op": "between",
                     "value": list(PHASE_WINDOWS[self.phase])},
                ] + [
                    {"fact": c.fact, "op": c.op, "value": c.value}
                    for c in self.extra_conditions
                ],
                "any": [], "none": [],
            },
            "expect": {"fact": self.action_fact, "op": "in",
                       "value": [self.action_value]},
        }


POSITION_CN = {
    "jungle": "打野", "mid": "中路", "roam": "游走",
    "side": "对抗路", "carry": "射手", "support": "辅助",
    "unknown": "未指定",
}
PHASE_CN = {"early": "前期", "mid": "中期", "late": "后期"}


# --------------------------------------------------------------------------
# 抽取
# --------------------------------------------------------------------------


def split_sentences(text: str) -> List[str]:
    out = []
    for part in re.split(r"(?<=[。！？；!?;\n])", text):
        s = part.strip()
        if s:
            out.append(s)
    return out


def _match_position(sentence: str) -> Optional[str]:
    """取**最先出现**的分路词作为该句的主体位置。

    中文里主语在前：「中期**游走位**要跟着打野入侵」讲的是游走，
    不是打野。取第一个出现的位置词才对；把所有出现过的都收集起来会
    判成「位置不唯一」而整句丢弃 —— 那是把「游走跟着打野」
    这种最常见的写法判没了。

    真出现两个并列主体（「打野和中路都要插眼」）仍然返回 None，
    因为那确实说不清这条规则是给谁看的。
    """
    best_pos = None
    best_at = len(sentence) + 1
    for word, value in POSITION_WORDS.items():
        at = sentence.find(word)
        if at == -1:
            continue
        # 同位置时取更长的词：「游走位」优于「游走」
        if at < best_at or (at == best_at and len(word) > len(
                next((w for w, v in POSITION_WORDS.items() if v == value), ""))):
            if at < best_at:
                best_at = at
                best_pos = value
    if best_pos is None:
        return None

    # 并列主体检测：「A和B」「A、B」紧挨着出现 → 说不清给谁看
    others = [(sentence.find(w), v) for w, v in POSITION_WORDS.items()
              if sentence.find(w) != -1]
    tight = [(at, v) for at, v in others
             if abs(at - best_at) <= 4 and v != best_pos]
    return None if tight else best_pos


def _match_phases(sentence: str) -> List[str]:
    """返回句子提到的**全部**阶段，按出现顺序。

    一句「刺客前期要gank，中期带线」对应两条规则 —— 前期该 gank、
    中期该带线。返回单个阶段会把其中一半丢掉，那比抽错更难发现：
    用户会以为攻略里就写了这一点。
    """
    hits = []
    for word, value in PHASE_WORDS.items():
        at = sentence.find(word)
        if at != -1:
            hits.append((at, value))
    seen = set()
    ordered = []
    for _, v in sorted(hits):
        if v not in seen:
            seen.add(v)
            ordered.append(v)
    return ordered


def _match_action(sentence: str) -> List[Tuple[str, str]]:
    out = []
    for word, pair in ACTION_WORDS.items():
        if word in sentence:
            out.append(pair)
    # 去重：控龙 / 开龙 映射到同一个取值
    seen = set()
    uniq = []
    for fact, value in out:
        if (fact, value) in seen:
            continue
        seen.add((fact, value))
        uniq.append((fact, value))
    return uniq


def _match_numeric(sentence: str, known_facts: set) -> List[Condition]:
    conds = []
    for fact, words in NUMERIC_FACT_WORDS.items():
        if fact not in known_facts:
            continue
        if not any(w in sentence for w in words):
            continue
        # 只取「领先 / 落后」这种能定成条件的表述
        if "落后" in sentence:
            conds.append(Condition(fact, "<=", -1))
        elif "领先" in sentence:
            conds.append(Condition(fact, ">=", 1))
    return conds


def _is_subjective(sentence: str) -> List[str]:
    return [w for w in SUBJECTIVE if w in sentence]


def extract_moba(
    text: str,
    known_facts: set,
    version: str = "S38",
    game: str = "hok",
    guide_title: str = "",
    guide_url: str = "",
    min_confidence: float = 0.4,
) -> List[MobaCandidate]:
    """从攻略文本抽取 MOBA 规则候选。

    只产出**位置 + 阶段 + 动作**三者齐备的候选。缺任何一项就丢弃，
    并在 ``skipped`` 里说明原因 —— 用户该知道哪些句子被丢了、为什么。
    """
    cands: List[MobaCandidate] = []
    seen = set()

    for raw in split_sentences(text):
        sentence = raw.strip()
        if len(sentence) < 6:
            continue

        subj = _is_subjective(sentence)
        pos = _match_position(sentence)
        phases = _match_phases(sentence)
        actions = _match_action(sentence)

        # 主谓宾缺一不可
        if not pos or not phases or not actions:
            continue
        if subj:
            continue

        fact, value = actions[0]
        conds = _match_numeric(sentence, known_facts)

        # 一句里的多个阶段各出一条：前期该 A、中期该 B，是两条规则
        for phase in phases:
            key = (pos, phase, fact, value)
            if key in seen:
                continue
            seen.add(key)
            cands.append(MobaCandidate(
                title=sentence.rstrip("。！？；.!?; "),
                position=pos,
                phase=phase,
                action_fact=fact,
                action_value=value,
                version=version,
                game=game,
                extra_conditions=conds,
                source_quote=sentence,
                source_guide=guide_title,
                source_url=guide_url,
                rationale=(
                    f"{POSITION_CN[pos]}在{PHASE_CN[phase]}"
                    f"（{PHASE_WINDOWS[phase][0]}-{PHASE_WINDOWS[phase][1]} 分钟）"
                    f"应当做「{value}」这类处理"
                ),
                deviation_hint=(
                    f"这一阶段你是{POSITION_CN[pos]}，"
                    f"应{POSITION_CN.get(value, value)}"
                ),
                confidence=0.95 if not conds else 0.8,
                notes=(
                    (["含局势条件：" + "，".join(c.fact for c in conds)] if conds else [])
                    + ([f"该句还提到阶段：{'、'.join(PHASE_CN[p] for p in phases)}"]
                       if len(phases) > 1 else [])
                ),
            ))

    cands.sort(key=lambda c: -c.confidence)
    return [c for c in cands if c.confidence >= min_confidence]


def explain_skips(text: str, known_facts: set) -> List[Dict[str, str]]:
    """说明哪些句子没被抽成规则、缺了什么。

    没有这个反馈，用户会以为抽取器"什么都看不出来"，然后去手工填 ——
    但工具的价值就在于把「哪些能自动化、哪些必须人来」摆清楚。
    """
    out = []
    for raw in split_sentences(text):
        s = raw.strip()
        if len(s) < 6:
            continue
        reasons = []
        if not _match_position(s):
            reasons.append("缺分路")
        if not _match_phases(s):
            reasons.append("缺阶段")
        if not _match_action(s):
            reasons.append("缺可判定动作")
        if _is_subjective(s):
            reasons.append("主观表述")
        if reasons:
            out.append({"sentence": s[:80], "reasons": "、".join(reasons)})
    return out


def stable_rule_id(cand: MobaCandidate) -> str:
    import hashlib

    digest = hashlib.sha1(
        f"{cand.position}|{cand.phase}|{cand.action_fact}|{cand.action_value}".encode()
    ).hexdigest()[:4]
    return f"HOK-{cand.position[:3].upper()}-{cand.phase[:3].upper()}-{digest}"
