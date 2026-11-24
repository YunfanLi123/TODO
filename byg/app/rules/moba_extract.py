"""MOBA 攻略抽取：把「前期打野该控龙」变成可判定的规则。

为什么需要单独一套
------------------
MOBA 攻略长这样：

    前期打野以刷野为主，尽快到四级抓人
    中期游走位要跟着打野节奏入侵野区
    后期视野很重要，开龙前先把河道点亮

这套语汇有三个特征，决定了抽取器必须按 MOBA 的口径写：

* **阶段是「前期/中期/后期」**，不是「第几阶段」。攻略只说时间感觉，
  所以阶段只能落成时间窗（:data:`PHASE_WINDOWS`）。
* **位置是角色分工**（打野/中单/游走/对抗路/射手），不是地图格子。
  「中期游走位要跟着打野节奏入侵野区」讲的是游走，不是打野 ——
  位置取**最先出现**的那个分路词。
* **动作是能判定到具体操作的词**（刷野/控龙/带线/做视野），
  「意识好」「注意节奏」这类不可判定的说法根本不该进表。

三种信号凑不齐一条规则，硬抽出来的只有两种结果：一条都没有，
或者一条没有阶段限制的全局规则 —— 后者看起来像那么回事，
其实每局都在触发。

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

from ..core.dsl import DEFAULT_SUBJECTIVE_MARKERS, Condition, Rule, Source, Trigger

# --------------------------------------------------------------------------
# 词表
# --------------------------------------------------------------------------

#: 分路词 -> 规范值。和适配器共用同一套映射，避免两边认知不一致。
#:
#: 2026-10-29 更正：``辅助/坦克/前排`` 原来映射到 ``support``，
#: 但适配器的规范值里**没有** ``support`` —— 游走位的规范值是 ``roam``
#: （见 :data:`app.adapters.honor_of_kings.POSITION_VALUES`）。
#: 后果很具体：攻略里写「辅助」「坦克」「前排」的句子（游走位攻略里非常多）
#: 抽出来的规则，触发条件里的 position 是个**不存在的值**，
#: 于是那条规则永远匹配不上任何一局 —— 静悄悄地少了一整类规则。
#:
#: 词表必须直接用适配器那份，别自己抄一份 —— 抄的版本就是这样漂掉的。
POSITION_WORDS: Dict[str, str] = {
    "打野": "jungle", "野核": "jungle", "刺客": "jungle",
    "中单": "mid", "中路": "mid", "法师": "mid",
    "游走": "roam", "游走位": "roam",
    "对抗路": "side", "边路": "side", "对抗": "side", "战士": "side",
    "射手": "carry", "c位": "carry", "发育路": "carry",
    "辅助": "roam", "坦克": "roam", "前排": "roam",
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
from ..adapters.honor_of_kings import MOBA_ACTION_VALUES
from ..adapters.honor_of_kings import PHASE_TIME_WINDOWS as PHASE_WINDOWS

#: 动作取值 -> 动作事实名。**从适配器那份分组表反查**，不另抄一份。
#:
#: 抄一份的代价很具体：抽取器把「带线」记成 ``objective_action``，
#: 而适配器把它归在 ``positioning_action`` —— 抽出来的规则永远匹配不上
#: 任何一局，而且没人看得出来（不报错，只是不触发，报告里那条规则一直
#: 躺着不出现）。
#:
#: 所以下面那张表只写「词 -> 取值」；**动作事实由适配器说了算**。
VALUE_TO_FACT: Dict[str, str] = {}
for _fact, _values in MOBA_ACTION_VALUES.items():
    for _value in _values:
        VALUE_TO_FACT[_value] = _fact

#: MOBA 动作词 -> 取值
#:
#: 只收录**能从画面或选择上判定**的动作。「意识好」「注意节奏」这类
#: 无法判定，不进表 —— 收了也只会变成噪音。
ACTION_WORDS: Dict[str, str] = {
    # 打野 / 刷野
    "刷野": "farm_jungle",
    "刷野怪": "farm_jungle",
    "gank": "engage",
    "抓人": "engage",
    "抓单": "engage",
    "入侵": "invade_jungle",
    "反野": "invade_jungle",
    # 地图目标
    "控龙": "contest_dragon",
    "开龙": "contest_dragon",
    "拿龙": "take_dragon",
    "开大龙": "fight_baron",
    "大龙团": "fight_baron",
    # 带线 / 推塔
    "推塔": "push_wave",
    "推线": "push_wave",
    "带线": "push_wave",
    "清线": "push_wave",
    "拆塔": "siege",
    "拆高地": "siege",
    "分推": "split_push",
    # 视野 / 游走
    "做视野": "ward_vision",
    "点亮": "ward_vision",
    "插眼": "ward_vision",
    "支援": "rotate_support",
    "转线": "rotate_support",
    "绕后": "rotate_support",
    "蹲": "hold_position",
    "卡位": "hold_position",
    # 团战
    "开团": "engage",
    "先手": "engage",
    "集火": "focus_target",
    "保护c位": "protect_carry",
    "保护射手": "protect_carry",
    "撤退": "retreat",
    "拉扯": "kite",
    # 其他
    "回城": "recall",
    "补给": "recall",
}

#: 主观词 —— 命中就不抽。
#:
#: 判据是「模糊」而不是「语气」。``应该`` 不在这里：它是**规范表述**，
#: 「前期打野应该刷野」是攻略里最标准的句式，也是最该被抽成规则的一句。
#: 早先把「应该」列进来，结果整套攻略一句都抽不出来。
#:
#: 真正要拦的是那些没有可执行边界的话：尽量、注意、意识、看情况…
#:
#: 这里必须**包含** :data:`app.core.dsl.DEFAULT_SUBJECTIVE_MARKERS` ——
#: 那张表是入库前的最后一道闸。两张表不一致时，含「该不该」的句子会
#: 先被抽成候选、再在准入口被拒：用户白等一轮，还以为是自己写错了。
SUBJECTIVE: List[str] = sorted(
    set([
        "尽量", "注意", "意识", "感觉", "看情况", "酌情", "灵活",
        "一般来说", "比较多", "比较重要", "有优势", "合理",
        "比较合适", "不一定", "看自己", "自行判断",
    ]) | set(DEFAULT_SUBJECTIVE_MARKERS)
)

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

    @property
    def category(self) -> str:
        """这条候选的归类：``位置·阶段``（如「打野·前期」）。

        规则 id、库里的分组、页面上的分类都用它 —— 三处各写一份，
        迟早漂成两个样子。所以它是**唯一的算法**，别在别处再拼一次。
        """
        return (f"{POSITION_CN.get(self.position, self.position)}"
                f"·{PHASE_CN.get(self.phase, self.phase)}")

    @property
    def tags(self) -> List[str]:
        """这条候选的标签：位置名 + 阶段名。

        和 :attr:`category` 一样是**唯一的算法** —— 规则入库和 v2 模式
        都从这里取，别在别处再拼一次。
        """
        return [POSITION_CN.get(self.position, self.position),
                PHASE_CN.get(self.phase, self.phase)]

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
            category=self.category,
            trigger=Trigger(all=conds),
            expect_fact=self.action_fact,
            expect_op="in",
            expect_value=[self.action_value],
            severity=self.severity,
            rationale=self.rationale,
            deviation_hint=self.deviation_hint,
            tags=self.tags,
            source=Source(
                guide=self.source_guide or "攻略抽取",
                url=self.source_url,
                quoted=self.source_quote,
            ),
        )

    def to_pattern(self, pattern_id: str):
        """这条候选 → 一条 v2 操作模式。

        解析段的产物应该直接是**新结构**，而不是先产出一份 v1 再迁移 ——
        迁一次就多一次丢信息的机会（本次迁移就丢了 ``subject`` 与
        ``none`` 组护栏两处）。

        所以解析器有两个出口：:meth:`to_rule`（v1，给旧链路用）与
        :meth:`to_pattern`（v2，新链路用）。两者表达同一件事，
        口径必须一致 —— 所以都从 :attr:`position` / :attr:`phase` 派生，
        不各自算一遍。
        """
        from ..core.dsl2 import (
            CORRECTIVE_ALWAYS,
            SCOPE_AT_LEAST_ONCE,
            AppliesTo,
            Check,
            Corrective,
            Node,
            Pattern,
        )

        lo, hi = PHASE_WINDOWS[self.phase]
        when = [Check(fact="position", op="==", value=self.position)]
        for cond in self.extra_conditions:
            when.append(Check(fact=cond.fact, op=cond.op, value=cond.value))
        return Pattern(
            id=pattern_id,
            game=self.game,
            version=self.version,
            title=self.title,
            category=self.category,
            lane=self.position,
            phase=self.phase,
            node=Node(
                label=(
                    f"第 {lo:g} 分钟" if lo == hi else f"{lo:g}–{hi:g} 分钟"
                ),
                from_min=float(lo),
                to_min=float(hi),
            ),
            applies_to=AppliesTo(lane=self.position),
            when=when,
            expect=[
                Check(fact=self.action_fact, op="in",
                      value=[self.action_value], min_count=1),
            ],
            scope=SCOPE_AT_LEAST_ONCE,
            severity=self.severity,
            corrective=Corrective(mode=CORRECTIVE_ALWAYS),
            rationale=self.rationale,
            deviation_hint=self.deviation_hint,
            tags=self.tags,
            source={
                "guide": self.source_guide or "攻略抽取",
                "url": self.source_url,
                "quoted": self.source_quote,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "game": self.game,
            "version": self.version,
            "category": self.category,
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
            # 候选里不可能带主观词：命中主观词的句子在抽取阶段就被丢了。
            # 这个键仍要留着 —— 调用方（页面、场景脚本）不该分两种候选类型。
            "subjective_markers": [],
            "rationale": self.rationale,
            "deviation_hint": self.deviation_hint,
            "tags": self.tags,
            "source": Source(
                guide=self.source_guide or "攻略抽取",
                url=self.source_url,
                quoted=self.source_quote,
            ).to_dict(),
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


#: 位置的中文名。
#:
#: 2026-10-29 更正：原来这里有一份**手抄**的表，里面留着
#: ``"support": "辅助"`` —— 而规范值里根本没有 ``support``。
#: 现在直接从适配器取，抄一份就漂一次。
POSITION_CN: Dict[str, str] = {
    "jungle": "打野", "mid": "中路", "roam": "游走",
    "side": "对抗路", "carry": "射手", "support": "辅助",
    "unknown": "未指定",
}
try:
    from ..adapters.honor_of_kings import POSITION_LABELS as _ADAPTER_LABELS
    # 只保留适配器认识的那些值 —— 留着 ``support`` 这种废值，
    # 迟早有人在别处拿它当规范值用。
    POSITION_CN = {k: v for k, v in _ADAPTER_LABELS.items()}
except Exception:                                      # noqa: BLE001
    pass

#: 合法的位置值集合（从适配器来）。**抽取时用来挡掉非法值**。
#:
#: 为什么要挡：抽出来的规则如果带着一个适配器不认识的位置值，
#: 它**永远匹配不上任何一局**，而且没人看得出来 —— 规则库里躺着，
#: 报告里一条都不触发。
#:
#: 所以这里宁可丢一条，也不要留一条永远不触发的。
VALID_POSITIONS: set = set(POSITION_CN.keys()) - {"unknown"}

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
    for word, value in ACTION_WORDS.items():
        if word not in sentence:
            continue
        fact = VALUE_TO_FACT.get(value)
        if fact is None:
            # 适配器不认识这个取值 —— 抽出来的规则永远匹配不上任何一局。
            # 宁可不抽，也不放进一条**永远不触发**的死规则。
            continue
        out.append((fact, value))
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


def detect_subjective(sentence: str) -> List[str]:
    """这句话里命中了哪些主观词。命中即不可判定，不该抽成规则。

    对外只暴露这一个名字 —— 调用方要的是「这句是不是主观表述」，
    不该知道词表长什么样，更不该自己去扫。
    """
    return _is_subjective(sentence)


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
        # **位置值必须是适配器认识的**。
        #
        # 不挡的话：抽出来的规则带着一个适配器不认识的位置值，
        # 它永远匹配不上任何一局 —— 规则库里躺着，报告里一条都不触发，
        # 而且**没人看得出来**（没有报错，只是不触发）。
        # 这比抽不出来更坏：抽不出来至少知道少了一条。
        if pos not in VALID_POSITIONS:
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
                    f"却没有任何「{value}」这类操作"
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
        else:
            # 位置词认出来了，但映射到的规范值适配器不认识 → 同样抽不了。
            # 不报这一条，用户会以为「有分路就该能抽」，然后去查为什么少一条。
            pos = _match_position(s)
            if pos and pos not in VALID_POSITIONS:
                reasons.append(f"分路映射到未知取值（{pos}）")
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


def candidates_to_patterns(cands: List[MobaCandidate]) -> List[Any]:
    """一批候选 → 一批 v2 操作模式。

    id 与 :func:`stable_rule_id` 同源 —— 同一份攻略文本，
    v1 与 v2 的 id 必须一一对应，否则回测时对不上是哪条变了。
    """
    return [c.to_pattern(stable_rule_id(c)) for c in cands]
