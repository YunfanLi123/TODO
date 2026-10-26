"""攻略抽取：非结构化攻略文本 → 结构化规则候选。

立场说明（重要，不要改）：
    本模块**不生产入库规则**，只产出「候选」。攻略文本到可判定规则之间
    存在语义鸿沟（「该稳一点」到「血量≥50 时不追星」需要人为补全阈值），
    任何声称全自动的方案都会在规则库里注水 —— 而报告的说服力恰恰
    来自每一条都站得住。

    因此流程是：抽取候选 → 可判定性准入筛选 → 人工确认 → 入库。
    抽取器做的是「把难的部分自动化」（切句、数值阈值、动作匹配、
    事实对齐），把「需要判断的部分」显式交给人。

抽取器的能力边界：
  ✓ 能抽：带显式阈值和阶段标记的条目（「4-1 血量 50 以上不要追星」）
  ✗ 不能抽：主观判断（「这波看情况打」）—— 它们会被词表拦下并标记原因
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..core.dsl import (
    DEFAULT_SUBJECTIVE_MARKERS,
    Condition,
    Rule,
    Source,
    Trigger,
)

# --------------------------------------------------------------------------
# 事实别名表：中文表述 -> 规范事实名
# --------------------------------------------------------------------------

#: 数值型事实 —— 抽取器会为其搜索数值阈值
#:
#: 只放**观测值的名词**，不放动作。「存钱」「攒钱」是玩家做什么，
#: 不是观测量什么；把它们列为金币别名会让「存钱吃利息」这句里的
#: 「存钱」吸走后面毫不相干的阈值。
NUMERIC_FACT_ALIASES: Dict[str, List[str]] = {
    "health": ["血量", "血量值", "生命值", "血条", "剩余血", "健康度"],
    "streak": ["连胜", "连胜数", "连胜场次", "连败", "连败数"],
    "gold": ["金币", "经济", "经济曲线", "经济值"],
    "level": ["人口", "等级", "人口数"],
    "income": ["收入", "经济收入"],
    "item_count": ["装备数量", "装备数", "装备件数"],
    "time_min": ["分钟", "游戏时间"],
    "gold_diff": ["经济差", "经济领先", "经济落后", "经济差值"],
    "tower_diff": ["防御塔差", "塔差", "推塔差"],
    "kill_diff": ["人头差", "击杀差", "人头数"],
}

#: 枚举型事实 —— 抽取器会搜索取值词
ENUM_FACT_ALIASES: Dict[str, List[str]] = {
    "stage": ["阶段", "回合", "第几阶段", "几阶段"],
    "pressure": ["局势", "局面", "压力", "血量健康", "健康"],
}

#: 动作事实 -> 中文动作表述
ACTION_FACT_ALIASES: Dict[str, List[str]] = {
    "econ_action": [
        "存钱", "攒钱", "不追星", "不升级", "断连", "断连胜", "连败", "省钱",
        "追星", "追三星", "冲人口", "上人口", "升级", "roll", "roll牌", "抽卡",
        "梭哈", "all in",
    ],
    "positioning_action": [
        "站位", "换位", "调整站位", "对位", "卡位", "前排", "开团站位",
    ],
    "objective_action": [
        "推一", "推塔", "开龙", "拿龙", "开大龙", "拿大龙", "抢龙", "开团", "打架",
        "支援", "游走", "抓单",
    ],
    "teamfight_action": ["开团", "接团", "集火", "后排", "前排开团"],
}

#: 动作取值词表：动作事实 -> [(取值, 中文表述)]
ACTION_VALUES: Dict[str, List[Tuple[str, str]]] = {
    "econ_action": [
        ("save", "存钱不追星"),
        ("break_streak", "主动断连胜"),
        ("rush_level", "冲人口上等级"),
        ("reroll", "大额抽卡"),
        ("all_in", "梭哈 all in"),
    ],
    "positioning_action": [
        ("adjust", "调整站位"),
        ("hold", "保持站位"),
        ("counter", "针对对位"),
    ],
    "objective_action": [
        ("take_baron", "开大龙"),
        ("take_dragon", "开小龙"),
        ("push_tower", "推一塔"),
        ("split_push", "分推"),
        ("retreat", "止损撤退"),
        ("fight", "开团"),
    ],
    "teamfight_action": [
        ("engage", "开团"),
        ("follow_up", "跟进"),
        ("protect_backline", "保护后排"),
    ],
}

# --------------------------------------------------------------------------
# 中文阈值表达
# --------------------------------------------------------------------------

#: 中文数量词 -> 数值
_CN_NUM = {
    "零": 0, "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}

_NUM_RE = re.compile(
    r"(?P<num>\d+(?:\.\d+)?|[零一两二三四五六七八九十]+)"
    r"\s*(?P<unit>血|点|个|级|人口|连胜|连败|金|金币|分钟|层)?"
)

#: 比较词 -> 操作符。
#:
#: **顺序至关重要**：必须从最长最具体的词开始匹配。
#: 之前「不低于」被「低于」抢先匹配（后者是前者的子串），于是
#: 「血量不低于 50」被抽成 ``health < 50`` —— 方向完全相反的规则会
#: 产生看似合理、实则彻底错误的结论。这类 bug 不会报错，只会安静地
#: 给出错误规则，所以这里刻意把否定式放在最前面。
_CMP_PATTERNS = [
    (r"(?:不少于|不低于|至少达到|至少)\s*", ">="),
    (r"(?:不多于|不高于|最多不超过|不超过|至多|最多)\s*", "<="),
    (r"(?:大于|超过|高于|多于)\s*", ">"),
    (r"(?:小于|低于|少于|不足)\s*", "<"),
    (r"(?:等于|恰好|正好)\s*", "=="),
    (r">=\s*", ">="),
    (r"<=\s*", "<="),
    (r">\s*", ">"),
    (r"<\s*", "<"),
]

_CMP_RE = re.compile("|".join(f"(?:{p})" for p, _ in _CMP_PATTERNS))


def _parse_number(text: str) -> Optional[float]:
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        pass
    if text in _CN_NUM:
        return float(_CN_NUM[text])
    if text.startswith("十") and len(text) == 2:
        return float(_CN_NUM.get(text[1], 0) + 10)
    if len(text) == 3 and text[1] == "十":
        return float(_CN_NUM.get(text[0], 0) * 10 + _CN_NUM.get(text[2], 0))
    if len(text) == 3 and text.endswith("十"):
        return float(_CN_NUM.get(text[0], 0) * 10)
    if len(text) == 2 and text.endswith("十"):
        return float(_CN_NUM.get(text[0], 0) * 10)
    return None


def _cn_to_int(text: str) -> str:
    val = _parse_number(text)
    return str(int(val)) if val is not None and val == int(val) else text


def _find_comparison(sentence: str) -> Optional[Tuple[str, Any]]:
    """在句子里找「比较词 + 数值」结构。

    顺序依赖 :data:`_CMP_PATTERNS` 的排列 —— 「不低于」必须先于「低于」
    被尝试，否则否定式会被当成肯定式，规则方向就反了。
    """
    m = _CMP_RE.search(sentence)
    if not m:
        # 没有显式比较词时，退而求其次找「数值 + 以上/以下」
        alt = re.search(r"(\d+(?:\.\d+)?)\s*(以上|以下|之内|以内)", sentence)
        if alt:
            op = ">=" if alt.group(2) == "以上" else "<="
            value = float(alt.group(1))
            # 与主分支保持一致：整数值不要留成 3.0 —— 规则里出现
            # ``streak >= 3.0`` 这种写法，读起来像是从别处算出来的
            return (op, int(value) if value == int(value) else value)
        return None

    matched = m.group(0).strip()
    op = "=="
    for pattern, candidate_op in _CMP_PATTERNS:
        if re.search(pattern, matched):
            op = candidate_op
            break

    rest = sentence[m.end():]
    num_match = _NUM_RE.search(rest)
    if not num_match:
        return None
    value = _parse_number(num_match.group("num"))
    if value is None:
        return None
    if op in (">=", "<=", ">", "<") and value == int(value):
        value = int(value)
    return (op, value)


def _find_comparison_after(sentence: str, alias: str, window: int = 8) -> Optional[Tuple[str, Any]]:
    """只在 ``alias`` 之后紧邻的窗口里找比较词。

    曾经这里有个「找不到就去整句找」的兜底，后果是「存钱吃利息」里的
    「存钱」匹配到金币别名后，在句尾找到了毫不相干的「达到 3 以上」，
    于是凭空生成一条 ``gold >= 3`` 的规则。

    阈值必须与它修饰的那个事实相邻 —— 不相邻就不该配对。宁可漏掉一条
    规则，也不要编一条错的。
    """
    start = 0
    while True:
        idx = sentence.find(alias, start)
        if idx == -1:
            return None
        tail = sentence[idx + len(alias): idx + len(alias) + window]
        found = _find_comparison(tail)
        if found is not None:
            return found
        start = idx + len(alias)


# --------------------------------------------------------------------------
# 候选规则
# --------------------------------------------------------------------------


@dataclass
class RuleCandidate:
    """抽取出的规则候选 —— 尚未入库，等待人工确认。"""

    title: str
    game: str
    version: str
    category: str = "待归类"
    trigger: Trigger = field(default_factory=Trigger)
    expect_fact: str = ""
    expect_op: str = "in"
    expect_value: Any = None
    severity: str = "medium"
    rationale: str = ""
    deviation_hint: str = ""
    tags: List[str] = field(default_factory=list)
    source: Optional[Source] = None
    #: 抽取置信度：数值阈值和动作都明确命中时高，只靠模糊匹配时低
    confidence: float = 0.0
    #: 抽取过程中发现的问题 / 待人工确认项
    notes: List[str] = field(default_factory=list)
    #: 命中的主观判断词
    subjective_markers: List[str] = field(default_factory=list)

    def to_rule(self, rule_id: str) -> Rule:
        return Rule(
            id=rule_id,
            game=self.game,
            version=self.version,
            title=self.title,
            category=self.category,
            trigger=self.trigger,
            expect_fact=self.expect_fact,
            expect_op=self.expect_op,
            expect_value=self.expect_value,
            severity=self.severity,
            status="active",
            rationale=self.rationale,
            deviation_hint=self.deviation_hint,
            tags=self.tags,
            source=self.source,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "game": self.game,
            "version": self.version,
            "category": self.category,
            "confidence": round(self.confidence, 2),
            "notes": self.notes,
            "subjective_markers": self.subjective_markers,
            "trigger": {
                "all": [
                    {"fact": c.fact, "op": c.op, "value": c.value}
                    for c in self.trigger.all
                ],
                "any": [
                    {"fact": c.fact, "op": c.op, "value": c.value}
                    for c in self.trigger.any
                ],
                "none": [
                    {"fact": c.fact, "op": c.op, "value": c.value}
                    for c in self.trigger.none
                ],
            },
            "expect": {
                "fact": self.expect_fact,
                "op": self.expect_op,
                "value": self.expect_value,
            },
            "rationale": self.rationale,
            "deviation_hint": self.deviation_hint,
            "tags": self.tags,
            "source": self.source.to_dict() if self.source else None,
        }


# --------------------------------------------------------------------------
# 抽取器
# --------------------------------------------------------------------------


def split_sentences(text: str) -> List[str]:
    """按中文标点切句，同时保留原文顺序。"""
    parts = re.split(r"(?<=[。！？；!?;\n])", text)
    return [p.strip() for p in parts if p and p.strip()]


def _match_fact(sentence: str, fact: str) -> bool:
    return any(alias in sentence for alias in _FACT_ALIASES.get(fact, []))


#: 动作事实 -> [(动作词, 动作取值编码)]。
#:
#: 结构说明：攻略里写的是动作**片段**（「存钱」「追三星」「别硬接」），
#: 不是规范取值名（「存钱不追星」）。匹配必须基于片段词表，
#: 再映射到取值编码 —— 否则整条规则会因为找不到「完整短语」而抽不出来。
ACTION_ALIAS_TO_VALUE: Dict[str, List[Tuple[str, str]]] = {
    "econ_action": [
        ("存钱", "save"),
        ("攒钱", "save"),
        ("省钱", "save"),
        ("吃利息", "save"),
        ("断连", "break_streak"),
        ("断连胜", "break_streak"),
        ("主动断", "break_streak"),
        ("连败", "break_streak"),
        ("追星", "all_in"),
        ("追三星", "all_in"),
        ("梭哈", "all_in"),
        ("all in", "all_in"),
        ("冲人口", "level_up"),
        ("上人口", "level_up"),
        ("升级", "level_up"),
        ("升人口", "level_up"),
        ("补人口", "level_up"),
        ("搜牌", "reroll"),
        ("抽卡", "reroll"),
        ("roll", "reroll"),
        ("锁血", "reroll"),
        ("强化主C", "upgrade_carry"),
        ("卖核心", "sell_core"),
    ],
    "positioning_action": [
        ("调整站位", "shift_lineup"),
        ("换位", "shift_lineup"),
        ("站位", "shift_lineup"),
        ("卡位", "shift_lineup"),
        ("对位", "counter_lineup"),
        ("保护主C", "protect_carry"),
        ("保护C位", "protect_carry"),
        ("缩角", "protect_carry"),
        ("保持站位", "hold_lineup"),
    ],
    "objective_action": [
        ("开大龙", "take_baron"),
        ("拿大龙", "take_baron"),
        ("大龙团", "take_baron"),
        ("开小龙", "take_dragon"),
        ("拿小龙", "take_dragon"),
        ("抢龙", "take_dragon"),
        ("推一塔", "push_tower"),
        ("拆高地", "siege"),
        ("分推", "split_push"),
        ("撤退", "retreat"),
        ("止损", "retreat"),
        ("游走", "ward_vision"),
        ("视野", "ward_vision"),
        ("刷野", "farm_jungle"),
        ("反野", "invade_jungle"),
        ("回防", "defend_inner"),
        ("支援", "rotate_support"),
        ("推线", "push_wave"),
        ("开团", "fight"),
        ("集火", "fight"),
    ],
    "teamfight_action": [
        ("开团", "engage"),
        ("先手", "engage"),
        ("跟进", "follow_up"),
        ("保护后排", "protect_backline"),
        ("保护C位", "protect_carry"),
        ("撤退", "retreat"),
        ("拉扯", "kite"),
        ("分散", "split_fight"),
    ],
}


def _match_action(sentence: str, fact: str) -> List[str]:
    """返回句中命中的动作取值编码（保持词表顺序，结果去重）。

    攻略原文用的是片段词（「存钱」），这里先按片段词表命中，
    再映射到规范取值编码。
    """
    hits: List[str] = []
    lowered = sentence.lower()
    for alias, code in ACTION_ALIAS_TO_VALUE.get(fact, []):
        if alias in lowered and code not in hits:
            hits.append(code)
    return hits


def _extract_stage(sentence: str) -> Optional[Condition]:
    """抽取「4-1」「第 4 阶段」「4 阶段」这类阶段标记。

    「N 阶段」这一形式很常见（「2 阶段金币超过 50」），漏掉它会让那个
    阶段编号被当成某个事实的阈值。
    """
    m = re.search(r"(\d+)\s*[-–]\s*(\d+)", sentence)
    if m:
        stage, phase = int(m.group(1)), int(m.group(2))
        # 4-1 是复合标记，用 stage 等值 + phase 表达
        return Condition("stage", "==", stage)

    m = re.search(r"(?:第\s*)?(\d+)\s*阶段", sentence)
    if m:
        return Condition("stage", "==", int(m.group(1)))
    return None


#: 阶段编号所在的位置。数值抽取必须避开这里，否则「2 阶段」里的 2
#: 会被误当成某个事实的阈值。
_STAGE_NUM_RE = re.compile(r"(?:第\s*)?\d+\s*[-–]\s*\d+|第?\s*\d+\s*阶段")


def _strip_stage_numbers(sentence: str) -> str:
    """把阶段标记替换成占位符，避免其中的数字被当成阈值。"""
    return _STAGE_NUM_RE.sub("【阶段】", sentence)


def _extract_numeric_condition(
    sentence: str, known_facts: set
) -> List[Tuple[Condition, str]]:
    """从句子中抽取「事实 + 比较」条件。

    只认与事实名紧邻的比较词 —— 详见 :func:`_find_comparison_after`
    的说明。宁可漏，不要错。
    """
    results: List[Tuple[Condition, str]] = []

    for fact, aliases in NUMERIC_FACT_ALIASES.items():
        if fact not in known_facts:
            continue
        # 在去掉阶段编号的文本上搜索：「2 阶段金币超过 50」里的 2 是阶段，
        # 不是金币阈值，放在一起就会被吸走
        stripped = _strip_stage_numbers(sentence)
        if not any(alias in stripped for alias in aliases):
            continue

        for alias in aliases:
            if alias not in stripped:
                continue
            cmp = _find_comparison_after(stripped, alias)
            if cmp:
                op, value = cmp
                results.append((Condition(fact, op, value), alias))
                break

    return results


def _detect_subjective(sentence: str) -> List[str]:
    return [m for m in DEFAULT_SUBJECTIVE_MARKERS if m in sentence]


def extract_candidates(
    text: str,
    game: str,
    version: str,
    known_facts: set,
    guide_title: str = "",
    guide_url: str = "",
    min_confidence: float = 0.4,
) -> List[RuleCandidate]:
    """从攻略文本抽取规则候选。

    返回的候选已按置信度降序排列。低于 ``min_confidence`` 的直接丢弃 ——
    连人工确认都不值得的条目，宁可不要。
    """
    candidates: List[RuleCandidate] = []

    for idx, sentence in enumerate(split_sentences(text)):
        cand = _extract_one(
            sentence, idx, game, version, known_facts, guide_title, guide_url
        )
        if cand is not None and cand.confidence >= min_confidence:
            candidates.append(cand)

    candidates.sort(key=lambda c: -c.confidence)
    return candidates


def _extract_one(
    sentence: str,
    idx: int,
    game: str,
    version: str,
    known_facts: set,
    guide_title: str,
    guide_url: str,
) -> Optional[RuleCandidate]:
    markers = _detect_subjective(sentence)
    cand = RuleCandidate(
        title=sentence.rstrip("。！？；.!?; "),
        game=game,
        version=version,
        source=Source(
            guide=guide_title or "未标注来源",
            locator=f"第 {idx + 1} 句",
            url=guide_url,
            quoted=sentence.strip(),
        ),
        subjective_markers=markers,
    )

    # --- 触发条件 ---
    conds: List[Condition] = []
    tags: List[str] = []

    stage_cond = _extract_stage(sentence)
    if stage_cond and "stage" in known_facts:
        conds.append(stage_cond)
        tags.append(f"{stage_cond.value}阶段")

    numeric = _extract_numeric_condition(sentence, known_facts)
    for cond, alias in numeric:
        conds.append(cond)
        tags.append(alias)

    # 谓词式条件：「血量健康」「局势不利」
    if "pressure" in known_facts:
        if re.search(r"(血量|生命值)?\s*(健康|充裕|充足)", sentence):
            conds.append(Condition("pressure", "in", ["low", "mid"]))
            tags.append("局面健康")
        if re.search(r"(局势|局面|血量)?\s*(危急|告急|紧张|不足|残血)", sentence):
            conds.append(Condition("pressure", "in", ["high"]))
            tags.append("局面危急")

    # --- 期望动作 ---
    expect_fact = ""
    expect_values: List[str] = []
    expect_op = "in"
    for fact in ACTION_ALIAS_TO_VALUE:
        if fact not in known_facts:
            continue
        hits = _match_action(sentence, fact)
        if hits:
            expect_fact = fact
            expect_values = hits
            break

    if not expect_fact:
        cand.notes.append("未识别出可判定的动作，需人工补全 expect 字段")
        return cand

    cand.trigger = Trigger(all=conds)
    cand.expect_fact = expect_fact
    cand.expect_op = expect_op
    cand.expect_value = expect_values
    cand.tags = tags

    # --- 置信度 ---
    # 数值阈值是判定的骨架；动作明确匹配次之；只有阶段标记最弱。
    score = 0.0
    if numeric:
        score += 0.5
    if stage_cond:
        score += 0.2
    if any(c.fact == "pressure" for c in conds):
        score += 0.1
    score += 0.3

    if markers:
        # 命中主观词直接压到不可用，避免靠其他加分项蒙混过关
        cand.notes.append(
            f"含主观判断词 {'/'.join(markers)}，需改写为可判定表述后才可入库"
        )
        score = min(score, 0.35)

    if not conds:
        cand.notes.append("未识别出触发条件，无法定位到具体回合")
        score = min(score, 0.3)

    if len(expect_values) > 1:
        cand.notes.append(
            f"识别到多个候选动作 {expect_values}，需人工确认应保留哪些"
        )
        score -= 0.1

    cand.confidence = max(0.0, min(1.0, score))
    return cand


_FACT_ALIASES: Dict[str, List[str]] = {}


def _build_alias_index() -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {}
    for table in (NUMERIC_FACT_ALIASES, ENUM_FACT_ALIASES, ACTION_FACT_ALIASES):
        for fact, aliases in table.items():
            merged.setdefault(fact, []).extend(aliases)
    return merged


_FACT_ALIASES = _build_alias_index()
