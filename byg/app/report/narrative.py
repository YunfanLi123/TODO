"""报告正文交给**文字模型**来写 —— 眼睛读画面，脑子写人话。

为什么要这一层
--------------
报告正文里最该是「人话」的两段 —— 「原视频操作的文字描述」与
「建议操作描述」 —— 过去是模板拼出来的：

- **我方动作**：``这一段做的是`` + 动作标签反查
  （「这一段做的是转线支援、回撤拉扯。」）
- **建议动作**：**逐字照搬**模式库条目（``rule.deviation_hint``）

模板能保证「句式统一、不出错」，但它**不知道这一局发生了什么**：
同一段文字贴在十局不同的对局上都不违和，因为它根本不在描述这一局。
玩家要的是「0 到 2 分钟你在发育路跟射手换血，却没顺势把兵线推进去」，
模板给不出这句话 —— 它只会说「这一段做的是换血拉扯」。

于是有了这一层：**把已经采到的事实交给文字模型，让它写成一段话**。

它不负责判断对错
----------------
判断早已做完（规则引擎 + 画面裁决，见 :mod:`app.collection.vlm_review`）。
这一层只做**表达**：把「这一局的素材」翻译成「玩家读得懂的一段话」。

三道接地闸
----------
模型会编。自由发挥的代价是「报告说得头头是道，说的却不是这局」——
那比模板更坏。所以模型写出来的东西必须过闸：

1. **不许出现素材里没有的英雄名** ——「敌方打野」可以写成「敌方打野」，
   不能写成「敌方孙悟空」（名册里有、素材里没有 = 编的）。
2. **不许出现素材里没有的数字** ——「33 秒进野区」里的 33 必须来自攻略原文。
3. **建议必须仍然是攻略那一条** —— 与攻略原文的字面重合度不够就退回。

另外还禁**评价词**与**命令口吻**：描述段只描述，理由与建议不许说教。

不合规 / 调用失败 → **回落模板**，报告照出，不阻塞。
回落原因要能说清（``NarrativeResult.note``），这样「为什么这段还是模板腔」
有答案，而不是默默换了个腔调。

缓存
----
叙述写一次就够：报告页每次打开都会**重建**报告（规则库可能已更新），
每次都调模型既慢又不可复现。所以叙述按「素材指纹」缓存进对局数据里，
素材没变就直接复用。

借鉴
----
设计思路参考 ``docs/research/llm_narrative_references.md`` 里调研的几个
开源项目（结构化输出 + 程序化校验 + 失败回退），以及项目内既有的
:mod:`app.collection.vlm_review` 四道闸与
:mod:`app.collection.state_describer` 的「状态→人话」做法。
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

#: 「脑子」那个槽位。眼睛（vlm）读画面，脑子（text）写人话。
SLOT_TEXT = "text"

#: 看画面那个槽位 —— **文字槽位不可用时的退路**。
#:
#: 文字槽位最常见的坏法是「一把 Key 配了两个供应方」
#: （见 :func:`app.collection.settings.effective_slot`），
#: 于是文字槽位拿不到 Key。但观察者模型**也能写字** ——
#: 与其让报告退回模板，不如用眼睛那个模型来写。
SLOT_VLM = "vlm"

#: 描述段禁用的词 —— **评价玩家**的说法。
#:
#: 「这一段做的是……」是描述，不是判决；对错已经在「建议理由」那一栏
#: 说清楚了，描述里再说一遍「你失误了」只会变成说教。
#:
#: 素材里本来就有的词不算违规（模型照抄攻略原文里的「失误」是允许的）——
#: 禁的是**模型自己加上去的评价**。
#:
#: ⚠️ 词表里**不能**放游戏术语。「坑」就曾经被误判过一次：
#: 「龙坑」「暴君坑」是这游戏的核心名词，模型写「龙坑最后一刻交惩」
#: 被当成评价玩家的话挡了回去 —— 闸太严会把好文本白白丢掉，
#: 那等于整个功能白做。宁可少禁一个词，也不要误伤术语。
_FORBIDDEN_IN_ACTUAL = (
    "失误", "错误", "不专业", "离谱", "丢人", "白给",
    "你应该", "你必须", "无脑", "愚蠢", "瞎打",
)

#: 建议段禁用的词 —— **命令玩家**的说法。
#:
#: 建议是参考不是命令，项目一直用「可以考虑」这种口吻。
#: 「失误」「失败」不在这里 —— 「避免失误」「等对方先失误」是攻略里
#: 正常的说法。
_FORBIDDEN_IN_ADVICE = (
    "你必须", "你应该", "必须立刻", "务必", "给我", "赶紧去", "快点去",
    "不专业", "离谱", "丢人", "白给", "无脑",
)


#: 长度上限（字符）。描述与建议各有各的合理长度。
MAX_ACTUAL_CHARS = 140
MAX_ADVICE_CHARS = 220
MAX_ACTUAL_SENTENCES = 4
MAX_ADVICE_SENTENCES = 5
#: 建议理由：它是对照式的一句话，比前两段短。
MAX_REASON_CHARS = 170
MAX_REASON_SENTENCES = 3

#: 接地阈值。
#:
#: ``precision`` = 输出里有多少内容能在素材里找到（按中文二元组）。
#: 它回答「模型是不是在编」。低于这个数就说明它说了素材里没有的东西。
#: ``recall``（对攻略原文）= 攻略那句话有多少被用上了，
#: 它回答「这段建议还是不是攻略那一条」。
MIN_PRECISION = 0.30
MIN_GUIDE_RECALL = 0.20
#: 理由段：它天然要同时说「攻略要什么」和「做了什么」，
#: 所以对素材的覆盖率要求比描述段略松一点。
MIN_REASON_PRECISION = 0.25

#: 一次报告最多写多少条叙述。
#:
#: 写叙述是**锦上添花**：写不动就退回模板，报告不能被它拖住。
#: 8 条足够覆盖一页报告，再多收益也很小、耗时却线性增长。
MAX_ENTRIES = 8

#: 单次调用超时（秒）。文字模型比看画面快得多。
DEFAULT_TIMEOUT = 45.0

#: **一次构建**里允许花在模型调用上的总秒数。
#:
#: 报告页是同步渲染的 —— 模型慢不能让页面一直转。
#: 预算用完就停，剩下的条目用模板写法（下次打开还会再试）。
TOTAL_BUDGET = 60.0

#: 写失败之后的**冷却时间**（秒）。
#:
#: 重建报告时若允许调模型，那么「模型没配 / 配额用尽」这种失败
#: 会在**每次打开页面**时重试一遍 —— 每一页都白等几十秒。
#: 冷却期内不再重试（那条退回模板），过了冷却再试。
#:
#: 这是**进程内**的账，不进对局数据：它说的是「现在别试了」，
#: 不是「这一条永远写不出来」。
FAILURE_COOLDOWN = 600.0

#: 失败冷却账：``键 -> 上次失败的时刻``。
_FAILED_AT: Dict[str, float] = {}

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_DIGIT_RE = re.compile(r"\d+(?:\.\d+)?")


# ---------------------------------------------------------------- 文本工具


def split_clauses(text: str) -> List[str]:
    """把一段话拆成一句一句 —— 报告里一行一句，读起来才不挤。

    条件式建议尤其需要：几句「什么情况下：做什么」排在一起，
    玩家要能一眼看出哪句对应哪个条件。
    """
    out: List[str] = []
    for part in re.split(r"(?<=。)", text or ""):
        part = part.strip()
        if part:
            out.append(part)
    return out


def _bigrams(text: str) -> Set[str]:
    """中文字符二元组。"""
    chars = [c for c in (text or "") if _CJK_RE.match(c)]
    if len(chars) < 2:
        return set(chars)
    return {chars[i] + chars[i + 1] for i in range(len(chars) - 1)}


def _digits(text: str) -> Set[str]:
    """文本里的数字（逐个记，不合并）。"""
    return set(_DIGIT_RE.findall(text or ""))


def _precision(out: str, material: str) -> float:
    """输出里有多少内容能在素材里找到。"""
    a, b = _bigrams(out), _bigrams(material)
    if not a:
        return 0.0
    return len(a & b) / len(a)


def _recall(out: str, material: str) -> float:
    """素材里有多少内容被输出用上了。"""
    a, b = _bigrams(out), _bigrams(material)
    if not b:
        return 1.0
    return len(a & b) / len(b)


def _sentences(text: str) -> List[str]:
    return [s for s in re.split(r"(?<=。)|(?<=！)|(?<=？)", text or "")
            if s.strip()]


def _ensure_period(text: str) -> str:
    text = (text or "").strip()
    if text and not text.endswith(("。", "！", "？", "；")):
        text += "。"
    return text


# ---------------------------------------------------------------- 证据卡


@dataclass
class NarrativeCard:
    """写一段叙述要用的**全部**素材 —— 只有已经采到的事实。

    这里没有「结论」。结论早就判完了（规则引擎 + 画面裁决），
    这一层只负责把素材翻译成话。放进卡里的每一样东西都必须有出处，
    闸就是拿卡里的内容去核对模型写出来的话。
    """

    entry_id: str = ""
    hero: str = ""
    lane_label: str = ""
    phase_label: str = ""
    time_range: str = ""

    # ---- 我方动作 ----
    #: 模型看画面时写的原话（「这一段做的是……」的素材）。
    did: str = ""
    #: 记录到的操作（人话标签）。
    actual_actions: List[str] = field(default_factory=list)
    #: 操作发生在哪里。
    actual_places: List[str] = field(default_factory=list)
    #: 画面证据（「几秒看到什么」）。
    evidence: List[str] = field(default_factory=list)
    #: 明确没做的动作类。
    absent: List[str] = field(default_factory=list)
    #: 这一段的**局面**（客观事实：「经济落后 1200」「大招未就绪」）。
    #:
    #: 为什么单独一栏：动作名只说了「做了什么」，说了不说明**当时怎么样**。
    #: 有了局面，叙述才可能写出「这一段你经济落后还硬接团」这种
    #: 具体的话，而不是把动作名念一遍。
    facts: List[str] = field(default_factory=list)

    # ---- 建议动作 ----
    #: 攻略要的是什么（分类说法）。
    expected: str = ""
    #: 这一支的前提。
    guide_premise: str = ""
    #: 这一支的做法（**攻略原文** —— 建议必须还是这一条）。
    guide_action: str = ""
    #: 照着做会怎样。
    guide_outcome: str = ""
    #: 玩家实际在做什么（归纳）。
    actual_doing: str = ""
    #: 模型写的理由。
    reason: str = ""
    #: 适用的是第几支。
    branch_no: int = 0

    # ---------------------------------------------------------------- 素材

    def context_material(self) -> str:
        """这一段的**定位信息**（时间区间 / 英雄 / 分路阶段）。

        它**也是素材** —— 提示词里给了这些，模型当然会用
        （写「第 8 分钟」不算编造数字）。
        不把它算进素材，闸就会把「用了时间区间里的数字」误判成「编造数字」，
        害得明明写得好的叙述被挡回去。
        """
        return " ".join(p for p in (
            self.time_range, self.hero, self.lane_label, self.phase_label,
        ) if p)

    def actual_core(self) -> str:
        """**这一局实际做的事**（不含定位信息）。

        定位信息（时间区间 / 分路 / 阶段）不算「实际做的事」——
        理由里提一句「发育路」不等于说到这一段做了什么，
        所以做对照锚的时候要用这一份，不要用 :meth:`actual_material`。
        """
        parts = [
            self.did,
            " ".join(self.actual_actions),
            " ".join(self.actual_places),
            " ".join(self.evidence),
            " ".join(self.absent),
            " ".join(self.facts),
            self.actual_doing,
        ]
        return "\n".join(p for p in parts if p)

    def actual_material(self) -> str:
        """我方动作这一段的**全部素材**（接地闸的比对基准）。"""
        core = self.actual_core()
        context = self.context_material()
        return "\n".join(p for p in (context, core) if p)

    def advice_material(self) -> str:
        """建议这一段的**全部素材**（接地闸的比对基准）。"""
        parts = [
            self.guide_premise,
            self.guide_action,
            self.expected,
            self.guide_outcome,
            self.actual_doing,
            self.reason,
            self.context_material(),
        ]
        return "\n".join(p for p in parts if p)

    def guide_material(self) -> str:
        """**攻略那一侧**的素材（不含玩家实际做的事）。

        理由段要「对照着说」，对照的两边一边是攻略、一边是这一局。
        这份素材就是「攻略那一边」——
        理由里至少得碰到它一点，才谈得上对照。
        """
        return "\n".join(p for p in (
            self.guide_premise,
            self.guide_action,
            self.expected,
            self.guide_outcome,
            self.reason,
        ) if p)


    def fingerprint(self) -> str:
        """素材指纹 —— 素材变了，缓存里的旧叙述就该作废。"""
        raw = json.dumps(
            [self.actual_material(), self.advice_material()],
            ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    def to_prompt(self) -> str:
        """给模型看的素材表。

        刻意写成**清单**，但**每行不写分类名** ——
        分类名统一写在「行序」那一行里。

        为什么不在每行前面写分类名
        --------------------------
        实测踩过：分类名写在行首，模型会把它当成**要填的格式**照抄出来 ——
        「记录到的操作：刷野补经济。归纳：刷野补经济。」、
        「前提：…做法：…要点：…结果：…」。
        那不是叙述，是把表格念了一遍 —— 跟模板腔一样死板。
        行首没有标签可抄，模型就只能用自己的话把素材组织成句子。

        分类名写在**系统提示词**里，不写在这里 ——
        写在这里它就是清单里的一行，模型会当成素材（实测回显成
        「操作行序是：分推牵制」）。
        """
        lines: List[str] = []
        where = self.context_material()
        if where:
            # 定位写成**一句话**，不是标签行 —— 模型引用「第 8 分钟」是好事，
            # 那正是「具体」的一部分。
            lines.append("这一段是：%s。" % where)
            lines.append("")

        actual_lines, _ = self._actual_lines()
        lines.append("【这一段实际发生了什么】")
        if actual_lines:
            lines.extend("- %s" % t for t in actual_lines)
        else:
            lines.append("- （这一段没有读到具体动作）")

        advice_lines, _ = self._advice_lines()
        lines.append("")
        lines.append("【攻略这一段要求怎么做】")
        if advice_lines:
            lines.extend("- %s" % t for t in advice_lines)
        else:
            lines.append("- （没有可参考的攻略做法）")
        return "\n".join(lines)

    # ---------------------------------------------------------------- 素材行

    @staticmethod
    def _dedupe(items: Sequence[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
        """``(分类, 内容)`` 列表 -> ``(内容列表, 分类列表)``，去掉重复的行。

        去重有两个层次，缺一不可：

        1. **整行相同** —— 动作标签与「归纳」常常是同一句话
           （「刷野补经济」既在操作里、也在归纳里），
           模型拿到两份一模一样的话，只能把这句话念两遍。
        2. **互相包含** —— 「归纳」往往是几个动作标签拼起来的那一句
           （「转线支援、回撤拉扯」），把它留在素材里，模型就会
           「转线支援；回撤拉扯；转线支援、回撤拉扯」地念两遍。
           反过来，「做法」常常把「前提」包在里面
           （「敌方刺客两名以上时：缩到塔下…」），此时**留下做法**、
           丢掉前提 —— 所以顺序很重要：**信息最多的排前面**。

        实测踩过：素材里重复一遍，输出里就重复一遍。
        """
        seen: Set[str] = set()
        texts: List[str] = []
        kinds: List[str] = []
        for kind, text in items:
            text = str(text or "").strip()
            if not text or text in seen:
                continue
            if any(text in kept for kept in texts):
                continue
            if any(kept in text for kept in texts):
                # 新的一行把前面某行整个包住了 —— 它没有新信息。
                continue
            seen.add(text)
            texts.append(text)
            kinds.append(kind)
        return texts, kinds

    def _actual_lines(self) -> Tuple[List[str], List[str]]:
        items: List[Tuple[str, str]] = []
        if self.did:
            items.append(("画面读到", self.did))
        for a in self.actual_actions:
            items.append(("操作", a))
        for p in self.actual_places:
            items.append(("位置", p))
        for e in self.evidence:
            items.append(("证据", e))
        for a in self.absent:
            items.append(("没做到", a))
        if self.facts:
            items.append(("局面", "；".join(self.facts)))
        if self.actual_doing:
            items.append(("归纳", self.actual_doing))
        return self._dedupe(items)

    def _advice_lines(self) -> Tuple[List[str], List[str]]:
        # **做法排第一** —— 它通常把前提包在里面，让它当代表，
        # 后面那条重复的前提就该被丢掉。
        items: List[Tuple[str, str]] = []
        if self.guide_action:
            items.append(("做法", self.guide_action))
        if self.guide_premise:
            items.append(("前提", self.guide_premise))
        if self.expected:
            items.append(("要点", self.expected))
        if self.guide_outcome:
            items.append(("结果", self.guide_outcome))
        if self.reason:
            items.append(("理由", self.reason))
        return self._dedupe(items)




# ---------------------------------------------------------------- 结果


@dataclass
class NarrativeResult:
    """一次叙述的结果。"""

    actual_text: str = ""
    advice_text: str = ""
    #: **建议理由**（第三段）。模型写的「差在哪一步」。
    reason_text: str = ""
    model: str = ""
    #: ``model`` = 用了模型写的；``template`` = 退回模板。
    source: str = "template"
    #: 回落 / 退回的原因（空 = 一切正常）。
    note: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.actual_text or self.advice_text or self.reason_text)

    @property
    def body_ok(self) -> bool:
        """正文（我方动作 + 建议）有没有换成模型写的。

        与 :attr:`ok` 分开是因为「只有理由换了模型写法」在报告页上
        看着还是「跟以前一样死板」 —— 那种情况**该说**，不该算成功。
        """
        return bool(self.actual_text or self.advice_text)


# ---------------------------------------------------------------- 接地闸


def _forbidden_hit(text: str, material: str,
                   words: Sequence[str]) -> List[str]:
    """输出里出现了「素材里没有的评价词 / 命令口吻」。

    素材里本来就有的词**不算违规** —— 模型照抄攻略原文里的说法是允许的，
    禁的是它自己加上去的评价。
    """
    hits: List[str] = []
    for word in words:
        if word in text and word not in material:
            hits.append(word)
    return hits


def _unknown_heroes(text: str, material: str, roster: Set[str],
                     hero: str) -> List[str]:
    """输出里出现了「名册里有、素材里没有」的英雄名 —— 那是编的。"""
    if not roster:
        return []
    bad: List[str] = []
    for name in roster:
        if len(name) < 2 or name == hero:
            continue
        if name in text and name not in material:
            bad.append(name)
    return sorted(bad)


def _unknown_numbers(text: str, material: str) -> List[str]:
    """输出里出现了素材里没有的数字。"""
    known = _digits(material)
    return sorted(d for d in _digits(text) if d not in known)


#: 「谁」的方向词。
#:
#: 为什么单列一道闸
#: --------------
#: 字符覆盖率查不出**方向说反**：素材写「己方红区」，模型写「敌方红区」——
#: 用词全在素材里，覆盖率满分，事实却整个反了。
#: 玩家看到「敌方红区」会以为说的是对面，实际那段是己方的。
_SELF_WORDS = ("己方", "我方")
_ENEMY_WORDS = ("敌方", "对方", "对面", "敌人")

#: 「方向词 + 名词」的组合。名词取两字，够分辨又不至于太死。
_DIR_RE = re.compile(r"(己方|我方|敌方|对方|对面|敌人)([\u4e00-\u9fff]{2})")


def _group_of(word: str) -> str:
    if word in _SELF_WORDS:
        return "self"
    if word in _ENEMY_WORDS:
        return "enemy"
    return ""


def _direction_flip(out: str, material: str) -> str:
    """输出把「己方 / 敌方」说反了吗？返回说明，空串 = 没反。

    查法：把素材里的「方向词 + 名词」记下来，看输出里**同一个名词**
    有没有挂到相反的方向词上。

    素材本身就两面都提过的（既说己方打野也说敌方打野）**不算反** ——
    那种情况下模型说到敌方打野完全正常，误判会让好文本被白白丢掉。
    """
    material_pairs = {(m.group(2), _group_of(m.group(1)))
                      for m in _DIR_RE.finditer(material)}
    flipped: List[str] = []
    for om in _DIR_RE.finditer(out):
        anchor = om.group(2)
        ogroup = _group_of(om.group(1))
        if not ogroup:
            continue
        own = [g for (a, g) in material_pairs if a == anchor]
        if not own or ogroup in own:
            # 素材里没这个名词，或者两面都提过 —— 不算反。
            continue
        flipped.append("%s%s" % (om.group(1), anchor))
    if flipped:
        return "把%s说反了" % "、".join(dict.fromkeys(flipped))
    return ""




def check_actual(text: str, card: NarrativeCard,
                 roster: Set[str] = frozenset()) -> str:
    """我方动作叙述的接地闸。返回**拒绝原因**，空串 = 通过。"""
    text = (text or "").strip()
    if not text:
        return "模型没有写出这一段"
    if len(text) > MAX_ACTUAL_CHARS:
        return "太长（%d 字，上限 %d）" % (len(text), MAX_ACTUAL_CHARS)
    if len(_sentences(text)) > MAX_ACTUAL_SENTENCES:
        return "句数太多（上限 %d 句）" % MAX_ACTUAL_SENTENCES
    material = card.actual_material()
    hits = _forbidden_hit(text, material, _FORBIDDEN_IN_ACTUAL)
    if hits:
        return "出现了评价词：%s" % "、".join(hits)
    unknown = _unknown_heroes(text, material, set(roster), card.hero)
    if unknown:
        return "出现了素材里没有的英雄：%s" % "、".join(unknown)
    numbers = _unknown_numbers(text, material)
    if numbers:
        return "出现了素材里没有的数字：%s" % "、".join(numbers)
    flip = _direction_flip(text, material)
    if flip:
        return flip
    if _precision(text, material) < MIN_PRECISION:
        return "说的东西在素材里找不到出处"
    return ""


def check_advice(text: str, card: NarrativeCard,
                 roster: Set[str] = frozenset()) -> str:
    """建议动作叙述的接地闸。返回**拒绝原因**，空串 = 通过。"""
    text = (text or "").strip()
    if not text:
        return "模型没有写出这一段"
    if len(text) > MAX_ADVICE_CHARS:
        return "太长（%d 字，上限 %d）" % (len(text), MAX_ADVICE_CHARS)
    if len(_sentences(text)) > MAX_ADVICE_SENTENCES:
        return "句数太多（上限 %d 句）" % MAX_ADVICE_SENTENCES
    material = card.advice_material()
    hits = _forbidden_hit(text, material, _FORBIDDEN_IN_ADVICE)
    if hits:
        return "出现了命令口吻：%s" % "、".join(hits)
    everything = material + "\n" + card.actual_material()
    unknown = _unknown_heroes(text, everything, set(roster), card.hero)
    if unknown:
        return "出现了素材里没有的英雄：%s" % "、".join(unknown)
    numbers = _unknown_numbers(text, everything)
    if numbers:
        return "出现了素材里没有的数字：%s" % "、".join(numbers)
    flip = _direction_flip(text, everything)
    if flip:
        return flip
    # 建议必须**还是攻略那一条** —— 这是这一栏最要紧的一条闸：
    # 模型很容易把「缩到塔下等对方头像露出」改写成「猥琐发育」，
    # 听着顺，但玩家照着做会打另一套打法。
    if card.guide_action and _recall(text, card.guide_action) < MIN_GUIDE_RECALL:
        return "和攻略原文对不上"
    return ""


def check_reason(text: str, card: NarrativeCard,
                 roster: Set[str] = frozenset()) -> str:
    """建议理由的接地闸。返回**拒绝原因**，空串 = 通过。

    理由这一段要**对照着说**：「攻略要什么」对「这一段做的是什么」。
    怎么验它真的在对照？—— 它必须说到**这一段实际做了什么**
    （与「我方动作」那段素材有实打实的重合），否则那就是一句
    正确的废话（「照这个打法能翻盘」），跟这一局没关系。
    """
    text = (text or "").strip()
    if not text:
        return "模型没有写出这一段"
    if len(text) > MAX_REASON_CHARS:
        return "太长（%d 字，上限 %d）" % (len(text), MAX_REASON_CHARS)
    if len(_sentences(text)) > MAX_REASON_SENTENCES:
        return "句数太多（上限 %d 句）" % MAX_REASON_SENTENCES
    material = card.actual_material()
    everything = material + "\n" + card.advice_material()
    hits = _forbidden_hit(text, everything, _FORBIDDEN_IN_ACTUAL)
    if hits:
        return "出现了评价词：%s" % "、".join(hits)
    unknown = _unknown_heroes(text, everything, set(roster), card.hero)
    if unknown:
        return "出现了素材里没有的英雄：%s" % "、".join(unknown)
    numbers = _unknown_numbers(text, everything)
    if numbers:
        return "出现了素材里没有的数字：%s" % "、".join(numbers)
    flip = _direction_flip(text, everything)
    if flip:
        return flip
    # **对照式的锚**：理由必须说到这一段实际做了什么。
    #
    # 锚必须是「**只有实际侧才有**」的词 ——
    # 「敌方」「对方」这种两边都有的词不算数：
    # 照抄攻略里那句「等对方打野位置露出来再动」就自带这两个词，
    # 拿它们当锚，等于没检查。
    guide = card.guide_material()
    actual_only = _bigrams(card.actual_core()) - _bigrams(guide)
    if len(_bigrams(text) & actual_only) < 2:
        return "没有对照这一段实际做的事"
    # 对照的**另一边**：还得碰到攻略那一边。
    # 只描述玩家做了什么、只字不提攻略要什么 —— 那是「描述」不是「理由」，
    # 跟上面那一栏重复了。门槛故意压得很低（一个二元组），
    # 宁可放过一句不够漂亮的理由，也不该把好文本整段丢掉。
    if guide and not (_bigrams(text) & _bigrams(guide)):
        return "没有对照攻略要怎么做"
    if _precision(text, everything) < MIN_REASON_PRECISION:
        return "说的东西在素材里找不到出处"
    return ""




# ---------------------------------------------------------------- 模型调用


_SYSTEM = """你是一位《王者荣耀》对局复盘作者。你拿到的是**这一局**已经记录下来的素材，
请把它写成玩家读得懂的三段话。

铁律：
1. 只写素材里有的东西。素材里没有的英雄名、事件、数字，一个都不要加。
2. 不评价对错，不说「失误」「你应该」这类词 —— 对错另有栏目在说，
   你只负责把事情说清楚、把做法讲具体。
3. 三段的分工：
   - 「actual」这一段**实际发生**的事；
   - 「advice」这一段**具体该怎么做**（必须是素材里那一条攻略做法，
     不要另起炉灶讲别的打法）；
   - 「reason」**差在哪一步** —— 把「攻略要什么」和「这一段做的是什么」
     **对照着说**，让玩家一眼看出自己差在哪，不是泛泛而谈。
     它要**点出这一段实际做的那件事**（用素材里的说法，不要换词），
     写成「攻略这一段要的是…，你做的是…」这种对照句最清楚。
     别写「照这个打法能翻盘」这种跟这一局无关的正确废话。
4. **写成通顺的话，不要照抄清单**。清单只是素材，不是句子模板 ——
   把它们组织成读得懂的一两句话，别写成「第 X 分钟，…记录到的操作是…」
   这种把清单念一遍的腔调。
5. **不要把清单里的分类名写进正文**（那些分类名只标给你看素材的种类：
   实际那一段依次是「当时的局面 / 画面读到的原话 / 记录到的操作 /
   发生的位置 / 画面证据 / 没做到的动作 / 归纳」，
   攻略那一段依次是「具体做法 / 这一支的前提 / 要点 / 结果 / 理由」）。
6. 直接输出 JSON，不要解释、不要 markdown 围栏。

输出格式：
{"actual": "……", "advice": "……", "reason": "……"}
"""


_REPAIR_SYSTEM = """你刚才写的那段复盘文字没有通过校验，被退回的原因是：

{reason}

请**只针对这一条原因**重写一遍，其余内容保持原样。
规矩与刚才一样：只写素材里有的东西，不评价对错，输出 JSON。

输出格式：
{"actual": "……", "advice": "……", "reason": "……"}
"""


def _ask(prompt: str, system: str, creds, model_name: str, slot: str,
         timeout: float, retry: int) -> Tuple[Optional[Dict[str, Any]], str]:
    """问模型一次。返回 ``(数据, 失败原因)``；成功时原因是空串。

    失败**不抛** —— 抛出来要一路传到报告层，而叙述写不出来
    **不是**报告的错。但原因必须留住：报告要能说清「为什么没写成」，
    「配额已用尽」和「回答不是 JSON」对用户是两件事。
    """
    from ..collection import vlm_client

    try:
        resp = vlm_client.call_model(
            model_name,
            [vlm_client.text_block(prompt)],
            creds=creds, slot=slot, system=system,
            max_tokens=800, timeout=timeout, retry=retry)
    except vlm_client.VlmError as exc:
        return None, vlm_client.explain_error(exc)
    except Exception as exc:                              # noqa: BLE001
        return None, "调用文字模型失败：%s" % exc

    try:
        data = resp.json()
    except vlm_client.VlmError as exc:
        return None, vlm_client.explain_error(exc)
    except Exception as exc:                              # noqa: BLE001
        return None, "模型回答不是预期的 JSON：%s" % exc
    if not isinstance(data, dict):
        return None, "模型回答不是一个对象"
    return data, ""



def write_card(card: NarrativeCard, *, slot: str = SLOT_TEXT,
               model: str = "", timeout: float = DEFAULT_TIMEOUT,
               retry: int = 1, roster: Optional[Set[str]] = None,
               repair: bool = True) -> NarrativeResult:
    """让文字模型把这段素材写成三段话（我方动作 / 建议 / 建议理由）。

    失败（没配模型 / 调用不通 / 回答不是 JSON）→ 返回 ``source='template'``
    并带上原因，调用方据此退回模板。**这一层永远不抛异常** ——
    叙述写不出来不该让整份报告出不来。

    **逐段退回**
    ------------
    三段是**各自独立**判定的：哪一段过不了闸，就只有那一段退回模板，
    另外两段照样用模型写的。不因为「理由写砸了」就把正文也丢掉 ——
    模型在这三件事上的水平并不一样。

    **修问一次**
    ------------
    闸不过就直接退回模板，等于白扔一次模型调用 ——
    而模型犯的错大多是一句话就能改的（多写了个英雄、把己方说成敌方）。
    所以闸不过时把**具体的违规理由**回喂给模型，让它改一遍再验。
    借鉴自 ``guardrails-ai/guardrails`` 的 ``on_fail=reask`` 与
    ``stanfordnlp/dspy`` 的 ``Refine``：不合规不只有「丢弃」一种处理。
    再不过就退回模板 —— 修问**只做一次**，不让模型无限自我修正。
    """
    from ..collection import vlm_client

    result = NarrativeResult()
    creds = vlm_client.find_credentials(slot)
    if creds is None and slot != SLOT_VLM:
        # 文字槽位不可用（最常见：一把 Key 配了两个供应方）→
        # 退到看画面那个模型：它也能写字。
        creds = vlm_client.find_credentials(SLOT_VLM)
        if creds is not None:
            result.note = "文字槽位不可用，改用观察者模型来写"
    if creds is None:
        result.note = "没有可用的文字模型（文字槽位与观察者槽位都没配好）"
        return result

    model_name = model or creds.model
    prompt = card.to_prompt()
    data, why = _ask(prompt, _SYSTEM, creds, model_name, slot, timeout, retry)
    if data is None:
        result.note = why
        return result

    actual = str(data.get("actual") or "").strip()
    advice = str(data.get("advice") or "").strip()
    reason_text = str(data.get("reason") or "").strip()
    if not actual and not advice and not reason_text:
        result.note = "模型没有写出内容"
        return result

    names = set(roster or ())
    actual_why = check_actual(actual, card, names) if actual else "模型没有写出这一段"
    advice_why = check_advice(advice, card, names) if advice else "模型没有写出这一段"
    reason_why = check_reason(reason_text, card, names) if reason_text else "模型没有写出这一段"

    # **修问一次**：把违规的具体理由回喂给模型。
    #
    # 三段是**一次问出来**的，所以三段一起修 —— 修问只花一次调用，
    # 拆成三次修既慢又容易修到一半。
    if repair and (actual_why or advice_why or reason_why):
        reason = "；".join(w for w in (actual_why, advice_why, reason_why) if w)
        retry_prompt = (
            prompt + "\n\n【你上一版写的】\n"
            + json.dumps({"actual": actual, "advice": advice, "reason": reason_text},
                         ensure_ascii=False)
            + "\n\n【被退回的原因】\n" + reason)
        again, _ = _ask(retry_prompt, _REPAIR_SYSTEM.replace("{reason}", reason),
                        creds, model_name, slot, timeout, retry)
        if again is not None:
            new_actual = str(again.get("actual") or "").strip()
            new_advice = str(again.get("advice") or "").strip()
            new_reason = str(again.get("reason") or "").strip()
            if new_actual:
                new_why = check_actual(new_actual, card, names)
                if not new_why:
                    actual, actual_why = new_actual, ""
                else:
                    reason = "%s；%s" % (reason, new_why)
            if new_advice:
                new_why = check_advice(new_advice, card, names)
                if not new_why:
                    advice, advice_why = new_advice, ""
                else:
                    reason = "%s；%s" % (reason, new_why)
            if new_reason:
                new_why = check_reason(new_reason, card, names)
                if not new_why:
                    reason_text, reason_why = new_reason, ""
                else:
                    reason = "%s；%s" % (reason, new_why)

    if actual and not actual_why:
        result.actual_text = _ensure_period(actual)
    elif actual:
        result.note = "我方动作叙述退回模板：%s" % actual_why
    if advice and not advice_why:
        result.advice_text = _ensure_period(advice)
    elif advice:
        prefix = (result.note + "；") if result.note else ""
        result.note = prefix + "建议动作叙述退回模板：%s" % advice_why
    if reason_text and not reason_why:
        result.reason_text = _ensure_period(reason_text)
    elif reason_text:
        prefix = (result.note + "；") if result.note else ""
        result.note = prefix + "建议理由退回模板：%s" % reason_why

    if result.actual_text or result.advice_text or result.reason_text:
        result.source = "model"
        result.model = model_name
    return result


# ---------------------------------------------------------------- 名单


def hero_roster(game: str) -> Set[str]:
    """这个游戏里全部英雄的名字（接地闸用来抓「编出来的英雄」）。

    名单从**适配器所在模块**里取 —— 适配器自己不暴露名册，
    但模块里通常有一份（``HEROES_BY_LANE``）。取不到就返回空集，
    那一道闸自动跳过（少一道闸总比整份报告出不来好）。

    ⚠️ 这里踩过两次坑：
      1. 写成 ``from app.adapters import get_adapter`` —— 那个导入根本不存在，
         而异常被 ``except`` 吞掉，于是名册**永远是空的**，
         「编出来的英雄」那道闸**静悄悄地失效了**；
      2. 只 ``from ..adapters.base import get_adapter`` —— 适配器是
         **导入 ``app.adapters.registry`` 时才注册**的
         （``app/adapters/__init__.py`` 按分层约定是空的），
         单独导入 base 拿不到任何游戏，``get_adapter`` 抛 KeyError，
         同样被吞掉。

    这种「闸还在、但不干活」的降级最难发现，所以现在：
    导入路径必须是 ``..adapters.registry``，并加一条单测盯住
    「王者名册不为空」（见 ``tests/test_report_narrative.py``）。
    """
    try:
        from ..adapters.registry import get_adapter
        adapter = get_adapter(game)
    except Exception:                                  # noqa: BLE001
        return set()
    try:
        module = importlib.import_module(type(adapter).__module__)
    except Exception:                                  # noqa: BLE001
        return set()
    by_lane = getattr(module, "HEROES_BY_LANE", None)
    if not isinstance(by_lane, dict):
        return set()
    out: Set[str] = set()
    for heroes in by_lane.values():
        for name in heroes or ():
            name = str(name).strip()
            if name:
                out.add(name)
    return out



# ---------------------------------------------------------------- 叙述器


def enabled_by_default() -> bool:
    """叙述默认开着，但**跑测试时关掉** —— 测试不该去调真模型。

    ``MATCH_NARRATIVE`` 可以显式开/关：``on`` / ``off``。
    """
    flag = os.environ.get("MATCH_NARRATIVE", "").strip().lower()
    if flag in ("off", "0", "false", "no", "none"):
        return False
    if flag in ("on", "1", "true", "yes"):
        return True
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    return True


class Narrator:
    """把模型写的叙述装进报告条目。带缓存，失败退回模板。

    缓存的键 = ``mode_id|时间区间|标题``；值里带**素材指纹**，
    素材变了（规则库更新了、这一局重新分析过）旧叙述自动作废。
    """

    def __init__(self, cache: Optional[Dict[str, Any]] = None,
                 store=None, slot: str = SLOT_TEXT, model: str = "",
                 enabled: Optional[bool] = None,
                 allow_model: Optional[bool] = None,
                 max_entries: int = MAX_ENTRIES,
                 timeout: float = DEFAULT_TIMEOUT,
                 budget: float = TOTAL_BUDGET,
                 roster: Optional[Set[str]] = None):
        self.cache: Dict[str, Any] = dict(cache or {})
        #: 写回缓存的回调（由调用方决定存到哪儿）。
        self.store = store
        self.slot = slot
        self.model = model
        #: 整层是否工作。关掉 = 连缓存里的叙述都不装，条目全是模板写法。
        self.enabled = enabled_by_default() if enabled is None else bool(enabled)
        #: 是否允许**调模型**。
        #:
        #: 报告页每次打开都会用当前规则库重建报告 —— 重建时只该读缓存，
        #: 不该又去调一遍模型（又慢又不可复现）。所以这两个开关要分开。
        self.allow_model = (self.enabled if allow_model is None
                            else bool(allow_model))
        self.max_entries = max_entries
        self.timeout = timeout
        #: 一次构建里允许花在模型调用上的总秒数。
        self.budget = budget
        self.roster: Set[str] = set(roster or ())
        #: 本次调用了几次模型（诊断用）。
        self.calls = 0
        #: 本次有几条条目用上了模型写的叙述（诊断用）。
        self.written = 0
        #: 其中有几条的**正文**（我方动作 / 建议）也换成了模型写的。
        #:
        #: 「只有理由换了模型写法」在报告页上看着还是老样子 ——
        #: 报告要说这件事时看的是这个数，不是 :attr:`written`。
        self.written_body = 0
        #: 回落 / 退回的原因（诊断用，会冒泡到报告的说明位）。
        self.notes: List[str] = []
        self._spent = 0.0

    # ------------------------------------------------------------------

    @staticmethod
    def entry_key(entry: Dict[str, Any]) -> str:
        """报告条目的稳定键。"""
        return "|".join([
            str(entry.get("mode_id") or ""),
            str(entry.get("time_range") or ""),
            str(entry.get("title") or ""),
        ])

    def apply(self, entry: Dict[str, Any], card: NarrativeCard) -> None:
        """把叙述写进一条报告条目（原地改）。

        写不出来就**什么都不改** —— 条目里留着模板那套话，报告照出。
        """
        if not self.enabled:
            return
        key = self.entry_key(entry)
        fingerprint = card.fingerprint()
        cached = self.cache.get(key)
        if isinstance(cached, dict) and cached.get("fingerprint") == fingerprint:
            if (cached.get("actual_text") or cached.get("advice_text")
                    or cached.get("reason_text")):
                self.written += 1
            if cached.get("actual_text") or cached.get("advice_text"):
                self.written_body += 1
            _apply_cached(entry, cached)
            return
        if not self.allow_model:
            # 只读缓存（重建报告时就是这个模式）。
            return
        if self.calls >= self.max_entries:
            self.notes.append(
                "叙述条数超过上限（%d 条），后面的条目仍用模板写法" % self.max_entries)
            return
        if self._spent >= self.budget:
            self.notes.append(
                "写叙述用掉了 %d 秒（上限 %d 秒），后面的条目仍用模板写法"
                % (int(self._spent), int(self.budget)))
            return
        # **失败冷却**：刚试过就失败了的，别在冷却期内再试 ——
        # 否则「模型没配」这种失败会让**每一次打开报告**都白等几十秒。
        last = _FAILED_AT.get(key)
        if last is not None and (time.monotonic() - last) < FAILURE_COOLDOWN:
            self.notes.append("刚才写叙述失败过，先用模板写法")
            return
        started = time.monotonic()
        result = write_card(card, slot=self.slot, model=self.model,
                            timeout=self.timeout, roster=self.roster)
        self._spent += time.monotonic() - started
        self.calls += 1
        if result.note:
            self.notes.append(result.note)
        value = {
            "fingerprint": fingerprint,
            "model": result.model,
            "actual_text": result.actual_text,
            "advice_text": result.advice_text,
            "reason_text": result.reason_text,
        }
        if result.actual_text or result.advice_text or result.reason_text:
            self.cache[key] = value
            if self.store is not None:
                self.store(key, value)
            _FAILED_AT.pop(key, None)
        else:
            # 一次都没写成 → 记进冷却账，别在冷却期内反复试。
            _FAILED_AT[key] = time.monotonic()
        if result.actual_text or result.advice_text or result.reason_text:
            self.written += 1
        if result.body_ok:
            self.written_body += 1
        _apply_cached(entry, value)



def _apply_cached(entry: Dict[str, Any], value: Dict[str, Any]) -> None:
    """把一段叙述装进报告条目。空的那一半不动（留模板）。"""
    actual = str(value.get("actual_text") or "").strip()
    advice = str(value.get("advice_text") or "").strip()
    reason = str(value.get("reason_text") or "").strip()
    if actual:
        entry["actual_text"] = actual
        entry["narrative_source"] = "model"
    if advice:
        entry["advice_text"] = advice
        entry["advice_lines"] = split_clauses(advice)
        entry["narrative_source"] = "model"
    if reason:
        # 理由也换成模型写的 —— **三段同源**才不会读着像两篇东西拼的。
        entry["reason_text"] = reason
        entry["narrative_source"] = "model"
    if value.get("model"):
        entry["narrative_model"] = str(value["model"])
