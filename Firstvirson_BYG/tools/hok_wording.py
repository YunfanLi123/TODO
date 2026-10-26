"""文案规范化与体检。

模式库里的每一句都要**说得清楚**，这是四个硬要求：

1. **时间口径统一** —— 窗口是单点就写那个时点，窗口是范围就写范围。
   「时间区间一定是范围，时点一定是准确的时刻」：
   一条规则覆盖第 2 到第 3 分钟，标题就不能写「两分钟」（那是个点，
   看起来像只管第 2 分钟）。
2. **状态记号去掉** —— 攻略里的「状态【反向开局完成】」是它内部记状态用的
   写法，读者看到「状态【…】」只知道有个状态，不知道是什么。
   留下内容、去掉记号：「反向开局完成」。
3. **模糊词不许进库** —— 「尽量」「必要时」「可能」「正常」「稳扎稳打」
   这类词一进来，结论就落不了地：玩家不知道自己该做什么，
   系统也没法说清「做对」是什么。
4. **一句话只留一个时刻** —— 攻略正文里常带裸时间戳（「1:00 与中单、游走
   控下中路河道之灵」）。标题已经写明这条规则管哪一段时间，正文再冒一个
   时刻，读者会以为规则管错了时间。所以：
   - 当它是**条件**（「1:00 河道之灵刷新」）→ 补一个「时」字，
     读作「1:00 时河道之灵刷新」；
   - 当它是**动作的时刻**（「1:00 与中单、游走控下中路河道之灵」）→ 去掉，
     时间由标题承担（「这个时间段该做的是：与中单、游走控下中路河道之灵」）。

本模块提供四样东西：
- :func:`normalize` —— 去状态记号、把「+」改成顿号、把裸时刻说成时刻；
- :func:`apply_fixes` —— 定点替换剩下的模糊表述；
- :func:`lint` —— 体检，列出命中的问题（生成器与测试都用它）；
- :func:`window_label` / :func:`retime_title` —— 时间口径。
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

#: 状态记号：「状态【反向开局完成】」与「【反向开局完成】」。
_STATE_RE = re.compile(r"状态\s*【([^】]*)】|【([^】]*)】")

#: 条件之间的「+」：攻略用它并列同一句里的多个条件。
_PLUS_RE = re.compile(r"\s*\+\s*")

#: 裸时刻：``1:00``。
_CLOCK = r"\d{1,2}:\d{2}"

#: 时刻区间：「1:30 至 2:00」。两端都是时刻，但整体是**一段**。
_RANGE_RE = re.compile(rf"({_CLOCK})\s*至\s*({_CLOCK})")

#: 时刻后面跟这些字时，它不是「在……时」，而是别的东西：
#: 「2:00 至 2:30」是区间，「1:50 前到 4 级」是截止时间。
_NOT_A_MOMENT_AFTER = set("至到前后之内起间左右多少时已未之每约超超")

#: 时刻前面跟这些字时同理（「至 2:00」是区间末端，「等 20:00 风暴龙王」
#: 是「等一个东西」，补「时」反而别扭）。
_NOT_A_MOMENT_BEFORE = set("至等")


def normalize(text: str, strip_clause_time: bool = False) -> str:
    """去掉状态记号，把「+」改成顿号，把裸时刻说成时刻。

    「【反向开局完成】+ 已到 4 级 + 敌方对抗路过线过深」
    → 「反向开局完成、已到 4 级、敌方对抗路过线过深」

    ``strip_clause_time=True`` 时，额外去掉**子句开头**的时刻：
    那是动作发生的时刻，而标题已经写明这条规则管哪段时间，
    正文再带一个时刻，读者会以为规则管错了时间。
    「1:00 与中单、游走控下中路河道之灵」→「与中单、游走控下中路河道之灵」
    """
    if not text:
        return ""

    def _mark(m: "re.Match[str]") -> str:
        inner = m.group(1) or m.group(2) or ""
        return inner.strip()

    out = _STATE_RE.sub(_mark, text)
    out = _PLUS_RE.sub("、", out)
    # 先去子句开头的时刻，再给剩下的时刻补「时」——
    # 顺序反过来的话，去完会剩一个孤零零的「时」。
    if strip_clause_time:
        out = _strip_clause_moments(out)
    out = _mark_moments(out)
    return out.strip()


def _protected_spans(text: str) -> List[Tuple[int, int]]:
    """时刻区间里两端的时刻不能当「时刻」处理。"""
    spans: List[Tuple[int, int]] = []
    for m in _RANGE_RE.finditer(text):
        spans.append((m.start(1), m.end(1)))
        spans.append((m.start(2), m.end(2)))
    return spans


def _mark_moments(text: str) -> str:
    """裸时刻补一个「时」字：「1:00 河道之灵刷新」→「1:00 时河道之灵刷新」。

    不动的三种：
    - 区间两端（「1:30 至 2:00」）—— 那是一段，不是时刻；
    - 后面跟着「至 / 前 / 后…」的（「1:50 前到 4 级」）—— 那是截止时间；
    - 区间末端（「至 2:00」）—— 由区间规则保护。
    """
    if not text:
        return text

    protected = _protected_spans(text)
    out = []
    last = 0
    for m in re.finditer(_CLOCK, text):
        if any(s <= m.start() < e for s, e in protected):
            continue
        end = m.end()
        after = text[end:]
        stripped = after.lstrip()
        nxt = stripped[:1]
        if nxt and nxt in _NOT_A_MOMENT_AFTER:
            continue
        before = text[:m.start()].rstrip()
        if before[-1:] in _NOT_A_MOMENT_BEFORE:
            continue
        out.append(text[last:end])
        # 有空格就把空格换成「时」，没有就直接接「时」。
        spaces = len(after) - len(stripped)
        if spaces:
            out.append(" 时")
            last = end + spaces
        else:
            out.append("时")
            last = end
    out.append(text[last:])
    return "".join(out)


def _strip_clause_moments(text: str) -> str:
    """去掉子句开头的裸时刻。

    只在**子句开头**去掉（分句首、冒号后、顿号后）：
    「1:00 与中单、游走控下中路河道之灵」→「与中单、游走控下中路河道之灵」。

    不动的：
    - 「1:30 至 2:00 用这个窗口抓人」—— 区间，去了会留下「至 2:00 …」；
    - 「1:50 前到 4 级」—— 截止时间。
    """
    if not text:
        return text

    out = []
    pos = 0
    pattern = re.compile(rf"(?:^|(?<=[；：，、]))\s*({_CLOCK})(?:\s*时)?\s+")
    for m in pattern.finditer(text):
        # 区间起点不动：「1:30 至 2:00 …」
        if text[m.end():].lstrip().startswith("至"):
            continue
        out.append(text[pos:m.start()])
        pos = m.end()
    out.append(text[pos:])
    return "".join(out)


# --------------------------------------------------------------------------
# 模糊词与空话
# --------------------------------------------------------------------------

#: 判不了、说不准的词。进了库，结论就落不了地。
VAGUE_WORDS: Tuple[str, ...] = (
    "尽量", "左右", "大概", "可能", "之类", "等等", "某些", "有时", "适当",
    "必要时", "看情况", "看局势", "酌情", "灵活", "感觉", "一般", "通常",
    "正常", "合理", "合适", "更好", "不少", "很多", "几乎", "差不多",
    "视情况", "自行判断", "大概齐", "或许", "也许", "说不定", "若干",
    "基本", "主要", "常常", "经常", "有些", "多数", "普遍", "原则上",
    "根据情况", "看情况办", "有的时候", "多少", "争取", "试试", "随便",
)

#: 只说了「要做什么类别」，没说「怎么做」。
EMPTY_TALK: Tuple[str, ...] = (
    "正常发育", "稳扎稳打", "见机行事", "随机应变", "灵活处理",
)


#: 定点替换：把剩下的模糊表述换成**具体**的说法。
#:
#: 只改确实读不懂的句子；每个替换都要能说出「原来那句话哪里含糊」。
#: 顺序有意义：长句先替换，避免短句把长句拦腰截断。
WORDING_FIXES: Dict[str, str] = {
    # 「正常清线」—— 正常是拿什么做标准？读者不知道什么叫正常。
    "正常清线": "照常清线",
    # 「必要时交控制」—— 什么时候是必要？没说。
    "必要时交控制": "对方拿到 BUFF 仍不退时才交控制",
    "必要时开三技能霸体": "被贴脸顶住时才开三技能霸体",
    # 「可能有埋伏」—— 说了等于没说：那就等于不敢进草。
    # 攻略的条件是「对方一级埋伏在我方兵线侧草丛」，把「可能」去掉，
    # 写法上就成了一句可执行的条件句。
    "对方一级可能埋伏在我方兵线侧草丛": "对方一级在我方兵线侧草丛时",
    "对方一级可能蹲在我方兵线侧草丛": "对方一级在我方兵线侧草丛时",
    # 「兵线很多」—— 多少才算多？
    "对方塔前兵线很多": "对方塔前有一大波兵线压到塔下",
    # 「稳扎稳打」—— 怎么打？
    "条件不齐就跟前排稳扎稳打": "条件不齐就跟着前排推进，不单独开团",
    # 孤立的「稳扎稳打」：只说了心态，没说动作。
    "稳扎稳打": "跟着前排推进，不单独开团",
    # 「正常发育」—— 发育成什么样？
    "正常发育": "按节奏补兵",
    # 「对方无法正常拆塔」—— 怎么就算正常拆塔？
    "对方无法正常拆塔": "对方拆不了塔",
    # ---- 英文词换成中文 --------------------------------------------
    #
    # 保留 BUFF、C位：那是游戏自己就在用的说法（王者界面里写的就是
    # 「红BUFF」「核心C位」），换成别的反而不像行话。
    # 但 gank / counter / ban / pick / CD / 平A / 5v5 这些是纯英文，
    # 报告里出现就是「文案的英文标签」，一律换成中文。
    "gank ": "抓",
    "终极 counter": "终极针对",
    "留一个 ban 位": "留一个禁人位",
    "时 ban 掉": "时禁掉",
    "露出两个 pick": "露出两个选择",
    "已暴露两个 pick": "已暴露两个选择",
    "三个 pick": "三个选择",
    "保留技能 CD": "保留技能冷却",
    "技能 CD": "技能冷却",
    "平A": "普攻",
    "不进 5v5": "不进 5 打 5",
    # 增益：游戏内叫「红BUFF」「蓝BUFF」，讲「拿到什么」时用「增益」更准
    "拿到 buff": "拿到增益",
    "无龙 buff": "无龙增益",
}


def apply_fixes(text: str) -> str:
    """把模糊表述换成具体说法。

    顺带把**已移除的机制**换成当前版本真实存在的机制
    （见 :mod:`tools.hok_mechanics`）—— 攻略原文可能记的是旧版本
    （实测 S45 已经没有「河道之灵」了），照抄进规则就会判一件
    游戏里不存在的东西，玩家永远做不到，报告只会冤枉人。
    """
    if not text:
        return ""
    out = text
    for wrong, right in WORDING_FIXES.items():
        if wrong in out:
            out = out.replace(wrong, right)
    from tools import hok_mechanics

    return hok_mechanics.apply(out)


def lint(text: str) -> List[str]:
    """体检：返回这段文字里命中的问题（模糊词 / 空话 / 残留状态记号）。"""
    problems: List[str] = []
    if not text:
        return problems
    for word in VAGUE_WORDS:
        if word in text:
            problems.append(f"模糊词「{word}」")
            break
    for word in EMPTY_TALK:
        if word in text:
            problems.append(f"空话「{word}」")
            break
    if _STATE_RE.search(text):
        problems.append("残留状态记号【】")
    return problems


# --------------------------------------------------------------------------
# 时间口径
# --------------------------------------------------------------------------

#: 分路名：标题的前缀。
LANE_NAMES: Tuple[str, ...] = (
    "打野", "中路", "对抗路", "发育路", "游走",
)

#: 阶段词：本身指一段（开局 / 前期 / 中期 / 后期）。
PHASE_WORDS: Tuple[str, ...] = ("开局", "前期", "中期", "后期", "决胜期")

#: 标题里的时间表达：「第 1 分钟」「两分钟」「6 ~ 9 分钟」。
_TIME_EXPR_RE = re.compile(
    r"^第?\s*(?:[0-9零一二三四五六七八九十两]+\s*(?:[-~]\s*[0-9零一二三四五六七八九十两]+\s*)?)?分钟"
)

#: 标题前缀：分路名 + 可能跟着阶段词 / 时间表达。
_TITLE_HEAD_RE = re.compile(
    rf"^(?P<lane>{'|'.join(LANE_NAMES)})(?P<rest>.*)$"
)


def window_label(lo: int, hi: int) -> str:
    """窗口 -> 时间标签。

    - 单点窗口 → 那个时点（「4:00」）
    - 一直到 60 分钟 → 「12:00 之后」
    - 其余 → 范围（「2:00 ~ 4:00」）

    口径：**区间就是范围，时点就是准确时刻**。
    写「2:00」而规则管的是第 2 到第 3 分钟，读者会以为只管第 2 分钟。
    """
    if lo == hi:
        return f"{lo}:00"
    if hi >= 60:
        return f"{lo}:00 之后"
    return f"{lo}:00 ~ {hi + 1}:00"


def retime_title(title: str, lo: int, hi: int) -> str:
    """把标题里的时间词按**规则实际覆盖的窗口**重写。

    「打野两分钟：…」在窗口是第 2 到第 3 分钟时要改成
    「打野 2:00 ~ 4:00：…」—— 写点会让人以为只管第 2 分钟。

    阶段词（「开局」「后期」）也一并换掉：它说的是一段，
    但没说清是哪一段，而窗口知道答案。
    """
    if not title:
        return title
    head, sep, rest = title.partition("：")
    if not sep:
        return title
    m = _TITLE_HEAD_RE.match(head.strip())
    if not m:
        return title

    lane = m.group("lane")
    tail = m.group("rest").strip()

    # 剥掉阶段词（「开局」「后期」）。
    for word in PHASE_WORDS:
        if tail.startswith(word):
            tail = tail[len(word):].strip()
            break
    # 剥掉时间表达（「两分钟」「第 1 分钟」「6 ~ 9 分钟」）。
    tm = _TIME_EXPR_RE.match(tail)
    if tm:
        tail = tail[tm.end():].strip()

    label = window_label(lo, hi)
    new_head = f"{lane} {label}"
    if tail:
        new_head += f" {tail}"
    return f"{new_head}：{rest}"
