# -*- coding: utf-8 -*-
"""分路裁决：**画面上多数时间在哪条路**，而不是模型随口说的那一个词。

为什么要有这一层
----------------
2026-11-07 实测（一局桑启）：模型看完整局，顶层 ``position`` 答了
``mid``（中路）—— 可它自己写的检索请求里全是「上路河道被包围」
「上路一塔附近」「上路野区」。**顶层那一个字段孤证独大，错了没人拦**，
于是：报告说「你走的是中路」、检索按中路筛攻略、桑启（游走位）的
攻略一条都进不来，覆盖表整列「—」。

所以现在的口径是**三处读数对账**：

======================  ==========================================
读数                    从哪来
======================  ==========================================
**画面逐段统计**        模型给每段标 ``lane``，按**众数**定
顶层那一个字段          模型看整局时说的 ``position``
英雄查表                按官方定位推断（摇摆位推不出，留空）
======================  ==========================================

**画面逐段统计优先** —— 它有几十条证据，顶层字段只有一条。
但只有证据**压倒性**时才用它（见 :func:`judge_lane` 的判定条件），
否则宁可保留顶层读数。

冲突必须留痕：说清「按哪条定的、别的为什么被否」。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

#: 模型逐段可以标的路。**别再编一套** —— 提示词里那份清单从这儿取。
LANE_VALUES: Tuple[str, ...] = (
    "上路", "中路", "下路", "发育路", "野区", "unknown",
)

#: 路 -> 规范分路码（规则库用的那套）。
#:
#: 上路和下路都归 ``side``（对抗路）—— 规则库按 ``side`` 筛，
#: 但**上路还是下路要单独记住**（玩家说的是「我走上路」，
#: 报告只写「对抗路」他会觉得没说清）。
LANE_TO_POSITION: Dict[str, str] = {
    "上路": "side",
    "下路": "side",
    "中路": "mid",
    "发育路": "carry",
    "野区": "jungle",
}

#: 分路码 -> 人话（留痕与报告里用）。
POSITION_LABELS: Dict[str, str] = {
    "side": "对抗路",
    "mid": "中路",
    "carry": "发育路",
    "jungle": "打野",
    "roam": "游走",
}

#: 规范分路码（规则库按它筛）。
POSITION_VALUES: Tuple[str, ...] = (
    "jungle", "mid", "side", "carry", "roam", "unknown",
)


def normalize_position(value: Any) -> str:
    """把模型说的分路归一成**规范码**。认不出就是 ``unknown``（不猜）。

    为什么顶层那一个字段也得归一
    ----------------------------
    提示词让模型写 ``jungle/mid/side/carry/roam``，但它经常写中文
    （「上路」「中路」「下路」）—— 逐段那个 ``lane`` 有 :func:`normalize_lane`
    兜着，顶层这个一直没有。

    **不归一的后果很安静**：分路成了「中路」而不是 ``mid``，
    于是每一条带分路条件的规则都被判成「不是给他写的」，
    整局一条结论都不出，而报告上**看不出哪里出了岔**
    （留痕只认 ``unknown``，「中路」看着像个正常的值）。

    归一顺序：规范码 → 逐段那套路名（上路/下路/野区…）→ 适配器那张别名表。
    """
    from ..adapters.honor_of_kings import normalize_position as _adapter_norm

    text = str(value or "").strip()
    if not text:
        return "unknown"
    if text in POSITION_VALUES:
        return text
    lane = normalize_lane(text)
    if lane in LANE_TO_POSITION:
        return LANE_TO_POSITION[lane]
    return _adapter_norm(text)


#: 至少要有这么多段画面指向同一条路，才算「证据够硬」。
MIN_SEGMENTS = 3

#: 指向同一条路的段数至少要占到这个比例，才压得过别的读数。
MIN_SHARE = 0.5


def normalize_lane(value: Any) -> str:
    """把模型给的 ``lane`` 归一。**认不出就是 unknown**（不猜）。

    归一得宽松一点：模型可能写「上路」「上路边」「上路一塔」，
    也可能写英文。按「含哪个词」判，不按全等。
    """
    text = str(value or "").strip()
    if not text:
        return "unknown"
    if text in LANE_VALUES:
        return text
    for lane in ("上路", "中路", "下路", "发育路", "野区"):
        if lane in text:
            return lane
    # 英文/别写法的兜底（模型偶尔会蹦英文）。
    low = text.lower()
    if "top" in low or "side" in low:
        return "上路"
    if "mid" in low or "middle" in low:
        return "中路"
    if "bottom" in low:
        return "下路"
    if "jungle" in low:
        return "野区"
    return "unknown"


@dataclass
class LaneJudgement:
    """分路裁决的结果。"""

    #: 规范分路码；定不出就是空串（**不是 unknown** —— unknown 是「模型说不知道」）。
    position: str = ""
    #: 哪条路（"上路" / "下路" …）；对抗路分不清上下就是空串。
    lane_detail: str = ""
    #: 这条分路是**怎么定的**（人话，进留痕）。
    source: str = ""
    #: 证据等级：``observed``（画面读出来的）/ ``inferred``（按英雄推的）/ ``declared``（玩家填的）。
    provenance: str = ""
    #: 各条路的段数（画面证据）。
    counts: Dict[str, int] = field(default_factory=dict)
    #: 顶层那一个字段说了什么（留着好对账）。
    watch_position: str = ""
    #: 跟别的读数对不上时的说明（人话）。
    conflict: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": self.position,
            "lane_detail": self.lane_detail,
            "source": self.source,
            "provenance": self.provenance,
            "counts": dict(self.counts),
            "watch_position": self.watch_position,
            "conflict": self.conflict,
        }


def count_lanes(events: Iterable[Any]) -> Dict[str, int]:
    """数一数各条路在画面里出现了多少段。

    只数**对局中**的段（选人/加载/结算那几段的路没有意义）。
    阶段全都没读到时（``unknown``）就都算数 —— 总比不数强。
    """
    counts: Dict[str, int] = {}
    for event in events or []:
        phase = str(getattr(event, "phase", "") or "unknown")
        lane = normalize_lane(getattr(event, "lane", ""))
        if lane == "unknown":
            continue
        if phase not in ("", "unknown", "in_game"):
            continue
        counts[lane] = counts.get(lane, 0) + 1
    return counts


def _lane_of_position(counts: Dict[str, int]) -> Tuple[str, str, int]:
    """按段数最多的路，挑出「哪条路」与它对应的分路码。"""
    best_lane = ""
    best_count = 0
    for lane, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if count > best_count:
            best_lane, best_count = lane, count
    if not best_lane:
        return "", "", 0
    return best_lane, LANE_TO_POSITION.get(best_lane, ""), best_count


def judge_lane(events: Sequence[Any],
               watch_position: str = "",
               hero_position: str = "",
               hero_name: str = "") -> LaneJudgement:
    """裁决这一局玩家走哪条路。

    判定顺序
    --------
    1. **画面逐段统计**（证据够硬时）；
    2. **顶层那一个字段**（模型看整局时说的）；
    3. **英雄查表**（按官方定位推断）。

    冲突时以证据硬的为准，但**必须留痕** —— 读者有权知道
    「按哪条定的、别的为什么被否」。

    **每一条冲突都要写，不能只写第一条**（2026-11-13）：原来是一串
    ``elif``，于是「看整局说的分路对不上」把「英雄的官方分路也对不上」
    盖住了 —— 而后者往往是**英雄认错**的信号（公孙离按发育路写，
    画面上却一直是中路），那正是读者最需要看见的一句。
    """
    counts = count_lanes(events)
    lane, position, count = _lane_of_position(counts)
    total = sum(counts.values())
    raw_watch = str(watch_position or "").strip()
    # **顶层那一个词也要归一** —— 模型写「中路」「上路」时，
    # 不归一就等于给每一条带分路条件的规则判「不是给他写的」。
    watch = normalize_position(raw_watch)
    hero = str(hero_position or "").strip()
    hero_label = str(hero_name or "").strip()

    out = LaneJudgement(counts=counts, watch_position=watch)

    def _hero_conflict(decided_lane: str) -> str:
        """英雄官方分路与画面不一致时那句话（一致就空串）。"""
        if not hero or hero == position or not decided_lane:
            return ""
        if hero_label:
            return ("按英雄「%s」的官方定位推的是「%s」，画面上多数时间在「%s」"
                    "—— 以画面为准（这个英雄读数可能认错了）。"
                    % (hero_label, POSITION_LABELS.get(hero, hero), decided_lane))
        return ("按英雄定位推的是「%s」，画面上多数时间在「%s」—— 以画面为准。"
                % (POSITION_LABELS.get(hero, hero), decided_lane))

    # -- 1. 画面证据够硬吗 ----------------------------------------------
    strong = bool(position) and count >= MIN_SEGMENTS and (
        total <= 0 or (float(count) / float(total)) >= MIN_SHARE)

    if strong:
        out.position = position
        out.lane_detail = lane if position == "side" else ""
        out.source = "按画面逐段统计（%s %d 段）" % (lane, count)
        out.provenance = "observed"
        conflicts: List[str] = []
        if raw_watch and raw_watch != "unknown" and watch == "unknown":
            conflicts.append(
                "看整局时说的分路「%s」认不出是哪条路 —— 以画面为准。"
                % raw_watch)
        elif watch and watch != position and watch != "unknown":
            conflicts.append(
                "看整局时说的分路是「%s」，画面上多数时间在「%s」—— "
                "以画面为准。" % (POSITION_LABELS.get(watch, watch), lane))
        hero_note = _hero_conflict(lane)
        if hero_note:
            conflicts.append(hero_note)
        out.conflict = " ".join(conflicts)
        return out

    # -- 2. 顶层那一个字段 ----------------------------------------------
    if watch and watch != "unknown":
        out.position = watch
        out.source = "按看整局时说的分路"
        out.provenance = "observed"
        conflicts = []
        if lane and position and position != watch:
            conflicts.append(
                "看整局时说的分路是「%s」，画面上「%s」更多（%d 段）—— "
                "没到压倒性的程度，先按看整局时说的算。"
                % (POSITION_LABELS.get(watch, watch), lane, count))
        if hero and hero != watch:
            conflicts.append(
                "按英雄「%s」的官方定位推的是「%s」，跟这个读数对不上"
                "—— 两个都不是画面上数出来的，请核对。"
                % (hero_label or "（没认出）", POSITION_LABELS.get(hero, hero)))
        out.conflict = " ".join(conflicts)
        return out

    # -- 3. 英雄查表 ----------------------------------------------------
    if hero:
        out.position = hero
        out.source = "按英雄推断"
        out.provenance = "inferred"
        return out

    # 模型说了分路但**认不出来** —— 说清是「认不出」，不是「没说」。
    if raw_watch and raw_watch != "unknown":
        out.conflict = ("看整局时说的分路「%s」认不出是哪条路 —— "
                        "分路这次没定出来。" % raw_watch)
    return out
