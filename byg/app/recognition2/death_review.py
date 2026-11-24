# -*- coding: utf-8 -*-
"""失败对局的**倒推复盘**：从水晶爆炸往开局推，看每次怎么死的、下次怎么做。

为什么要有这一层
----------------
规则库判不了死因 —— 种子模式里写着「『有没有被切』读不到」，
所以那一类结论只能落在「参与团战的回合」上，**不指认是哪一次死亡**。
用户要的是恰恰这一句：**这局我死了几次，每次怎么死的，下次怎么办**。

口径（四条，缺一条就会变成编）
------------------------------
1. **只对失败局做**（``result == "lose"``）—— 赢了的局没有「败因」要推；
2. **倒序** —— 离结局越近的死亡越靠前（用户要的是「从水晶爆炸往开局推」）；
3. **死因要有画面依据** —— 模型给的 ``cause`` 只有在**那一段画面叙述里
   确实有死亡迹象**时才写进报告；没有依据就丢掉这一条，
   宁可不写，也不把模型编的死因当结论；
4. **绕过规则库** —— 这一层**独立于规则库的判定与模式筛选**：
   规则库判不了死因（种子模式里写着「『有没有被切』读不到」），
   所以死亡复盘自己走 —— 死因取模型看到的画面，
   「下次怎么做」直接从**攻略语料**里检索那一刻的打法。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: 画面里出现这些词，才算「玩家真的死了」。
#:
#: **只收玩家死的迹象**，不收「他击杀了敌人」——
#: 「击杀」「收掉」「人头」这些词在解说里两边都有，
#: 混进去会把玩家杀敌当成玩家阵亡。
#:
#: 「**倒在**」是 2026-11-13 补的：那一局的叙述写「最终**倒在**塔前完成一换一」，
#: 而死亡清单里那一次因此**被判成「画面里没读到死亡依据」丢掉**了
#: （说明位只能写「记下来的 2 次里有 1 次没读到死亡依据」）。
#: 「倒在 X 前」是解说里最常见的阵亡说法，漏掉它等于白丢一次复盘。
DEATH_MARKS = ("阵亡", "复活倒计时", "倒地", "倒在地上", "倒在", "死亡",
               "被击杀", "被杀", "被秒")

#: 模型说「**玩家自己杀了人**」的几种说法 ——
#:
#: 出现它们时，那句里的「击杀」是玩家的动作，不是玩家被杀。
#: 「被压制，随后完成击杀」——这一局玩家活着，不能报成死亡。
KILL_BY_PLAYER = ("完成击杀", "反杀", "收掉", "击杀敌人", "击杀敌方",
                  "拿下人头", "双杀", "三杀", "四杀", "五杀")

#: 「被…击杀」—— 中间**不能隔标点**。
#:
#: 「被先手控制并迅速击杀」是玩家死了；
#: 「被压制，随后完成击杀」是玩家杀了别人 —— 隔了逗号就不算。
_DEATH_BY_RE = re.compile(r"被[^，。；、！？\n]{0,12}击杀")


def looks_like_death(text: Any) -> bool:
    """这段话里有没有「玩家阵亡」的迹象。

    **宁可漏，不可编** —— 假死亡会凭空多出一条复盘结论，
    而漏掉的死亡只是少一条复盘。
    所以那个弱信号（「被…击杀」）要卡紧。
    """
    t = str(text or "")
    for mark in DEATH_MARKS:
        if mark in t:
            return True
    if any(k in t for k in KILL_BY_PLAYER):
        # 玩家自己杀了人 —— 那句「被…击杀」多半说的是别人的遭遇。
        return False
    return bool(_DEATH_BY_RE.search(t))


def _clip(text: Any, limit: int = 140) -> str:
    """截到一句话那么长。"""
    s = " ".join(str(text or "").split())
    if len(s) <= limit:
        return s
    return s[:limit].rstrip("，。、 ") + "…"


def _split_clauses(text: str) -> List[str]:
    """一句一句 —— 建议挤成一段，条件与做法就糊在一起了。"""
    import re

    out: List[str] = []
    for part in re.split(r"(?<=。)", str(text or "")):
        part = part.strip()
        if part:
            out.append(part)
    return out


@dataclass
class DeathMoment:
    """一次死亡（连同它在画面里的依据与对应的攻略）。"""

    #: 第几次（按**模型报的那份清单**里的自然时间编号：第一次死亡 = 1）。
    #:
    #: **按清单里的序号，不是按留下来的条数** —— 有一条因为没有画面依据
    #: 被丢掉时，后面几条的序号不变（不然「第 1 次」会指错那一次）。
    index: int = 0
    #: 模型报的**死亡总次数** —— 标题里的「这局共 N 次」用它。
    total: int = 0
    #: 原视频里的秒数。
    video_start: float = 0.0
    #: 局内时刻（MM:SS），读不到就空。
    clock: str = ""
    #: 死在哪儿。
    where: str = ""
    #: 模型说的死因（**已核对画面**）。
    cause: str = ""
    #: 玩家自己哪一步出的问题（模型说看不出就是空）。
    fault: str = ""
    #: 那一段的画面叙述（原文）—— 「看到了什么」的主料。
    narrative: str = ""
    #: 这一局的时间区间标签（如「16:00 ~ 18:00」）。
    time_range: str = ""
    #: 那一刻的攻略（``title/premise/action/outcome/advice``）——
    #: 直接从**攻略语料**检索来的，不走规则库的判定层。
    guide: Dict[str, Any] = field(default_factory=dict)


#: 一段（桶）有多宽 —— 与 :func:`app.recognition2.bridge.bucket_beats`
#: 的默认桶宽一致（2 分钟）。
_BUCKET_SECONDS = 120.0


def _mmss_to_sec(text: Any) -> Optional[float]:
    """``MM:SS`` -> 秒。解析不了就返回 None（不猜）。"""
    t = str(text or "").strip()
    if not t:
        return None
    if ":" in t:
        head, _, tail = t.partition(":")
        try:
            return int(head) * 60 + int(tail)
        except ValueError:
            return None
    try:
        return float(t)
    except ValueError:
        return None


def _round_span(rnd: Any) -> Optional[Tuple[float, float]]:
    """这一段的**局内时间**区间（秒）。

    标签形如「10:00 ~ 12:00」（局内分钟）——
    那是这一段**覆盖的时间范围**，不是它的第一帧。
    """
    label = str(getattr(rnd, "label_override", "")
                or getattr(rnd, "label", "") or "").strip()
    if not label:
        return None
    parts = label.split("~")
    lo = _mmss_to_sec(parts[0].strip())
    if lo is None:
        return None
    if len(parts) >= 2:
        hi = _mmss_to_sec(parts[1].strip())
        if hi is None:
            return None
        return (lo, hi)
    return (lo, lo + _BUCKET_SECONDS)


def _game_second(raw: Dict[str, Any]) -> Optional[float]:
    """这次死亡的**局内时刻**（秒）。读不到就 None（不猜）。"""
    minute = raw.get("minute")
    if minute is not None:
        try:
            return float(minute) * 60.0
        except (TypeError, ValueError):
            pass
    return _mmss_to_sec(raw.get("clock") or "")


def _nearest_round(rounds: Sequence[Any], video_start: float,
                   game_second: Optional[float] = None) -> Optional[Any]:
    """离死亡时刻最近的那一段。

    **按区间求交**，不是按「这一段的第一帧」就近比。

    段的首帧是那一段的**开头**：死亡落在段的后半段时，
    按首帧比会把它判给**下一段**（下一段的叙述里根本没有这次死亡）——
    接地的依据就错了，表现是「明明那一段写着阵亡，这次死亡却没被接住」。

    优先用**局内时间**（标签里的分钟区间）；
    局内时刻读不出来时退回按视频秒比（老办法，只在没有更好选择时用）。
    """
    if game_second is not None:
        spanned: List[Tuple[Any, float, float]] = []
        for rnd in rounds or []:
            span = _round_span(rnd)
            if span is not None:
                spanned.append((rnd, span[0], span[1]))
        if spanned:
            best = None
            best_gap = None
            for rnd, lo, hi in spanned:
                if game_second < lo:
                    gap = lo - game_second
                elif game_second > hi:
                    gap = game_second - hi
                else:
                    gap = 0.0
                if best_gap is None or gap < best_gap:
                    best, best_gap = rnd, gap
            return best

    best = None
    best_gap = None
    for rnd in rounds or []:
        vt = getattr(rnd, "video_time", None)
        try:
            vt = float(vt or 0.0)
        except (TypeError, ValueError):
            vt = 0.0
        gap = abs(vt - float(video_start))
        if best_gap is None or gap < best_gap:
            best, best_gap = rnd, gap
    return best


def collect(deaths: Sequence[Any], rounds: Sequence[Any]) -> List[DeathMoment]:
    """把模型给的死亡清单接上画面依据。**没有画面依据的丢掉**。

    ``deaths`` 是落库的死亡清单（``meta["deaths"]``），``rounds`` 是
    这一局的逐轮记录（每轮带画面叙述与视频秒）。

    **编号按模型报的那份清单** —— 被丢掉的那几次不占号，
    剩下的几条仍报自己真正的「第 N 次」。
    """
    out: List[DeathMoment] = []
    # **先按时间排，再编号** —— 「第几次死亡」要按自然时间算
    # （第一次 = 最早那一次），不能跟着模型报的顺序走。
    ordered: List[Tuple[float, Dict[str, Any]]] = []
    for raw in deaths or []:
        if not isinstance(raw, dict):
            continue
        try:
            vs = float(raw.get("video_start") or 0.0)
        except (TypeError, ValueError):
            continue
        ordered.append((vs, raw))
    ordered.sort(key=lambda item: item[0])
    total = len(ordered)
    for index, (vs, raw) in enumerate(ordered, 1):
        rnd = _nearest_round(rounds, vs, _game_second(raw))
        narrative = str(getattr(rnd, "narrative", "") or "").strip()
        # **接地闸**：画面里没有死亡迹象，这条就不算 —— 模型可能编死因。
        if not narrative or not looks_like_death(narrative):
            continue
        out.append(DeathMoment(
            index=index,
            total=total,
            video_start=vs,
            clock=str(raw.get("clock") or ""),
            where=str(raw.get("where") or "").strip(),
            cause=str(raw.get("cause") or "").strip(),
            fault=str(raw.get("fault") or "").strip(),
            narrative=narrative,
            time_range=str(getattr(rnd, "label_override", "")
                           or getattr(rnd, "label", "") or ""),
            guide=dict(raw.get("guide") or {}),
        ))
    return out


def _title(moment: DeathMoment, total: int) -> str:
    """条目标题：第几次 + 死在哪 / 怎么死的。"""
    where = moment.where or ""
    cause = _clip(moment.cause, 24)
    tail = where or cause
    if tail and cause and where and cause != where:
        tail = "%s（%s）" % (where, cause)
    if not tail:
        tail = moment.time_range or "画面里没读到具体位置"
    return "第 %d 次死亡（这局共 %d 次）· %s" % (moment.index, total, tail)


def _reason(moment: DeathMoment) -> str:
    """「为什么有用」—— 攻略要 X，你这局是 Y（**对照着说**）。"""
    guide = moment.guide or {}
    premise = str(guide.get("premise") or "").strip()
    outcome = str(guide.get("outcome") or "").strip()
    did = _clip(moment.fault or moment.cause or moment.narrative, 60)
    parts: List[str] = []
    if premise:
        parts.append("攻略在这个节点的前提是「%s」" % premise)
    if outcome:
        parts.append("照着做会得到「%s」" % outcome)
    if did:
        parts.append("你这次是%s" % did)
    if not parts:
        return "这一段没查到可对照的攻略，只能把画面上的死因摆出来。"
    return "；".join(parts) + "。"


def build_entries(moments: Sequence[DeathMoment],
                  video_url: str = "") -> List[Dict[str, Any]]:
    """把死亡清单做成报告正文里的条目（每次死亡一条）。

    **倒序**：离结局越近的死亡越靠前 —— 用户要的是「从水晶爆炸往开局推」。

    「该怎么做」直接来自**攻略语料**（检索那一刻的打法）——
    这一层**绕过规则库**：规则库那套判不了死因
    （种子模式里写着「『有没有被切』读不到」），也不该由它来挑。
    """
    # **总数用模型报的那份清单** —— 不是留下来的条数。
    # 少了几次是因为画面里没读到依据（报告层会说明），
    # 标题里写「这局共 1 次」而说明位写「这局有 6 次」，那是自相矛盾。
    total = 0
    for moment in moments or []:
        total = max(total, int(getattr(moment, "total", 0) or 0))
    if not total:
        total = len(moments or [])
    if not total:
        return []
    ordered = sorted(moments, key=lambda m: -float(m.video_start or 0.0))
    entries: List[Dict[str, Any]] = []
    for moment in ordered:
        guide = moment.guide or {}
        advice = str(guide.get("advice") or "").strip()
        if not advice:
            advice = "这一段没查到可对照的攻略。"
        entries.append({
            #: 条目类型 —— 报告正文的三段式靠它分清「明细」与「模式判定」。
            "kind": "death",
            "mode_id": "death-%d" % moment.index,
            "category": "死亡复盘",
            "title": _title(moment, total),
            "lane_label": "",
            "phase_label": "",
            "verdict_label": "这里可以调整",
            "time_range": moment.time_range,
            "clip": {"video_url": video_url, "moments": [], "seconds": 0},
            # **模型写的白描不上前端**（用户拍板）—— 死因留在标题与理由里，
            # 整段原话不展示：这一页要回答的是「我该怎么改」。
            "actual_text": "",
            "evidence": [],
            "absent_kinds": [],
            "advice_text": advice,
            "advice_lines": _split_clauses(advice),
            "reason_text": _reason(moment),
            "low_confidence": False,
            "confidence": 1.0,
        })
    return entries
