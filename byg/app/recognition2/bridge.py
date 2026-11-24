"""**桥**：把 v2 的时间线与判定结果，转成报告层认识的形状。

为什么要有它
------------
报告层（:mod:`app.report.builder`）读的是一套很具体的老形状：
``ctx.rounds[].actions``、``verdicts[].status``、窗口 ``label``。
v2 的数据形状不一样（``Beat.narrative`` 是模型自己的话，
``RuleJudgement`` 是纯代码判定）。

改报告层要动前端字段位（「前端冻结」范畴），所以这里做**桥**：
**只填已经在渲染的字段位，不动模板** —— 与 :mod:`app.harness.bridge`
用的是同一套约定。

一处关键的差别
--------------
老报告的「分析覆盖」表第二列读的是**动作枚举**（来自 OpenCV 探测器），
所以探测器认不出就显示「这段没识别到明确操作」。
这里填的是**模型自己写的句子** —— 表格里每一行都是模型的话。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .beat import Beat, Timeline
from .judge import (STATUS_DEVIATED, STATUS_NOT_APPLICABLE,
                    STATUS_SATISFIED, STATUS_UNDETERMINED)

#: 报告里那三列的标题对应的计数位。
_COUNT_KEYS = ("checked", "satisfied", "deviated", "undetermined")


def _mmss(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return "%d:%02d" % (seconds // 60, seconds % 60)


def _bucket_label(start_min: int, end_min: Optional[int],
                  is_last: bool) -> str:
    """窗口标签 —— 跟老报告一个口气。

    老报告的标签是「0:00」「0:00 ~ 2:00」「12:00 之后」，
    这里沿用同一套写法，玩家看过去不用重新学。
    """
    if is_last and end_min is None:
        return "%d:00 之后" % start_min
    if end_min is None or end_min <= start_min:
        return "%d:00" % start_min
    return "%d:00 ~ %d:00" % (start_min, end_min)


def bucket_beats(timeline: Timeline, bucket_minutes: float = 2.0
                 ) -> List[Tuple[str, List[Beat]]]:
    """把段落按**局内时间**分桶。

    **画面驱动**，不是规则窗口倒推 —— 画面里发生的事只要看得见，
    就落进某一个桶。老报告按攻略窗口切，画面里发生的事只要不落在
    攻略边界上就归不进任何一段（那是它「这段没识别到明确操作」的
    结构性来源）。
    """
    beats = [b for b in timeline.sorted_beats() if b.game_time >= 0]
    if not beats:
        return []
    step = max(0.5, float(bucket_minutes))
    buckets: Dict[int, List[Beat]] = {}
    for beat in beats:
        minute = int(beat.game_time // 60.0)
        # 桶的键取**桶起点**：第 0~2 分钟就是 0，第 2~4 分钟就是 2。
        key = int(minute // step) * int(step)
        buckets.setdefault(key, []).append(beat)
    ordered = sorted(buckets.items())
    out: List[Tuple[str, List[Beat]]] = []
    for i, (key, items) in enumerate(ordered):
        start_min = key
        end_min = key + int(step)
        is_last = (i == len(ordered) - 1)
        out.append((_bucket_label(start_min, end_min, is_last), items))
    return out


def beat_sentences(beats: Sequence[Beat]) -> List[str]:
    """这一桶里**模型自己写的**话。

    优先用叙述（narrative）；叙述为空但有动作名时退回动作名 ——
    宁可给个粗的，也不能给空（「空」最容易被读成「没分析」）。
    """
    out: List[str] = []
    for beat in beats:
        text = (beat.narrative or "").strip()
        if not text:
            label = (beat.action_label or "").strip()
            text = label if label and label != "unclear" else ""
        if not text:
            continue
        if text not in out:
            out.append(text)
    return out


def coverage_windows(timeline: Timeline, verdicts: Iterable[Any] = (),
                     bucket_minutes: float = 2.0) -> List[Dict[str, Any]]:
    """转成报告「分析覆盖」表要的那几行。

    每一行：``label`` / ``actions`` / ``action_text`` / ``checked`` /
    ``satisfied`` / ``deviated`` / ``undetermined`` / ``sampled``。

    ``verdicts`` 接受两种形状：纯代码的 ``RuleJudgement``
    与模型裁决的 ``ModelVerdict`` —— 两者都有 ``rule_id`` / ``status``，
    差别只在额外字段，这里只读这三个。
    """
    counts: Dict[int, Dict[str, int]] = {}

    def _bucket_of(minute: float) -> int:
        step = max(0.5, float(bucket_minutes))
        return int(minute // step) * int(step)

    for verdict in verdicts or []:
        status = str(getattr(verdict, "status", "") or "")
        span = getattr(verdict, "span", None)
        if span:
            key = _bucket_of(float(span[0]) / 60.0)
        else:
            window = getattr(verdict, "window", None)
            key = _bucket_of(float(window[0])) if window else 0
        cell = counts.setdefault(key, {k: 0 for k in _COUNT_KEYS})
        cell["checked"] += 1
        if status == STATUS_SATISFIED:
            cell["satisfied"] += 1
        elif status == STATUS_DEVIATED:
            cell["deviated"] += 1
        elif status == STATUS_UNDETERMINED:
            cell["undetermined"] += 1

    windows: List[Dict[str, Any]] = []
    for label, beats in bucket_beats(timeline, bucket_minutes):
        sentences = beat_sentences(beats)
        start_min = 0
        first = beats[0] if beats else None
        if first is not None:
            start_min = int(first.game_time // 60.0)
        key = _bucket_of(float(start_min))
        cell = counts.get(key, {k: 0 for k in _COUNT_KEYS})
        windows.append({
            "label": label,
            "actions": sentences,
            "action_text": "；".join(sentences) if sentences
                           else "这段没识别到明确操作",
            "checked": cell["checked"],
            "satisfied": cell["satisfied"],
            "deviated": cell["deviated"],
            "undetermined": cell["undetermined"],
            #: 采到了画面 —— 有模型的话就算采到了。
            "sampled": bool(sentences),
        })
    return windows


def coverage_summary(windows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """表头那几个汇总数。"""
    return {
        "windows": list(windows),
        "window_count": len(windows),
        "checked": sum(int(w.get("checked") or 0) for w in windows),
        "satisfied": sum(int(w.get("satisfied") or 0) for w in windows),
        "deviated": sum(int(w.get("deviated") or 0) for w in windows),
    }


def rule_entries(timeline: Timeline, verdicts: Iterable[Any],
                 patterns_by_id: Optional[Dict[str, Any]] = None
                 ) -> List[Dict[str, Any]]:
    """把判定结果转成「操作模式分析」那一块要的条目形状。

    老报告的条目长这样（``builder.py`` 里 ``item.get(...)`` 读的那几个）：

    ``kind`` / ``mode_id`` / ``lane`` / ``lane_label`` / ``title`` /
    ``verdict`` / ``did`` / ``why`` / ``evidence`` / ``advice`` /
    ``action`` / ``branch_action`` / ``branch_no`` / ``moments`` /
    ``frames`` / ``confidence``

    ``verdict`` 用的是**老口径**的名字：``ok`` / ``deviation`` / ``never`` /
    ``unclear``。这里做一次映射，**不新增口径**。
    """
    out: List[Dict[str, Any]] = []
    for verdict in verdicts or []:
        rule_id = str(getattr(verdict, "rule_id", "") or "")
        status = str(getattr(verdict, "status", "") or "")
        pattern = (patterns_by_id or {}).get(rule_id)
        title = ""
        lane = ""
        lane_label = ""
        action = ""
        if pattern is not None:
            title = str(getattr(pattern, "title", "") or "")
            lane = str(getattr(pattern, "lane", "") or "")
            action = str(getattr(pattern, "action", "") or "")
            lane_label = {"jungle": "打野", "mid": "中路", "side": "边路",
                          "carry": "发育路", "roam": "游走"}.get(lane, lane)
        did = str(getattr(verdict, "did", "") or "")
        why = str(getattr(verdict, "why", "") or "")
        advice = str(getattr(verdict, "advice", "") or "")
        entry: Dict[str, Any] = {
            "kind": "mode",
            "mode_id": rule_id,
            "lane": lane,
            "lane_label": lane_label,
            "title": title,
            "verdict": _legacy_verdict(status),
            "did": did,
            "why": why,
            "evidence": str(getattr(verdict, "evidence", "") or ""),
            "advice": advice,
            "action": action,
            "branch_action": advice or action,
            "branch_no": 0,
            "frames": list(getattr(verdict, "frames", []) or []),
            "confidence": float(getattr(verdict, "confidence", 0.0) or 0.0),
        }
        out.append(entry)
    return out


def _legacy_verdict(status: str) -> str:
    """把 v2 的四态映射成老报告认的那几个名字。

    ``not_applicable`` 映射成 ``unclear``：**老报告没有「不适用」这一档**，
    而「不适用」也不是「没做到」—— 混进 ``deviation`` 会冤枉人。
    """
    if status == STATUS_SATISFIED:
        return "ok"
    if status == STATUS_DEVIATED:
        return "deviation"
    if status == STATUS_UNDETERMINED:
        return "unclear"
    if status == STATUS_NOT_APPLICABLE:
        return "unclear"
    return "unclear"


# ---------------------------------------------------------------- 报告层形状


def _frame_times(beat) -> Dict[str, float]:
    """这张 Beat 的**帧文件名 → 原视频秒数**。

    ``Beat.frames`` 与 ``detail["frame_times"]`` 是**一一对应**的
    （同一个循环里追加的）—— 报告的「看这一段」要靠它把证据帧
    定位回原视频的那一秒。
    """
    out: Dict[str, float] = {}
    names = list(getattr(beat, "frames", []) or [])
    times = list((getattr(beat, "detail", {}) or {}).get("frame_times") or [])
    for i, name in enumerate(names):
        if i < len(times):
            try:
                out[str(name)] = float(times[i])
            except (TypeError, ValueError):
                continue
    return out


def _moments_for(beat, frame_names: Sequence[str],
                  why: str) -> List[Dict[str, Any]]:
    """证据帧 → 报告要的 ``moments``（带原视频秒数与文件名）。"""
    times = _frame_times(beat)
    out: List[Dict[str, Any]] = []
    for name in frame_names:
        if name not in times:
            continue
        out.append({"t": times[name], "image": name, "why": why})
    return out


def to_legacy_rounds(timeline: Timeline, verdicts: Iterable[Any],
                     patterns_by_id: Optional[Dict[str, Any]] = None,
                     bucket_minutes: float = 2.0) -> List[Dict[str, Any]]:
    """v2 的时间线 + 裁决 → 报告层认识的 ``rounds``。

    **报告层不用改一行代码** —— 它读的就是这个形状：
    ``ctx.rounds[].review``（模型逐条裁决）与 ``ctx.rounds[].actions``
    （动作）。v2 把自己的叙述写进 ``narrative``，报告的「分析覆盖」
    表第二列就显示**模型自己写的句子**，而不是探测器认出来的动作码。

    一条一条对应：

    ==================  ============================================
    报告层要的          v2 给的
    ==================  ============================================
    ``review``          模型裁决（:class:`ModelVerdict`）
    ``moments``         证据帧 + 原视频秒数
    ``did`` / ``why``   模型原话：看到了什么、凭什么这么判
    ``advice``          模型给的建议，没有就用规则的 ``deviation_hint``
    ``narrative``       这一段模型说的话（**新增字段**）
    ==================  ============================================
    """
    patterns_by_id = patterns_by_id or {}
    # 裁决按规则 id 归类，好挂到对应的窗口上。
    by_rule: Dict[str, List[Any]] = {}
    for verdict in verdicts or []:
        rule_id = str(getattr(verdict, "rule_id", "") or "")
        if rule_id:
            by_rule.setdefault(rule_id, []).append(verdict)

    rounds: List[Dict[str, Any]] = []
    buckets = bucket_beats(timeline, bucket_minutes)
    #: 每个时间片的局内时间范围（秒）。
    bucket_range: Dict[str, Tuple[float, float]] = {}
    for label, beats in buckets:
        lows = [b.game_time_start for b in beats if b.game_time_start >= 0]
        highs = [b.game_time_end for b in beats if b.game_time_end >= 0]
        bucket_range[label] = (min(lows) if lows else 0.0,
                               max(highs) if highs else 0.0)

    def _best_bucket(verdict) -> Optional[str]:
        """这条裁决该落在**哪个**时间片 —— 重叠最多的那个。

        为什么不能「落进所有重叠的时间片」：那会让每条结论都出现在
        第一段时间片上（第一片先被遍历到），报告上每条的「对局时间区间」
        都变成 ``0:00 ~ 2:00``，而证据帧明明在第 8 分钟。
        """
        span = getattr(verdict, "span", None)
        if not span or not any(span):
            return None
        lo, hi = float(span[0]), float(span[1])
        best: Optional[str] = None
        best_ov = 0.0
        for name, (b_lo, b_hi) in bucket_range.items():
            ov = min(hi, b_hi) - max(lo, b_lo)
            if ov > best_ov:
                best_ov = ov
                best = name
        return best

    #: 每条裁决的归属时间片。
    best_of: Dict[int, Optional[str]] = {}
    for items in by_rule.values():
        for verdict in items:
            best_of[id(verdict)] = _best_bucket(verdict)
    #: 没有区间的裁决归到**第一段** —— 挂在每一段等于没挂
    #: （那正是「每条结论的时间区间都是 0:00 ~ 2:00」的成因）。
    fallback_label = buckets[0][0] if buckets else None

    for label, beats in buckets:
        first = beats[0] if beats else None
        positions = [str((b.detail or {}).get("position") or "")
                     for b in beats]
        positions = [p for p in positions if p and p != "unknown"]
        position = ""
        if positions:
            position = max(set(positions), key=positions.count)

        # 这一段的模型裁决：规则的时间窗落在这一段里才算。
        review: List[Dict[str, Any]] = []
        for cand_rule_id, items in by_rule.items():
            pattern = patterns_by_id.get(cand_rule_id)
            for verdict in items:
                span = getattr(verdict, "span", None)
                if not span or not any(span):
                    # 不知道这段是哪一段 —— 挂第一段，别挂每一段。
                    if label != fallback_label:
                        continue
                else:
                    # 只进**归属的那一个**时间片。
                    if label != best_of.get(id(verdict)):
                        continue
                frames = list(getattr(verdict, "frames", []) or [])
                why = str(getattr(verdict, "why", "") or "")
                did = str(getattr(verdict, "did", "") or "")
                advice = str(getattr(verdict, "advice", "") or "")
                deviation_hint = ""
                if pattern is not None:
                    deviation_hint = str(
                        getattr(pattern, "deviation_hint", "") or "")
                entry: Dict[str, Any] = {
                    "kind": "mode",
                    "mode_id": cand_rule_id,
                    "lane": str(getattr(pattern, "lane", "") or ""),
                    "lane_label": _lane_label(
                        str(getattr(pattern, "lane", "") or "")),
                    "title": str(getattr(pattern, "title", "") or ""),
                    "premise": str(getattr(pattern, "premise", "") or ""),
                    "action": str(getattr(pattern, "action", "") or ""),
                    "outcome": str(getattr(pattern, "outcome", "") or ""),
                    "advice": advice or deviation_hint,
                    #: v2 的裁决是**对整条规则**的（模型看画面判「做到没有」），
                    #: 没有「第几支」这一层。老形状里 ``branch_no`` 既是
                    #: 「适用第几支」也是「这条适用不适用的标记」——
                    #: 给 1 表示「这一条适用」，``branch_premise`` 留空，
                    #: 报告就不会写出「第 N 支」那种话。
                    "branch_no": 1,
                    "branch_premise": "",
                    "branch_action": advice or deviation_hint,
                    "branch_outcome": "",
                    "verdict": _legacy_verdict(
                        str(getattr(verdict, "status", "") or "")),
                    "did": did,
                    "why": why,
                    "evidence": str(getattr(verdict, "evidence", "") or ""),
                    "frames": list(range(1, len(frames) + 1)),
                    "moments": [],
                    "confidence": float(
                        getattr(verdict, "confidence", 0.0) or 0.0),
                    "round_label": label,
                    # **检验区间**：报告要靠它把这条落到某一段时间上。
                    # 没有它，裁决会落进每一个时间片。
                    "span": [float(span[0]), float(span[1])] if span else [],
                    # **纯代码口径**：同一批事实用规则条件直接比对的结果。
                    # 只用来跟模型的裁决对账，不参与判定、不进执行度分母。
                    "code_status": str(
                        getattr(verdict, "code_status", "") or ""),
                }
                # 证据帧 → 原视频时刻。
                by_name = {str(n): n for n in frames}
                moments: List[Dict[str, Any]] = []
                for beat in beats:
                    for name, seconds in _frame_times(beat).items():
                        if name in by_name:
                            moments.append({"t": seconds, "image": name,
                                            "why": why or did})
                entry["moments"] = moments
                review.append(entry)

        narrative = "；".join(beat_sentences(beats))
        rounds.append({
            "label": label,
            "label_override": label,
            # **局内分钟**（整分）—— 适配器靠它算阶段、算规则时间窗。
            #
            # 没有它，适配器会把 ``time_min`` 判成缺失，整局每一段都落在
            # 「第 0 分钟」—— 于是所有带时间窗的规则一条都不触发，
            # 而报告上只是「没结论」，看不出是时间没对上。
            "time_min": int(first.game_time // 60) if first else 0,
            "stage": 0,
            "phase": 1,
            "time_index": round((first.game_time / 60.0), 2) if first else 0.0,
            "video_time": round(first.video_time, 2) if first else 0.0,
            "facts": {"position": position} if position else {},
            "actions": [],
            "missing_facts": [],
            "review": review,
            "probe_moments": [],
            "image": (beats[0].frames[0] if beats and beats[0].frames else ""),
            "narrative": narrative,
        })
    return rounds


def _lane_label(lane: str) -> str:
    """分路 → 报告里显示的中文。"""
    return {"jungle": "打野", "mid": "中路", "side": "边路",
            "carry": "发育路", "roam": "游走"}.get(lane, lane)

