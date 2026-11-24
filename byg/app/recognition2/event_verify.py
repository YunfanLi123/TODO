"""用顶栏 K/D/A 重建「玩家击杀/阵亡」事实，再把模型事件挂上去。

核心顺序（2026-11-14 用户指出的问题）
------------------------------------
旧做法：拿模型给的事件 → 在它附近找 KDA +1 → 把事件时刻改到那一帧 → 同类型再合并。
问题：**同一时段两次击杀，会被并成一次**；而且「找最近的 +1」会把多个事件
都吸到同一个增量上。

新做法：**KDA 是事实骨架，模型描述只是挂在骨架上的文字**。
1. 在候选事件的前后 ±30 秒范围内，**每秒**读一次顶栏 K/D/A；
2. 击杀数每 +1 生成一个「玩家击杀」事实，死亡数每 +1 生成一个「阵亡」事实
   （一次 +2 就生成两条 —— 双杀不会被吞）；
3. 模型事件按时间**一对一**匹配事实：匹配上的用事实时刻，匹配不上的
   （队友击杀）删掉；没被匹配的事实补一条空描述事件，交给细看那段去写；
4. **不再按时间合并事件** —— 两次击杀哪怕只隔 1 秒，也是两条事实。

读 K/D/A 仍然只让模型照抄三个数字（裁顶栏、放大），加event由代码算。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: 候选事件前后各看多少秒。
SPAN = 30.0

#: 采样间隔：2 秒一次（双杀隔几秒也能分开，调用量减半）。
STEP = 2.0

#: 只校验这两类（拆塔/丢塔没有 KDA 可对）。
VERIFY_TYPES = ("kill", "death")


def _sample_times(events: Sequence[Any], span: float = SPAN,
                  step: float = STEP) -> List[float]:
    """所有候选窗口取并集，按 step 采样。"""
    spans: List[Tuple[float, float]] = []
    for event in events:
        if str(getattr(event, "event_type", "") or "") not in VERIFY_TYPES:
            continue
        at = float(getattr(event, "video_start", 0.0) or 0.0)
        spans.append((max(0.0, at - span), at + span))
    if not spans:
        return []
    spans.sort()
    merged: List[List[float]] = [list(spans[0])]
    for lo, hi in spans[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])

    times: List[float] = []
    for lo, hi in merged:
        t = lo
        while t <= hi + 1e-6:
            times.append(round(t, 1))
            t += step
    return times


def _truths(samples: Sequence[Any]) -> Dict[str, List[float]]:
    """从 K/D/A 序列推出玩家击杀/阵亡时刻。

    相邻两次读数：击杀 +1 记一次击杀；+2 就记两次（双杀）。
    时刻取**后一次读数**（增量在这一秒内被看到）。

    **防误读**：模型偶尔把数字读错一个（3 读成 4），下一帧又读回 3 ——
    那是一次假击杀。所以增量必须**被后面的读数确认**（后面还保持 ≥ 新值），
    单帧跳变不算。
    """
    out: Dict[str, List[float]] = {"kill": [], "death": []}
    ok = [s for s in samples if getattr(s, "ok", False)]
    ok.sort(key=lambda s: float(getattr(s, "t", 0.0) or 0.0))
    n = len(ok)
    for i in range(n - 1):
        prev, cur = ok[i], ok[i + 1]
        for attr, kind in (("k", "kill"), ("d", "death")):
            a = getattr(prev, attr, None)
            b = getattr(cur, attr, None)
            if a is None or b is None:
                continue
            delta = int(b) - int(a)
            if delta <= 0:
                continue
            confirmed = any(
                getattr(ok[j], attr, None) is not None
                and int(getattr(ok[j], attr)) >= int(b)
                for j in range(i + 2, n))
            if not confirmed:
                continue
            for _ in range(max(0, delta)):
                out[kind].append(float(cur.t))
    return out


def _match(candidates: Sequence[Any], truths: Sequence[float],
           span: float) -> Tuple[List[Tuple[Any, float]], List[Any], List[float]]:
    """事件与事实一对一匹配（按时间就近，不重复使用）。

    返回 (匹配上的 (事件, 事实时刻), 没匹配上的事件, 没被用的事实)。
    """
    used_truth: set = set()
    matched: List[Tuple[Any, float]] = []
    unmatched_events: List[Any] = []
    used_event: set = set()
    order = sorted(range(len(candidates)),
                   key=lambda i: float(getattr(candidates[i], "video_start", 0.0)))
    for index in order:
        event = candidates[index]
        at = float(getattr(event, "video_start", 0.0) or 0.0)
        best = None
        for ti, when in enumerate(truths):
            if ti in used_truth:
                continue
            gap = abs(float(when) - at)
            if gap <= span and (best is None or gap < best[0]):
                best = (gap, ti, when)
        if best is None:
            unmatched_events.append(event)
            continue
        used_truth.add(best[1])
        matched.append((event, best[2]))
        used_event.add(index)
    leftover = [when for ti, when in enumerate(truths) if ti not in used_truth]
    return matched, unmatched_events, leftover


def _dedupe_towers(events: List[Any]) -> int:
    """同类型、相差 ≤2 秒的拆塔/丢塔只留一条（同一条「已摧毁」提示）。"""
    events.sort(key=lambda e: float(getattr(e, "video_start", 0.0)))
    unique: List[Any] = []
    merged = 0
    for event in events:
        kind = str(getattr(event, "event_type", "") or "")
        if (unique and kind in ("tower_lost", "tower_taken")
                and str(getattr(unique[-1], "event_type", "")) == kind
                and abs(float(getattr(event, "video_start", 0.0))
                        - float(getattr(unique[-1], "video_start", 0.0))) <= 2.0):
            if len(str(getattr(event, "narrative", "") or "")) > len(
                    str(getattr(unique[-1], "narrative", "") or "")):
                unique[-1] = event
            merged += 1
            continue
        unique.append(event)
    events[:] = unique
    return merged


def verify_events(events: List[Any], video_path: str, *, model: str,
                  creds: Any = None, data_dir: str = "",
                  transform: Any = None, span: float = SPAN,
                  read_series: Optional[Callable[..., Any]] = None,
                  on_note: Optional[Callable[[str], None]] = None
                  ) -> Dict[str, Any]:
    """用 KDA 事实重排 kill/death 事件。"""
    if read_series is None:
        from .kda_read import read_series as _read
        read_series = _read

    notes: List[str] = []
    candidates = [e for e in events
                  if str(getattr(e, "event_type", "") or "") in VERIFY_TYPES]
    if not candidates:
        merged = _dedupe_towers(events)
        if merged:
            notes.append("拆塔/丢塔同一提示重复：合并 %d 条。" % merged)
        return {"fixed": 0, "dropped": 0, "added": 0, "notes": notes,
                "calls": 0, "tokens": 0}

    times = _sample_times(candidates, span)
    try:
        result = read_series(video_path, times, model=model,
                             data_dir=data_dir, transform=transform)
    except Exception as exc:                           # noqa: BLE001
        notes.append("读 K/D/A 失败（%s）—— 事件保持原样。" % exc)
        return {"fixed": 0, "dropped": 0, "added": 0, "notes": notes,
                "calls": 0, "tokens": 0}

    calls = int(getattr(result, "calls", 0) or 0)
    tokens = int(getattr(result, "tokens", 0) or 0)
    samples = list(getattr(result, "samples", []) or [])
    if not [s for s in samples if getattr(s, "ok", False)]:
        notes.append("K/D/A 一个数都没读到 —— 事件保持原样。")
        return {"fixed": 0, "dropped": 0, "added": 0, "notes": notes,
                "calls": calls, "tokens": tokens}

    facts = _truths(samples)
    fixed = dropped = added = 0
    by_kind = {kind: [e for e in candidates
                      if str(getattr(e, "event_type", "")) == kind]
               for kind in VERIFY_TYPES}

    for kind in VERIFY_TYPES:
        matched, unmatched, leftover = _match(by_kind[kind], facts[kind], span)
        for event, when in matched:
            event.video_start = float(when)
            event.video_end = max(float(when), float(event.video_end))
            fixed += 1
        for event in unmatched:
            # 击杀和阵亡都按事实删：K/D/A 是玩家自己的数字，没有 +1 就不是他的。
            dropped += 1
            label = "击杀" if kind == "kill" else "阵亡"
            notes.append("第 %.0f 秒的「%s」不是玩家的（K/D/A 没有对应增量），已删。"
                         % (float(getattr(event, "video_start", 0.0)), label))
            if event in events:
                events.remove(event)
        for when in leftover:
            from .watcher import WatchEvent
            events.append(WatchEvent(video_start=float(when),
                                     video_end=float(when),
                                     event_type=kind,
                                     narrative=""))
            added += 1

    # 事实事件已经一一对上；不再按时间合并（相隔 1 秒的双杀是两条）。
    # 但**拆塔/丢塔**没有 KDA 可用：同一条「已摧毁」提示会被两条事件都抓到，
    # 同类型、相差 ≤2 秒的只留一条。
    merged_towers = _dedupe_towers(events)
    if merged_towers:
        notes.append("拆塔/丢塔同一提示重复：合并 %d 条。" % merged_towers)
    if fixed or dropped or added:
        notes.append("K/D/A 事实：校正 %d 条、删除 %d 条队友击杀、补入 %d 条。"
                     % (fixed, dropped, added))
    if on_note is not None:
        for note in notes:
            on_note(note)
    return {"fixed": fixed, "dropped": dropped, "added": added,
            "notes": notes, "calls": calls, "tokens": tokens}
