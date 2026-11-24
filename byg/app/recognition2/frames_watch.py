# -*- coding: utf-8 -*-
"""**抽帧看整局** —— 给「收不了视频」的模型留的那条路（2026-11-13）。

为什么要有它
-----------
这套识别链原本是「把整局录像打包后**直接给模型看**」（README §17b/§28），
而 OpenAI 兼容的接口只认 ``text / image_url / file`` —— 实测拿 DeepSeek
发视频块，服务端回 ``422 unknown variant `video_url```。也就是说：
**DeepSeek 能看图，看不了视频**，整局那条路直接断掉。

所以这里给一条兜底：**按画面变化抽帧 → 每 5 分钟一批发过去 → 按时间拼起来**。

三条设计取舍（都不是随手定的）
------------------------------
1. **按画面变化抽，不按时间平均抽**
   —— 打架、转场、抢资源的时段切得密，清线刷野切得疏。
   复用 :mod:`app.recognition2.sampler` 那套（小地图差分 + 场景差分），
   它已经在真实录屏上标定过。

2. **分批，一波 5 分钟、最多 32 张**
   —— 整局一次发 150 张：一是上下文撑不下，二是**输出一定会被截断**
   （旧视频链实测在 16384 token 处断，整局只剩前 86 段）。
   分批之后每批只讲 5 分钟，输出量小、时间点也更准。
   批与批之间只传一句「上一批讲到哪」，不传全部事件 —— 传了它就跟着复读。

3. **时间一律用「原视频第几秒」，而且每张都标出来**
   —— 抽帧丢的是连续性，不能连时间也丢。模型的 ``t`` 只许取标出来的秒数
   （或相邻两张之间），下游的事件时刻、证据帧、死亡时刻全靠它。

质量上它比整局视频差在哪
------------------------
两次采样之间发生的事**就是看不到**：技能放没放、位移走没走、团战谁先手，
这些只有连续画面能看出来。所以这里**不假装看得到**：
提示词明写「两帧之间怎么到的，不要编」，读不出的字段照旧写 unclear。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import watcher
from .selector import parse_requests
from .watch_prompt import (
    FRAMES_SYSTEM_PROMPT as _FRAMES_SYSTEM_PROMPT,
    build_frames_prompt as _build_frames_prompt,
)
from .watcher import (Death, WatchConfig, WatchEvent, WatchResult,
                      _parse_deaths, _resolve_match_start, _token_count,
                      _norm_result, parse_clock, parse_events)

#: **抽帧密度**（2026-11-14 用户要求：抽到 1~2 fps 量级）。
#:
#: 以前是「按画面变化抽、一局约 128 张」≈ 1 张/9 秒，两次采样之间
#: 的技能、位移、先手全看不到。现在按 **1 秒 1 张** 铺满整局，
#: 再用总量上限压到可承受的规模（见 ``MAX_TOTAL``）。
FRAME_FPS = 1.0

#: 一批覆盖多少秒。60 秒一批：1 fps 的网格下每批候选 60 张，
#: 受 ``PER_WINDOW``（32）与全局配额压缩后落在 25~32 张；
#: 15 分钟一局 ≈ 15 批，比原来 4 批密一个量级。
WINDOW_SEC = 60.0

#: 一批最多给几张。
#:
#: 为什么是 32：一张 1280×574 的 JPEG 在视觉模型那边大约占 700~1500 token，
#: 32 张 + 提示词 ≈ 3~5 万 token，稳在上下文里；再多就是拿钱换不出信息量的帧。
PER_WINDOW = 32

#: 整局总共最多抽几张（超了按窗口等比压缩）。
#:
#: 420 张 ≈ 15 分钟对局的平均 2 秒一张、关键窗口 1 秒一张。
#: 这个量级的代价要说清楚：420 张 × 约 800~1200 token ≈ 40 万 token/局，
#: 是原来（128 张）的三倍多 —— 换来的是「两次采样之间不再瞎猜」。
MAX_TOTAL = 420

#: 抽帧的采样步长（秒）—— 跟着 :data:`FRAME_FPS` 走（1 fps → 1 秒一探）。
SAMPLE_STEP = 1.0 / FRAME_FPS

#: 每张图最长边（超过就缩到 1280 宽）。
#:
#: 不缩的话 1080p 原片的帧会把 token 翻倍，而画面上要读的字
#: （顶栏时刻、战绩面板）在 1280 宽下已经够清楚。
MAX_WIDTH = 1280

#: 抽帧专用系统提示：系统提示 + 一段「这次是抽帧」的说明（见 watch_prompt）。
FRAMES_SYSTEM_PROMPT = _FRAMES_SYSTEM_PROMPT


@dataclass
class Frame:
    """一张抽出来的画面。"""

    t: float = 0.0
    jpeg: bytes = b""

    def label(self) -> str:
        return "第 %.1f 秒" % float(self.t)


# ---------------------------------------------------------------- 抽帧


def _encode(frame: Any, transform: Any = None) -> bytes:
    """把一帧转成 JPEG 字节（方向归一 + 限宽 + 压缩）。"""
    import cv2

    img = frame
    if transform is not None:
        try:
            img = transform(img)
        except Exception:                              # noqa: BLE001
            img = frame
    if img is None or getattr(img, "size", 0) <= 0:
        return b""
    h, w = img.shape[:2]
    if w > MAX_WIDTH:
        img = cv2.resize(img, (MAX_WIDTH, max(1, int(h * MAX_WIDTH / float(w)))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return bytes(buf) if ok else b""


def read_frames(video_path: str, times: Sequence[float],
                transform: Any = None) -> List[Frame]:
    """按时间戳取帧。

    **按 ``CAP_PROP_POS_MSEC`` seek，不用 grab() 递进** —— 实测这段录屏上
    ``grab()`` 会跳帧，「grab 了 N 次」不等于「第 N 帧」，时间戳会指错
    （见 :meth:`app.recognition2.sampler.ContentSampler.sample` 的注释）。
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    out: List[Frame] = []
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t)) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            actual = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if not actual or abs(actual - float(t)) > 12.0:
                actual = float(t)       # seek 没落在预期位置时按目标时间记
            data = _encode(frame, transform)
            if data:
                out.append(Frame(t=round(float(actual), 1), jpeg=data))
    finally:
        cap.release()
    return out


def candidate_times(video_path: str, step: float = SAMPLE_STEP,
                    transform: Any = None) -> List[float]:
    """按画面变化给出**候选时刻**（不取图，只算时间）。

    复用 :class:`app.recognition2.sampler.ContentSampler`：小地图差分
    （玩家位置变了）或场景差分（镜头换地方了）就切一刀，每段留 2~3 帧。
    打架的时段段切得密 → 自然拿到更多帧；清线刷野的时段段少 → 帧也少。
    """
    from .sampler import ContentSampler, SamplerConfig

    cfg = SamplerConfig(step=max(1.0, float(step)),
                        max_frames_per_segment=3,
                        min_frames_per_segment=2)
    try:
        segments = ContentSampler(video_path, cfg).sample()
    except Exception:                                  # noqa: BLE001
        return []
    times: List[float] = []
    seen: set = set()
    for seg in segments:
        for t in seg.frames:
            key = round(float(t), 1)
            if key in seen:
                continue
            seen.add(key)
            times.append(key)
    times.sort()
    # **按帧率补一张网**（2026-11-14）：内容采样在平静段只给 2~3 帧，
    # 一局下来才一百多张 ≈ 1 张/9 秒 —— 技能、位移、先手全落在缝里。
    # 这里再铺一层 1/fps 的均匀网格，与内容候选合并；
    # 总量由 :func:`pick_windows` 的 ``max_total`` 统一压。
    step = 1.0 / max(0.2, float(FRAME_FPS))
    grid: List[float] = []
    if times:
        start, end = times[0], times[-1]
        t = start
        while t <= end:
            grid.append(round(t, 1))
            t += step
    merged = sorted(set(round(x, 1) for x in list(times) + grid))
    return merged


def pick_windows(times: Sequence[float], *, window_sec: float = WINDOW_SEC,
                 per_window: int = PER_WINDOW,
                 max_total: int = MAX_TOTAL) -> List[Tuple[float, float, List[float]]]:
    """把候选时刻分进「第 N 批」，每批不超过 ``per_window`` 个。

    返回 ``[(批起点秒, 批终点秒, [时刻…]), …]``。窗口里的时刻**已经过密度筛选**
    （忙的时段候选就多），这里只是按窗口裁到上限，裁的时候**均匀取**，
    保证这一批的首尾都在。
    """
    if not times:
        return []
    window_sec = max(30.0, float(window_sec))
    per_window = max(4, int(per_window))
    buckets: Dict[int, List[float]] = {}
    for t in times:
        buckets.setdefault(int(float(t) // window_sec), []).append(float(t))

    # 全局上限：窗口数 × 每窗上限超了，就等比压每窗的配额。
    n_win = max(1, len(buckets))
    quota = max(4, min(per_window, int(max_total // n_win) or 4))

    out: List[Tuple[float, float, List[float]]] = []
    for key in sorted(buckets):
        items = sorted(buckets[key])
        if len(items) > quota:
            idx = [round(i * (len(items) - 1) / float(quota - 1))
                   for i in range(quota)]
            items = [items[i] for i in sorted(set(int(x) for x in idx))]
        lo = key * window_sec
        out.append((lo, lo + window_sec, items))
    return out


# ---------------------------------------------------------------- 提示词


def build_frames_prompt(lo: float, hi: float, times: Sequence[float],
                        index: int, total: int,
                        prev_summary: str = "",
                        hero: str = "", position: str = "") -> str:
    """造这一批的提问（提示词本体在 watch_prompt）。"""
    return _build_frames_prompt(lo, hi, times, index, total, prev_summary,
                                hero, position)


# ---------------------------------------------------------------- 编排


def _merge_events(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """各批事件拼起来：按时间排序，同一时刻重复的去掉。"""
    out: List[Dict[str, Any]] = []
    seen: List[Tuple[float, str]] = []
    for item in sorted(items, key=lambda x: float(x.get("t") or 0.0)):
        t = item.get("t")
        text = str(item.get("narrative") or "").strip()
        if t is None:
            out.append(item)
            continue
        dup = False
        for tt, tt_text in seen:
            if abs(tt - float(t)) <= 3.0 and (not text or text == tt_text):
                dup = True
                break
        if dup:
            continue
        seen.append((float(t), text))
        out.append(item)
    return out


def _merge_deaths(items: Sequence[Any]) -> List[Death]:
    """各批死亡清单拼起来：按时间排序，15 秒内的重复只留一条。

    ⚠️ 进出的都是 :class:`watcher.Death` **对象**，不是字典 ——
    ``_parse_deaths()`` 给的就是对象。2026-11-13 实测就栽在这上面：
    拿 ``item.get("video_start")`` 去读对象，整局扫描直接抛
    ``AttributeError: 'Death' object has no attribute 'get'``，
    用户在页面上看到的是「扫描失败」。
    """
    out: List[Death] = []
    for item in sorted(items,
                       key=lambda d: float(getattr(d, "video_start", 0.0) or 0.0)):
        t = float(getattr(item, "video_start", 0.0) or 0.0)
        if any(abs(t - float(getattr(x, "video_start", 0.0) or 0.0)) <= 15.0
               for x in out):
            continue
        out.append(item)
    return out


def _merge_requests(items: Sequence[Any]) -> List[Any]:
    """各批的检索请求合并：按时间排，同一条只留一次，最多 6 条。"""
    out: List[Any] = []
    seen: set = set()
    for item in sorted(items, key=lambda r: (r.at if r.at is not None else 1e9)):
        key = str(getattr(item, "query", "") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out[:6]


def _majority(values: Sequence[str]) -> str:
    """出现最多的那个（空串不算）。"""
    counts: Dict[str, int] = {}
    for v in values:
        v = str(v or "").strip()
        if not v or v == "unknown":
            continue
        counts[v] = counts.get(v, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda kv: kv[1])[0]


def watch_by_frames(video_path: str, *, cfg: WatchConfig,
                    model: str, creds: Any,
                    transform: Any = None,
                    hero: str = "", position: str = "") -> WatchResult:
    """抽帧把整局「看」一遍（给收不了视频的模型）。

    返回与 :func:`watcher.watch_video` 同形状的 :class:`WatchResult` ——
    下游（时间线、证据帧、检索、裁决）一行都不用改。
    """
    from app.collection import vlm_client

    out = WatchResult()
    times = candidate_times(video_path, transform=transform)
    if not times:
        out.notes.append("抽帧失败：这一局一张画面都没取到 —— 整局没有分析。")
        return out
    windows = pick_windows(times)
    out.notes.append(
        "这个模型收不了视频 —— 改用抽帧：整局抽了 %d 张，分 %d 批（每批 %.0f 分钟）。"
        % (sum(len(w[2]) for w in windows), len(windows), WINDOW_SEC / 60.0))

    events: List[Dict[str, Any]] = []
    deaths: List[Dict[str, Any]] = []
    requests: List[Any] = []
    heads: Dict[str, List[str]] = {"hero": [], "position": [], "result": []}
    match_start: Optional[float] = None
    summaries: List[str] = []
    prev_summary = ""
    failed = 0

    for index, (lo, hi, times_in) in enumerate(windows):
        frames = read_frames(video_path, times_in, transform)
        if not frames:
            failed += 1
            out.notes.append("第 %d 批（%.0f~%.0f 秒）一张图都没取到，跳过。"
                             % (index + 1, lo, hi))
            continue
        blocks: List[Dict[str, Any]] = [vlm_client.text_block(
            build_frames_prompt(lo, hi, [f.t for f in frames], index,
                                len(windows), prev_summary=prev_summary,
                                hero=hero, position=position))]
        for frame in frames:
            blocks.append(vlm_client.image_block(frame.jpeg))
        try:
            result = vlm_client.call_model(
                model, blocks, creds=creds, system=FRAMES_SYSTEM_PROMPT,
                max_tokens=cfg.max_tokens, timeout=cfg.timeout)
        except vlm_client.VlmError as exc:
            failed += 1
            out.notes.append("第 %d 批（%.0f~%.0f 秒）没看成：%s"
                             % (index + 1, lo, hi,
                                vlm_client.explain_error(exc)))
            continue
        out.calls += 1
        out.tokens += _token_count(result.usage or {})

        try:
            raw = result.json()
        except vlm_client.VlmError as exc:
            salvaged = watcher.salvage_events(result.text or "")
            if not salvaged:
                out.notes.append("第 %d 批的回答不是合法 JSON（%s）—— 这批跳过。"
                                 % (index + 1, exc))
                continue
            out.notes.append("第 %d 批的回答被截断 —— 捞回前面写完的 %d 段。"
                             % (index + 1, len(salvaged)))
            raw = {"events": salvaged}
        if not isinstance(raw, dict):
            continue

        parsed, drops = parse_events(raw)
        for item in parsed:
            if item.get("t") is None:
                continue
            # 模型偶尔会写这一批范围外的时间 —— 把它**夹回这一批**，
            # 免得一段事件被拼到别的时段去（报告上的时段就全错了）。
            t = float(item["t"])
            if t < lo - 30 or t > hi + 30:
                item["t"] = min(max(t, lo), hi)
                if item.get("t_end") is not None:
                    item["t_end"] = min(max(float(item["t_end"]), lo), hi)
            events.append(item)
        for drop in drops:
            out.notes.append("第 %d 批：%s" % (index + 1, drop))

        deaths.extend(_parse_deaths(raw))
        requests.extend(parse_requests(raw.get("queries")))
        for key in ("hero", "position", "result"):
            value = str(raw.get(key) or "").strip()
            if value and value.lower() not in ("null", "unknown", "none"):
                heads[key].append(value)
        try:
            start = raw.get("match_start")
            if start is not None and match_start is None:
                match_start = float(start)
        except (TypeError, ValueError):
            pass
        summary = str(raw.get("summary") or "").strip() or watcher._stages_summary(raw)
        if summary:
            summaries.append(summary)
            prev_summary = summary

    if not events:
        out.notes.append("抽帧这条路一条事件都没读出来 —— 整局没有分析。")
        return out
    if failed:
        out.notes.append("有 %d 批没看成（其余 %d 批照常用）—— 这一局的时间线有缺口。"
                         % (failed, len(windows) - failed))

    merged = _merge_events(events)
    out.events = _to_watch_events(merged)
    out.deaths = _merge_deaths(deaths)
    out.requests = _merge_requests(requests)
    out.hero = heads["hero"][0] if heads["hero"] else ""
    if not out.hero and hero:
        out.hero = str(hero).strip()
        out.notes.append("英雄用你上传前填的「%s」。" % out.hero)
    out.position = _majority(heads["position"]) or "unknown"
    if out.position in ("", "unknown") and position:
        out.position = str(position).strip()
        out.notes.append("分路用你上传前填的「%s」。" % out.position)
    out.result = _norm_result(heads["result"][-1]) if heads["result"] else "unknown"
    out.summary = " ".join(summaries)[:600]
    out.match_start, how = _resolve_match_start(out.events, match_start)
    watcher.apply_stages(out.events, out.match_start)
    if out.hero:
        for event in out.events:
            event.narrative = watcher.strip_skills(watcher.strip_lane_words(
                watcher.strip_time_refs(watcher.clean_words(event.narrative, out.hero))))
            event.where = watcher.canonical_place(
                watcher.clean_words(event.where, out.hero))
        for death in out.deaths:
            death.cause = watcher.clean_words(death.cause, out.hero)
            death.fault = watcher.clean_words(death.fault, out.hero)
            death.where = watcher.canonical_place(
                watcher.clean_words(death.where, out.hero))
    if out.match_start is not None:
        out.notes.append("对局起点：原视频第 %.0f 秒（%s）。"
                         % (out.match_start, how or "抽帧估计"))

    # -- K/D/A 校验：队友击杀不能算成玩家的 ---------------------------------
    if cfg.verify_events and out.events:
        from . import event_verify

        verified = event_verify.verify_events(
            out.events, video_path, model=model, data_dir=cfg.data_dir,
            transform=transform)
        out.calls += int(verified.get("calls") or 0)
        out.tokens += int(verified.get("tokens") or 0)
        out.notes.extend(verified.get("notes") or [])
        watcher.apply_stages(out.events, out.match_start)
        out.events.sort(key=lambda e: e.video_start)

    # -- 位置不出现在报告里：清空 where（小地图只作内部校验，不外显）-------
    for event in out.events:
        event.where = ""
    for death in out.deaths:
        death.where = ""

    # -- 关键时刻再细看一遍：±30 秒、一秒一帧 ----------------------------
    if cfg.detail_events and out.events:
        from . import event_detail

        detail = event_detail.detail_events(
            out.events, video_path, model=model, creds=creds,
            transform=transform, hero=out.hero or hero,
            span=cfg.detail_span)
        out.calls += int(detail.get("calls") or 0)
        out.tokens += int(detail.get("tokens") or 0)
        out.notes.extend(detail.get("notes") or [])
        if detail.get("calls"):
            out.notes.append("关键时刻按 1 秒 1 帧细看了 %d 段（每段前后各 %.0f 秒）。"
                             % (detail["calls"], cfg.detail_span))
        if detail.get("fixed"):
            out.notes.append("细看时按画面提示帧校正了 %d 个事件时刻。"
                             % detail["fixed"])
        watcher.apply_stages(out.events, out.match_start)
        out.events.sort(key=lambda e: e.video_start)
        if out.hero:
            for event in out.events:
                event.narrative = watcher.clean_place(
                    watcher.clean_words(event.narrative, out.hero))
        lost = [e for e in out.events
                if str(getattr(e, "event_type", "")) == "tower_lost"]
        if lost:
            out.events = [e for e in out.events if e not in lost]
            out.notes.append("本版不输出丢塔：去掉 %d 条。" % len(lost))
    return out


def _to_watch_events(items: Sequence[Dict[str, Any]]) -> List[WatchEvent]:
    """规整后的事件 → :class:`WatchEvent`。"""
    out: List[WatchEvent] = []
    for item in items:
        t = item.get("t")
        if t is None:
            continue
        t_end = item.get("t_end")
        out.append(WatchEvent(
            video_start=float(t),
            video_end=float(t_end if t_end is not None else t),
            narrative=watcher.clean_words(str(item.get("narrative") or "")),
            action=str(item.get("action") or "unclear"),
            action_label=str(item.get("action_label") or ""),
            position=str(item.get("position") or "unknown"),
            lane=str(item.get("lane") or "unknown"),
            where=watcher.clean_words(str(item.get("where") or "")),
            phase=str(item.get("phase") or "unknown"),
            confidence=float(item.get("confidence") or 0.0),
            clock=item.get("clock"),
            stage=str(item.get("stage") or ""),
            event_type=str(item.get("event_type") or ""),
            frames=[float(x) for x in (item.get("frames") or [])],
        ))
    out.sort(key=lambda e: e.video_start)
    return out
