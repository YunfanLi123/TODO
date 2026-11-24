# -*- coding: utf-8 -*-
"""**读顶栏的 K/D/A，把「死没死」变成事实**（2026-11-13）。

为什么要有这一层
---------------
用户的要求是「分析这次死亡对局内有益还是有害」—— 这句话**必须有事实支撑**：

* 这一局到底死了几次？
* 死之前 45 秒我方有没有拿到人头？
* 死了之后对方有没有拿到资源？

而原来的链条里，**死几次是模型自己数的**：实测那一局真值 3 次，
模型报了 2 次；而「换到了什么」只能靠模型看着画面猜。

怎么做的（不是问模型「你死了几次」）
----------------------------------
参照高星项目的做法（OpenDota 那套：**先把回放解析成结构化事件，再在事件上做分析**），
这里把「读数字」和「下判断」拆开：

    抽帧 → 裁出顶栏右上角那一小块 → **只让模型做一件事：照抄三个数字**
         → 代码按「死亡数 +1」推出**死亡时刻**、按「击杀数 +1」推出**我方 kills**
         → 死亡复盘拿着这份事实去判「赚还是亏」

第二行是关键：**模型只负责读字，事件由代码算**。
读字是它擅长的（README §32：拉近之后 5 个昵称全读对），
而「数数、算增量」交给代码 —— 那部分不该有概率。

读不出来怎么办
--------------
逐张写 null，整段作废：报告里说「这一项没读到」，**不拿缺的数字推事件**。
数错死亡次数比没数更糟（报告会指着不存在的那次死亡讲道理）。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import frames_watch

#: 顶栏右上角 K/D/A 的位置（相对整帧；1280×574 实测标定）。
#:
#: 三个数字在 x≈0.74~0.83、y≈0.03~0.07（1280×574 下是 950~1060 × 18~42）。
#: 框给宽一点：两个数字、三位数都装得下，多出来的边不影响模型读。
KDA_BOX: Tuple[float, float, float, float] = (0.715, 0.015, 0.855, 0.090)

#: 裁出来放大几倍（数字原高 ~13px，×4 之后模型读得稳）。
ZOOM = 4

#: 一次调用塞几张（拼成一张竖图）。
#:
#: 一张 4 倍放大的裁图约 180×50；12 张竖着拼 ≈ 180×700，模型一眼能扫完。
PER_CALL = 12

#: 采样间隔（秒）。30 秒一次：死亡/击杀的时刻误差 ≤30 秒，够判「前后 45 秒」。
STEP_SEC = 30.0

#: 一局最多采几个点（19 分钟 ≈ 38 个）。
MAX_SAMPLES = 48

PROMPT = (
    "下面是同一局《王者荣耀》录像里 {n} 个时刻的**顶栏右上角**裁图（竖着排，"
    "每张上面标着它在原视频的第几秒）。\n"
    "那一小块里印着**玩家自己的三组数字**：击杀 / 死亡 / 助攻"
    "（每组数字左边有一个小图标：剑=击杀、骷髅=死亡、握手=助攻）。\n\n"
    "请**逐张照抄**这三个数字。只输出一个 JSON：\n"
    '{{"readings": [{{"i": 1, "k": 2, "d": 0, "a": 1}}, ...]}}\n'
    "规则：\n"
    "1. **只照抄你看到的**；看不清就写 null（不要猜、不要沿用上一张的数字）。\n"
    "2. 数字可能是一位也可能是两位，**别把图标当成数字**。\n"
    "3. i 就是图上标的序号，一张都不能漏。"
)


@dataclass
class KdaSample:
    """某一个时刻读到的 K/D/A。"""

    t: float = 0.0
    k: Optional[int] = None
    d: Optional[int] = None
    a: Optional[int] = None

    @property
    def ok(self) -> bool:
        return self.k is not None and self.d is not None and self.a is not None

    def to_dict(self) -> Dict[str, Any]:
        return {"t": round(float(self.t), 1), "k": self.k, "d": self.d,
                "a": self.a}


@dataclass
class KdaResult:
    """整局的 K/D/A 读数 + 由它推出来的事件。"""

    samples: List[KdaSample] = field(default_factory=list)
    #: 推出来的事件：``[{"kind": "death", "t": 320.0}, {"kind": "kill", ...}]``
    events: List[Dict[str, Any]] = field(default_factory=list)
    ok: bool = False
    notes: List[str] = field(default_factory=list)
    calls: int = 0
    tokens: int = 0

    @property
    def deaths(self) -> List[Dict[str, Any]]:
        return [e for e in self.events if e.get("kind") == "death"]

    @property
    def kills(self) -> List[Dict[str, Any]]:
        return [e for e in self.events if e.get("kind") == "kill"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "samples": [s.to_dict() for s in self.samples],
            "events": [dict(e) for e in self.events],
            "deaths_detected": len(self.deaths),
            "kills_detected": len(self.kills),
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------- 采样与裁图


def sample_times(start: float, duration: float,
                 step: float = STEP_SEC,
                 limit: int = MAX_SAMPLES) -> List[float]:
    """从对局起点开始，隔 ``step`` 秒取一个点。"""
    start = max(0.0, float(start or 0.0))
    duration = float(duration or 0.0)
    if duration <= start:
        return []
    step = max(10.0, float(step))
    times: List[float] = []
    t = start
    while t < duration and len(times) < limit:
        times.append(round(t, 1))
        t += step
    return times


def crop_hud(frame: Any) -> Any:
    """裁出顶栏 K/D/A 那一块并放大（返回 BGR 图）。"""
    import cv2

    if frame is None or getattr(frame, "size", 0) <= 0:
        return None
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = KDA_BOX
    crop = frame[int(y0 * h):max(int(y0 * h) + 1, int(y1 * h)),
                 int(x0 * w):max(int(x0 * w) + 1, int(x1 * w))]
    if crop.size == 0:
        return None
    big = cv2.resize(crop, (crop.shape[1] * ZOOM, crop.shape[0] * ZOOM),
                     interpolation=cv2.INTER_CUBIC)
    return big


def _tile(shots: Sequence[Tuple[float, Any]]) -> Any:
    """把若干张裁图竖着拼成一张（每张上面写「第几秒」）。"""
    import cv2
    import numpy as np

    tiles = []
    for t, img in shots:
        if img is None:
            continue
        label = np.zeros((26, img.shape[1], 3), np.uint8)
        cv2.putText(label, "%.0fs" % t, (4, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([label, img]))
    if not tiles:
        return None
    width = max(t.shape[1] for t in tiles)
    rows = []
    for tile in tiles:
        if tile.shape[1] < width:
            pad = np.zeros((tile.shape[0], width - tile.shape[1], 3), np.uint8)
            tile = np.hstack([tile, pad])
        rows.append(tile)
    return np.vstack(rows)


# ---------------------------------------------------------------- 读


def _encode(img: Any) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return bytes(buf) if ok else b""


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value if value is not None else "").strip()
    if not text or text.lower() in ("null", "none", "unknown", "?", "-"):
        return None
    m = re.search(r"-?\d+", text)
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


def parse_readings(raw: Any, times: Sequence[float]) -> List[KdaSample]:
    """把模型回答解析成 K/D/A 采样。**缺项就是 None**（不猜、不沿用）。"""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    if isinstance(raw, dict):
        raw = raw.get("readings")
    if not isinstance(raw, list):
        return []
    out: List[KdaSample] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        index = _as_int(item.get("i") or item.get("index"))
        if index is None or index < 1 or index > len(times):
            continue
        sample = KdaSample(
            t=float(times[index - 1]),
            k=_as_int(item.get("k")),
            d=_as_int(item.get("d")),
            a=_as_int(item.get("a")))
        out.append(sample)
    out.sort(key=lambda s: s.t)
    return out


def derive_events(samples: Sequence[KdaSample]) -> List[Dict[str, Any]]:
    """按「数字变大」推出事件。**只比较相邻两个都读到的采样。**

    * 死亡数 +1 → 一次死亡（时刻取后一个采样的时间，误差 ≤ 采样间隔）；
    * 击杀数 +1 → 我方拿到一个人头（用来判「这次死亡换到了没有」）。

    采样点之间的差值大于 1 时按增量补齐（30 秒里连死两次也记得住），
    但时刻只能落在后一个采样点上 —— 所以每条事件都带 ``tolerance``，
    下游看到的是「这个时刻附近」。
    """
    out: List[Dict[str, Any]] = []
    prev: Optional[KdaSample] = None
    for sample in sorted(samples, key=lambda s: s.t):
        if not sample.ok:
            continue
        if prev is not None:
            for kind, field_name, delta_label in (
                    ("death", "d", "死亡数"), ("kill", "k", "击杀数")):
                before = getattr(prev, field_name) or 0
                now = getattr(sample, field_name) or 0
                if now > before:
                    for n in range(int(now - before)):
                        out.append({
                            "kind": kind,
                            "t": float(sample.t),
                            "t_from": float(prev.t),
                            "delta": n + 1,
                            "counter": delta_label,
                            "tolerance": float(max(1.0, sample.t - prev.t)),
                        })
        prev = sample
    return out


def cross_check(model_deaths: Sequence[Dict[str, Any]],
                derived: Sequence[Dict[str, Any]],
                tolerance: float = 45.0) -> Dict[str, Any]:
    """模型报的死亡 vs HUD 推出来的死亡，对一遍。

    两种错都要说清楚：

    * **模型漏报**（HUD 有、模型没写）—— 复盘少写一次死亡；
    * **模型多报**（模型写了、HUD 没有）—— 报告在讲一次没发生过的死亡。

    对不上的时候**不自动改谁的数**：报告写明两边各是几次，
    让读者自己核（一次多报比一次漏报复盘更伤人）。
    """
    derived = [d for d in (derived or []) if str(d.get("kind")) == "death"]
    model = [dict(m) for m in (model_deaths or []) if isinstance(m, dict)]
    used = set()
    matched = 0
    for item in model:
        try:
            t = float(item.get("video_start") or 0.0)
        except (TypeError, ValueError):
            continue
        best = None
        for i, d in enumerate(derived):
            if i in used:
                continue
            if abs(float(d.get("t") or 0.0) - t) <= tolerance:
                best = i
                break
        if best is not None:
            used.add(best)
            matched += 1
    missing = [d for i, d in enumerate(derived) if i not in used]
    extra = []
    for item in model:
        try:
            t = float(item.get("video_start") or 0.0)
        except (TypeError, ValueError):
            continue
        if not any(abs(float(d.get("t") or 0.0) - t) <= tolerance
                   for d in derived):
            extra.append(item)
    return {
        "model_count": len(model),
        "hud_count": len(derived),
        "matched": matched,
        "missed_by_model": [dict(d) for d in missing],
        "not_on_hud": extra,
        "ok": len(model) == len(derived) and not extra,
    }


# ---------------------------------------------------------------- 编排


def read_series(video_path: str, times: Sequence[float], *,
                slot: str = "vlm", model: str = "", data_dir: str = "",
                transform: Any = None, per_call: int = PER_CALL,
                max_tokens: int = 1500,
                timeout: float = 300.0) -> KdaResult:
    """按时刻读整局的 K/D/A。**读不到就空着**（不推事件）。"""
    from app.collection import vlm_client

    out = KdaResult()
    if not times:
        out.notes.append("没给采样时刻 —— 没读 K/D/A。")
        return out
    try:
        creds = vlm_client.find_credentials(slot, data_dir or None)
        if not model:
            from app.collection import settings
            model = settings.model_name(slot)
    except Exception as exc:                           # noqa: BLE001
        out.notes.append("取不到模型配置：%s" % exc)
        return out
    if not creds or not model:
        out.notes.append("没有配置「%s」槽位的模型 —— 没读 K/D/A。" % slot)
        return out
    if not os.path.isfile(video_path):
        out.notes.append("视频不存在：%s" % video_path)
        return out

    frames = frames_watch.read_frames(video_path, list(times), transform)
    if not frames:
        out.notes.append("一张画面都没取到 —— 没读 K/D/A。")
        return out
    shots = [(f.t, crop_hud_cv(f.jpeg)) for f in frames]
    shots = [(t, img) for t, img in shots if img is not None]
    if not shots:
        out.notes.append("裁不出顶栏那一块 —— 没读 K/D/A。")
        return out

    read_any = False
    for i in range(0, len(shots), max(1, int(per_call))):
        chunk = shots[i:i + per_call]
        sheet = _tile(chunk)
        if sheet is None:
            continue
        prompt = PROMPT.format(n=len(chunk))
        blocks = [vlm_client.text_block(prompt),
                  vlm_client.image_block(_encode(sheet))]
        try:
            result = vlm_client.call_model(
                model, blocks, creds=creds, max_tokens=max_tokens,
                timeout=timeout)
        except vlm_client.VlmError as exc:
            out.notes.append("这批 K/D/A 没读到（%s）：%s"
                             % ("第 %d 组" % (i // max(1, per_call) + 1),
                                vlm_client.explain_error(exc)))
            continue
        out.calls += 1
        try:
            usage = result.usage or {}
            out.tokens += sum(int(usage.get(key) or 0) for key in
                              ("input_tokens", "output_tokens",
                               "prompt_tokens", "completion_tokens",
                               "total_tokens"))
        except Exception:                              # noqa: BLE001
            pass
        try:
            raw = result.json()
        except vlm_client.VlmError as exc:
            out.notes.append("K/D/A 那批回答不是合法 JSON（%s）" % exc)
            continue
        samples = parse_readings(raw, [t for t, _ in chunk])
        if samples:
            read_any = True
            out.samples.extend(samples)

    if not read_any:
        out.notes.append("K/D/A 一个数都没读到 —— 这一局没有用它推事件。")
        return out
    out.samples.sort(key=lambda s: s.t)
    out.events = derive_events(out.samples)
    out.ok = bool(out.events) or bool([s for s in out.samples if s.ok])
    out.notes.append(
        "K/D/A：读了 %d 个时刻（%d 个读全），推出 %d 次死亡、%d 个人头。"
        % (len(out.samples), len([s for s in out.samples if s.ok]),
           len(out.deaths), len(out.kills)))
    return out


def crop_hud_cv(jpeg: bytes) -> Any:
    """把 JPEG 字节解成图再裁顶栏（``read_frames`` 给的是字节）。"""
    import cv2
    import numpy as np

    if not jpeg:
        return None
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return crop_hud(img)


def window_facts(events: Sequence[Dict[str, Any]], at: float,
                 before: float = 45.0, after: float = 45.0) -> Dict[str, int]:
    """某一刻前后各 ``before/after`` 秒里，我方拿到几个人头、死几次。"""
    kills = 0
    deaths = 0
    for e in events or []:
        try:
            t = float(e.get("t") or 0.0)
        except (TypeError, ValueError):
            continue
        if t < at - before or t > at + after:
            continue
        if str(e.get("kind")) == "kill":
            kills += 1
        elif str(e.get("kind")) == "death":
            deaths += 1
    return {"kills": kills, "deaths": deaths}
