# -*- coding: utf-8 -*-
"""**血条读数**：像素法为主，模型裁剪放大复核。

「模型能不能读出对局里的血量」—— 实测答案是**不能**（2026-11-08）：

- 让模型看整局录像，它每次都答 ``100``（满血），**连亲眼确认只有
  约 20% 血的那一帧也答 100** —— 它是**按满血猜**的，不是读出来的；
- 原因和「认名字」一样：血条是画面中央一条约 95×10 px 的细条，
  视频降采样之后彻底糊掉了。

但**拉近就能读**：把那一块裁出来放大 4 倍再问，模型答得出来
（实测 84 / 14 / 14，与像素量到的 90% / 23% / 23% 吻合）。

于是这一层**两条腿走路**
----------------------

1. **像素法**（主）：血条位置极稳 —— 横屏王者里被跟随英雄的
   「名字 + 血条 + 蓝条」画在英雄身上方，摄像机跟着英雄，
   所以血条在屏幕上基本不动（实测左端 x=596、行 y≈192）。
   量绿条长度 ÷ 条总长 = 血量。**零模型成本，能出全程曲线。**
2. **模型复核**（辅）：只在残血时刻把那一块裁出来放大问一次，
   与像素读数**对账** —— 对不上就留痕，不装作没这回事。

口径（三条，缺一条就会变成编）
------------------------------

1. **别拿「最长的绿段」当满血** —— 玩家全程没满过血时，
   那会把所有读数一起放大。所以用**血条与蓝条各自的 99 分位取大者**：
   两条轨道**总长一样**，只要有一条曾经满过，就能定出总长。
2. **绿色不等于血条** —— 草地也是绿的。所以必须再验一条：
   血条**正下方**紧贴着一条同样长的蓝条（蓝量轨道），
   草下面不会有。找不到蓝条就当这一帧没有血条。
3. **读不出就没有** —— 找不到血条就返回空，报告写「这一局没读到血量」，
   **不拿 0 充数**（0 会被读成「血空了」）。
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: 搜索窗口（相对画面）。血条在画面**中央偏上** —— 摄像机跟着英雄，
#: 名字与血条跟着英雄走，所以血条在屏幕上基本不动。
SEARCH_BOX = (0.28, 0.14, 0.72, 0.52)

#: 血条填充绿：实测 RGB ≈ (80, 190, 51)。
#: 草地更偏黄绿（G-R 只有 30 左右），所以 G-R / G-B 都要 ≥ 45 才算血条。
GREEN_MIN = 130
GREEN_DELTA = 45

#: 蓝量条：实测饱和蓝（用它把血条和草地分开）。
BLUE_MIN = 120
BLUE_DELTA = 40
BLUE_GREEN_DELTA = 25

#: 一段绿至少这么长才算「条」（不是噪点）。
MIN_RUN = 10

#: 残血阈值（百分比）。低于它算「残血时段」。
LOW_PCT = 35.0

#: 两段残血之间隔这么久就并成一段（画面偶尔漏检一两帧）。
EPISODE_GAP = 5.0

#: 残血时段至少要这么长（秒）才留 —— 太短的是一帧的抖动。
EPISODE_MIN_SEC = 2.0

#: 条总长的合理范围（像素）。超出这个范围说明量错了。
BASELINE_MIN = 30
BASELINE_MAX = 220

#: **至少读到这么多帧**才算测过（2 fps 下约 15 秒）。
#:
#: 为什么要有这个闸：只剩三五帧时，血条可能只是碰巧露过一次脸 ——
#: 那时候说「这一局没出现残血」是**把「几乎没测」说成「测了没问题」**。
MIN_SAMPLES = 30

#: **至少这么大比例**的帧读到血条，才算测过。
#:
#: 同一局里死亡结算、选人、暂停这些时候画面上根本没有血条，
#: 所以不是每一帧都该读到；但读到的比例太低，说明多半是**认错了界面**。
MIN_COVERAGE = 0.2

#: 蓝条长度至少要达到条总长的这个比例，才算数。
#:
#: 反草地的闸只看「蓝条起点对不对得上」，一个 8 像素的蓝点也能通过；
#: 而**队友/敌人自己头顶的血条**正好就是「绿条 + 蓝条」这个形状。
#: 比长度能把「蹭到一个小蓝点」这种误检剔掉。
MIN_BLUE_RATIO = 0.25


# ---------------------------------------------------------------- 数据形状


@dataclass
class BarHit:
    """一帧里量到的血条。"""

    #: 这一帧在**扫描的视频**里的秒数。
    t: float = 0.0
    #: 血条左端 x（整段视频里基本不变）。
    x0: int = 0
    #: 血条所在行。
    y: int = 0
    #: 绿色填充长度（像素）。
    green: int = 0
    #: 下方蓝条长度（像素）—— 用来定条的总长。
    blue: int = 0


@dataclass
class HpEpisode:
    """一段残血。"""

    #: 起点 / 终点（扫描视频里的秒）。
    start: float = 0.0
    end: float = 0.0
    #: 最险的那一刻（秒）。
    worst_t: float = 0.0
    #: 最低血量（百分比）。
    min_pct: float = 0.0
    #: 平均血量（百分比）。
    avg_pct: float = 0.0
    #: 最险那一刻的绿条长度（像素）。
    green_px: int = 0
    #: 模型复核读到的血量（没复核就是 None）。
    model_pct: Optional[int] = None
    #: 模型自己给的一句话依据。
    model_note: str = ""
    #: 两个读数**档位**是否一致（None = 没复核）。
    agree: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": round(float(self.start), 2),
            "end": round(float(self.end), 2),
            "worst_t": round(float(self.worst_t), 2),
            "min_pct": round(float(self.min_pct), 1),
            "avg_pct": round(float(self.avg_pct), 1),
            "green_px": int(self.green_px),
            "model_pct": (int(self.model_pct)
                          if self.model_pct is not None else None),
            "model_note": self.model_note,
            "agree": self.agree,
        }


@dataclass
class HpScan:
    """一局的血条读数。"""

    ok: bool = False
    #: 每一帧的读数（t / 绿长 / 百分比）。
    hits: List[BarHit] = field(default_factory=list)
    #: 血条左端（整段视频的锚点）。
    anchor_x: int = 0
    #: 条的总长（像素）—— 绿长 ÷ 它 = 血量。
    baseline_px: int = 0
    #: 总长是「血条」还是「蓝条」定的（两个取大者）。
    baseline_source: str = ""
    #: 一共扫了多少帧。
    sampled: int = 0
    #: 其中读到血条的比例（0~1）。
    coverage: float = 0.0
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------- 像素


def _runs(mask_row: Any, min_len: int = MIN_RUN) -> List[Tuple[int, int]]:
    """一行里的连续 True 段。"""
    out: List[Tuple[int, int]] = []
    start = None
    for x in range(len(mask_row)):
        if mask_row[x]:
            if start is None:
                start = x
        elif start is not None:
            if x - start >= min_len:
                out.append((start, x - 1))
            start = None
    if start is not None and len(mask_row) - start >= min_len:
        out.append((start, len(mask_row) - 1))
    return out


def _blue_below(blue: Any, row: int, x0: int, span: int = 14,
                 need: int = 12) -> int:
    """血条**正下方**几行里，有一条从同一处开始的蓝条吗？返回它的长度。

    这是把血条和「草地上的绿色」区分开的关键 —— 草也会是绿的，
    但草下面不会紧贴着一条同样长的蓝条（那是蓝量轨道）。

    **只看起点**是不够的（一个 8 像素的蓝点也能对上）——
    长度由 :func:`scan_video` 拿条总长再卡一道。
    """
    height = blue.shape[0]
    best = 0
    for r in range(row + 3, min(height, row + 3 + span)):
        for (a, b) in _runs(blue[r], min_len=need):
            # 起点要对得上（允许几像素的错位）。
            if a <= x0 + 8 and b >= x0 - 8:
                best = max(best, b - a + 1)
    return best


def _find_hit(frame: Any, box: Tuple[int, int, int, int],
              t: float) -> Optional[BarHit]:
    """一帧里找血条。找不到返回 None。"""
    import numpy as np

    x0, y0, x1, y1 = box
    band = frame[y0:y1, x0:x1]
    if band.size == 0:
        return None
    b = band[:, :, 0].astype(np.int16)
    g = band[:, :, 1].astype(np.int16)
    r = band[:, :, 2].astype(np.int16)
    green = (g >= GREEN_MIN) & (g - r >= GREEN_DELTA) & (g - b >= GREEN_DELTA)
    blue = ((b >= BLUE_MIN) & (b - r >= BLUE_DELTA)
            & (b - g >= BLUE_GREEN_DELTA))
    best: Optional[BarHit] = None
    for i in range(green.shape[0]):
        for (a, b2) in _runs(green[i]):
            if b2 - a + 1 < MIN_RUN:
                continue
            blue_len = _blue_below(blue, i, a)
            if blue_len <= 0:
                continue
            if best is None or (b2 - a + 1) > best.green:
                best = BarHit(t=t, x0=a + x0, y=i + y0,
                              green=b2 - a + 1, blue=blue_len)
    return best


def _percentile(values: Sequence[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    if len(ordered) == 1:
        return int(ordered[0])
    pos = (len(ordered) - 1) * (q / 100.0)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return int(round(ordered[low] * (1 - frac) + ordered[high] * frac))


def _token_count(usage: Dict[str, Any]) -> int:
    """这次调用花了多少 token（口径与 watcher 一致）。"""
    total = 0
    for key in ("input_tokens", "output_tokens", "prompt_tokens",
                "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, (int, float)):
            total += int(value)
    return total


def scan_video(path: str, *, transform: Optional[Callable[[Any], Any]] = None,
               target_fps: float = 2.0) -> HpScan:
    """扫一遍视频，把每一帧的血条读数取回来。

    ``transform`` 是画面归一（转正 / 裁黑边）—— 扫的是**原片**时才需要；
    扫打包片时不用（打包时已经烙进去了）。
    """
    import cv2

    out = HpScan()
    if not path or not os.path.exists(path):
        out.notes.append("没读到血量：视频文件不在（%s）。" % path)
        return out
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        out.notes.append("没读到血量：这个视频打不开。")
        return out
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            out.notes.append("没读到血量：这个视频没有画面尺寸。")
            return out
        if fps <= 0:
            # **没有帧率就没有时间轴** —— 算出来的「秒」其实是帧号，
            # 残血时段会整体错位（局内时刻全错）。宁可读不出。
            out.notes.append("没读到血量：这个视频读不到帧率，"
                             "算不出残血发生在哪一刻。")
            return out
        # 采样密度：目标 2 fps。高帧率的原片按步长跳着读，
        # 否则 60 fps 的 19 分钟录像要读 6 万多帧。
        stride = 1
        if target_fps > 0:
            stride = max(1, int(round(fps / float(target_fps))))
        # **搜索框按「归一之后」的画面算** —— 转正（旋转）、
        # 裁黑边都会改变尺寸，拿原视频的尺寸算就是**框到别的地方**，
        # 扫半天只会得出「画面里没有血条」。所以第一帧过了归一再定框。
        box: Optional[Tuple[int, int, int, int]] = None
        raw: List[BarHit] = []
        sampled = 0
        transform_failures = 0
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if index % stride == 0:
                sampled += 1
                if transform is not None:
                    try:
                        frame = transform(frame)
                    except Exception:                      # noqa: BLE001
                        transform_failures += 1
                        frame = None
                if frame is not None:
                    if box is None:
                        fh, fw = frame.shape[:2]
                        box = (int(fw * SEARCH_BOX[0]), int(fh * SEARCH_BOX[1]),
                               int(fw * SEARCH_BOX[2]), int(fh * SEARCH_BOX[3]))
                    hit = _find_hit(frame, box, index / fps)
                    if hit is not None:
                        raw.append(hit)
            index += 1
    finally:
        cap.release()

    if transform_failures and transform_failures >= sampled:
        # **归一失败不能当成「画面里没有血条」** —— 那会把编码问题
        # 误诊成「不是横屏王者界面」，读者会去调错的东西。
        out.notes.append("没读到血量：画面归一（转正 / 裁黑边）失败，"
                         "读出来的坐标对不上。")
        return out

    if not raw:
        out.notes.append(
            "没读到血量：画面里找不到被跟随英雄的血条"
            "（这一局可能不是横屏王者界面，或血条被裁掉了）。")
        return out

    # **锚点**：整段视频里血条左端基本不动 —— 用众数把它钉住，
    # 离锚点太远的读数当误检丢掉（草地、别的绿色元素）。
    anchor = Counter(h.x0 for h in raw).most_common(1)[0][0]
    kept = [h for h in raw if abs(h.x0 - anchor) <= 8]
    if len(kept) < 3:
        out.notes.append("没读到血量：血条位置飘忽，没能钉住。")
        return out

    # **条的总长**：血条与蓝条两条轨道**总长一样**，
    # 取两者的 99 分位里更大的那个 —— 玩家全程没满过血时，
    # 蓝条那一条还能顶上来（两条都满过才准）。
    base_green = _percentile([h.green for h in kept], 99)
    base_blue = _percentile([h.blue for h in kept], 99)
    baseline = max(base_green, base_blue)
    if not (BASELINE_MIN <= baseline <= BASELINE_MAX):
        out.notes.append("没读到血量：量出来的血条长度不合理（%d 像素）。"
                         % baseline)
        return out

    # **蓝条长度要对得上条总长** —— 只对上起点不够：一个蹭到
    # 小蓝点的误检也得剔掉（队友头顶那条「血条 + 蓝条」形状更真）。
    min_blue = max(MIN_RUN, int(baseline * MIN_BLUE_RATIO))
    trimmed = [h for h in kept if h.blue >= min_blue]
    if len(trimmed) >= 3:
        kept = trimmed
        base_green = _percentile([h.green for h in kept], 99)
        base_blue = _percentile([h.blue for h in kept], 99)
        baseline = max(base_green, base_blue)
        if not (BASELINE_MIN <= baseline <= BASELINE_MAX):
            out.notes.append("没读到血量：量出来的血条长度不合理（%d 像素）。"
                             % baseline)
            return out

    # **测过没有** —— 只有三五帧不算测过。
    #
    # 少了这道闸，「血条只露过三次脸」也会被写成
    # 「这一局没出现残血（血量一直在 35% 以上）」——
    # **把「几乎没测」说成「测了没问题」**，正好是口径第三条禁止的。
    coverage = (float(len(kept)) / float(sampled)) if sampled else 0.0
    if not _enough_samples(len(kept), sampled):
        out.notes.append(
            "没读到血量：整局只读到 %d 帧血条（%d 帧里的 %.0f%%），"
            "太少，判不了残血。" % (len(kept), sampled, coverage * 100.0))
        return out

    out.ok = True
    out.hits = kept
    out.anchor_x = int(anchor)
    out.baseline_px = int(baseline)
    out.baseline_source = ("血条与蓝条" if base_green >= base_blue else "蓝条")
    out.sampled = sampled
    out.coverage = coverage
    return out


def _pct(green: int, baseline: int) -> float:
    if baseline <= 0:
        return 0.0
    return max(0.0, min(100.0, 100.0 * float(green) / float(baseline)))


def build_episodes(scan: HpScan, low_pct: float = LOW_PCT) -> List[HpEpisode]:
    """从逐帧读数里切出**残血时段**。

    相邻两段之间隔几秒的算同一段（画面偶尔漏检一两帧）；
    短于 ``EPISODE_MIN_SEC`` 的丢掉（一帧抖动不算一段残血）。
    """
    if not scan.ok or not scan.hits:
        return []
    samples = sorted(scan.hits, key=lambda h: h.t)
    groups: List[List[BarHit]] = []
    current: List[BarHit] = []
    for hit in samples:
        pct = _pct(hit.green, scan.baseline_px)
        if pct < low_pct:
            if current and hit.t - current[-1].t > EPISODE_GAP:
                groups.append(current)
                current = []
            current.append(hit)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    out: List[HpEpisode] = []
    for group in groups:
        if not group:
            continue
        if group[-1].t - group[0].t < EPISODE_MIN_SEC:
            continue
        worst = min(group, key=lambda h: h.green)
        pcts = [_pct(h.green, scan.baseline_px) for h in group]
        out.append(HpEpisode(
            start=group[0].t, end=group[-1].t, worst_t=worst.t,
            min_pct=min(pcts), avg_pct=sum(pcts) / len(pcts),
            green_px=worst.green,
        ))
    out.sort(key=lambda e: e.start)
    return out


# ---------------------------------------------------------------- 模型复核


_CONFIRM_PROMPT = (
    "下面几张图是王者对局录屏里**同一块画面裁出来放大的**"
    "（被跟随英雄身上那一块）。\n"
    "每一张里都有一个英雄（玩家自己，被摄像机跟随），"
    "他**名字正下方**有一条细长的横条 —— 那是血条：\n"
    "   * 整条都是绿的 = 满血；\n"
    "   * 绿色只占一部分、其余是黑的 = 残血"
    "（绿色占整条的几分之几，就报几）；\n"
    "   * 整条都是黑的（没有绿色）= 血量空了。\n"
    "按图片顺序回答一个 JSON：\n"
    '{"frames": [{"i": 1, "hp_pct": 0~100 的整数 或 null, '
    '"note": "一句话依据（那条横条绿了多少 / 是不是全黑）"}]}\n'
    "**看不清就写 null**；条是黑的就写 0。"
)


def _parse_frames(text: str) -> List[Dict[str, Any]]:
    """从模型回答里抠出 ``frames`` 数组。抠不出就返回空。

    **别用贪婪的 ``\\{.*\\}``** —— 回答里只要多一段带括号的话
    （模型很爱写「（整条都是黑的）」），贪婪匹配会把尾巴一起吞进来，
    JSON 直接解析失败 —— 复核就这么没了，而且**一声不响**。
    """
    raw = str(text or "")
    if not raw.strip():
        return []
    for candidate in _json_candidates(raw):
        try:
            data = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        frames = data.get("frames") if isinstance(data, dict) else None
        if isinstance(frames, list):
            return [f for f in frames if isinstance(f, dict)]
    return []


def _json_candidates(text: str) -> List[str]:
    """从一段话里挑出**可能**是 JSON 的片段（从大到小试）。"""
    out: List[str] = []
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_str = False
        escape = False
        for index in range(start, len(text)):
            ch = text[index]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[start:index + 1])
                    break
        start = text.find("{", start + 1)
    return out


def _crop_around_bar(frame: Any, hit: BarHit, zoom: int = 4) -> bytes:
    """把血条那一块裁出来放大（模型读得清的关键）。

    裁剪范围：血条左端往左 200 px、往右 200 px，条所在行往上 40 px
    （名字就在正上方）、往下 25 px —— 名字与条都在画面里。
    """
    import cv2

    height, width = frame.shape[:2]
    x0 = max(0, hit.x0 - 200)
    x1 = min(width, hit.x0 + 200)
    y0 = max(0, hit.y - 40)
    y1 = min(height, hit.y + 25)
    if x1 - x0 < 40 or y1 - y0 < 12:
        return b""
    crop = frame[y0:y1, x0:x1]
    big = cv2.resize(crop, None, fx=zoom, fy=zoom,
                     interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", big, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return buf.tobytes() if ok else b""


def confirm_with_model(path: str, scan: HpScan,
                       episodes: Sequence[HpEpisode], *,
                       data_dir: str = "", max_shots: int = 4,
                       timeout: float = 180.0,
                       model: str = "",
                       transform: Optional[Callable[[Any], Any]] = None
                       ) -> Dict[str, Any]:
    """让模型复核最险的那几段残血 —— **一次调用问完**。

    为什么不逐段问：每段一次调用太贵（一次就是一次模型往返），
    而这几张图都是同一个问题，可以一次发过去。

    ``transform`` 必须和扫描时用的是**同一个** —— 扫描是在归一后的
    画面上找血条的，裁剪也得从归一后的画面上裁，否则坐标对不上
    （裁出来的是别的地方，模型读的是另一块画面）。
    """
    import cv2

    from ..collection import vlm_client

    result: Dict[str, Any] = {"calls": 0, "tokens": 0, "shots": [],
                               "notes": []}
    if not episodes:
        return result

    # 挑最险的几段（按最低血量），再按时间排回去。
    picked = sorted(episodes, key=lambda e: e.min_pct)[:max(1, int(max_shots))]
    picked.sort(key=lambda e: e.start)

    by_time = {round(h.t, 1): h for h in scan.hits}
    images: List[bytes] = []
    targets: List[HpEpisode] = []
    for episode in picked:
        hit = by_time.get(round(episode.worst_t, 1))
        if hit is None:
            hit = min(scan.hits, key=lambda h: abs(h.t - episode.worst_t),
                      default=None)
        if hit is None:
            continue
        ok = False
        frame = None
        cap = cv2.VideoCapture(path)
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(episode.worst_t) * 1000.0)
            ok, frame = cap.read()
        except Exception:                                  # noqa: BLE001
            ok = False
            frame = None
        finally:
            cap.release()
        if not ok or frame is None:
            continue
        if transform is not None:
            try:
                frame = transform(frame)
            except Exception:                              # noqa: BLE001
                continue
        if frame is None:
            continue
        data = _crop_around_bar(frame, hit)
        if not data:
            continue
        images.append(data)
        targets.append(episode)
    if not images:
        result["notes"].append("血量复核没做成：截不到残血那一刻的画面。")
        return result

    creds = vlm_client.find_credentials("vlm", data_dir or None)
    if creds is None:
        result["notes"].append("血量复核没做成：没配「vlm」槽位的模型。")
        return result
    blocks = [vlm_client.text_block(_CONFIRM_PROMPT)]
    for data in images:
        blocks.append(vlm_client.image_block(data))
    try:
        answer = vlm_client.call_model(
            model or creds.model or "MiniMax-M2", blocks, creds=creds,
            max_tokens=1200, timeout=timeout)
    except vlm_client.VlmError as exc:
        result["notes"].append("血量复核没做成：%s" % vlm_client.explain_error(exc))
        return result
    result["calls"] = 1
    result["tokens"] = _token_count(answer.usage or {})

    frames = _parse_frames(answer.text)
    if not frames:
        # **模型答了但抠不出答案** —— 不留痕的话，读者以为
        # 「像素 + 模型双保险」验过了，其实模型那腿根本没成。
        result["notes"].append("血量复核没做成：模型没给出能读的回答。")
        return result

    # **按模型自己标的 ``i`` 配对** —— 不能按顺序配。
    by_index = _by_index(frames)

    for index, episode in enumerate(targets, 1):
        entry: Dict[str, Any] = {"episode": episode.start,
                                  "pixel_pct": round(episode.min_pct, 1)}
        entry.update(_shot_entry(by_index.get(index), episode.min_pct))
        episode.model_pct = entry["model_pct"]
        episode.model_note = entry["note"]
        episode.agree = entry["agree"]
        result["shots"].append(entry)

    if not any(s.get("model_pct") is not None for s in result["shots"]):
        result["notes"].append("血量复核没做成：模型看不清那几条血条。")
    return result


def _same_band(model_pct: int, pixel_pct: float) -> bool:
    """两个读数是不是落在同一档（残血 / 不残血）。"""
    band = lambda v: v < LOW_PCT                            # noqa: E731
    return band(model_pct) == band(pixel_pct)


def _by_index(frames: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """按模型自己标的 ``i`` 归位（**不按顺序**）。

    少答一张、或者把顺序答反的时候，按顺序配会把读数**整段错位**：
    90% 挂到第 1 段、8% 挂到第 2 段，与模型本意相反 ——
    「对账」这一层就白做了。
    """
    out: Dict[int, Dict[str, Any]] = {}
    for item in frames or []:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("i"))
        except (TypeError, ValueError):
            continue
        if idx not in out:
            out[idx] = item
    return out


def _shot_entry(item: Optional[Dict[str, Any]],
                pixel_pct: float) -> Dict[str, Any]:
    """一张图的对账结果。

    **越界的读数不算读数** —— 模型答 150% 或 -5% 时，
    那个数既不能写进「模型读到几成」，也不该被算成「不一致」
    （否则一致性统计会被垃圾读数污染）。
    """
    entry: Dict[str, Any] = {"model_pct": None, "note": "", "agree": None}
    if item is None:
        entry["note"] = "模型没回答这一张"
        return entry
    value = item.get("hp_pct")
    try:
        pct = int(value)
    except (TypeError, ValueError):
        pct = None
    if pct is not None and 0 <= pct <= 100:
        entry["model_pct"] = pct
        # **按档位比，不按数字比** —— 模型读这种细条本来就
        # 偏保守（实测低 5~9 个点），比数字会天天报「不一致」。
        entry["agree"] = _same_band(pct, pixel_pct)
    entry["note"] = str(item.get("note") or "")[:120]
    return entry


def _enough_samples(kept: int, sampled: int) -> bool:
    """**测过没有** —— 帧数太少不算测过。

    少了这道闸，「血条只露过三次脸」也会被写成
    「这一局没出现残血」—— **把「几乎没测」说成「测了没问题」**。
    """
    if sampled <= 0 or kept <= 0:
        return False
    if float(kept) / float(sampled) < MIN_COVERAGE:
        return False
    if kept < MIN_SAMPLES and sampled > MIN_SAMPLES:
        return False
    return True


# ---------------------------------------------------------------- 汇总


def analyze(path: str, *, transform: Optional[Callable[[Any], Any]] = None,
            data_dir: str = "", confirm: bool = True,
            match_start: Optional[float] = None, time_scale: float = 1.0,
            source_start: float = 0.0,
            max_shots: int = 4, low_pct: float = LOW_PCT,
            model: str = "") -> Dict[str, Any]:
    """读一局的血量，产出可以直接落库的形状。

    ``match_start`` 是对局在**原视频**里开始的秒数 —— 有了它才能说
    「局内 11:26」；``time_scale`` 是打包视频 → 原视频的时间比例，
    ``source_start`` 是打包片在原视频里的起点（切片时不为 0）。
    """
    out: Dict[str, Any] = {
        "ok": False,
        "samples": 0,
        "baseline_px": 0,
        "anchor_x": 0,
        "low_pct": float(low_pct),
        "low_seconds": 0.0,
        "episodes": [],
        "confirmed": 0,
        "disagreed": 0,
        "notes": [],
    }
    scan = scan_video(path, transform=transform)
    out["notes"].extend(scan.notes)
    if not scan.ok:
        return out

    episodes = build_episodes(scan, low_pct=low_pct)
    out["ok"] = True
    out["samples"] = len(scan.hits)
    out["scanned"] = int(scan.sampled)
    out["coverage"] = round(float(scan.coverage), 3)
    out["baseline_px"] = int(scan.baseline_px)
    out["baseline_source"] = scan.baseline_source
    out["anchor_x"] = int(scan.anchor_x)
    out["low_seconds"] = round(
        sum(max(0.0, e.end - e.start) for e in episodes), 1)

    if confirm and episodes:
        check = confirm_with_model(path, scan, episodes, data_dir=data_dir,
                                   max_shots=max_shots, model=model,
                                   transform=transform)
        out["notes"].extend(check.get("notes") or [])
        out["confirm_calls"] = int(check.get("calls") or 0)
        out["confirm_tokens"] = int(check.get("tokens") or 0)
        out["confirmed"] = sum(1 for s in check.get("shots") or []
                               if s.get("agree") is True)
        out["disagreed"] = sum(1 for s in check.get("shots") or []
                               if s.get("agree") is False)
        if out["disagreed"]:
            out["notes"].append(
                "血量复核：%d 段两个读数不在同一档（像素说残血、模型说不残血，"
                "或反过来）—— 已按像素读数记，模型的说法留在明细里。"
                % out["disagreed"])

    scale = float(time_scale or 1.0)
    offset = float(source_start or 0.0)
    for episode in episodes:
        item = episode.to_dict()
        source_t = float(episode.start) * scale + offset
        item["video_start"] = round(source_t, 2)
        # **时长也要按比例换算** —— 写「有 14 秒都在 35% 以下」
        # 说的是**原视频**的秒数，不是打包片里的。
        item["duration"] = round((float(episode.end) - float(episode.start))
                                 * scale, 1)
        if match_start is not None:
            game_t = max(0.0, source_t - float(match_start))
            item["clock"] = "%d:%02d" % (int(game_t) // 60, int(game_t) % 60)
            item["minute"] = round(game_t / 60.0, 2)
            game_end = max(0.0, float(episode.end) * scale + offset
                           - float(match_start))
            item["clock_end"] = "%d:%02d" % (int(game_end) // 60,
                                             int(game_end) % 60)
        out["episodes"].append(item)
    if not episodes:
        out["notes"].append(
            "这一局没出现残血（血量一直在 %d%% 以上；整局读到 %d 帧血条，"
            "占 %d%%）。" % (int(low_pct), len(scan.hits),
                             int(round(float(scan.coverage) * 100.0))))
    return out


# ---------------------------------------------------------------- 报告条目


def _split_clauses(text: str) -> List[str]:
    """一句一句 —— 建议挤成一段，条件与做法就糊在一起了。"""
    out: List[str] = []
    for part in re.split(r"(?<=。)", str(text or "")):
        part = part.strip()
        if part:
            out.append(part)
    return out


def _hp_title(episode: Dict[str, Any]) -> str:
    """条目标题：哪段时间 · 血量最低到多少。

    血量读数是**结论**（量出来的），所以留在标题里 ——
    模型的白描才不上屏。
    """
    clock = str(episode.get("clock") or "")
    clock_end = str(episode.get("clock_end") or "")
    if clock and clock_end:
        when = "%s ~ %s" % (clock, clock_end)
    elif clock:
        when = clock
    else:
        when = "局内时间没读到"
    try:
        low = int(round(float(episode.get("min_pct") or 0.0)))
    except (TypeError, ValueError):
        low = 0
    return "%s · 血量最低 %d%%" % (when, low)


def _hp_reason(episode: Dict[str, Any], low_pct: float) -> str:
    """「为什么有用」—— 攻略要 X，你这局是 Y（**对照着说**）。"""
    guide = episode.get("guide") or {}
    parts: List[str] = []
    premise = str(guide.get("premise") or "").strip()
    outcome = str(guide.get("outcome") or "").strip()
    if premise:
        parts.append("攻略在这个节点的前提是「%s」" % premise)
    if outcome:
        parts.append("照着做会得到「%s」" % outcome)
    try:
        low = int(round(float(episode.get("min_pct") or 0.0)))
    except (TypeError, ValueError):
        low = 0
    try:
        duration = int(round(float(episode.get("duration") or 0.0)))
    except (TypeError, ValueError):
        duration = 0
    parts.append("你这段血量最低只剩 %d%%，有 %d 秒都在 %d%% 以下"
                 % (low, duration, int(low_pct)))
    model_pct = episode.get("model_pct")
    if model_pct is not None:
        try:
            value = int(model_pct)
        except (TypeError, ValueError):
            value = None
        if value is not None:
            if episode.get("agree") is False:
                parts.append("拉近复核的模型读到 %d%%，跟像素读数不在同一档"
                             % value)
            else:
                parts.append("拉近复核的模型也读到 %d%%，两个读数对得上"
                             % value)
    return "；".join(parts) + "。"


def build_entries(episodes: Sequence[Dict[str, Any]], *,
                  low_pct: float = LOW_PCT,
                  video_url: str = "") -> List[Dict[str, Any]]:
    """把残血时段做成报告正文里的条目（每段一条）。

    **只留三样**：时间区间 / 该做什么（模式库原文）/ 为什么有用 ——
    模型写的白描不上屏（用户拍板），血量读数是量出来的结论，留在标题与理由里。
    """
    out: List[Dict[str, Any]] = []
    for index, episode in enumerate(episodes or [], 1):
        if not isinstance(episode, dict):
            continue
        guide = episode.get("guide") or {}
        advice = str(guide.get("advice") or "").strip()
        if not advice:
            if episode.get("guide_searched"):
                advice = "这个节点没查到可对照的攻略。"
            else:
                # **没查过 ≠ 查了没找到** —— 这次只对照了最险的几段，
                # 说「没查到」等于替检索结果撒谎（2026-11-12 复核发现）。
                advice = "这一段没做攻略对照（这次只对照了最险的几段）。"
        when = ""
        clock = str(episode.get("clock") or "")
        clock_end = str(episode.get("clock_end") or "")
        if clock and clock_end:
            when = "%s ~ %s" % (clock, clock_end)
        elif clock:
            when = clock
        out.append({
            #: 条目类型 —— 报告正文的三段式靠它分清「明细」与「模式判定」
            #: （2026-11-13：正文只留明细，规则命中不再上屏）。
            "kind": "hp",
            "mode_id": "hp-%d" % index,
            "category": "血线",
            "title": _hp_title(episode),
            "lane_label": "",
            "phase_label": "",
            "verdict_label": "这里可以调整",
            "time_range": when,
            "clip": {"video_url": video_url, "moments": [], "seconds": 0},
            # **模型写的白描不上前端**（用户拍板）。
            "actual_text": "",
            "evidence": [],
            "absent_kinds": [],
            "advice_text": advice,
            "advice_lines": _split_clauses(advice),
            "reason_text": _hp_reason(episode, low_pct),
            "low_confidence": False,
            "confidence": 1.0,
        })
    return out
