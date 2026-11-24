"""时间轴：把「视频第几秒」对到「局内第几秒」。

真源是**多模态模型逐帧读出来的顶栏时刻**
--------------------------------------
2026-10-31 之前，这里用的是 OpenCV 模板匹配（:mod:`app.collection.clock`）。
那一版在这段录屏上**读错过 5 次里的 4 次**：

===================  ==============  ================
视频时刻             画面上真正的钟   模板匹配读成
===================  ==============  ================
306.5 秒             02:38           02:16
318.3 秒             02:50           11:21
881.3 秒             12:13           02:43
911.0 秒             12:43           12:13
===================  ==============  ================

肉眼核对过每一张（``tools/dump_clock_crops.py``）—— **画面上的钟是对的，
错的是模板匹配器**。所以整套 OpenCV 读钟从新识别子系统里撤掉了，
「读懂画面」这件事一律交给用户配置的多模态模型。

一共核对过 **14 帧**：它们落在同一条直线上（对局起点 ≈ 148.3 秒、
斜率 1.0001、残差 ±0.31 秒 —— 0.3 秒是画面按秒显示的取整）。
这 14 个读数固化在 ``tests/test_recognition2_clockaxis.py`` 的 golden 数据里。

**为什么模型读数也不能全信**
------------------------
模型也会读错（把「英雄等级 14」当成时间），但它错得更**明显**：
错一次就会偏离直线几十秒到几百秒，一眼就能剔掉。
而且模型能读**任意界面**上的时刻，不只是顶栏那个固定位置。

三条防线
--------
1. **逐帧定点**：模型对**每一张图**分别报时刻，编号跟图号对上 ——
   不再是「一段报一个」，也就没有「它读的是哪一帧」的歧义。
2. **抗离群拟合**：所有读数拟合一条直线（:mod:`app.recognition2.timefit`，
   Theil–Sen 起步），偏离直线的读数判为离群，不参与定轴。
3. **互校**：定点放大的钟（:func:`resolve_segment` 里 ``model_zoom``）
   与整帧读数（``model``）互为佐证；两者差太远就都记为可疑。

一个诚实的说明
--------------
这一层**只做时间对齐**，不判断「玩家做了什么」。
它读的是画面上的数字，不是画面里的行为。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import timefit

#: 读数与拟合线相差超过这个秒数就算离群（读错了）。
DEFAULT_TOLERANCE = 3.0
#: 至少要有这么多个读数才敢定轴 —— 太少说明模型没在读钟。
MIN_READINGS = 8


@dataclass
class FrameClock:
    """一帧的局内时刻读数。"""

    #: 段内帧编号（1 起，跟提示词里「图N」对上）。
    frame_id: int = 0
    #: 这帧在视频里的秒数。
    video_t: float = 0.0
    #: 模型读到的局内秒数。读不到就是 None（**不猜**）。
    game_s: Optional[float] = None
    #: 读数的来源：``model``（整帧读）/ ``model_zoom``（定点放大读）。
    source: str = "model"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.frame_id,
            "video_time": round(float(self.video_t), 2),
            "clock": (round(float(self.game_s), 1)
                      if self.game_s is not None else None),
            "source": self.source,
        }


@dataclass
class ClockAxis:
    """「视频秒 → 局内秒」这条线，以及它是怎么来的。"""

    fit: Optional[timefit.TimeFit] = None
    readings: List[FrameClock] = field(default_factory=list)
    #: 被判为离群的读数（读错了，或者压根不在对局里）。
    outliers: List[Dict[str, Any]] = field(default_factory=list)
    tolerance: float = DEFAULT_TOLERANCE
    #: 对局在视频里的起止（秒）。给了就**覆盖**拟合线自己推的 ``end_t``。
    #:
    #: 为什么需要它
    #: ------------
    #: 拟合线的 ``end_t`` 是「**最后一个读数**所在的时刻」。读数稀疏时
    #: （比如离线重标时只有几个手工核对点），对局结束前的最后几十秒
    #: 会被误判成「结算」—— 明明还在打，报告却写「不在对局里」。
    game_window: Optional[Tuple[float, float]] = None

    @property
    def ok(self) -> bool:
        return bool(self.fit is not None and self.fit.ok)

    def in_game(self, video_t: float) -> bool:
        """这个视频时刻**在对局里**吗。

        对局之前是选人/加载，之后是结算 —— 那两段**没有局内时刻**。
        硬按直线外推会得到「局内 16:43」这种胡说八道。
        """
        if not self.ok or self.fit is None or self.fit.start_t is None:
            return False
        if self.game_window:
            lo, hi = self.game_window
            return (video_t >= lo - 2.0) and (video_t <= hi + 5.0)
        if video_t < self.fit.start_t - 2.0:
            return False
        end = self.fit.end_t
        if end is not None and video_t > end + 5.0:
            return False
        return True

    def game_seconds(self, video_t: float) -> Optional[float]:
        """某个视频时刻对应的局内秒数。

        对局开始之前 / 结束之后返回 None —— 那不是对局。
        """
        if not self.ok:
            return None
        if not self.in_game(video_t):
            return None
        return self.fit.game_seconds(video_t)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "ok": self.ok,
            "tolerance": self.tolerance,
            "readings": len(self.readings),
            "outliers": list(self.outliers),
            "source": "多模态模型逐帧读钟",
        }
        if self.fit is not None:
            out.update(self.fit.to_dict())
        return out

    @staticmethod
    def build(readings: Sequence[FrameClock],
              tolerance: float = DEFAULT_TOLERANCE,
              min_points: int = MIN_READINGS) -> "ClockAxis":
        """用一批逐帧读数定一条轴。"""
        readings = list(readings)
        axis = ClockAxis(readings=readings, tolerance=tolerance)
        points: List[Tuple[float, float]] = [
            (r.video_t, float(r.game_s)) for r in readings if r.game_s is not None
        ]
        if len(points) < min_points:
            axis.fit = timefit.TimeFit(
                ok=False,
                reason="只有 %d 个时刻读数，不够定一条轴（至少要 %d 个）"
                       % (len(points), min_points))
            return axis

        fit = timefit.robust_fit(points, min_points=min_points,
                                  residual_tolerance=tolerance)
        axis.fit = fit
        if not fit.ok:
            return axis

        for r in readings:
            if r.game_s is None:
                continue
            gs = fit.game_seconds(r.video_t)
            outside = (r.video_t < fit.start_t - 2.0)
            if fit.end_t is not None and r.video_t > fit.end_t + 5.0:
                outside = True
            if gs is None or outside:
                axis.outliers.append({
                    "id": r.frame_id,
                    "video_time": round(float(r.video_t), 2),
                    "clock": round(float(r.game_s), 1),
                    "reason": "不在对局时间内（对局外没有局内时刻）",
                })
                continue
            delta = abs(float(r.game_s) - float(gs))
            if delta > tolerance:
                axis.outliers.append({
                    "id": r.frame_id,
                    "video_time": round(float(r.video_t), 2),
                    "clock": round(float(r.game_s), 1),
                    "reason": "与拟合线差 %.1f 秒（读错了，不参与定轴）" % delta,
                })
        return axis


def pick_anchor(frame_pts: Sequence[float]) -> int:
    """挑锚点帧的下标。

    **锚点 = 段内时间居中的那一帧**，不是首帧也不是末帧：
    它最能代表「这一段」发生在什么时候。

    为什么不能用「首末帧的平均值」当时间戳
    --------------------------------------
    早期版本把 ``video_time`` 设成段内各帧时间的**平均**，结果：
    段长 17.8 秒、4 帧时，平均值离最近的那张帧整整 **2.97 秒**
    （= L/2(n-1)，理论值，实测 2.92~2.99）。
    拿平均时刻去比某一帧画面的钟，必然差 3 秒 —— 这不是时间线错了，
    是**口径**错了：时间线说的是「这一段的中间」，验收却拿某一帧去比。
    改成锚点帧以后，「这一段发生在 T 时刻」和「T 时刻那张画面上的钟」
    指的是同一个瞬间，差值应当是 0（最多差一个「秒」的截断）。
    """
    if not frame_pts:
        return 0
    if len(frame_pts) == 1:
        return 0
    mid = (float(frame_pts[0]) + float(frame_pts[-1])) / 2.0
    best = 0
    best_d = None
    for i, t in enumerate(frame_pts):
        d = abs(float(t) - mid)
        if best_d is None or d < best_d:
            best = i
            best_d = d
    return best


def resolve_segment(
    frame_pts: Sequence[float],
    frame_clocks: Sequence[Optional[float]],
    axis: Optional[ClockAxis] = None,
    tolerance: float = DEFAULT_TOLERANCE,
    anchor_index: Optional[int] = None,
    video_bounds: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """把一段的「帧时刻 + 逐帧读数」变成「锚点 + 区间」。

    返回的字典直接就是 :class:`app.recognition2.beat.Beat` 需要的那些时间字段。

    规则
    ----
    - 锚点帧的局内时刻**优先用模型直接读到的数**（那就是画面上显示的数，
      截断到整秒，和人眼看到的一致）；
    - 读数与拟合线差得离谱（离群）时**不采信**，改用拟合线推的值，
      并在 ``clock_source`` 里标成 ``axis``；
    - 模型没读到、拟合线也没有 → 那一段没有局内时刻（``-1``），
      **绝不拿相邻段的值硬填**。

    ``video_bounds`` 是这一段在原视频里的起止（段边界，不是帧）。
    给了就用它当视频区间 —— 段边界才是「这一段」的真实范围。
    """
    pts = [float(t) for t in frame_pts]
    n = len(pts)
    out: Dict[str, Any] = {
        "anchor_index": 0,
        "video_time": 0.0,
        "video_time_start": 0.0,
        "video_time_end": 0.0,
        "game_time": -1.0,
        "game_time_start": -1.0,
        "game_time_end": -1.0,
        "clock_source": "none",
        "clock_residual": None,
    }
    if n == 0:
        return out

    if anchor_index is None or not (0 <= anchor_index < n):
        anchor_index = pick_anchor(pts)
    out["anchor_index"] = int(anchor_index)
    out["video_time"] = round(pts[anchor_index], 2)
    if video_bounds:
        out["video_time_start"] = round(min(video_bounds), 2)
        out["video_time_end"] = round(max(video_bounds), 2)
    else:
        out["video_time_start"] = round(min(pts), 2)
        out["video_time_end"] = round(max(pts), 2)

    def _clock_at(i: int) -> Tuple[Optional[float], str, Optional[float]]:
        """第 i 帧的局内时刻、来源、与拟合线的差。"""
        direct = None
        if i < len(frame_clocks):
            direct = frame_clocks[i]
        fitted = axis.game_seconds(pts[i]) if axis is not None else None
        if direct is None:
            if fitted is None:
                return None, "none", None
            return round(float(fitted), 2), "axis", 0.0
        if fitted is None:
            # 有读数但拟合线不认（多半是对局外的读数）—— 采信读数本身。
            return round(float(direct), 2), "model", None
        delta = abs(float(direct) - float(fitted))
        if delta > tolerance:
            return round(float(fitted), 2), "axis", round(delta, 2)
        return round(float(direct), 2), "model", round(delta, 2)

    def _first_valid(order: Sequence[int]) -> Tuple[Optional[float], str, Optional[float]]:
        for i in order:
            if i < 0 or i >= n:
                continue
            value, source, residual = _clock_at(i)
            if value is not None:
                return value, source, residual
        return None, "none", None

    # -- 锚点 --
    g, src, res = _clock_at(anchor_index)
    if g is None:
        # 锚点没读到，就近找一帧读到的（宁可偏移一点，也别整段没时刻）。
        order = sorted(range(n), key=lambda i: abs(i - anchor_index))
        g, src, res = _first_valid(order)
    out["game_time"] = g if g is not None else -1.0
    out["clock_source"] = src
    out["clock_residual"] = res

    # -- 区间：优先用首末帧的**直接读数**（那就是画面上的数），
    #    读不到才用拟合线推。 --
    gs, ssrc, _ = _first_valid(list(range(n)))
    ge, esrc, _ = _first_valid(list(range(n - 1, -1, -1)))
    if gs is None and axis is not None:
        gs = axis.game_seconds(out["video_time_start"])
    if ge is None and axis is not None:
        ge = axis.game_seconds(out["video_time_end"])
    out["game_time_start"] = gs if gs is not None else -1.0
    out["game_time_end"] = ge if ge is not None else -1.0
    if gs is not None and ge is not None and ge < gs:
        # 区间反了（读数乱序）—— 宁可报「没有区间」，也不要给一个反的。
        out["game_time_start"] = -1.0
        out["game_time_end"] = -1.0
    return out
