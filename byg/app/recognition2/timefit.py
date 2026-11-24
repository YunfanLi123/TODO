"""把一串「读数」拟合成时间线 —— **抗离群**，不是普通最小二乘。

为什么不用现成的 :func:`app.collection.clock.build_clock_track`
--------------------------------------------------------------
它先做一次**普通最小二乘**、再按残差剔离群点。本地实测（2026-10-31，
``赵云.mp4``，每 5 秒读一次，共 231 个点）：

- 对局内（t≥150 秒）读得**完全正确**且严格线性（t=150 → 00:02，
  t=1100 → 15:52）；
- 但**选人界面**和**结算界面**上，模板匹配会锁到「英雄等级 14」
  「技能冷却 44」这种数字上，读出 ``14:44`` / ``24:44`` / ``19:44``
  —— 231 个点里有 17 个是这种。

坏点不是随机撒的，它们**集中在时间轴的一端**（开局前那一大段）。
普通最小二乘对这种系统性偏置没有抵抗力：斜率被从 1.000 拽到 0.942，
于是**好点**的残差跟着变大，又被当成离群点删掉，最后整条线判废
（实测残差 34.4 秒 → ``ok=False``）。

Theil–Sen
---------
所有点对斜率的**中位数**。它对少数一端的离群点免疫 ——
这本就是为「有 20% 脏数据」设计的统计量（Theil 1960）。
先用它把线找出来，再按残差剔离群点，最后用**剔干净的点**做一次
普通最小二乘（此时数据已经干净，最小二乘效率最高）。

所以这里做三步：
1. **Theil–Sen** 估斜率/截距（抗离群）；
2. 按残差剔离群点；
3. 在干净点上重新最小二乘，并做斜率/残差自检。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class TimeFit:
    """拟合出来的一条时间线。"""

    ok: bool = False
    reason: str = ""
    #: 游戏秒 / 视频秒。正常正好 1。
    slope: Optional[float] = None
    #: 截距（游戏秒 = slope * 视频秒 + intercept）。
    intercept: Optional[float] = None
    #: 对局第 0 秒对应的视频时刻。
    start_t: Optional[float] = None
    #: 对局**结束**对应的视频时刻（由最大读数反推）。录屏常在对局后继续。
    end_t: Optional[float] = None
    #: 残差（秒）的 95 分位。
    residual: Optional[float] = None
    #: 参与拟合的点数 / 总点数。
    kept: int = 0
    total: int = 0
    #: 被剔掉的点（``(视频秒, 读数)``）。
    dropped: List[Tuple[float, float]] = field(default_factory=list)

    def game_seconds(self, video_t: float) -> Optional[float]:
        """某个视频时刻对应的局内秒数。

        对局开始之前返回 None —— 那不是对局（选人/加载），
        **不要**返回 0，那会被当成「开局第 0 秒」。
        """
        if not self.ok or self.start_t is None:
            return None
        value = (video_t - self.start_t) * (self.slope or 1.0)
        if value < -1.0:
            return None
        return max(0.0, value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "slope": round(self.slope, 4) if self.slope is not None else None,
            "start_t": round(self.start_t, 1) if self.start_t is not None else None,
            "end_t": round(self.end_t, 1) if self.end_t is not None else None,
            "residual": round(self.residual, 2) if self.residual is not None else None,
            "kept": self.kept,
            "total": self.total,
            "dropped": len(self.dropped),
        }


#: 至少要有这么多个点才敢下结论。
MIN_POINTS = 8
#: 斜率允许的偏差（正常正好 1）。
SLOPE_TOLERANCE = 0.06
#: 残差（秒）超过这个值就判不可信。
RESIDUAL_TOLERANCE = 5.0


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return 0.0
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return (float(ordered[mid - 1]) + float(ordered[mid])) / 2.0


def _theil_sen(points: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    """所有点对斜率的中位数（Theil–Sen 估计）。"""
    slopes: List[float] = []
    n = len(points)
    # 点太多时随机取子集（结果几乎一样，快很多）。
    if n > 400:
        step = n / 400.0
        idx = [int(i * step) for i in range(400)]
    else:
        idx = list(range(n))
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            dt = points[j][0] - points[i][0]
            if abs(dt) < 1.0:
                continue
            slopes.append((points[j][1] - points[i][1]) / dt)
    if not slopes:
        return 1.0, 0.0
    slope = _median(slopes)
    intercept = _median([g - slope * t for t, g in points])
    return slope, intercept


def _least_squares(points: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    """普通最小二乘（只在**已经干净**的数据上用）。"""
    n = len(points)
    if n < 2:
        return 1.0, 0.0
    sx = sum(t for t, _ in points)
    sy = sum(g for _, g in points)
    sxx = sum(t * t for t, _ in points)
    sxy = sum(t * g for t, g in points)
    den = n * sxx - sx * sx
    if abs(den) < 1e-9:
        return 1.0, 0.0
    slope = (n * sxy - sx * sy) / den
    intercept = (sy - slope * sx) / n
    return slope, intercept


def robust_fit(points: Sequence[Tuple[float, float]],
               min_points: int = MIN_POINTS,
               slope_tolerance: float = SLOPE_TOLERANCE,
               residual_tolerance: float = RESIDUAL_TOLERANCE) -> TimeFit:
    """把 ``(视频秒, 读数)`` 拟合成一条时间线。

    读数可以是任意来源 —— OpenCV 模板匹配读出来的、模型读出来的。
    坏点会被剔掉，但**剔不掉的会留下痕迹**（``dropped``）。
    """
    fit = TimeFit(total=len(points))
    usable = [(float(t), float(g)) for t, g in points if g is not None]
    if len(usable) < min_points:
        fit.reason = ("只读到 %d 个时刻，至少要 %d 个才敢定时间线"
                      % (len(usable), min_points))
        return fit

    slope, intercept = _theil_sen(usable)
    resid = [abs(g - (slope * t + intercept)) for t, g in usable]
    thresh = max(residual_tolerance, _median(resid) * 3.0 + 2.0)
    keep = [p for p, r in zip(usable, resid) if r <= thresh]
    dropped = [p for p, r in zip(usable, resid) if r > thresh]

    if len(keep) >= min_points:
        slope, intercept = _least_squares(keep)
        resid = [abs(g - (slope * t + intercept)) for t, g in keep]
    else:
        dropped = usable
        keep = []

    fit.slope = float(slope)
    fit.intercept = float(intercept)
    fit.kept = len(keep)
    fit.dropped = dropped
    fit.residual = float(_percentile(resid, 95)) if resid else 0.0

    if len(keep) < min_points:
        fit.reason = ("剔掉离群点后只剩 %d 个（本来 %d 个）—— 读数太杂，不敢下结论"
                      % (len(keep), len(usable)))
        return fit
    if abs(slope - 1.0) > slope_tolerance:
        fit.reason = ("拟合出来游戏时间走得比真实快慢 %.3f 倍（应当是 1.000）—— "
                      "多半是帧率元数据不对，或者读的根本不是时钟"
                      % slope)
        return fit
    if fit.residual > residual_tolerance:
        fit.reason = ("读数到时间线的偏差最大 %.1f 秒 —— 时钟不该这样跳，"
                      "多半是匹配到了别的东西" % fit.residual)
        return fit

    fit.start_t = float(-intercept / slope)
    # 对局结束点：由最大读数反推 —— 录屏往往在对局结束后还继续着。
    max_game = max(g for _, g in keep)
    fit.end_t = float((max_game - intercept) / slope)
    fit.ok = True
    fit.reason = ("拟合出时间线：%d/%d 个读数可用（剔掉 %d 个离群点），"
                  "对局从视频第 %.0f 秒开始，误差 ±%.1f 秒"
                  % (len(keep), len(usable), len(dropped),
                     fit.start_t, fit.residual))
    return fit


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * pct / 100.0
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return float(ordered[low] * (1.0 - frac) + ordered[high] * frac)
