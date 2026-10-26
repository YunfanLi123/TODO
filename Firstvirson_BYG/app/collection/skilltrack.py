"""英雄位置轨迹：跨帧追踪 + 运动特征。

为什么需要这一层
----------------
「这一段玩家在做什么」这个问题的答案，**不在单帧里，在帧与帧之间**。

单帧认绿圈会认错：实测密集采样（2 秒一帧）时，检出率只有 45%，
而且最强的绿块常常是**一片不动的绿色地形** —— 选出来的是草地，
位置在几帧之间乱跳。

但地形有一个玩家标记**没有**的性质：**它不动**。
所以这里的做法是：

1. **先剔除固定物**：把所有候选按位置聚类，某一簇如果「在几乎所有帧里
   都出现在同一个点」，那它就是地形（玩家的标记不可能整局钉在一个点
   上还被检测到八成的帧），把它从所有帧里剔掉。
   ⇒ 这一步把「最强绿块是草地」这个坑填平了。
2. **再跨帧追踪**：剩下的候选里，逐帧选**与上一帧连得上**的那个；
   连不上就留空（这一帧读不到），不硬凑。
3. **最后算运动特征**：总路程、净位移、直线性、方向熵、折返率、
   各区域停留时间。

这些特征就是**操作模式项**的判据：
刷野是「小范围高频折返」、带线是「单向持续推进」、转线是「净位移大而路径短」。

读不出的就说读不出
------------------
追踪断了就断，不插值成假的轨迹 ——
一条假轨迹会让「在龙坑刷野」判成「在龙坑开龙」，报告上完全看不出来。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: 判定「这一簇是固定物（地形）」的帧占比。
#:
#: 地形在**每一帧**都在；玩家的标记不可能整局钉在一个点、
#: 而且还被检测到这么大比例的帧（实测整体检出率才 45%）。
STATIC_FRAME_RATIO = 0.80

#: 固定簇的容差：簇内位置两两距离都小于它，才算「没动过」。
STATIC_SPREAD = 0.012

#: 相邻采样点之间的最大位移（归一化，采样间隔 2 秒）。
#:
#: 王者英雄的移速有限，2 秒走不了太远；超过这个值的多半是认错了。
#: 取得偏宽是为了容得下技能位移（闪现/突进），宁可留缺口也不硬连。
MAX_STEP = 0.40

#: 追踪时允许的最大「跨帧跳跃」：从上一个已知点出发，
#: 下一帧的候选离它超过这个距离就认为连不上。
LINK_STEP = 0.45

#: 中位数滤波的窗口（点数）。孤立离群点会被中位数压掉。
MEDIAN_WINDOW = 5

#: 「空心」的加权：空心候选在挑选时**便宜这么多**。
#:
#: 为什么按形状而不是按亮度
#: ----------------------
#: 实测同一帧里，玩家标记的绿度峰值可能只有 22（在泉水那种亮背景上），
#: 而一片绿色地形能到 116 —— 亮度**分不开**二者。
#: 但形状分得开：玩家的标记是**圆弧**（中心不绿），地形是**实心**的。
#:
#: 所以候选要**宽松**地收（低阈值），挑的时候按空心优先。
HOLLOW_WEIGHT = 0.25

#: 走丢了以后重新接上的最大跨度（归一化）。
#:
#: 闪现、跨墙跑、或者中间有一长段没认出来 —— 都可能造成大跨度。
#: 超过这个值就不接了（宁可留缺口）。
REANCHOR_STEP = 0.55

#: 特征至少要有这么多个有效采样点才算数。
MIN_POINTS = 6

#: 方向量化成几个方向桶（用于方向熵）。
DIRECTION_BINS = 8


@dataclass(frozen=True)
class TrackPoint:
    """轨迹上的一个点。"""

    t: float
    nx: float
    ny: float
    conf: float = 0.0

    def dist(self, other: "TrackPoint") -> float:
        return math.hypot(self.nx - other.nx, self.ny - other.ny)

    def to_dict(self) -> Dict[str, Any]:
        return {"t": round(self.t, 2), "nx": round(self.nx, 4),
                "ny": round(self.ny, 4), "conf": round(self.conf, 3)}


@dataclass
class Motion:
    """一段轨迹的运动特征。

    全部是**可解释**的几何量 —— 报告里要能写清
    「因为这段在野区来回走了 12 次、方向一直在变」，
    而不是抛一个模型分数。
    """

    samples: int = 0
    #: 被采信的采样点占比（0~1）
    covered: float = 0.0
    #: 时间跨度（秒）
    span_sec: float = 0.0
    #: 总路程（归一化单位）
    path_len: float = 0.0
    #: 首尾直线距离
    net_len: float = 0.0
    #: 直线性 = 净位移 / 总路程（0~1）。带线高、刷野低。
    straightness: float = 0.0
    #: 方向熵（0~1，越大方向越乱）
    direction_entropy: float = 0.0
    #: 折返率（0~1，越大越像在原地来回打）
    revisit: float = 0.0
    #: 各区域的停留秒数
    zone_sec: Dict[str, float] = field(default_factory=dict)
    #: 各区域的停留占比
    zone_ratio: Dict[str, float] = field(default_factory=dict)
    #: 轨迹两端
    start: Optional[TrackPoint] = None
    end: Optional[TrackPoint] = None
    ok: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "samples": self.samples,
            "covered": round(self.covered, 3),
            "span_sec": round(self.span_sec, 1),
            "path_len": round(self.path_len, 4),
            "net_len": round(self.net_len, 4),
            "straightness": round(self.straightness, 3),
            "direction_entropy": round(self.direction_entropy, 3),
            "revisit": round(self.revisit, 3),
            "zone_sec": {k: round(v, 1) for k, v in self.zone_sec.items()},
            "zone_ratio": {k: round(v, 3) for k, v in self.zone_ratio.items()},
            "ok": self.ok,
            "reason": self.reason,
        }


def _cluster_positions(items: Sequence[Tuple[int, float, float]],
                       tol: float) -> List[Dict[str, Any]]:
    """按位置把 (帧号, nx, ny) 聚成簇。

    用的不是单链聚类，而是**贪心成簇**：一个点归到第一个中心够近的簇
    里（或自己开新簇）。单链会被「A 近 B、B 近 C、C 不近 A」这种链式
    情况拖成一团 —— 那正好会把整局的轨迹连成一大坨。
    """
    clusters: List[Dict[str, Any]] = []
    for frame, nx, ny in items:
        hit = None
        for c in clusters:
            if math.hypot(nx - c["cx"], ny - c["cy"]) <= tol:
                hit = c
                break
        if hit is None:
            clusters.append({"frames": [frame], "pts": [(nx, ny)],
                             "cx": nx, "cy": ny})
        else:
            hit["frames"].append(frame)
            hit["pts"].append((nx, ny))
            n = len(hit["pts"])
            hit["cx"] = sum(p[0] for p in hit["pts"]) / n
            hit["cy"] = sum(p[1] for p in hit["pts"]) / n
    return clusters


def _cluster_spread(pts: Sequence[Tuple[float, float]]) -> float:
    """簇内位置的最大两两距离。"""
    if len(pts) < 2:
        return 0.0
    worst = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
            if d > worst:
                worst = d
    return worst


def drop_static_clusters(
    frames_candidates: Sequence[Tuple[float, Sequence[Any]]],
    tol: float = STATIC_SPREAD,
) -> Tuple[List[List[Any]], int]:
    """剔除「整局钉在一个点」的候选（地形/固定图标）。

    返回 ``(每帧剩下的候选, 被剔掉的簇数)``。
    """
    total = len(frames_candidates)
    if total == 0:
        return [], 0
    items: List[Tuple[int, float, float]] = []
    for i, (_t, cands) in enumerate(frames_candidates):
        for c in cands:
            items.append((i, c.nx, c.ny))
    if not items:
        return [list(c) for _t, c in frames_candidates], 0
    clusters = _cluster_positions(items, tol)
    static: List[Tuple[float, float]] = []
    for c in clusters:
        if len(c["frames"]) < STATIC_FRAME_RATIO * total:
            continue
        if _cluster_spread(c["pts"]) > tol:
            continue
        static.append((c["cx"], c["cy"]))
    if not static:
        return [list(c) for _t, c in frames_candidates], 0
    kept: List[List[Any]] = []
    for _t, cands in frames_candidates:
        row = []
        for c in cands:
            if any(math.hypot(c.nx - sx, c.ny - sy) <= tol for sx, sy in static):
                continue
            row.append(c)
        kept.append(row)
    return kept, len(static)


def _median_filter(points: Sequence[Optional[TrackPoint]],
                   window: int = MEDIAN_WINDOW) -> List[Optional[TrackPoint]]:
    """中位数滤波：把孤立离群点压回邻域的中位位置。

    没有它的话，一个认错的点会把「直线推进」变成「折返」——
    而「带线 vs 刷野」正是靠方向熵区分的。
    """
    out: List[Optional[TrackPoint]] = []
    half = window // 2
    for i, p in enumerate(points):
        if p is None:
            out.append(None)
            continue
        xs: List[float] = []
        ys: List[float] = []
        for j in range(max(0, i - half), min(len(points), i + half + 1)):
            q = points[j]
            if q is not None:
                xs.append(q.nx)
                ys.append(q.ny)
        if not xs:
            out.append(p)
            continue
        xs.sort()
        ys.sort()
        out.append(TrackPoint(t=p.t, nx=xs[len(xs) // 2], ny=ys[len(ys) // 2],
                              conf=p.conf))
    return out


def _pick(cands: Sequence[Any], last: Optional[TrackPoint],
          link_step: float, reanchor_step: float) -> Optional[Any]:
    """在候选里挑「最像玩家标记」的那一个。

    代价 = 离上一个点多远 − 空心加成（空心的是弧，不是实心地形）。
    离得太远就**不接**（留缺口），除非还在可接受的跨度内重新接上。
    """
    if not cands:
        return None
    if last is None:
        # 第一帧：空心的优先，其次看单帧分
        return min(cands, key=lambda c: (-c.hollow, -c.score))
    scored = []
    for c in cands:
        d = math.hypot(c.nx - last.nx, c.ny - last.ny)
        scored.append((d - HOLLOW_WEIGHT * c.hollow, d, c))
    scored.sort(key=lambda x: x[0])
    best_cost, best_d, best = scored[0]
    if best_d <= link_step:
        return best
    # 走丢了：看看能不能在一个更大的跨度内重新接上（要空心的）
    hollows = [c for c in cands if c.hollow > 0]
    if hollows:
        cb = min(hollows, key=lambda c: math.hypot(c.nx - last.nx, c.ny - last.ny))
        if math.hypot(cb.nx - last.nx, cb.ny - last.ny) <= reanchor_step:
            return cb
    return None


def build_track(frames_candidates: Sequence[Tuple[float, Sequence[Any]]],
                static_tol: float = STATIC_SPREAD,
                link_step: float = LINK_STEP,
                reanchor_step: float = REANCHOR_STEP
                ) -> Tuple[List[Optional[TrackPoint]], Dict[str, Any]]:
    """把逐帧候选串成一条轨迹。

    候选要用**宽松**阈值收（见
    :func:`app.collection.minimap.find_hero_candidates` 的 ``ratio`` 参数）——
    这里靠形状（空心弧）而不是亮度认玩家。

    返回 ``(每帧的轨迹点（读不到就是 None）, 诊断信息)``。
    """
    diag: Dict[str, Any] = {"frames": len(frames_candidates), "dropped_clusters": 0,
                            "picked": 0}
    if not frames_candidates:
        return [], diag

    kept, dropped = drop_static_clusters(frames_candidates, static_tol)
    diag["dropped_clusters"] = dropped

    picks: List[Optional[TrackPoint]] = []
    last: Optional[TrackPoint] = None
    for (t, _cands), row in zip(frames_candidates, kept):
        c = _pick(row, last, link_step, reanchor_step)
        if c is None:
            picks.append(None)
            continue
        p = TrackPoint(t=t, nx=c.nx, ny=c.ny, conf=c.score)
        picks.append(p)
        last = p
    diag["picked"] = sum(1 for p in picks if p is not None)
    return _median_filter(picks), diag


def points_from_dicts(rows: Sequence[Dict[str, Any]]) -> List[TrackPoint]:
    """把（存进 JSON 之后）的一串轨迹行还原成 :class:`TrackPoint`。

    **带缺口的行**（只有 ``t``、没有 ``nx``）会被跳过 ——
    缺口表示「这一帧读不出位置」，不是「位置在原点」。

    轨迹要跟着扫描结果走（扫描结果会落库、会进摘要），
    所以中途一定要能序列化成普通字典。
    """
    out: List[TrackPoint] = []
    for r in rows or []:
        if r.get("nx") is None or r.get("ny") is None:
            continue
        try:
            out.append(TrackPoint(
                t=float(r.get("t", 0.0)),
                nx=float(r.get("nx", 0.0)),
                ny=float(r.get("ny", 0.0)),
                conf=float(r.get("conf", 0.0)),
            ))
        except (TypeError, ValueError):
            continue
    return out


def slice_track_rows(rows: Sequence[Dict[str, Any]], t0: float,
                     t1: float) -> Tuple[List[TrackPoint], int]:
    """切出一段轨迹，返回 ``(点, 这一段的采样点总数)``。

    总数**包含缺口** —— 覆盖率要按「这一段本来该有多少个采样点」算，
    才看得出这一段有多少是读不出来的。
    """
    sel = [r for r in (rows or [])
           if t0 - 1e-6 <= float(r.get("t", -1e9)) <= t1 + 1e-6]
    pts = points_from_dicts([r for r in sel if r.get("nx") is not None])
    return pts, len(sel)


def slice_track(points: Sequence[Optional[TrackPoint]],
                t0: float, t1: float) -> List[TrackPoint]:
    """取轨迹在 ``[t0, t1]`` 这段里的点（丢掉读不到的）。"""
    out = []
    for p in points:
        if p is None:
            continue
        if t0 - 1e-6 <= p.t <= t1 + 1e-6:
            out.append(p)
    return out


def _direction_entropy(dirs: Sequence[float]) -> float:
    """方向直方图的归一化熵（0~1）。

    0 = 一直朝一个方向走（带线）；1 = 方向很乱（刷野、来回跑）。
    """
    if not dirs:
        return 0.0
    bins = [0] * DIRECTION_BINS
    for d in dirs:
        idx = int((d % (2 * math.pi)) / (2 * math.pi) * DIRECTION_BINS)
        bins[idx % DIRECTION_BINS] += 1
    total = float(sum(bins))
    if total <= 0:
        return 0.0
    ent = 0.0
    for b in bins:
        if b <= 0:
            continue
        p = b / total
        ent -= p * math.log(p)
    return ent / math.log(DIRECTION_BINS)


def motion_of(points: Sequence[TrackPoint],
              zone_of=None,
              flipped: Optional[bool] = None,
              total: Optional[int] = None) -> Motion:
    """算一段轨迹的运动特征。

    ``zone_of(nx, ny) -> Optional[str]`` 用来把点归到区域；
    没有区域表（区域为空）时只算运动学特征。

    ``total`` 是这一段**本来该有多少个采样点**（含读不到的那些）。
    覆盖率 = 认出来的点 / total —— 没有它就不知道这一段有多少是读不出来的，
    而「读不出来」和「读出来是空的」是两件事。
    """
    m = Motion()
    m.samples = len(points)
    m.covered = (len(points) / float(total)) if total else (
        1.0 if points else 0.0)
    if len(points) < MIN_POINTS:
        m.reason = (f"有效采样点只有 {len(points)} 个，"
                    f"不足 {MIN_POINTS} 个算不出运动特征")
        return m

    m.ok = True
    m.start = points[0]
    m.end = points[-1]
    m.span_sec = points[-1].t - points[0].t
    if m.span_sec <= 0:
        m.ok = False
        m.reason = "采样点时间跨度为零"
        return m

    steps: List[float] = []
    dirs: List[float] = []
    for a, b in zip(points, points[1:]):
        dx = b.nx - a.nx
        dy = b.ny - a.ny
        d = math.hypot(dx, dy)
        if d <= 0:
            continue
        steps.append(d)
        dirs.append(math.atan2(dy, dx))

    m.path_len = sum(steps)
    m.net_len = points[-1].dist(points[0])
    m.straightness = (m.net_len / m.path_len) if m.path_len > 1e-9 else 0.0
    m.direction_entropy = _direction_entropy(dirs)

    # 折返率：相邻两步方向相反（差 > 120°）的比例
    opposite = 0
    for a, b in zip(dirs, dirs[1:]):
        diff = abs((a - b + math.pi) % (2 * math.pi) - math.pi)
        if diff > 2.0:      # ≈115°
            opposite += 1
    m.revisit = (opposite / len(dirs)) if len(dirs) > 1 else 0.0

    # 区域停留
    if zone_of is not None:
        per_point = m.span_sec / max(1, len(points) - 1)
        for p in points:
            z = zone_of(p.nx, p.ny)
            if not z:
                continue
            m.zone_sec[z] = m.zone_sec.get(z, 0.0) + per_point
        total = sum(m.zone_sec.values())
        if total > 0:
            m.zone_ratio = {k: v / total for k, v in m.zone_sec.items()}
    return m
