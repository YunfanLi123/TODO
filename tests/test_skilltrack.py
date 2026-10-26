"""英雄轨迹：跨帧追踪与运动特征。

盯三件事：
1. **孤立离群点被中位数压掉** —— 一个认错的点会把「直线推进」变成「折返」，
   而「带线 vs 刷野」正是靠方向熵区分的；
2. **走丢了就留缺口**，不硬连 —— 宁可少一段，不要编一段；
3. **运动特征算得对**：总路程、净位移、直线性、折返率。
"""
from __future__ import annotations

import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from app.collection import skilltrack as st  # noqa: E402


class _Cand:
    """一个候选（只需要 skilltrack 用到的几个字段）。"""

    def __init__(self, nx, ny, hollow=1.0, score=100.0):
        self.nx = nx
        self.ny = ny
        self.hollow = hollow
        self.score = score


def _track(points):
    return [st.TrackPoint(t=2.0 * i, nx=p[0], ny=p[1]) for i, p in
            enumerate(points)]


def test_median_filter_removes_an_isolated_outlier():
    """一个孤立的错点要被邻域中位数压回去。"""
    pts = [st.TrackPoint(t=0, nx=0.1, ny=0.1),
           st.TrackPoint(t=1, nx=0.1, ny=0.1),
           st.TrackPoint(t=2, nx=0.9, ny=0.9),     # 离群
           st.TrackPoint(t=3, nx=0.1, ny=0.1),
           st.TrackPoint(t=4, nx=0.1, ny=0.1)]
    out = st._median_filter(pts)
    assert abs(out[2].nx - 0.1) < 1e-9
    assert abs(out[2].ny - 0.1) < 1e-9


def test_build_track_links_neighbours_and_leaves_gaps():
    """连得上的接上；跨得太远的**留缺口**，不硬连。"""
    frames = [
        (0.0, [_Cand(0.10, 0.10)]),
        (2.0, [_Cand(0.12, 0.10)]),
        (4.0, [_Cand(0.14, 0.10)]),
        # 跳到地图另一头 —— 连不上
        (6.0, [_Cand(0.90, 0.90)]),
        (8.0, [_Cand(0.16, 0.10)]),
    ]
    points, diag = st.build_track(frames)
    assert points[0] is not None
    assert points[1] is not None and points[2] is not None
    assert points[3] is None, "跨了大半个地图还接上，就是编轨迹"
    assert diag["picked"] == 4


def test_build_track_prefers_the_hollow_candidate():
    """空心的是玩家标记（弧），实心的是绿色地形 —— 形状分得开它们。"""
    frames = [
        (0.0, [_Cand(0.10, 0.10, hollow=1.0)]),
        (2.0, [_Cand(0.10, 0.10, hollow=0.0, score=400.0),
               _Cand(0.12, 0.10, hollow=1.0, score=100.0)]),
    ]
    points, _diag = st.build_track(frames)
    assert points[1] is not None
    # 空心的那个离得近、又空心 → 被选中
    assert abs(points[1].nx - 0.12) < 1e-6


def test_motion_features_of_a_straight_line():
    """直线推进：直线性接近 1，折返率 0，总路程≈净位移。"""
    pts = _track([(0.10 + 0.02 * i, 0.10) for i in range(20)])
    m = st.motion_of(pts)
    assert m.ok
    assert m.straightness > 0.95
    assert m.revisit == 0.0
    assert abs(m.path_len - m.net_len) < 1e-6


def test_motion_features_of_an_oscillating_path():
    """来回折返：折返率高，直线性低。"""
    pts = _track([(0.20 + (0.05 if i % 2 else 0.0), 0.20) for i in range(20)])
    m = st.motion_of(pts)
    assert m.ok
    assert m.revisit > 0.5
    assert m.straightness < 0.5


def test_too_few_points_gives_no_features():
    """点不够就说读不出来，不硬算。"""
    m = st.motion_of(_track([(0.1, 0.1), (0.2, 0.2)]))
    assert not m.ok
    assert m.reason


def test_points_round_trip_through_dicts():
    """轨迹要能存进 JSON 再取回来（它跟着扫描结果落库）。"""
    pts = _track([(0.1, 0.2), (0.3, 0.4)])
    rows = [p.to_dict() for p in pts]
    back = st.points_from_dicts(rows)
    assert len(back) == len(pts)
    assert abs(back[1].nx - pts[1].nx) < 1e-6
    assert abs(back[1].t - pts[1].t) < 1e-6


def test_slice_track_keeps_only_the_window():
    pts = _track([(0.1, 0.1) for _ in range(10)])
    got = st.slice_track(pts, 4.0, 12.0)
    assert got and all(4.0 <= p.t <= 12.0 for p in got)
