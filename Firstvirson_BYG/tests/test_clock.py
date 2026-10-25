"""游戏内时钟：用合成读数验证「拟合出时间线」这件事本身。

这些测试用的是**合成读数**：直接造出 (视频秒, 游戏秒) 的样本对，
不碰像素。这样「该推出什么结论」是确定的，与机型、分辨率、字体都无关。

必须说清楚的一件事：这里验的是**机制**，不是「真的能读懂王者荣耀」。
模板与槽位坐标要对着真实录屏标定出来（``tools/build_clock_templates.py``），
没有那段录屏，任何声称「已验证能读王者时钟」的说法都是空的。
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.collection.clock import (
    MAX_MINUTES,
    MIN_SAMPLES,
    RESIDUAL_TOLERANCE,
    SLOPE_TOLERANCE,
    ClockSample,
    build_clock_track,
)


def _readings(start_t=8.0, game_end=68.0, step=0.4, game_start_at=None):
    """造一串「读得出时钟」的采样。

    BP 段读不出（``game_s`` 为 None），进入游戏后每秒读一次 ——
    真实对局就是这样：加载界面没有时钟。
    """
    game_start_at = start_t if game_start_at is None else game_start_at
    out, t = [], 0.0
    while t <= game_end + 5.0:
        g = None if t < start_t else int(round(t - game_start_at))
        out.append(ClockSample(t=round(t, 3), game_s=g, confidence=0.9))
        t += step
    return out


def test_readings_map_to_the_right_game_time():
    """读出来的时间线要能把视频秒换算成游戏分钟 —— 这就是阶段划分的依据。"""
    track = build_clock_track(_readings())
    assert track.ok, track.reason

    assert track.game_start_t == pytest.approx(8.0, abs=0.5)
    assert track.minutes_at(track.game_start_t + 600.0) == pytest.approx(10.0, abs=0.2)
    # 开始前一律按 0 算，而不是负数
    assert track.minutes_at(track.game_start_t - 5.0) == 0.0


def test_game_start_is_derived_not_guessed():
    """对局起点是**拟合出来的截距**，不是「第一次读到数」。

    早先的数跳变方案只能得到相对秒数，起点得另外猜；
    猜错的话整个时间轴平移，每一条带时间窗的模式都判到错误阶段，
    而报告上只是「没触发」—— 看不出来。
    """
    for true_start in (3.0, 8.0, 199.0, 412.5):
        samples = _readings(start_t=true_start, game_end=true_start + 120,
                            game_start_at=true_start)
        track = build_clock_track(samples)
        assert track.ok, track.reason
        assert track.game_start_t == pytest.approx(true_start, abs=1.0), \
            f"起点 {true_start} 被拟合成 {track.game_start_t}"


def test_a_region_that_never_reads_is_reported_not_guessed():
    """读不出时钟必须报「没读到」，不能给一个看着很像的时间。

    这是整套方案最重要的一条：猜错时间，每一条带时间窗的模式都会判到
    错误的阶段，而报告上完全看不出来 —— 它们只是「没触发」。
    """
    samples = [ClockSample(t=i * 0.4, game_s=None) for i in range(200)]
    track = build_clock_track(samples)
    assert not track.ok
    assert track.reason
    assert track.minutes_at(30.0) is None, "没读到就不该给出分钟数"


def test_too_few_samples_is_not_read_as_a_clock():
    """样本太少时要说「样本太少」，不能硬拟合出一条线来。"""
    track = build_clock_track([ClockSample(t=0.0, game_s=0),
                               ClockSample(t=0.4, game_s=0)])
    assert not track.ok
    assert "只读到" in track.reason
    assert track.minutes_at(10.0) is None


def test_slow_clock_is_rejected():
    """游戏时间比视频时间走得慢，不能当成正常的时钟。

    可能的原因：帧率元数据不对，或者压根没在读时钟。
    硬用会得到一条斜率不是 1 的线，于是每一秒的阶段判定都偏一点，
    而累积起来能偏出整个阶段。所以斜率不对必须拒绝。
    """
    samples = []
    t = 0.0
    while t <= 300.0:
        # 游戏时间只走了视频时间的 0.9 倍
        samples.append(ClockSample(t=round(t, 3), game_s=int(0.9 * t)))
        t += 0.4
    track = build_clock_track(samples)
    assert not track.ok, f"斜率 {track.slope} 不该被接受"
    assert track.slope is not None
    assert abs(track.slope - 1.0) > SLOPE_TOLERANCE


def test_jumpy_readings_are_rejected():
    """读数乱跳要拒绝 —— 时钟是连续走的，不会这样。"""
    rng = list(range(0, 200, 3))
    samples = [ClockSample(t=float(t), game_s=v)
               for t, v in zip(range(0, 400, 2), rng)]
    track = build_clock_track(samples)
    assert not track.ok


def test_outliers_are_dropped_but_the_line_survives():
    """偶发读错一两格不该毁掉整条时间线。

    实战里偶尔有一帧被技能特效糊住。真实做法是剔掉离群点再拟合，
    而不是因为一个坏点就宣布「读不出时钟」。
    """
    samples = _readings(start_t=8.0, game_end=180.0, step=0.4)
    # 在中间掺几个明显错的读数
    for i in (40, 80, 120):
        s = samples[i]
        samples[i] = ClockSample(t=s.t, game_s=(s.game_s or 0) + 37)
    track = build_clock_track(samples)
    assert track.ok, track.reason
    assert track.game_start_t == pytest.approx(8.0, abs=1.5)
    assert track.residual is not None and track.residual <= RESIDUAL_TOLERANCE


def test_tolerance_bounds_are_sane():
    """游戏时间应当与视频时间 1:1 走，容差必须包住 1。"""
    assert SLOPE_TOLERANCE < 0.5
    assert RESIDUAL_TOLERANCE > 0
    assert MIN_SAMPLES >= 4
    assert 15 <= MAX_MINUTES <= 60


def test_track_serialises_with_its_reason():
    """读不出来时原因必须能带出去 —— 界面上要能说清为什么没读到。"""
    d = build_clock_track([ClockSample(t=i * 0.4, game_s=None) for i in range(50)]).to_dict()
    assert d["ok"] is False
    assert d["reason"]
    for key in ("game_start_t", "slope", "residual", "samples_read"):
        assert key in d
