"""切片按**模式项存在的时间区间**切 —— 区间内切够密、且不冗余。

用户定的口径：

> 「对模式项存在的所有时间区间内，对对局视频切很多片，
> 要包含所有有效信息，但不冗余。然后每个时间区间的切片去匹配模式项」

盯四件事：

1. **切片落在模式项存在的时间区间里** —— 区间才是判定单位，证据按区间组织；
2. **区间内切很多片** —— 一段里发生两次龙坑，只切一片就漏掉一次；
3. **同一件事不切多片** —— 一次团战切出七八片是冗余；
4. **不同的事都切到** —— 漏掉一件就是漏掉一条结论。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.collection import video_upload as vu  # noqa: E402


def _points(minutes=20, per_minute=20, peak_minutes=()):
    """造一段变化序列；``peak_minutes`` 里的分钟变化量大（有大事发生）。"""
    pts = []
    for m in range(minutes):
        for i in range(per_minute):
            t = m * 60 + i * 3.0
            d = 5.0 if m in peak_minutes else 1.0
            pts.append((t, d + (i % 3) * 0.1))
    return pts


#: 模式项存在的时间区间（分钟）：前 2 分钟、中间 2 分钟、后面一大段。
INTERVALS = [(0, 1), (2, 3), (12, 60)]

#: 每个局内分钟能判出的规则条数。
WEIGHTS = {0: 200, 1: 200, 2: 300, 3: 300, 12: 250, 13: 250, 14: 250}


def test_slices_land_inside_the_mode_item_intervals():
    """切片只落在模式项存在的时间区间里（其余时间没有可判的模式）。"""
    pts = _points()
    times = vu.density_slices(pts, budget=24, weights=WEIGHTS,
                              game_start_t=0.0, intervals=INTERVALS)
    assert times
    for t in times:
        minute = int(t // 60)
        assert any(lo <= minute <= hi for lo, hi in INTERVALS), \
            f"第 {minute} 分钟没有模式项，不该切"


def test_an_interval_without_rules_is_not_cut():
    """第 4~11 分钟没有模式项 —— 切了也判不出东西。"""
    pts = _points()
    times = vu.density_slices(pts, budget=24, weights=WEIGHTS,
                              game_start_t=0.0, intervals=INTERVALS)
    assert not [t for t in times if 60 * 4 <= t < 60 * 12]


def test_a_busy_interval_gets_more_slices():
    """规则多的区间切得更密。"""
    pts = _points()
    times = vu.density_slices(pts, budget=24, weights=WEIGHTS,
                              game_start_t=0.0, intervals=INTERVALS)

    def count(lo, hi):
        return len([t for t in times if lo * 60 <= t < (hi + 1) * 60])

    assert count(2, 3) > count(0, 1)


def test_one_event_is_not_cut_repeatedly():
    """同一场团战（连续剧烈变化）只切一片 —— 那是冗余。"""
    pts = []
    for i in range(40):
        pts.append((100.0 + i * 0.5, 9.0))       # 20 秒连着的团战
    for i in range(40):
        pts.append((200.0 + i * 3.0, 1.0))       # 安静的两分钟
    times = vu.density_slices(pts, budget=8, weights={1: 300, 2: 300},
                              game_start_t=0.0, intervals=[(1, 3)])
    busy = [t for t in times if 95 <= t < 130]
    assert len(busy) <= 2, f"同一场团战切了 {len(busy)} 片"


def test_two_separate_events_both_get_a_slice():
    """两件分开的事都要有片 —— 漏掉一件就是漏掉一条结论。"""
    pts = []
    for i in range(20):            # 第一次团战：30~40 秒
        pts.append((30.0 + i * 0.5, 9.0))
    for i in range(40):            # 安静
        pts.append((60.0 + i * 1.0, 1.0))
    for i in range(20):            # 第二次团战：110~120 秒
        pts.append((110.0 + i * 0.5, 9.0))
    times = vu.density_slices(pts, budget=8, weights={0: 300, 1: 300},
                              game_start_t=0.0, intervals=[(0, 1)])
    first = [t for t in times if 25 <= t < 50]
    second = [t for t in times if 105 <= t < 130]
    assert first, "第一次事件没切到"
    assert second, "第二次事件漏了"


def test_without_intervals_it_falls_back_to_minute_buckets():
    """没有模式项时间区间时，老路（按分钟）仍然能用。"""
    pts = _points()
    times = vu.density_slices(pts, budget=12, weights=WEIGHTS, game_start_t=0.0)
    assert times
    minutes = {int(t // 60) for t in times}
    assert minutes and all(m in WEIGHTS for m in minutes)
