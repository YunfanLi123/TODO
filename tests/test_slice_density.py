"""切片按**规则密度**分配预算 —— 不是按时间平均分。

盯三件事：

1. **没有规则的分钟不切** —— 切了也判不出任何东西，预算浪费在空气上。
   （王者 S45 的第 5 分钟与第 11 分钟就是这种空档：攻略的节点在
   4:00–6:00、10:00–12:00 之间留着。）
2. **密度大的分钟多切** —— 那一分钟能判出的规则多，多切几刀才不浪费。
3. **桶 = 一个局内分钟** —— 桶号就是局内分钟号，密度直接查得到，
   不会「桶算的是视频第几分钟、密度查的是局内第几分钟」这种错位。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.collection import video_upload as vu  # noqa: E402
from app.rules.density import (  # noqa: E402
    density_weights,
    minute_density,
    seed_density,
)


def _points(minutes=20, per_minute=20):
    """造一段变化序列：每分钟 20 个采样点，变化量固定。"""
    pts = []
    for m in range(minutes):
        for i in range(per_minute):
            t = m * 60 + i * 3.0
            pts.append((t, 1.0 + (i % 3)))
    return pts


def test_the_density_shape_comes_from_the_rule_library():
    """密度 = 每个局内分钟能判出多少条规则。"""
    rules = [
        {"id": "a", "trigger": {"all": [{"fact": "time_min", "op": "between",
                                        "value": [0, 0]}]}},
        {"id": "b", "trigger": {"all": [{"fact": "time_min", "op": "between",
                                        "value": [2, 3]}]}},
        {"id": "c", "trigger": {"all": [{"fact": "time_min", "op": "between",
                                        "value": [4, 4]}]}},
        {"id": "d", "trigger": {"all": [{"fact": "time_min", "op": "between",
                                        "value": [4, 4]}]}},
    ]
    density = minute_density(rules)
    assert density[0] == 1
    assert density[2] == 1 and density[3] == 1, "窗口 [2,3] 覆盖的两分钟都要算"
    assert density[4] == 2
    assert 5 not in density, "没有规则覆盖的分钟不该出现在密度里"


def test_the_real_library_has_gaps_and_peaks():
    """真实库（王者 S45）确实有空档与高峰。"""
    density = seed_density("hok")
    assert density, "读不到王者种子库的密度"
    assert density[4] > density[0], "第 4 分钟（龙坑/带线/惩击）该比第 0 分钟密"
    assert density.get(5, 0) == 0, "第 5 分钟没有对应模式（4:00 与 6:00 之间的空档）"
    assert density.get(11, 0) == 0, "第 11 分钟同样没有对应模式"


def test_minutes_without_rules_get_no_slice():
    """没有规则的分钟不切。"""
    weights = density_weights({0: 1.0, 1: 1.0, 4: 1.0})
    times = vu.density_slices(
        _points(minutes=8), budget=12, span=(0.0, 8 * 60.0),
        weights=weights, game_start_t=0.0,
    )
    minutes = {int(t // 60) for t in times}
    assert 2 not in minutes, "第 2 分钟没有规则，不该切"
    assert 3 not in minutes
    assert 0 in minutes and 4 in minutes, "有规则的分钟要切到"


def test_denser_minutes_get_more_slices():
    """密度大的分钟切得多。

    用真实库里的密度差距来测（第 4 分钟 327 条 vs 第 0 分钟 237 条）——
    差距不大，但正是要体现在切片数上的那点差距。
    """
    weights = density_weights({0: 237, 1: 237, 4: 327})
    pts = []
    for m in range(8):
        for i in range(20):
            pts.append((m * 60 + i * 3.0, 1.0 + (i % 3)))
    times = vu.density_slices(pts, budget=12, span=(0.0, 8 * 60.0),
                              weights=weights, game_start_t=0.0)
    per = {}
    for t in times:
        per[int(t // 60)] = per.get(int(t // 60), 0) + 1
    assert per.get(4, 0) > per.get(0, 0), (
        f"规则更密的分钟应该切得更多：{per}"
    )
    assert sum(per.values()) == 12, f"预算没切完：{per}"


def test_the_budget_is_fully_used():
    """预算要切完 —— 切不满说明密度表或分配有问题。"""
    weights = density_weights({0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})
    times = vu.density_slices(
        _points(minutes=5), budget=10, span=(0.0, 5 * 60.0),
        weights=weights, game_start_t=0.0,
    )
    assert len(times) == 10, f"预算 10 只切了 {len(times)} 片"


def test_the_slice_lands_on_the_busiest_moment():
    """桶内落点取画面变化最剧烈的时刻。"""
    pts = []
    for m in range(3):
        for i in range(20):
            t = m * 60 + i * 3.0
            d = 9.0 if i == 11 else 0.5      # 第 12 个采样点是这一分钟的高峰
            pts.append((t, d))
    weights = density_weights({0: 1.0, 1: 1.0, 2: 1.0})
    times = vu.density_slices(pts, budget=3, span=(0.0, 3 * 60.0),
                              weights=weights, game_start_t=0.0)
    assert len(times) == 3
    for t in times:
        assert abs((t % 60) - 33.0) < 0.01, f"落点没有落在变化峰值：{t}"


def test_without_a_density_table_every_minute_gets_one():
    """没有密度表（别的游戏）就退化成每分钟一片，不乱切。"""
    times = vu.density_slices(_points(minutes=4), budget=4,
                              span=(0.0, 4 * 60.0), weights=None,
                              game_start_t=0.0)
    assert len(times) == 4
    assert {int(t // 60) for t in times} == {0, 1, 2, 3}
