"""分析单元改成时间窗：窗口推导、分组、事实合并。

用户的要求：「逐时间片标注不适合王者」「要在一个时间区间对帧做综合分析」。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.collection.window_analysis import (
    Window,
    aggregate_actions,
    aggregate_facts,
    bundle_slices,
    merge_windows,
    rule_windows,
    segments,
    window_for_minute,
    windows_from_conditions,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "app" / "seed"


def _seed_rules() -> list:
    out = []
    for name in ("hok_modes.json", "hok_hero_modes.json"):
        p = SEED / name
        if p.exists():
            out.extend(json.loads(p.read_text(encoding="utf-8")))
    return out


# ---------------------------------------------------------------- 窗口推导


def test_windows_from_conditions():
    wins = windows_from_conditions([
        {"fact": "time_min", "op": "between", "value": [2, 3]},
        {"fact": "position", "op": "==", "value": "jungle"},
        {"fact": "time_min", "op": "==", "value": 4},
    ], merge=False)
    assert [(w.lo, w.hi) for w in wins] == [(2, 3), (4, 4)]


def test_merge_windows_merges_adjacent():
    ws = merge_windows([(0, 1), (2, 3), (6, 9)])
    assert [(w.lo, w.hi) for w in ws] == [(0, 3), (6, 9)]


def test_merge_windows_can_keep_gaps():
    """相邻但中间有空档的窗，默认可以并；分析段要的是「只并重叠的」。"""
    ws = merge_windows([(0, 1), (2, 3), (4, 4)], adjacent=False)
    assert [(w.lo, w.hi) for w in ws] == [(0, 1), (2, 3), (4, 4)]


def test_rule_windows_from_seed_are_seven():
    """王者 S45 的规则库只用 8 种时间窗 —— 这就是天然的判定单元。"""
    wins = rule_windows(_seed_rules())
    assert len(wins) == 7
    assert [(w.lo, w.hi) for w in wins] == [
        (0, 0), (0, 1), (2, 3), (4, 4), (6, 9), (10, 10), (12, 60),
    ]


def test_segments_merge_to_six():
    """8 个判定单元合成 6 个分析段 —— 玩家/识别器对着 6 段做综合分析。"""
    segs = segments(rule_windows(_seed_rules()))
    assert [(w.lo, w.hi) for w in segs] == [
        (0, 1), (2, 3), (4, 4), (6, 9), (10, 10), (12, 60),
    ]


def test_every_rule_window_starts_inside_its_own_window():
    """判定单元的起点必须落在它自己的窗里。

    否则「规则在第 1 分钟」这种规则会永远不触发 —— 报告上只是「没结论」，
    看不出是哪里错了。
    """
    for w in rule_windows(_seed_rules()):
        assert w.contains(w.lo)


def test_window_label():
    assert Window(4, 4).label() == "4:00"
    assert Window(2, 3).label() == "2:00 ~ 4:00"
    assert Window(12, 60).label() == "12:00 之后"


def test_window_for_minute():
    wins = [Window(0, 1), Window(2, 3), Window(4, 4)]
    assert window_for_minute(wins, 0).key == "0-1"
    assert window_for_minute(wins, 3).key == "2-3"
    assert window_for_minute(wins, 5) is None      # 没有规则的分钟不硬塞


# ---------------------------------------------------------------- 分组


def test_bundle_slices_groups_by_window():
    slices = [
        {"round_no": 1, "game_minute": 0, "timestamp": 1.0},
        {"round_no": 2, "game_minute": 1, "timestamp": 2.0},
        {"round_no": 3, "game_minute": 4, "timestamp": 3.0},
        {"round_no": 4, "game_minute": 5, "timestamp": 4.0},   # 无规则的分钟
    ]
    wins = [Window(0, 1), Window(4, 4)]
    bundles = bundle_slices(slices, wins)
    assert [b.window.key for b in bundles] == ["0-1", "4-4"]
    assert len(bundles[0].slices) == 2
    assert len(bundles[1].slices) == 1


def test_bundle_slices_lets_one_slice_join_several_windows():
    """规则窗可能重叠（第 1 分钟同时属于 0~1 和 1~1）。

    同一片证据同时服务两个窗 —— 它们判的是不同批规则，证据相同。
    """
    slices = [{"round_no": 1, "game_minute": 1, "timestamp": 1.0}]
    wins = [Window(0, 1), Window(1, 1)]
    bundles = bundle_slices(slices, wins)
    assert len(bundles) == 2
    assert all(len(b.slices) == 1 for b in bundles)


def test_bundle_drops_minutes_no_rule_covers():
    slices = [{"round_no": 1, "game_minute": 5, "timestamp": 1.0}]
    assert bundle_slices(slices, [Window(0, 1)]) == []


# ---------------------------------------------------------------- 事实合并


def test_aggregate_facts_keeps_only_consistent():
    """不一致的事实在窗内本来就会变（打了人就变）—— 不给就是编数字。"""
    out = aggregate_facts([
        {"gold_diff": -200, "kill_diff": 0},
        {"gold_diff": -200, "kill_diff": 1},
    ])
    assert out == {"gold_diff": -200}


def test_aggregate_facts_empty():
    assert aggregate_facts([]) == {}


def test_aggregate_actions_unions_and_dedupes():
    class A:
        def __init__(self, fact, value):
            self.fact = fact
            self.value = value

    out = aggregate_actions([
        [A("move", "lane")],
        [A("move", "lane"), A("gank", "mid")],
    ])
    assert len(out) == 2
