"""方向判定：多数票 + 最终画布必须横屏。

实测背景（720p 录屏，12.5 分钟）：手机竖着录横屏游戏，游戏在画面里是
**躺着**的。单帧最高分被 BP 选人界面那一帧抢走（它没有小地图，布局判据
纯属巧合却得分最高），于是方向判反 —— 画布里小地图在右下、摇杆在右上，
时钟、锚点、切片全部读不对。

盯三件事：
1. **多数票**：一帧的巧合不能压过好几帧的一致；
2. **最终画布必须横屏**（王者是横屏游戏）；
3. 识别可行性判定要按**画布**说，不能按容器分辨率说。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from app.collection import clock  # noqa: E402
from app.collection.video_upload import _vote_rotation  # noqa: E402


# ---------------------------------------------------------------- 投票


def test_majority_beats_one_lucky_high_score():
    """一帧得分最高不算数，多数才算。"""
    choices = [
        {"rotation": 270, "score": 1.95},
        {"rotation": 270, "score": 1.93},
        {"rotation": 270, "score": 1.97},
        {"rotation": 180, "score": 2.03},      # BP 界面那帧：单帧最高
    ]
    assert _vote_rotation(choices) == 270


def test_ties_are_broken_by_total_score():
    choices = [
        {"rotation": 90, "score": 1.9},
        {"rotation": 90, "score": 1.8},
        {"rotation": 270, "score": 1.7},
        {"rotation": 270, "score": 1.6},
    ]
    assert _vote_rotation(choices) == 90


def test_vote_is_reproducible():
    """同分同票要给出**同一个**结果（不然每次跑出来可能不一样）。"""
    choices = [
        {"rotation": 90, "score": 2.0},
        {"rotation": 270, "score": 2.0},
    ]
    assert _vote_rotation(choices) == _vote_rotation(list(choices))


def test_empty_choices_are_not_voted():
    # 空列表不能抛（detect_view 里有判空，但投票函数自己也该稳）
    with pytest.raises(ValueError):
        _vote_rotation([])


# ---------------------------------------------------------------- 识别可行性


def test_recognition_feasibility_says_what_is_missing():
    """判定要**说清楚差在哪**，不能只说「识别不了」。"""
    out = clock.recognition_feasibility(464, 211)
    assert out["ok"] is False
    assert "336" in out["note"] and "545" in out["note"]
    assert "像素" in out["note"]


def test_recognition_feasibility_passes_a_clear_canvas():
    out = clock.recognition_feasibility(1920, 1080)
    assert out["ok"] is True
    assert "够自动识别" in out["note"]
    assert out["digit_px"] == pytest.approx(25 * 1080 / 1200, abs=0.1)


def test_recognition_feasibility_uses_the_canvas_not_the_container():
    """容器 960×464 但画布只有 464×211 —— 按画布算就是不够。"""
    out = clock.recognition_feasibility(464, 211)
    assert out["ok"] is False
    assert out["digit_px"] == pytest.approx(25 * 211 / 1200, abs=0.1)
