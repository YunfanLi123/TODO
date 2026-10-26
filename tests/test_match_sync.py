"""局内时间同步：对局起点检测 + 两条来源互验 + 时钟可读性闸门。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.collection import clock, match_sync
from app.collection.video_upload import (
    JOYSTICK_BRIGHT_MAX,
    in_match,
)


# ---------------------------------------------------------------- 对局起点


def test_match_start_takes_longest_run_not_first_hit():
    """BP 界面偶尔也会撞上对局图案，起点必须取**最长连续段**。

    用户说过：不同对局视频进入对局的时间不同 —— 起点是测出来的，
    不能取「第一次撞上」的那一刻。
    """
    samples = []
    samples += [(t, False) for t in range(0, 10)]
    # BP 界面里的巧合段：只有 4 秒
    samples += [(t, True) for t in range(10, 14)]
    samples += [(t, False) for t in range(14, 120)]
    # 真正的对局：一路持续到录屏结束
    samples += [(t, True) for t in range(120, 400)]

    out = match_sync.detect_match_start(samples)
    assert out.ok
    assert out.t == 120
    assert out.source == "hud"
    assert "第 120 秒" in out.reason


def test_match_start_needs_a_long_enough_run():
    """只撞上几秒不算对局。"""
    samples = [(t, False) for t in range(0, 20)]
    samples += [(t, True) for t in range(20, 22)]
    out = match_sync.detect_match_start(samples)
    assert not out.ok
    assert out.t is None
    assert "不敢认" in out.reason


def test_match_start_without_any_match():
    samples = [(t, False) for t in range(0, 100)]
    out = match_sync.detect_match_start(samples)
    assert not out.ok
    assert "BP" in out.reason or "结算" in out.reason


def test_match_start_empty():
    out = match_sync.detect_match_start([])
    assert not out.ok
    assert out.t is None


def test_match_start_sorted_input():
    """输入乱序也要能用。"""
    samples = [(t, t >= 50) for t in range(0, 200)]
    out = match_sync.detect_match_start(list(reversed(samples)))
    assert out.ok
    assert out.t == 50


def test_match_start_merges_short_gaps():
    """对局是连续的一整场，中间不该有洞。

    实测：技能特效扫过左下角会让判据断掉一次（最长约 9 秒），
    断点两侧仍然是同一场对局 —— 起点要取整段的开头，不是最后一段的开头。
    """
    samples = [(t, False) for t in range(0, 20)]
    samples += [(t, True) for t in range(20, 60)]
    samples += [(t, False) for t in range(60, 68)]        # 8 秒的洞
    samples += [(t, True) for t in range(68, 300)]

    out = match_sync.detect_match_start(samples)
    assert out.ok
    assert out.t == 20
    # 平滑会把两端各修掉一点点，所以不是整 280 秒
    assert out.run_sec == pytest.approx(280, abs=3)


def test_match_start_does_not_merge_long_gaps():
    """BP 与对局之间隔着几十秒，不能被合并到一起。"""
    samples = [(t, True) for t in range(0, 10)]
    samples += [(t, False) for t in range(10, 200)]       # 190 秒的间隔
    samples += [(t, True) for t in range(200, 400)]

    out = match_sync.detect_match_start(samples)
    assert out.ok
    assert out.t == 200


def test_game_seconds_at():
    assert match_sync.game_seconds_at(100.0, 160.0) == pytest.approx(60.0)
    assert match_sync.game_seconds_at(100.0, 50.0) == 0.0     # 不给负数
    assert match_sync.game_seconds_at(None, 160.0) is None     # 没测出来就不猜


def test_cross_check_consistent():
    out = match_sync.cross_check(120.0, 122.0)
    assert out is not None
    assert out["consistent"] is True
    assert out["diff_sec"] == pytest.approx(-2.0)


def test_cross_check_inconsistent():
    out = match_sync.cross_check(120.0, 300.0)
    assert out is not None
    assert out["consistent"] is False
    assert "相差" in out["note"]


def test_cross_check_needs_both():
    assert match_sync.cross_check(None, 100.0) is None
    assert match_sync.cross_check(100.0, None) is None


# ---------------------------------------------------------------- 画面判据


def _feats(joy_bright: float, minimap_edges: float = 100.0):
    """造一份四角特征：左下（摇杆）亮度可调，左上（小地图）边密度最高。"""
    return {
        "上左": (minimap_edges, 0.02, 0),
        "上右": (50.0, 0.02, 0),
        "下左": (40.0, joy_bright, 0),
        "下右": (45.0, 0.02, 0),
    }


LAY = {"minimap": "上左", "skills": "下右", "joystick": "下左"}


def test_in_match_accepts_dark_joystick():
    """对局里左下角压着摇杆，整块是暗的。"""
    assert in_match(_feats(0.005), LAY) is True


def test_in_match_rejects_bright_bottom_left():
    """BP / 选人 / 加载界面的左下角是队友头像列表，一堆亮图标。"""
    assert in_match(_feats(0.06), LAY) is False


def test_in_match_threshold_boundary():
    assert in_match(_feats(JOYSTICK_BRIGHT_MAX), LAY) is True
    assert in_match(_feats(JOYSTICK_BRIGHT_MAX + 0.001), LAY) is False


def test_in_match_empty_features():
    assert in_match({}, LAY) is False


# ---------------------------------------------------------------- 时钟可读性


def test_digit_height_scales_with_canvas():
    """数字大小随画布高度缩放。"""
    assert clock.digit_height_px(1200) == pytest.approx(clock.SLOT_H)
    assert clock.digit_height_px(240) == pytest.approx(clock.SLOT_H * 0.2)


def test_too_small_to_read_gate():
    """画布太矮就别认了 —— 认出来的是噪声（实测会读出 1111/1144）。"""
    assert clock.too_small_to_read(218) is True      # 聊天软件压缩过的录屏
    assert clock.too_small_to_read(1080) is False    # 正常手机录屏


def test_legibility_note_says_the_number():
    note = clock.legibility_note(218)
    assert "218" in note
    assert "读不出局内时间" in note
    assert "像素" in note


def test_read_clock_anchored_refuses_tiny_canvas():
    """小画布上就算给了锚点也认不出 —— 返回「没读到」而不是乱码。"""
    import numpy as np

    frame = np.zeros((220, 480, 3), dtype=np.uint8)
    anchor = {"rotation": 0, "x": 1800, "score": 0.9}
    assert clock.read_clock_anchored(frame, anchor) == (None, 0.0)


def test_detect_anchor_refuses_tiny_canvas():
    import numpy as np

    frame = np.zeros((220, 480, 3), dtype=np.uint8)
    assert clock.detect_anchor(frame, rotations=(0,)) is None
