"""录屏规格检查：横屏 + 游戏画面占满全屏（用户定的标准）。

盯四件事：
1. **按画布判，不按容器判** —— 容器 960×464 的那段，游戏画面只占中间 210px；
2. **横屏** —— 王者是横屏游戏，画面竖着就是录反了；
3. **占满全屏** —— 剩下的是黑边，等于把可识别的像素砍掉；
4. **说清楚差在哪** —— 用户要知道该重录成什么样，而不是只听到「识别不了」。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from app.collection.video_upload import (  # noqa: E402
    FULLSCREEN_MIN_RATIO,
    recording_spec,
)


def test_a_standard_recording_passes():
    """横屏 + 占满全屏 + 够大 → 全部通过。"""
    out = recording_spec(1920, 1080, 1920, 1080)
    assert out["ok"] is True
    assert out["landscape"] is True
    assert out["fullscreen"] is True
    assert out["problems"] == []
    assert "符合标准" in out["note"]


def test_a_portrait_recording_is_rejected():
    out = recording_spec(1080, 1920, 1080, 1920)
    assert out["ok"] is False
    assert out["landscape"] is False
    assert any("竖" in p for p in out["problems"])


def test_a_letterboxed_recording_is_rejected():
    """手机画面只占中间一条 —— 这就是聊天软件转发那种。"""
    out = recording_spec(1080, 487, 1080, 2230)
    assert out["ok"] is False
    assert out["landscape"] is True
    assert out["fullscreen"] is False
    assert out["fill_ratio"] == pytest.approx(1080 * 487 / (1080 * 2230), abs=0.01)
    assert any("黑边" in p for p in out["problems"])


def test_a_small_recording_is_rejected_for_legibility():
    """横屏、占满全屏，但画布太小 → 数字只有几个像素，认不出。

    「规格」与「清晰度」是两件事：规格归 ``recording_spec``，
    清晰度归 ``recognition``（在 spec 里面，但不算进 problems）。
    """
    out = recording_spec(480, 218, 480, 218)
    assert out["ok"] is True          # 规格没问题：横屏 + 满屏
    assert out["recognition"]["ok"] is False
    assert out["recognition"]["digit_px"] == pytest.approx(
        25 * 218 / 1200, abs=0.1)


def test_the_fill_ratio_is_reported():
    out = recording_spec(1080, 487, 1080, 1080)
    assert out["fill_ratio"] == pytest.approx(487 / 1080, abs=0.01)
    assert out["fullscreen"] is (487 / 1080 >= FULLSCREEN_MIN_RATIO)
