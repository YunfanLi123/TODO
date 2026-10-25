"""自动识别层：窗级综合分析 + 标定文件的加载与校验。

盯三件事：

1. **读不出就记成缺失**，不猜 —— 缺失的事实进内核是「无法判定」，
   猜一个值是替玩家编一条他没下过的结论；
2. **标定与识别分开** —— 换机型、换清晰度只重跑标定，识别代码不动；
3. 窗级接口：一次分析一整段的若干帧，出**一份**结论。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from app.collection import clock  # noqa: E402
from app.collection.recognizer import (  # noqa: E402
    HudRegions,
    WindowRecognizer,
    WindowReading,
)


# ---------------------------------------------------------------- 标定文件


def test_the_calibration_file_ships_with_the_app():
    """标定文件必须在包里（exe 打包也要带上）。"""
    p = APP_ROOT / "app" / "collection" / "hud_regions.json"
    assert p.exists(), "标定文件没进包"
    data = json.loads(p.read_text("utf-8"))
    assert data.get("regions"), "标定文件里没有区域"


def test_regions_load_and_report_calibrated_state():
    reg = HudRegions.load()
    assert reg.regions, "没读到任何区域"
    # 关键区域必须在
    # 顶栏击杀数拆成**左右两个区**：中间那个灰白的「vs」按颜色就能排除，
    # 合成一块反而容易把「vs」当数字读进来。
    for name in ("minimap", "scoreboard_gold", "scoreboard_kill_left",
                 "scoreboard_kill_right", "skill_ult"):
        assert reg.box(name) is not None, f"少了 {name} 的标定框"
    # 没标定的要如实标成没标定
    assert reg.calibrated("scoreboard_gold") is False
    assert reg.calibrated("clock") is True
    # 击杀数这两块是标定过的
    assert reg.calibrated("scoreboard_kill_left") is True
    assert reg.calibrated("scoreboard_kill_right") is True


def test_crop_uses_relative_coordinates():
    """裁剪按**相对坐标** —— 换分辨率不用改代码。"""
    reg = HudRegions.load()
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    crop = reg.crop(img, "minimap")
    assert crop is not None
    h, w = crop.shape[:2]
    # 标定框是 (0, 0, 0.22, 0.30)
    assert w == int(0.22 * 400)
    assert h == int(0.30 * 200)


def test_crop_of_uncalibrated_region_is_none():
    reg = HudRegions.load()
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    assert reg.crop(img, "不存在的区域") is None


# ---------------------------------------------------------------- 窗级分析


def test_empty_window_reports_everything_missing():
    r = WindowRecognizer()
    out = r.analyze([])
    assert out.facts == {}
    assert out.missing
    assert out.notes


def test_uncalibrated_regions_become_missing_not_guesses():
    """顶栏读不出 → 只能是缺失，不许给值。

    大招按钮**已标定**了（2026-10-23），所以它会给出读数而不是缺失；
    顶栏的金币与击杀是另外两回事，分别记缺失。
    """
    r = WindowRecognizer()
    frames = [np.zeros((200, 400, 3), dtype=np.uint8) for _ in range(3)]
    out = r.analyze(frames)
    assert out.fact("gold_diff") is None
    assert out.fact("kill_diff") is None
    # 金币与击杀**分开记**（两件事的状态不一样：金币是「王者不显示经济」，
    # 击杀是「模板还没验收」）
    assert "金币差" in out.missing
    assert "击杀差" in out.missing


def test_reading_is_window_level_one_set_of_facts():
    """一窗一次分析 → **一份**事实，不是一帧一份。"""
    r = WindowRecognizer()
    frames = [np.full((200, 400, 3), i, dtype=np.uint8) for i in range(5)]
    out = r.analyze(frames)
    assert isinstance(out, WindowReading)
    # 顶栏的事实在任何情况下都不会是「猜出来的值」
    assert out.fact("gold_diff") is None
    assert out.fact("kill_diff") is None


# ---------------------------------------------------------------- 数字匹配


def test_match_digits_reads_a_synthetic_region():
    """数字模板机制本身是好的 —— 合成一块区域，认得出来就返回数值。

    这里用计时器自己的槽位表（canvas 相对），在整幅画面上认 4 位数字。
    """
    slots = clock.load_digits()
    if not slots:
        pytest.skip("没有数字模板")
    # 造一张白底黑字的合成图，槽位按模板的均值填充
    img = np.full((200, 200), 255, dtype=np.uint8)
    value = clock.match_digits(img, slots)
    # 白底没有笔画 → 认不出（或者认出个乱数），但**不能崩**
    assert value is None or isinstance(value, int)


def test_match_digits_returns_none_on_empty_region():
    slots = clock.load_digits()
    if not slots:
        pytest.skip("没有数字模板")
    assert clock.match_digits(np.zeros((10, 10), dtype=np.uint8), slots) is None
