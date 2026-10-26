"""键位档位：右下角按钮的检测与档位选择。

盯三件事：
1. **跨帧稳定**才认按钮 —— 草地上的假圆每帧位置都不同，会被滤掉；
2. **档位能选对** —— 用例1 的大招在 (0.85, 0.582)、用例2 的在 (0.794, 0.610)，
   两段录屏的画布、时钟、小地图位置完全一致，只有技能簇不同；
3. **认不出档位就不读大招** —— 框对不上时读到的是别的按钮，
   「就绪/冷却」会变成随机数，而报告上只是「没触发」，看不出错在哪。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from app.collection import skillbar                       # noqa: E402
from app.collection.recognizer import HudRegions          # noqa: E402

W, H = 1280, 574


def _canvas(buttons=(), noise=()) -> np.ndarray:
    """造一幅画面：在指定位置画**圆环**（模拟圆形按钮的外圈）。

    ``buttons`` 是 (x, y, r) 的像素坐标；``noise`` 同义，画在别的位置。
    画圆环而不是实心圆：按钮本来就是个圈，而实心圆在纯色背景上
    Hough 的边缘点太少，检不出来（那是合成图的毛病，不是算法的毛病）。
    """
    import cv2

    img = np.full((H, W, 3), 40, dtype=np.uint8)
    for x, y, r in list(buttons) + list(noise):
        cv2.circle(img, (int(x), int(y)), int(r), (200, 190, 180), 3)
    return img


def _buttons_in(img, xs, ys):
    import cv2

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=40,
                               param1=100, param2=28, minRadius=17, maxRadius=63)
    if circles is None:
        return []
    return [(int(x), int(y), int(r)) for x, y, r in np.round(circles[0]).astype(int)]


def test_detect_buttons_finds_circles_in_the_right_area():
    img = _canvas(buttons=[(1088, 334, 36), (1096, 494, 55)])
    got = skillbar.detect_buttons(img)
    assert got, "右下角画了圆却一个都没检出"
    for b in got:
        assert b.x * W > W * skillbar.REGION_X0 - 1, f"检出了区域外的按钮 {b}"
        assert b.y * H > H * skillbar.REGION_Y0 - 1, f"检出了区域外的按钮 {b}"


def test_stable_buttons_drops_moving_false_circles():
    """每帧位置都不同的「假按钮」必须被滤掉；固定位置的留下。"""
    fixed = (1088, 334, 36)
    frames = []
    for i in range(10):
        # 假按钮每帧挪位置（像草地上的纹理被 Hough 误检）
        fake_x = 900 + i * 7
        frames.append(_canvas(buttons=[fixed], noise=[(fake_x, 300, 30)]))
    got = skillbar.stable_buttons(frames)
    assert len(got) == 1, f"应该只剩一个稳定按钮，实际 {len(got)} 个"
    assert abs(got[0].x * W - fixed[0]) < 12
    assert abs(got[0].y * H - fixed[1]) < 12


def test_select_profile_picks_the_right_layout():
    """两个档位的大招位置不同，按钮群决定选哪一档。"""
    reg = HudRegions.load()
    assert reg.profiles, "标定文件里应该有档位"
    case1 = skillbar.Button(1088 / W, 334 / H, 36 / H)
    case2 = skillbar.Button(1016 / W, 350 / H, 31 / H)
    assert reg.select_profile([case1]) == "标准键位"
    assert reg.select_profile([case2]) == "熬隐键位"


def test_no_profile_means_ult_is_not_read():
    """认不出档位 → 大招框不可信 → 不读（记缺失），不给随机数。"""
    reg = HudRegions.load()
    reg.select_profile([])
    assert reg.profile_name is None
    assert not reg.calibrated("skill_ult"), \
        "没匹配到档位时不该信任默认的大招框"
    assert "键位档位没认出来" in reg.layout_note()


def test_selected_profile_overrides_the_ult_box():
    """选中的档位用自己的大招框，不用公共区域那个。"""
    reg = HudRegions.load()
    reg.select_profile([skillbar.Button(1016 / W, 350 / H, 31 / H)])
    box = reg.box("skill_ult")
    assert box == (0.7625, 0.5453, 0.825, 0.6742)
    assert reg.calibrated("skill_ult")


def test_other_regions_are_not_layout_sensitive():
    """时钟、击杀数这些跟键位无关，认不出档位也照常用。"""
    reg = HudRegions.load()
    reg.select_profile([])
    assert reg.calibrated("clock")
    assert reg.calibrated("scoreboard_kill_left")
