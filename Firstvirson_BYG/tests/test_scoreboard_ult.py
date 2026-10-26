"""识别层：顶栏与大招的读数判据。

盯三件事：
1. **大招就绪**：靠按钮中心的**色彩饱和度**（就绪=彩色、冷却=灰暗），
   阈值取在两簇之间；
2. **击杀差**：模板**没验收就不启用** —— 宁可记缺失，也不给可能读错的数；
3. **英雄位置**：区域判定**与阵营无关**（红方视角地图被中心反转 180°，
   但主宰/暴君的图标颜色是绝对的）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from app.collection import minimap as mm  # noqa: E402
from app.collection import scoreboard as sb  # noqa: E402
from app.collection import skillbar  # noqa: E402
from app.collection.recognizer import (  # noqa: E402
    HudRegions,
    WindowRecognizer,
)


def _frame_with_ult(color: tuple) -> np.ndarray:
    """造一整幅画面，只把大招按钮那个位置画成指定颜色。

    标定区域是**相对坐标**，所以按钮必须画在它该在的地方。
    """
    reg = HudRegions.load()
    box = reg.box("skill_ult")
    w, h = 1000, 600
    img = np.full((h, w, 3), 25, dtype=np.uint8)
    x0, y0, x1, y1 = box
    px0, py0 = int(x0 * w), int(y0 * h)
    px1, py1 = int(x1 * w), int(y1 * h)
    # 按钮的外圈是暗的，中间 50% 是技能图标
    img[py0:py1, px0:px1] = color[::-1]
    cx0, cy0 = px0 + (px1 - px0) // 4, py0 + (py1 - py0) // 4
    cx1, cy1 = px1 - (px1 - px0) // 4, py1 - (py1 - py0) // 4
    img[cy0:cy1, cx0:cx1] = (40, 40, 40)          # 外圈暗
    img[cy0 + 4:cy1 - 4, cx0 + 4:cx1 - 4] = color[::-1]   # 中心是图标
    return img


def _standard_layout_regions() -> HudRegions:
    """区域表 + **已选中的键位档位**。

    大招框是「档位敏感」的：王者键位能自己拖，不同玩家的技能簇能差
    几十像素，框对不上读到的是**别的按钮**。所以读大招之前必须先
    认出错位档位 —— 这里显式选「标准键位」，等价于流水线里
    「扫到按钮群 → 匹配上标准键位」那一步。
    """
    reg = HudRegions.load()
    reg.select_profile([skillbar.Button(0.85, 0.5819, 0.0600)])
    return reg


def test_ult_ready_reads_colour_of_the_button():
    """就绪=彩色、冷却=灰暗 —— 两者差得很开，阈值好取。"""
    r = WindowRecognizer(_standard_layout_regions())
    ready = _frame_with_ult((60, 180, 220))      # 青色的技能图标
    cooling = _frame_with_ult((70, 70, 70))     # 灰暗的冷却遮罩
    out = r.analyze([ready] * 5)
    assert out.fact("has_ult_ready") is True
    out2 = r.analyze([cooling] * 5)
    assert out2.fact("has_ult_ready") is False


def test_ult_ready_takes_the_majority():
    """一帧被特效盖住不算数 —— 取多数。"""
    r = WindowRecognizer(_standard_layout_regions())
    ready = _frame_with_ult((60, 180, 220))
    cooling = _frame_with_ult((70, 70, 70))
    frames = [ready] * 4 + [cooling] * 1
    assert r.analyze(frames).fact("has_ult_ready") is True


def test_ult_is_not_read_without_a_matched_layout():
    """没认出键位 → **不读**大招（记缺失），不给随机数。

    框偏了几十像素时读到的是**别的按钮**，「就绪/冷却」会变成随机数 ——
    而报告上只是「没触发」，根本看不出错在哪。所以宁可不给。
    """
    reg = HudRegions.load()
    reg.select_profile([])          # 一个档位都没匹配上
    r = WindowRecognizer(reg)
    ready = _frame_with_ult((60, 180, 220))
    out = r.analyze([ready] * 5)
    assert out.fact("has_ult_ready") is None
    assert "大招就绪" in out.missing



def test_unverified_templates_are_not_used():
    """模板没验收 → 读数返回 None（而不是给一个可能错的数）。"""
    assert sb.load_templates() == {}, "未验收的模板不应该被启用"
    crop = np.zeros((20, 30, 3), dtype=np.uint8)
    assert sb.read_number(crop, "left") is None


def test_lane_judgement_is_the_same_after_rotating_180():
    """**视角无关**：地图被中心反转 180° 之后，「对抗路」还是对抗路。

    这是王者机制决定的：不管哪一方，主宰图标所在的那一侧永远是
    玩家的对抗路（蓝方视角主宰在对抗路上半区；红方视角地图转了
    180°，主宰转到下半区，而红方玩家的对抗路也在下半区）。
    """
    marks = {"主宰": (0.30, 0.28), "暴君": (0.70, 0.74)}
    flipped = {"主宰": (1 - 0.30, 1 - 0.28), "暴君": (1 - 0.70, 1 - 0.74)}
    for pt in ((0.2, 0.2), (0.8, 0.8), (0.5, 0.5)):
        assert mm.lane_of(pt[0], pt[1], marks) == \
            mm.lane_of(1 - pt[0], 1 - pt[1], flipped)


def test_objective_marks_decide_the_pit():
    """站在龙坑图标附近 → 那个坑的名字。"""
    geo = mm.MapGeometry()
    marks = {"主宰": (0.30, 0.28), "暴君": (0.70, 0.74)}
    assert geo.zone_of(0.30, 0.28, marks) == "主宰坑"
    assert geo.zone_of(0.70, 0.74, marks) == "暴君坑"
    assert geo.zone_of(0.5, 0.5, marks) == "中路附近"
