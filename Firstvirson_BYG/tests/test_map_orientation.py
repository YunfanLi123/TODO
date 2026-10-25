"""小地图：中心反转（红方视角）与区域查询。

为什么测这个
------------
王者对**红方玩家**会把地图「中心反转」180°（腾讯官方《王者荣耀作战指南》：
「如果你的阵营为红方……界面会进行 180° 旋转，暴君、主宰、法阵等也会交换位置」）。
所以小地图上的坐标**不是**绝对地图坐标 —— 判错朝向，「在主宰坑」就会判成
「在暴君坑」，报告上完全看不出来。

这套测试盯三件事：
1. **朝向判据**：敌方防御塔是红色的（颜色不随旋转改变）；
2. **旋转是中心对称的**：旋转两次回到原处；
3. **判不出来就不猜**：宁可没有区域名，也不给一个错的。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from app.collection import minimap as mm  # noqa: E402


def _blank(h: int = 100, w: int = 100):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _paint(img, x0, y0, x1, y1, color):
    """画一块纯色（color 是 **BGR**，和 OpenCV 一致）。"""
    img[y0:y1, x0:x1] = color
    return img


def test_rotate_180_is_center_symmetric():
    """旋转两次回到原处 —— 旋转必须是**中心对称**的，不能是平移。"""
    nx, ny = 0.2, 0.8
    rx, ry = mm.rotate_180(nx, ny)
    assert abs(rx - 0.8) < 1e-9 and abs(ry - 0.2) < 1e-9
    bx, by = mm.rotate_180(rx, ry)
    assert abs(bx - nx) < 1e-9 and abs(by - ny) < 1e-9


def test_red_towers_top_right_means_blue_side():
    """红塔集中在**右上** → 蓝方视角（不反转）。"""
    img = _blank(120, 120)
    # 右上角放一簇红像素（模拟敌方防御塔）
    _paint(img, 95, 15, 112, 40, (30, 30, 220))
    assert mm.map_is_flipped(img) is False


def test_red_towers_bottom_left_means_red_side():
    """红塔集中在**左下** → 红方视角（地图被中心反转过，要旋 180°）。"""
    img = _blank(120, 120)
    _paint(img, 8, 80, 25, 105, (30, 30, 220))
    assert mm.map_is_flipped(img) is True


def test_unclear_orientation_returns_none():
    """分不出来就返回 None —— 宁可没有区域名，也不给一个错的。"""
    img = _blank(120, 120)
    assert mm.map_is_flipped(img) is None
    # 红色像素到处都有（像敌人头像）→ 也分不出来
    img2 = _blank(120, 120)
    for x in range(0, 120, 7):
        for y in range(0, 120, 7):
            _paint(img2, x, y, x + 5, y + 5, (30, 30, 220))
    assert mm.map_is_flipped(img2) is None


def test_zone_table_is_in_blue_side_coordinates():
    """区域表按**蓝方基准**存；红方视角要先转 180° 再查。"""
    geo = mm.MapGeometry(regions=[{"name": "中路", "box": [0.40, 0.40, 0.50, 0.60]}])
    # 蓝方视角：小地图上的 (0.45, 0.5) 落在中路
    assert geo.zone_of(0.45, 0.5, flipped=False) == "中路"
    # 红方视角：小地图上的 (0.45, 0.5) 要先旋转 → 蓝方的 (0.55, 0.5)，不在中路
    assert geo.zone_of(0.45, 0.5, flipped=True) == ""
    assert geo.zone_of(0.55, 0.5, flipped=True) == "中路"
    # 中心点旋转后不动，两种视角都在中路
    assert geo.zone_of(0.45, 0.5, flipped=False) == "中路"


def test_empty_region_table_gives_no_zone():
    """区域表为空 → 没有区域名（不猜）。"""
    geo = mm.MapGeometry()
    assert geo.zone_of(0.5, 0.5) == ""


def test_the_shipped_geometry_file_documents_the_rotation():
    """标定文件里必须写清朝向约定 —— 不写清楚，后来人一定会按绝对坐标用。"""
    import json

    p = APP_ROOT / "app" / "collection" / "map_geometry.json"
    data = json.loads(p.read_text("utf-8"))
    blob = json.dumps(data, ensure_ascii=False)
    assert "180" in blob and "红方" in blob, "标定文件没写清 180° 中心反转"
