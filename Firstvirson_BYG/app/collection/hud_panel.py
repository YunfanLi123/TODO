"""合成 HUD 面板：把「看不清的小东西」放大拼成一张图。

为什么要有这一层
----------------
王者画面里，模型真正需要读的东西都很小：

- 顶部计时器：数字只有十来个像素高；
- 左上角小地图：整块只占画面宽度的 22%；
- 右下角大招按钮：就一个圆。

把这些区域**裁出来、放大、拼成一张图**再和整帧一起给模型，
它才读得准。实测（2026-10-24，用例2 的第 2~4 分钟）模型读整帧时
已经把计时器读到了秒，但小地图的判断很粗 —— 合成面板就是补这一块。

**没标定的区域不画。** 画一个偏了几十像素的框，比不画更坏：
模型会照着错的东西答，而报告上看不出哪里错了。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

#: 面板里放哪几块、每块放大到多少像素宽。
#: 顺序按「模型最容易用错」排前面：先小地图（判断位置），再技能（状态），最后计时器。
PANEL_SPECS: List[Tuple[str, str, int]] = [
    ("minimap", "小地图", 460),
    ("skill_ult", "大招", 260),
    ("clock", "计时器", 200),
]


def _crop(frame, box: Tuple[float, float, float, float]):
    """按归一化框裁一块。框是 ``(x0, y0, x1, y1)``。"""
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    px0, py0 = max(0, int(x0 * w)), max(0, int(y0 * h))
    px1, py1 = min(w, int(x1 * w)), min(h, int(y1 * h))
    if px1 - px0 < 4 or py1 - py0 < 4:
        return None
    return frame[py0:py1, px0:px1].copy()


def _resize(img, width: int):
    import cv2

    h, w = img.shape[:2]
    if w == width:
        return img
    scale = width / float(w)
    return cv2.resize(img, (width, max(1, int(h * scale))),
                       interpolation=cv2.INTER_CUBIC)


def _label(img, text: str):
    """给一块图加个标题条 —— 不然模型不知道哪块是小地图。"""
    import cv2
    import numpy as np

    bar = 26
    h, w = img.shape[:2]
    out = np.zeros((h + bar, w, 3), dtype=np.uint8)
    out[bar:, :, :] = img
    # 标题用深色字（白底）
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (40, 40, 70), 1, cv2.LINE_AA)
    return out


def build_panel(frame, regions: Any, max_width: int = 1400) -> Optional[bytes]:
    """把 HUD 面板拼成一张 JPEG。

    ``regions`` 是 :class:`app.collection.recognizer.HudRegions`（已选好键位档位）。
    一块都裁不到就返回 None（**不拼一张空图糊弄模型**）。
    """
    import cv2

    if frame is None or regions is None:
        return None
    pieces: List[Any] = []
    for name, label, width in PANEL_SPECS:
        box = regions.box(name)
        if not box:
            continue
        piece = _crop(frame, box)
        if piece is None:
            continue
        piece = _resize(piece, width)
        pieces.append(_label(piece, label))
    if not pieces:
        return None

    # 横向拼；总宽超了就把每块按比例缩一遍。
    total = sum(p.shape[1] for p in pieces) + 8 * (len(pieces) - 1)
    if total > max_width:
        scale = max_width / float(total)
        pieces = [_resize(p, max(1, int(p.shape[1] * scale))) for p in pieces]
        total = sum(p.shape[1] for p in pieces) + 8 * (len(pieces) - 1)

    height = max(p.shape[0] for p in pieces)
    canvas = None
    import numpy as np

    canvas = np.full((height, total, 3), 255, dtype=np.uint8)
    x = 0
    for p in pieces:
        canvas[: p.shape[0], x: x + p.shape[1]] = p
        x += p.shape[1] + 8
    ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return None
    return bytes(buf)


def panel_available(regions: Any) -> bool:
    """有没有至少一块能画。"""
    if regions is None:
        return False
    for name, _label, _w in PANEL_SPECS:
        if regions.box(name):
            return True
    return False
