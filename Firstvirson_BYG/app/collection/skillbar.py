"""右下角技能按钮的检测，以及「键位档位」的匹配。

为什么需要这一层
----------------
王者荣耀的**键位是能自己拖的**：同一款游戏，不同玩家的技能按钮位置能差
几十像素。这不是「换机型」，是「换人」。

大招按钮位置一旦对不上，饱和度判据就会读到**别的按钮**上 ——
「就绪/冷却」变成随机数，而报告上只会显示「没触发」，看不出错在哪。
实测：测试用例1（赵云）的大招在 (1088,334)，测试用例2（熬隐）的大招在
(1014,350)，两个视频的画布分辨率、时钟位置、小地图位置**完全一致**，
只有技能簇的位置不同。

所以这里**不猜**：把右下角那一圈圆形按钮**跨帧稳定地**检测出来，
当成一份签名，跟标定文件里存好的档位比对：

- 匹配上某一档 → 用那一档的大招框；
- 一档都匹配不上 → **记缺失**，并在报告里写明「键位没认出来」。

跨帧稳定这一条是关键：单帧里 Hough 会在草地上检出大量假圆，
但真按钮在每一帧都在同一个位置 —— 位置对不上的就是噪声。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: 只看右下角这一块。技能键位都在这儿。
REGION_X0 = 0.55
REGION_Y0 = 0.45

#: 圆形按钮的半径范围（相对画布高度）。太大的是普攻键以外的东西，
#: 太小的只是图标细节点，按钮描边没被 Hough 当成圆。
MIN_RADIUS = 0.030
MAX_RADIUS = 0.110

#: 按钮中心归一化坐标的容差：差这么多以内算同一个按钮。
MERGE_TOL = 0.014

#: 同一个按钮在各帧里的**位置散布**上限（像素）。
#:
#: 为什么要单独一条：光靠「距离近就并到同一簇」，**缓慢漂移**的假圆会被
#: 链式合并 —— 每帧挪 7 像素，十帧挪了 70 像素，仍然被当成同一个按钮。
#: 而真实世界里镜头一直在动，草地上的一个纹理在屏幕上就是**慢慢挪**的，
#: 它恰恰是最像「稳定按钮」的噪声。
#:
#: 真按钮的检测位置每帧只抖几个像素；漂移的假圆散布几十像素。
#: 所以按**簇内的位置散布**再筛一道。
SPREAD_MAX_PX = 14.0

#: 一个按钮要在多少比例的帧里都出现，才算「稳定」。
STABLE_RATIO = 0.40


@dataclass(frozen=True)
class Button:
    """一个圆形按钮。坐标与半径都是**归一化**的（相对画布宽/高）。"""

    x: float
    y: float
    r: float

    def to_dict(self) -> Dict[str, float]:
        return {"x": round(self.x, 4), "y": round(self.y, 4), "r": round(self.r, 4)}


def detect_buttons(frame) -> List[Button]:
    """单帧里检出右下角的圆形按钮。

    用 Hough 圆检测 —— 技能键位是**正圆**，这个形状特征比颜色可靠。
    """
    import cv2
    import numpy as np

    if frame is None or getattr(frame, "size", 0) == 0:
        return []
    h, w = frame.shape[:2]
    x0, y0 = int(w * REGION_X0), int(h * REGION_Y0)
    region = frame[y0:h, x0:w]
    if region.size == 0:
        return []
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=40,
                               param1=100, param2=28,
                               minRadius=int(h * MIN_RADIUS),
                               maxRadius=int(h * MAX_RADIUS))
    out: List[Button] = []
    if circles is None:
        return out
    for cx, cy, r in np.round(circles[0]).astype(int):
        bx = (cx + x0) / float(w)
        by = (cy + y0) / float(h)
        br = r / float(h)
        if not (MIN_RADIUS <= br <= MAX_RADIUS):
            continue
        b = Button(bx, by, br)
        # 同一帧里同一个圆可能被检出多次，去重
        if any(abs(b.x - o.x) < MERGE_TOL and abs(b.y - o.y) < MERGE_TOL
               and abs(b.r - o.r) < MERGE_TOL for o in out):
            continue
        out.append(b)
    return out


def stable_buttons(frames: Sequence[Any],
                   ratio: float = STABLE_RATIO) -> List[Button]:
    """跨帧统计，只留下**在大多数帧里都在同一位置**的按钮。

    草地上的假圆每帧位置都不同，会被这一步滤掉；
    真按钮位置固定，留下来。

    两道筛：
    1. **出现得多**（>= ``ratio`` 比例的帧里都在）；
    2. **位置不散**（簇内位置的散布 <= :data:`SPREAD_MAX_PX` 像素）——
       镜头在动时草地上的纹理会缓慢漂移，它出现得也多，但位置散布大，
       第 2 道筛才拦得住它。
    """
    frames = [f for f in frames if f is not None and getattr(f, "size", 0) > 0]
    if not frames:
        return []
    h, w = frames[0].shape[:2]
    seen: List[Dict[str, Any]] = []
    for f in frames:
        for b in detect_buttons(f):
            hit = None
            for s in seen:
                if abs(b.x - s["x"]) < MERGE_TOL and abs(b.y - s["y"]) < MERGE_TOL:
                    hit = s
                    break
            if hit is None:
                seen.append({"x": b.x, "y": b.y, "r": b.r, "n": 1,
                             "minx": b.x, "maxx": b.x,
                             "miny": b.y, "maxy": b.y})
            else:
                hit["n"] += 1
                # 位置取平均，稳一点
                hit["x"] = (hit["x"] * (hit["n"] - 1) + b.x) / hit["n"]
                hit["y"] = (hit["y"] * (hit["n"] - 1) + b.y) / hit["n"]
                hit["minx"] = min(hit["minx"], b.x)
                hit["maxx"] = max(hit["maxx"], b.x)
                hit["miny"] = min(hit["miny"], b.y)
                hit["maxy"] = max(hit["maxy"], b.y)
    need = max(2, int(len(frames) * ratio))
    out: List[Button] = []
    for s in seen:
        if s["n"] < need:
            continue
        spread = max((s["maxx"] - s["minx"]) * w, (s["maxy"] - s["miny"]) * h)
        if spread > SPREAD_MAX_PX:
            continue
        out.append(Button(s["x"], s["y"], s["r"]))
    return sorted(out, key=lambda b: (b.y, b.x))


def match_button(buttons: Sequence[Button], target: Dict[str, float],
                 tol: float = MERGE_TOL) -> Optional[Button]:
    """在检出的按钮里找**最接近**目标坐标的那个。找不到返回 None。"""
    if not buttons or not target:
        return None
    tx, ty = float(target.get("x", 0.0)), float(target.get("y", 0.0))
    best: Optional[Button] = None
    best_d = 1e9
    for b in buttons:
        d = abs(b.x - tx) + abs(b.y - ty)
        if d < best_d:
            best_d = d
            best = b
    if best is None or best_d > tol:
        return None
    return best
