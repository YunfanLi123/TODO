"""顶栏数字：切字形、认数字。

为什么单独一个模块
------------------
顶栏的字体和计时器**不是同一套**（位数、字形都不一样），
所以不能复用 ``app/collection/clock.py`` 的模板。

怎么分左右
----------
王者顶栏里左队（蓝方）的数字偏**蓝**、右队（红方）偏**红**，
中间那个「vs」是**灰**的 —— 灰的两通道差接近 0，按颜色一筛就分开了。
比按位置猜稳：数字位数变了（1 位变 2 位），位置就跟着挪。

怎么切字形
----------
按颜色掩膜取连通域，每个连通域是一个数字。字号只有 7~10 像素高，
所以字形取的是**以连通域中心为中心的固定窗口**（8x12），
不是紧贴的包围盒 —— 紧贴的话「1」会缩成一条线，跟别的数字长得一样。

读不出来就返回 None
-------------------
任何一位认不准，整串就不给。顶栏数字读错比读不到危险得多：
读错之后每一条带局势条件的模式都会判到错误的局面，而报告上只是「没触发」。

模板从哪来
----------
``app/collection/scoreboard_digits.json``，由
``tools/build_scoreboard_templates.py`` 从**真实录屏**里长出来 ——
抽字形 → 聚类 → 出拼图**由人肉眼标号** → 写模板。
程序自己认字形等于让它自己编数字，那步不能省。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: 模板文件（随包打包）。
_TEMPLATES_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "scoreboard_digits.json")

#: 归一后的**字形窗口**（宽 x 高）。
#:
#: 字形是**按自身长宽比**缩放后居中放进去的，所以「1」这种窄字
#: 仍然是窄的 —— 否则「1」被拉宽就跟「7」分不开了。
GLYPH_W, GLYPH_H = 16, 24

#: 数字高度占**区域高度**的比例（区域是标定出来的，按设备量）。
#:
#: 只用来**估算**数字高度（真实高度在切的时候按行投影量出来，这个只是
#: 给阈值自适应用的初值）。
DIGIT_IN_REGION = 0.9

#: 归一裁剪到的高度（越大越稳，最后字形各自缩放，不受它影响）。
#:
#: 24 是这么定的：这段录屏的区域高 19 画布像素、数字宽 11，
#: 归一后数字宽 14、高 23 —— 都还在下面的尺寸上限之内。
CROP_H = 24

#: 字位在归一后的高度范围。
GLYPH_MIN_H = 8
GLYPH_MAX_H = 30

#: 字位的宽度范围。
GLYPH_MIN_W = 2
GLYPH_MAX_W = 22

#: **一个**数字的最大宽度（归一后）。
#:
#: 用它区分「一个数字被切开」和「两个数字连在一起」：前者合起来仍然
#: 窄（实测 13），后者合起来宽（实测 24）。
GLYPH_ONE_MAX_W = 18

#: 「够亮」的下限：字形比背景亮得多。
#:
#: 实测：半透明的数字（叠在游戏画面上）亮度只有 60~85，
#: 早先用 85 会把它们整片丢掉 —— 表现是「某些时刻一个字都读不出」。
BRIGHT_MIN = 55

#: 「够蓝」/「够红」：通道差要超过这个数。
#:
#: 中间那个「vs」是**白的**（R≈G≈B，通道差接近 0），所以这个阈值
#: 同时把「vs」挡在外面。
COLOR_MIN = 8

#: 匹配得分下限（字形与模板的余弦相似度）。
MIN_SCORE = 0.55

#: 最优与次优的分差下限。
MIN_MARGIN = 0.02


def normalize_crop(crop):
    """把区域裁剪缩放到**固定高度** ``CROP_H``。

    只为让阈值自适应稳定一点 —— **字形最后各自按长宽比缩放**，
    所以这个高度不影响字形的可比性。
    """
    import cv2

    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    h = float(crop.shape[0])
    if h <= 0:
        return None
    scale = CROP_H / h
    nw = max(4, int(round(crop.shape[1] * scale)))
    nh = max(4, int(round(crop.shape[0] * scale)))
    return cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)


def extract_glyphs(crop, side: str = "") -> Tuple[List[Any], List[str]]:
    """从一块顶栏裁剪里切出字形。

    返回 ``(字形列表, 归属列表)``。

    切分用**亮度自适应阈值**（区域内最亮与最暗的中点）
    --------------------------------------------------
    试过的几种方案（已知时刻能切对字形个数的比例）：

    ==========================  ======
    阈值                        对上
    ==========================  ======
    固定 60                       4/10
    固定 85                       8/10
    区域内 90 分位                6/10
    区域内 80 分位                7/10
    **最亮与最暗的中点**          **10/10**
    ==========================  ======

    原因：数字是**半透明**的（叠在游戏画面上），绝对亮度随背景变 ——
    固定阈值要么切不到、要么把背景连成一片。取中点则自适应。

    内部先**归一**（见 :func:`normalize_crop`），所以切出来的字形跨设备可比。

    ``side`` 是这一块的归属（"left"/"right"）—— 现在**区域本身就是分开的**
    （左队一块、右队一块），颜色只用来剔除中间那个白的「vs」；
    传空串表示不按颜色分。
    """
    if crop is None or getattr(crop, "size", 0) == 0:
        return [], []
    import cv2
    import numpy as np

    crop = normalize_crop(crop)
    if crop is None:
        return [], []
    bgr = crop if crop.ndim == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)
    lum = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = lum.shape[:2]

    thr = (float(lum.min()) + float(lum.max())) / 2.0
    thr = max(thr, 45.0)
    mask = (lum >= thr).astype(np.uint8)
    # 「vs」是白的：两通道差≈0，用颜色把它挡在外面（区域边界外的部分）
    if side:
        if side == "left":
            mask &= ((b - r) >= COLOR_MIN).astype(np.uint8)
        else:
            mask &= ((r - b) >= COLOR_MIN).astype(np.uint8)

    out: List[Any] = []
    sides: List[str] = []
    # **按列投影分段**，不用连通域。
    #
    # 为什么不用连通域：两个数字的抗锯齿边缘会**连成一片**，
    # 于是 "20" 被当成一个字形 —— 读出来是某个乱七八糟的两位数
    # （实测读出过 46 / 56 / 33）。
    #
    # 列投影看的是「这一列有没有笔画」，两个数字之间那一两列是空的，
    # 天然分得开；而一个数字内部的笔画在**列上互相重叠**
    # （"1" 的斜杠和竖、"7" 的横和斜都共用列），所以不会被切开。
    col = mask.sum(axis=0)
    solid = col >= 1
    segments = []
    x = 0
    while x < w:
        if solid[x]:
            x0 = x
            while x < w and solid[x]:
                x += 1
            segments.append((x0, x))
        else:
            x += 1
    if not segments:
        return out, sides
    # 数字的**行范围**（同一行里的数字上下对齐）—— 用行投影量出来。
    rows = np.where(mask.any(axis=1))[0]
    if len(rows) == 0:
        return out, sides
    ry0 = int(rows.min())
    ry1 = int(rows.max()) + 1
    if not (GLYPH_MIN_H <= (ry1 - ry0) <= GLYPH_MAX_H):
        return out, sides

    # **合并相邻的窄段**：一个数字被列投影切开（笔画之间的低谷）
    # 合并后总宽仍然像**一个**数字，就当成一个。
    # 判据用**总宽**而不是间隙 —— 两位数的两段紧挨着时，间隙
    # 和「一个数字内部」的间隙差不多大，只有总宽能分开（实测
    # "20" 的两段合起来宽 24，"3" 被切开的两段合起来只有 13）。
    #
    # 合并时**跳过极窄的段**（1~2 像素，多半是噪点），
    # 否则一个噪点会把两个数字粘成一坨。
    real = [s for s in segments if s[1] - s[0] >= 2]
    merged_segs: List[Tuple[int, int]] = []
    for x0, x1 in real:
        if merged_segs and (x1 - merged_segs[-1][0]) <= GLYPH_ONE_MAX_W:
            merged_segs[-1] = (merged_segs[-1][0], x1)
        else:
            merged_segs.append((x0, x1))
    # **过宽的段切开**：两个数字连成了一段（抗锯齿边缘搭上了）。
    # 在段内找**最深的谷**（笔画最少的列）切开。
    final_segs: List[Tuple[int, int]] = []
    for x0, x1 in merged_segs:
        if x1 - x0 <= GLYPH_MAX_W:
            final_segs.append((x0, x1))
            continue
        inner = col[x0 + 1:x1 - 1]
        if inner.size == 0:
            continue
        cut = int(np.argmin(inner)) + x0 + 1
        final_segs.append((x0, cut))
        final_segs.append((cut, x1))

    for x0, x1 in final_segs:
        seg_w = x1 - x0
        if seg_w < GLYPH_MIN_W or seg_w > GLYPH_MAX_W:
            continue
        sub = mask[ry0:ry1, x0:x1]
        win = lum[ry0:ry1, x0:x1]
        ph, pw = win.shape[:2]
        if ph < 4 or pw < 2:
            continue
        keep = cv2.dilate(sub.astype(np.uint8), np.ones((3, 3), np.uint8))
        # **按自身长宽比**缩放：高固定，宽按比例（上限 GLYPH_W）
        scale = float(GLYPH_H) / float(ph)
        nw = max(2, min(GLYPH_W, int(round(pw * scale))))
        nh = GLYPH_H
        small = cv2.resize(win * keep, (nw, nh), interpolation=cv2.INTER_AREA)
        patch = np.zeros((GLYPH_H, GLYPH_W), np.float32)
        ox = max(0, (GLYPH_W - nw) // 2)
        patch[:, ox:ox + nw] = small
        if float(patch.max()) < BRIGHT_MIN:
            continue
        out.append(patch)
        sides.append(side)
    return out, sides


def unit(vec) -> Any:
    """零均值 + 单位长度：匹配时做内积，比对形状不看亮度。

    **先展平** —— 字形是 12x18 的二维块，直接做内积会变成矩阵乘法，
    结果是个矩阵而不是一个相似度分数。
    """
    import numpy as np

    v = np.asarray(vec, dtype="float32").reshape(-1)
    v = v - float(v.mean())
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.zeros_like(v)
    return v / n


_templates_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None
#: 缓存时记下的模板文件修改时间（文件变了自动重载）。
_templates_mtime: float = 0.0


def load_templates(path: Optional[str] = None
                   ) -> Dict[str, List[Dict[str, Any]]]:
    """加载数字模板。没文件或格式不对就返回空表 → 上层报「没读到」。

    刻意不抛异常：模板是**标定值**，换机型时可能还没重跑标定脚本。
    那时正确的表现是「读不出击杀数」，而不是整个扫描崩掉。
    """
    global _templates_cache, _templates_mtime
    try:
        mtime = os.path.getmtime(_TEMPLATES_JSON)
    except OSError:
        mtime = 0.0
    if (path is None and _templates_cache is not None
            and mtime == _templates_mtime):
        return _templates_cache
    if path is None:
        _templates_mtime = mtime
    p = path or _TEMPLATES_JSON
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    # **未验收的模板不启用**：2026-10-23 这套模板在 17 个「肉眼确认过
    # 读数」的时刻里只有 2 个读对 —— 字位切分在某些帧仍会出错。
    # 读错比读不到危险：顶栏数字读错之后，每一条带局势条件的模式都会
    # 判到错误的局面，而报告上只是「没触发」。宁可记成缺失。
    if not data.get("已验收", False):
        return {}
    import numpy as np

    out: Dict[str, List[Dict[str, Any]]] = {"all": []}
    for d in data.get("digits") or []:
        tmpl = d.get("template")
        if not tmpl or len(tmpl) != GLYPH_W * GLYPH_H:
            continue
        out["all"].append({"label": int(d.get("label", 0)),
                           "vec": unit(np.asarray(tmpl, dtype="float32"))})
    if not out["all"]:
        return {}
    if path is None:
        _templates_cache = out
    return out


def read_number(crop, side: str,
                templates: Optional[Dict[str, List[Dict[str, Any]]]] = None
                ) -> Optional[int]:
    """读一侧的击杀数。认不准任何一位就返回 None。"""
    import numpy as np

    tmpls = (templates if templates is not None else load_templates())
    # **两侧共用一套模板** —— 顶栏左右是同一套字体，只是颜色不同。
    # 分侧各存一套的话，一边出现得少的数字（比如 "4"）就没模板了。
    pool = tmpls.get("all") or tmpls.get(side) or []
    if not pool:
        return None
    # 区域本身就是这一侧的（左右各一块），所以按区域定侧
    glyphs, _ = extract_glyphs(crop, side)
    if not glyphs:
        return None
    # 按横坐标排序 = 从左往右读
    items = []
    for g in glyphs:
        ys, xs = np.where(g >= BRIGHT_MIN)
        items.append((float(xs.mean()) if len(xs) else 0.0, g))
    items.sort(key=lambda t: t[0])

    digits: List[int] = []
    for _, g in items:
        v = unit(g)
        best, second = None, None
        for t in pool:
            s = float(t["vec"] @ v)
            if best is None or s > best[0]:
                second = best
                best = (s, t["label"])
            elif second is None or s > second[0]:
                second = (s, t["label"])
        if best is None or best[0] < MIN_SCORE:
            return None
        if second is not None and best[0] - second[0] < MIN_MARGIN:
            return None
        digits.append(int(best[1]))
    if not digits:
        return None
    value = 0
    for d in digits:
        value = value * 10 + d
    return value


def read_kill_score(left_crop, right_crop,
                    templates: Optional[Dict[str, List[Dict[str, Any]]]] = None
                    ) -> Optional[Tuple[int, int]]:
    """读顶栏的击杀数，返回 ``(左队, 右队)``。

    两块裁剪**分开传**（左队一块、右队一块）—— 王者顶栏是「N vs M」，
    数字颜色不同，而且中间那个灰白的「vs」不能混进来。

    读不出返回 None —— **不给猜的值**。
    """
    left = read_number(left_crop, "left", templates)
    right = read_number(right_crop, "right", templates)
    if left is None or right is None:
        return None
    return left, right
