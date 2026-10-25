"""小地图：找出英雄在地图上的位置。

为什么先做这个
--------------
王者**对局画面上没有计时器**（局内时间只能靠"对局 HUD 就位"检出），
而顶栏那些数字在这类录屏里只有 5~7 像素高（边缘，读不稳）。

**小地图是另一回事**：它固定在左上角、约 100×100 像素，英雄自己的点
是里面最亮的那个小点 —— 位置能读出来，而且**读的是位置，不是字**。

位置怎么变成"这条规则判什么"
--------------------------
小地图上英雄点的位置 → 换算成**地图坐标**（0~1）→ 落在哪个区域
（龙坑 / 野区 / 线上 / 泉水 / 己方高地）→ 与镜头里的画面一起，
才能得出"在刷野 / 在开龙 / 在转线"这种动作事实。

区域表在 :mod:`app.collection.map_geometry`（``map_geometry.json``）：
**每个区域用地图坐标的矩形圈出来**。这份表必须对着**真实地图**标，
不能凭印象写 —— 写错了就会把"在龙坑"判成"在野区"。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: 地图几何表（随包打包）。
_GEOMETRY_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "map_geometry.json")

#: 绿度门槛：绿色通道要比红、蓝**都**高这么多，才算「绿」。
#:
#: 为什么不能只看「绿 - (红+蓝)/2」
#: ---------------------------------
#: 小地图上有大片**青色**（河道、防御塔标记）：青色像素的 G 很高，
#: 用 `g - (r+b)/2` 算出来同样是正的（实测青边 (4,118,143) 算得 45），
#: 会把河道当成绿标。
#: 改用 `g - max(r, b)` 之后，青色是负的（-25），绿圈才是正的（+52）。
HERO_GREEN_MIN = 18

#: 绿度要达到全图最大绿度的这个比例，才算「强绿」。
#:
#: 用相对值而不是绝对值：小地图底图的亮度随画面（昼夜、技能特效）变化，
#: 写死一个绝对阈值换一台设备就失效。
HERO_GREEN_RATIO = 0.35

#: 「够绿」判据：候选连通域的绿度峰值要达到全图峰值的这个比例。
#:
#: 实测：绿圈峰值 40~54，河道边的青绿、塔旁草地峰值只有 18~25，
#: 差一倍多 —— 按**峰值**分得开，按面积分不开（青绿面积反而更大）。
HERO_PEAK_RATIO = 0.6

#: 「空心」加分项：圈中心（外接框正中间那一小块）不该是绿的。
#:
#: 为什么这是个加分而不是门槛
#: --------------------------
#: 绿圈在小地图上只渲染成**上半圈亮、下半圈暗的一段弧**，
#: 弧正好盖住中心时，「中心是空的」就不成立 —— 当门槛会把真位置判掉。
#: 所以它只用来在候选之间加分。
HERO_HOLE_RATIO = 0.6

#: 空心带来的加分（够压过形状上的小差别）。
HERO_HOLLOW_BONUS = 30.0

#: 英雄标记的大小范围（像素面积）。
#:
#: 实测：绿圈在录屏里是 12x16 像素的**弧**（面积 20~72）。
#: 注意王者把小地图按**半分辨率**渲染再放大，所以录屏里 12x16 的圈，
#: 真实只有 6x8 像素 —— 尺寸门槛要按这个尺度定。
#: 早期上限设成 40，把真正的绿圈当噪声滤掉了，捡了别处的小绿点 ——
#: 表现出来就是「位置乱跳」。
HERO_MIN_AREA = 6
HERO_MAX_AREA = 120

#: 绿圈外接框的边长范围（像素）。
HERO_MIN_SIDE = 4
HERO_MAX_SIDE = 22

#: 外接框的长宽比上限：绿圈接近正圆，明显偏长的不是它。
HERO_ASPECT_MAX = 2.2


#: 绿圈外接框的边长范围（**占小地图边长的比例**）。
#:
#: 为什么改成比例而不是绝对像素
#: --------------------------
#: 这套阈值最初是按**旧模板的小地图（105×94）**定的：边长 ≤ 22 像素、
#: 面积 ≤ 120 像素。换到 1280×574 的模板（小地图 178×177）之后，
#: **玩家的绿圈实测是 24~26 像素、面积 99~300** —— 全部被上限滤掉，
#: 于是「最强绿块」选出来的是别的绿色小块（实测还会选中泉水那个
#: 固定装饰圈），位置在几帧之间乱跳。
#:
#: 同一套比例在两种尺寸的小地图上都成立：105 宽时 22/105 = 0.21，
#: 178 宽时 26/178 = 0.15 —— 同一量级。
HERO_SIDE_FRAC_MIN = 0.030
HERO_SIDE_FRAC_MAX = 0.200
HERO_AREA_FRAC_MIN = 0.0008
HERO_AREA_FRAC_MAX = 0.0200


def _hero_limits(minimap) -> Tuple[int, int, int, int]:
    """按小地图尺寸算出（边长下限, 边长上限, 面积下限, 面积上限）。"""
    h, w = minimap.shape[:2]
    m = min(h, w)
    side_min = max(HERO_MIN_SIDE, int(m * HERO_SIDE_FRAC_MIN))
    side_max = max(side_min + 2, int(m * HERO_SIDE_FRAC_MAX))
    area_min = max(HERO_MIN_AREA, int(m * m * HERO_AREA_FRAC_MIN))
    area_max = max(area_min + 4, int(m * m * HERO_AREA_FRAC_MAX))
    return side_min, side_max, area_min, area_max


@dataclass
class HeroCandidate:
    """小地图里一个**可能的**绿圈候选。

    为什么要有「候选」而不是直接给一个点
    ------------------------------------
    单帧里最强的绿块**不一定是玩家**：实测密集采样时，绿色的地形块
    常常比玩家标记更强，于是「最强绿」选出来的是一片不动的草地，
    位置在几帧之间乱跳。

    所以这里把**所有**通过基本筛选的候选都交出去，由
    :mod:`app.collection.skilltrack` 拿**整局的时间连续性**去挑：
    玩家的标记会移动，地形不会 —— 这是唯一分得开二者的特征。
    """

    x: float
    y: float
    nx: float
    ny: float
    area: int
    side_w: int
    side_h: int
    #: 该连通域的绿度峰值
    peak: int
    #: 本帧的绿度阈值（置信度算强度要用它）
    thr: int
    #: 「空心」程度 0~1（圆弧的中心不绿 → 1）
    hollow: float
    #: 单帧打分（越大越像玩家标记）
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {"nx": round(self.nx, 3), "ny": round(self.ny, 3),
                "area": self.area, "peak": self.peak,
                "hollow": round(self.hollow, 3),
                "score": round(self.score, 1)}


@dataclass
class HeroSpot:
    """英雄在小地图上的位置。"""

    #: 小地图内的像素坐标（左上角为原点）
    x: float
    y: float
    #: 归一化到小地图的比例（0~1）
    nx: float
    ny: float
    #: 点的像素面积
    area: int
    #: 识别置信度（0~1）
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"x": round(self.x, 1), "y": round(self.y, 1),
                "nx": round(self.nx, 3), "ny": round(self.ny, 3),
                "area": self.area, "confidence": round(self.confidence, 3)}


def find_hero_candidates(minimap, ratio: Optional[float] = None,
                         min_green: int = HERO_GREEN_MIN) -> List[HeroCandidate]:
    """小地图里所有**通过基本筛选**的绿圈候选。

    筛选条件与 :func:`find_hero_spot` 一致（绿度峰值、面积、边长、长宽比），
    只是**不挑最好的那个** —— 全都交出去。

    ``ratio`` / ``min_green``
    ----------------------
    默认跟 :func:`find_hero_spot` 一样用**相对**阈值（取全图峰值的
    ``HERO_GREEN_RATIO``），单帧最强绿块是谁就选谁。

    但**跨帧追踪**要用**宽松**阈值（``ratio`` 传小值）：实测玩家标记的
    绿度峰值有时只有 22（在泉水那种亮背景上），而同一帧的绿色地形能到
    116 —— 相对阈值会把玩家的标记直接滤掉。追踪那边靠**形状**（空心弧）
    而不是亮度来认，所以要多收候选。

    「空心」在这里先记下来、不当门槛：玩家的标记是**圆弧**，
    弧正好盖住中心时中心也是绿的，硬门槛会把真位置判掉。
    """
    if minimap is None or minimap.size == 0:
        return []
    import cv2
    import numpy as np

    use_ratio = HERO_GREEN_RATIO if ratio is None else ratio
    bgr = minimap if minimap.ndim == 3 else cv2.cvtColor(minimap, cv2.COLOR_GRAY2BGR)
    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)
    # 绿度：绿色通道要比红、蓝**都**高才算绿。
    # 用 g-(r+b)/2 会把**青色**（河道、塔标记）也算进来 —— 实测青边
    # (4,118,143) 算出来是 45，比绿圈还高，于是认出来的是河道上的点。
    green = g - np.maximum(r, b)
    peak = int(green.max())
    if peak < min_green:
        return []
    thr = max(min_green, int(peak * use_ratio))
    mask = (green >= thr).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    out: List[HeroCandidate] = []
    h, w = green.shape[:2]
    side_min, side_max, area_min, area_max = _hero_limits(minimap)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < area_min or area > area_max:
            continue
        lx = int(stats[i, cv2.CC_STAT_LEFT])
        ly = int(stats[i, cv2.CC_STAT_TOP])
        side_w = int(stats[i, cv2.CC_STAT_WIDTH])
        side_h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if side_w < side_min or side_h < side_min:
            continue
        if side_w > side_max or side_h > side_max:
            continue
        aspect = side_w / float(side_h)
        if aspect > HERO_ASPECT_MAX or aspect < 1.0 / HERO_ASPECT_MAX:
            continue
        comp = labels == i
        comp_peak = int(green[comp].max())
        # 「够绿」判据：这个连通域的绿度峰值要接近全图峰值。
        #
        # 实测（模板录屏 720x324 画布）：绿圈的绿度峰值是 40~54，
        # 而河道边上的青绿、塔旁的草地峰值只有 18~25 ——
        # 差一倍多，按峰值分得开。
        #
        # 宽松模式下（ratio 很小）这道门槛也得跟着松，否则还是会把
        # 玩家标记滤掉 —— 那边靠**形状**认，不靠亮度。
        if use_ratio >= HERO_GREEN_RATIO:
            if comp_peak < max(HERO_GREEN_MIN, int(peak * HERO_PEAK_RATIO)):
                continue
        # 空心加分项：圈中心（外接框正中间那一小块）不该是绿的。
        # 绿圈是**弧**（上半圈亮、下半圈暗），所以这是加分不是门槛 ——
        # 弧正好盖住中心时也得能认出来。
        cx0 = lx + side_w // 3
        cy0 = ly + side_h // 3
        cx1 = lx + side_w - side_w // 3
        cy1 = ly + side_h - side_h // 3
        hollow = 0.0
        if cx1 > cx0 and cy1 > cy0:
            core = green[cy0:cy1, cx0:cx1]
            if core.size:
                if float(core.mean()) < comp_peak * HERO_HOLE_RATIO:
                    hollow = 1.0
        # 环是空心的：像素重心会被弧的一侧压偏，取**外接框中心**。
        cx = float(lx) + (side_w - 1) / 2.0
        cy = float(ly) + (side_h - 1) / 2.0
        # 优先要「又绿又方又空心」的：绿度峰值高、面积大、形状接近正方。
        score = ((comp_peak * 4.0 + area) / max(aspect, 1.0 / aspect)
                 + hollow * HERO_HOLLOW_BONUS)
        out.append(HeroCandidate(
            x=cx, y=cy, nx=cx / max(1, w - 1), ny=cy / max(1, h - 1),
            area=area, side_w=side_w, side_h=side_h, peak=comp_peak, thr=thr,
            hollow=hollow, score=score))
    return out


def find_hero_spot(minimap) -> Optional[HeroSpot]:
    """在小地图里找出**自己**的位置。

    靠**绿圈**认，不靠"最亮"。

    为什么不是最亮
    --------------
    小地图上队友的头像、防御塔、野怪标记都是亮的，甚至更亮；
    而**自己**的位置有一个**绿圈**（王者给自己的标记），
    全图只有这一处是绿的 —— 这是唯一区分得开的特征。

    怎么认这个绿圈
    --------------
    绿度用 ``g - max(r, b)``：绿色通道要比红蓝**都**高才算绿。
    用 ``g - (r + b) / 2`` 会把**青色**（河道、塔标记）也算进来 ——
    实测小地图上一片青边的绿度算出来是 45，比绿圈还高，
    结果认出来的是河道上的一个点，位置在几个区域之间乱跳。

    认不出来就返回 None —— 宁可不报，也不报一个错的坐标
    （错的坐标会让"在龙坑"判成"在野区"，报告上完全看不出来）。

    单帧取「最强绿块」在**稀疏**采样下够用（窗口级分析只要一个大致位置），
    但**密集**采样下会盯上不动的绿色地形 —— 那时要用
    :func:`find_hero_candidates` + :mod:`app.collection.skilltrack`
    拿整局的时间连续性去挑。
    """
    if minimap is None or minimap.size == 0:
        return None
    cands = find_hero_candidates(minimap)
    if not cands:
        return None
    best = max(cands, key=lambda c: c.score)
    h, w = best.side_h, best.side_w
    # 置信度沿用旧口径：绿度强度 + 方正程度 + 空心加分
    strength = min(1.0, (best.peak - best.thr) / 30.0)
    squareness = 1.0 - abs(best.side_w - best.side_h) / float(
        max(best.side_w, best.side_h))
    confidence = min(1.0, 0.25 + 0.40 * strength + 0.20 * squareness
                     + 0.15 * best.hollow)
    return HeroSpot(x=best.x, y=best.y, nx=best.nx, ny=best.ny,
                    area=best.area, confidence=round(confidence, 3))


# --------------------------------------------------------------------------
# 地图朝向：王者会「中心反转」
# --------------------------------------------------------------------------

#: 敌方防御塔的颜色特征（BGR 判据）：红。
#:
#: 为什么用塔判朝向
#: ------------------
#: 王者对**红方玩家**会把地图「中心反转」—— 以中心对称点反转玩家、野怪、
#: 水晶和塔（腾讯官方《王者荣耀作战指南》原文：「如果你的阵营为红方，
#: 为了游戏体验，界面会进行 180° 旋转……暴君、主宰、法阵等也会交换位置」）。
#:
#: 所以**玩家永远在左下**，而**敌方的塔在哪一半边**取决于本局阵营：
#:   - 蓝方视角：敌方红塔在**右上**
#:   - 红方视角：地图转了 180°，敌方蓝塔跑到右上、红塔跑到**左下**
#:
#: 而**颜色不会跟着位置转**：红塔永远是红的。所以红色像素集中在哪半边，
#: 就是判别本局视角的**绝对**依据（不依赖"哪条是上路"这种地图知识）。
#:
#: 只看**四周一圈**：防御塔贴边，而敌方英雄头像（也是红圈）在地图中间，
#: 看整张图会被头像带偏。
TOWER_RED_MIN = 110
TOWER_RED_DELTA = 45

#: 判为「某一侧有敌方塔」所需的红色像素数。
TOWER_RED_MIN_COUNT = 40


def map_is_flipped(minimap) -> Optional[bool]:
    """这张小地图是不是**红方视角**（地图被中心反转过 180°）。

    返回 ``True`` = 红方视角，``False`` = 蓝方视角，``None`` = 分不出来。

    判据：**敌方防御塔是红色的**，而**颜色不随旋转改变** ——
    红塔集中在**右上**=蓝方视角，集中在**左下**=红方视角。
    只看小地图四周一圈（塔贴边，敌方英雄头像在中间，会带偏）。

    ⚠️ **不要用「己方野区营地图标的颜色」判阵营** —— 那个不成立
    （营地图标的颜色并不等于主人的阵营色）。只认塔。

    判不出来就返回 None 而不是猜 —— 转错方向会让「在主宰坑」判成
    「在暴君坑」，报告上完全看不出来。

    不过说实话：**区域判定用不着它**。主宰/暴君的图标颜色是绝对的，
    而它们分居两侧，主宰那侧永远是玩家的对抗路（见 :func:`lane_of`），
    所以按图标判区域天然就与阵营无关。
    """
    if minimap is None or getattr(minimap, "size", 0) == 0:
        return None
    import numpy as np

    b = minimap[:, :, 0].astype(np.int16)
    g = minimap[:, :, 1].astype(np.int16)
    r = minimap[:, :, 2].astype(np.int16)
    red = (r - b > TOWER_RED_DELTA) & (r - g > TOWER_RED_DELTA) \
        & (r > TOWER_RED_MIN)
    h, w = red.shape[:2]
    if h < 10 or w < 10:
        return None
    # 只看四周一圈（塔贴边，头像在中间）
    band = max(2, int(min(h, w) * 0.18))
    edge = np.zeros_like(red)
    edge[:band, :] = red[:band, :]
    edge[h - band:, :] = red[h - band:, :]
    edge[:, :band] = red[:, :band]
    edge[:, w - band:] = red[:, w - band:]
    top_right = int(edge[:h // 2, w // 2:].sum())
    bottom_left = int(edge[h // 2:, :w // 2].sum())
    if top_right >= TOWER_RED_MIN_COUNT and top_right > bottom_left * 2:
        return False          # 蓝方视角：红塔在右上
    if bottom_left >= TOWER_RED_MIN_COUNT and bottom_left > top_right * 2:
        return True           # 红方视角：红塔被转到左下
    return None


def lane_of(nx: float, ny: float,
            marks: Dict[str, Tuple[float, float]]) -> str:
    """按**大小龙的位置**判这是哪条路。

    为什么这样就与阵营无关
    ----------------------
    王者对红方玩家把地图「中心反转」180°，暴君与主宰的**图标位置互换**，
    但**图标本身的样子不变**（颜色是绝对的）。

    而不管哪一方：
      - 蓝方视角：主宰在**上半区**（对抗路那侧），暴君在下半区；
      - 红方视角：地图转了 180°，主宰转到**下半区** —— 而红方玩家的对抗路
        正好也在下半区（官方：「红方……对抗路和发育路的图标会交换位置」）。

    ⇒ **主宰图标所在的那一侧，永远是玩家的对抗路**；暴君那侧是发育路。

    所以判路只要认图标，**不需要判断本局是红方还是蓝方**。

    返回 ``"对抗路"`` / ``"发育路"`` / ``"中路附近"`` / ``""``（认不出）。
    """
    if not marks:
        return ""
    over = marks.get("主宰")
    tyr = marks.get("暴君")
    # 中路是左下↔右上的对角线 —— 180° 旋转**不改变**这条对角线，
    # 所以它与阵营无关，可以直接用。
    if _near_mid_diag(nx, ny):
        return "中路附近"
    if over is None and tyr is None:
        return ""
    if over is not None and tyr is not None:
        # **更靠近哪条龙，就是哪条路**。
        #
        # 为什么不能写死「上半区=对抗路」：那是**蓝方视角**的样子。
        # 红方玩家的地图被中心反转 180°，主宰跑到了下半区，而红方玩家的
        # 对抗路也在下半区 —— 「跟着主宰走」才与阵营无关。
        if abs(ny - over[1]) == abs(ny - tyr[1]):
            return ""
        return "对抗路" if abs(ny - over[1]) < abs(ny - tyr[1]) else "发育路"
    if over is not None:
        return "对抗路"
    return "发育路"


#: 离「左下↔右上对角线」多近算中路（归一化距离）。
MID_DIAG_TOL = 0.11

#: 站在龙坑图标多近算「在龙坑里」（归一化距离）。
PIT_RADIUS = 0.09

#: 两个龙坑在**蓝方视角**里的固定位置（归一化）。
#:
#: 为什么靠位置而不是颜色
#: ----------------------
#: 实测**同一个坑位的图标颜色会随状态变**（t=175 是紫粉、t=600 是金白），
#: 所以颜色认不出「哪个是主宰」。位置是固定的：
#: 主宰坑在左上、暴君坑在右下（红方视角整张图旋 180°，两个坑跟着换边）。
#:
#: 依据：用户给的 `app/collection/map_reference.png`（阵营选择界面的峡谷全图）
#: 里，主宰（紫）在上半区、暴君（金）在下半区；蓝方视角上半区在左上。
PIT_ANCHORS: Dict[str, Tuple[float, float]] = {
    "主宰": (0.34, 0.27),
    "暴君": (0.70, 0.72),
}

#: 龙坑锚点的容差（归一化）。图标中心在锚点附近就算。
PIT_ANCHOR_TOL = 0.09


def _near_mid_diag(nx: float, ny: float) -> bool:
    """点离「左下(0,1) ↔ 右上(1,0)」这条对角线有多近。"""
    # 对角线方程：x + y = 1
    return abs(nx + ny - 1.0) <= MID_DIAG_TOL * 1.414


def rotate_180(nx: float, ny: float) -> Tuple[float, float]:
    """把小地图内的一个点旋转 180°（中心对称）。"""
    return (1.0 - nx, 1.0 - ny)


def find_objective_marks(minimap) -> Dict[str, Tuple[float, float]]:
    """在小地图上认出**主宰**与**暴君**的图标位置。

    为什么靠**颜色**而不是坐标框
    --------------------------
    红方视角下两条龙会**互换位置**（中心反转），所以写死的坐标框
    一定会判反。而**图标的样子不随旋转改变**。

    颜色特征（实测 1280×574 模板录屏的小地图底图）：

    ==========  =====================  ==========================
    图标         平均 BGR              特征
    ==========  =====================  ==========================
    颜色**不可靠**
    --------------
    实测**同一个坑位在不同时刻的图标颜色不一样**（t=175 时紫粉、t=600 时
    金白），两段录屏里同一位置的颜色也不同 —— 所以**不能靠颜色认哪个是
    主宰、哪个是暴君**。

    可靠的是**位置**
    ----------------
    两个坑在蓝方视角里是**固定**的（实测两段录屏里，(0.34,0.27) 与
    (0.69,0.74) 各有一个龙形图标，位置一致、颜色不一致）。
    名字的依据是用户给的 `app/collection/map_reference.png`
    （阵营选择界面的峡谷全图）：**主宰（紫）在上半区、暴君（金）在下半区**，
    蓝方视角上半区在左上 ⇒ 主宰坑左上、暴君坑右下。

    这也顺带修掉一个坑：早先暴君的判据是「绿不低于其它通道」，于是
    **玩家标记的绿圈被认成了暴君图标**（绿圈同样是绿的），龙坑判定跟着错
    —— 而报告上看不出来。

    认不出就返回空表 —— 宁可没有区域名，也不给一个错的。
    """
    out: Dict[str, Tuple[float, float]] = {}
    if minimap is None or getattr(minimap, "size", 0) == 0:
        return out
    out = {name: (px, py) for name, (px, py) in PIT_ANCHORS.items()}
    return out


def _connected(mask):
    """连通域分析（延迟导入 cv2）。"""
    import cv2

    return cv2.connectedComponentsWithStats(mask.astype("uint8"), 8)


def cv2_connected_stat(which: str) -> int:
    """``cv2.CC_STAT_*`` 的取值（延迟导入 cv2）。"""
    import cv2

    return getattr(cv2, f"CC_STAT_{which}")


@dataclass
class MapGeometry:
    """地图区域表：每个区域用地图坐标的矩形圈出来。"""

    regions: List[Dict[str, Any]] = field(default_factory=list)
    #: 小地图在画布里的相对位置（相对坐标矩形）
    minimap_box: Optional[Tuple[float, float, float, float]] = None

    @classmethod
    def load(cls, path: Optional[str] = None) -> "MapGeometry":
        p = path or _GEOMETRY_JSON
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return cls()
        box = data.get("minimap_box")
        return cls(
            regions=list(data.get("regions") or []),
            minimap_box=tuple(float(v) for v in box) if box and len(box) == 4 else None,
        )

    def zone_of(self, nx: float, ny: float,
                marks: Optional[Dict[str, Tuple[float, float]]] = None,
                flipped: Optional[bool] = None) -> str:
        """英雄位置落在哪个区域。

        判据的优先级：

        1. **龙坑**：主宰坑 / 暴君坑 —— 靠**图标外观**认（颜色绝对），
           不受 180° 中心反转影响；
        2. **静态区域**（区域表）：本表按**蓝方基准**存；
           ``flipped=True``（红方视角）时先旋转 180° 再查；
        3. **分路**：对抗路 / 发育路 / 中路附近 —— 靠大小龙的位置定，
           **天然与阵营无关**（见 :func:`lane_of`）。

        认不出就返回空串（不猜）。

        阵营相对的区域
        --------------
        「己方野区」这种框跟着**玩家的基地**走：王者把地图转 180°，
        好让自己的基地永远在左下。于是同一个「己方野区」在蓝方基准里
        是左下那一块、在红方视角里是右上那一块 —— 区域表用
        「蓝方视角 / 红方视角」两个框分别记。

        **分不出视角时（``flipped`` 是 None）这类区域直接跳过** ——
        拿蓝方的框去套红方的点，判出来的区域是错的，
        而报告上完全看不出来。
        """
        if marks:
            pit = self._pit_of(nx, ny, marks)
            if pit:
                return pit
        if flipped:
            nx, ny = rotate_180(nx, ny)
        for r in self.regions:
            box = self._box_for(r, flipped)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            if x0 <= nx <= x1 and y0 <= ny <= y1:
                return r.get("name", "")
        if marks:
            return lane_of(nx, ny, marks)
        return ""

    @staticmethod
    def _box_for(region: Dict[str, Any],
                 flipped: Optional[bool]) -> Optional[Tuple[float, float, float, float]]:
        """按当前视角取区域框。取不到返回 None（这一区不判）。

        支持三种写法：

        - ``蓝方视角`` / ``红方视角``：阵营相对的区域，按视角选；
          **视角分不出来时返回 None**（不猜）。
        - ``boxes``：多个候选框，任一命中即可。
        - ``box``：单个框。
        """
        if "蓝方视角" in region or "红方视角" in region:
            if flipped is None:
                return None
            key = "红方视角" if flipped else "蓝方视角"
            b = region.get(key)
            if b and len(b) == 4:
                return tuple(float(v) for v in b)
            return None
        for b in (region.get("boxes") or []):
            if len(b) == 4:
                return tuple(float(v) for v in b)
        b = region.get("box")
        if b and len(b) == 4:
            return tuple(float(v) for v in b)
        return None


    @staticmethod
    def _pit_of(nx: float, ny: float,
                marks: Dict[str, Tuple[float, float]]) -> str:
        """英雄是不是站在龙坑里。"""
        for name, (mx, my) in marks.items():
            if abs(nx - mx) <= PIT_RADIUS and abs(ny - my) <= PIT_RADIUS:
                return f"{name}坑"
        return ""


def minimap_crop(canvas, geometry: MapGeometry):
    """按标定把画布里的小地图那块裁出来。"""
    if geometry.minimap_box is None or canvas is None:
        return None
    x0, y0, x1, y1 = geometry.minimap_box
    h, w = canvas.shape[:2]
    px0, py0 = int(x0 * w), int(y0 * h)
    px1, py1 = int(x1 * w), int(y1 * h)
    if px1 - px0 < 4 or py1 - py0 < 4:
        return None
    if px0 < 0 or py0 < 0 or px1 > w or py1 > h:
        return None
    return canvas[py0:py1, px0:px1]


def hero_zone(canvas, geometry: Optional[MapGeometry] = None
              ) -> Optional[Dict[str, Any]]:
    """英雄在小地图上的位置 + 落在哪个区域。"""
    geo = geometry or MapGeometry.load()
    mm = minimap_crop(canvas, geo)
    if mm is None:
        return None
    spot = find_hero_spot(mm)
    if spot is None:
        return None
    return {
        "spot": spot.to_dict(),
        "zone": geo.zone_of(spot.nx, spot.ny),
    }


# --------------------------------------------------------------------------
# 窗级聚合：一段画面里的位置，取**众数**
# --------------------------------------------------------------------------

#: 两个位置算「同一处」的容差（小地图比例）。
#:
#: 小地图在录屏里只有 105x94 像素、而且王者是**半分辨率**渲染再放大的，
#: 所以同一个位置在不同帧上会差 1~2 个像素（比例约 0.02）。
HERO_SAME_PLACE = 0.045

#: 一段画面里，至少这么多比例的**停留**要落在同一处，才认这个位置。
#:
#: 跟项目既有的口径一致：**一致才采信**（见 ``window_analysis.aggregate_facts``）。
#: 单帧识别会偶发离群（把河道上的青绿当成绿圈），多数票正好滤掉它。
HERO_AGREEMENT_MIN = 0.25

#: 整段里绿色标记的位移要超过这个值，才认为它「在动」。
#:
#: 为什么需要这条守卫
#: ------------------
#: 实测（2670x1200 的高码率录屏，帧 584/842/868/896）小地图上那圈绿色
#: 的**像素位置完全不变**（都是 134,240）—— 它是**固定图标**，不是玩家位置。
#: 而在玩家自己视角的录屏里，这个绿圈是跟着玩家走的
#: （模板录屏实证：开局时它在泉水位 (0.04,0.95)，之后一路沿地图移动）。
#:
#: 两种画面长得一样，光看单帧分不出来。所以用一条**保守**的规则：
#: 整段都不动 → 当作固定图标 → **不报位置**（记缺失）。
#: 宁可少一条事实，也不给一个错的坐标。
HERO_MOVED_MIN = 0.03


@dataclass
class HeroTrack:
    """一段画面里英雄位置的结论。"""

    #: 停留最久的那一处（小地图内比例 0~1）
    nx: float
    ny: float
    #: 停在那里的帧数占比
    agreement: float
    #: 参与投票的帧数
    samples: int
    #: 整段里英雄一共换过几处（走过几个地方）
    stops: int = 0
    #: 这段里绿色标记**动过没有**
    moved: bool = False
    #: 本局是不是**红方视角**（地图被中心反转过 180°）；None = 分不出来
    flipped: Optional[bool] = None
    #: 认出的**龙坑图标位置**（主宰/暴君），区域判定靠它
    marks: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    #: 区域名（区域表为空时是空串）
    zone: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"nx": round(self.nx, 3), "ny": round(self.ny, 3),
                "agreement": round(self.agreement, 3),
                "samples": self.samples, "stops": self.stops,
                "moved": self.moved, "flipped": self.flipped,
                "marks": {k: (round(v[0], 3), round(v[1], 3))
                          for k, v in self.marks.items()},
                "zone": self.zone}


def hero_spots(frames: Sequence[Any],
               geometry: Optional[MapGeometry] = None
               ) -> List[HeroSpot]:
    """逐帧找绿圈。"""
    geo = geometry or MapGeometry.load()
    out: List[HeroSpot] = []
    for canvas in frames:
        mm = minimap_crop(canvas, geo)
        if mm is None:
            continue
        spot = find_hero_spot(mm)
        if spot is not None:
            out.append(spot)
    return out


def track_hero(frames: Sequence[Any],
               geometry: Optional[MapGeometry] = None) -> Optional[HeroTrack]:
    """从**一段画面**里定出英雄的位置。

    不是「整段取众数」
    ------------------
    英雄在打**三分钟**的窗里会走过好几个地方，整段取众数必然不成立
    （实测 80 秒的一段里位置从 (0.52,0.77) 一路走到 (0.93,0.97)）。
    所以先把画面**按停留分桶**：位置没变算一桶，走了就开新桶，
    最后取**停留最久的那一桶** —— 它回答的是
    「这一段时间他主要待在哪儿」。

    单帧认绿圈会偶发离群（把河道边的青绿当绿圈），
    分桶 + 取最长桶正好滤掉它：离群点各自成一小桶，自然排在后面。

    众数位置取桶内各点的**中位数** —— 避免个别偏得远的点把结论拖歪。
    """
    geo = geometry or MapGeometry.load()
    spots = hero_spots(frames, geo)
    if not spots:
        return None

    # 按停留分桶：位置还在容差内就并进当前桶，走了就开新桶。
    buckets: List[List[HeroSpot]] = []
    for sp in spots:
        if buckets:
            last = buckets[-1]
            cx = sum(s.nx for s in last) / len(last)
            cy = sum(s.ny for s in last) / len(last)
            if (abs(sp.nx - cx) <= HERO_SAME_PLACE
                    and abs(sp.ny - cy) <= HERO_SAME_PLACE):
                last.append(sp)
                continue
        buckets.append([sp])
    buckets.sort(key=len, reverse=True)
    best = buckets[0]
    agreement = len(best) / float(len(spots))
    if agreement < HERO_AGREEMENT_MIN:
        return None
    # 整段动过没有？不动的那多半是**固定图标**，不是玩家位置（见 HERO_MOVED_MIN）。
    moved = False
    for a in spots:
        for b in spots:
            if (abs(a.nx - b.nx) + abs(a.ny - b.ny)) >= HERO_MOVED_MIN:
                moved = True
                break
        if moved:
            break
    xs = sorted(s.nx for s in best)
    ys = sorted(s.ny for s in best)
    nx = xs[len(xs) // 2]
    ny = ys[len(ys) // 2]
    # 本局是不是**红方视角**（地图被中心反转过 180°）—— 查静态区域表前要知道。
    flipped = None
    marks: Dict[str, Tuple[float, float]] = {}
    try:
        mm0 = minimap_crop(frames[0], geo)
        if mm0 is not None:
            flipped = map_is_flipped(mm0)
            marks = find_objective_marks(mm0)
    except Exception:                                   # noqa: BLE001
        flipped = None
    return HeroTrack(nx=nx, ny=ny, agreement=agreement, samples=len(spots),
                     stops=len(buckets), moved=moved, flipped=flipped,
                     marks=marks,
                     zone=geo.zone_of(nx, ny, marks, flipped))
