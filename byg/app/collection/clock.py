"""游戏内时钟：直接读出 MM:SS。

为什么改成「认字」而不是「数跳」
--------------------------------
早先的方案是数跳变：时钟每秒跳一次，数跳了几次就是过了几秒。
想法本身不坏，但拿真实录屏一试就站不住 —— 时钟数字是压在
**会动的游戏画面**上的（草地中灰、河道亮蓝、后期地形亮红橙）：

    整块灰度均值当指纹   背景一动就误判成跳变，间隔只有 0.03 秒
    二值化 + 亮度阈值    有的帧整块找不到一个亮像素，分段每帧都不一样
    先缩放后二值化      笔画被 INTER_AREA 平均掉，指纹几乎不随秒跳变

三种做法都数不出稳定的 1 秒 1 跳。数字本身却又亮又稳：
白字、固定位置、固定字体。所以改成**归一化互相关**匹配 ——
它对局部亮度和对比度都不敏感，背景是草地还是雪地都不影响判定。

换来的不只是「能读」，还有两样原来没有的东西：

  1. **绝对时间**。数跳变只能得到相对秒数，还得另外猜对局从哪一秒开始；
     读出 MM:SS 之后，对局起点是拟合出来的，不是猜的。
  2. **强得多的自检**。读出的 (视频秒, 游戏秒) 是成对的，可以直接做线性拟合：
     斜率必须≈1（游戏时间按真实速度走），残差必须小。斜率明显偏离 1
     只有两种可能 —— 帧率元数据不对，或者把别的东西当成时钟了。

读不出就返回「没读到」（:attr:`ClockTrack.ok` 为 False），绝不猜。
猜错时间的代价是每一条带时间窗的模式都判到错误的阶段，
而报告上完全看不出错在哪 —— 那些模式看起来只是「没触发」。

模板来源
--------
``app/collection/clock_digits.json``，由 ``tools/build_clock_templates.py``
从真实录屏聚类学出。**那个文件是标定值**：槽位位置随分辨率与屏幕比例变化，
换机型必须重跑脚本。本模块只负责加载与匹配。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: 模板文件。放在包内，随 exe 一起打包。
_DIGITS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "clock_digits.json")

#: 最佳与次佳的得分差要大于这个值才认。
#:
#: 实测标定录屏：20 个数字位全部由 argmax 选对，但有几帧正确项只领先
#: 0.04 —— 那个时刻画面正压着别的亮东西（例如技能特效）。早先把闸门
#: 定在 0.08，结果把**读对了**的时刻判成「没读到」。
#: 真正兜底的不是这道闸，而是 :func:`build_clock_track` 的拟合与离群剔除：
#: 认错数字会让读数偏几十秒，离直线足够远，一定会被剔掉。
MIN_MARGIN = 0.02

#: 最佳模板本身的得分下限。
#:
#: 必须有这道闸：模板是**按位**建的，而某一位未必覆盖全部十个数字
#: （一局 12 分钟，分钟十位只出现过 0 和 1）。这一位真显示出模板里
#: 没有的数字时，argmax 会硬挑一个最像的 —— 不设下限就会把「2」读成「1」。
#: 绝对分不够就报「没读到」，这是刻意的：读错时间比读不到时间危险得多，
#: 因为读错之后每一条带时间窗的模式都会判到错误的阶段，而报告上只是「没触发」。
MIN_SCORE = 0.35

#: 读出来的分钟数上限。王者荣耀单局不会超过 40 分钟；
#: 超过就说明匹配到了别的东西（比如金币数字）。
MAX_MINUTES = 40

#: 拟合自检：斜率允许的偏差。斜率 = 游戏秒 / 视频秒，正常正好 1。
SLOPE_TOLERANCE = 0.06

#: 拟合自检：残差（秒）超过这个值就算不可信。
RESIDUAL_TOLERANCE = 3.0

#: 至少要有这么多个可读样本才敢下结论。
MIN_SAMPLES = 8

#: 计时器数字在画布上的最小像素高度。
#:
#: 低于这个高度，模板匹配认出来的是**噪声**而不是数字 ——
#: 实测聊天软件转发的王者荣耀录屏，画面被压到 217 像素宽，计时器数字只剩
#: 3~4 像素高，认出来的是 1111 / 1144 这种乱码（而且每次还不一样）。
#:
#: 认错时间比读不到时间危险得多：读错之后每一条带时间窗的模式都会判到
#: 错误的阶段，而报告上只是「没触发」，看不出错在哪。所以宁可明确说
#: 「这段录屏的画质读不出局内时间」，也不给一个编出来的时间。
MIN_DIGIT_PX = 7.0


def digit_height_px(canvas_h: int) -> float:
    """计时器数字在画布上的**实际**像素高度。

    数字的大小随画布高度缩放（标定时是 1200 高、25 像素），
    所以换算就是按比例缩一下。
    """
    if canvas_h <= 0:
        return 0.0
    return SLOT_H * (float(canvas_h) / float(WORK_H))


def too_small_to_read(canvas_h: int) -> bool:
    """画布矮到数字只剩几个像素 —— 这时模板匹配认出来的是噪声。"""
    return digit_height_px(canvas_h) < MIN_DIGIT_PX


def legibility_note(canvas_h: int) -> str:
    """给「为什么读不到」一句人话。"""
    h = digit_height_px(canvas_h)
    if too_small_to_read(canvas_h):
        return (
            f"画布只有 {canvas_h} 像素高，计时器数字实际只有 {h:.1f} 像素"
            f"（至少要 {MIN_DIGIT_PX:.0f} 像素才认得出）—— "
            f"这段录屏的画质读不出局内时间"
        )
    return f"计时器数字在画布上约 {h:.1f} 像素高"


#: 自动识别（读局内时间 + 读局势数字 + 定位小地图）需要的**最小画布**。
#:
#: 三个要求取最严的：
#:
#: - 计时器数字 ≥ 7 像素高 → 画布高 ≥ 336；
#: - 顶栏的金币/击杀数字同样是 7 像素量级 → 画布高 ≥ 336；
#: - 小地图要能定出英雄的点 → 小地图得够宽（画布宽 × 0.22 ≥ 120）→ 画布宽 ≥ 545。
#:
#: 实测对照：聊天软件转发的录屏画布 480×218（远不够）；
#: 同样是「容器 960×464」的那段，画布 464×211 —— **容器分辨率不算数，
#: 画布分辨率才算数**（容器里可能一大半是黑边）。
RECOGNITION_MIN_W = 545
RECOGNITION_MIN_H = 336


def recognition_feasibility(canvas_w: int, canvas_h: int) -> Dict[str, Any]:
    """这份录屏**够不够自动识别**。

    返回 ``{"ok": bool, "note": str, "digit_px": float}``。
    说清楚差在哪：画布多高、离门槛还差多少 —— 玩家要重录的时候
    知道该录多清楚，而不是只听到一句「识别不了」。
    """
    digit = digit_height_px(canvas_h)
    ok_w = canvas_w >= RECOGNITION_MIN_W
    ok_h = canvas_h >= RECOGNITION_MIN_H
    if ok_w and ok_h:
        return {
            "ok": True,
            "note": (f"画布 {canvas_w}×{canvas_h}，计时器数字约 {digit:.1f} 像素高"
                     f" —— 够自动识别"),
            "digit_px": round(digit, 1),
        }
    need = []
    if not ok_h:
        need.append(f"画布高 ≥ {RECOGNITION_MIN_H}（现在 {canvas_h}）")
    if not ok_w:
        need.append(f"画布宽 ≥ {RECOGNITION_MIN_W}（现在 {canvas_w}）")
    return {
        "ok": False,
        "note": (f"画布只有 {canvas_w}×{canvas_h}（计时器数字约 {digit:.1f} 像素高）"
                 f" —— 自动识别需要 {'、'.join(need)}"),
        "digit_px": round(digit, 1),
    }


@dataclass
class ClockSample:
    """一次采样留下的痕迹。"""

    t: float                       # 视频内秒数
    game_s: Optional[int] = None   # 读到的游戏内秒数（MM*60+SS）
    confidence: float = 0.0        # 最佳与次佳模板的得分差


@dataclass
class ClockTrack:
    """从一串采样里还原出的游戏内时间线。"""

    ok: bool = False
    reason: str = ""
    #: 视频秒 -> 游戏内分钟。只含读成功的采样点。
    game_minutes_at: Dict[float, float] = field(default_factory=dict)
    #: 对局第 0 秒对应的视频时刻；由线性拟合得出，不是猜的
    game_start_t: Optional[float] = None
    #: 拟合结果，用于自检与展示
    slope: Optional[float] = None
    residual: Optional[float] = None
    samples_read: int = 0
    samples_total: int = 0

    def minutes_at(self, video_t: float) -> Optional[float]:
        """某个视频时刻对应的游戏内分钟数。

        采样点之间线性插值 —— 相邻两次采样最多隔几秒，
        时钟在这中间最多差几秒，插值误差远小于一个阶段的宽度。
        """
        if not self.ok or self.game_start_t is None:
            return None
        return max(0.0, (video_t - self.game_start_t) / 60.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "game_start_t": round(self.game_start_t, 1) if self.game_start_t is not None else None,
            "slope": round(self.slope, 4) if self.slope is not None else None,
            "residual": round(self.residual, 2) if self.residual is not None else None,
            "samples_read": self.samples_read,
            "samples_total": self.samples_total,
            # 可靠读数所在的视频时刻（升序）。采集层靠最后一个读数确定
            # 「对局到哪一秒为止」—— 录屏往往在对局结束后还继续着，
            # 那段不是对局，不该被抽帧抽进去。
            "readable_times": sorted(self.game_minutes_at.keys()),
        }


# --------------------------------------------------------------------------
# 模板加载
# --------------------------------------------------------------------------

_digits_cache: Optional[List[Dict[str, Any]]] = None
#: 缓存时记下的模板文件修改时间。
#:
#: 为什么记它
#: --------
#: 模板是**标定值**，标定完会换文件；而服务往往已经在跑了，
#: 进程里缓存的旧模板不会自动更新 —— 于是扫描时读到的是**上一次**标定的
#: 模板，读出来的是垃圾（实测：同一段视频，独立跑斜率 1.000，
#: 服务里跑出来 0.079，差了一个数量级），而界面只显示「读不出时钟」，
#: 看不出是模板过期了。
#:
#: 记下 mtime，文件一变就重载 —— 标定完直接重扫，不用记得重启服务。
_digits_mtime: float = 0.0


def load_digits() -> List[Dict[str, Any]]:
    """加载数字模板。缺文件或格式不对就返回空列表，交给上层报「没读到」。

    刻意不在这里抛异常：模板是**标定值**，换机型时可能还没重跑脚本。
    那时正确的表现是「读不出时钟，阶段要手填」，而不是整个扫描崩掉。

    文件变了会自动重载（见 :data:`_digits_mtime`）。
    """
    global _digits_cache, _digits_mtime
    try:
        mtime = os.path.getmtime(_DIGITS_JSON)
    except OSError:
        mtime = 0.0
    if _digits_cache is not None and mtime == _digits_mtime:
        return _digits_cache
    _digits_mtime = mtime

    try:
        with open(_DIGITS_JSON, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []

    pw, ph = data.get("size", (0, 0))

    slots: List[Dict[str, Any]] = []
    for s in data.get("slots", []):
        labels = s.get("labels")
        tmpl = s.get("templates") or []
        if not labels or len(labels) != len(tmpl):
            # 没标号或数量对不上就不认：认出来的数字无从解释
            return []
        arr = np_array(tmpl, ph, pw)
        if arr is None:
            return []
        slots.append({
            "name": s.get("name", ""),
            "box": tuple(s["box"]),
            "shape": (ph, pw),
            "templates": _l2_normalize(arr),   # (K, ph*pw)，每行已归一化
            "labels": list(labels),
        })
    if not slots:
        return []
    _digits_cache = slots
    return slots


def np_array(rows: Sequence[Sequence[float]], h: int, w: int):
    """把 JSON 里的模板转成 ``(K, h*w)`` 的 numpy 数组；形状不对返回 None。

    模板在文件里是**展平**的（每个数字一行 500 个数）—— 行列式反而更小，
    读取端不关心长什么样，只做内积。这里同时接受已经排好的二维形式，
    免得手工改文件时把两种布局搞混。
    """
    import numpy as np

    out = np.asarray(rows, dtype=np.float32)
    if out.ndim == 2 and out.shape[1] == h * w:
        return out
    if out.ndim == 3 and out.shape[1:] == (h, w):
        return out.reshape(out.shape[0], h * w)
    return None


def _l2_normalize(arr):
    import numpy as np

    n = np.linalg.norm(arr, axis=1, keepdims=True)
    n[n < 1e-6] = 1.0
    return arr / n


# --------------------------------------------------------------------------
# 读时钟
# --------------------------------------------------------------------------

def match_digits(crop, slots: Optional[List[Dict[str, Any]]] = None,
                 min_score: float = MIN_SCORE) -> Optional[int]:
    """在**一块裁剪好的区域**里按槽位读数字，认出就返回数值。

    槽位坐标是**相对这块裁剪**的（占裁剪宽高的比例）——
    所以同一套槽位既能用于整幅画面（计时器），也能用于顶栏的某一小块
    （金币、击杀）。

    任何一位认不准就返回 None：宁可说「没读到」，不给乱码。
    """
    import numpy as np

    slots = load_digits() if slots is None else slots
    if not slots or crop is None:
        return None
    h, w = crop.shape[:2]
    digits: List[int] = []
    for slot in slots:
        x, y, bw, bh = slot["box"]
        px, py = int(round(x * w)), int(round(y * h))
        pw, ph = int(round(bw * w)), int(round(bh * h))
        if px < 0 or py < 0 or px + pw > w or py + ph > h or pw < 1 or ph < 1:
            return None
        patch = crop[py:py + ph, px:px + pw]
        sh, sw = slot["shape"]
        if patch.shape[0] != sh or patch.shape[1] != sw:
            import cv2

            patch = cv2.resize(patch, (sw, sh), interpolation=cv2.INTER_AREA)
        v = patch.astype(np.float32).flatten()
        v = v - v.mean()
        n = float(np.linalg.norm(v))
        if n < 1e-3:
            return None
        scores = slot["templates"] @ (v / n)
        best = int(np.argmax(scores))
        top = float(scores[best])
        if top < min_score:
            return None
        if scores.size > 1:
            margin = top - float(np.partition(scores, -2)[-2])
            if margin < MIN_MARGIN:
                return None
        digits.append(int(slot["labels"][best]))
    if not digits:
        return None
    value = 0
    for d in digits:
        value = value * 10 + d
    return value


def read_clock_frame(frame):
    """读一帧，返回 ``(游戏内秒数, 置信度)``；任何一位认不准就返回 ``(None, 0)``。

    置信度取四位里最弱的那一位 —— 取 min 而不是 mean，
    因为一个数字认错整条时间线就是错的，平均会把坏点稀释掉。
    """
    import numpy as np

    slots = load_digits()
    if not slots:
        return None, 0.0

    H, W = frame.shape[:2]
    gray = frame
    if len(gray.shape) == 3:
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    digits: List[int] = []
    conf = 1.0
    for slot in slots:
        x, y, w, h = slot["box"]
        px, py = int(round(x * W)), int(round(y * H))
        pw, ph = int(round(w * W)), int(round(h * H))
        if px < 0 or py < 0 or px + pw > W or py + ph > H:
            return None, 0.0
        crop = gray[py:py + ph, px:px + pw]
        sh, sw = slot["shape"]
        if crop.shape[0] != sh or crop.shape[1] != sw:
            return None, 0.0

        v = crop.astype(np.float32).flatten()
        v = v - v.mean()
        n = float(np.linalg.norm(v))
        if n < 1e-3:
            return None, 0.0

        scores = slot["templates"] @ (v / n)
        best = int(np.argmax(scores))
        top = float(scores[best])
        # 绝对分不够 = 这一位压根没认出来（多半是模板里没有这个数字）
        if top < MIN_SCORE:
            return None, 0.0
        if scores.size > 1:
            margin = top - float(np.partition(scores, -2)[-2])
            if margin < MIN_MARGIN:
                return None, 0.0
        digits.append(int(slot["labels"][best]))
        conf = min(conf, top)

    minutes = digits[0] * 10 + digits[1]
    seconds = digits[2] * 10 + digits[3]
    # 秒位不可能超过 59。出现这种读数只可能是匹配到了别的东西。
    if seconds > 59 or minutes > MAX_MINUTES:
        return None, 0.0
    return minutes * 60 + seconds, conf


def read_clock(frame) -> Optional[int]:
    """只要游戏内秒数，不关心置信度。"""
    return read_clock_frame(frame)[0]


# --------------------------------------------------------------------------
# 搜索式读时钟
# --------------------------------------------------------------------------
#
# 为什么要有「搜索」这一套
# ----------------------
# 上面那套按位切窗的读法，槽位是**标定值**：某台设备的分辨率、屏幕比例、
# 游戏版本下，时钟正好落在那些像素上。换一台设备、换一个比例、换一个
# 清晰度，槽位就偏了 —— 读出来的是「没读到」，而不是「读错了」。
# 用户换手机、换个录屏软件，整套时间线就废了。
#
# 所以再加一条路：**把画布按高度归一到标定几何，然后在顶边横向搜索**，
# 找出时钟那一组数字落在哪儿。数字的大小随画布高度缩放，所以归一之后
# 字的形状与间距都一致；位置随屏幕比例变，那就搜出来。
#
# 顺带解决**方向**：数字是横排的才说明画面正着。躺着的时候搜不到横排，
# 于是「哪个方向能搜到时钟」就是画面朝向的判据 ——
# 不依赖容器里的旋转元数据（有些录屏软件写错了，实测就遇到过）。
#
# 代价是搜索比定点慢，所以只在一开始做一次（:func:`detect_anchor`），
# 找到之后每帧都用固定的偏移去读。

#: 标定几何：把画布高度归一到这个值，数字的大小与间距随之归一。
WORK_H = 1200

#: 四个数字位在标定几何里的横向偏移与宽度（相对整组的起点）。
#:
#: 数值来自标定时的像素：M1 1819-1839、M2 1836-1856、
#: S1 1859-1879、S2 1876-1896，整组从 1819 开始。
GROUP_SLOTS: Tuple[Tuple[int, int], ...] = ((0, 20), (17, 20), (40, 20), (57, 20))

#: 整组数字的横向总跨度（像素，标定几何）。
GROUP_W = 77

#: 数字位的纵向位置与高度（标定几何，贴着画布顶边）。
SLOT_Y = 10
SLOT_H = 25

#: 横向搜索的步长。
SEARCH_STEP = 4

#: 搜索时允许的最低分。低于这个数就认为「这里没有时钟」。
#:
#: 定得低是因为归一之后字形被缩放、清晰度也未必够；
#: 真正的安全网是 :func:`build_clock_track` 的拟合自检 ——
#: 认错数字会让读数偏几十秒，离「游戏时间按真实速度走」那条直线足够远，
#: 一定会被剔掉。
SEARCH_MIN_SCORE = 0.16


def _slot_best_score(gray, slot, x: int, y: int, w: int, h: int) -> float:
    """一个数字位上的最佳模板得分。取不到就 -1。

    **切出来的块会被缩放到模板尺寸再比** —— 这一步是「换手机也能用」的关键：

    槽位是**相对坐标**（占画布宽高的比例），所以换设备、换分辨率、换录屏
    软件之后，数字相对画布的大小几乎不变；按高度归一后，数字的**绝对**像素
    也就和标定时一致（标定 20×25）。

    但归一后的宽度除不尽：0.007491 × 2571 = 19.3 → 切出 19×25，
    而模板是 20×25。早先的写法要求尺寸**完全相等**，差一个像素就整条路
    被否掉 —— 于是换任何一台设备都「读不出时钟」。缩放到模板尺寸再比，
    才是尺度无关的匹配。
    """
    import numpy as np

    if x < 0 or y < 0 or x + w > gray.shape[1] or y + h > gray.shape[0]:
        return -1.0
    crop = gray[y:y + h, x:x + w]
    sh, sw = slot["shape"]
    if crop.shape[0] != sh or crop.shape[1] != sw:
        import cv2

        crop = cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_AREA)
    v = crop.astype(np.float32).flatten()
    v = v - v.mean()
    n = float(np.linalg.norm(v))
    if n < 1e-3:
        return -1.0
    return float((slot["templates"] @ (v / n)).max())


def _normalize_canvas(frame, rotation: int):
    """按高度归一到标定几何，并按 ``rotation`` 转正。

    归一**只按高度**：数字大小随画布高度缩放，按高度归一之后字的形状与
    间距就和标定时一致；按宽度归一会把字横向拉扁，模板就不认了。
    """
    import cv2

    img = frame
    if rotation == 90:
        img = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        img = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        img = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    h, w = img.shape[:2]
    if h <= 0:
        return None, 0
    scale = WORK_H / float(h)
    w2 = max(GROUP_W + 8, int(round(w * scale)))
    work = cv2.resize(img, (w2, WORK_H), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    return gray, w2


def detect_anchor(frame, rotations: Sequence[int] = (0, 90, 180, 270)) -> Optional[Dict[str, Any]]:
    """找时钟：返回 ``{rotation, x, score}``，找不到返回 None。

    ``rotation`` 是把画面转正的旋转角（0/90/180/270），
    ``x`` 是时钟整组数字在标定几何里的横向起点。

    ``rotations`` 可以限定只试某几个方向 —— 方向已经由画面布局判出来的时候，
    就别让搜索再乱挑（搜索本身会挑到别的位置的相似纹理上）。
    """
    import numpy as np

    slots = load_digits()
    if not slots:
        return None

    # 数字太小就别搜了：认出来的只会是噪声，而搜索本身还会把
    # 「最像的地方」当成时钟（比如自家英雄头像）—— 那比读不到更糟。
    if frame is None or too_small_to_read(int(frame.shape[0])):
        return None

    best: Optional[Dict[str, Any]] = None
    for rotation in rotations:
        gray, w2 = _normalize_canvas(frame, rotation)
        if gray is None:
            continue
        best_x = -1
        best_score = -1.0
        for x0 in range(0, max(1, w2 - GROUP_W), SEARCH_STEP):
            total = 0.0
            ok = True
            for slot, (dx, dw) in zip(slots, GROUP_SLOTS):
                s = _slot_best_score(gray, slot, x0 + dx, SLOT_Y, dw, SLOT_H)
                if s < 0:
                    ok = False
                    break
                total += s
            if not ok:
                continue
            if total > best_score:
                best_score = total
                best_x = x0
        if best_x < 0:
            continue
        # 四位平均分才是可比的量（不同朝向可能只搜到部分位）
        avg = best_score / len(slots)
        if best is None or avg > best["score"]:
            best = {"rotation": rotation, "x": best_x, "score": avg}

    if best is None or best["score"] < SEARCH_MIN_SCORE:
        return None
    return best


def read_clock_anchored(frame, anchor: Dict[str, Any],
                        min_score: float = 0.18) -> Tuple[Optional[int], float]:
    """按已找到的时钟位置读数。

    找到位置之后就不用再搜了，每帧只切四块、算四次相关 —— 快得多。

    ``min_score`` 默认放得比 :data:`MIN_SCORE` 低：归一之后字形被缩放过，
    清晰度也未必够。真正的安全网是 :func:`build_clock_track` 的拟合自检 ——
    认错数字会让读数偏几十秒，离「游戏时间按真实速度走」那条直线足够远，
    一定会被剔掉。
    """
    import numpy as np

    slots = load_digits()
    if not slots or not anchor:
        return None, 0.0

    # 数字太小就明确认不出，不给乱码。
    if frame is None or too_small_to_read(int(frame.shape[0])):
        return None, 0.0

    gray, _ = _normalize_canvas(frame, int(anchor.get("rotation", 0)))
    if gray is None:
        return None, 0.0

    x0 = int(anchor.get("x", 0))
    digits: List[int] = []
    conf = 1.0
    for slot, (dx, dw) in zip(slots, GROUP_SLOTS):
        crop_gray = gray
        x = x0 + dx
        y = SLOT_Y
        w, h = dw, SLOT_H
        if x < 0 or y < 0 or x + w > crop_gray.shape[1] or y + h > crop_gray.shape[0]:
            return None, 0.0
        patch = crop_gray[y:y + h, x:x + w]
        sh, sw = slot["shape"]
        if patch.shape[0] != sh or patch.shape[1] != sw:
            import cv2

            patch = cv2.resize(patch, (sw, sh), interpolation=cv2.INTER_AREA)
        v = patch.astype(np.float32).flatten()
        v = v - v.mean()
        n = float(np.linalg.norm(v))
        if n < 1e-3:
            return None, 0.0
        scores = slot["templates"] @ (v / n)
        best = int(np.argmax(scores))
        top = float(scores[best])
        if top < min_score:
            return None, 0.0
        if scores.size > 1:
            margin = top - float(np.partition(scores, -2)[-2])
            if margin < MIN_MARGIN:
                return None, 0.0
        digits.append(int(slot["labels"][best]))
        conf = min(conf, top)

    minutes = digits[0] * 10 + digits[1]
    seconds = digits[2] * 10 + digits[3]
    if seconds > 59 or minutes > MAX_MINUTES:
        return None, 0.0
    return minutes * 60 + seconds, conf


# --------------------------------------------------------------------------
# 时间线
# --------------------------------------------------------------------------

def build_clock_track(samples: Sequence[ClockSample]) -> ClockTrack:
    """把读数拟合成「视频秒 → 游戏秒」的直线。

    自检靠的是**拟合**，不是数跳变：

      斜率    游戏秒 / 视频秒。正常正好 1；明显偏离只有两种可能 ——
              帧率元数据不对，或者这块根本不是时钟。
      残差    读数到直线的距离。时钟连续走，残差应当只有一两秒；
              残差很大说明读数在乱跳，那是匹配错了，不是时间在变。

    对局起点是这条直线的**截距**，不是「时钟第一次跳变」那种猜测。
    """
    import numpy as np

    track = ClockTrack(samples_total=len(samples))
    usable = [(s.t, float(s.game_s)) for s in samples if s.game_s is not None]
    track.samples_read = len(usable)

    if len(usable) < MIN_SAMPLES:
        track.reason = (
            f"只读到 {len(usable)} 个时刻的时钟，"
            f"至少要 {MIN_SAMPLES} 个才敢定时间线"
        )
        return track

    ts = np.array([p[0] for p in usable], dtype=np.float64)
    gs = np.array([p[1] for p in usable], dtype=np.float64)

    # 时间轴上抖动 ±1 秒是正常的（时钟在两次采样之间跳了）。
    # 先做一次中位数粗拟合，剔掉明显离群的点，再正式拟合。
    slope, intercept = _fit(ts, gs)
    resid = np.abs(gs - (slope * ts + intercept))
    keep = resid <= max(RESIDUAL_TOLERANCE, float(np.median(resid)) * 4 + 2)
    if keep.sum() >= MIN_SAMPLES:
        ts, gs = ts[keep], gs[keep]
        slope, intercept = _fit(ts, gs)
        resid = np.abs(gs - (slope * ts + intercept))

    track.slope = float(slope)
    track.residual = float(np.percentile(resid, 95))

    if abs(slope - 1.0) > SLOPE_TOLERANCE:
        track.reason = (
            f"读到 {len(ts)} 个时刻，但拟合出来游戏时间走得比真实快慢 "
            f"{slope:.3f} 倍（应当是 1.000）—— "
            f"多半是帧率元数据不对，或者这块不是时钟"
        )
        return track
    if track.residual > RESIDUAL_TOLERANCE:
        track.reason = (
            f"读数到时间线的偏差最大 {track.residual:.1f} 秒 —— "
            f"时钟不该这样跳，多半是匹配到了别的东西"
        )
        return track

    track.game_start_t = float(-intercept / slope)
    track.game_minutes_at = {float(t): float(g) / 60.0 for t, g in zip(ts, gs)}
    track.ok = True
    track.reason = (
        f"认出时钟：读到 {len(ts)} 个时刻，对局从视频第 "
        f"{track.game_start_t:.0f} 秒开始，误差 ±{track.residual:.1f} 秒"
    )
    return track


def _fit(ts, gs):
    """最小二乘拟合 gs = slope * ts + intercept。"""
    import numpy as np

    A = np.vstack([ts, np.ones_like(ts)]).T
    sol, *_ = np.linalg.lstsq(A, gs, rcond=None)
    return float(sol[0]), float(sol[1])


#: 时钟区域二值化的灰度下限：数字是那一块最亮的东西，
#: 比游戏场景漏过来的亮点高。
CLOCK_BIN_LEVEL = 150

#: 掩码归一到的尺寸：只用来比较「变了没有」，不需要看清数字。
CLOCK_MASK_W, CLOCK_MASK_H = 24, 10


def clock_mask(canvas, anchor: Dict[str, Any]) -> Optional[Any]:
    """时钟区域里「亮着的数字」的二值掩码。

    读不出数字时（录屏清晰度不够，数字只有两三个像素宽），
    这个掩码仍然有用：**它每秒变一次**（MM:SS 走秒），
    数它变了多少次就知道过了多少秒 —— 时间的绝对起点由 HUD 就位来定。

    返回值是定尺后的掩码（``CLOCK_MASK_W`` × ``CLOCK_MASK_H`` 的 0/1 数组），
    直接比相等与否即可。
    """
    import cv2
    import numpy as np

    if canvas is None or not anchor:
        return None
    h, w = canvas.shape[:2]
    if h <= 0 or w <= 0:
        return None

    # 锚点是在「按高度归一」的几何里算出来的，换算回原画布
    scale = h / float(WORK_H)
    x0 = int(int(anchor.get("x", 0)) * scale)
    y0 = int(SLOT_Y * scale)
    x1 = int((int(anchor.get("x", 0)) + GROUP_W) * scale)
    y1 = int((SLOT_Y + SLOT_H) * scale)
    if x1 - x0 < 3 or y1 - y0 < 2:
        return None
    crop = canvas[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    mask = (gray > CLOCK_BIN_LEVEL).astype(np.uint8)
    return cv2.resize(mask, (CLOCK_MASK_W, CLOCK_MASK_H),
                      interpolation=cv2.INTER_NEAREST)


def mask_changed(a, b) -> bool:
    """两个掩码是否不同（用**像素总数 + 内容**一起比，避免抖动误判）。"""
    import numpy as np

    if a is None or b is None:
        return False
    if a.shape != b.shape:
        return True
    return int(a.sum()) != int(b.sum()) or not np.array_equal(a, b)
