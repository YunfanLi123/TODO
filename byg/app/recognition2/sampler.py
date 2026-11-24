"""**按画面变化切段** —— 新识别子系统的第一步。

为什么不能按规则窗口切
----------------------
旧系统的分析单元是从**规则库**倒推的（``rules/density.py``：
每个局内分钟有多少条规则能判 → 那里密就多切几刀）。

问题很具体：真实对局是**连续**的，规则窗口是**稀疏**的。
画面里 3:10~3:40 发生的事，只要不落在规则窗口的边界上，
就归不进任何一段 —— 那一段的信息**整个丢掉**。

而且方向是反的：应该是「画面里发生了什么」决定切分，
→「规则要什么」去对齐它（见 :mod:`app.recognition2.reader`）。

切分信号（从便宜到贵）
----------------------
1. **小地图区域差异** —— 玩家位置变了就是一段新的
   （用现成的 ``map_geometry.json`` 标定）；
2. **全图模糊后差异** —— 画面整体变了（团战、推塔、换场景）；
3. **固定间隔兜底** —— 画面没变也要定期切，
   否则「什么都没发生」的那 2 分钟会黏成一大段。

只用 OpenCV，不调模型 —— 切分是廉价的，判断「变了没有」不需要模型。

阈值是**量出来的**，不是猜的
----------------------------
2026-10-31 在 ``赵云.mp4``（1280x574, 60fps, 1151 秒）上实测
（``tools/measure_signals.py``），相邻采样点（2 秒间隔）的差异分布：

=================  ======  ======  ======  ======  ======
信号                min    p25    中位    p90    p95
=================  ======  ======  ======  ======  ======
小地图逐像素差      0.000  0.027  0.036  0.060  0.079
全图逐像素差        0.002  0.052  0.096  0.128  0.139
全图**模糊后**差    0.001  0.027  0.069  0.101  0.112
=================  ======  ======  ======  ======  ======

**全图逐像素差不可用**：中位数就 0.096 —— 游戏画面一直在动
（伤害数字、粒子特效、技能动画），阈值压到 0.16 只能命中 1~2% 的点，
于是每段都切不动（第一版就是这么废掉的）。
**模糊后**把局部高频抹掉，大尺度变化才浮出来，中位降到 0.069。

所以阈值取在 p90 略上：小地图 0.075、场景 0.14。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Segment:
    """切出来的一段。"""

    index: int
    start: float              # 视频秒数
    end: float                # 视频秒数
    #: 这一段里挑出的**代表帧**（视频秒数）。
    frames: List[float] = field(default_factory=list)
    #: 切这一刀的理由（写下来，方便人核对切得对不对）。
    reason: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "duration": round(self.duration, 2),
            "frames": [round(f, 2) for f in self.frames],
            "reason": self.reason,
        }


@dataclass
class SamplerConfig:
    """切分参数。阈值来自真实录屏实测（见模块文档）。"""

    #: 采样间隔（秒）。抽帧的最小步长。
    step: float = 2.0
    #: 小地图差异超过这个值就切一刀（玩家位置明显变了）。
    map_diff_threshold: float = 0.075
    #: 全图模糊后差异超过这个值就切一刀（场景明显变了）。
    scene_diff_threshold: float = 0.14
    #: 一段最长多少秒（到点必切，防止黏成一大段）。
    max_segment: float = 20.0
    #: 一段最短多少秒（太短不成立，合并进上一段）。
    min_segment: float = 5.0
    #: 每段最多给模型几帧。
    max_frames_per_segment: int = 4
    #: 一段至少给几帧。
    min_frames_per_segment: int = 2
    #:
    #: **这里没有「读局内时间」的开关**
    #: ---------------------------------
    #: 2026-10-31 之前有（``read_clock`` + ``clock_sample_step``），走的是
    #: OpenCV 模板匹配。那条路已经撤掉 —— 它在这段录屏上 5 次读错 4 次
    #: （02:38 读成 02:16、02:50 读成 11:21、12:13 读成 02:43…），
    #: 错得毫无规律，光靠抗离群拟合也拦不住一部分。
    #:
    #: 现在局内时刻由**多模态模型逐帧读钟**建出来，
    #: 见 :mod:`app.recognition2.clockaxis`。
    #: 采样器只管切段 —— 切段靠画面差分，不需要知道几点几分。


class ContentSampler:
    """按画面变化把视频切成段。

    用法::

        sampler = ContentSampler(video_path)
        segments = sampler.sample()
    """

    def __init__(self, video_path: str, config: Optional[SamplerConfig] = None):
        self.video_path = video_path
        self.config = config or SamplerConfig()
        self._geom = None

    # -- 标定 ------------------------------------------------------------

    def _geometry(self):
        """小地图标定。**复用旧模块**（口径必须一致，不能各标一套）。"""
        if self._geom is None:
            try:
                from app.collection.minimap import MapGeometry
                self._geom = MapGeometry.load()
            except Exception:                          # noqa: BLE001
                self._geom = None
        return self._geom

    def _minimap_crop(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """裁出小地图。

        ``map_geometry.json`` 里的 ``minimap_box`` 是**相对坐标**
        ``[x0, y0, x1, y1]``（0~1），不是像素 —— 第一版按像素解析，
        裁出来是空的，于是小地图信号永远是 0。
        """
        geom = self._geometry()
        if geom is None:
            return None
        try:
            from app.collection.minimap import minimap_crop
            return minimap_crop(frame, geom)
        except Exception:                              # noqa: BLE001
            return None

    # -- 差异 ------------------------------------------------------------

    @staticmethod
    def _hist_diff(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        """两张图的直方图差异（0~1）。"""
        if a is None or b is None or a.size == 0 or b.size == 0:
            return 0.0
        ga = cv2_hist(a)
        gb = cv2_hist(b)
        denom = float(np.sum(ga) + np.sum(gb))
        if denom <= 0:
            return 0.0
        return float(np.sum(np.abs(ga - gb)) / denom)

    @staticmethod
    def _pixel_diff(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        """逐像素平均差（0~1）。比直方图更能看出位移。"""
        if a is None or b is None or a.shape != b.shape:
            return 0.0
        ga = a.astype("float32")
        gb = b.astype("float32")
        return float(np.mean(np.abs(ga - gb)) / 255.0)

    # -- 采样 ------------------------------------------------------------

    def sample(self) -> List[Segment]:
        """切段。返回按时间排好的段列表。

        **取帧一律按时间戳 seek，不用 ``grab()`` 逐帧推进**
        --------------------------------------------------
        实测（2026-10-31，赵云.mp4，1280x574）：
        ``cap.grab()`` 推进 21601 次取到的帧，画面上的局内时间是 **03:40**（220 秒），
        而帧号 21600 的**真实 PTS** 是 360.985 秒、对应画面是 **03:33**（212 秒）——
        **差 6.7 秒，是另一张画面**（平均像素差 38）。

        也就是说这段录屏上 ``grab()`` 会**跳过帧**，「grab 了 N 次」不等于
        「显示顺序的第 N 帧」。用它推进取到的画面，**时间戳会指错**：
        段边界落在哪一秒、证据帧是哪一秒，全都会错。

        正确做法：按 ``CAP_PROP_POS_MSEC`` seek 到目标时刻再读，
        并用读回来的 **真实 PTS**（``cap.get(POS_MSEC)``）当时间戳。
        慢一点（每次 seek 要从关键帧解起），但时间是对的。
        """
        import cv2

        cfg = self.config
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError("打不开视频：%s" % self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = (total / fps) if fps else 0.0

        segments: List[Segment] = []
        cur: Optional[Segment] = None
        prev_full: Optional[np.ndarray] = None
        prev_map: Optional[np.ndarray] = None
        target = 0.0

        while True:
            cap.set(cv2.CAP_PROP_POS_MSEC, target * 1000.0)
            ok, frame = cap.read()
            if not ok:
                break
            # 用**真实 PTS** 当时间戳 —— 不用 ``target`` 推算。
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            full_small = _blur(frame, 64)
            map_crop = self._minimap_crop(frame)
            map_small = _shrink(map_crop, 96) if map_crop is not None else None

            map_diff = self._pixel_diff(prev_map, map_small) if prev_map is not None else 0.0
            full_diff = self._pixel_diff(prev_full, full_small) if prev_full is not None else 0.0

            if cur is None:
                cur = Segment(index=0, start=t, end=t)
            else:
                span = t - cur.start
                reason = ""
                if span >= cfg.max_segment:
                    reason = "到段长上限"
                elif span >= cfg.min_segment:
                    if map_diff >= cfg.map_diff_threshold:
                        reason = "玩家位置变化 %.3f" % map_diff
                    elif full_diff >= cfg.scene_diff_threshold:
                        reason = "画面场景变化 %.3f" % full_diff
                if reason:
                    # 段尾就是**这一刀的位置**（下一段从这里开始）——
                    # 写成上一帧会白丢一个采样步，段与段之间看着像有缺口。
                    cur.end = t
                    cur.reason = reason
                    self._fill_frames(cur)
                    segments.append(cur)
                    cur = Segment(index=len(segments), start=t, end=t)

            cur.frames.append(t)
            cur.end = t
            prev_full = full_small
            prev_map = map_small

            target += cfg.step
            if duration and target >= duration:
                break

        cap.release()
        if cur is not None and cur.frames:
            cur.end = max(cur.end, cur.frames[-1])
            self._fill_frames(cur)
            segments.append(cur)

        return segments

    def _fill_frames(self, seg: Segment) -> None:
        """每段的代表帧：均匀取 N 个（首尾必留）。"""
        cfg = self.config
        if not seg.frames:
            return
        n = len(seg.frames)
        want = max(cfg.min_frames_per_segment,
                   min(cfg.max_frames_per_segment, n))
        if n <= want:
            return
        idx = np.linspace(0, n - 1, want).round().astype(int).tolist()
        picked = [seg.frames[i] for i in sorted(set(idx))]
        seg.frames = picked


def _shrink(img: Optional[np.ndarray], width: int) -> Optional[np.ndarray]:
    """缩到固定宽度（比较用的）。"""
    if img is None or img.size == 0:
        return None
    import cv2

    h = max(1, int(img.shape[0] * width / max(1, img.shape[1])))
    return cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)


def _blur(img: Optional[np.ndarray], width: int,
          ksize: int = 5) -> Optional[np.ndarray]:
    """重采样 + 高斯模糊。

    **为什么必须模糊**：游戏画面里伤害数字、技能特效、粒子全是**高频局部**变化，
    逐像素差里全是它们的动静（中位数 0.096），真正的大尺度变化反而被淹没。
    模糊之后局部细节被抹平，剩下的是「镜头换地方了 / 团战了」这类真变化。
    """
    small = _shrink(img, width)
    if small is None:
        return None
    import cv2

    return cv2.GaussianBlur(small, (ksize, ksize), 0)


def cv2_hist(img: np.ndarray) -> np.ndarray:
    """灰度直方图（归一化）。"""
    import cv2

    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = cv2.calcHist([g], [0], None, [32], [0, 256])
    total = float(np.sum(h))
    if total <= 0:
        return h.flatten()
    return (h / total).flatten()
