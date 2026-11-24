"""确定性校验：拿龙/拆塔必须「玩家在跟前 + 目标已消失/爆」。

小地图能给出两块**不含模型**的事实：
1. 玩家位置（绿圈 find_hero_spot）；
2. 龙（主宰/暴君）图标位置（find_objective_marks）—— 图标在=龙活着，
   图标消失=龙死了。

所以「拿龙」可以完全确定性判定：
- 玩家在龙前：绿圈落在龙坑附近；
- 龙死：事件前该龙图标在、事件后图标消失。

拆塔需要主画面的塔血条（小地图没有塔的独立标记）——这一块暂留空，
先用模型闸兜底，塔血条像素检测是下一步。
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional, Sequence, Tuple

from ..collection import minimap as mm

#: 判「玩家在龙坑里」的归一化距离阈值（与 minimap.PIT_RADIUS 一致）。
PIT_RADIUS = 0.09

#: 事件前 / 后各看多少秒。
SPAN = 30.0


def _crop(frame: Any, geometry: mm.MapGeometry):
    return mm.minimap_crop(frame, geometry)


def _player_near_pit(crop: Any, geometry: mm.MapGeometry) -> Optional[str]:
    """玩家绿圈是不是在某个龙坑附近。返回 pit 名或 None。

    坑的位置是固定的（PIT_ANCHORS），跟龙活着没有关系 —— 所以这里
    不查龙图标，只用固定锚点 + 玩家绿圈，避免和「龙死没死」耦合。
    """
    spot = mm.find_hero_spot(crop)
    if spot is None:
        return None
    best = None
    best_d = 1e9
    for name, (nx, ny) in mm.PIT_ANCHORS.items():
        d = ((spot.nx - nx) ** 2 + (spot.ny - ny) ** 2) ** 0.5
        if d < best_d:
            best_d = d
            best = name
    if best is not None and best_d <= PIT_RADIUS:
        return best
    return None


def _has_mark(frames: Sequence[Any], geo: mm.MapGeometry) -> bool:
    """这一段画面里有没有龙（主宰/暴君）图标。"""
    for f in frames:
        crop = _crop(f, geo)
        if crop is not None and mm.find_objective_marks(crop):
            return True
    return False


def verify_dragon(video_path: str, t: float, *,
                  geometry: Optional[mm.MapGeometry] = None,
                  transform: Any = None,
                  read_frames: Optional[Callable[..., List[Any]]] = None
                  ) -> Tuple[bool, bool]:
    """确定性判断「玩家在龙前」与「龙死」。

    返回 (player_in_front, dragon_died)。
    """
    if read_frames is None and not os.path.isfile(video_path):
        return (False, False)

    geo = geometry or mm.MapGeometry.load()

    def _frames_between(lo: float, hi: float, step: float = 2.0) -> List[Any]:
        if read_frames is not None:
            times = [round(x, 1) for x in _arange(lo, hi, step)]
            return read_frames(video_path, times, transform)
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        out: List[Any] = []
        try:
            cursor = lo
            while cursor <= hi + 1e-6:
                cap.set(cv2.CAP_PROP_POS_MSEC, cursor * 1000.0)
                ok, frame = cap.read()
                if ok and frame is not None:
                    out.append(frame)
                cursor += step
        finally:
            cap.release()
        return out

    before = _frames_between(max(0.0, t - SPAN), t)
    after = _frames_between(t + 2.0, t + SPAN)

    player_in_front = any(_player_near_pit(_crop(f, geo), geo) for f in before)
    # 龙死：事件前有龙图标、事件后没了
    dragon_died = (_has_mark(before, geo) and not _has_mark(after, geo)
                   and bool(before) and bool(after))
    return (bool(player_in_front), bool(dragon_died))


def _arange(lo: float, hi: float, step: float) -> List[float]:
    out: List[float] = []
    x = lo
    while x <= hi + 1e-6:
        out.append(x)
        x += step
    return out
