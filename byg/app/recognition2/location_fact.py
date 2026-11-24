"""位置事实：从小地图读「玩家在哪条路」，不再让模型猜。

两条经验（2026-11-14 实测）
--------------------------
1. **死亡/击杀那一帧经常没有绿圈**（玩家已死、标记被击杀提示盖住）——
   所以取事件**前 12 秒**的帧投票，而不是只看事件帧；
2. 区域表（zone_of）会把上路边线判成「己方野区」——**先判路**
   （lane_of：看主宰/暴君图标），判不出路再看河道/野区。

输出只用：上路 / 中路 / 下路 / 河道 / 野区；判不出就空（报告写「未定位」）。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ..collection import minimap as mm

#: 事件前取多少秒投票（死亡前的走位常被击杀提示盖住，取长一点）。
PRE_SECONDS = 25


def _lane_label(lane: str, flipped: Optional[bool]) -> str:
    """lane_of 的结果 -> 物理路名（上路/中路/下路）。"""
    if lane == "中路附近":
        return "中路"
    if lane == "对抗路":
        if flipped is True:
            return "下路"
        if flipped is False:
            return "上路"
        return ""
    if lane == "发育路":
        if flipped is True:
            return "上路"
        if flipped is False:
            return "下路"
        return ""
    return ""


def _vote_lane(frames: List[Any], geometry: mm.MapGeometry) -> str:
    """多帧投票：哪条路出现最多就是哪条。"""
    counts: Dict[str, int] = {}
    for frame in frames:
        crop = mm.minimap_crop(frame, geometry)
        if crop is None:
            continue
        spot = mm.find_hero_spot(crop)
        if spot is None:
            continue
        marks = mm.find_objective_marks(crop)
        flipped = mm.map_is_flipped(crop)
        label = _lane_label(mm.lane_of(spot.nx, spot.ny, marks), flipped)
        if label:
            counts[label] = counts.get(label, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _vote_area(frames: List[Any], geometry: mm.MapGeometry) -> str:
    """判不出路时再看河道 / 野区（区域表）。"""
    counts: Dict[str, int] = {}
    for frame in frames:
        crop = mm.minimap_crop(frame, geometry)
        if crop is None:
            continue
        spot = mm.find_hero_spot(crop)
        if spot is None:
            continue
        marks = mm.find_objective_marks(crop)
        flipped = mm.map_is_flipped(crop)
        zone = geometry.zone_of(spot.nx, spot.ny, marks=marks, flipped=flipped)
        if zone == "河道":
            label = "河道"
        elif zone in ("己方野区", "敌方野区"):
            label = "野区"
        else:
            label = ""
        if label:
            counts[label] = counts.get(label, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda kv: kv[1])[0]


def locate_at(video_path: str, t: float, *,
              geometry: Optional[mm.MapGeometry] = None,
              pre: float = PRE_SECONDS,
              transform: Any = None) -> str:
    """读事件前 pre 秒的小地图，返回 上路/中路/下路/河道/野区（或空）。"""
    import cv2

    if not os.path.isfile(video_path):
        return ""
    geo = geometry or mm.MapGeometry.load()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return ""
    frames: List[Any] = []
    try:
        start = max(0.0, float(t) - pre)
        cursor = start
        while cursor <= float(t) + 1e-6 and len(frames) < int(pre) + 1:
            cap.set(cv2.CAP_PROP_POS_MSEC, cursor * 1000.0)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
            cursor += 1.0
    finally:
        cap.release()
    if not frames:
        return ""
    lane = _vote_lane(frames, geo)
    if lane:
        return lane
    return _vote_area(frames, geo)
