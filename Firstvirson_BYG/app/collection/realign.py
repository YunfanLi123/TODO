"""对局起点校正：改一个数，就把每一段**重新读一遍**。

为什么不能把算出来的结果平移一下
----------------------------------
「对局从视频第几秒开始」这个数一改，「第 6 分钟」对应的画面就换了。

原来那批帧是按**旧起点**摆的 —— 起点偏了 30 秒，它们显示的局内时刻
就整体偏了 30 分钟里的半分钟。于是平移旧结论是不行的：拿错时刻的
画面去说「这局你在 6 分钟刷了野」，报告上看着句句有据，其实句句
说的都不是那一段的事。

这类错最难被发现：结论的方向全对（该刷野刷了野、该转线转了线），
只是**证据对不上画面**。玩家点开「看这一段」看到的是另一段，
却看不出来哪里出了问题。

所以这里做的事只有一件：**按新起点重新抽帧、重新识别**。

    玩家给的数 → 只决定「去看哪几秒」
    画面里读到什么 → 一律重新读

代价是一次重读：重新 seek 抽帧（几十帧，十几秒）＋逐窗识别。
换来的是报告里每一条结论都对得上它标的那几分钟画面。

只读的部分
----------
- **英雄轨迹**（小地图候选）是在扫描时按**视频秒**采的，与起点无关，
  不重采。窗口切轨迹时按新的 t0/t1 重新切即可。
- **局内分钟**（``time_min``）是规则窗自己的时间轴（窗从规则的
  ``time_min between`` 倒推），跟视频起点无关，不动。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import video_upload as vu
from .window_analysis import Window, WindowBundle

#: 起点改动小于这个秒数就不重读。
#:
#: 差一秒看不出区别，而重读一遍要重新抽帧 —— 拿几十秒换一秒不划算。
MIN_SHIFT_SEC = 1.0


def shift_of(summary: Dict[str, Any], new_start: float) -> float:
    """新起点相对旧起点挪了多少秒（正数=往后挪）。"""
    old = summary.get("match_start_t")
    if old is None:
        return 0.0
    return float(new_start) - float(old)


def needs_realign(summary: Dict[str, Any], new_start: Optional[float]) -> bool:
    """这次提交要不要重读一遍。

    没给数 → 不重读；给的是同一个数 → 不重读；差不到一秒 → 不重读。
    """
    if new_start is None:
        return False
    if summary.get("match_start_t") is None:
        return False
    return abs(shift_of(summary, float(new_start))) >= MIN_SHIFT_SEC


def _reshoot(video, base_dir: str, summary: Dict[str, Any],
             delta: float) -> List[Dict[str, Any]]:
    """按新起点重新抽帧，返回新的时间片列表。

    每一帧都**重新从视频里读** —— 挪位置不行，画面内容才是证据。
    文件名沿用原来的（同样的序号、同样的前缀），报告里的图片地址
    不用改，也不用留下两份。
    """
    shots = list(summary.get("rounds") or [])
    if not shots:
        return []
    prefix = "phase" if (summary.get("product") or "") == "phase" else "slice"
    wanted: List[Tuple[str, float]] = [
        (str(s.get("phase") or ""), float(s.get("timestamp") or 0.0) + delta)
        for s in shots
    ]
    fresh = vu.extract_frames_at(video, wanted, base_dir, prefix)
    out: List[Dict[str, Any]] = []
    for old, shot in zip(shots, fresh):
        merged = dict(old)
        merged.update({
            "timestamp": shot.get("timestamp", 0.0),
            "image": shot.get("image", ""),
            "image_url": shot.get("image_url", ""),
        })
        out.append(merged)
    return out


def realign_windows(
    video,
    base_dir: str,
    summary: Dict[str, Any],
    new_start: float,
    hero_track: Optional[Sequence[Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """按新起点重读一遍每一窗。

    返回 ``(新窗口列表, 新时间片列表)``。窗口列表的形状与扫描时
    产出的完全一致（``lo``/``hi``/``slices``/``facts``…），
    因此可以直接交给 :func:`app.collection.video_upload.build_skeleton`。
    """
    delta = shift_of(summary, new_start)
    shots = _reshoot(video, base_dir, summary, delta)
    if not shots:
        return [], []
    # 时间片按 round_no 索引：同一片可能同时落在好几个窗里
    # （规则窗本来就可能重叠），按序号对齐才不会串。
    by_round: Dict[Any, Dict[str, Any]] = {
        s.get("round_no"): s for s in shots
    }

    # 键位档位按新帧重挑 —— 帧换了，档位可能也变了。
    regions = vu._hud_regions_for(video, base_dir, shots)

    out: List[Dict[str, Any]] = []
    for w in summary.get("windows") or []:
        slices: List[Dict[str, Any]] = []
        for s in w.get("slices") or []:
            m = by_round.get(s.get("round_no"))
            if m is None:
                continue
            slices.append({
                "round_no": m.get("round_no"),
                "timestamp": m.get("timestamp"),
                "game_minute": m.get("game_minute"),
                "image": m.get("image", ""),
                "image_url": m.get("image_url", ""),
            })
        if not slices:
            continue
        lo = int(w.get("lo", 0))
        hi = int(w.get("hi", 0))
        bundle = WindowBundle(window=Window(lo=lo, hi=hi), slices=slices)
        # 窗内重新识别：事实、探测器、证据时刻全部重新算。
        # 轨迹按**新的窗区间**切 —— 区间变了，切出来的点就变了。
        reading = vu._recognize_window(
            video, base_dir, bundle,
            str(w.get("segment_key") or w.get("key") or ""),
            regions, list(hero_track or []),
            t0=float(new_start) + lo * 60.0,
            t1=float(new_start) + (hi + 1) * 60.0,
        )
        d = dict(w)
        d["slices"] = slices
        d["facts"] = reading
        out.append(d)
    return out, shots
