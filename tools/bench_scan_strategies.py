"""实测几种扫描策略在**真实录屏**上的速度。

为什么要有这个脚本
------------------
扫描策略的选择曾依赖「seek 更快」这个假设定的 —— 代码注释里写着
「跳过去的那几帧根本不解码」。拿真实录屏一测，这个假设是错的：
1.7 GB / 89.7 fps 的文件关键帧稀疏，每次 seek 都要从上一个关键帧解起。

于是要换策略。但「顺序解码一定更快」也只是另一个猜测，得测。
这个脚本把三种策略放在同一段真实视频上跑同一件事：

    seek      每 N 帧 seek 一次再 read（当前实现）
    seq-read  顺序解码每一帧，只在采样点做处理
    seq-grab  顺序解码每一帧，但跳过的帧只 grab 不 retrieve

判据只有一个：**处理的有效样本数 / 墙钟秒数**。三个策略做的是同一件事
（采样 + 缩略图 + 读时钟），所以这个比值直接可比。

用法::

    .\\.venv\\Scripts\\python.exe tools\\bench_scan_strategies.py "<录屏>" [--seconds 60]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# 让 `from app.collection import clock` 能用 —— 脚本在 tools\ 下，
# 直接跑时 sys.path[0] 是 tools\，不是项目根。
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

sys.stdout.reconfigure(encoding="utf-8")


def _clock_read(frame) -> Optional[int]:
    """读时钟；模板不在（换机型）时安静跳过。"""
    from app.collection import clock

    try:
        return clock.read_clock(frame)
    except Exception:                       # noqa: BLE001
        return None


def _thumb(cv2, frame):
    import numpy as np

    return cv2.cvtColor(
        cv2.resize(frame, (16, 16), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2GRAY,
    ).astype(np.float32)


def bench_seek(cv2, path: str, fps: float, total: int,
               sample_sec: float, limit: int) -> Dict[str, Any]:
    """当前实现：每 sample_sec 秒 seek 一次。"""
    cap = cv2.VideoCapture(path)
    step = max(1, int(round(sample_sec * fps)))
    max_idx = min(total - 1, limit) if limit else total - 1
    samples = 0
    frames_decoded = 0
    t0 = time.perf_counter()
    idx = 0
    while idx <= max_idx:
        if idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames_decoded += 1
        _thumb(cv2, frame)
        _clock_read(frame)
        samples += 1
        idx += step
    dt = time.perf_counter() - t0
    cap.release()
    return {
        "strategy": "seek",
        "samples": samples,
        "frames_decoded": frames_decoded,
        "seconds": round(dt, 2),
        "samples_per_sec": round(samples / dt, 2) if dt else 0,
    }


def bench_seq_read(cv2, path: str, fps: float, total: int,
                   sample_sec: float, limit: int) -> Dict[str, Any]:
    """顺序解码，每一帧都 retrieve，只在采样点做处理。"""
    cap = cv2.VideoCapture(path)
    step = max(1, int(round(sample_sec * fps)))
    max_idx = min(total - 1, limit) if limit else total - 1
    samples = 0
    frames_decoded = 0
    t0 = time.perf_counter()
    for idx in range(max_idx + 1):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames_decoded += 1
        if idx % step == 0:
            _thumb(cv2, frame)
            _clock_read(frame)
            samples += 1
    dt = time.perf_counter() - t0
    cap.release()
    return {
        "strategy": "seq-read",
        "samples": samples,
        "frames_decoded": frames_decoded,
        "seconds": round(dt, 2),
        "samples_per_sec": round(samples / dt, 2) if dt else 0,
    }


def bench_seq_grab(cv2, path: str, fps: float, total: int,
                   sample_sec: float, limit: int) -> Dict[str, Any]:
    """顺序解码，跳过的帧只 grab 不 retrieve。"""
    cap = cv2.VideoCapture(path)
    step = max(1, int(round(sample_sec * fps)))
    max_idx = min(total - 1, limit) if limit else total - 1
    samples = 0
    frames_decoded = 0
    t0 = time.perf_counter()
    for idx in range(max_idx + 1):
        if idx % step == 0:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            _thumb(cv2, frame)
            _clock_read(frame)
            samples += 1
        else:
            if not cap.grab():
                break
        frames_decoded += 1
    dt = time.perf_counter() - t0
    cap.release()
    return {
        "strategy": "seq-grab",
        "samples": samples,
        "frames_decoded": frames_decoded,
        "seconds": round(dt, 2),
        "samples_per_sec": round(samples / dt, 2) if dt else 0,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="扫描策略基准")
    ap.add_argument("video", help="录屏路径")
    ap.add_argument("--seconds", type=float, default=60.0,
                    help="只跑视频开头的多少秒（默认 60）")
    ap.add_argument("--sample-sec", type=float, default=0.3,
                    help="采样间隔秒（默认 0.3，与线上一致）")
    args = ap.parse_args(argv)

    if not os.path.exists(args.video):
        print(f"找不到视频：{args.video}")
        return 1

    import cv2

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("打不开视频")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    limit = int(args.seconds * fps) if args.seconds else 0

    print(f"视频：{os.path.basename(args.video)}")
    print(f"{w}x{h}  {fps:.2f} fps  {total} 帧  "
          f"{total / fps / 60:.1f} 分钟")
    print(f"基准区间：前 {args.seconds:.0f} 秒（{limit} 帧），"
          f"采样间隔 {args.sample_sec}s\n")

    results = []
    for fn in (bench_seek, bench_seq_read, bench_seq_grab):
        r = fn(cv2, args.video, fps, total, args.sample_sec, limit)
        results.append(r)
        print(f"{r['strategy']:<10} 样本 {r['samples']:>5}  "
              f"解码 {r['frames_decoded']:>6} 帧  "
              f"{r['seconds']:>7.2f}s  "
              f"{r['samples_per_sec']:>6.2f} 样本/秒")

    best = max(results, key=lambda r: r["samples_per_sec"])
    print(f"\n最快：{best['strategy']}  "
          f"({best['samples_per_sec']:.2f} 样本/秒，"
          f"是 seek 的 {best['samples_per_sec'] / max(1e-9, results[0]['samples_per_sec']):.2f} 倍)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
