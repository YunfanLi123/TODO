"""从录屏里导出**干净的小地图底图**（标定区域用）。

为什么要导出底图
----------------
小地图上除了地形，还有会动的标记（英雄头像、绿圈、信号），
直接在录屏上框区域很难框准 —— 框的时候分不清哪块是地形、哪块是标记。

按时间取**中位数**就能把动的去掉：同一个像素在整个对局里大多数时候
是同一块地形，中位数就是地形本身；偶尔被标记盖住的那几帧被中位数吃掉。

用法
----
::

    python tools/extract_minimap_reference.py <视频> \
        --out app/collection/minimap_reference.png
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.collection import minimap as mm_mod          # noqa: E402
from app.collection import video_upload as vu         # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="导出小地图底图")
    ap.add_argument("video")
    ap.add_argument("--out", default=os.path.join(ROOT, "app", "collection",
                                                  "minimap_reference.png"))
    ap.add_argument("--samples", type=int, default=120,
                    help="取多少帧做中位数（覆盖整局即可）")
    ap.add_argument("--scale", type=int, default=4,
                    help="导出时放大几倍（标定时看得清）")
    args = ap.parse_args()

    geo = mm_mod.MapGeometry.load()
    if geo.minimap_box is None:
        print("小地图框还没标定，先跑 tools/calibrate_hud.py 或手填 map_geometry.json")
        return 1

    video = vu.UploadedVideo(video_id="mmref", original_name="t.mp4",
                             path=args.video, size=1)
    video = vu.probe(video)
    view = vu.video_view(video, hud_layout=vu.DEFAULT_HUD_LAYOUT)
    rot = int(view.get("rotation", 0))
    band = view.get("band")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: list = []
    step = max(1, total // max(1, args.samples))
    idx = 0
    while len(frames) < args.samples:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if not ok:
            break
        canvas = vu.crop_to_band(vu.orient_frame(cv2, f, rot), band)
        idx += step
        if canvas is None:
            continue
        mm = mm_mod.minimap_crop(canvas, geo)
        if mm is None:
            continue
        frames.append(mm)
    cap.release()
    if len(frames) < 5:
        print("取到的帧太少")
        return 1
    stack = np.stack(frames)
    ref = np.median(stack, axis=0).astype(np.uint8)
    big = cv2.resize(ref, (ref.shape[1] * args.scale, ref.shape[0] * args.scale),
                     interpolation=cv2.INTER_NEAREST)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, big)
    print(f"{len(frames)} 帧取中位数 -> {args.out}（原图 {ref.shape[1]}x{ref.shape[0]}，"
          f"放大 {args.scale} 倍）")
    print("接下来用 tools/calibrate_map.py 在这张图上框区域。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
