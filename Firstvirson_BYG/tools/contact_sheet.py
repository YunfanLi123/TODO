"""把视频里的一批时刻拼成一张联系表（contact sheet）。

用途
----
设计「BP 帧序列」的取样落点时，不能靠猜 —— 得先看清 0~200 秒里
每一段到底是什么画面（禁用选人 / 加载 / 开局）。

做法：按给定时刻各取一帧，等比缩到小尺寸，按网格拼成一张图，
每格左上角标出视频秒数与游戏内读数（读不到就只标视频秒）。

用法::

    .\\.venv\\Scripts\\python.exe tools\\contact_sheet.py "<录屏>" 0 10 30 60 120 180 195 [--out .tmp\\sheet.jpg]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

sys.stdout.reconfigure(encoding="utf-8")


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="生成视频时刻联系表")
    ap.add_argument("video")
    ap.add_argument("times", nargs="+", type=float, help="视频秒数")
    ap.add_argument("--out", default=".tmp/contact_sheet.jpg")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--cell-width", type=int, default=560)
    args = ap.parse_args(argv)

    import cv2
    import numpy as np

    from app.collection import clock

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("打不开视频")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    cells = []
    for t in args.times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"  t={t}: 取不到帧")
            continue
        gs = clock.read_clock(frame)
        h, w = frame.shape[:2]
        nw = args.cell_width
        nh = int(h * nw / w)
        small = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        label = f"t={t:.0f}s" + (f"  {gs // 60:02d}:{gs % 60:02d}" if gs is not None else "  --:--")
        cv2.rectangle(small, (0, 0), (nw, 34), (0, 0, 0), -1)
        cv2.putText(small, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cells.append(small)
        print(f"  t={t:.0f}s -> {label}")

    cap.release()
    if not cells:
        print("没有取到任何帧")
        return 1

    ch, cw = cells[0].shape[:2]
    cols = max(1, min(args.cols, len(cells)))
    rows = (len(cells) + cols - 1) // cols
    sheet = np.zeros((rows * ch, cols * cw, 3), dtype=np.uint8)
    for i, c in enumerate(cells):
        r, cc = divmod(i, cols)
        sheet[r * ch:(r + 1) * ch, cc * cw:(cc + 1) * cw] = c

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"\n已写入 {args.out}  ({sheet.shape[1]}x{sheet.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
