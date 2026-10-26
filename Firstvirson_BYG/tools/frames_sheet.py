"""把扫描抽出的帧拼成联系表，用来看每个阶段画面上实际在发生什么。

用途
----
标注要填「这一帧做了什么」。凭空猜会让报告的结论建立在编造的数据上，
而报告恰恰是要告诉玩家「你哪里做错了」—— 猜出来的动作会让它指错方向。

所以先把帧看一遍：野区就是在野区、龙坑就是在龙坑、路上了就在路上。
看得出来的就是事实，看不出来的就留空。

用法::

    .\\.venv\\Scripts\\python.exe tools\\frames_sheet.py "<帧目录>" [--out .tmp\\sheet.jpg]
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
    ap = argparse.ArgumentParser(description="抽帧联系表")
    ap.add_argument("frames_dir")
    ap.add_argument("--out", default=".tmp/frames_sheet.jpg")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--cell-width", type=int, default=460)
    ap.add_argument("--prefix", default="", help="只拼这个前缀的文件")
    args = ap.parse_args(argv)

    import cv2
    import numpy as np

    names = sorted(
        f for f in os.listdir(args.frames_dir)
        if f.lower().endswith(".jpg")
        and (not args.prefix or f.startswith(args.prefix))
    )
    if not names:
        print(f"{args.frames_dir} 里没有 jpg")
        return 1

    cells = []
    for name in names:
        img = cv2.imread(os.path.join(args.frames_dir, name))
        if img is None:
            continue
        h, w = img.shape[:2]
        nw = args.cell_width
        nh = int(h * nw / w)
        small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        cv2.rectangle(small, (0, 0), (nw, 30), (0, 0, 0), -1)
        cv2.putText(small, name, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(small)
        print(f"  {name}")

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
