"""在真实录屏上验证时钟读数 —— 只做「已知时刻对答案」这一步。

对答案的依据是**事先看图确认过的读数**：t=220→00:21、t=320→02:01、
t=420→03:41、t=520→05:21、t=700→08:21，与模板无关，
所以能独立判模板对不对。

全程扫描另由 tools/scan_clock_on_video.py 做 —— 那个要顺序解完整个视频，
慢，不该和这一步混在一起。

用法::

    .venv\\Scripts\\python.exe tools\\verify_clock_on_video.py <录屏>
"""

from __future__ import annotations

import os
import sys
from typing import Dict

import cv2

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: 事先看图确认的读数（与模板无关）
KNOWN: Dict[int, str] = {220: "00:21", 320: "02:01", 420: "03:41",
                         520: "05:21", 700: "08:21"}


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("用法：python tools/verify_clock_on_video.py <录屏>")
    src = sys.argv[1]

    from app.collection import clock as C

    slots = C.load_digits()
    if not slots:
        raise SystemExit("模板没加载成功（缺文件或格式不对）")
    print("模板加载：")
    for s in slots:
        print(f"  {s['name']:3s} 覆盖数字 {s['labels']}")
    print(flush=True)

    cap = cv2.VideoCapture(src)
    ok_all = True
    print("=== 已知时刻对答案 ===", flush=True)
    for t, truth in KNOWN.items():
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            print(f"  t={t:4d}  读帧失败", flush=True)
            ok_all = False
            continue
        gs, conf = C.read_clock_frame(fr)
        got = f"{gs//60:02d}:{gs%60:02d}" if gs is not None else "  --  "
        hit = "✓" if got == truth else "✗"
        if got != truth:
            ok_all = False
        print(f"  t={t:4d}  期望 {truth}  实读 {got}  置信 {conf:.3f}  {hit}",
              flush=True)
    cap.release()

    print("\n结论：" + ("全部已知时刻读对" if ok_all else "有读错的时刻"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
