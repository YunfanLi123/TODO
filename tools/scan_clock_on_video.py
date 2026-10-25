"""在真实录屏上全程读时钟，检查时间线与阶段落点。

与 verify_clock_on_video.py 的分工：
  verify —— 只 seek 五个已知时刻，快速判模板对不对
  scan   —— 顺序解完整个视频，看读出率、拟合结果与阶段落点

顺序解码 2670x1200@90fps 很慢（十几分钟），所以这一步单独跑。

用法::

    .venv\\Scripts\\python.exe tools\\scan_clock_on_video.py <录屏> [采样秒]
"""

from __future__ import annotations

import os
import sys

import cv2

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("用法：python tools/scan_clock_on_video.py <录屏> [采样秒]")
    src = sys.argv[1]
    every = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

    from app.collection import clock as C

    slots = C.load_digits()
    if not slots:
        raise SystemExit("模板没加载成功")
    print("模板：" + "  ".join(f"{s['name']}={s['labels']}" for s in slots), flush=True)

    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(every * fps)))

    samples = []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            gs, conf = C.read_clock_frame(fr)
            samples.append(C.ClockSample(t=t, game_s=gs, confidence=conf))
        i += 1
        del fr
    cap.release()

    read = [s for s in samples if s.game_s is not None]
    print(f"\n采样 {len(samples)} 次 (每 {every}s)，读出 {len(read)} 次 "
          f"({len(read)/max(1,len(samples))*100:.1f}%)", flush=True)

    # 读不出来的集中在哪儿？那才是问题所在
    miss = [s.t for s in samples if s.game_s is None]
    if miss:
        runs, start = [], miss[0]
        for a, b in zip(miss, miss[1:] + [None]):
            if b is None or b - a > every * 2.5:
                runs.append((start, a))
                if b is not None:
                    start = b
        print("读不出来的区段：")
        for a, b in runs[:20]:
            print(f"  视频 {a:7.1f}s - {b:7.1f}s")

    track = C.build_clock_track(samples)
    print(f"\nok={track.ok}  斜率={track.slope}  残差={track.residual}")
    print(f"{track.reason}")

    if track.ok:
        print(f"\n对局起点：视频第 {track.game_start_t:.1f} 秒")
        print("\n阶段落点抽查（early <5 / mid 5-10 / late >=11）：")
        vt = max(0.0, track.game_start_t)
        while vt < 960:
            m = track.minutes_at(vt)
            ph = ("early" if m < 5 else "mid" if m < 11 else "late")
            print(f"  视频 {vt:7.1f}s -> 游戏 {m:6.2f} 分 ({ph})", flush=True)
            vt += 60
    return 0 if track.ok else 1


if __name__ == "__main__":
    sys.exit(main())
