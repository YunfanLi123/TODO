"""直接对本地视频跑一遍采集层，产出扫描产物（不上传、不入库）。

为什么要有它
------------
验证「按阶段抽帧」的落点需要反复调：抽帧规则一改就要重看落点对不对。
而走 HTTP 上传意味着每次都要传 1.7 GB（十几秒）再等十分钟扫描。
这个脚本绕开上传与入库，只跑**采集层本身**，出结果快得多。

用法::

    .\\.venv\\Scripts\\python.exe tools\\scan_video_product.py "<录屏>" [--out .tmp\\frames]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

sys.stdout.reconfigure(encoding="utf-8")


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="本地扫描：产出采集层产物")
    ap.add_argument("video")
    ap.add_argument("--out", default=".tmp/frames",
                    help="抽帧存放目录（相对项目根）")
    ap.add_argument("--game", default="hok")
    ap.add_argument("--skip-scan", action="store_true",
                    help="跳过扫描阶段，只跑抽帧（时钟读数从 --clock 传入）")
    ap.add_argument("--clock", default="",
                    help="跳过扫描时使用的时钟信息 JSON 文件")
    args = ap.parse_args(argv)

    from app.adapters import registry as _registry  # noqa: F401  触发适配器注册
    from app.adapters.base import get_adapter
    from app.collection import video_upload as vu

    adapter = get_adapter(args.game)
    video = vu.UploadedVideo(
        video_id="local",
        path=os.path.abspath(args.video),
        original_name=os.path.basename(args.video),
        size=os.path.getsize(args.video),
    )
    video = vu.probe(video)
    print(f"视频 {video.width}x{video.height} {video.fps:.2f}fps "
          f"{video.duration_sec / 60:.1f} 分钟")

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    clock_info = {}
    if args.clock:
        import json

        clock_info = json.loads(open(args.clock, encoding="utf-8").read())

    if not args.skip_scan:
        t0 = time.perf_counter()
        frames, scan = vu.gate_round_boundaries(video, vu.ScanConfig())
        print(f"\n扫描 {time.perf_counter() - t0:.1f}s")
        print(f"  采样 {scan['scanned_samples']} 次  边界 {scan['boundary_count']} 个")
        print(f"  {scan['note']}")
        clock_info = scan.get("clock") or {}

    ck = clock_info or {}
    print(f"\n时钟 ok={ck.get('ok')} 起点={ck.get('game_start_t')}s "
          f"斜率={ck.get('slope')} 残差={ck.get('residual')}s "
          f"读到 {ck.get('samples_read')}/{ck.get('samples_total')}")

    if not ck.get("ok"):
        print("读不到时钟，按阶段抽帧这条路走不通")
        return 1

    t0 = time.perf_counter()
    result = vu.sample_by_phase(
        video, ck, out_dir, dict(adapter.phase_time_windows),
    )
    if result is None:
        print("按阶段抽帧失败")
        return 1
    print(f"\n抽帧 {time.perf_counter() - t0:.1f}s  "
          f"对局 {result['match_start_t']}s → {result['match_end_t']}s "
          f"（{result['match_minutes']} 分钟）")

    print("\n阶段抽帧：")
    for r in result["rounds"]:
        print(f"  #{r['round_no']:>2}  {r['phase']:<6} "
              f"第 {r['game_minute']} 分钟  视频 {r['timestamp']}s  {r['image']}")

    print("\nBP 帧序列：")
    for f in result["bp"]:
        print(f"  {f['label']:<6} 视频 {f['timestamp']}s  {f['image']}")

    # 落点是否压在阶段边界上 —— 压上去就会一帧算进两个阶段
    print("\n边界检查：")
    bad = 0
    for a, b in zip(result["rounds"], result["rounds"][1:]):
        if a["phase"] == b["phase"]:
            continue
        gap = b["game_minute"] - a["game_minute"]
        print(f"  {a['phase']} → {b['phase']}  相隔 {gap:.2f} 分钟"
              f"{'  ← 两帧挤在同一分钟' if gap < 0.6 else ''}")
        if gap < 0.6:
            bad += 1
    print("  没有边界重复" if not bad else f"  有 {bad} 处边界重复")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
