"""用真实录屏跑通「上传 → 扫描 → 报告」整条链路。

这不是单元测试，是**端到端实测**：真的把 1.7 GB 的录屏通过 HTTP 传上去，
等后台扫完，把扫描产物打印出来。

为什么要有它
------------
采集层的变化（顺序解码、BP 帧序列、按阶段抽帧）只有接在真实数据上
才验得出来。单帧合成的视频读不出时钟，也就没有阶段可分 ——
而「阶段分得对不对」恰恰是这次改动的核心。

流程里**没有标注页**了：分路、胜负、BP、每段操作都由模型从画面读，
扫完直接出报告（``POST /api/videos/finalize``）。

用法::

    .\\.venv\\Scripts\\python.exe tools\\verify_real_video_flow.py "<录屏>" [--base http://127.0.0.1:8849]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

sys.stdout.reconfigure(encoding="utf-8")


def _fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="真实录屏端到端验证")
    ap.add_argument("video")
    ap.add_argument("--base", default="http://127.0.0.1:8849")
    ap.add_argument("--game", default="hok")
    ap.add_argument("--timeout", type=float, default=3600.0)
    args = ap.parse_args(argv)

    import httpx

    base = args.base.rstrip("/")
    size = os.path.getsize(args.video)
    print(f"录屏：{os.path.basename(args.video)}  {_fmt_size(size)}")

    t0 = time.perf_counter()
    with open(args.video, "rb") as fh:
        with httpx.Client(timeout=httpx.Timeout(args.timeout, connect=30.0)) as client:
            resp = client.post(
                f"{base}/api/videos/upload",
                files={"file": (os.path.basename(args.video), fh,
                                "video/mp4")},
                data={"game": args.game},
            )
    if resp.status_code != 200:
        print(f"上传失败 {resp.status_code}: {resp.text[:400]}")
        return 1

    up = resp.json()
    print(f"上传完成 用时 {time.perf_counter() - t0:.1f}s")
    print(f"  job_id   {up['job_id']}")
    print(f"  match_id {up['match_id']}")
    print(f"  视频     {up['video']['resolution']} {up['video']['fps']}fps "
          f"{up['video']['duration_min']} 分钟")

    # -- 轮询扫描进度 --------------------------------------------------
    last_stage = ""
    while True:
        time.sleep(3.0)
        with httpx.Client(timeout=60.0) as client:
            st = client.get(f"{base}/api/videos/jobs/{up['job_id']}").json()
        stage = f"{st['status']} {st['stage_text']}"
        if stage != last_stage:
            print(f"  [{time.perf_counter() - t0:7.1f}s] {stage} "
                  f"{st['progress'] * 100:.0f}%")
            last_stage = stage
        if st["status"] in ("done", "failed"):
            break
        if time.perf_counter() - t0 > args.timeout:
            print("超时")
            return 1

    if st["status"] == "failed":
        print(f"扫描失败：{st['error']}")
        return 1

    summary: Dict[str, Any] = st["summary"]
    print(f"\n扫描完成 总用时 {time.perf_counter() - t0:.1f}s")
    print(f"  产物类型   {summary.get('product')}")
    print(f"  扫描说明   {summary['scan'].get('note')}")
    clock = summary["scan"].get("clock") or {}
    print(f"  时钟       ok={clock.get('ok')} 起点={clock.get('game_start_t')}s "
          f"斜率={clock.get('slope')} 残差={clock.get('residual')}s "
          f"读到 {clock.get('samples_read')}/{clock.get('samples_total')}")
    if summary.get("match_minutes"):
        print(f"  对局       {summary['match_start_t']}s → {summary['match_end_t']}s "
              f"（{summary['match_minutes']} 分钟）")
    print(f"  阶段抽帧   {summary['round_count']} 张   BP 帧 {summary['bp_count']} 张")

    print("\n抽帧落点：")
    for r in summary["rounds"]:
        print(f"  #{r['round_no']:>2}  {r.get('phase','?'):<6} "
              f"第 {r.get('game_minute')} 分钟   视频 {r['timestamp']}s   "
              f"{r['image']}")

    print("\nBP 帧落点：")
    for f in summary.get("bp") or []:
        print(f"  {f['label']:<6} 视频 {f['timestamp']}s   {f['image']}")

    # -- 建对局 → 出报告（中间没有人工那一步） -------------------------
    print("\n建对局：")
    with httpx.Client(timeout=httpx.Timeout(600.0)) as client:
        fin = client.post(f"{base}/api/videos/finalize",
                          json={"match_id": up["match_id"]})
    if fin.status_code != 200:
        print(f"建对局失败 {fin.status_code}: {fin.text[:400]}")
        return 1
    data = fin.json()
    print(f"  match_id   {data['match_id']}")
    print(f"  结论条数   {data.get('gap_count')}")
    print(f"  报告       {base}{data['report_url']}")

    prefill = summary.get("prefill") or {}
    print(f"  模型读出   分路={prefill.get('position') or '—'} "
          f"胜负={prefill.get('result') or '—'} 英雄={prefill.get('hero') or '—'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
