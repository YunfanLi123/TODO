"""批量体检真实录屏：方向、黑边、对局段、时钟、切片分布。

用法::

    python tools/verify_recordings.py <录屏1.mp4> [录屏2.mp4 ...]
    python tools/verify_recordings.py --dir "<本机目录>"

它跑的是**采集层的真实链路**（跟上传后跑的同一条），把每段录屏的关键
数字打成一栏表：

    录屏          方向  画布      对局段(视频秒)      时钟   切片  切片分钟分布

看这张表是为了回答三个问题：

1. **方向判对了吗** —— 画布尺寸和朝向对不对（错 = 截图全倒着看）；
2. **对局段找准了吗** —— 录屏前后有加载、BP、结算，还有可能混进别的
   应用画面（噪声）；切错 = 整局的时间轴都偏；
3. **切片落在哪** —— 每分钟切了几片、规则密度高的地方是不是切得多。

诊断本身也写进 ``.tmp/recordings_probe/``，可以直接看图核对。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.collection import clock as ck              # noqa: E402
from app.collection import video_upload as vu        # noqa: E402
from app.rules.density import density_weights, seed_density  # noqa: E402

OUT_DIR = ROOT / ".tmp" / "recordings_probe"

HUD_LAYOUT = {"minimap": "上左", "skills": "下右", "joystick": "下左"}


def probe_one(path: str) -> Dict[str, Any]:
    """跑一遍采集链路的关键环节，返回一张诊断表。"""
    import cv2
    import numpy as np

    p = Path(path)
    size_mb = p.stat().st_size / 1024 ** 2
    video = vu.UploadedVideo(video_id=p.stem[:12], original_name=p.name,
                             path=str(p), size=p.stat().st_size)
    t0 = time.time()
    video = vu.probe(video)
    view = vu.video_view(video, hud_layout=HUD_LAYOUT)
    rot = int(view["rotation"])
    band = view.get("band")

    # 画布尺寸（转正 + 裁黑边之后）
    cap, _ = vu.open_capture(video.path)
    canvas = None
    if cap.isOpened():
        ok, frame = cap.read()
        if ok:
            canvas = vu.crop_to_band(vu.orient_frame(cv2, frame, rot), band)
        cap.release()

    # 密集扫描 → 变化序列
    cfg = vu.ScanConfig()
    cfg.sample_sec = vu.adapt_sample_sec(video.duration_sec)
    points, stats = vu.activity_scan(video, cfg)
    span = vu.activity_span(points) if points else None

    # 时钟能不能读
    clock_ok = bool((stats.get("clock") or {}).get("ok"))

    # 对局起点（逐段视频单独检测）：对局中才有局内时间，
    # 不同录屏进入对局的时刻不同，所以不能写死。
    match_start_t = stats.get("match_start_t")
    match_start_reason = stats.get("match_start_reason", "")

    # 切片
    weights = density_weights(seed_density("hok"))
    product = None
    if points:
        product = vu.moba_slice_product(
            video, points, str(OUT_DIR), {"early": [0, 4], "mid": [5, 10],
                                          "late": [11, 60]},
            budget=32, hud_layout=HUD_LAYOUT, weights=weights,
            game_start_t=match_start_t or stats.get("hud_start_t"),
        )

    per_minute: Dict[int, int] = {}
    if product:
        for r in product["rounds"]:
            m = int(r["game_minute"] or 0)
            per_minute[m] = per_minute.get(m, 0) + 1

    # 存一张画布图，方便肉眼核对方向对不对
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shot = ""
    if canvas is not None:
        name = f"{p.stem[:12]}_canvas.jpg"
        cv2.imwrite(str(OUT_DIR / name), canvas)
        shot = name

    return {
        "name": p.name,
        "size_mb": round(size_mb, 1),
        "duration": round(video.duration_sec, 1),
        "raw": f"{video.width}x{video.height}",
        "rotation": rot,
        "canvas": f"{canvas.shape[1]}x{canvas.shape[0]}" if canvas is not None else "-",
        "band": band,
        "span": [round(span[0], 1), round(span[1], 1)] if span else None,
        "span_min": round((span[1] - span[0]) / 60.0, 1) if span else None,
        "clock_ok": clock_ok,
        "clock_note": (stats.get("clock") or {}).get("reason", ""),
        "slices": len(product["rounds"]) if product else 0,
        "per_minute": per_minute,
        "slice_minutes": product.get("slice_minutes") if product else [],
        "match_start_t": match_start_t,
        "match_start_reason": match_start_reason,
        "diff_median": stats.get("diff_median"),
        "shot": shot,
        "elapsed": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", help="录屏文件")
    ap.add_argument("--dir", help="目录（里面的 .mp4 全跑）")
    args = ap.parse_args()

    paths: List[str] = list(args.videos)
    if args.dir:
        d = Path(args.dir)
        paths += [str(p) for p in sorted(d.glob("*.mp4"))]
    paths = [p for p in paths if Path(p).exists()]
    if not paths:
        print("没有可跑的录屏")
        return 1

    results = []
    for path in paths:
        print(f"\n=== {Path(path).name}")
        try:
            r = probe_one(path)
        except Exception as exc:                      # noqa: BLE001
            print(f"  !! {type(exc).__name__}: {exc}")
            continue
        results.append(r)
        print(f"  画布 {r['canvas']}（原帧 {r['raw']}，方向 {r['rotation']}°）"
              f"  黑边 {r['band']}")
        print(f"  对局段 {r['span']}（{r['span_min']} 分钟）"
              f"  时钟 {'可读' if r['clock_ok'] else '读不出'}")
        if not r["clock_ok"]:
            print(f"       {r['clock_note']}")
        if r["match_start_t"] is not None:
            print(f"  对局起点：视频第 {r['match_start_t']} 秒"
                  f"（{r['match_start_reason']}）")
        print(f"  切片 {r['slices']} 片，每分钟：{r['per_minute']}")
        print(f"  用时 {r['elapsed']}s")

    print("\n" + "=" * 78)
    print(f"{'录屏':<24}{'方向':>5}{'画布':>11}{'对局段(分)':>12}"
          f"{'对局起点':>10}{'时钟':>7}{'切片':>6}")
    for r in results:
        span = f"{r['span_min']}分" if r["span_min"] else "-"
        mst = f"{r['match_start_t']}s" if r["match_start_t"] is not None else "-"
        print(f"{r['name'][:22]:<24}{r['rotation']:>5}{r['canvas']:>11}"
              f"{span:>12}{mst:>10}{'可读' if r['clock_ok'] else '无':>7}{r['slices']:>6}")

    out = ROOT / ".tmp" / "recordings_report.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n详细结果：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
