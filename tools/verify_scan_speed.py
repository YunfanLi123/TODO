"""验证扫描提速：按时间采样 vs 原来的按帧采样。

造一段「长视频」（低分辨率、长时间，成本可控），量两件事：

  1. 采样量降了多少（这是耗时下降的根本原因）
  2. 回合边界有没有漏（提速不能以漏边界为代价）

第 2 条是关键 —— 一个把采样率降到 1/10 却切不出回合的实现，
提速再多也没有意义。所以拿「按帧」的结果当基准来对比。
"""

from __future__ import annotations

import io
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np                                     # noqa: E402
import cv2                                             # noqa: E402

from app.collection import video_upload as vu          # noqa: E402

FAILURES = []


def check(cond, msg):
    print(("  [PASS] " if cond else "  [FAIL] ") + msg)
    if not cond:
        FAILURES.append(msg)


def make_long_video(path: str, seconds: float = 240.0,
                    fps: float = 30.0, round_sec: float = 12.0,
                    size=(320, 180)) -> int:
    """造一段长视频：每个回合内轻微抖动，回合切换时整体压暗。

    尺寸和时长都调到能快速跑完，但帧数与真实长视频同量级。
    """
    W, H = size
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not vw.isOpened():
        raise RuntimeError("无法创建视频")

    n = int(seconds * fps)
    rnd = 0
    for i in range(n):
        t = i / fps
        rnd = int(t / round_sec)
        base = 45 + rnd * 5
        frame = np.full((H, W, 3), base, dtype=np.uint8)
        # 回合内的轻微动画
        frame[:, :, 1] = (base + int(5 * np.sin(i / 3.0))) % 255
        # 每隔一段时间画个方块当"棋子"
        cx = int(W / 2 + 40 * np.sin(i / 7.0))
        cv2.circle(frame, (cx, H // 2), 12, (200, 180, 90), -1)
        # 回合切换：整体压暗到接近黑
        into = (t % round_sec) / round_sec
        if into < 0.05:
            frame = (frame * 0.25).astype(np.uint8)
        vw.write(frame)
    vw.release()
    return n


def scan(video: vu.UploadedVideo, cfg: vu.ScanConfig) -> tuple:
    t0 = time.perf_counter()
    frames, info = vu.gate_round_boundaries(video, cfg)
    return time.perf_counter() - t0, frames, info


def main() -> int:
    vid = os.path.join(ROOT, "data", "sample", "long.mp4")
    os.makedirs(os.path.dirname(vid), exist_ok=True)
    if os.path.exists(vid):
        os.remove(vid)

    print("\n[1] 造一段 4 分钟 / 30fps 的视频")
    n_frames = make_long_video(vid)
    size_mb = os.path.getsize(vid) / 1024 ** 2
    print(f"      {n_frames} 帧 · {size_mb:.1f} MB")
    check(n_frames == 7200, f"帧数 {n_frames}")

    up = vu.UploadedVideo(
        video_id="longtest", path=vid, original_name="long.mp4",
        size=os.path.getsize(vid),
    )
    up = vu.probe(up)
    print(f"      时长 {up.duration_sec:.0f}s · {up.fps:.0f}fps · "
          f"{up.width}x{up.height}")

    print("\n[2] 按帧采样（旧做法，每 3 帧一次 = 每秒 10 次）")
    old_cfg = vu.ScanConfig()
    old_cfg.sample_sec = 3.0 / up.fps          # 0.1s/次
    t_old, f_old, i_old = scan(up, old_cfg)
    print(f"      采样 {i_old['scanned_samples']} 次 · "
          f"用时 {t_old:.2f}s · 切出 {len(f_old)} 个边界")

    print("\n[3] 按时间采样（新做法，时长自适应）")
    auto_sec = vu.adapt_sample_sec(up.duration_sec)
    print(f"      自适应采样间隔 {auto_sec:.2f}s")
    new_cfg = vu.ScanConfig()
    t_new, f_new, i_new = scan(up, new_cfg)
    print(f"      采样 {i_new['scanned_samples']} 次 · "
          f"用时 {t_new:.2f}s · 切出 {len(f_new)} 个边界")

    print("\n[4] 提速与漏检")
    speedup = t_old / t_new if t_new else 0
    print(f"      提速 {speedup:.1f}x　采样量减少到 "
          f"{i_new['scanned_samples'] / max(i_old['scanned_samples'], 1):.1%}")
    check(speedup > 1.5, f"扫描提速 {speedup:.1f}x")
    check(i_new["scanned_samples"] < i_old["scanned_samples"],
          "采样次数确实减少了")

    # 真值：4 分钟 / 每回合 12 秒 = 20 个回合（+1 个首帧边界）
    expected = int(up.duration_sec // 12) + 1
    print(f"      真值约 {expected} 个回合")
    check(abs(len(f_new) - expected) <= 2,
          f"边界数 {len(f_new)} 接近真值 {expected}")
    check(abs(len(f_new) - len(f_old)) <= 2,
          f"新旧采样切出的边界数接近（{len(f_old)} vs {len(f_new)}）"
          f" —— 没因为提速而漏检")

    print("\n[5] 进度回调是真实的")
    seen = []

    def cb(frac, stage):
        seen.append(frac)

    vu.gate_round_boundaries(up, vu.ScanConfig(), progress=cb)
    check(len(seen) > 5, f"进度回调触发 {len(seen)} 次")
    check(seen == sorted(seen), "进度单调递增")
    check(seen[-1] > 0.9, f"最终进度 {seen[-1]:.2f}")

    print("\n[6] 自适应采样表")
    for d in (120, 300, 600, 1800, 2700, 5400):
        print(f"      {d // 60:>2} 分钟 -> 每 {vu.adapt_sample_sec(d):.2f}s 采一次"
              f"（{d / vu.adapt_sample_sec(d):.0f} 次）")

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"失败 {len(FAILURES)} 项：")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("扫描提速验证通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
