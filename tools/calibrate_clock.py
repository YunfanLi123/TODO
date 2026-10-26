"""从清晰录屏里标定**时钟数字模板** —— 设备无关的做法。

跟老工具（``tools/build_clock_templates.py``）的区别
------------------------------------------------
老工具把四个数字位写成**绝对像素**（M1 1819-1839…），那是某台设备的几何，
换设备就得重写常量。

这里改成**按画布高度归一**（跟运行时同一套：``clock.WORK_H = 1200``）：
  - 数字组在归一空间里的**横向位置自动找**（顶条亮像素的列分布里有 4 个峰）；
  - 组内四个数字位的**相对偏移固定**（``clock.GROUP_SLOTS``，与设备无关）；
  - 标签由「一个已知读数」推出来（锚点），不做无监督聚类。

用法::

    python tools/calibrate_clock.py <录屏路径> --at 220 --shows 0013
    #  --at   视频第几秒
    #  --shows 那一刻屏幕上显示的 MMSS
    #  --from/--to 扫描区间（默认整段）
    #  --out  输出文件（默认 app/collection/clock_digits.json）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.collection import clock as ck  # noqa: E402
from app.collection import video_upload as vu  # noqa: E402

HUD = {"minimap": "上左", "skills": "下右", "joystick": "下左"}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(ROOT, "app", "collection", "clock_digits.json")

#: 每个数字至少要这么多样本才建模板 —— 三五张图平均出来的「模板」认不准，
#: 而认错比不认更糟。
MIN_SAMPLES_PER_DIGIT = 6


def normalized_canvas(path: str, rot: int, band, t: float, fps: float):
    """取某一时刻的画布，并按高度归一到标定几何。"""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    canvas = vu.crop_to_band(vu.orient_frame(cv2, frame, rot), band)
    h, w = canvas.shape[:2]
    work = cv2.resize(canvas, (max(8, int(w * ck.WORK_H / h)), ck.WORK_H),
                      interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)


def find_group_x(gray: np.ndarray) -> Optional[int]:
    """在归一空间里找数字组的横向起点。

    找**结构**，不找"最亮的那一撮"
    --------------------------------
    早先的实现是"顶条里最靠右的窄亮峰" —— 在**加载画面**上跑，
    顶条里根本没有时钟，找到的是别的东西，于是模板是从垃圾内容建的
    （而且因为标签是按时间推的，"每一位都覆盖了 0~9" 看着完全正常，
    根本看不出建错了 —— 这个坑很隐蔽）。

    现在改成**结构搜索**：滑过整个顶条，要求
      - 四个数字位**每一位都有墨迹**（各占槽位 4% 以上）；
      - 没有哪一位是**整块亮**的（那是别的东西，比如血条）；
    取总墨迹最多的那个位置。
    """
    h, w = gray.shape[:2]
    strip = gray[ck.SLOT_Y:ck.SLOT_Y + ck.SLOT_H, :]
    if strip.size == 0 or w < ck.GROUP_W:
        return None
    thr = max(130, int(np.percentile(strip, 99)))
    ink = (strip > thr)
    col = ink.sum(axis=0)
    best: Optional[Tuple[float, int]] = None
    for x0 in range(0, w - ck.GROUP_W, 2):
        inks = []
        for dx, dw in ck.GROUP_SLOTS:
            patch = ink[:, x0 + dx:x0 + dx + dw]
            inks.append(float(patch.mean()) if patch.size else 0.0)
        if min(inks) < 0.04:          # 有一位没墨 → 不是数字组
            continue
        if max(inks) > 0.70:          # 整块亮 → 是别的 UI
            continue
        # 四个位置要**都**有墨，且不能太不均（数字组的四个位墨迹量相近）
        score = sum(inks) - 0.5 * (max(inks) - min(inks))
        if best is None or score > best[0]:
            best = (score, x0)
    if best is None:
        return None
    # 顶栏里时钟在中偏右；但也可能整条顶条只有一个数字组 —— 取**最右**
    # 的高分位置，避免把左边别的东西当成时钟。
    cand = []
    for x0 in range(0, w - ck.GROUP_W, 2):
        inks = []
        for dx, dw in ck.GROUP_SLOTS:
            patch = ink[:, x0 + dx:x0 + dx + dw]
            inks.append(float(patch.mean()) if patch.size else 0.0)
        if min(inks) < 0.04 or max(inks) > 0.70:
            continue
        cand.append((sum(inks) - 0.5 * (max(inks) - min(inks)), x0))
    if not cand:
        return None
    cand.sort(reverse=True)
    top = cand[0][0]
    good = [x0 for sc, x0 in cand if sc >= top * 0.9]
    return int(max(good))


def collect(path: str, rot: int, band, fps: float, t_from: float, t_to: float,
            step: float, group_x: Optional[int]):
    """采样每一帧的四个数字位。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"打不开视频：{path}")
    patches: List[np.ndarray] = []
    times: List[float] = []
    slot_idx: List[int] = []
    t = t_from
    gx = group_x
    while t <= t_to:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            break
        canvas = vu.crop_to_band(vu.orient_frame(cv2, frame, rot), band)
        ch, cw = canvas.shape[:2]
        work = cv2.resize(canvas, (max(8, int(cw * ck.WORK_H / ch)), ck.WORK_H),
                          interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        if gx is None:
            gx = find_group_x(gray)
            if gx is None:
                t += step
                continue
        for i, (dx, dw) in enumerate(ck.GROUP_SLOTS):
            x0 = gx + dx
            y0 = ck.SLOT_Y
            if x0 < 0 or y0 < 0 or x0 + dw > gray.shape[1] \
                    or y0 + ck.SLOT_H > gray.shape[0]:
                continue
            p = gray[y0:y0 + ck.SLOT_H, x0:x0 + dw].astype(np.float32)
            p = p - p.mean()
            s = p.std()
            p = (p / s if s > 1e-3 else p)
            patches.append(p.flatten())
            times.append(t)
            slot_idx.append(i)
        t += step
    cap.release()
    return patches, np.array(times), np.array(slot_idx), gx


def build(patches, times, slot_idx, start_t: float):
    """按标签求均值，得到每个数字的模板。"""
    acc: Dict[Tuple[int, int], List[np.ndarray]] = {}
    for p, t, i in zip(patches, times, slot_idx):
        g = int(round(float(t) - start_t))
        if g < 0:
            continue
        mm, ss = divmod(g, 60)
        if mm > 99 or ss > 59:
            continue
        lab = [mm // 10, mm % 10, ss // 10, ss % 10][int(i)]
        acc.setdefault((int(i), lab), []).append(p)
    slots: List[Dict] = []
    for i, (dx, dw) in enumerate(ck.GROUP_SLOTS):
        labels: List[int] = []
        rows: List[List[float]] = []
        for lab in range(10):
            group = acc.get((i, lab))
            if not group or len(group) < MIN_SAMPLES_PER_DIGIT:
                continue
            mean = np.mean(np.array(group), axis=0)
            labels.append(lab)
            rows.append([round(float(v), 5) for v in mean])
        if not labels:
            print(f"  位 {i}：样本不够，不建模板")
            continue
        slots.append({
            "name": ["M1", "M2", "S1", "S2"][i],
            "box": [0.0, 0.0, 1.0, 1.0],   # 运行时的锚定路径不用这个
            # **形状是 [高, 宽]** —— 运行时 ``sh, sw = slot["shape"]``
            # 然后 ``cv2.resize(patch, (sw, sh))``。
            "shape": [ck.SLOT_H, dw],
            "labels": labels,
            "templates": rows,
        })
        print(f"  位 {i}：认出 {len(labels)} 个数字（{labels}）")
    return slots


def main() -> int:
    ap = argparse.ArgumentParser(description="标定时钟数字模板（设备无关）")
    ap.add_argument("video")
    ap.add_argument("--at", type=float, required=True,
                    help="锚点：视频第几秒")
    ap.add_argument("--shows", required=True,
                    help="锚点那一刻屏幕上的 MMSS，例如 0013")
    ap.add_argument("--from", dest="t_from", type=float, default=None)
    ap.add_argument("--to", dest="t_to", type=float, default=None)
    ap.add_argument("--step", type=float, default=1.5)
    ap.add_argument("--group-x", type=int, default=None,
                    help="数字组在归一空间里的起点（不给就自动找）")
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args()

    shows = args.shows.strip()
    if len(shows) != 4 or not shows.isdigit():
        print("--shows 要四位 MMSS，例如 0013")
        return 1
    reading = int(shows[:2]) * 60 + int(shows[2:])
    start_t = args.at - reading

    video = vu.UploadedVideo(video_id="cal", original_name="c.mp4",
                             path=args.video, size=1)
    video = vu.probe(video)
    view = vu.video_view(video, hud_layout=HUD)
    rot = int(view["rotation"])
    band = view.get("band")
    fps = video.fps or 30.0
    print(f"方向 {rot}°  黑边 {band}  锚点 t={args.at:.0f}s 读数 {shows}"
          f" → 对局起点 {start_t:.0f}s")

    t_from = args.t_from if args.t_from is not None else start_t + 2
    t_to = args.t_to if args.t_to is not None else start_t + 900
    patches, times, slot_idx, gx = collect(
        args.video, rot, band, fps, t_from, t_to, args.step, args.group_x)
    if not patches:
        print("没采到样本")
        return 1
    print(f"采到 {len(patches)} 个字位样本，数字组起点（归一空间）= {gx}")
    slots = build(patches, times, slot_idx, start_t)
    if len(slots) < 4:
        print("四个位没建齐，不写文件")
        return 1

    data = {
        # **size 是 [宽, 高]** —— 运行时的 ``load_digits`` 按
        # ``pw, ph = data["size"]`` 读，形状再取 ``(ph, pw)``。
        # 写反了会把补丁缩成另一个比例，匹配全是噪声。
        "size": [ck.GROUP_SLOTS[0][1], ck.SLOT_H],
        "slots": slots,
        "group_x": gx, "start_t": round(start_t, 1),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"已写 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
