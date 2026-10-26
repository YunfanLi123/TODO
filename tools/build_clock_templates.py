"""从真实对局录屏里学出时钟数字模板，产出 ``app/collection/clock_digits.json``。

为什么需要它
------------
原方案是「数秒跳」：时钟每秒跳一次，数跳变次数就等于经过的游戏秒数。
拿真实录屏一试就走不通了 —— 时钟数字压在**会动的游戏画面**上，
草地中灰、河道亮蓝、后期地形亮红橙，背景明暗跨度大到：

  * 整块灰度均值当指纹 → 背景一动就误判成「跳变」，间隔只有 0.03 秒
  * 二值化 + 亮度阈值 → 有的帧整块找不到一个亮像素，分段结果每帧都不一样
  * 先缩放后二值化 → 笔画被 INTER_AREA 平均掉，指纹几乎不随秒跳变化

数字本身却是又亮又稳的：白字、固定位置、固定字体。
所以改用**归一化互相关**匹配 —— 它对局部亮度和对比度都不敏感。

模板怎么来的（关键：不是聚类，是监督）
--------------------------------------
无监督聚类试过，很难干净：同一个「0」会因为在四个位置上落点差半像素
被拆成三四簇，凑不出 10 类，还混进「画面被遮挡」产生的糊簇。

改用锚点法：**先有一个已知的读数**，对局起点就定了，于是每一帧
每个数字位的标签都是确定的，直接按标签求均值就是模板 ——
没有簇数要凑，没有阈值要调。

唯一的人工输入是**一个**锚点（本项目默认 t=220s → 00:21），
而且它会被自动校验：对起点做小范围扫描，取「类内最紧」的那个。
起点错一秒就会让标签整体错位，类内立刻变散，扫描会把错的那个排除。

用法::

    .venv\\Scripts\\python.exe tools\\build_clock_templates.py <录屏路径> [--reuse]

``--reuse`` 复用 ``.tmp/clock_patches.npz``（解码 2670x1200@90fps 很贵，
调参数时不该每次重跑）。
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(ROOT, "app", "collection", "clock_digits.json")
OUT_PNG = os.path.join(ROOT, ".tmp", "clock_templates.png")
CACHE = os.path.join(ROOT, ".tmp", "clock_patches.npz")

#: 四个数字位的取样窗，**统一宽 20**，各自对准该位数字的中心。
#:
#: 列段实测（thr=140, 见 tools/README 标定记录）：
#: M1 1823-1835(中心 1829) / M2 1839-1852(中心 1845.5) /
#: 冒号 1855-1859 / S1 1861-1875(中心 1868) / S2 1880-1887(中心 1883.5)。
#:
#: 统一宽度是对齐的关键：宽度不一致时，同一个数字在四个位置上取出来的
#: 像素落在不同位置，「0」会被拆成好几个簇。
SLOTS: List[Tuple[str, int, int, int, int]] = [
    ("M1", 1819, 1839, 10, 35),
    ("M2", 1836, 1856, 10, 35),
    ("S1", 1859, 1879, 10, 35),
    ("S2", 1876, 1896, 10, 35),
]
assert len({(x1 - x0) for _, x0, x1, _, _ in SLOTS}) == 1, "四扇窗必须等宽"

#: 锚点：视频第 ANCHOR_T 秒，屏幕上显示 ANCHOR_READING。
#: 这是**唯一**需要人工确认的输入，且会被下面的起点扫描校验。
ANCHOR_T = 220.0
ANCHOR_READING = "0021"          # MMSS

#: 采样：顺序解一遍对局区间，每 N 帧取一个。
#:
#: 必须覆盖**整局**，不能只采开头几分钟：一局 12 分钟里每个分钟只占
#: 1/12，早先只密集采了前 3 分钟，结果分钟位只建出数字 0-2 的模板，
#: 第 5 分钟和第 8 分钟全被硬凑成 0 —— 实读 05:21 / 08:21 全变成 00:21。
#: 每分钟拿到 40 个样本，任何一个数字都够。
SCAN_FROM, SCAN_TO, SCAN_STEP = 200.0, 960.0, 135     # 约 1.5 秒一帧

#: 起点扫描范围（相对锚点反推值的偏移）
START_SEARCH = 12

#: 某一位的某个数字至少要这么多样本才建模板。低于此数不建 ——
#: 用三五张图平均出来的「模板」认不准数字，而认错比不认更糟。
MIN_SAMPLES_PER_DIGIT = 8


def _patch(gray: np.ndarray, slot) -> np.ndarray:
    """切一个字位并零均值单位方差。归一化之后背景明暗就被抵消掉了。"""
    _, x0, x1, y0, y1 = slot
    p = gray[y0:y1, x0:x1].astype(np.float32)
    p = p - p.mean()
    s = p.std()
    return (p / s if s > 1e-3 else p).flatten()


def collect(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"打不开视频：{path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    patches: List[np.ndarray] = []
    meta: List[Tuple[float, int]] = []          # (视频秒, 槽位下标)

    def take(frame, t):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for i, slot in enumerate(SLOTS):
            patches.append(_patch(g, slot))
            meta.append((t, i))

    cap.set(cv2.CAP_PROP_POS_MSEC, SCAN_FROM * 1000)
    idx = 0
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t >= SCAN_TO:
            break
        if idx % SCAN_STEP == 0:
            take(fr, t)
            n += 1
            if n % 50 == 0:
                print(f"  ...{t:.0f}s ({n} 帧)", flush=True)
        idx += 1
        del fr
    cap.release()
    return W, H, patches, meta, n


def labels_for(start_t: float, t: float, slot_idx: int) -> List[int]:
    """按起点推算这一帧四个数字位各是什么。

    时钟从 00:00 走到 99:99 是不可能的，实际不到 30 分钟，
    所以直接从秒数取十位个位，不做进位。
    """
    g = int(round(t - start_t))
    if g < 0:
        return []
    mm, ss = divmod(g, 60)
    if mm > 99 or ss > 59:
        return []
    return [mm // 10, mm % 10, ss // 10, ss % 10]


def assign_labels(start_t: float, times: np.ndarray, slots: np.ndarray) -> np.ndarray:
    """给每个样本的**它自己那一位**打上应有数字；无法判定记 -1。"""
    out = np.full(len(times), -1, np.int64)
    for i in range(len(times)):
        lab = labels_for(start_t, float(times[i]), int(slots[i]))
        if lab:
            out[i] = lab[int(slots[i])]
    return out


def score_start(X: np.ndarray, times: np.ndarray, slots: np.ndarray,
                start_t: float, ph: int, pw: int):
    """起点候选的得分：类内平方距离和（越小越紧）。

    起点错一秒，所有标签整体错位，同一个数字就被拆散，代价立刻变大 ——
    这就是校验手段。

    这里对**每一位**分别算，再取平均：不同位的字形落点不一样，
    混在一起算会让「0」这种出现位置多的数字主导整个得分。
    """
    lab_arr = assign_labels(start_t, times, slots)
    ok = lab_arr >= 0

    per_slot_costs, n_used = [], 0
    for s in range(len(SLOTS)):
        ms = slots == s
        cts = np.zeros(10, np.int64)
        cents = np.zeros((10, ph * pw), np.float32)
        for d in range(10):
            m = ok & ms & (lab_arr == d)
            cts[d] = int(m.sum())
            if cts[d]:
                cents[d] = X[m].mean(axis=0)
        used = (cts >= 3).sum()
        if used < 2:
            continue
        tot, n = 0.0, 0
        for d in range(10):
            m = ok & ms & (lab_arr == d)
            if cts[d] < 3:
                continue
            diff = X[m] - cents[d]
            tot += float((diff * diff).sum())
            n += int(cts[d])
        per_slot_costs.append(tot / max(1, n))
        n_used += n

    # 至少要覆盖 8 个不同数字，否则某个数字根本没有模板
    digits_seen = int(sum(1 for d in range(10)
                          if int(((ok & (lab_arr == d)).sum())) >= 3))
    if digits_seen < 8 or n_used < 200:
        return None, None
    return float(np.mean(per_slot_costs)), cents


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("用法：python tools/build_clock_templates.py <录屏> [--reuse]")
    src = sys.argv[1]

    if os.path.isfile(CACHE) and "--reuse" in sys.argv:
        z = np.load(CACHE)
        X = z["X"].astype(np.float32)
        W, H = int(z["W"]), int(z["H"])
        times, slots = z["times"].astype(np.float64), z["slots"].astype(np.int64)
        print(f"  复用缓存 {CACHE}: {X.shape[0]} 个样本, {W}x{H}", flush=True)
    else:
        print(f"读取 {src}")
        W, H, patches, meta, n = collect(src)
        X = np.asarray(patches, np.float32)
        times = np.array([m[0] for m in meta], np.float64)
        slots = np.array([m[1] for m in meta], np.int64)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez_compressed(CACHE, X=X, times=times, slots=slots, W=W, H=H)
        print(f"  {W}x{H}, 密集段 + 稀疏段 {n} 帧, 共 {X.shape[0]} 个样本", flush=True)

    ph, pw = 25, 20
    base = ANCHOR_T - (int(ANCHOR_READING[:2]) * 60 + int(ANCHOR_READING[2:]))
    print(f"\n锚点 t={ANCHOR_T:.0f}s 读作 {ANCHOR_READING} → 起点约 {base:.0f}s")
    print("扫描起点 (±%d 秒，取类内最紧的)：" % START_SEARCH)

    best = None
    for off in range(-START_SEARCH, START_SEARCH + 1):
        st = base + off
        sc, counts = score_start(X, times, slots, st, ph, pw)
        if sc is None:
            print(f"  起点 {st:7.1f}s (偏移 {off:+d})  覆盖数字不足，跳过")
            continue
        mark = ""
        if best is None or sc < best[0]:
            best = (sc, st, off)
            mark = "  <== 当前最好"
        print(f"  起点 {st:7.1f}s (偏移 {off:+d})  类内代价 {sc:.4f}{mark}")

    if best is None:
        raise SystemExit("没有任何起点候选能覆盖足够的数字，检查锚点是否正确")
    _, start_t, off = best
    print(f"\n采用起点 {start_t:.1f}s（偏移 {off:+d} 秒）")

    # ---- 按「位 × 标签」建模板 ----
    #
    # 必须**按位各建一套**。同一个数字在四位上的落点不一样，
    # 混在一起求均值会把它们抹平 —— 实测「0」在分钟十位与秒个位之间
    # 的相关只有 0.86，模板直接糊成一团。
    lab_arr = assign_labels(start_t, times, slots)
    ok = lab_arr >= 0

    slot_templates = []
    for si, (name, _, _, _, _) in enumerate(SLOTS):
        ms = slots == si
        cents = np.zeros((10, ph * pw), np.float32)
        cts = np.zeros(10, np.int64)
        for d in range(10):
            m = ok & ms & (lab_arr == d)
            cts[d] = int(m.sum())
            if cts[d]:
                cents[d] = X[m].mean(axis=0)
        have = [d for d in range(10) if cts[d] >= MIN_SAMPLES_PER_DIGIT]
        slot_templates.append((name, cents, cts, have))
        print(f"  {name}: 样本 {cts.tolist()}  可用数字 {have}")

    grid = []
    for name, cents, cts, have in slot_templates:
        for d in have:
            v = cents[d] - cents[d].min()
            if v.max() > 1e-6:
                v = v / v.max()
            img = (v.reshape(ph, pw) * 255).round().astype(np.uint8)
            img = cv2.resize(img, (pw * 5, ph * 5), interpolation=cv2.INTER_NEAREST)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(img, (0, 0), (img.shape[1]-1, img.shape[0]-1), (0, 0, 255), 1)
            cv2.putText(img, f"{name}{d}", (2, 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 255, 255), 1)
            grid.append(img)

    out: Dict[str, object] = {
        "calibrated_on": {"width": W, "height": H},
        "anchor": {"video_t": ANCHOR_T, "reading": ANCHOR_READING,
                   "game_start_t": round(start_t, 1)},
        "note": "由 tools/build_clock_templates.py 从真实录屏生成，勿手改。"
                "换机型/分辨率请重跑该脚本。每位一套模板：同一个数字在四个"
                "位置上的落点不同，混用会把字形抹平。某位缺哪个数字，"
                "读时钟时该位出现这个数字就会读不出——这是刻意的，不猜。",
        "size": [pw, ph],
        "slots": [],
    }
    for (name, x0, x1, y0, y1), (_, cents, cts, have) in zip(SLOTS, slot_templates):
        out["slots"].append({                      # type: ignore[union-attr]
            "name": name,
            "box": [round(x0 / W, 6), round(y0 / H, 6),
                    round((x1 - x0) / W, 6), round((y1 - y0) / H, 6)],
            "labels": have,
            "templates": [cents[d].tolist() for d in have],
        })
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    while len(grid) % 5:
        grid.append(np.zeros_like(grid[0]))
    rows = [np.hstack(grid[i:i+5]) for i in range(0, len(grid), 5)]
    cv2.imwrite(OUT_PNG, np.vstack(rows))

    print(f"\n产物 -> {OUT_JSON} ({os.path.getsize(OUT_JSON)/1024:.1f} KB)")
    print(f"模板图 -> {OUT_PNG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
