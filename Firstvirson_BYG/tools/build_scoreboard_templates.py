"""顶栏数字模板：抽字形 → 聚类 → 出拼图给人标号 → 写模板。

为什么单独做一套
----------------
顶栏的字体和计时器**不是同一套**（位数、字形都不一样），
所以不能复用 ``app/collection/clock_digits.json``。

也不能凭印象写模板
------------------
模板必须从**真实录屏**里长出来。这套流程是：

1. 从录屏里按固定间隔抽帧；
2. 在顶栏区域里按**颜色**分开两边的数字
   （左队偏蓝、右队偏红，中间「vs」是灰的 —— 颜色天然分得开）；
3. 把每个数字切成一个字形块，**聚类**（同一个数字的像素长得一样）；
4. 出一张拼图，**由人肉眼标号**（这一步不能省 ——
   让程序自己认字形，等于让它自己编数字）；
5. 把标好的模板写进 ``app/collection/scoreboard_digits.json``。

用法
----
::

    # 1) 抽字形并聚类，出拼图
    python tools/build_scoreboard_templates.py <视频> --out .tmp/sb_sheet.png

    # 2) 照着拼图填 labels.json（键是簇号，值是数字），然后写模板
    python tools/build_scoreboard_templates.py <视频> --labels labels.json \
        --write app/collection/scoreboard_digits.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#: 字形切分与尺寸**只在 app 里定义一份** —— 工具和识别层用同一套，
#: 两处各写一遍迟早会对不上（标出来的模板认不出）。
from app.collection import scoreboard as sb  # noqa: E402

#: 模板文件（随包打包）。
TEMPLATE_JSON = os.path.join(ROOT, "app", "collection", "scoreboard_digits.json")

GLYPH_W, GLYPH_H = sb.GLYPH_W, sb.GLYPH_H

#: 聚类时，两个字形算同一类的距离上限（归一化后的欧氏距离）。
CLUSTER_DIST = 0.62


def _region_box(regions_path: str, name: str) -> Optional[Tuple[float, float, float, float]]:
    try:
        with open(regions_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    box = ((data.get("regions") or {}).get(name) or {}).get("box")
    if not box or len(box) != 4:
        return None
    return tuple(float(v) for v in box)


#: 字形切分**只用 app 里那一份**（`app/collection/scoreboard.py`）。
#:
#: 之前这里有一份自己的副本，改阈值时只改了一边、另一边的常量没了 ——
#: 两份实现迟早会对不上，而「切出来的字形不一样」这种错没人看得出来。
extract_glyphs = sb.extract_glyphs


def _unit(vec: np.ndarray) -> np.ndarray:
    """零均值 + 单位长度：匹配时做内积，比对形状不看亮度。**先展平**。"""
    v = np.asarray(vec, dtype="float32").reshape(-1)
    v = v - float(v.mean())
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.zeros_like(v)
    return (v / n).astype(np.float32)


def cluster(glyphs: Sequence[np.ndarray], sides: Sequence[str],
            dist: float = CLUSTER_DIST) -> List[Dict[str, Any]]:
    """把字形聚成类。

    **不分左右** —— 顶栏左右是同一套字体（只是颜色不同），分成两套聚类
    只会让"出现得少的那一侧"缺模板（实测一局里 4/5/6/8 只在一侧出现过）。
    分侧只用来**切哪一块**，不用来分模板。

    每个簇存两份：``raw``（平均后的原始灰度，给人看）、
    ``centroid``（零均值单位向量，给匹配用）。
    """
    clusters: List[Dict[str, Any]] = []
    for raw in glyphs:
        vec = _unit(raw)
        best = None
        for cl in clusters:
            d = float(np.linalg.norm(cl["centroid"] - vec))
            if d < dist and (best is None or d < best[0]):
                best = (d, cl)
        if best is None:
            clusters.append({"side": "all", "centroid": vec.copy(),
                             "raw": raw.copy(), "members": [raw], "count": 1})
        else:
            cl = best[1]
            cl["members"].append(raw)
            cl["raw"] = np.mean(cl["members"], axis=0).astype(np.float32)
            cl["centroid"] = _unit(cl["raw"])
            cl["count"] = len(cl["members"])
    clusters.sort(key=lambda c: -c["count"])
    for i, cl in enumerate(clusters):
        cl["id"] = i
    return clusters


def sheet_image(clusters: Sequence[Dict[str, Any]], cell: int = 110,
                scale: int = 8, cols: int = 4) -> np.ndarray:
    """出聚类拼图：每簇一个格子，画出平均字形 + 簇号 + 样本数。

    字形本身只有 8x12 像素，放大 ``scale`` 倍才认得出是几 ——
    标号这一步是人做的，字太小会标错。

    每张字形**单独做对比度拉伸**（各自减自己的最小值再按最大值归一）：
    顶栏的数字本来就暗，直接画出来是一团灰，认不出是几。
    """
    rows = (len(clusters) + cols - 1) // cols
    img = np.zeros((rows * cell, cols * cell, 3), np.uint8)
    for cl in clusters:
        i = cl["id"]
        r, c = divmod(i, cols)
        img[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = (30, 30, 30)
        g = cl["raw"].reshape(GLYPH_H, GLYPH_W).astype(np.float32)
        lo, hi = float(g.min()), float(g.max())
        if hi - lo > 1e-3:
            g = (g - lo) / (hi - lo) * 255.0
        else:
            g = np.zeros_like(g)
        small = cv2.resize(g.astype(np.uint8),
                           (GLYPH_W * scale, GLYPH_H * scale),
                           interpolation=cv2.INTER_NEAREST)
        for yy in range(small.shape[0]):
            for xx in range(small.shape[1]):
                y0 = r * cell + 18 + yy
                x0 = c * cell + 12 + xx
                if y0 < img.shape[0] and x0 < img.shape[1]:
                    img[y0, x0] = (small[yy, xx],) * 3
        color = (200, 200, 200)
        cv2.putText(img, "%d n=%d" % (i, cl["count"]),
                    (c * cell + 6, r * cell + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def collect(video_path: str, boxes: Sequence[Tuple[str, Sequence[float]]],
            start: float, stop: float, step: float
            ) -> Tuple[List[np.ndarray], List[str]]:
    """按间隔抽帧、切字形。

    ``boxes`` 是 ``[(侧, 相对坐标框), ...]`` —— 每一侧一个区域。
    王者顶栏是**两行**：第一行 MM:SS，第二行「N vs M」，左右两队的数字
    颜色不同（蓝/红），所以**每侧各切一块**最干净。

    **取景方式与主程序完全一致**（``video_upload.video_view`` + ``orient_frame``
    + ``crop_to_band``）—— 标定出来的相对坐标必须对应主程序看到的那张画面，
    这里自己另搞一套转正规则，两边就会对不上。
    """
    from app.collection import video_upload as vu

    # 先把流打开，再探针。
    #
    # 顺序不能反：在某些环境里，**这个进程里第一次**打开某个路径的
    # 视频会失败（第二次就好），而 ``vu.probe`` 内部会自己开一次。
    # 先开一次流，probe 里那次就是第二次 —— 稳。
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"打不开视频：{video_path}")

    video = vu.UploadedVideo(video_id="sb_template", original_name="t.mp4",
                             path=video_path, size=1)
    video = vu.probe(video)
    view = vu.video_view(video, hud_layout=vu.DEFAULT_HUD_LAYOUT)
    rot = int(view.get("rotation", 0))
    band = view.get("band")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    glyphs: List[np.ndarray] = []
    sides: List[str] = []
    t = start
    while t <= stop:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            break
        canvas = vu.crop_to_band(vu.orient_frame(cv2, frame, rot), band)
        if canvas is None:
            t += step
            continue
        h, w = canvas.shape[:2]
        for side, box in boxes:
            x0, y0, x1, y1 = box
            crop = canvas[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
            # 区域本身就分左右了（每侧一块），所以**按区域定侧**，
            # 不再靠颜色猜 —— 颜色只用来把中间那个白的「vs」挡在外面。
            g, s = extract_glyphs(crop, side)
            glyphs.extend(g)
            sides.extend(s)
        t += step
    cap.release()
    return glyphs, sides


def main() -> int:
    ap = argparse.ArgumentParser(description="顶栏数字模板构建")
    ap.add_argument("video", help="录屏路径")
    ap.add_argument("--regions", default=os.path.join(
        ROOT, "app", "collection", "hud_regions.json"))
    ap.add_argument("--region", action="append", default=None,
                    help="区域名，可给多次（默认左右两块击杀数）")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--stop", type=float, default=600.0)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(ROOT, ".tmp", "sb_sheet.png"))
    ap.add_argument("--labels", help="标号文件：{\"簇号\": 数字}")
    ap.add_argument("--write", help="把模板写到这个文件")
    args = ap.parse_args()

    names = args.region or ["scoreboard_kill_left", "scoreboard_kill_right"]
    boxes = []
    for name in names:
        box = _region_box(args.regions, name)
        if box is None:
            print(f"标定文件里没有 {name} 的框")
            return 1
        side = "left" if name.endswith("_left") else "right"
        boxes.append((side, box))
        print(f"区域 {name}（{side}）相对坐标 {box}")

    glyphs, sides = collect(args.video, boxes, args.start, args.stop, args.step)
    print(f"抽出字形 {len(glyphs)} 个")
    clusters = cluster(glyphs, sides)
    print(f"聚成 {len(clusters)} 类：")
    for cl in clusters:
        print(f"  簇 {cl['id']:3d} {'蓝' if cl['side'] == 'left' else '红'} "
              f"{cl['count']:5d} 个")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, sheet_image(clusters))
    print(f"拼图 -> {args.out}")

    if args.labels:
        with open(args.labels, encoding="utf-8") as f:
            labels = json.load(f)
        out: Dict[str, Any] = {
            "版本": 1,
            "说明": "顶栏数字模板（王者顶栏字体，与计时器不是同一套）",
            "size": [GLYPH_W, GLYPH_H],
            "digits": [],
        }
        for cl in clusters:
            lab = labels.get(str(cl["id"]))
            if lab is None:
                print(f"簇 {cl['id']} 没有标号 —— 先照拼图标上再写")
                return 1
            out["digits"].append({
                "label": int(lab),
                "side": cl["side"],
                "template": [round(float(v), 4) for v in cl["centroid"]],
            })
        dest = args.write or TEMPLATE_JSON
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
            f.write("\n")
        print(f"模板已写入 {dest}（{len(out['digits'])} 个）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
