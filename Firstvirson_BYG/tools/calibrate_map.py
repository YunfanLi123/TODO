"""地图区域标定：在小地图底图上框出各区域，写进 ``map_geometry.json``。

为什么必须对着图标
------------------
「在龙坑」和「在野区」只差一个区域框的位置。框歪一点，
报告上就会出现「这局开龙判成了刷野」，而且**报告上看不出来** ——
玩家只看到一条不对的建议。

所以这份表只能从**真实底图**上框出来，不许凭印象写。
底图由 ``tools/extract_minimap_reference.py`` 从录屏里取中位数得到。

用法
----
::

    # 看当前标定
    python tools/calibrate_map.py --list

    # 框一个区域（坐标是底图上的像素，左上角为原点）
    python tools/calibrate_map.py --region 暴君坑 --box 258,196,290,228

    # 框完把那块裁出来核对一眼
    python tools/calibrate_map.py --region 暴君坑 --dump
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GEOMETRY_FILE = os.path.join(ROOT, "app", "collection", "map_geometry.json")
REFERENCE_PNG = os.path.join(ROOT, "app", "collection", "minimap_reference.png")

#: 底图放大倍数（导出时用的，标定坐标要换算回原图）。
REFERENCE_SCALE = 4


def _load() -> dict:
    try:
        with open(GEOMETRY_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"minimap_box": None, "regions": []}


def _save(data: dict) -> None:
    with open(GEOMETRY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def list_regions() -> None:
    data = _load()
    print(f"标定文件：{GEOMETRY_FILE}")
    print(f"底图：{REFERENCE_PNG}（放大 {REFERENCE_SCALE} 倍）")
    regions = data.get("regions") or []
    if not regions:
        print("\n还没有标任何区域。")
        print("用法：--region 名字 --box x0,y0,x1,y1（底图像素）")
        return
    print(f"\n已标 {len(regions)} 个区域：")
    print(f"{'区域':<14}{'小地图相对坐标':<34}备注")
    print("-" * 78)
    for r in regions:
        x0, y0, x1, y1 = r["box"]
        print(f"{r.get('name', ''):<14}"
              f"({x0:.3f}, {y0:.3f}) ~ ({x1:.3f}, {y1:.3f})   "
              f"{r.get('备注', '')}")


def main() -> int:
    ap = argparse.ArgumentParser(description="王者地图区域标定")
    ap.add_argument("--region", help="区域名（如 暴君坑 / 己方野区 / 中路）")
    ap.add_argument("--box", help="底图上的像素框 x0,y0,x1,y1")
    ap.add_argument("--note", default="", help="这个框是怎么定的（写进表里，便于复核）")
    ap.add_argument("--dump", action="store_true", help="把裁剪结果存成图核对")
    ap.add_argument("--list", action="store_true", help="只看当前标定")
    args = ap.parse_args()

    if args.list or not args.region:
        list_regions()
        return 0
    if not args.box:
        print("要标定就得给 --box x0,y0,x1,y1")
        return 1
    try:
        import cv2
    except ImportError:
        print("没有 opencv")
        return 1

    try:
        x0, y0, x1, y1 = [int(v) for v in args.box.replace(" ", "").split(",")]
    except ValueError:
        print("框要写成 x0,y0,x1,y1")
        return 1

    ref = cv2.imread(REFERENCE_PNG)
    if ref is None:
        print(f"读不出底图：{REFERENCE_PNG}")
        print("先跑 tools/extract_minimap_reference.py 从录屏里导出")
        return 1
    h, w = ref.shape[:2]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        print(f"框超出底图范围（底图 {w}x{h}）")
        return 1

    # 换算成**小地图内**的相对坐标（0~1）
    rel = [round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4)]
    data = _load()
    regions = data.setdefault("regions", [])
    for r in regions:
        if r.get("name") == args.region:
            r["box"] = rel
            if args.note:
                r["备注"] = args.note
            _save(data)
            print(f"已更新 {args.region}：{rel}")
            break
    else:
        regions.append({"name": args.region, "box": rel, "备注": args.note})
        _save(data)
        print(f"已标定 {args.region}：{rel}")

    if args.dump:
        out = os.path.join(ROOT, ".tmp", f"map_{args.region}.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cv2.imwrite(out, ref[y0:y1, x0:x1])
        print(f"裁剪结果：{out}（核对一下框对不对）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
