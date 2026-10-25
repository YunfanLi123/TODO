"""HUD 区域标定：把截图上的目标框，换成**相对坐标**写进标定文件。

为什么要它
----------
王者的 UI 位置是**固定的百分比**（摇杆在 (0.164, 0.798)、技能在右下、
信号按钮在右上角 —— wzry_ai 的 ``argparses.py`` 就是这么记的）。
但**清晰度与机型不同**，同一个元素在不同录屏上的像素尺寸不一样。

所以：识别代码只认**相对坐标**，换设备/换录屏软件时重跑这个工具标定一次，
识别代码一行不改。

用法
----
::

    # 1) 在截图上框出「顶栏金币」那块，写进标定文件
    python tools/calibrate_hud.py shot.png --region scoreboard_gold --box 310,0,390,14

    # 2) 只看当前标定
    python tools/calibrate_hud.py --list

    # 3) 标定完把裁剪结果存出来，肉眼核对框对不对
    python tools/calibrate_hud.py shot.png --region scoreboard_gold --dump

坐标是**截图上的像素**（左上角为原点），工具会自己换算成相对坐标。

已知的锚点（可以直接当起点）
----------------------------
    摇杆     (0.164, 0.798)      小地图   左上角约 (0.0, 0.0) ~ (0.22, 0.30)
    1 技能   (0.71, 0.874)       信号按钮 (0.926, 0.14)
    2 技能   (0.76, 0.69)        装备     (0.133, 0.4) / (0.133, 0.51)
    3 技能   (0.844, 0.58)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
REGIONS_FILE = ROOT / "app" / "collection" / "hud_regions.json"

#: 已知锚点（相对坐标），用来提示。
ANCHORS = {
    "minimap": "左上角，约 (0.00, 0.00) ~ (0.22, 0.30)",
    "skills": "右下，约 (0.67, 0.48) ~ (0.84, 0.91)",
    "joystick": "左下，约 (0.164, 0.798)",
    "clock": "顶部中偏右，约 (0.68, 0.00) ~ (0.71, 0.022)",
    "scoreboard_gold": "顶栏金币（两串数字，中间是双方差值区）",
    "scoreboard_kill": "顶栏击杀数",
    "skill_ult": "大招按钮（右下那排技能里最大的那个）",
}


def _load() -> dict:
    try:
        with open(REGIONS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"版本": 1, "regions": {}, "anchors": {}}


def _save(data: dict) -> None:
    with open(REGIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def list_regions() -> None:
    data = _load()
    print(f"标定文件：{REGIONS_FILE}")
    print(f"{'区域':<18}{'相对坐标':<34}{'已标定'}")
    print("-" * 68)
    for name, spec in (data.get("regions") or {}).items():
        box = spec.get("box") or []
        rel = "(" + ", ".join(f"{float(v):.4f}" for v in box) + ")" if box else "-"
        print(f"{name:<18}{rel:<34}{'是' if spec.get('已标定') else '否'}")
    print("\n已知锚点：")
    for name, hint in ANCHORS.items():
        print(f"  {name:<18}{hint}")


def main() -> int:
    ap = argparse.ArgumentParser(description="王者 HUD 区域标定")
    ap.add_argument("image", nargs="?", help="截图路径")
    ap.add_argument("--region", help="要标定的区域名")
    ap.add_argument("--box", help="截图上的像素框 x0,y0,x1,y1")
    ap.add_argument("--point", help="截图上的一个点 x,y（作为区域中心）")
    ap.add_argument("--size", help="配合 --point：区域宽高（像素），如 80,14")
    ap.add_argument("--dump", action="store_true", help="把裁剪结果存成图，肉眼核对")
    ap.add_argument("--list", action="store_true", help="只看当前标定")
    args = ap.parse_args()

    if args.list or not args.image:
        list_regions()
        return 0

    if not args.region:
        print("要标定就得给 --region（区域名）")
        return 1
    if not args.box and not args.point:
        print("给 --box x0,y0,x1,y1 或 --point x,y（配合 --size 宽,高）")
        return 1

    try:
        import cv2
    except ImportError:
        print("没有 opencv")
        return 1

    img = cv2.imread(args.image)
    if img is None:
        print(f"读不出图片：{args.image}")
        return 1
    h, w = img.shape[:2]

    if args.box:
        x0, y0, x1, y1 = [int(v) for v in args.box.replace(" ", "").split(",")]
    else:
        px, py = [int(v) for v in args.point.replace(" ", "").split(",")]
        sw, sh = [int(v) for v in args.size.replace(" ", "").split(",")]
        x0, y0, x1, y1 = px - sw // 2, py - sh // 2, px + sw // 2, py + sh // 2

    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        print("框太小了")
        return 1

    rel = [round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4)]
    data = _load()
    regions = data.setdefault("regions", {})
    spec = regions.setdefault(args.region, {})
    spec["box"] = rel
    spec["已标定"] = True
    spec.setdefault("用途", ANCHORS.get(args.region, ""))
    data["版本"] = int(data.get("版本", 1))
    _save(data)
    print(f"已标定 {args.region}：像素 ({x0},{y0})-({x1},{y1})  相对 {rel}")

    if args.dump:
        out = ROOT / ".tmp" / f"calib_{args.region}.png"
        os.makedirs(out.parent, exist_ok=True)
        cv2.imwrite(str(out), img[y0:y1, x0:x1])
        print(f"裁剪结果：{out}（核对一下框对不对）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
