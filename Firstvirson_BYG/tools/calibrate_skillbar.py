r"""键位档位标定：把「这套键位的大招按钮在哪」写进标定文件。

为什么需要它
------------
王者荣耀的**键位能自己拖**：同一款游戏，不同玩家的技能按钮位置能差
几十像素（实测用例1 的大招在 (1088,334)、用例2 在 (1014,350)，
两段录屏的画布分辨率、时钟位置、小地图位置**完全一致**）。

大招框一旦对不上，饱和度判据读到的就是**别的按钮**，「就绪/冷却」
变成随机数 —— 而报告上只是「没触发」，看不出错在哪。所以按键位分档标定，
扫描时按右下角按钮群自动认档；认不出就记缺失，不猜。

用法
----
::

    # 1) 看看这段视频里检出了哪些按钮（编号即下标，从 1 开始）
    python tools/calibrate_skillbar.py <录屏路径> --start 175

    # 2) 把第 2 个按钮定成大招，写进标定文件
    python tools/calibrate_skillbar.py <录屏路径> --start 175 --name 熬隐键位 --ult 2

``--start`` 是**对局起点在视频里的秒数**（读时钟能算出来）。
工具会在这之后的一段时间里跨帧稳定地检出按钮群 ——
草地上的假圆每帧位置都不同，会被滤掉；真按钮位置固定，留下来。

标定结果写进 ``app/collection/hud_regions.json`` 的「档位」里，
识别代码不动。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.collection import skillbar                      # noqa: E402
from app.collection import video_upload as vu             # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REGIONS_FILE = ROOT / "app" / "collection" / "hud_regions.json"

#: 采样帧数。少了容易把噪声当按钮，多了慢。
SAMPLES = 16
#: 采样起点相对对局起点的偏移（秒）：跳过开局，HUD 才稳定。
START_OFFSET = 180.0
#: 采样间隔（秒）。
STEP = 5.0

#: 大招框相对按钮的**半宽/半高**（按画布比例）。跟标定几何一致：
#: 框要比按钮大一点，才能框住图标而不是只框住圆心。
HALF_W = 0.031
HALF_H = 0.065


def collect(path: str, start: float, samples: int = SAMPLES) -> list:
    """跨帧稳定地检出右下角按钮群。"""
    import cv2

    video = vu.UploadedVideo(video_id="cal", original_name="c.mp4",
                             path=path, size=1)
    video = vu.probe(video)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    for i in range(samples):
        t = start + START_OFFSET + i * STEP
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            break
        canvas = vu.crop_to_band(vu.orient_frame(cv2, frame, 0),
                                 vu.video_view(video).get("band"))
        if canvas is not None:
            frames.append(canvas)
    cap.release()
    return skillbar.stable_buttons(frames)


def main() -> int:
    ap = argparse.ArgumentParser(description="标定王者键位档位（大招按钮位置）")
    ap.add_argument("video", help="录屏路径")
    ap.add_argument("--start", type=float, required=True,
                    help="对局起点在视频里的秒数")
    ap.add_argument("--name", default="", help="档位名（要写进标定文件就带上）")
    ap.add_argument("--ult", type=int, default=0,
                    help="第几个按钮是大招（从 1 开始）")
    ap.add_argument("--desc", default="", help="档位说明")
    ap.add_argument("--dump", default="", help="把标注图存到这个路径")
    ap.add_argument("--list", action="store_true", help="只看当前已有的档位")
    args = ap.parse_args()

    data = json.loads(REGIONS_FILE.read_text(encoding="utf-8"))
    if args.list:
        for name, spec in (data.get("档位") or {}).items():
            ult = spec.get("大招") or {}
            print(f"{name}: 大招 ({ult.get('x')}, {ult.get('y')}) r={ult.get('r')}")
            print(f"    {spec.get('描述', '')}")
        return 0

    buttons = collect(args.video, args.start)
    if not buttons:
        print("没检出稳定的圆形按钮 —— 这段录屏可能不是王者对局画面")
        return 1
    print(f"检出 {len(buttons)} 个稳定按钮（从 1 开始编号）：")
    for i, b in enumerate(buttons, 1):
        print(f"  {i:2d}. ({b.x:.4f}, {b.y:.4f}) r={b.r:.4f}")

    if args.dump:
        import cv2
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES,
                int((args.start + START_OFFSET) * fps))
        ok, frame = cap.read()
        cap.release()
        if ok:
            h, w = frame.shape[:2]
            vis = frame.copy()
            for i, b in enumerate(buttons, 1):
                cv2.circle(vis, (int(b.x * w), int(b.y * h)), int(b.r * h),
                           (0, 0, 255), 2)
                cv2.putText(vis, str(i), (int(b.x * w) + 4, int(b.y * h) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imwrite(args.dump, vis)
            print(f"标注图 -> {args.dump}")

    if not args.name or not args.ult:
        print("\n（没给 --name/--ult，只看检测结果，不写文件）")
        return 0
    if not 1 <= args.ult <= len(buttons):
        print(f"--ult 超出范围（1..{len(buttons)}）")
        return 1

    ult = buttons[args.ult - 1]
    box = [round(ult.x - HALF_W, 4), round(ult.y - HALF_H, 4),
           round(ult.x + HALF_W, 4), round(ult.y + HALF_H, 4)]
    profiles = data.setdefault("档位", {})
    profiles[args.name] = {
        "描述": args.desc or f"{args.name}（{args.video} 标定）",
        "来源": os.path.basename(args.video),
        "大招": {"x": round(ult.x, 4), "y": round(ult.y, 4),
                 "r": round(ult.r, 4)},
        "按钮": [b.to_dict() for b in buttons],
        "覆盖": {"skill_ult": {"box": box, "已标定": True,
                               "用途": f"大招按钮（3技能）—— {args.name}"}},
    }
    REGIONS_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已写档位「{args.name}」到 {REGIONS_FILE}")
    print(f"  大招 ({ult.x:.4f}, {ult.y:.4f})  box={box}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
