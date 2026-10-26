"""尖峰：用真实的一窗去打一次模型，看通道通不通、答案可不可用。

用法::

    .\\.venv\\Scripts\\python.exe tools\\spike_vlm_window.py
    .\\.venv\\Scripts\\python.exe tools\\spike_vlm_window.py --video videos\\ec5ea4672d90.mp4 \\
        --start 175 --lo 2 --hi 4 --frames 8

它做的事：抽这一窗的若干帧 → 带上**封闭选项**词表 → 发给模型 → 打印原始回答与解析结果。

这是「步 0 尖峰」：判据是
1. 端点收得下媒体块（图片 base64）；
2. 回答是**可解析的 JSON**；
3. 内容不是胡话（人工看一眼）。

跑不通就别往下写功能 —— 先解决通道或画质。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.collection import video_upload as vu          # noqa: E402
from app.collection import vlm_client                  # noqa: E402

#: 送给模型的封闭选项 —— 取自适配器自己的动作词表，
#: 不用我们自己另编一套（两套词表对不上，判定层就认不出来了）。
def action_vocabulary() -> str:
    from app.adapters.honor_of_kings import MOBA_ACTION_VALUES

    lines: List[str] = []
    for kind, values in MOBA_ACTION_VALUES.items():
        if not values:
            continue
        lines.append(f"{kind}：")
        for code, label in values.items():
            lines.append(f"  - {code}（{label}）")
    return "\n".join(lines)


def grab_frames(video_path: str, times: List[float],
                width: int = 768) -> List[Tuple[float, bytes]]:
    """抽帧并压成 JPEG 字节。

    用采集层自己的开视频/裁画面逻辑（``open_capture`` + ``video_view`` +
    ``orient_frame`` + ``crop_to_band``）—— 和流水线抽出来的画面一致，
    不另发明一套，免得尖峰通过了、正式跑却对不上。
    """
    import cv2

    cap, meta_rot = vu.open_capture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"打不开视频：{video_path}")
    view = vu.video_view(vu.UploadedVideo(
        video_id="spike", path=video_path, original_name=os.path.basename(video_path),
        size=0, duration_sec=0.0))
    band = view.get("band")
    rotation = int(view.get("rotation", meta_rot))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out: List[Tuple[float, bytes]] = []
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, img = cap.read()
            if not ok or img is None:
                continue
            img = vu.crop_to_band(vu.orient_frame(cv2, img, rotation), band)
            h, w = img.shape[:2]
            if w > width:
                scale = width / float(w)
                img = cv2.resize(img, (width, int(h * scale)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                out.append((round(t, 1), bytes(buf)))
    finally:
        cap.release()
    return out


def build_prompt(frames: List[Tuple[float, bytes]], lo: int, hi: int,
                 start: float) -> Tuple[str, str]:
    """(系统提示, 用户提示)。"""
    system = (
        "你是一个王者荣耀对局复盘助手。你只看画面，只描述你**看到**的东西，"
        "不要猜、不要美化、不要给攻略建议。看不清的就说看不清。\n"
        "只能从给定的动作清单里选值，不要发明新词。"
    )
    lines: List[str] = []
    for i, (t, _b) in enumerate(frames, start=1):
        minute = (t - start) / 60.0
        lines.append(f"第 {i} 帧 = 局内第 {int(minute)} 分 {int((minute % 1) * 60):02d} 秒")
    user = (
        f"下面是同一局王者里局内第 {lo} 分 00 秒到第 {hi} 分 00 秒这一段的画面，"
        f"共 {len(frames)} 帧，按时间先后编号：\n"
        + "\n".join(lines)
        + "\n\n可选动作清单（只能用这些值）：\n"
        + action_vocabulary()
        + "\n\n请只回答一个 JSON 对象，结构如下（不要加别的文字）：\n"
        + json.dumps({
            "actions": [{
                "kind": "objective_action",
                "value": "farm_jungle",
                "frames": [1, 3],
                "why": "第 1、3 帧小地图在野区反复移动",
            }],
            "position": "jungle",
            "unknown": ["看不出是哪一类的动作"],
        }, ensure_ascii=False, indent=2)
        + "\n\n规则：\n"
        + "1. actions 里每一条都要给出**帧号**（1 起），没有证据就不要写。\n"
        + "2. 看不清的整类动作放进 unknown，不要猜。\n"
        + "3. value 必须是清单里的值。\n"
    )
    return system, user


def make_clip(video_path: str, t0: float, t1: float, out_path: str,
              width: int = 960, fps: float = 15.0) -> str:
    """把这一窗剪成一个小视频（给「视频理解」通道用）。

    机器上**没有 ffmpeg**，只能用 OpenCV 自己编（mp4v）。
    帧率降到 15：小地图的移动在 15 帧/秒下照样看得清，
    而 60 帧/秒编两分钟要编 7200 帧，慢到不可接受。
    """
    import cv2

    cap, meta_rot = vu.open_capture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"打不开视频：{video_path}")
    view = vu.video_view(vu.UploadedVideo(
        video_id="spike", path=video_path, original_name=os.path.basename(video_path),
        size=0, duration_sec=0.0))
    band = view.get("band")
    rotation = int(view.get("rotation", meta_rot))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / fps)))
    writer = None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t0 * src_fps))
        idx = 0
        while True:
            t = t0 + (idx / fps)
            if t >= t1:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * src_fps))
            ok, img = cap.read()
            if not ok or img is None:
                break
            img = vu.crop_to_band(vu.orient_frame(cv2, img, rotation), band)
            h, w = img.shape[:2]
            if w > width:
                scale = width / float(w)
                img = cv2.resize(img, (width, int(h * scale)), interpolation=cv2.INTER_AREA)
            if writer is None:
                hh, ww = img.shape[:2]
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                        fps, (ww, hh))
            writer.write(img)
            idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if not os.path.isfile(out_path):
        raise SystemExit("片段没写出来")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="videos/ec5ea4672d90.mp4")
    ap.add_argument("--start", type=float, default=175.0, help="局内 0:00 对应的视频秒")
    ap.add_argument("--lo", type=int, default=2, help="窗口起始局内分钟")
    ap.add_argument("--hi", type=int, default=4, help="窗口结束局内分钟")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--model", default=vlm_client.DEFAULT_MODEL)
    ap.add_argument("--out", default="", help="把原始回答写到文件")
    ap.add_argument("--clip", default="", help="同时把这一窗剪成小视频（给视频理解通道用）")
    ap.add_argument("--save-frames", default="", help="把抽到的帧存到这个目录")
    args = ap.parse_args()

    creds = vlm_client.find_credentials()
    if creds is None:
        print("!! 没找到凭据：设环境变量 MINIMAX_API_KEY 或放一个 config.yaml")
        return 2
    print(f"凭据来源：{creds.source}")
    print(f"端点：{creds.base_url}")
    print(f"模型：{args.model}")

    t0 = args.start + args.lo * 60.0
    t1 = args.start + args.hi * 60.0
    n = max(1, args.frames)
    times = [t0 + (t1 - t0) * (i + 0.5) / n for i in range(n)]

    if args.clip:
        path = make_clip(args.video, t0, t1, args.clip)
        print(f"片段已写到 {path}（{os.path.getsize(path) // 1024} KB）")

    frames = grab_frames(args.video, times, args.width)
    if not frames:
        print("!! 一帧都没抽到")
        return 2
    size_kb = sum(len(b) for _t, b in frames) // 1024
    print(f"抽到 {len(frames)} 帧，合计 {size_kb} KB；"
          f"时间 {[t for t, _b in frames]}")

    if args.save_frames:
        out_dir = Path(args.save_frames)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, (t, data) in enumerate(frames, start=1):
            (out_dir / f"frame-{i:02d}.jpg").write_bytes(data)
        print(f"帧已存到 {out_dir}")

    system, user = build_prompt(frames, args.lo, args.hi, args.start)
    blocks = [vlm_client.text_block(user)]
    for _t, data in frames:
        blocks.append(vlm_client.image_block(data))

    try:
        result = vlm_client.call_model(
            args.model, blocks, creds=creds, system=system, max_tokens=3000)
    except vlm_client.VlmError as exc:
        print(f"!! 调用失败：{exc}（kind={exc.kind} status={exc.status}）")
        if exc.detail:
            print("   细节：", exc.detail[:800])
        return 1

    print("\n===== 原始回答 =====")
    print(result.text)
    print("\n===== 解析 =====")
    try:
        parsed = result.json()
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    except vlm_client.VlmError as exc:
        print(f"!! 解析失败：{exc}")
        return 1
    if args.out:
        Path(args.out).write_text(
            json.dumps({"raw": result.text, "parsed": parsed},
                       ensure_ascii=False, indent=2), "utf-8")
        print(f"\n已写到 {args.out}")
    print("\nusage：", json.dumps(result.usage, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
