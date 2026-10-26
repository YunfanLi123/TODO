"""让模型**先读一遍**，玩家只做核对。

哪些是画面里读得出来的
----------------------
- **分路**：玩家在地图上主要活动在哪条线 —— 小地图看得到。
- **胜负**：结算画面上写着 —— 录屏结尾那一帧。
- **BP 禁用与选人**：BP 阶段画面上是英雄头像与禁用标记 —— 抽帧那一步已经抽了。

这三样以前要用户一格一格填。玩家原话：**「难道还要我自己去选？」**

原则
----
**预填 ≠ 替玩家决定。**
- 读不出来就留空（**不猜**），由用户自己填；
- 读出来的都标「模型读的，请核对」；
- 用户改过之后，以用户为准（前端不再覆盖）。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import vlm_client
from .vlm_recognizer import SYSTEM_PROMPT

#: 位置的可选值（从适配器取，**不另编一套**）。
def position_values() -> List[str]:
    try:
        from ..adapters.honor_of_kings import POSITION_VALUES
        return list(POSITION_VALUES)
    except Exception:                              # noqa: BLE001
        return ["jungle", "mid", "side", "carry", "roam"]


def _frame_bytes(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError:
        return None


def _grab(video, base_dir: str, times: Sequence[float], prefix: str,
          width: int = 768) -> List[Tuple[float, bytes]]:
    """抽几帧并压成 JPEG 字节（复用采集层的开视频/裁画面逻辑）。"""
    from . import video_upload as vu

    try:
        import cv2
    except Exception:                              # noqa: BLE001
        return []
    cap, meta_rot = vu.open_capture(video.path)
    if not cap.isOpened():
        return []
    view = vu.video_view(vu.UploadedVideo(
        video_id="prefill", path=video.path,
        original_name=os.path.basename(video.path), size=0, duration_sec=0.0))
    band = view.get("band")
    rotation = int(view.get("rotation", meta_rot))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out: List[Tuple[float, bytes]] = []
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(t * fps)))
            ok, img = cap.read()
            if not ok or img is None:
                continue
            img = vu.crop_to_band(vu.orient_frame(cv2, img, rotation), band)
            h, w = img.shape[:2]
            if w > width:
                scale = width / float(w)
                img = cv2.resize(img, (width, int(h * scale)),
                                 interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                out.append((round(t, 1), bytes(buf)))
    finally:
        cap.release()
    return out


def read_lane_and_result(video, base_dir: str, start_t: Optional[float] = None,
                          duration_sec: float = 0.0) -> Dict[str, Any]:
    """读出「玩家打什么位置」「这局赢没赢」。

    读不出来就是 ``unknown`` —— **不猜**。
    """
    from . import settings

    creds = settings.credentials_for("vlm")
    if creds is None:
        return {"position": "unknown", "result": "unknown",
                "reason": "没有配置「看画面」的模型"}
    model = settings.model_name("vlm") or creds.model

    t0 = float(start_t or 0.0)
    # 开局几帧（判位置）+ 结尾几帧（判结算画面）
    early = [t0 + 20.0 + 40.0 * i for i in range(4)]
    late = [max(0.0, duration_sec - 12.0 - 6.0 * i) for i in range(3)]
    frames = _grab(video, base_dir, early + late, "prefill")
    if not frames:
        return {"position": "unknown", "result": "unknown",
                "reason": "抽不到画面"}

    lines = [f"第 {i} 帧 = 视频第 {t:.1f} 秒"
             + ("（接近录屏结尾，可能是结算画面）" if i > 4 else "（对局刚开始）")
             for i, (t, _b) in enumerate(frames, start=1)]
    prompt = (
        "下面是同一局《王者荣耀》录屏里的几帧：前 4 帧是开局附近，"
        "后 3 帧接近录屏结尾（可能是结算画面）。\n"
        + "\n".join(lines)
        + "\n\n只回答一个 JSON：\n"
        + '{"position": "' + "、".join(position_values()) + '、unknown",'
        + ' "result": "win|lose|unknown",'
        + ' "why": "一句话依据（比如「一直在对抗路」「结算画面写着失败」）"}\n'
        + "\n规则：只能从上面的取值里选；看不出来就写 unknown，不要猜。"
    )
    blocks = [vlm_client.text_block(prompt)]
    for _t, data in frames:
        blocks.append(vlm_client.image_block(data))
    try:
        result = vlm_client.call_model(
            model, blocks, creds=creds, system=SYSTEM_PROMPT,
            max_tokens=800, timeout=180.0)
        answer = result.json()
    except vlm_client.VlmError as exc:
        return {"position": "unknown", "result": "unknown",
                "reason": f"模型没调通：{exc.kind}"}
    if not isinstance(answer, dict):
        return {"position": "unknown", "result": "unknown",
                "reason": "模型回答不是对象"}
    pos = str(answer.get("position") or "").strip()
    if pos not in position_values():
        pos = "unknown"
    res = str(answer.get("result") or "").strip().lower()
    if res not in ("win", "lose"):
        res = "unknown"
    return {"position": pos, "result": res,
            "why": str(answer.get("why") or "").strip()}


def read_bp_steps(video, base_dir: str) -> Dict[str, Any]:
    """从 BP 阶段的画面读出「谁禁了谁、谁选了谁」+ 阵容缺了什么。

    BP 帧在抽帧那一步已经抽好了（``<video_id>_frames/bp-*.jpg``）。
    读不出来的**整条不写** —— 宁可少一条，也不要编一条禁用。

    返回 ``{"decisions": [...], "my_missing_frontline": bool, ...}``。
    """
    from . import settings
    from ..adapters.honor_of_kings import HEROES_BY_LANE

    creds = settings.credentials_for("vlm")
    if creds is None:
        return {}
    model = settings.model_name("vlm") or creds.model
    frames_dir = os.path.join(
        __import__("app.collection.video_upload", fromlist=["x"]).video_store_dir(base_dir),
        video.video_id + "_frames")
    if not os.path.isdir(frames_dir):
        return {}
    names = sorted(n for n in os.listdir(frames_dir) if n.startswith("bp-"))
    if not names:
        return {}
    frames: List[Tuple[float, bytes]] = []
    for i, name in enumerate(names, start=1):
        data = _frame_bytes(os.path.join(frames_dir, name))
        if data:
            frames.append((float(i), data))
    if not frames:
        return {}

    heroes = sorted({h for lane in HEROES_BY_LANE.values() for h in lane})
    try:
        from ..adapters.honor_of_kings import BP_ACTION_LABELS, BP_PHASE_LABELS
        actions = sorted(BP_ACTION_LABELS.values())
        phases = sorted(BP_PHASE_LABELS.values())
    except Exception:                              # noqa: BLE001
        actions, phases = [], []
    prompt = (
        f"下面是同一局《王者荣耀》BP 阶段的 {len(frames)} 帧画面（按时间先后）。\n"
        + "\n".join(f"第 {i} 帧" for i in range(1, len(frames) + 1))
        + "\n\n请按时间顺序读出 BP 里发生的每一手，以及最终阵容缺了什么：\n"
        + '{"steps": [{"side": "my|opp", "phase": "禁用|选人",'
        + ' "action": "操作类别", "hero": "英雄名"}],'
        + ' "my_missing_frontline": true|false,'
        + ' "my_missing_engage": true|false,'
        + ' "my_missing_protect": true|false,'
        + ' "my_missing_damage": true|false,'
        + ' "opp_missing_frontline": true|false}\n'
        + f"\n可选英雄：{'、'.join(heroes)}\n"
        + f"可选操作类别：{'、'.join(actions)}\n"
        + f"可选环节：{'、'.join(phases)}\n"
        + "\n规则：只写画面上**确实发生**的手；读不出来就别写；"
        "英雄名必须来自上面的清单。"
    )
    blocks = [vlm_client.text_block(prompt)]
    for _t, data in frames:
        blocks.append(vlm_client.image_block(data))
    try:
        result = vlm_client.call_model(
            model, blocks, creds=creds, system=SYSTEM_PROMPT,
            max_tokens=2000, timeout=240.0)
        answer = result.json()
    except vlm_client.VlmError:
        return {}
    if not isinstance(answer, dict):
        return {}

    steps = answer.get("steps")
    known = set(heroes)
    decisions: List[Dict[str, Any]] = []
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            hero = str(step.get("hero") or "").strip()
            if hero not in known:
                continue      # 不在清单里的英雄：认错了，丢掉
            decisions.append({
                "side": "opp" if str(step.get("side") or "my").strip() == "opp" else "my",
                "phase": str(step.get("phase") or "").strip(),
                "action": str(step.get("action") or "").strip(),
                "hero": hero,
            })
    out: Dict[str, Any] = {}
    if decisions:
        out["decisions"] = decisions
    for key in ("my_missing_frontline", "my_missing_engage", "my_missing_protect",
                "my_missing_damage", "opp_missing_frontline"):
        if key in answer and isinstance(answer[key], bool):
            out[key] = answer[key]
    return out
