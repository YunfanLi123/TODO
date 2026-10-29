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
from collections import Counter
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


def normalize_position(raw: Any) -> str:
    """把模型答的分路写法归成规范值（打野 / 对抗路 / 野区 都认）。

    提示词里列的是英文码，但模型常按中文答（「对抗路」「野区」）。
    早先这里用 ``pos not in position_values()`` 做严格比对，中文答案
    一律落成 unknown —— 分路一空，这一局所有带位置的规则都不触发，
    报告于是整片空白。词表只管提示词怎么写，解析必须走适配器的
    别名表（与玩家自己填分路时同一套口径）。
    """
    try:
        from ..adapters.honor_of_kings import normalize_position as _norm
        return _norm(raw)
    except Exception:                              # noqa: BLE001
        return "unknown"


def _frame_bytes(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError:
        return None


def _grab(video, base_dir: str, times: Sequence[float], prefix: str,
          width: int = 1280) -> List[Tuple[float, bytes]]:
    """抽几帧并压成 JPEG 字节（复用采集层的开视频/裁画面逻辑）。

    ``width`` 默认 1280 而**不是**统一压到 768：认英雄靠的是屏幕中央那个
    小人，768 会把 1080p 里的 67 像素压回 40 像素上下 —— 明明录得很清楚，
    却在送进模型前自己先把信息丢了（960×432 的素材压到 768×345 更是雪上加霜）。
    只有小到 1280 的图才不放大，原样送出。
    """
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


def _prompt_for(label: str, times: Sequence[float],
                position_choices: Sequence[str], heroes: Sequence[str]) -> str:
    lines = [f"第 {i} 帧 = 视频第 {t:.1f} 秒" for i, t in enumerate(times, start=1)]
    return (
        f"下面是同一局《王者荣耀》**玩家第一视角**录屏里「{label}」的几帧画面。\n"
        + "\n".join(lines)
        + "\n\n只回答一个 JSON：\n"
        + '{"position": "玩家主要打哪条路",'
        + ' "result": "win|lose|unknown",'
        + ' "hero": "玩家这一局自己操控的英雄名",'
        + ' "why": "一句话依据"}\n'
        + "\nposition 只能从这些里选："
        + "、".join(position_choices) + "、unknown。\n"
        + f"hero 只能从下面的英雄名里**原样**选一个：\n{'、'.join(heroes)}\n"
        + "认玩家自己英雄的依据：这是第一视角录屏，自己的英雄整局在屏幕中央附近、"
        "右下角是它的技能轮盘；别把队友或敌方英雄当成玩家英雄。\n"
        + "**看不清就写 unknown，不要从技能特效或常见搭配去推测。**\n"
        + "规则：三项都只能从给定取值里选；哪一项看不出来就写 unknown"
        "（hero 看不出来写 unknown）。"
    )


def _vote(values: Sequence[str], min_votes: int = 1) -> str:
    """多组帧投票：过半且达到最小独立票数才采纳，否则留空。

    同一帧连问两次能得到两个不同英雄（实测），所以「只读一次」等于
    把一次猜测当成事实。改成几组**不同时刻**的画面各问一次：
    过半一致才认，各说各的就当认不出来。
    """
    vals = [v for v in values if v]
    if not vals:
        return ""
    top, count = Counter(vals).most_common(1)[0]
    return top if count >= min_votes and count * 2 > len(vals) else ""


def infer_lane_from_track(rows: Sequence[Dict[str, Any]], start_t: float,
                          early_minutes: float = 4.0) -> Dict[str, Any]:
    """从开局小地图轨迹给出可解释的分路候选。

    分路是整局级事实，不能让一两张全屏截图决定。这里仅在轨迹覆盖足够且
    某条路有明显多数停留时给结论；游走这种天然跨区的职责仍交给 VLM 复核。
    返回值刻意带计数，便于报告和回归集检查依据。
    """
    from . import minimap

    end_t = float(start_t) + early_minutes * 60.0
    # 只有中路对角线在红/蓝方 180° 翻转后仍保持不变；边路与野区必须
    # 先可靠识别阵营和野区框，不能用「离哪条龙近」把野区硬归成边路。
    counts = {"mid": 0, "other": 0}
    seen = 0
    for row in rows or []:
        try:
            t = float(row.get("t"))
            nx, ny = float(row.get("nx")), float(row.get("ny"))
        except (TypeError, ValueError):
            continue
        if t < float(start_t) or t > end_t:
            continue
        seen += 1
        if minimap._near_mid_diag(nx, ny):
            counts["mid"] += 1
        else:
            counts["other"] += 1

    if seen < 8:
        return {"position": "unknown", "confidence": 0.0,
                "reason": "开局小地图轨迹样本不足", "counts": counts}
    confidence = counts["mid"] / float(seen)
    # 轨迹只能安全地给中路正证据；其他分路交给 VLM 的画面判断，避免
    # 野区点被错误地按距离归到对抗/发育路。
    if confidence < 0.60:
        return {"position": "unknown", "confidence": round(confidence, 3),
                "reason": "轨迹未形成稳定中路证据，交给画面复核", "counts": counts}
    return {"position": "mid", "confidence": round(confidence, 3),
            "reason": "开局轨迹多数停留在中路对角线", "counts": counts}


def read_lane_and_result(video, base_dir: str, start_t: Optional[float] = None,
                          duration_sec: float = 0.0,
                          position_hint: str = "") -> Dict[str, Any]:
    """读出「玩家打什么位置」「这局赢没赢」「玩家自己玩的是哪个英雄」。

    读不出来就是 ``unknown`` / 空串 —— **不猜**。

    英雄这一项必须读：英雄层规则（如甄姬、云中君的专属打法）的触发条件是
    ``position + my_hero + 时间``，分路认对了但英雄空着，那些规则一条都不会触发，
    报告只剩分路通用规则。

    **为什么问几组、还要投票**
    ------------------------
    早先只把 7 帧（开局 4 + 结尾 3）一次性问一遍，而模型的默认采样温度不为 0：
    同一帧连问四次会给出韩信 / 哪吒 / 孙策 / 曜 四个答案，同一段视频跑三遍
    得到亚连 / 马超 / 李白 —— 那其实是「把一次随机猜测当成了事实」，再被逐回合
    复制 7 份，看着还挺稳定。现在拆成开局 / 中段 / 后段几组分别问、温度固定为 0，
    过半一致才采纳。
    """
    from . import settings
    from ..adapters.honor_of_kings import HEROES_BY_LANE

    creds = settings.credentials_for("vlm")
    if creds is None:
        return {"position": "unknown", "result": "unknown", "hero": "",
                "reason": "没有配置「看画面」的模型"}
    model = settings.model_name("vlm") or creds.model

    # 英雄白名单：从适配器的分路英雄表汇总，和规则库 / BP 识别用**同一份**，
    # 不另编一套。模型答的英雄不在表里就当认错丢弃 ——
    # 看错英雄会让英雄层规则对错对象下结论，比留空更糟。
    known_heroes = sorted({
        h for lane in HEROES_BY_LANE.values() for h in lane
    })

    t0 = float(start_t or 0.0)
    dur = float(duration_sec or 0.0)
    # 英雄名通常只在开局/技能栏附近短暂可辨。四张稀疏全屏图经常刚好都
    # 没拍到名字，因此提高到六张、缩短间隔；仍分组投票，避免单帧认错。
    early_t = [t0 + 16.0 + 28.0 * i for i in range(6)]
    groups: List[Tuple[str, List[float]]] = [("开局", early_t)]
    # 中段 / 后段只在视频够长、且与上一组不重叠时才加 —— 否则同一批帧问几遍，
    # 投出来的「一致」是假的（同一张图当然是同一个答案）。
    if dur > early_t[-1] + 120.0:
        mid_t = [dur * p for p in (0.36, 0.42, 0.48, 0.54, 0.60)]
        if mid_t[0] > early_t[-1] + 30.0:
            groups.append(("中段", mid_t))
            if dur > mid_t[-1] + 120.0:
                # 胜负只在结算 / 水晶爆炸后的尾段可靠可见。以前取到的是
                # “后期对战”的普通画面，和开局、中段一起投票必然多数 unknown。
                late_t = [dur - offset for offset in (3.0, 7.0, 12.0, 18.0, 26.0)]
                if min(late_t) > mid_t[-1] + 30.0:
                    groups.append(("结算尾段", late_t))

    answers: List[Tuple[str, Dict[str, Any]]] = []
    for label, times in groups:
        frames = _grab(video, base_dir, times, "prefill")
        if not frames:
            continue
        blocks = [vlm_client.text_block(
            _prompt_for(label, [t for t, _b in frames],
                        position_values(), known_heroes))]
        blocks.extend(vlm_client.image_block(data) for _t, data in frames)
        try:
            result = vlm_client.call_model(
                model, blocks, creds=creds, system=SYSTEM_PROMPT,
                max_tokens=800, timeout=180.0, temperature=0.0)
            answer = result.json()
        except vlm_client.VlmError:
            continue
        if isinstance(answer, dict):
            answers.append((label, answer))

    if not answers:
        return {"position": "unknown", "result": "unknown", "hero": "",
                "reason": "模型没调通或回答不是对象"}

    def _hero_of(ans: Dict[str, Any]) -> str:
        hero = str(ans.get("hero") or "").strip()
        return hero if hero in known_heroes else ""   # 不在清单里 = 认错了

    pos = _vote([normalize_position(a.get("position")) for _l, a in answers])
    # 英雄误判会把整套英雄专属规则套到错误对象上，代价远高于留空。
    # 一组帧只是同一次模型判断，不能伪装成「多数票」。
    hero = _vote([_hero_of(a) for _l, a in answers], min_votes=2)
    # 胜负只能以结算尾段为证据。开局和中段即使模型给了 win/lose 也不采信：
    # 那是把局势顺逆误当最终结果，最容易造成“看起来识别出来、实际写反”。
    res = _vote([
        str(a.get("result") or "").strip().lower()
        for label, a in answers
        if label == "结算尾段"
        and str(a.get("result") or "").strip().lower() in ("win", "lose")
    ])
    # 票型留在返回值里：报告与排查都要能看出「这次是真的读到了，还是碰巧一致」。
    votes = {label: {"position": normalize_position(a.get("position")),
                     "hero": _hero_of(a), "result": str(a.get("result") or "")}
             for label, a in answers}
    why = "；".join(f"{lb}: {str(a.get('why') or '').strip()[:60]}"
                    for lb, a in answers)
    # 轨迹是跨帧的客观证据，置信度达标时比语言模型看几张画面更可靠。
    # 它只覆盖分路，不影响英雄、胜负仍由各自的视觉证据读取。
    if position_hint in position_values():
        pos = position_hint
    return {"position": pos or "unknown", "result": res or "unknown",
            "hero": hero, "why": why, "votes": votes,
            "position_source": "track" if position_hint in position_values() else "vlm"}


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
            max_tokens=2000, timeout=240.0, temperature=0.0)
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
