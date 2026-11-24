"""**对局级事实**：这一局是谁、打的哪条线、赢没赢、BP 选了谁。

为什么是单独一层
----------------
「玩家做了什么」由 :mod:`app.recognition2.reader` **逐段**读；
而「这一局是谁」「什么结果」是**对局级**的 —— 读一次就够。

而且**不能靠模型在整帧里猜**
----------------------------
实测（2026-10-29，赵云那局）：让模型在整帧里从 130 个英雄名里挑，
同一段素材问三次，答出过 上官婉儿 / 云中君 / 澜 / 宫本武藏 ——
**全是猜**。猜错英雄比没有英雄更糟：英雄层规则（甄姬、云中君的专属打法）
会拿它去套**错对象**，而报告里看不出哪里错了。

分工：每一样事实都有它的依据
----------------------------
======================  ==================================================
事实                    怎么读
======================  ==================================================
**英雄**                选人界面那一段的画面上**印着的字**（裁决层定）
                        —— :mod:`app.recognition2.hero_read`
**分路**                **从英雄身份查表**（赵云→打野）
                        —— :func:`app.adapters.honor_of_kings.infer_position_from_hero`
**胜负**                结算画面 —— 模型读
**地图位置**            小地图/画面 —— 模型读
**BP 禁用与选人**       对局开始**之前**的那几帧 —— 模型读
======================  ==================================================

读不出来就是读不出来 —— **不猜**。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: 「只看画面，只描述看到的」这一层用的系统提示。
#:
#: 与 v1 那份的差别：v1 那份是为**逐窗动作识别**写的（一整套动作码词表、
#: 点单机制、动作清单），这里只需要模型答两三个字段，用不着那一套。
#:
#: 但两条硬规矩留着 —— **看不清就写 unknown**、**不许编造**。
SYSTEM_PROMPT = (
    "你是一个王者荣耀对局复盘助手。你**只看画面**，只描述你看到的东西。\n"
    "硬规矩：\n"
    "1. 看不清的字段一律写 unknown，不要猜、不要美化。\n"
    "2. 不许编造英雄名字、不许编造击杀数、不许提语音播报（录屏没有声音）。\n"
    "3. 只在**真有画面依据**的时候才下结论。\n"
    "4. **不要为了给答案而猜** —— 写「看不清」比编一个像模像样的结论有用得多。"
)


def _frame_bytes(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError:
        return None


def _grab(video, base_dir: str, times: Sequence[float], width: int = 768
          ) -> List[Tuple[float, bytes]]:
    """抽几帧并压成 JPEG 字节（复用采集层的开视频/裁画面逻辑）。

    画面归一（转正 + 裁黑边）**必须**与识别层用同一套 ——
    不然这里读到的是歪的画面，认英雄就认不出来了。
    """
    from app.collection import video_upload as vu

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


def _map_position_values() -> List[str]:
    """地图位置的**规范值**（从适配器取，**不另编一套**）。"""
    try:
        from app.adapters.honor_of_kings import MAP_POSITION_VALUES
        return list(MAP_POSITION_VALUES)
    except Exception:                                  # noqa: BLE001
        return ["野区", "河道", "中路", "对抗路", "发育路", "泉水", "unknown"]


#: 位置码 → 画面上的样子（写给模型看的解释）。
_MAP_POSITION_HINT = {
    "泉水": "刚复活在基地",
    "野区": "在野怪区打怪",
    "河道": "在中间的河道上",
    "中路": "在中间那条路",
    "对抗路": "上路或下路",
    "发育路": "发育路那条（射手路）",
    "unknown": "看不出来",
}


def _map_position_lines() -> List[str]:
    """提示词里那份位置清单。

    **清单从适配器取**（:func:`_map_position_values`）——
    两份各写一遍，迟早只改一处，于是模型答的归一失败 → ``unknown`` →
    报告里「当时局面」永远是「未知」，而界面上看不出是哪里对不上。
    """
    return ["  %s = %s" % (v, _MAP_POSITION_HINT.get(v, "画面上的那块区域"))
            for v in _map_position_values()]


def _normalize_map_position(value: str) -> str:
    """把模型给的**地图位置**归一化。

    为什么单独一个函数
    ------------------
    2026-10-29 之前，这里有个归一**分路**的函数干同样的事，但两件事被混
    成一件，后果是：模型从画面答「野区」（它看到的是**地图位置**），
    被当成分路去归一；归一失败 → ``unknown`` → 84 条规则全不触发。
    更糟的是**同一个问题模型每次答得不一样**：赵云那局先后答过 ``jungle``
    （对）和 ``side``（错）—— 因为「打野」这件事**从画面本就看不准**
    （打野前期也在边路出现）。

    现在分工明确：
    - **地图位置**（野区/河道/中路）→ 模型读画面，可靠；
    - **分路**（打野/中路/…）→ 从**英雄身份**推断，可靠。
    """
    text = (value or "").strip()
    if not text:
        return "unknown"
    try:
        from app.adapters.honor_of_kings import MAP_POSITION_VALUES
    except Exception:                                  # noqa: BLE001
        return "unknown"
    if text in MAP_POSITION_VALUES:
        return text
    # 模型偶尔答「己方野区」「上半野区」这类带修饰的。
    # 只认**完整包含**一个合法位置词的情形，不做逐字拼接。
    for v in MAP_POSITION_VALUES:
        if v == "unknown":
            continue
        if v in text:
            return v
    return "unknown"


#: **读对局级事实失败**时的留痕前缀。
#:
#: 为什么不用「模型没调通」：那个标记是**逐段**失败用的（读画面那一步），
#: 报告层认它，并写成「模型接口调不通，有 N 段没读出来」。
#: 这里失败的是**读胜负/地图位置**那一次调用 —— 逐段画面读得好好的，
#: 混用同一个标记，报告就会说错「有几段没读出来」。
FAIL_MARK = "读对局级事实没调通"


def _read_result_failed(why: str) -> Dict[str, Any]:
    """读不出胜负时的返回值（**统一一个说法**）。"""
    return {"map_position": "unknown", "result": "unknown",
            "why": "%s：%s" % (FAIL_MARK, why)}


def read_result(video, base_dir: str, start_t: Optional[float] = None,
                duration_sec: float = 0.0) -> Dict[str, Any]:
    """读「这局赢没赢」+「玩家大部分时间在哪块区域」。

    读不出来就是 ``unknown`` —— **不猜**。猜错胜负会让报告在复盘里
    说反话；猜错区域会让「当时局面」指错地方。

    读不出来时 ``why`` 里带上**原因**（人话，不是错误码）——
    报告顶上的结果栏因此会空着，玩家得知道为什么空。
    """
    from app.collection import settings, vlm_client

    creds = settings.credentials_for("vlm")
    if creds is None:
        return _read_result_failed("没有配置「看画面」的模型")
    model = settings.model_name("vlm") or creds.model

    t0 = float(start_t or 0.0)
    # 开局几帧（判在哪块区域）+ 结尾几帧（判结算画面）
    early = [t0 + 20.0 + 40.0 * i for i in range(4)]
    late = [max(0.0, duration_sec - 12.0 - 6.0 * i) for i in range(3)]
    frames = _grab(video, base_dir, early + late)
    if not frames:
        return _read_result_failed("抽不到画面")

    lines = [f"第 {i} 帧 = 视频第 {t:.1f} 秒"
             + ("（接近录屏结尾，可能是结算画面）" if i > 4 else "（对局刚开始）")
             for i, (t, _b) in enumerate(frames, start=1)]
    prompt = (
        "下面是同一局《王者荣耀》**玩家第一视角**录屏里的几帧：前 4 帧是开局附近，"
        "后 3 帧接近录屏结尾（可能是结算画面）。\n"
        + "\n".join(lines)
        + "\n\n只回答一个 JSON（不要有别的文字）：\n"
        + '{"where": "<玩家大部分时间在哪个区域>",'
        + ' "result": "<win|lose|unknown>",'
        + ' "why": "<一句话依据（比如「一直在中路」「结算画面写着失败」）"}\n'
        + "\nwhere 取值只有这几个，**原样抄一个，不要写别的**：\n"
        + "\n".join(_map_position_lines()) + "\n"
        + "判断依据：看玩家**大多数时间在哪个区域活动**。\n"
        + "规则：result 只能是 win / lose / unknown；"
        "看不出胜负就写 unknown，不要猜。\n"
        + "**不要回答玩家玩的是哪个英雄** —— 英雄由另一条链路"
        "（局内右下角的技能按钮）单独读，这一项在这里答了也是猜。\n"
        + "**也不要说玩家打哪条分路**（打野/中路/对抗路/发育路/游走）—— "
        + "分路由英雄身份推断，比看画面猜准得多。"
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
        # 原因要翻成**人话**（"模型没有返回内容"）而不是错误码（"empty"）——
        # 这句话最后会出现在报告顶上，玩家看到的是它。
        return _read_result_failed(vlm_client.explain_error(exc))
    if not isinstance(answer, dict):
        return _read_result_failed("模型回答不是对象")
    where = _normalize_map_position(str(answer.get("where") or ""))
    res = str(answer.get("result") or "").strip().lower()
    if res not in ("win", "lose"):
        res = "unknown"
    return {"map_position": where, "result": res,
            "why": str(answer.get("why") or "").strip()}


def read_bp(video, base_dir: str) -> Dict[str, Any]:
    """从 BP 阶段的画面读出「谁禁了谁、谁选了谁」+ 阵容缺了什么。

    BP 帧在抽帧那一步抽好了（``<video_id>_frames/bp-*.jpg``）。
    读不出来的**整条不写** —— 宁可少一条，也不要编一条禁用。

    ⚠️ **这份记录里的英雄名是模型「看头像」认的，不是画面上印着的字**
    ------------------------------------------------------------------
    2026-11-13 核对了 ``hok-video-8178dc031675`` 的选人帧：
    选人界面每一行只印**昵称、分路与「预选」状态**；英雄名只出现在
    右边那个**候选英雄网格**、以及鼠标悬停时的英雄资料卡上 ——
    而网格里那个名字**不代表玩家选了谁**。

    实测后果：同一份 BP 帧读出来的一队是「司空震 / 廉颇 / 姬小满 / 公孙离」，
    而局内战绩面板上印着的是「吕布 / 马超 / 妲己 / 敖隐 / 西施」——
    **五手全错**。凭美术认英雄这条链本项目实测 10 次没有一次对
    （README §32）。

    所以这份记录**只是线索**：返回的 ``source`` 标成 ``"model"``，
    裁决层（:mod:`app.recognition2.pipeline`）据此决定「不作数」还是
    「只当参考」—— 不能让它冒充「玩家确认过的 BP 记录」。

    返回 ``{"decisions": [...], "source": "model", ...}``。
    """
    from app.collection import settings, vlm_client
    from app.adapters.honor_of_kings import HEROES_BY_LANE

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
        from app.adapters.honor_of_kings import BP_ACTION_LABELS, BP_PHASE_LABELS
        actions = sorted(BP_ACTION_LABELS.values())
        phases = sorted(BP_PHASE_LABELS.values())
    except Exception:                              # noqa: BLE001
        actions, phases = [], []
    prompt = (
        f"下面是同一局《王者荣耀》BP 阶段的 {len(frames)} 帧画面（按时间先后）。\n"
        + "\n".join(f"第 {i} 帧" for i in range(1, len(frames) + 1))
        + "\n\n请按时间顺序读出 BP 里发生的每一手，以及最终阵容缺了什么：\n"
        + '{"steps": [{"side": "my|opp", "phase": "禁用|选人",'
        + ' "action": "操作类别", "hero": "英雄名", '
        + '"quote": "你照着画面抄下来的、能证明这一手是谁的那行字"}],'
        + ' "my_missing_frontline": true|false,'
        + ' "my_missing_engage": true|false,'
        + ' "my_missing_protect": true|false,'
        + ' "my_missing_damage": true|false,'
        + ' "opp_missing_frontline": true|false}\n'
        + f"\n可选英雄：{'、'.join(heroes)}\n"
        + f"可选操作类别：{'、'.join(actions)}\n"
        + f"可选环节：{'、'.join(phases)}\n"
        + "\n规则：只写画面上**确实发生**的手；读不出来就别写；"
        "英雄名必须来自上面的清单。\n"
        "⚠️ **只采信画面上印着的字**（2026-11-13 核对过这一屏）：\n"
        "* 选人界面左边那 5 行**只印昵称、分路与「预选」状态**，"
        "通常**不印英雄名**；\n"
        "* 右边那张**候选英雄网格**里的名字是「可以选谁」，"
        "悬停弹出的**英雄资料卡**同理 —— **都不代表谁选了它**；\n"
        "* 所以：**读不到印着的名字，这一手就别写**（宁可少一条）。"
        "凭头像认出来的名字会被当成记录，而它十次有十次是错的。\n"
        "* quote 写你真的读到的那行字；找不到这样的字就不要写这一手。"
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
                # **这个英雄名是模型看头像认的** —— 下游据此决定要不要采信
                # （见 :func:`read_bp` 的说明）。
                "source": "model",
            })
    out: Dict[str, Any] = {}
    if decisions:
        out["decisions"] = decisions
        #: 整份记录的来源：``model`` = 模型看 BP 帧认的（可能认错）。
        #: 玩家在 BP 记录页手填的那份**没有这个键**。
        out["source"] = "model"
    for key in ("my_missing_frontline", "my_missing_engage", "my_missing_protect",
                "my_missing_damage", "opp_missing_frontline"):
        if key in answer and isinstance(answer[key], bool):
            out[key] = answer[key]
    return out


def resolve_match_facts(video, base_dir: str, start_t: Optional[float] = None,
                        duration_sec: float = 0.0) -> Dict[str, Any]:
    """这一局的**对局级事实** —— 报告里「这一局是按什么定的」靠它。

    这里给的是**胜负、地图位置、BP 记录**三样。

    **英雄与分路不在这里**
    ----------------------
    原来英雄是从「局内技能按钮 → 裁图 → 像素比对」那条链读的
    —— **那条链 2026-11-03 退役，2026-11-04 连代码一起删了**：
    冷却数字叠在图标上，美术被压掉一大半；前两名只差 0.0119（纯噪声）；
    而且它分不清「玩家自己」和「队友」。

    现在英雄由**裁决层**（:mod:`app.recognition2.pipeline`）定：
    先看选人界面那一段**读画面上的字**，再拿 BP 记录的禁用名单
    与我方选人名单**否掉不可能的读数**。分路仍从英雄身份查表。

    **分路从英雄身份推断**，不从画面猜 ——
    「打野」这件事从画面本就看不准（打野前期也在边路出现），
    而英雄认出来之后，分路是可以查表定的：有依据、不猜、每次一样。
    """
    out: Dict[str, Any] = {
        "position": "",
        "position_source": "",
        "hero": "",
        "hero_source": "",
        "hero_confidence": 0.0,
        "hero_candidates": [],
        "hero_types": [],
        "map_position": "",
        "result": "",
        "why": "",
    }

    result = read_result(video, base_dir, start_t, duration_sec)
    if isinstance(result, dict):
        out["map_position"] = str(result.get("map_position") or "")
        out["result"] = str(result.get("result") or "")
        out["why"] = str(result.get("why") or "")

    # **BP 记录也一起读**（2026-11-03 新增）。
    #
    # 为什么在这读：认英雄要拿它**否掉不可能的读数** ——
    # 「这一局被禁用的英雄」不可能是玩家自己用的那个
    # （实测：看完整局的模型说了一个我方禁用的英雄）。
    # 裁决层（:mod:`app.recognition2.pipeline`）要这个名单。
    if start_t:
        try:
            from app.collection import video_upload as vu

            vu.extract_bp_frames(video, base_dir, float(start_t))
            bp = read_bp(video, base_dir)
            if bp:
                out["bp"] = bp
        except Exception:                              # noqa: BLE001
            pass
    return out
