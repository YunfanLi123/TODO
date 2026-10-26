"""视频上传的 API 层与页面编排。

完整链路：

    上传视频 → 回合门控扫描 → 逐回合标注 → 生成对局 → 分析 → 网页报告

分工说明（这一层存在的理由）
------------------------
系统从画面里能确定的只有「回合怎么切」和「血条多长」。玩家的**操作**
在视觉上几乎不可辨 —— 存钱就是没点任何按钮，追星是点升级再点两次，
两者在几帧画面里几乎没有区别。所以让模型去猜动作，猜错就会得出
「你该存钱」这种颠倒结论，比不判更糟。

于是：系统干活（切回合、读血条、抽帧），玩家干活（勾动作、补数字）。
原本手工录 20 局要几小时，这样降到十几分钟。
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from . import scan_jobs
from . import realign
from . import video_upload as vu
from .video import RecognitionResult, RecognizedRound, to_raw_match
from ..adapters.base import get_adapter
from ..db import session_scope
from ..report.views import render_page


def db_session():
    """FastAPI 依赖：每个请求一个会话，结束自动提交/回滚。"""
    with session_scope() as s:
        yield s


def _register_session(job, base_dir: str) -> None:
    """扫描完成后把结果挂到「建对局」可用的会话上。

    在后台线程里跑，但只写内存字典 —— 不碰数据库，不涉及跨线程 session。

    **不检查 ``job.status``**：扫描任务在调本函数之后才标 ``DONE``
    （先挂会话、再标完成，前端一看到 done 就来建对局，
    顺序反了会拿到「解析结果已失效」）。到这里说明扫描已经成功了，
    有摘要就挂上。
    """
    if not job.summary:
        return
    shots = job.summary["rounds"]
    # 分析单元是**时间窗**：窗从规则库倒推（每条规则的 time_min between）。
    # 窗内做一次综合分析，窗里的规则共用这份结论；时间片降级成证据。
    windows = job.summary.get("windows") or None
    skeleton = vu.build_skeleton(
        job.video, shots, job.game, job.game_version, job.match_id,
        windows=windows,
    )
    _SESSIONS[job.match_id] = {
        "summary": job.summary,
        "skeleton": skeleton,
        "adapter": get_adapter(job.game),
        # 视频本身要留着：玩家校正对局起点时，要按新起点**重读**一遍
        # （重新抽帧 + 重新识别），光有摘要不够 —— 得能再打开视频。
        "video": job.video,
        # 模型读出来的整体情况（分路/胜负/BP）—— 建对局时直接用。
        "prefill": (job.summary or {}).get("prefill") or {},
    }


#: 扫描任务缓存。进程内保存「视频 → 扫描结果」，标注页要用。
#: 用内存而不是数据库：这是一次性的中转状态，进程重启后重新上传即可。
_SESSIONS: Dict[str, Dict[str, Any]] = {}


class RoundAnnotation(BaseModel):
    """一段（时间窗）的人工标注。"""

    round_no: int
    #: 所属**分析段**的键。
    #:
    #: 玩家是按段答的（对着整段画面做一次综合判断），
    #: 而判定单元是规则原窗 —— 段里的所有窗共用这份答案。
    #: 留空则退回到按 ``round_no`` 逐条对应（老协议）。
    segment_key: str = ""
    #: 局面事实。留空表示没填 —— 内核会自动当成「无法判定」，
    #: 而不是当成 0 或默认值。
    facts: Dict[str, Any] = Field(default_factory=dict)
    #: 玩家动作。格式同 RecognitionInput 的 actions。
    actions: List[Dict[str, Any]] = Field(default_factory=list)
    #: 玩家对这一回合的补充说明，会带进报告
    note: str = ""
    #: 这一帧玩家没看懂（画面糊、视角不在自己身上、时机尴尬…）。
    #:
    #: 留这个开关是因为「我看不清这帧」和「这帧我什么都没做」是**两件事**，
    #: 而协议原本只能表达后者。混为一谈的后果很直接：内核把「没勾动作」
    #: 判成「该做而没做」，于是报告里出现一条玩家从没下过的结论 ——
    #: 看上去是系统在给建议，其实是在编。
    #:
    #: 标了跳过的回合，动作事实会被记成缺失，内核判「无法判定」。
    skipped: bool = False


class AnnotateInput(BaseModel):
    match_id: str
    game: str
    game_version: str
    player: Dict[str, Any] = Field(default_factory=dict)
    outcome: Dict[str, Any] = Field(default_factory=dict)
    played_at: str = ""
    #: 动作标注的来源。
    #:
    #: ``player``  = 玩家自己在标注页上勾的（正常路径）
    #: ``derived`` = 从画面反推出来的（**只有验证链路会这么用**）
    #:
    #: 这两种来源的结论可信度完全不同：前者是玩家的自述，后者是我
    #: 看着帧猜的。报告必须标出来 —— 否则用户打开报告会以为那些
    #: 「视频中做了什么」是玩家自己说的，那就是替玩家编话。
    annotation_source: str = "player"
    rounds: List[RoundAnnotation]
    #: 玩家改过的「对局从视频第几秒开始」。
    #:
    #: 标注页上**没有**逐段时间输入框了（玩家不逐段标），所以起点校正
    #: 没法在前端改输入框 —— 提交时把改后的起点带上来，服务端按它
    #: 重算每一段的时间窗。偏了几分钟，每条带时间窗的模式都会判到
    #: 错误的阶段，而报告上只是「没触发」。
    match_start_t: Optional[float] = None


def _mark_actions_missing(rnd, adapter, reason: str) -> None:
    """把这个回合的动作事实标成「缺失」。

    缺失的事实在内核里判成**无法判定**，而不是「该做而没做」——
    后者是在替玩家编一条他没下过的结论。

    写一条置信度 0 的 :class:`DetectedFact` 就够：``to_raw_match`` 会把
    未采信的事实收进 ``missing_facts``，同时把原因记进降级报告，
    报告里能看出「这一帧是玩家说看不懂，不是不合格」。
    """
    from .video import DetectedFact

    for kind in sorted(adapter.known_action_facts()):
        if any(f.name == kind for f in rnd.facts):
            continue
        rnd.facts.append(
            DetectedFact(
                name=kind,
                value=None,
                confidence=0.0,
                method="manual",
                failure=reason,
            )
        )


def _apply_annotation(rnd, adapter, ann) -> None:
    """把玩家对**一段**的标注写进该段里的每一个判定单元。

    人工填的数字一律按「用户填的」计置信度 1.0 —— 它比任何识别都可靠。
    没填的**不写进去**，让 to_raw_match 把它算作缺失，
    而不是塞一个 0 或默认值。

    但**读时钟读出来的**局内时间要留下：它不是用户填的，
    冲掉之后每一段都变成「第 0 分钟」，所有带时间窗的模式
    一条都不触发 —— 报告看起来只是「没结论」，看不出是哪里错了。
    """
    from .video import DetectedAction, DetectedFact

    clock_facts = [f for f in rnd.facts if f.method != "manual"]
    rnd.facts.clear()
    rnd.facts.extend(clock_facts)
    for name, value in ann.facts.items():
        # 用户填的局内时间优先于读出来的 —— 他要是手填了，说明
        # 他知道自己那局的时间，不该被机器的读数盖掉。
        if any(f.name == name for f in clock_facts):
            rnd.facts = [f for f in rnd.facts if f.name != name]
        rnd.facts.append(
            DetectedFact(name=name, value=value, confidence=1.0, method="manual")
        )
    # 动作：**合并**而不是清空。
    #
    # 玩家勾了哪个动作种类，就用玩家的；玩家没勾的种类，
    # 保留识别器认出来的（画面里看出来的）。
    #
    # 为什么不能一上来就 clear：自动识别认出的动作是**证据**，
    # 玩家只补自己没看懂的 —— 把两边都清掉，报告里就只剩玩家勾的那几件，
    # 「你实际在做什么」就答不出来了。
    #
    # 现在标注页**不再逐段勾动作**了（玩家明确说不想标），
    # 这里的合并是给「玩家想推翻某一类」留的口子。
    if ann.skipped:
        # 「这段看不清」＝这一段的**识别结果**不算数。
        #
        # 动作和**动作事实**都要清：只清动作的话，那条动作事实还留着
        # （值是「刷野」），内核会判成「该做而没做」——
        # 而这一段的真实情况是**看不出来**，不是「没做」。
        #
        # 但玩家填的**局面事实**（分路等）要留着：
        # 「看不清」说的是这一段画面里的操作，不是「你打什么位置」。
        rnd.actions.clear()
        action_kinds = set(adapter.known_action_facts())
        rnd.facts = [f for f in rnd.facts
                     if f.name not in action_kinds or f.method == "manual"]
        _mark_actions_missing(rnd, adapter, "玩家标注为「这段看不清」")
        return
    if ann.actions:
        kinds = {str(a.get("kind") or "") for a in ann.actions}
        rnd.actions = [a for a in rnd.actions if a.kind not in kinds]
    for a in ann.actions:
        rnd.actions.append(DetectedAction(
            kind=a.get("kind", ""),
            value=a.get("value"),
            label=a.get("label", ""),
            confidence=float(a.get("confidence", 1.0)),
        ))


def _hud_layout_for(game: str) -> Dict[str, str]:
    """该游戏的 HUD 布局（从适配器取，不在采集层写死一份）。"""
    try:
        from ..adapters.base import get_adapter

        adapter = get_adapter(game)
    except Exception:                              # noqa: BLE001
        return {}
    return dict(getattr(adapter, "hud_layout", {}) or {})


def _apply_start_correction(state: Dict[str, Any], base_dir: str,
                            new_start: Optional[float]) -> Dict[str, Any]:
    """玩家改了「对局从视频第几秒开始」→ 按新起点把每一段**重读一遍**。

    改一个数不等于把算出来的结果平移一下。

    起点偏了多少，原来那批帧显示的局内时刻就偏了多少 —— 平移旧结论
    等于拿错时刻的画面去说那一段的事：方向全对，证据对不上，
    而且这种错在报告上完全看不出来。

    所以这里重新抽帧、重新识别：玩家给的数只决定**去看哪几秒**，
    看到什么一律重新读（见 :mod:`app.collection.realign`）。
    """
    summary = state.get("summary") or {}
    if not realign.needs_realign(summary, new_start):
        return state
    video = state.get("video")
    if video is None:
        # 没有视频就没法重读 —— 宁可什么都不做，也不平移旧结论。
        return state
    start = float(new_start)
    duration = float(getattr(video, "duration_sec", 0.0) or 0.0)
    if duration and not (0.0 <= start <= duration):
        raise HTTPException(
            status_code=422,
            detail=(f"对局起点要落在视频的第 0 ~ {int(duration)} 秒之间，"
                    f"你填的是第 {int(start)} 秒。"),
        )
    hero_track = ((summary.get("scan") or {}).get("hero_track") or [])
    windows, shots = realign.realign_windows(
        video, base_dir, summary, start, hero_track)
    if not windows:
        return state

    skeleton = vu.build_skeleton(
        video, shots, summary.get("game", ""),
        summary.get("game_version", ""), summary.get("match_id", ""),
        windows=windows,
    )
    old = summary.get("match_start_t")
    reason = summary.get("match_start_reason") or ""
    new_summary = dict(summary)
    new_summary["match_start_t"] = round(start, 1)
    new_summary["match_start_corrected"] = True
    new_summary["match_start_reason"] = (
        f"玩家手动校正：第 {int(old) if old is not None else 0} 秒 → "
        f"第 {int(start)} 秒，各段已按新起点重新读过一遍"
        + (f"（自动测得：{reason}）" if reason else "")
    )
    new_summary["windows"] = windows
    new_summary["rounds"] = shots
    state["summary"] = new_summary
    state["skeleton"] = skeleton
    return state


#: 「测试连接」用的那张小图：一个**纯红色**的方块。
#:
#: 为什么要带图：这是「这个模型能不能看画面」唯一可靠的判据 ——
#: DeepSeek 与百炼都有视觉型号，按供应方断定会错。
#: 答出「红」就说明它真的看见了。
_PROBE_COLOR = (40, 40, 220)          # BGR


def _probe_png() -> bytes:
    """64×64 的纯红方块（PNG 字节）。"""
    import cv2
    import numpy as np

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :] = _PROBE_COLOR
    ok, buf = cv2.imencode(".png", img)
    return bytes(buf) if ok else b""


#: 回答里出现这些词就算「它看见了红色」。
_COLOR_WORDS = ("红", "red", "朱", "赤")


def _looks_like_color(answer: str) -> bool:
    text = (answer or "").lower()
    return any(w in text for w in _COLOR_WORDS)


def _find_video_match(session: Session, video_id: str):
    """按视频 id 找已经建好的对局（幂等用）。

    同一个视频只该有一局对局：用户刷新页面、重试提交都走这里，
    不能建出第二条。
    """
    if not video_id:
        return None
    from ..collection import match as match_col

    for record in match_col.list_matches(session, player_id="local", limit=200):
        payload = record.get_payload() or {}
        rec = payload.get("recognition") or {}
        source = str((rec.get("stats") or {}).get("source") or "")
        if source and source == video_id:
            return record
    return None


def register_routes(app: FastAPI, base_dir: str) -> None:
    """把视频相关路由挂到 app 上。

    用函数而不是模块级装饰器，是因为要拿到 ``base_dir``（项目根）
    来定位视频存储目录。
    """

    # 首页由 app/report/home.py 提供。这里不再定义 —— 两处都注册的话，
    # 先注册的那个赢，而我不想让「哪个首页生效」取决于 import 顺序。

    # -------------------------------------------------------------- 前端配置
    @app.get("/api/config")
    def frontend_config() -> Dict[str, Any]:
        """给前端下发的运行配置。

        上传上限必须由后端下发。让前端写死字面量必然出事：上限从 500MB
        提到 4GB 时文案忘了改，页面还写着「最大 500 MB」，用户据此
        判断自己传不上去 —— 文案与实现不一致比没有文案更糟。
        """
        return {
            "max_upload_bytes": vu.MAX_UPLOAD_BYTES,
            "max_upload_mb": round(vu.MAX_UPLOAD_BYTES / 1024 ** 2),
            "allowed_suffixes": sorted(vu.ALLOWED_SUFFIXES),
            "temp_dir": vu.temp_dir_hint(base_dir),
            "sample_sec_default": vu.GATE_SAMPLE_SEC,
        }

    # ------------------------------------------------------------ 模型配置
    @app.get("/api/model/config")
    def model_config() -> Dict[str, Any]:
        """当前的模型配置（**密钥已掩码**）+ 供应方清单 + 限额。"""
        from . import model_presets, settings

        return {
            "slots": settings.display_slots(),
            "providers": model_presets.catalog(),
            "slot_labels": model_presets.SLOT_LABELS,
            "slot_hints": model_presets.SLOT_HINTS,
            "recognizer": settings.recognizer_kind(),
            "limits": settings.limits(),
            "status": settings.status(),
        }

    @app.post("/api/model/config")
    def save_model_config(payload: Dict[str, Any]) -> Dict[str, Any]:
        """保存模型配置。

        **只填 API Key 就能配好**：供应方、端点、协议、关思考的字段名
        都按供应方预设推出来，用户不用碰。
        密钥留空 = 不改动已存的那个（界面不回显明文，只能这样改）。
        """
        from . import settings

        body = payload or {}
        slot = str(body.get("slot") or "vlm")
        if slot not in settings.SLOTS:
            raise HTTPException(status_code=422, detail=f"没有这个槽位：{slot}")
        try:
            settings.save_slot(
                slot,
                provider=str(body.get("provider") or ""),
                api_key=str(body.get("api_key") or ""),
                model=str(body.get("model") or ""),
                endpoint=str(body.get("endpoint") or ""),
                protocol=str(body.get("protocol") or ""),
                can_see_images=body.get("can_see_images"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        if body.get("recognizer"):
            settings.set_recognizer(str(body.get("recognizer")))
        if body.get("limits"):
            settings.set_limits(**(body.get("limits") or {}))
        return {"ok": True, **model_config()}

    @app.post("/api/model/test")
    def test_model(payload: Dict[str, Any]) -> Dict[str, Any]:
        """测一下这个模型通不通、**能不能看图**。

        带一张 64px 的小图问颜色 —— 这是「能不能看画面」唯一可靠的判据
        （DeepSeek 与百炼都有视觉型号，按供应方断定会错）。

        **这一下不写盘**：测就只是测。用户可能测完就改主意，
        一个「测试连接」按钮顺手把配置改了，是会咬人的。
        测出来的结论（能不能看图）由前端暂着，等点「保存」时一起写进去。
        """
        from . import settings
        from .vlm_client import Credentials, VlmError, call_model, image_block, text_block

        body = payload or {}
        slot = str(body.get("slot") or "vlm")
        provider = str(body.get("provider") or "")
        model = str(body.get("model") or "")
        key = str(body.get("api_key") or "").strip()

        if provider:
            from . import model_presets
            resolved = model_presets.resolve(provider, model, slot)
            endpoint = str(body.get("endpoint") or "").strip() or resolved["endpoint"]
            protocol = str(body.get("protocol") or "").strip() or resolved["protocol"]
            thinking_field = resolved["thinking_field"]
            model = model or resolved["model"]
        else:
            saved = settings.slot_config(slot)
            endpoint = str(body.get("endpoint") or "").strip() or str(saved.get("endpoint") or "")
            protocol = str(body.get("protocol") or "").strip() or str(saved.get("protocol") or "openai")
            thinking_field = str(saved.get("thinking_field") or "thinking")
            model = model or str(saved.get("model") or "")
            key = key or str(saved.get("api_key") or "")
        # Key 框出于安全不回显，测试时常留空 —— 无论哪个供应商分支，
        # 留空都回落到已保存的 Key（保存接口本来就是「留空=不改」）。
        # 不做这步回落，选了「自定义」时点测试会被 422 打回，
        # 而前端只认 error 字段，最终显示成看不懂的「未知错误」。
        if not key:
            key = str(settings.slot_config(slot).get("api_key") or "")
        if not key:
            raise HTTPException(status_code=422, detail="先填 API Key")
        if not endpoint:
            raise HTTPException(status_code=422, detail="先选供应方")
        if not model:
            raise HTTPException(status_code=422, detail="先填模型名")

        creds = Credentials(api_key=key, endpoint=endpoint, protocol=protocol,
                            thinking_field=thinking_field, model=model,
                            source="test")
        started = time.time()
        try:
            result = call_model(
                model,
                [text_block("这张图是什么颜色？只回答颜色。"),
                 image_block(_probe_png())],
                creds=creds, max_tokens=64, timeout=60.0, retry=0)
        except VlmError as exc:
            return {"ok": False, "kind": exc.kind, "error": str(exc),
                    "detail": exc.detail[:400],
                    "elapsed_ms": int((time.time() - started) * 1000),
                    "can_see_images": False}
        elapsed = int((time.time() - started) * 1000)
        answer = (result.text or "").strip()
        can_see = _looks_like_color(answer)
        note = "" if can_see else "这个模型没认出颜色 —— 多半读不了画面"
        return {"ok": True, "answer": answer[:200], "elapsed_ms": elapsed,
                "can_see_images": can_see, "note": note, "usage": result.usage}

    # ------------------------------------------------------------ 视频上传
    @app.post("/api/videos/upload")
    async def upload_video(
        file: UploadFile = File(...),
        game: str = Form("hok"),
        session: Session = Depends(db_session),
    ) -> Dict[str, Any]:
        """上传视频并**立即返回**，扫描在后台跑。

        同步扫完再返回对 2GB 的录屏意味着几十秒的白屏等待，
        而且看不出是在跑还是卡了。所以这里只负责落盘 + 建任务，
        真正的扫描交给 :mod:`scan_jobs` 的后台线程。

        版本与扫描灵敏度都不由调用方给 —— 见下面两段解析。
        """
        # 这个产品只做王者荣耀。别的游戏没有「时间片」这套切法，
        # 扫出来的东西玩家没法用 —— 与其让它跑完再报错，不如上传这一秒就说清楚。
        if game != "hok":
            raise HTTPException(
                status_code=422,
                detail="目前只支持王者荣耀的对局录屏。",
            )

        # 版本取该游戏规则库里**最新**的那一个，不接受前端传值。
        #
        # 版本填错不会报错，只会让扫描完之后拿到的报告是空的 ——
        # 而那种错误在界面上完全看不出来：上传成功、扫描成功、
        # 报告页面也打开了，就是没有一条模式被检验。所以这个字段干脆不给填。
        #
        # 一个版本都没有就当场拒掉：让用户白等几分钟扫完视频再撞上
        # 「规则库为空」，不如上传这一秒就告诉他。
        from ..rules.store import latest_version

        version = latest_version(session, game)
        if not version:
            raise HTTPException(
                status_code=409,
                detail=f"{game} 还没有操作模式库，请先到操作模式库载入",
            )

        video_id = uuid.uuid4().hex[:12]
        match_id = f"{game}-video-{video_id}"

        try:
            # 流式落盘：边读边写，2GB 视频也不会整个进内存
            video = await vu.save_upload_stream(file, base_dir, video_id)
            video = vu.probe(video)
        except vu.VideoProcessingError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        # 扫描参数固定用调好的默认值。界面曾经给两个滑块（边界阈值 /
        # 最短回合间隔），但让人自己拖只会得到更差的切片：调高漏回合，
        # 调低把一次转场切成好几段 —— 而这种错误在报告里看不出来，
        # 只会让后面所有模式的判定都建立在错误的回合切分上。
        cfg = vu.ScanConfig()
        job = scan_jobs.RUNNER.create(
            uuid.uuid4().hex[:12], match_id, video, game, version
        )
        scan_jobs.RUNNER.run_async(
            job, base_dir, cfg,
            on_done=lambda j: _register_session(j, base_dir),
        )
        scan_jobs.RUNNER.cleanup()

        # **录屏规格当场核对**：横屏 + 游戏画面占满全屏（用户定的标准）。
        # 不符合就说清楚 —— 让用户白等 18 分钟扫完再知道这段录屏认不出来，
        # 比当场说一句话糟糕得多。
        spec = vu.quick_spec(video, hud_layout=_hud_layout_for(game))

        return {
            "job_id": job.job_id,
            "match_id": match_id,
            "video": video.to_dict(),
            "status_url": f"/api/videos/jobs/{job.job_id}",
            "spec": spec,
            "message": (
                f"上传完成（{video.size / 1024 ** 3:.2f} GB），"
                f"正在后台扫描 {video.duration_sec / 60:.1f} 分钟的视频…"
                + ("" if spec.get("ok")
                   else f"　⚠ 录屏规格：{spec['note']}")
                + ("" if (spec.get("recognition") or {}).get("ok")
                   else f"　⚠ {(spec.get('recognition') or {}).get('note', '')}")
            ),
        }

    # ------------------------------------------------------------ 任务状态
    @app.get("/api/videos/jobs/{job_id}")
    def job_status(job_id: str) -> Dict[str, Any]:
        """轮询这个接口拿扫描进度。

        进度是真实的 —— 后台线程每采一个时间点就更新一次，
        不是前端演的动画条。
        """
        job = scan_jobs.RUNNER.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="任务不存在或已过期")
        return job.to_dict()

    # ---------------------------------------------------------- 标注页（已撤）
    @app.get("/videos/{match_id}/annotate", response_class=HTMLResponse)
    def annotate_page(match_id: str) -> HTMLResponse:
        """这一页没有了。

        流程改成了「上传 → 模型读 → 直接出报告」，
        中间不再有让人标注的一步（旧书签会走到这里）。

        留这个路由只是为了给旧链接一句**说人话**的交代，
        而不是甩一个 404。
        """
        raise HTTPException(
            status_code=410,
            detail="流程已改成自动出报告：上传完直接出结论，不用再标注。"
                   "请从「上传」重新走一遍。",
        )

    # --------------------------------------- 扫描完 → 直接建对局、出报告
    @app.post("/api/videos/finalize")
    def finalize(payload: Dict[str, Any], session: Session = Depends(db_session)
                 ) -> Dict[str, Any]:
        """模型读完了 → 落库 → 分析 → 返回报告地址。

        **中间没有人工那一步**：分路、胜负、BP、每段操作都是模型从画面读的
        （用户原话：「我只要模型去读出报告，其余一切步骤都省去」）。

        模型读不出来的就记「读不出来」—— 报告照出，相关规则判不了，
        **不回头让人填**。
        """
        match_id = str((payload or {}).get("match_id") or "")
        state = _SESSIONS.get(match_id)
        if state is None:
            raise HTTPException(status_code=404,
                                detail="解析结果已失效，请重新上传视频")

        adapter = state["adapter"]
        summary = state.get("summary") or {}
        video = summary.get("video") or {}
        video_id = str(video.get("video_id") or "")

        # 幂等：这一局建过了就直接给地址（用户刷新/重试都走这里）。
        existing = _find_video_match(session, video_id)
        if existing is not None:
            report = _analyze(session, existing, adapter.game_id)
            return {
                "match_id": existing.id,
                "stats": report["stats"],
                "gap_count": len(report["gaps"]),
                "lanes": report["lanes"],
                "degradation": {},
                "issues": [],
                "report_url": f"/api/matches/{existing.id}/report",
            }

        prefill = state.get("prefill") or {}
        raw, degradation = to_raw_match(
            state["skeleton"], adapter.known_facts(), adapter.known_action_facts()
        )
        raw["match_id"] = match_id
        # 分路与胜负是**模型读出来的**；读不出来就留空
        # （留空 = 那些带分路条件的规则不触发，而不是「没做到」）。
        raw["player"] = {
            "position": str(prefill.get("position") or ""),
            "matchup": "unknown",
            "hero": str(prefill.get("hero") or ""),
        }
        raw["outcome"] = {"result": str(prefill.get("result") or "")}
        raw["played_at"] = ""
        # 动作是**模型从画面读出来的**，不是玩家自己勾的 ——
        # 报告据此说明来源（见 report.html 的 derived 提示）。
        raw["annotation_source"] = "derived"
        # **哪一个模型读出来的**：换模型就换结论，报告顶部要写出来。
        # 写进 ``recognition`` 是因为那一块本来就会随对局存下来 ——
        # 以后重算历史对局时也知道当时是谁读的。
        from . import settings as _settings

        _slot = (_settings.display_slots().get("vlm") or {})
        _recog = dict(raw.get("recognition") or {})
        _recog["model"] = _settings.model_name("vlm") or _slot.get("model", "")
        _recog["provider"] = _slot.get("provider_label", "")
        raw["recognition"] = _recog
        video_name = str(video.get("original_name") or "")
        if video_name:
            raw["video_name"] = video_name
        bp = prefill.get("bp") or {}
        if bp.get("decisions"):
            raw["bp"] = bp

        from ..collection import match as match_col

        try:
            record, issues = match_col.save_match(
                session, adapter, raw, source_kind="video"
            )
        except match_col.MatchValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        report = _analyze(session, record, adapter.game_id)

        return {
            "match_id": record.id,
            "stats": report["stats"],
            "gap_count": len(report["gaps"]),
            "lanes": report["lanes"],
            "degradation": degradation.to_dict(),
            "issues": [i.to_dict() for i in issues],
            "report_url": f"/api/matches/{record.id}/report",
        }

    # ---------------------------------------------------------- 帧图片
    @app.get("/api/videos/{video_id}/frames/{name}")
    def get_frame(video_id: str, name: str):
        path = os.path.join(
            vu.video_store_dir(base_dir), f"{video_id}_frames", name
        )
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="帧图片不存在")
        return FileResponse(path, media_type="image/jpeg")

    # ------------------------------------------------------- 视频预览
    @app.get("/api/videos/{video_id}/file")
    def get_video_file(video_id: str):
        try:
            video = vu.load_video(base_dir, video_id)
        except vu.VideoProcessingError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return FileResponse(video.path, media_type="video/mp4")


def _action_options(adapter) -> Dict[str, List[Dict[str, str]]]:
    """按动作事实分组列出可选取值。

    取值来自**该游戏自己的**动作词表（``action_values_by_fact``），
    不再用共享的 :data:`ACTION_VALUES` —— 那是自走棋时代的词表。
    塞进王者标注页就会出现「开大龙」「调整站位」这种这个游戏没有的动作，
    而且显示成英文原始码，玩家根本不知道勾的是啥；勾了也不对应任何
    一条王者模式，只是噪音。

    换一款游戏、动作集变了，这里自动跟着变，前端不用改。
    """
    from ..rules.extractor import ACTION_VALUES

    by_fact = dict(getattr(adapter, "action_values_by_fact", {}) or {})
    out: Dict[str, List[Dict[str, str]]] = {}
    # 只列**局内**动作事实（``action_fact_names``）——
    # ``known_action_facts()`` 还含 BP 阶段的 ``bp_action``，
    # 那是给 BP 录入页用的，塞进局内标注页就是一栏玩家用不上的选项。
    for kind in sorted(getattr(adapter, "action_fact_names", []) or []):
        values = by_fact.get(kind)
        if values is None:
            # 该游戏没给出分组表，退回共享词表（自走棋用它）
            values = dict(ACTION_VALUES.get(kind, []))
        out[kind] = [
            {"value": v, "label": adapter.action_labels.get(v, v)}
            for v in values
        ]
    return out


def _numeric_facts(adapter) -> List[str]:
    """该游戏里适合用数字输入框填的事实。

    优先用适配器声明的 ``annotate_numeric_facts`` —— 那份清单是
    **从规则库倒推**出来的：只有真正影响判定的事实才问。

    早先这里是「所有带标签的非动作事实」，于是王者标注页每张卡渲染
    13 个数字输入框，玩家面对 13 个框、真正该填的只有 1 个。而且
    多出来的那些不是「问了也没事」，是**问了会给错答案**：局势阶段、
    分路、英雄、对位优劣都由顶部选择或派生逻辑决定，在数字框里填
    是填不进去的，玩家以为自己改了什么，其实被适配器覆盖掉了。

    适配器没有声明时退回旧口径 —— 新增游戏不写声明也不会白屏。
    """
    declared = list(getattr(adapter, "annotate_numeric_facts", []) or [])
    if declared:
        return declared
    action_facts = set(adapter.known_action_facts())
    return sorted(
        f for f in adapter.fact_names
        if f not in action_facts and f in adapter.fact_labels
    )


def _bool_facts(adapter) -> List[str]:
    """该游戏里适合用勾选框填的事实。

    「是 / 否」型的事实用数字框问是折磨：玩家得在 1 / 0 / 留空之间猜，
    填错就等于没填。所以单独一份清单，渲染成勾选框。
    """
    return list(getattr(adapter, "annotate_bool_facts", []) or [])


def _position_choices(adapter) -> List[Dict[str, str]]:
    """标注页上的分路下拉选项；没有分路概念的游戏返回空表。

    分路必须在这里收，因为操作模式全部带 position 条件：这一局不填分路，
    position 就是 ``unknown``，模式一条都不会触发，报告会是空的 —— 而原因
    在界面上看不出来，只会显得规则库不灵。
    """
    return [{"value": v, "label": adapter.position_labels.get(v, v)}
            for v in adapter.position_values]


#: 阶段顺序与中文名，供标注页把同一阶段的模式并成一组。
PHASE_ORDER = ["early", "mid", "late", "endgame"]


def _modes_by_lane(version: str) -> Dict[str, List[Dict[str, Any]]]:
    """把操作模式库按「分路 → 阶段」整理好，供标注页第一屏展示。

    只读不改：这里返回的是**展示用**的摘要（这条模式要什么动作、怎么改），
    不是判定用的规则。判定仍然走内核，前端拿到的只是提前看一眼。
    """
    from ..core.lane_analysis import mode_cell
    from ..rules.store import get_rules
    from ..db import session_scope
    from ..adapters.base import get_adapter

    if not version:
        return {}
    with session_scope() as s:
        rules = get_rules(s, "hok", version)
        if not rules:
            return {}
        adapter = get_adapter("hok")
        action_labels = dict(adapter.action_labels)
        phase_labels = dict(adapter.phase_labels)

    out: Dict[str, List[Dict[str, Any]]] = {}
    for rule in rules:
        cell = mode_cell(rule)
        if cell is None:
            continue
        lane, window = cell
        values = rule.expect_value
        values = values if isinstance(values, list) else [values]
        lo = window[0] if window else 0
        phase = ("early" if lo < 5 else "mid" if lo < 11
                 else "late" if lo < 25 else "endgame")
        out.setdefault(lane, []).append({
            "mode_id": rule.id,
            "title": rule.title,
            "phase": phase,
            "phase_label": phase_labels.get(phase, phase),
            "window": window or [],
            "should_do": "、".join(action_labels.get(v, v) for v in values),
            "how_to_fix": rule.deviation_hint,
        })

    for lane in out:
        out[lane].sort(
            key=lambda m: PHASE_ORDER.index(m["phase"])
            if m["phase"] in PHASE_ORDER else 9
        )
    return out


def _analyze(session, record, game: str):
    from ..core.attribution import analyze_match
    from ..rules.store import get_rules

    adapter = get_adapter(game)
    rules = get_rules(session, game, record.game_version)
    if not rules:
        raise HTTPException(
            status_code=409,
            detail=f"{game} 版本 {record.game_version} 没有规则库，请先载入",
        )
    ctx = adapter.build_context(record.get_payload())
    report = analyze_match(rules, ctx)
    # 统计必须转成字典再返回。
    #
    # ``ExecutionStats`` 上的 ``compliance_rate`` / ``decided`` / ``coverage``
    # 都是**属性**而不是字段，dataclass 序列化不会带上它们 ——
    # 接口于是把「执行度 0%」报成「没有执行度」（null），
    # 而这跟「一条规则都没被检验」完全是两回事。
    # 报告层用的是 ``stats.to_dict()``，这里跟上，两个接口才说同一种话。
    report["stats"] = report["stats"].to_dict()

    # **只报能拿来纠正的差距**。
    #
    # 报告正文里也只呈现这些（见 :func:`app.report.builder._corrective_ids`）——
    # 接口报 3 处、报告上只有 1 条，那玩家打开页面会觉得「少了两条」。
    # 前提读不出来的那些不是「差距」，是「还没法判」。
    #
    # 只对**视频来源**的对局生效（用户原话：「视频做的没错与数据库不一致的
    # 不用纠正」）。玩家自己录的对局，事实是他给的，按规则库字面来判。
    payload = record.get_payload()
    derived = str((payload or {}).get("annotation_source") or "").lower() == "derived"
    corrective = {r.id for r in rules if getattr(r, "corrective", True)}
    if derived and corrective:
        report["gaps"] = [g for g in report["gaps"]
                          if g.rule_id in corrective]
    return report
