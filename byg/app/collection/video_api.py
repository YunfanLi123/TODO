"""视频上传的 API 层与页面编排。

完整链路：

    上传视频 → 回合门控扫描 → 逐回合标注 → 生成对局 → 分析 → 网页报告

分工说明（这一层存在的理由）
------------------------
系统从画面里能确定的只有「回合怎么切」和「血条多长」。玩家的**操作**
在视觉上几乎不可辨 —— 刷野就是对着野怪打，带线就是让小兵往前推，
两者在几帧画面里几乎没有区别。所以让模型去猜动作，猜错就会得出
「你该刷野」这种颠倒结论，比不判更糟。

于是：系统干活（切回合、读血条、抽帧），玩家干活（勾动作、补数字）。
原本手工录 20 局要几小时，这样降到十几分钟。
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from . import chunk_upload
from . import scan_jobs
from . import video_upload as vu
from .video import DegradationReport
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

    挂的是 **v2 的产物**（每段带模型裁决与模型原话）——
    它已经是报告层认识的形状，不必再「搭骨架 → 降级成 raw」转一手。
    """
    if not job.summary:
        return
    _SESSIONS[job.match_id] = {
        "summary": job.summary,
        "rounds": list(job.summary.get("rounds") or []),
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




def _hud_layout_for(game: str) -> Dict[str, str]:
    """该游戏的 HUD 布局（从适配器取，不在采集层写死一份）。"""
    try:
        from ..adapters.base import get_adapter

        adapter = get_adapter(game)
    except Exception:                              # noqa: BLE001
        return {}
    return dict(getattr(adapter, "hud_layout", {}) or {})



#: 「测试连接」用的探针图：白底黑字，写着一串**数字**。
#:
#: 为什么要画数字而不是问颜色：纯文本模型也会顺着问题瞎猜。
#: 问「这张图是什么颜色」，它猜「红色」—— 猜中一次就被误判成
#: 「能看画面」，然后整份报告是空的，而页面上看着一切正常。
#:
#: 数字答案**不在提示词里**，纯文本模型只能瞎蒙（蒙中的概率约千分之一），
#: 蒙不出来它会老实说「没有图片」。真的视觉模型读几个大数字不在话下。
#:
#: 宁可误判「读不了画面」也不要让纯文本模型冒充能看图：
#: 前者的代价是用户换个型号，后者的代价是报告悄悄空掉。
_PROBE_DIGITS = "7391"

#: 探针图里那串数字的提问。**提示词里绝不能出现数字本身**。
_VLM_PROBE = "图里写着一串数字。请只回答那串数字本身，不要解释。"

#: 纯文本槽位（对账措辞）用的探针：只回一句话，**不带图**。
_TEXT_PROBE = "这是一次连通性测试。请只回复「好的」两个字，不要输出其它内容。"


def _probe_png(digits: str = _PROBE_DIGITS) -> bytes:
    """160×80 的白底黑字探针图（PNG 字节）。"""
    import cv2
    import numpy as np

    img = np.full((80, 160, 3), 255, dtype=np.uint8)
    cv2.putText(img, digits, (12, 58), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 0, 0), 3, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", img)
    return bytes(buf) if ok else b""


def _host_of(endpoint: str) -> str:
    """从完整请求地址里取出域名（给人看的）。"""
    try:
        from urllib.parse import urlparse
        return urlparse(endpoint or "").netloc
    except Exception:                              # noqa: BLE001
        return ""


def _upstream_said(detail: str, limit: int = 100) -> str:
    """把供应方自己说的那句原话压成一行（``Authentication Fails…``）。

    为什么要带它（2026-11-13 实测）：401 的自造提示只说「Key 不对，
    或不是这家供应方的 Key」，用户看不出**到底是哪一边不对**；
    而供应方原话里写着 ``Your api key: ****ecbe is invalid`` ——
    一眼就知道是 Key 本身的问题。

    供应方大多回 JSON（``{"error": {"message": "…"}}``）——
    原样贴一坨 JSON 没人读，先把那句话抠出来。太长（整页 HTML）就截断。
    """
    raw = str(detail or "")
    text = " ".join(raw.split())
    if not text:
        return ""
    try:
        data = json.loads(raw)
    except Exception:                                   # noqa: BLE001
        data = None
    if isinstance(data, dict):
        wrap = data.get("error")
        cand = ""
        if isinstance(wrap, dict):
            cand = str(wrap.get("message") or "")
        if not cand:
            cand = str(data.get("message") or data.get("detail") or "")
        if cand.strip():
            text = " ".join(cand.split())
    # 有的接口把 request_id 塞在句子里 —— 那是排查用的，不是给人读的。
    text = re.sub(r"\s*\(?request[_\s-]?id[:：][^)]*\)?", "", text).strip()
    if len(text) > limit:
        text = text[:limit].rstrip("，。、 ,.") + "…"
    return text


def _friendly_error(exc, endpoint: str, provider: str, model: str,
                    slot: str, key: str = "") -> str:
    """把调用失败说成一句**能照着做**的话。

    光说「HTTP 401」用户看不出该干什么 —— 最常见的原因是
    **拿 A 家的 Key 去请求 B 家的接口地址**。这句话得直接说出来；
    能看出 Key 形状就不对时（见 :func:`model_presets.key_shape_hint`），
    要放在最前面 —— 那是**一眼能改对**的那一种。
    """
    from . import model_presets

    host = _host_of(endpoint)
    # 「自定义」时 provider 是 ``custom``，只能按地址反查是哪一家。
    owner = provider if provider in ("qwen", "deepseek", "minimax") \
        else model_presets.provider_of_endpoint(endpoint)
    shape = model_presets.key_shape_hint(key, endpoint)
    said = _upstream_said(getattr(exc, "detail", ""))

    if exc.kind == "auth":
        parts = []
        if shape:
            parts.append("这把 Key 看着就不对：%s" % shape)
        msg = ("HTTP %d：Key 不对，或不是这家供应方的 Key —— "
               "Key 和请求地址必须同一家" % (exc.status or 401))
        if host:
            msg += "（当前地址：%s）" % host
        if owner:
            label = model_presets.get(owner)["label"]
            msg += "。%s 的 Key 只能配他们自己的地址" % label
        parts.append(msg)
        if said:
            parts.append("供应方原话：%s" % said)
        return "｜".join(parts)
    if exc.kind == "bad_request":
        if slot == "vlm" and model_presets.vision_hints(model) is False:
            return ("HTTP %d：这个型号不收图片 —— 纯文本型号当不了"
                    "「观察者模型」，换一个能看图的（%s）"
                    % (exc.status or 400, model_presets.VISION_MODEL_HINTS))
        text = "HTTP %d：请求被拒绝 —— %s" % (
            exc.status or 400, (exc.detail or "")[:120])
        if said and "model" in said.lower():
            # 「模型名写错」也是 400，而它和「Key 不对」在页面上长得一样。
            text += "（型号名到供应方控制台核对一遍）"
        return text
    if exc.kind == "rate_limit":
        return "被限流了（HTTP 429）—— 稍等一会儿再测"
    if exc.kind == "timeout":
        return "超时 —— 网络不通，或这个模型太慢"
    if exc.kind == "server":
        return "供应方那边出错了（HTTP %d）—— 稍后再试" % (exc.status or 500)
    return str(exc)


def _reads_the_digits(answer: str, digits: str = _PROBE_DIGITS) -> bool:
    """回答里有没有把那串数字**原样**说出来。

    只认原样出现。模型把 7391 说成「七千三百九十一」也是看见了，
    但那是另一套识别能力，不值得为它放宽 —— 放宽了纯文本模型就混进来。
    """
    text = (answer or "").replace(" ", "").replace(",", "").replace("，", "")
    return digits in text


def _find_video_match(session: Session, video_id: str, player_id: str = "local"):
    """按视频 id 找已经建好的对局（幂等用）。

    同一个视频只该有一局对局：用户刷新页面、重试提交都走这里，
    不能建出第二条。

    **必须按玩家查**：公网部署开着设备隔离时，每台设备只该看到自己那几局。
    查全部再挑一个，等于让 A 设备能认领 B 设备传的视频。
    """
    if not video_id:
        return None
    from ..collection import match as match_col

    for record in match_col.list_matches(session, player_id=player_id, limit=200):
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
                test_note=str(body.get("test_note") or "")[:200],
                tested_at=str(body.get("tested_at") or "")[:40],
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        if body.get("limits"):
            settings.set_limits(**(body.get("limits") or {}))
        return {"ok": True, **model_config()}

    @app.post("/api/model/test")
    def test_model(payload: Dict[str, Any]) -> Dict[str, Any]:
        """测一下这个模型通不通、**能不能看图**。

        两个槽位测的东西不一样：

        - **观察者模型**：发一张探针图（图里写着四个数字），
          答对那串数字才算真看见了画面。纯文本模型蒙不出来。
        - **分析者模型**：**不带图**，只发一句话。
          文本模型本来就不看图，拿带图的探针去测它，
          就算 Key 完全正确也会被判成「读不了画面」。

        型号名先验优先：确定是纯文本的型号压根不给它发图 ——
        有些接口会直接 400，有些会默默忽略图片然后**猜一个答案**。

        **这一下不写盘**：测就只是测。用户可能测完就改主意，
        一个「测试连接」按钮顺手把配置改了，是会咬人的。
        测出来的结论（能不能看图）由前端暂着，等点「保存」时一起写进去。
        """
        from . import model_presets, settings
        from .vlm_client import Credentials, VlmError, call_model, image_block, text_block

        body = payload or {}
        slot = str(body.get("slot") or "vlm")
        provider = str(body.get("provider") or "")
        model = str(body.get("model") or "")
        key = str(body.get("api_key") or "").strip()

        if provider:
            resolved = model_presets.resolve(provider, model, slot)
            endpoint = str(body.get("endpoint") or "").strip() or resolved["endpoint"]
            protocol = str(body.get("protocol") or "").strip() or resolved["protocol"]
            thinking_field = resolved["thinking_field"]
            model = model or resolved["model"]
        else:
            saved = settings.effective_slot(slot)
            endpoint = str(body.get("endpoint") or "").strip() or str(saved.get("endpoint") or "")
            protocol = str(body.get("protocol") or "").strip() or str(saved.get("protocol") or "openai")
            thinking_field = str(saved.get("thinking_field") or "thinking")
            model = model or str(saved.get("model") or "")
            key = key or str(saved.get("api_key") or "")
        # Key 框出于安全不回显，测试时常留空 —— 无论哪个供应商分支，
        # 留空都回落到已保存的 Key（保存接口本来就是「留空=不改」）。
        # 不做这步回落，选了「自定义」时点测试会被 422 打回，
        # 而前端只认 error 字段，最终显示成看不懂的「未知错误」。
        # 自己的槽位没有密钥就借另一个槽位的（供应方相同才行）——
        # 两个槽位往往用同一把 Key，让人贴两遍纯属折磨。
        if not key:
            key = settings.slot_key(slot, provider=provider)
        if not key:
            raise HTTPException(status_code=422, detail="先填 API Key")
        if not endpoint:
            raise HTTPException(status_code=422, detail="先选供应方")
        if not model:
            raise HTTPException(status_code=422, detail="先填模型名")

        # 型号名先验：确定是纯文本的型号不要给它发图。
        prior = model_presets.vision_hints(model) if slot == "vlm" else None
        if slot == "vlm" and prior is not False:
            blocks = [text_block(_VLM_PROBE),
                      image_block(_probe_png(), media_type="image/png")]
        else:
            blocks = [text_block(_TEXT_PROBE)]

        creds = Credentials(api_key=key, endpoint=endpoint, protocol=protocol,
                            thinking_field=thinking_field, model=model,
                            source="test")
        started = time.time()
        try:
            result = call_model(
                model, blocks,
                creds=creds, max_tokens=64, timeout=60.0, retry=0)
        except VlmError as exc:
            return {"ok": False, "kind": exc.kind,
                    "error": _friendly_error(exc, endpoint, provider, model,
                                             slot, key=key),
                    "detail": exc.detail[:400],
                    "elapsed_ms": int((time.time() - started) * 1000),
                    "can_see_images": False if slot == "vlm" else None}
        elapsed = int((time.time() - started) * 1000)
        answer = (result.text or "").strip()
        if not answer:
            return {"ok": False, "kind": "empty", "error": "模型连话都没接上（空回答）",
                    "detail": "", "elapsed_ms": elapsed,
                    "can_see_images": False if slot == "vlm" else None}
        if slot != "vlm":
            # 文本槽位：只验「通不通」，跟看画面没关系。
            return {"ok": True, "answer": answer[:200], "elapsed_ms": elapsed,
                    "can_see_images": None, "note": "", "usage": result.usage}
        if prior is False:
            can_see = False
            note = ("「%s」按型号名判是纯文本型号，看不了画面 —— "
                    "换个能看图的（%s）"
                    % (model, model_presets.VISION_MODEL_HINTS))
        elif _reads_the_digits(answer):
            can_see = True
            note = ""
        else:
            can_see = False
            if prior is True:
                note = ("「%s」看着是视觉型号，却没读出图里的数字 —— "
                        "Key 或型号可能不支持图片" % model)
            else:
                note = "没读出图里的数字 —— 多半读不了画面"
        return {"ok": True, "answer": answer[:200], "elapsed_ms": elapsed,
                "can_see_images": can_see, "note": note, "usage": result.usage}

    # ------------------------------------------------------------ 视频上传
    def _resolve_version(session: Session, game: str) -> str:
        """取该游戏规则库里**最新**的版本。

        版本填错不会报错，只会让扫描完之后拿到的报告是空的 ——
        而那种错误在界面上完全看不出来：上传成功、扫描成功、
        报告页面也打开了，就是没有一条模式被检验。所以这个字段干脆不给填。

        一个版本都没有就当场拒掉：让用户白等几分钟扫完视频再撞上
        「规则库为空」，不如上传这一秒就告诉他。
        """
        from ..rules.store import latest_version

        version = latest_version(session, game)
        if not version:
            raise HTTPException(
                status_code=409,
                detail=f"{game} 还没有操作模式库，请先到操作模式库载入",
            )
        return version

    def _identity(hero: Any, position: Any):
        """归一玩家填的英雄/分路。分路只收五个码（也认中文）。"""
        names = {"对抗路": "side", "发育路": "carry", "中路": "mid",
                 "打野": "jungle", "游走": "roam",
                 "side": "side", "carry": "carry", "mid": "mid",
                 "jungle": "jungle", "roam": "roam"}
        hero_text = str(hero or "").strip()[:20]
        pos_text = str(position or "").strip()
        pos_text = names.get(pos_text, names.get(pos_text.lower(), ""))
        return hero_text, pos_text

    def _start_scan(video, game: str, version: str, match_id: str,
                    hero: str = "", position: str = "") -> Dict[str, Any]:
        """视频已经落盘之后的公共流程：建任务、开扫、核对录屏规格。

        一次性上传与分片上传走的是同一条路 —— 分片传完最后一片时，
        那个文件就已经是完整的视频了，剩下的事完全一样。
        """
        # 扫描参数固定用调好的默认值。界面曾经给两个滑块（边界阈值 /
        # 最短回合间隔），但让人自己拖只会得到更差的切片：调高漏回合，
        # 调低把一次转场切成好几段 —— 而这种错误在报告里看不出来，
        # 只会让后面所有模式的判定都建立在错误的回合切分上。
        cfg = vu.ScanConfig()
        job = scan_jobs.RUNNER.create(
            uuid.uuid4().hex[:12], match_id, video, game, version,
            hero=hero, position=position,
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

    @app.post("/api/videos/upload")
    async def upload_video(
        file: UploadFile = File(...),
        game: str = Form("hok"),
        hero: str = Form(""),
        position: str = Form(""),
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

        version = _resolve_version(session, game)

        video_id = uuid.uuid4().hex[:12]
        match_id = f"{game}-video-{video_id}"

        try:
            # 流式落盘：边读边写，2GB 视频也不会整个进内存
            video = await vu.save_upload_stream(file, base_dir, video_id)
            video = vu.probe(video)
        except vu.VideoProcessingError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        hero, position = _identity(hero, position)
        return _start_scan(video, game, version, match_id, hero, position)

    # ------------------------------------------------------- 分片上传（大文件）
    #
    # 为什么需要：云端网关对**单次请求体**有硬上限，实测 20MB 能过、
    # 24MB 直接断连（客户端看到的是 TLS EOF，不是像样的 4xx）。
    # 一局王者录屏几百 MB，一次性 POST 必然失败 ——
    # 而且失败得很难看：网关回 HTML 错误页，前端当 JSON 解析，
    # 报「Unexpected token '<'」，跟真实原因毫无关系。
    #
    # 三步：init 拿到分片计划 → chunk 一片一片传 → complete 收尾开扫。

    @app.post("/api/videos/upload/init")
    def upload_init(
        payload: Dict[str, Any],
        session: Session = Depends(db_session),
        player_id: str = Query("local"),
    ) -> Dict[str, Any]:
        """开一个分片上传会话，返回分片计划。"""
        game = str((payload or {}).get("game") or "hok")
        if game != "hok":
            raise HTTPException(
                status_code=422,
                detail="目前只支持王者荣耀的对局录屏。",
            )
        _resolve_version(session, game)

        filename = str((payload or {}).get("filename") or "")
        try:
            size = int((payload or {}).get("size") or 0)
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="文件大小不是数字")

        hero, position = _identity((payload or {}).get("hero"),
                                   (payload or {}).get("position"))
        try:
            plan = chunk_upload.start(filename, size, base_dir, player_id,
                                      hero=hero, position=position)
        except vu.VideoProcessingError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        return {
            "upload_id": plan.upload_id,
            "chunk_size": plan.chunk_size,
            "total_chunks": plan.total_chunks,
            "size": plan.size,
            "video_id": plan.video_id,
        }

    @app.post("/api/videos/upload/chunk")
    async def upload_chunk(
        request: Request,
        upload_id: str = Query(...),
        index: int = Query(...),
        player_id: str = Query("local"),
    ) -> Dict[str, Any]:
        """收一片。

        片是**裸二进制**（请求体直接就是这一片的字节），不是 multipart、
        也不是 base64 的 JSON：

        - multipart 每片都要带一份边界和头部，780MB 分成上百片，
          白白多传好几 MB；
        - base64 会让体积涨 33% —— 780MB 变成 1GB 出网流量，
          而云端的流量是按量计费的。

        片号与上传 id 放查询串里，因为请求体已经被视频字节占满了。
        """
        plan = chunk_upload.get(upload_id, player_id)
        if plan is None:
            raise HTTPException(
                status_code=404, detail="上传会话不存在或已过期，请重新上传"
            )

        blob = await request.body()

        try:
            received = chunk_upload.put_chunk(plan, index, blob)
        except vu.VideoProcessingError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        return {
            "upload_id": upload_id,
            "received": received,
            "total_chunks": plan.total_chunks,
            "received_bytes": plan.received_bytes,
        }

    @app.post("/api/videos/upload/complete")
    def upload_complete(
        payload: Dict[str, Any],
        session: Session = Depends(db_session),
        player_id: str = Query("local"),
    ) -> Dict[str, Any]:
        """全部传完 → 校验 → 落成正式视频 → 开扫。"""
        upload_id = str((payload or {}).get("upload_id") or "")
        game = str((payload or {}).get("game") or "hok")

        plan = chunk_upload.get(upload_id, player_id)
        if plan is None:
            raise HTTPException(
                status_code=404, detail="上传会话不存在或已过期，请重新上传"
            )

        version = _resolve_version(session, game)

        try:
            video = chunk_upload.finish(plan)
            video = vu.probe(video)
        except vu.VideoProcessingError as exc:
            chunk_upload.drop(upload_id)
            raise HTTPException(status_code=422, detail=str(exc))

        match_id = f"{game}-video-{plan.video_id}"
        return _start_scan(video, game, version, match_id,
                           plan.hero, plan.position)


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
    def finalize(payload: Dict[str, Any], session: Session = Depends(db_session),
                 player_id: str = Query("local")) -> Dict[str, Any]:
        """模型读完了 → 落库 → 分析 → 返回报告地址。

        **中间没有人工那一步**：分路、胜负、BP、每段操作都是模型从画面读的
        （用户原话：「我只要模型去读出报告，其余一切步骤都省去」）。

        模型读不出来的就记「读不出来」—— 报告照出，相关规则判不了，
        **不回头让人填**。

        **player_id 必须跟着走**：公网部署开着设备隔离，对局是按设备存的。
        这里落库时若写死 ``"local"``，报告页按设备查就查不到 ——
        表现为「上传成功、扫描成功、跳到报告却是 404」，
        而用户看到的是自己刚传的局不见了。
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

        # ⚠️ **这里原来有一处「幂等早退」**（2026-11-14 删掉）：
        #
        #     existing = _find_video_match(session, video_id, player_id)
        #     if existing is not None:
        #         return 旧报告          # ← 新分析一个字都没写进去
        #
        # 它按**视频 id** 认「这一局建过了」，于是**重新扫描同一段录像**
        # 永远拿回上次那份 payload —— 用户重启到最新 exe、重扫、
        # 页面上看到的还是几天前的结果（实测：03:52 那局，重扫到 11:48
        # 仍然显示旧的）。这不是打包问题，是数据没被覆盖。
        #
        # 现在一律走完整流程：``save_match`` 本身就是按 ``match_id``
        # 覆盖写的（同 id 就更新 payload），重复提交不会建出第二条，
        # 但**一定会把新分析写进去**。

        prefill = state.get("prefill") or {}
        rounds = list(state.get("rounds") or [])
        notes = list(summary.get("notes") or [])
        # **整局综述 +「攻略之外的问题」** —— 报告正文要靠它出「复盘建议」。
        # 2026-11-13 之前这一段没落库：模型算出来的综述到这儿就丢了。
        overview = summary.get("overview") or {}
        if not isinstance(overview, dict):
            overview = {}

        # **v2 的产物已经是报告层认识的形状** ——
        # 每一段自带「模型原话」与「模型逐条裁决」，不必再走
        # 「搭骨架 → 降级成 raw」那一层中间转换。
        raw: Dict[str, Any] = {
            "match_id": match_id,
            "game": adapter.game_id,
            "game_version": str(summary.get("game_version") or ""),
            "player": {},
            "outcome": {},
            "played_at": "",
            "rounds": rounds,
        }
        # 分路与胜负是**模型读出来的**；读不出来就留空
        # （留空 = 那些带分路条件的规则不触发，而不是「没做到」）。
        #
        # **英雄与分路取识别链的最终决定**（``meta``），不是 prefill 的读数 ——
        # 决定权在裁决层（看选人界面的字 / BP 记录 / 看完整局），
        # 报告上显示的那个必须就是**判规则时用的那个**。
        #
        # 认不准时 hero 是空串 —— 英雄层规则一条都不触发，
        # 比拿一个猜的英雄去套错对象强。
        # hero_source 非空 = 这是**读出来的**（不是玩家自己录的），报告里要标「请核对」。
        _meta = summary.get("meta") or {}
        _hero = str(_meta.get("hero") or "").strip()
        _position = str(_meta.get("position") or "").strip()
        raw["player"] = {
            "position": _position or str(prefill.get("position") or ""),
            # **对线优劣 = 模型看这一局的对线期判的**（2026-11-14 用户拍板：
            # 「不要对位表了，让模型去看」）。对线劣势不是英雄决定的 ——
            # 同一个对位被反野三次就是劣势，表回答不了这个。
            # 判不出来就是 unknown：带这个前提的规则按「不适用」处理。
            "matchup": str((_meta.get("matchup") or {}).get("matchup") or "unknown"),
            "matchup_why": str((_meta.get("matchup") or {}).get("why") or ""),
            "hero": _hero or str(prefill.get("hero") or ""),
            "hero_source": str(_meta.get("hero_source") or ""),
            # **分路的证据等级与细到哪条路** —— 模型读出来的分路
            # 不能再被当成「你填的」（2026-11-07 实测：那局分路是模型
            # 答的，却因为没有 position_source 被标成 declared，
            # 报告就按「玩家填的」去用了）。
            "position_provenance": str(_meta.get("position_provenance") or ""),
            "lane_detail": str(_meta.get("lane_detail") or ""),
        }
        # 分路是**推断**来的（按英雄），要标出来源 ——
        # 报告里「这一局是按什么定的分路」得说清，玩家才有办法核对。
        # 优先用**识别链的最终决定**那份来源说明（英雄换过，来源说明也得跟着换）。
        _position_source = str(_meta.get("position_source") or "") \
            or str(prefill.get("position_source") or "")
        if _position_source:
            raw["player"]["position_source"] = _position_source
        # **地图位置**（野区/河道/中路）—— 模型读画面读出来的，
        # 与**分路**是两回事：分路是「你打哪条线」，地图位置是「你人在哪」。
        # 报告的「当时局面」里用得上。
        if prefill.get("map_position"):
            raw["map_position"] = str(prefill["map_position"])
        # **胜负**：以识别链的最终决定为准 ——
        # 「看整局」那次读到的（结算画面就在整局里）与「读最后几帧」那次
        # 对过账了（见 ``recognition2/pipeline.py``）。
        # 两处都读不到就留空 —— **留空 ≠ 失败**（那是「没读到」，不是「输了」）。
        _result = str(_meta.get("result") or "").strip().lower()
        if _result not in ("win", "lose"):
            _result = str(prefill.get("result") or "").strip().lower()
        raw["outcome"] = {"result": _result if _result in ("win", "lose") else ""}
        # **胜负没读到** —— 报告顶上因此没有结果栏，得让用户知道为什么。
        #
        # 2026-11-02 实测（熬隐那局）：读胜负那一步的模型调用返回了空，
        # 识别层如实记下 ``"unknown"``；而页面模板是
        # ``{'失败' if result == 'lose' else '胜利'}`` —— 任何非空值都会
        # 被渲染成「胜利」，那一局明明是输的。
        # 报告层已改成「读不到就不给结果栏」，这里负责**说明为什么空**。
        if raw["outcome"]["result"] not in ("win", "lose"):
            why = str(prefill.get("why") or "").strip().rstrip("。")
            text = "这一局的胜负没读到"
            if why:
                text += "（%s）" % why
            notes.append(text + "。")
        raw["played_at"] = ""
        # **整局综述**（model 自己写的 summary / 阶段 / 攻略之外的问题）。
        # 报告层把 findings 变成「复盘建议」条目 —— 那是**按这一局的真实
        # 画面**推出来的，模式库只是参考（2026-11-13）。
        if overview:
            raw["overview"] = overview
        # **逐次死亡复盘**（这波赚还是亏 + 模型自己查的攻略）。
        # 报告层据此出「死亡复盘」条目；没有它就回落到旧的「怎么死的」那套。
        coach = _meta.get("coach") or {}
        if isinstance(coach, dict) and (coach.get("deaths")
                                        or coach.get("match")):
            raw["coach"] = coach
        # **顶栏 K/D/A 读出来的事实**（死亡时刻、我方人头）——
        # 报告要写「模型报了几次、顶栏推出来几次」，对不上就得说。
        kda = _meta.get("kda") or {}
        if isinstance(kda, dict) and (kda.get("events") or kda.get("samples")):
            raw["kda"] = kda
            if _meta.get("kda_check"):
                raw["kda_check"] = _meta["kda_check"]
        # 动作是**模型从画面读出来的**，不是玩家自己勾的 ——
        # 报告据此说明来源（见 report.html 的 derived 提示）。
        raw["annotation_source"] = "derived"
        # **哪一个模型读出来的**：换模型就换结论，报告顶部要写出来。
        # 写进 ``recognition`` 是因为那一块本来就会随对局存下来 ——
        # 以后重算历史对局时也知道当时是谁读的。
        from . import settings as _settings

        _slot = (_settings.display_slots().get("vlm") or {})
        # **识别诊断**：v2 留的痕（哪几段没读出来、为什么、额度够不够）。
        #
        #: 这段必须**说人话** —— 空表最容易被读成「打得没问题」，
        #: 而实际是「根本没分析」。所以模型调不通、读不到的那些都进去。
        degradation = DegradationReport(
            total_rounds=len(rounds),
            usable_rounds=sum(1 for r in rounds
                              if str(r.get("narrative") or "").strip()),
            dropped_rounds=sum(1 for r in rounds
                               if not str(r.get("narrative") or "").strip()),
            warnings=notes,
        )
        raw["recognition"] = {
            # **哪套识别器、哪一版**：报告要写出来。
            # 换一版识别器结论就可能不同，读者得知道这局是哪套得出的。
            "recognizer_id": "recognition2",
            # v3 = 「模型直接看视频」那条链（v2 是抽帧那条）。
            "recognizer_version": "v3",
            # 哪一个模型读出来的：换模型就换结论，报告顶部要写出来。
            "model": _settings.model_name("vlm") or _slot.get("model", ""),
            "provider": _slot.get("provider_label", ""),
            "stats": {
                "rounds": len(rounds),
                "segments": _meta.get("segments", 0),
                # **看视频**那一步的调用与 token（新链路）。
                "watch_calls": _meta.get("watch_calls", 0),
                "watch_tokens": _meta.get("watch_tokens", 0),
                "judge_calls": _meta.get("judge_calls", 0),
                "rules_total": _meta.get("rules_total", 0),
                "candidates": _meta.get("candidates", 0),
                "coverage": _meta.get("coverage", {}),
                "source": video_id,
            },
            # **候选是哪来的**：模型自己检索的（``model``）还是退回顾则筛的
            # （``fallback``）。退回了就得在报告上说明 —— 口径变了。
            # **死亡清单** —— 失败局的倒推复盘用它（每次死亡一条）。
        # 读不到就是空数组（**不是失败**）。
        "deaths": list(_meta.get("deaths") or []),
        # **模型报的总次数**（落库的清单可能是截断过的）——
        # 报告层靠它说清「这局死了几次、复盘里写了几次」。
        "deaths_total": int(_meta.get("deaths_total") or 0),
        # **血条读数** —— 残血时段（像素法量的，模型复核过）。
        # 报告里的「血线」这一组条目用它；读不到就是空字典（**不是失败**）。
        "hp": dict(_meta.get("hp") or {}),
        "selection": {
                "mode": str(_meta.get("selection_mode") or ""),
                "queries": list(_meta.get("selection_queries") or []),
                "candidates": int(_meta.get("candidates") or 0),
                # **具体是哪几条** —— 只存个数的话，事后没法追查
                # 「这局到底查了哪几条攻略」。
                "candidate_ids": list(_meta.get("candidate_ids") or []),
                # **检索能力有没有降级**（向量那路没走通 → 只按字面匹配）。
                "degraded": str(_meta.get("selection_degraded") or ""),
                # **每一次检索请求的结果**（含失败原因）。
                #
                # 为什么必须落这一份：只存「6 个请求 → 0 条打法」的话，
                # 读者分不清是「检索调不通」还是「库里确实没有适用的打法」——
                # 这两件事要改的东西完全不同（前者查接口，后者补资料）。
                # 2026-11-07 实测：这一局的 error 被丢在了这一步，
                # 报告只能写「0 条打法」，像库里没东西。
                "hits": list(_meta.get("selection_hits") or []),
            },
            # **英雄没认准**（模型说的和技能按钮认的对不上）——
            # 英雄层的打法这次一条都没判，得让人知道为什么。
            "hero_conflict": dict(_meta.get("hero_conflict") or {}),
            # **英雄是按哪条读数定的** + **哪些读数被否掉了、为什么**
            # （2026-11-03 新增：被禁用 / 不在我方名单里的读数一律作废）。
            "hero_source": str(_meta.get("hero_source") or ""),
            "hero_voided": list(_meta.get("hero_voided") or []),
            # **只是提醒、不作废**的读数（BP 记录是模型看头像认的时，
            # 它没资格否掉别处读出来的名字 —— 2026-11-13）。
            "hero_soft": list(_meta.get("hero_soft") or []),
            # **这份 BP 记录是哪来的**：``model`` = 模型看 BP 帧认的（可能认错），
            # ``record`` = 玩家自己录的。报告据此决定要不要提醒「名字请核对」。
            "bp_source": str(_meta.get("bp_source") or ""),
            # **判规则时当成谁** —— 与 ``player.hero`` 分开：
            # 英雄层的攻略不覆盖这局打法时（英雄按 A 路写、这局打 B 路），
            # 这里会是空串，报告要说清「这次按分路通则判」。
            "judge_hero": str(_meta.get("judge_hero") or ""),
            "hero_off_lane": bool(_meta.get("hero_off_lane") or False),
            # **这一局是怎么「看」的**：``video`` = 整局录像直接给模型看；
            # ``frames`` = 按画面变化抽帧、每 5 分钟一批（模型收不了视频时）。
            # 两种看法的粗细差很多，报告要说出来（2026-11-13）。
            "watch_mode": str(_meta.get("watch_mode") or ""),
            "frames": [],
            "degradation": degradation.to_dict(),
            # 原视频 id：报告里「看这一段」靠它找视频文件
            "video_id": video_id,
        }
        video_name = str(video.get("original_name") or "")
        if video_name:
            raw["video_name"] = video_name
        bp = prefill.get("bp") or {}
        if bp.get("decisions"):
            raw["bp"] = bp

        from ..collection import match as match_col

        try:
            record, issues = match_col.save_match(
                session, adapter, raw, player_id=player_id, source_kind="video"
            )
        except match_col.MatchValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        report = _analyze(session, record, adapter.game_id)

        # **叙述在这里写好** —— 报告页第一次打开就不用等模型。
        # 写不出来不影响这一步：报告页会退回模板写法。
        try:
            from ..main import _warm_narratives

            _warm_narratives(session, record, adapter.game_id)
        except Exception:                               # noqa: BLE001
            pass

        return {
            "match_id": record.id,
            "stats": report["stats"],
            # **结论条数** —— 与报告页正文同一份（模型逐条裁决里「该改」的）。
            "gap_count": report.get("conclusion_count", len(report["gaps"])),
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

    取值**只**来自该游戏自己的动作词表（``action_values_by_fact``）。
    拿不到分组表就报错 —— 那是配置问题，不该拿别的游戏的词表糊弄：
    那种词表里的动作（开大龙、调整站位…）这个游戏根本没有，塞进来
    只会列出选不上的假选项，显示成英文原始码，玩家根本不知道勾的是啥，
    勾了也不对应任何一条模式，只是噪音。

    换一款游戏、动作集变了，这里自动跟着变，前端不用改。
    """
    by_fact = dict(getattr(adapter, "action_values_by_fact", {}) or {})
    out: Dict[str, List[Dict[str, str]]] = {}
    # 只列**局内**动作事实（``action_fact_names``）——
    # ``known_action_facts()`` 还含 BP 阶段的 ``bp_action``，
    # 那是给 BP 录入页用的，塞进局内标注页就是一栏玩家用不上的选项。
    for kind in sorted(getattr(adapter, "action_fact_names", []) or []):
        values = by_fact.get(kind)
        if values is None:
            # 没有分组表 = 没有「这个游戏有哪些动作」这个事实。
            # 编一份词表进来只会让页面列出选不上的动作，比报错更糟。
            raise HTTPException(
                status_code=422,
                detail=(
                    "该游戏没有提供动作分组表（action_values_by_fact），"
                    f"无法列出「{kind}」的可选动作"
                ),
            )
        out[kind] = [
            {"value": v, "label": adapter.action_labels.get(v, v)}
            for v in values
        ]
    return out


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


def _analyze(session, record, game: str):
    """分析这一局 —— **视频来源走 v2（判定权归模型）**。

    以前这里对视频对局也调老内核（:func:`app.core.attribution.analyze_match`）：
    对局是 v2 的形状（每段自带模型裁决与模型原话），
    老内核读不懂 → 触发 0 条、结论 0 条；
    而报告页走的是 v2 的裁决（触发 7 条）。
    于是**接口说「0 条结论」、页面有 7 条结论** —— 两个口径对不上，
    而且老内核那一趟在视频路径上是白跑的。

    现在两边用同一套（:func:`app.report.builder.build_match_report`），
    接口报的数字与页面上看到的**一定是同一份**。
    """
    from ..rules.store import get_rules

    adapter = get_adapter(game)
    rules = get_rules(session, game, record.game_version)
    if not rules:
        raise HTTPException(
            status_code=409,
            detail=f"{game} 版本 {record.game_version} 没有规则库，请先载入",
        )
    ctx = adapter.build_context(record.get_payload())
    payload = record.get_payload()
    derived = str((payload or {}).get("annotation_source") or "").lower() == "derived"
    if derived:
        return _analyze_by_model(rules, ctx)

    from ..core.attribution import analyze_match

    # v2 操作模式库（有就用，没有就退回 v1 规则引擎）。
    try:
        from ..rules.patterns import load_patterns
        patterns = load_patterns(game, record.game_version)
    except Exception:                                       # noqa: BLE001
        patterns = None
    report = analyze_match(rules, ctx, patterns=patterns)
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
        # v2 引擎给出的**这一局**结论优先：分支前提被指认了，
        # 那条模式这一局就能纠正（静态库里它仍是「不可纠正」）。
        runtime = {
            v.rule_id: v.corrective
            for v in (report.get("verdicts") or [])
            if getattr(v, "corrective", None) is not None
        }
        def _keep(gap):
            flag = runtime.get(gap.rule_id)
            if flag is not None:
                return bool(flag)
            return gap.rule_id in corrective
        report["gaps"] = [g for g in report["gaps"] if _keep(g)]
    report["conclusion_count"] = len(report["gaps"])
    return report


def _analyze_by_model(rules, ctx) -> Dict[str, Any]:
    """视频来源的对局：**判定权归模型**（与报告页同一套口径）。

    结论条数取报告正文的那一份（模型逐条裁决里「偏离/没做」的那些）——
    别拿老内核的 gap 数来报，那是另一套口径。
    """
    from ..report.builder import build_match_report

    try:
        report = build_match_report(rules, ctx)
    except Exception as exc:                               # noqa: BLE001
        # 报告层算不出来时**说清原因**，别报「0 条结论」——
        # 空表最容易被读成「打得没问题」，实际是根本没算出来。
        return {
            "stats": {
                "triggered": 0, "satisfied": 0, "deviated": 0,
                "undetermined": 0, "compliance_rate": None,
            },
            "gaps": [],
            "lanes": {"lane": "", "lanes": [], "cells": [], "findings": [],
                      "problem_count": 0},
            "mode_analysis": [],
            "conclusion_count": 0,
            "warnings": ["报告算不出来（%s）" % exc],
        }
    report["conclusion_count"] = len(report.get("mode_analysis") or [])
    return report
