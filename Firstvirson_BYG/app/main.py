"""FastAPI 应用入口：把采集层 / 规则层 / 内核 / 报告层串成一条链路。

接口设计围绕产品的真实工作流：
    1. 装规则库   POST /api/games/{game}/rules/seed
    2. 导攻略     POST /api/guides/extract  →  人工确认  →  POST /api/guides/confirm
    3. 录对局     POST /api/matches        （自动触发分析）
    4. 看报告     GET  /api/matches/{id}/report
    5. 看成长     GET  /api/growth/{game}
    6. 版本适配   POST /api/versions/{game}/diff
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union


from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from . import adapters  # noqa: F401  确保适配器完成注册
from .adapters.base import get_adapter, registered_games
from .adapters.registry import REGISTRY
from .collection import guide as guide_col
from .collection import guide_sessions
from .collection import match as match_col
from .collection.collector import (
    CollectionError,
    HttpGuideCollector,
    SourceNotAllowed,
    TextCollector,
    collect_and_extract,
    confirm_collected,
)
from .collection.video import (
    ACTION_ACCEPT_THRESHOLD,
    ACTION_FLAG_THRESHOLD,
    FACT_ACCEPT_THRESHOLD,
    FACT_FLAG_THRESHOLD,
    GATE_TRUST_THRESHOLD,
    DetectedAction,
    DetectedFact,
    GateFrame,
    RecognitionResult,
    RecognizedRound,
    summarize_cost,
    to_raw_match,
)
from .core.attribution import analyze_match
from .core.context import MatchContext
from .core.curve import build_curve
from .core.dsl import Rule
from .core import device as device_mod
from .core import session as session_mod
from .db import Base, get_engine, session_scope
from .report.builder import (
    build_bp_report,
    build_growth_report,
    build_match_report,
    build_version_report,
)
from .report.export import (
    attachment_headers,
    build_export_page,
    export_filename,
    export_json,
)
from .report.views import (
    STATIC_DIR,
    render_bp_form,
    render_bp_report,
    render_report,
    render_review,
)
from .rules.authoring import (
    RuleWriteError,
    add_source,
    delete_source,
    list_sources,
    write_rule,
    write_rules_bulk,
)
from .rules.review import (
    ReviewError,
    build_review_view,
    compute_coverage,
    coverage_summary,
    revision_history,
    set_status,
    update_rule,
)
from .rules.store import (
    count_rules,
    upsert_rule,
    deactivate_missing,
    get_rules,
    latest_version,
    list_guide_sources,
    list_versions,
    load_seed_library,
    load_bp_seed_library,
)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """建表 + 启动自检 +（即焚模式）自动灌规则库 + 起清扫协程。"""
    get_engine()
    _check_clock_template()
    _autoseed_ephemeral()

    sweeper = None
    if session_mod.enabled():
        sweeper = asyncio.create_task(_session_sweeper())
    try:
        yield
    finally:
        if sweeper is not None:
            sweeper.cancel()


def _autoseed_ephemeral() -> None:
    """即焚模式：数据库是临时的，所以启动时自己灌一次种子库。

    不灌的话页面全是空的 —— 而且不报错，只是「没有内容」，
    是最难发现的那种问题。
    """
    if not session_mod.enabled():
        return
    from .rules.store import load_bp_seed_library, load_seed_library

    with session_scope() as s:
        for game in sorted(REGISTRY):
            try:
                load_seed_library(s, game)
                load_bp_seed_library(s, game)
            except Exception as exc:  # noqa: BLE001
                print(f"[警告] 启动时灌 {game} 规则库失败：{exc}")


async def _session_sweeper() -> None:
    """定期清扫：把到期会话的数据删掉。

    这是 beacon 丢了时的兜底 —— 浏览器崩溃、断网、被杀，
    都不会发出「我走了」的通知。
    """
    while True:
        # 每 5 秒扫一次。间隔要小：用户关掉网页后是「多久之后没」，
        # 扫得慢就等于承诺了一个做不到的时间。
        await asyncio.sleep(5)
        try:
            await asyncio.to_thread(session_mod.sweep, BASE_DIR)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"[警告] 会话清扫出错：{exc}")


def _check_clock_template() -> None:
    """启动时确认时钟模板能加载。

    模板是**标定值**，打包时可能没打进去、或者被打成了目录
    （PyInstaller 对「文件」源的 ``--add-data`` 目标给成完整文件名时，
    会把那个名字建成目录）。这两种情况都不会让程序报错 ——
    只是安静地「读不出时钟」，于是每一局都退回「阶段要手填」，
    而界面上看不出任何异常。

    与其等扫完 5 分钟才发现，不如启动就说一声。
    """
    from .collection.clock import load_digits

    try:
        digits = load_digits()
    except Exception as exc:                    # noqa: BLE001
        digits = []
        print(f"[警告] 时钟模板加载出错：{exc}")
    if not digits:
        print(
            "[警告] 读不到时钟模板（app/collection/clock_digits.json）。\n"
            "        视频扫描仍能跑，但**认不出游戏内时间** —— "
            "每一局都会退回「阶段要手填」。\n"
            "        打包时它必须作为一个文件放进包内，不能被打成目录。"
        )


app = FastAPI(
    title="基于攻略规范的个人对局执行度分析系统",
    description=(
        "把现成攻略转化成可判定的规则，逐时间段检查玩家实际操作，"
        "定位到「差在哪一局、哪一段时间、该做而没做的是哪一步」。"
    ),
    version="0.2.0",
    lifespan=lifespan,
)

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ==========================================================================
# 设备隔离（公网部署时用）
#
# 所有接口的 player_id 都是 Query("local")，所以在这里统一注入一次，
# 比改三十多个接口省事，也比让每个接口自己读 Cookie 更不容易漏。
#
# 隔离默认**关闭**（环境变量 MULTI_TENANT=1 才开）——
# 本地单人使用、现有测试的行为一个字都不变。
# ==========================================================================


@app.middleware("http")
async def _device_scope(request: Request, call_next):
    from urllib.parse import parse_qsl, urlencode

    if not device_mod.isolation_enabled():
        return await call_next(request)

    device_id = request.cookies.get(device_mod.DEVICE_COOKIE) or device_mod.new_device_id()
    # 记一下这个设备来过：即焚模式靠它判断会话死没死。
    session_mod.touch(device_id)

    # 把设备标识塞进查询串。
    # **必须自己解析原始 query_string**：一旦读过 request.query_params，
    # Starlette 就把它缓存了，之后再改 scope 也不生效 ——
    # 注入会静默地不灵，而且没有任何报错。
    raw = request.scope.get("query_string", b"").decode("latin-1")
    params = parse_qsl(raw, keep_blank_values=True)
    if not any(key == "player_id" for key, _ in params):
        params.append(("player_id", device_id))
        request.scope["query_string"] = urlencode(params).encode("latin-1")

    response = await call_next(request)
    device_mod.ensure_device_cookie(request, response, device_id)
    return response


# ==========================================================================
# 会话结束（即焚模式）
#
# 用户要的是「关掉网页就没了」。但「关掉」这件事服务器看不见，
# 只能靠浏览器告诉它 —— 而这个通知会丢（崩溃、断网、浏览器被杀）。
# 所以这里是「排一次删除」，真正的删除由清扫协程做，
# 判据是「标记之后还有没有任何请求」。
# ==========================================================================


@app.post("/api/session/end")
def api_session_end(request: Request) -> Dict[str, Any]:
    """页面关了 → 排一次删除。

    前端用 ``navigator.sendBeacon`` 调它 —— 页面卸载时普通请求会被
    浏览器掐掉，beacon 才是专门干这个的。
    """
    if not session_mod.enabled():
        return {"message": "本服务没有开启即焚模式", "ephemeral": False}

    device_id = device_mod.device_id(request)
    session_mod.mark_closed(device_id)
    return {
        "message": "已记下，离开后数据会被清除",
        "ephemeral": True,
        "grace_seconds": session_mod.grace_seconds(),
    }


@app.get("/api/session/policy")
def api_session_policy() -> Dict[str, Any]:
    """这个服务留不留存数据。

    给 API 使用者与排查用的 —— 页面上的提示条已经去掉了，
    所以想确认「这台实例到底留不留数据」只能问它。
    """
    ephemeral = session_mod.enabled()
    return {
        "ephemeral": ephemeral,
        "isolated": device_mod.isolation_enabled(),
        "grace_seconds": session_mod.grace_seconds() if ephemeral else 0,
        "ttl_seconds": session_mod.ttl_seconds() if ephemeral else 0,
    }


#: 项目根目录。视频上传用它定位存储目录。
#:
#: 打包成 exe 后必须指向 **exe 所在目录**，不能指向解包目录 ——
#: onefile 的解包目录在进程退出时会被清掉，而 onedir 的 _internal 是
#: 程序内部实现，用户的视频和缓存不该混在里面。
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 视频路由在模块级注册 —— 放进 lifespan 的话，测试和 CLI 都拿不到它们，
# 而「接口存在与否」恰恰是测试要断言的东西。
from .collection.video_api import register_routes as _register_video_routes
from .collection.video_upload import (
    MAX_UPLOAD_BYTES,
    raise_multipart_limit,
    redirect_tempdir,
)

_register_video_routes(app, BASE_DIR)

# 页面：只有三个 —— 上传视频 / 对局记录 / 规则库
from .report.pages import register_pages as _register_pages

_register_pages(app, BASE_DIR)

#: 上传相关的全局设置在这里生效，而不是 lifespan ——
#: TestClient 直接调用时不会跑 lifespan，那样测试环境等于没配，
#: 会写出一个「测试过了、实际一传大文件就炸」的假象。
TMP_DIR = redirect_tempdir(BASE_DIR)
raise_multipart_limit(MAX_UPLOAD_BYTES)


def db_session():
    """FastAPI 依赖：每个请求一个会话，结束自动提交/回滚。"""
    with session_scope() as session:
        yield session


# ==========================================================================
# 请求模型
# ==========================================================================


class RoundInput(BaseModel):
    """一个回合的录入结构。

    ``phase`` 允许字符串：自走棋用整数小阶段（1/2/3…），而 MOBA 用
    ``early`` / ``mid`` / ``late`` / ``endgame`` 这样的枚举。内核的
    RoundRecord 用 ``phase_code``（整数）承载游戏内位置，展示用的原始
    phase 放在 facts 里 —— 两种游戏的时间坐标本来就不该强行统一。

    ``time_min`` 必须在这里声明。少了它，MOBA 对局从 API 录进来时局内时间
    会被 pydantic **静默丢掉** —— 适配器随后把 time_min 标成「缺失」，
    而所有带时间窗的模式在缺失事实上一律判不成立。表现是：
    接口返回 200、库也是满的，报告却一条结论都没有。
    """

    stage: int
    phase: Union[int, str] = 1
    #: 局内分钟。MOBA 的一切都以它为准 —— 阶段由它推导。
    time_min: Optional[int] = None
    facts: Dict[str, Any] = Field(default_factory=dict)
    actions: List[Dict[str, Any]] = Field(default_factory=list)
    missing_facts: List[str] = Field(default_factory=list)


class MatchInput(BaseModel):
    match_id: str = ""
    game: str
    game_version: str
    player: Dict[str, Any] = Field(default_factory=dict)
    outcome: Dict[str, Any] = Field(default_factory=dict)
    played_at: str = ""
    rounds: List[RoundInput]
    #: **BP（选人禁用）记录**，可选。
    #:
    #: 它属于这一局的数据，但不在局内回合里 —— BP 在比赛之前。
    #: 不在这里声明的话，pydantic 会**静默丢掉**整个 bp 段：
    #: 接口照样返回 200，对局也存进去了，到了 BP 复盘页却显示
    #: 「还没录 BP」，而没人会想到是接口把它吃了。
    bp: Optional[Dict[str, Any]] = None


class GuideExtractInput(BaseModel):
    text: str
    game: str
    version: str
    guide_title: str
    guide_url: str = ""
    guide_kind: str = "community"
    min_confidence: float = 0.4


class GuideConfirmInput(BaseModel):
    game: str
    version: str
    guide_title: str
    guide_url: str = ""
    accept: List[str] = Field(
        default_factory=list,
        description="人工确认放行的候选标题；空列表表示一条都不要",
    )


class VersionDiffInput(BaseModel):
    game: str
    old_version: str
    new_version: str
    player_id: str = "local"


# ==========================================================================
# 元信息
# ==========================================================================


@app.get("/", response_class=PlainTextResponse)
def root() -> str:
    return (
        "基于攻略规范的个人对局执行度分析系统\n"
        f"已注册游戏：{'、'.join(sorted(REGISTRY)) or '（无）'}\n"
        "接口文档：/docs\n"
    )


@app.get("/api/games")
def api_games() -> Dict[str, Any]:
    return {"games": registered_games()}


@app.get("/api/games/{game}/rules")
def api_rules(
    game: str,
    version: Optional[str] = None,
    session: Session = Depends(db_session),
) -> Dict[str, Any]:
    _ensure_game(game)
    version = version or latest_version(session, game)
    if version is None:
        return {"game": game, "version": None, "rules": [], "count": {}}
    rules = get_rules(session, game, version)
    from .rules import library_source

    return {
        "game": game,
        "version": version,
        "count": count_rules(session, game, version),
        "rules": [r.to_dict() for r in rules],
        # 这批规则是从**哪一份**模式库来的（用户自己的仓库还是内置兜底）。
        # 「我在改的是哪一份」必须查得到 —— 否则改了半天没生效都不知道。
        "library": library_source.describe(),
    }


@app.get("/api/games/{game}/versions")
def api_versions(game: str, session: Session = Depends(db_session)) -> Dict[str, Any]:
    _ensure_game(game)
    versions = list_versions(session, game)
    return {"game": game, "versions": versions, "latest": versions[0] if versions else None}


@app.post("/api/games/{game}/rules/seed")
def api_seed_rules(game: str, session: Session = Depends(db_session)) -> Dict[str, Any]:
    _ensure_game(game)
    result = load_seed_library(session, game)
    # BP 阶段的库是另一个版本号，单独载入（strict 导入不许一次跨版本）。
    bp_result = load_bp_seed_library(session, game)
    if not result.accepted and not bp_result.accepted:
        # 一条都没载进来时还说「已载入规则库」，用户会以为页面该有内容却空着 ——
        # 实际是这款游戏还没做资料（模块留着，资料后补）。说清楚是哪一种。
        return {
            "message": f"{game} 还没有操作模式库资料",
            "empty": True,
            **result.to_dict(),
        }
    return {
        "message": f"已载入 {game} 的操作模式库（{len(result.accepted)} 条）",
        "empty": False,
        # 这里的 accepted 是**局内库**的条数（BP 库另算）——
        # 调用方（校验脚本、界面）关心的是「局内模式库有多少条」，
        # 把两个口径混在一起就说不清了。
        **result.to_dict(),
        "bp": {
            "version": bp_result.version,
            "accepted_count": len(bp_result.accepted),
            "rejected_count": len(bp_result.rejected),
        },
    }


class RuleWriteBody(BaseModel):
    """人工写规则。字段与候选结构一致，方便前端复用同一套编辑器。"""

    game: str
    version: str
    title: str
    category: str = "未分类"
    trigger: Dict[str, Any] = Field(default_factory=dict)
    expect: Dict[str, Any] = Field(default_factory=dict)
    severity: str = "medium"
    rationale: str = ""
    deviation_hint: str = ""
    source_guide: str = ""
    source_url: str = ""


class RulesBulkBody(BaseModel):
    game: str
    version: str
    guide_title: str = ""
    rules: List[Dict[str, Any]] = Field(default_factory=list)


class SourceBody(BaseModel):
    game: str
    title: str
    url: str = ""
    kind: str = "user"
    note: str = ""


@app.post("/api/guides/rules")
def api_write_rule(payload: RuleWriteBody, session: Session = Depends(db_session)
                    ) -> Dict[str, Any]:
    """手写一条规则入库。

    和抽取候选走同一套准入校验 —— 手写不代表能跳过。
    唯一放宽的是主观词：人写「尽量」可能是有意的口语化，
    而抽取出来的文字里出现「尽量」通常意味着本来就没法判定。
    """
    _ensure_game(payload.game)
    try:
        out = write_rule(session, payload.game, payload.version, payload.model_dump())
        upsert_rule(session, out["rule"])
    except RuleWriteError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {
        "rule_id": out["rule"].id,
        "warnings": out["warnings"],
        "message": "已写入",
    }


@app.post("/api/guides/rules/bulk")
def api_write_rules_bulk(payload: RulesBulkBody, session: Session = Depends(db_session)
                         ) -> Dict[str, Any]:
    """批量写入（攻略页勾选候选后走这里）。"""
    _ensure_game(payload.game)
    return write_rules_bulk(session, payload.game, payload.version, payload.rules)


@app.post("/api/guides/sources")
def api_add_source(payload: SourceBody, session: Session = Depends(db_session)
                   ) -> Dict[str, Any]:
    """登记一个来源。

    登记不等于授权抓取 —— 抓取仍走采集器的域名白名单。两者刻意分开：
    「知道这个站点存在」和「允许程序去访问它」是两件不同的事。
    """
    _ensure_game(payload.game)
    try:
        out = add_source(
            session, game=payload.game, title=payload.title,
            url=payload.url, kind=payload.kind, note=payload.note,
        )
    except RuleWriteError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return out


class MobaExtractInput(BaseModel):
    """MOBA 攻略抽取。

    抽出来的是**候选**，仍然要人工确认才入库 —— 这条链路上没有旁路。
    """

    text: str = ""
    url: str = ""
    allowed_hosts: List[str] = Field(default_factory=list)
    game: str = "hok"
    version: str = "S38"
    guide_title: str = ""
    min_confidence: float = 0.4
    #: 预览模式（不建采集会话）—— 只想看看能抽出什么时用
    preview: bool = True


class MobaConfirmInput(BaseModel):
    """人工确认 MOBA 候选后入库。

    直接吃 ``MobaCandidate.to_dict()`` 的输出，所以前端可以原样回传，
    中间不改结构 —— 改了就对不上了，还很难发现。
    """

    game: str = "hok"
    version: str = "S38"
    guide_title: str = ""
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    #: 预览模式下算出来的稳定 id，按标题对齐回传
    ids: Dict[str, str] = Field(default_factory=dict)


@app.post("/api/guides/moba/extract")
def api_moba_extract(payload: MobaExtractInput,
                     session: Session = Depends(db_session)) -> Dict[str, Any]:
    """从攻略抽取「阶段 × 位置 → 该做什么」。

    缺位置 / 缺阶段 / 缺可判定动作的句子会被丢弃，并在 ``skipped``
    里逐条说明缺什么 —— 用户据此判断哪些得手工补，而不是干等工具出结果。
    """
    _ensure_game(payload.game)
    from .collection.collector import HttpGuideCollector, TextCollector
    from .rules.moba_extract import explain_skips, extract_moba, stable_rule_id

    if payload.url:
        if not payload.allowed_hosts:
            raise HTTPException(
                status_code=422,
                detail="抓取网页必须声明来源站点（allowed_hosts），默认拒绝一切域名",
            )
        try:
            doc = HttpGuideCollector(
                allowed_hosts=payload.allowed_hosts
            ).fetch(payload.url, game=payload.game)
            text = doc.text
            title = payload.guide_title or doc.title
        except Exception as exc:                       # noqa: BLE001
            raise HTTPException(status_code=422, detail=str(exc))
    else:
        doc = TextCollector({"t": payload.text}).fetch(
            "t", game=payload.game, kind="user")
        text = doc.text
        title = payload.guide_title or "手工录入"

    adapter = get_adapter(payload.game)
    cands = extract_moba(
        text,
        known_facts=adapter.known_facts(),
        version=payload.version,
        game=payload.game,
        guide_title=title,
        guide_url=payload.url,
        min_confidence=payload.min_confidence,
    )

    ids = {c.title: stable_rule_id(c) for c in cands}
    return {
        "game": payload.game,
        "version": payload.version,
        "guide_title": title,
        "candidates": [c.to_dict() for c in cands],
        "ids": ids,
        "skipped": explain_skips(text, adapter.known_facts()),
        "total_sentences": len([s for s in text.split("。") if s.strip()]),
        "message": f"抽出 {len(cands)} 条候选",
    }


@app.post("/api/guides/moba/confirm")
def api_moba_confirm(payload: MobaConfirmInput,
                     session: Session = Depends(db_session)) -> Dict[str, Any]:
    """人工确认 MOBA 候选并入库。

    候选的 ``to_dict()`` 结构与 :func:`write_rule` 期望的入参一致，
    所以这里可以直接转交 —— 不重新组装，也就不会两边字段对不上。

    准入校验和自走棋抽取完全同一套：未注册事实、期望动作不是动作事实、
    缺触发条件，一律拦住。
    """
    from .rules.authoring import write_rules_bulk

    guide = payload.guide_title or "攻略抽取"
    rules = []
    skipped_no_position = 0
    for raw in payload.candidates:
        # MOBA 专属约束：规则必须挂位置。
        # 「前期该控龙」不说清是哪个位置，对每个位置都成立 ——
        # 那是一条没有分工的规则，进了库只会稀释 MOBA 规则库。
        conds = (raw.get("trigger") or {}).get("all") or []
        if not any(c.get("fact") == "position" for c in conds):
            skipped_no_position += 1
            continue

        rule = dict(raw)                       # 浅拷贝，不改调用方的数据
        rule["category"] = "{}·{}".format(
            raw.get("position_label", ""), raw.get("phase_label", "")
        )
        rule["source_guide"] = guide
        rules.append(rule)

    result = write_rules_bulk(
        session, payload.game, payload.version, rules
    )
    result["skipped_no_position"] = skipped_no_position
    if skipped_no_position:
        result["message"] += f"，{skipped_no_position} 条因未指定位置被拒"
    return result


@app.get("/api/guides/sources")
def api_guide_sources(
    game: Optional[str] = None,
    session: Session = Depends(db_session),
) -> Dict[str, Any]:
    srcs = list_guide_sources(session, game)
    return {
        "sources": [
            {
                "game": s.game,
                "title": s.title,
                "url": s.url,
                "kind": s.kind,
                "note": s.note,
                "collected_at": s.collected_at.isoformat() if s.collected_at else "",
            }
            for s in srcs
        ]
    }


# ==========================================================================
# 采集层：对局
# ==========================================================================


@app.post("/api/matches")
def api_ingest_match(
    payload: MatchInput,
    player_id: str = Query("local"),
    session: Session = Depends(db_session),
) -> Dict[str, Any]:
    _ensure_game(payload.game)
    adapter = get_adapter(payload.game)

    raw = payload.model_dump()
    raw["rounds"] = [r for r in raw["rounds"]]

    try:
        record, issues = match_col.save_match(
            session, adapter, raw, player_id=player_id, source_kind="manual"
        )
    except match_col.MatchValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    report = _run_analysis(session, record, payload.game)

    return {
        "match_id": record.id,
        "game": record.game,
        "game_version": record.game_version,
        "issues": [i.to_dict() for i in issues],
        "stats": report["stats"],
        "gap_count": len(report["gaps"]),
        # 分路结论跟着一起返回 —— 调用方拿到「哪条分路哪一段做错了、
        # 应该怎么做」，不必再自己去取一次报告页面
        "lanes": report["lanes"],
        "report_url": f"/api/matches/{record.id}/report",
    }


def _run_analysis(session: Session, record, game: str) -> Dict[str, Any]:
    """对已落库的对局执行分析并持久化结果快照。"""
    from .db import Analysis

    adapter = get_adapter(game)
    ctx = adapter.build_context(record.get_payload())
    version = record.game_version
    rules = get_rules(session, game, version)

    if not rules:
        # 规则库为空时不做分析 —— 明确报错比给一份空报告诚实
        raise HTTPException(
            status_code=409,
            detail=f"{game} 版本 {version} 的规则库为空，请先载入规则库",
        )

    curve = _build_player_curve(session, game, record.player_id)
    report = build_match_report(rules, ctx, curve=curve)

    analysis = Analysis(
        match_id=record.id,
        rules_version=version,
        compliance_rate=report["stats"].get("compliance_rate"),
        triggered=report["stats"].get("triggered", 0),
        satisfied=report["stats"].get("satisfied", 0),
        deviated=report["stats"].get("deviated", 0),
        undetermined=report["stats"].get("undetermined", 0),
    )
    analysis.set_detail(
        {"gaps": report["gaps"], "by_category": report["by_category"]}
    )
    session.add(analysis)
    session.flush()

    return report


def _build_player_curve(session: Session, game: str, player_id: str) -> Dict[str, Any]:
    """用该玩家全部历史对局的分析结果构建成长曲线。"""
    records = match_col.list_matches(session, player_id=player_id, game=game)
    analyses: List[Dict[str, Any]] = []

    for record in reversed(records):   # 按时间正序
        snap = (
            record.analyses[0]
            if record.analyses
            else None
        )
        if snap is None:
            continue
        detail = snap.get_detail()
        analyses.append(
            {
                "match_id": record.id,
                "played_at": record.played_at.isoformat() if record.played_at else "",
                "game": record.game,
                "game_version": record.game_version,
                "compliance_rate": snap.compliance_rate,
                "triggered": snap.triggered,
                "satisfied": snap.satisfied,
                "deviated": snap.deviated,
                "outcome": record.get_outcome(),
                "gaps": detail.get("gaps", []),
                "verdicts": [],
            }
        )
    return build_curve(analyses)


@app.get("/api/matches")
def api_list_matches(
    player_id: str = Query("local"),
    game: Optional[str] = None,
    limit: int = Query(50, le=500),
    session: Session = Depends(db_session),
) -> Dict[str, Any]:
    records = match_col.list_matches(session, player_id=player_id, game=game, limit=limit)
    return {
        "matches": [
            {
                "match_id": r.id,
                "game": r.game,
                "game_version": r.game_version,
                "played_at": r.played_at.isoformat() if r.played_at else "",
                "outcome": r.get_outcome(),
                "source_kind": r.source_kind,
            }
            for r in records
        ]
    }


# ==========================================================================
# 采集层：视频识别结果接入
# ==========================================================================


class DetectedFactInput(BaseModel):
    name: str
    value: Any = None
    confidence: float = 1.0
    method: str = "roi"
    failure: str = ""


class DetectedActionInput(BaseModel):
    kind: str
    value: Any = None
    label: str = ""
    confidence: float = 1.0
    ts: float = 0.0


class RecognizedRoundInput(BaseModel):
    stage: int
    phase: Union[int, str] = 1
    label_override: str = ""
    time_index: float = 0.0
    facts: List[DetectedFactInput] = Field(default_factory=list)
    actions: List[DetectedActionInput] = Field(default_factory=list)
    gate_confidence: float = 1.0


class GateFrameInput(BaseModel):
    index: int
    timestamp: float = 0.0
    is_boundary: bool = False
    confidence: float = 1.0


class RecognitionInput(BaseModel):
    """视频识别结果的录入入口。

    识别器（无论来自真实视频管线还是测试）产出这个结构，服务端负责
    **降级成内核能消费的对局数据**。降级规则见
    :func:`app.collection.video.to_raw_match`。
    """

    match_id: str
    game: str
    game_version: str
    player: Dict[str, Any] = Field(default_factory=dict)
    outcome: Dict[str, Any] = Field(default_factory=dict)
    played_at: str = ""
    recognizer_id: str = ""
    recognizer_version: str = ""
    frames: List[GateFrameInput] = Field(default_factory=list)
    rounds: List[RecognizedRoundInput] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


@app.post("/api/matches/from-recognition")
def api_ingest_recognition(
    payload: RecognitionInput,
    player_id: str = Query("local"),
    analyze: bool = Query(True, description="是否立即用规则库分析"),
    session: Session = Depends(db_session),
) -> Dict[str, Any]:
    """接收视频识别结果，降级入库并（可选）立即分析。

    返回体里的 ``degradation`` 是这套接口的重点：它如实说明哪些事实
    被采信、哪些因为识别不可靠被丢进 ``missing_facts``、以及为什么。
    识别质量不达标时，调用方能看见原因，而不是拿到一份「看起来完整」
    但混着猜测的报告。
    """
    _ensure_game(payload.game)
    adapter = get_adapter(payload.game)

    result = RecognitionResult(
        match_id=payload.match_id,
        game=payload.game,
        game_version=payload.game_version,
        frames=[
            GateFrame(
                index=f.index, timestamp=f.timestamp,
                is_boundary=f.is_boundary, confidence=f.confidence,
            )
            for f in payload.frames
        ],
        rounds=[
            RecognizedRound(
                stage=r.stage,
                phase=_coerce_phase(r.phase),
                label_override=r.label_override,
                time_index=r.time_index,
                facts=[
                    DetectedFact(
                        name=d.name, value=d.value,
                        confidence=d.confidence, method=d.method,
                        failure=d.failure,
                    )
                    for d in r.facts
                ],
                actions=[
                    DetectedAction(
                        kind=a.kind, value=a.value, label=a.label,
                        confidence=a.confidence, ts=a.ts,
                    )
                    for a in r.actions
                ],
                gate_confidence=r.gate_confidence,
            )
            for r in payload.rounds
        ],
        stats=payload.stats,
        warnings=payload.warnings,
    )
    result.recognizer_id = payload.recognizer_id
    result.recognizer_version = payload.recognizer_version

    raw, degradation = to_raw_match(
        result, adapter.known_facts(), adapter.known_action_facts()
    )
    raw["player"] = dict(payload.player)
    raw["outcome"] = dict(payload.outcome)
    raw["played_at"] = payload.played_at

    try:
        record, issues = match_col.save_match(
            session, adapter, raw, player_id=player_id, source_kind="video"
        )
    except match_col.MatchValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    response: Dict[str, Any] = {
        "match_id": record.id,
        "source_kind": record.source_kind,
        "issues": [i.to_dict() for i in issues],
        "degradation": degradation.to_dict(),
        "cost": summarize_cost(result),
    }

    if analyze:
        report = _run_analysis(session, record, payload.game)
        response["stats"] = report["stats"]
        response["gap_count"] = len(report["gaps"])
        response["low_confidence_facts"] = report.get("low_confidence_facts", [])
        response["report_url"] = f"/api/matches/{record.id}/report"
    return response


def _coerce_phase(value: Union[int, str]) -> int:
    """把录入的 phase 规整成内核用的整数坐标。

    自走棋的小阶段本来就是 1/2/3；MOBA 传的是 early/mid/late 这类枚举，
    而内核排序需要数值，这里按已知枚举映射，取不到就退化为 1。
    """
    if isinstance(value, int):
        return value
    return {"early": 1, "mid": 2, "late": 3, "endgame": 4}.get(
        str(value).lower(), 1
    )


# ==========================================================================
# 规则库人工审核
# ==========================================================================


class RuleUpdateInput(BaseModel):
    """人工修改一条规则。

    只接受 :data:`app.rules.review.EDITABLE_FIELDS` 里的字段。改 id、
    游戏、版本会破坏规则库的主键语义和版本隔离，那不是审核该做的事。
    """

    changes: Dict[str, Any] = Field(
        default_factory=dict, description="要修改的字段，仅传需要改的"
    )
    reviewer: str = ""
    note: str = ""


class RuleStatusInput(BaseModel):
    status: str = Field(description="active 或 deprecated")
    reviewer: str = ""
    note: str = ""


@app.get("/api/rules/review", response_class=HTMLResponse)
def api_review_page(
    game: str = Query(...),
    version: Optional[str] = Query(None),
    player_id: str = Query("local"),
    session: Session = Depends(db_session),
) -> HTMLResponse:
    """规则库审核工作台（网页）。"""
    _ensure_game(game)
    data = build_review_view(
        session, game, version=version, player_id=player_id
    )
    return HTMLResponse(render_review(data))


@app.get("/api/games/{game}/rules/review")
def api_review_data(
    game: str,
    version: Optional[str] = None,
    player_id: str = Query("local"),
    coverage: bool = Query(True, description="是否现算历史对局覆盖率"),
    session: Session = Depends(db_session),
) -> Dict[str, Any]:
    """审核工作台数据（JSON）。"""
    _ensure_game(game)
    return build_review_view(
        session, game, version=version,
        player_id=player_id, with_coverage=coverage,
    )


@app.get("/api/games/{game}/rules/review/coverage")
def api_rule_coverage(
    game: str,
    version: Optional[str] = None,
    player_id: str = Query("local"),
    session: Session = Depends(db_session),
) -> Dict[str, Any]:
    """只取覆盖率统计。

    现算而非读快照：快照只存了「偏离」，规则被触发但玩家做对了的情况
    没留痕，只看偏离会漏掉「这条规则一直在生效且玩家一直做对」。
    """
    _ensure_game(game)
    version = version or latest_version(session, game) or ""
    if not version:
        raise HTTPException(status_code=404, detail=f"{game} 尚无规则库")
    coverage = compute_coverage(session, game, version, player_id)
    return {"game": game, "version": version, **coverage_summary(coverage)}


@app.patch("/api/games/{game}/rules/{version}/{rule_id}")
def api_update_rule(
    game: str,
    version: str,
    rule_id: str,
    payload: RuleUpdateInput,
    session: Session = Depends(db_session),
) -> Dict[str, Any]:
    """人工修改规则。改完仍走可判定性校验 —— 人工改不等于绕过准入。"""
    _ensure_game(game)
    try:
        result = update_rule(
            session, game, version, rule_id,
            payload.changes, reviewer=payload.reviewer, note=payload.note,
        )
    except ReviewError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {
        "rule_id": result.rule_id,
        "changed_fields": result.changed_fields,
        "message": result.message,
    }


@app.post("/api/games/{game}/rules/{version}/{rule_id}/status")
def api_set_rule_status(
    game: str,
    version: str,
    rule_id: str,
    payload: RuleStatusInput,
    session: Session = Depends(db_session),
) -> Dict[str, Any]:
    """启用 / 停用一条规则。

    停用走 ``deprecated`` 而不是删除：历史对局复盘仍需当时的规则状态。
    """
    _ensure_game(game)
    try:
        result = set_status(
            session, game, version, rule_id, payload.status,
            reviewer=payload.reviewer, note=payload.note,
        )
    except ReviewError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {
        "rule_id": result.rule_id,
        "changed_fields": result.changed_fields,
        "message": result.message,
    }


@app.get("/api/games/{game}/rules/{version}/{rule_id}/history")
def api_rule_history(
    game: str,
    version: str,
    rule_id: str,
    session: Session = Depends(db_session),
) -> Dict[str, Any]:
    """某条规则的修改历史。"""
    _ensure_game(game)
    return {
        "rule_id": rule_id,
        "history": revision_history(session, game, version, rule_id),
    }


@app.get("/api/matches/{match_id}/report", response_class=HTMLResponse)
def api_report_html(
    match_id: str,
    player_id: str = Query("local"),
    session: Session = Depends(db_session),
) -> HTMLResponse:
    report = _load_report(session, match_id, player_id)
    return HTMLResponse(render_report(report))


@app.get("/api/matches/{match_id}/report.json")
def api_report_json(
    match_id: str,
    player_id: str = Query("local"),
    session: Session = Depends(db_session),
) -> Dict[str, Any]:
    return _load_report(session, match_id, player_id)


@app.get("/api/matches/{match_id}/report/export")
def api_report_export(
    match_id: str,
    format: str = Query("html", pattern="^(html|json)$"),
    player_id: str = Query("local"),
    session: Session = Depends(db_session),
) -> Response:
    """把这一局的报告导出成一个文件，留在用户自己电脑上。

    为什么要这个：报告是网页 —— 数据在服务器上，样式也指着服务器。
    用户真正想要的却是「把这一局的分析留在我自己电脑里」：
    发给队友看、存档、离线翻。所以导出物是**自包含**的单文件，
    样式和脚本都内联进去，双击就能打开，服务器关不关都无所谓。

    视频片段带不走 —— 片段是服务器上的文件。
    导出页顶部会写明这一点，不假装它能播。
    """
    report = _load_report(session, match_id, player_id)
    meta = report.get("meta") or {}
    title = meta.get("report_title") or "操作模式分析"
    stamp = meta.get("generated_at") or ""

    if format == "json":
        return Response(
            export_json(report),
            media_type="application/json; charset=utf-8",
            headers=attachment_headers(export_filename(title, stamp, "json")),
        )

    page = build_export_page(
        render_report(report),
        title=title,
        stamp=stamp,
        note="视频片段仍在服务器上，离线打开时放不了。",
    )
    return Response(
        page,
        media_type="text/html; charset=utf-8",
        headers=attachment_headers(export_filename(title, stamp, "html")),
    )


@app.delete("/api/matches/{match_id}")
def api_delete_match(
    match_id: str,
    player_id: str = Query("local"),
    session: Session = Depends(db_session),
) -> Dict[str, Any]:
    """删掉一局对局（含它的分析结果）。

    对局记录页每一条后面都有一个删除按钮 —— 录错的那一局、
    不想再看见的那一局，得能删掉，否则列表越积越长，
    最后没人愿意往下翻。

    删的是**这一局**，不是规则库、不是别的对局。找不到就 404，
    理由要说清楚：静默返回成功会让人以为删掉了、刷新却还在。
    """
    deleted = match_col.delete_match(session, match_id, player_id=player_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"对局 {match_id} 不存在")
    return {"message": f"已删除对局 {match_id}", "match_id": match_id}


def _load_report(session: Session, match_id: str, player_id: str) -> Dict[str, Any]:
    record = match_col.load_match(session, match_id, player_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"对局 {match_id} 不存在")
    snap = record.analyses[0] if record.analyses else None
    if snap is None:
        # 还没分析过 —— 现算一次并落库
        return _run_analysis(session, record, record.game)
    return _rebuild_report(session, record, snap)


# ==========================================================================
# BP（选人禁用）阶段
#
# BP 在比赛之前，判的是「这一步该禁谁、该选谁」。
# 它有自己的规则库版本（``S45-BP``）、自己的事实与动作、自己的复盘页 ——
# 跟局内报告分开，是因为两者的坐标根本不是一回事
# （局内是「第几分钟」，BP 是「哪个环节」）。
# ==========================================================================


class BpDecision(BaseModel):
    """BP 记录里的一条决策。"""

    side: str = Field(default="my", pattern="^(my|opp)$")
    phase: str
    action: str
    hero: str = ""


class BpRecordInput(BaseModel):
    """一局的 BP 记录。

    这里**只有玩家在 BP 记录页上真正答得出来的问题**：
    阵容缺了什么、每个环节动了谁。
    段位与预选冲突已从页面上移除，所以也不再收 ——
    收了也不会用（判定层不登记这两条事实）。
    旧客户端带这两个键上来，Pydantic 会直接忽略，不报错。
    """

    my_missing_frontline: Optional[bool] = None
    my_missing_engage: Optional[bool] = None
    my_missing_protect: Optional[bool] = None
    my_missing_damage: Optional[bool] = None
    opp_missing_frontline: Optional[bool] = None
    decisions: List[BpDecision] = Field(default_factory=list)


def _bp_rules(session: Session, game: str) -> List[Rule]:
    """取 BP 阶段的规则库。

    BP 库是**另一个版本号**，与局内库并存 —— 所以这里不能拿对局自己的
    ``rules_version``（那是局内版本）去查。
    """
    from .rules.store import BP_VERSION

    return get_rules(session, game, BP_VERSION)


def _build_bp_report(session: Session, record, match_id: str) -> Dict[str, Any]:
    adapter = get_adapter(record.game)
    payload = record.get_payload() or {}
    if not payload.get("bp"):
        raise HTTPException(
            status_code=404,
            detail=f"对局 {match_id} 还没有 BP 记录 —— 先录 BP 再看复盘",
        )
    rules = _bp_rules(session, record.game)
    if not rules:
        raise HTTPException(
            status_code=409,
            detail="BP 操作模式库是空的 —— 先载入规则库",
        )
    ctx = adapter.build_bp_context(payload)
    return build_bp_report(rules, ctx)


@app.get("/api/matches/{match_id}/bp/report.json")
def api_bp_report_json(
    match_id: str,
    player_id: str = Query("local"),
    session: Session = Depends(db_session),
) -> Dict[str, Any]:
    record = match_col.load_match(session, match_id, player_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"对局 {match_id} 不存在")
    return _build_bp_report(session, record, match_id)


@app.get("/api/matches/{match_id}/bp/report", response_class=HTMLResponse)
def api_bp_report_html(
    match_id: str,
    player_id: str = Query("local"),
    session: Session = Depends(db_session),
) -> HTMLResponse:
    record = match_col.load_match(session, match_id, player_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"对局 {match_id} 不存在")
    report = _build_bp_report(session, record, match_id)
    return HTMLResponse(render_bp_report(report, match_id=match_id))


@app.get("/api/matches/{match_id}/bp/report/export")
def api_bp_report_export(
    match_id: str,
    format: str = Query("html", pattern="^(html|json)$"),
    player_id: str = Query("local"),
    session: Session = Depends(db_session),
) -> Response:
    """BP 复盘的导出（自包含单文件），存到自己电脑上。

    没录 BP 的对局返回 404 —— 与 BP 复盘页同一个口径：
    「没有的东西」要说没有，不能导出一个空壳给人。
    """
    record = match_col.load_match(session, match_id, player_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"对局 {match_id} 不存在")
    report = _build_bp_report(session, record, match_id)
    meta = report.get("meta") or {}
    title = meta.get("report_title") or "BP 复盘"
    stamp = meta.get("generated_at") or ""

    if format == "json":
        return Response(
            export_json(report),
            media_type="application/json; charset=utf-8",
            headers=attachment_headers(export_filename(title, stamp, "json")),
        )

    page = build_export_page(
        render_bp_report(report, match_id=match_id, standalone=True),
        title=title,
        stamp=stamp,
    )
    return Response(
        page,
        media_type="text/html; charset=utf-8",
        headers=attachment_headers(export_filename(title, stamp, "html")),
    )


@app.get("/api/matches/{match_id}/bp", response_class=HTMLResponse)
def api_bp_form_html(
    match_id: str,
    player_id: str = Query("local"),
    session: Session = Depends(db_session),
) -> HTMLResponse:
    """BP 录入页。"""
    record = match_col.load_match(session, match_id, player_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"对局 {match_id} 不存在")
    bp = match_col.load_bp_record(session, match_id, player_id) or {}
    return HTMLResponse(render_bp_form(record, bp))


@app.post("/api/matches/{match_id}/bp")
def api_save_bp(
    match_id: str,
    body: BpRecordInput,
    player_id: str = Query("local"),
    session: Session = Depends(db_session),
) -> Dict[str, Any]:
    """录一局的 BP 记录。

    录完立刻能看复盘 —— 录 BP 是为了得到「这一步该怎么打」的结论，
    让玩家录完还要自己去找入口，是把结论藏起来了。
    """
    from .adapters.honor_of_kings import (
        BP_ACTION_LABELS,
        BP_PHASE_LABELS,
        BP_PHASE_LEGACY_LABELS,
    )

    record = match_col.load_match(session, match_id, player_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"对局 {match_id} 不存在")
    if record.game != "hok":
        raise HTTPException(
            status_code=400, detail="目前只有王者荣耀有 BP 操作模式库"
        )

    # 「进入 BP」那一环已经撤下（它的规则全依赖段位与预选），
    # 但旧记录里可能有这一步 —— 得容得下它，不然老记录连保存都保存不了。
    known_phases = {**BP_PHASE_LABELS, **BP_PHASE_LEGACY_LABELS}
    for d in body.decisions:
        if d.phase not in known_phases:
            raise HTTPException(
                status_code=422,
                detail=f"不认识的 BP 环节：{d.phase!r}（可用：{sorted(BP_PHASE_LABELS)}）",
            )
        if d.action not in BP_ACTION_LABELS:
            raise HTTPException(
                status_code=422,
                detail=f"不认识的 BP 操作类别：{d.action!r}",
            )

    payload = body.model_dump()
    payload["decisions"] = [d.model_dump() for d in body.decisions]
    if not match_col.save_bp_record(session, match_id, payload, player_id=player_id):
        raise HTTPException(status_code=404, detail=f"对局 {match_id} 不存在")

    report = _build_bp_report(session, record, match_id)
    return {
        "message": f"已录入 {len(body.decisions)} 条 BP 决策",
        "report_url": f"/api/matches/{match_id}/bp/report",
        "entries": len(report["entries"]),
    }



def _rebuild_report(session: Session, record, snap) -> Dict[str, Any]:
    """用当前规则库重建报告（规则库可能已更新到新版本）。"""
    adapter = get_adapter(record.game)
    ctx = adapter.build_context(record.get_payload())
    rules = get_rules(session, record.game, snap.rules_version)
    if not rules:
        raise HTTPException(
            status_code=409, detail=f"版本 {snap.rules_version} 的规则库为空"
        )
    curve = _build_player_curve(session, record.game, record.player_id)
    return build_match_report(rules, ctx, curve=curve)


# ==========================================================================
# 采集层：攻略
# ==========================================================================


# ==========================================================================
# 采集层：攻略自动采集
# ==========================================================================


class CollectInput(BaseModel):
    """采集请求。

    ``allowed_hosts`` 必须由调用方显式给出。采集器默认拒绝一切来源 ——
    一个分析工具主动去抓哪些站点不该是默认值，赶工期时最容易被忘掉，
    所以做成默认拒绝。
    """

    url: str
    game: str
    version: str
    allowed_hosts: List[str] = Field(default_factory=list)
    kind: str = "community"
    title: str = ""
    min_confidence: float = 0.4


#: 进程内采集会话，保存「采集 → 确认」之间的候选
_COLLECT_SESSIONS = guide_sessions._STORE


@app.delete("/api/guides/sources")
def api_delete_source(
    title: str = Query(...),
    game: str = Query(...),
    session: Session = Depends(db_session),
) -> Dict[str, Any]:
    n = delete_source(session, game, title)
    return {"deleted": n, "message": f"删除 {n} 条来源"}


@app.post("/api/guides/collect")
def api_collect_guide(
    payload: CollectInput,
    session: Session = Depends(db_session),
) -> Dict[str, Any]:
    """采集一篇攻略并抽取候选 —— **不入库规则**。

    返回 ``session_id`` 与候选列表，后续用 ``/api/guides/collect/confirm``
    做人工确认入库。采集与确认刻意分两步：自动抽取必然有误差，
    让它能直接写规则库就等于取消了审核。
    """
    _ensure_game(payload.game)

    if not payload.allowed_hosts:
        raise HTTPException(
            status_code=422,
            detail="allowed_hosts 不能为空：采集来源必须显式声明，默认全部拒绝",
        )

    collector = HttpGuideCollector(allowed_hosts=payload.allowed_hosts)
    try:
        result = collect_and_extract(
            collector, session,
            source=payload.url,
            game=payload.game,
            version=payload.version,
            title=payload.title,
            kind=payload.kind,
            min_confidence=payload.min_confidence,
        )
    except SourceNotAllowed as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except CollectionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    session_id = uuid.uuid4().hex[:12]
    _COLLECT_SESSIONS.put(session_id, result)

    return {
        "session_id": session_id,
        "expires_in_seconds": int(guide_sessions.DEFAULT_TTL_SECONDS),
        "message": (
            f"已抽取 {len(result.candidates)} 条候选，"
            f"拦下 {len(result.rejected_subjective)} 条主观表述。确认后才会入库。"
        ),
        **result.to_dict(),
    }


class ConfirmCollectInput(BaseModel):
    """确认采集候选。

    整个请求体是一个模型而不是散装参数：``session_id`` 若单独作为函数签名
    的普通参数，FastAPI 会当成查询参数，结果所有 JSON 请求全部 422。
    踩过一次就固定成模型。
    """

    session_id: str
    #: 允许为空 —— 空列表表达「一条都不要」，是明确的表态而非缺省
    accept: List[str] = Field(default_factory=list)


@app.post("/api/guides/collect/confirm")
def api_confirm_collected(
    payload: ConfirmCollectInput,
    session: Session = Depends(db_session),
) -> Dict[str, Any]:
    """确认采集候选并入库规则。

    ``accept`` 传候选标题列表；空列表表示一条都不要。
    不重新抓取正文 —— 采集时的抽取结果已在会话里，这也避免了
    「你看到的是 A、入库的是 B」这种站点内容漂移。
    """
    stored = _COLLECT_SESSIONS.get(payload.session_id)
    if stored is None:
        raise HTTPException(
            status_code=404, detail="采集会话不存在或已过期，请重新采集"
        )

    outcome = confirm_collected(session, stored.result, payload.accept)
    _COLLECT_SESSIONS.drop(payload.session_id)
    return {"session_id": payload.session_id, **outcome}


@app.post("/api/guides/text")
def api_guide_from_text(
    payload: GuideExtractInput,
    session: Session = Depends(db_session),
) -> Dict[str, Any]:
    """从现成文本抽取候选（用户粘贴 / 测试）。与采集走同一条下游路径。"""
    _ensure_game(payload.game)
    collector = TextCollector({payload.guide_title or "user": payload.text})
    try:
        result = collect_and_extract(
            collector, session,
            source=payload.guide_title or "user",
            game=payload.game,
            version=payload.version,
            title=payload.guide_title,
            kind="user",
            min_confidence=payload.min_confidence,
        )
    except CollectionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    session_id = uuid.uuid4().hex[:12]
    _COLLECT_SESSIONS.put(session_id, result)
    return {"session_id": session_id, **result.to_dict()}


@app.post("/api/guides/extract")
def api_extract_guide(
    payload: GuideExtractInput,
    session: Session = Depends(db_session),
) -> Dict[str, Any]:
    _ensure_game(payload.game)
    result = guide_col.ingest_guide(
        session,
        text=payload.text,
        game=payload.game,
        version=payload.version,
        guide_title=payload.guide_title,
        guide_url=payload.guide_url,
        guide_kind=payload.guide_kind,
        accept=None,               # 预览模式：只抽取不入库
        min_confidence=payload.min_confidence,
    )
    return result.to_dict()


@app.post("/api/guides/confirm")
def api_confirm_guide(
    payload: GuideConfirmInput,
    session: Session = Depends(db_session),
) -> Dict[str, Any]:
    _ensure_game(payload.game)
    if not payload.accept:
        raise HTTPException(
            status_code=422,
            detail="accept 为空：没有人工确认放行的条目，本次不入库",
        )
    result = guide_col.ingest_guide(
        session,
        text="",
        game=payload.game,
        version=payload.version,
        guide_title=payload.guide_title,
        guide_url=payload.guide_url,
        accept=payload.accept,
    )
    return {
        "imported": result.imported,
        "failed": result.failed,
        "message": f"已入库 {len(result.imported)} 条规则",
    }


# ==========================================================================
# 成长曲线与版本适配
# ==========================================================================


@app.get("/api/growth/{game}")
def api_growth(
    game: str,
    player_id: str = Query("local"),
    session: Session = Depends(db_session),
) -> Dict[str, Any]:
    _ensure_game(game)
    version = latest_version(session, game) or ""
    return build_growth_report(
        _player_analyses(session, game, player_id), game, version
    )


@app.get("/api/growth/{game}/report", response_class=HTMLResponse)
def api_growth_report(
    game: str,
    player_id: str = Query("local"),
    session: Session = Depends(db_session),
) -> HTMLResponse:
    """成长曲线的网页报告。

    与单局报告共用同一套模板 —— 成长报告的图表主体是执行度曲线，
    只给 JSON 的话玩家还得自己找工具看，不算交付完整。
    """
    _ensure_game(game)
    version = latest_version(session, game) or ""
    report = build_growth_report(
        _player_analyses(session, game, player_id), game, version
    )
    return HTMLResponse(render_report(report))


def _player_analyses(session: Session, game: str, player_id: str) -> List[Dict[str, Any]]:
    records = match_col.list_matches(session, player_id=player_id, game=game)
    out: List[Dict[str, Any]] = []
    for record in reversed(records):
        if not record.analyses:
            continue
        snap = record.analyses[0]
        detail = snap.get_detail()
        out.append(
            {
                "match_id": record.id,
                "played_at": record.played_at.isoformat() if record.played_at else "",
                "game": record.game,
                "game_version": record.game_version,
                "compliance_rate": snap.compliance_rate,
                "triggered": snap.triggered,
                "satisfied": snap.satisfied,
                "deviated": snap.deviated,
                "outcome": record.get_outcome(),
                "gaps": detail.get("gaps", []),
                "verdicts": [],
            }
        )
    return out


def _rebuild_contexts(
    session: Session, game: str, player_id: str, version: Optional[str] = None
) -> List:
    """从持久化的原始数据重建历史对局上下文。

    这是版本回测能成立的关键：回测要比对新旧两套规则，就必须拿**同一批
    原始对局数据**各判一次，而不是复用旧版本的判定结果。原始数据一直
    保留在 ``Match.payload`` 里（采集层的设计前提），所以重建是可行的。

    ``version`` 缺省时回溯到所有版本 —— 新版本刚建立时，玩家历史对局都
    跑在旧版本上，那正是回测最该覆盖的场景。
    """
    records = match_col.list_matches(session, player_id=player_id, game=game)
    contexts = []
    for record in reversed(records):
        if version and record.game_version != version:
            continue
        try:
            ctx = MatchContext.from_dict(record.get_payload())
        except Exception:                     # noqa: BLE001
            # 单局重建失败不该毁掉整批回测，跳过并继续
            continue
        if ctx.rounds:
            contexts.append(ctx)
    return contexts


@app.post("/api/versions/{game}/diff")
def api_version_diff(
    game: str,
    payload: VersionDiffInput,
    session: Session = Depends(db_session),
) -> Dict[str, Any]:
    """比较两个版本的规则库，并用玩家历史对局回测变更规则的影响。"""
    _ensure_game(game)
    old_rules = get_rules(session, game, payload.old_version, include_deprecated=True)
    new_rules = get_rules(session, game, payload.new_version, include_deprecated=True)
    if not old_rules:
        raise HTTPException(status_code=404, detail=f"旧版本 {payload.old_version} 规则库为空")
    if not new_rules:
        raise HTTPException(status_code=404, detail=f"新版本 {payload.new_version} 规则库为空")

    contexts = _rebuild_contexts(session, game, payload.player_id)
    return build_version_report(game, old_rules, new_rules, contexts)


class VersionActivateInput(BaseModel):
    """版本切换：把某个新版本设为生效版本并标记失效条目。"""

    game: str
    from_version: str
    to_version: str
    player_id: str = "local"
    dry_run: bool = Field(
        default=False, description="只预览将要发生的变更，不实际写库"
    )


@app.post("/api/versions/{game}/activate")
def api_activate_version(
    game: str,
    payload: VersionActivateInput,
    session: Session = Depends(db_session),
) -> Dict[str, Any]:
    """执行版本切换：回测 → 标记失效条目 → 返回可复查的切换报告。

    顺序是有意的：先回测再落库。回测会告诉玩家哪些老习惯该改、哪些可以
    放松，这些结论必须能在「切换前」看到；一旦写入就再也分不清了。
    """
    _ensure_game(game)
    old_rules = get_rules(session, game, payload.from_version, include_deprecated=True)
    new_rules = get_rules(session, game, payload.to_version, include_deprecated=True)
    if not old_rules:
        raise HTTPException(status_code=404, detail=f"旧版本 {payload.from_version} 规则库为空")
    if not new_rules:
        raise HTTPException(status_code=404, detail=f"新版本 {payload.to_version} 规则库为空")

    contexts = _rebuild_contexts(session, game, payload.player_id)
    report = build_version_report(game, old_rules, new_rules, contexts)

    # 标记失效：新版本中不再存在的条目降级为 deprecated（不删除，
    # 历史对局复盘仍需当时的规则状态）
    active_ids = [r.id for r in new_rules if r.status == "active"]
    if payload.dry_run:
        marked = [r.id for r in old_rules
                  if r.status == "active" and r.id not in set(active_ids)]
        return {
            "dry_run": True,
            "would_deprecate": marked,
            "backtest_summary": report["backtest_summary"],
            "version_note": report["version_note"],
            "message": (
                f"预览：将 {len(marked)} 条规则标记为失效，"
                f"生效版本切到 {payload.to_version}"
            ),
        }

    marked = deactivate_missing(
        session, game, payload.to_version, active_ids
    )

    return {
        "dry_run": False,
        "activated_version": payload.to_version,
        "deprecated": marked,
        "backtest": report["backtest"],
        "backtest_summary": report["backtest_summary"],
        "version_diff": report["version_diff"],
        "version_note": report["version_note"],
        "message": (
            f"已切换到 {payload.to_version}，"
            f"标记 {len(marked)} 条规则失效"
        ),
    }


@app.get("/api/versions/{game}/backtest")
def api_version_backtest(
    game: str,
    old_version: str = Query(...),
    new_version: str = Query(...),
    player_id: str = Query("local"),
    as_html: bool = Query(False, description="返回 HTML 报告而非 JSON"),
    session: Session = Depends(db_session),
) -> Any:
    """回测报告。可直接出 HTML，与单局报告共用同一套模板。"""
    _ensure_game(game)
    old_rules = get_rules(session, game, old_version, include_deprecated=True)
    new_rules = get_rules(session, game, new_version, include_deprecated=True)
    if not old_rules:
        raise HTTPException(status_code=404, detail=f"版本 {old_version} 规则库为空")
    if not new_rules:
        raise HTTPException(status_code=404, detail=f"版本 {new_version} 规则库为空")

    contexts = _rebuild_contexts(session, game, player_id)
    report = build_version_report(game, old_rules, new_rules, contexts)

    if as_html:
        return HTMLResponse(render_report(report))
    return report


# ==========================================================================
# 辅助
# ==========================================================================


def _ensure_game(game: str) -> None:
    if game not in REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"未注册的游戏 {game!r}，已注册：{sorted(REGISTRY)}",
        )


def cli() -> None:
    """本地启动入口：``python run.py``。"""
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    cli()
