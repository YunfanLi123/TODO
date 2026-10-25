"""上传即出报告：**中间没有让人填的一步**。

整条链路是：

    上传 → 后台扫描（模型读画面） → finalize 建对局 → 报告

以前中间夹着一个「逐段标注」页：分路、胜负、BP、每段操作全靠人点。
现在这些都由模型从画面里读，读不出来的记「读不出来」——
**不回头让人填**（用户原话：「我只要模型去读出报告，其余一切步骤都省去」）。

盯的是几件「错了也不报错」的事：

1. 扫描完了就得出报告 —— 不跳标注页。
2. ``finalize`` **幂等**：刷新页面、重试提交都走它，不能建出第二局。
3. 报告要写清楚「这局结论是哪个模型读出来的」——
   换模型就换结论，读者得知道。
4. **没配模型也得能出报告** —— 少一环功能就整个瘫掉，是最糟的降级。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def client(tmp_path, monkeypatch):
    """独立临时库 + 独立数据目录的 TestClient。"""
    from fastapi.testclient import TestClient

    from app import db as dbmod
    from app.db import session_scope
    from app.main import app
    from app.rules.store import latest_version, load_seed_library

    dbmod._engine = None
    dbmod._SessionFactory = None
    dbmod.get_engine(str(tmp_path / "auto_flow.db"))
    monkeypatch.setenv("MATCH_DATA_DIR", str(tmp_path / "data"))
    for name in ("MATCH_VLM_API_KEY", "MATCH_VLM_ENDPOINT", "MATCH_VLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    with session_scope() as s:
        if not latest_version(s, "hok"):
            load_seed_library(s, "hok")

    yield TestClient(app)

    dbmod._engine = None
    dbmod._SessionFactory = None


def _fake_session(match_id: str = "hok-video-auto"):
    """往采集层挂一个假的扫描会话（结果在进程内存里，不落库）。"""
    from app.adapters.base import get_adapter
    from app.collection import video_api
    from app.collection import video_upload as vu

    video = vu.UploadedVideo(video_id="v-auto", path="", original_name="对局.mp4",
                             size=1, duration_sec=600.0, fps=30.0)
    shots = [
        {"round_no": 1, "phase": "early", "game_minute": 0.5, "timestamp": 30.0,
         "image": "phase-001.jpg", "image_url": "/api/videos/v-auto/frames/phase-001.jpg",
         "confidence": 1.0},
        {"round_no": 2, "phase": "mid", "game_minute": 5.5, "timestamp": 360.0,
         "image": "phase-002.jpg", "image_url": "/api/videos/v-auto/frames/phase-002.jpg",
         "confidence": 1.0},
    ]
    summary = {
        "video": video.to_dict(),
        "scan": {"note": "认出时钟", "fps": 30.0, "clock": {"ok": True}},
        "product": "phase",
        "game": "hok",
        "game_version": "S45",
        "round_count": len(shots),
        "rounds": shots,
        "bp": [],
        "bp_count": 0,
        "match_minutes": 10.0,
        "match_start_t": 0.0,
    }
    video_api._SESSIONS[match_id] = {
        "summary": summary,
        "skeleton": vu.build_skeleton(video, shots, "hok", "S45", match_id),
        "adapter": get_adapter("hok"),
        "video": video,
        "prefill": {"position": "野区", "result": "win", "hero": "梦奇"},
    }
    return video_api._SESSIONS[match_id]


# ---------------------------------------------------------------- 页面接线


def test_the_upload_page_goes_straight_to_the_report(client):
    """扫完 → finalize → 报告。中间没有标注页。"""
    html = client.get("/upload").text
    assert "/api/videos/finalize" in html, "上传页没有调用 finalize"
    assert "/annotate" not in html, "上传页还在跳标注页"
    assert "打开标注页" not in html, "上传页的文案还在说标注页"


def test_the_navigation_has_no_annotate_entry(client):
    """侧边栏不该有标注页的入口。"""
    for path in ("/upload", "/matches", "/model"):
        html = client.get(path).text
        assert "/annotate" not in html, path


def test_the_annotate_route_says_the_flow_changed(client):
    """旧书签要得到一个**说清来由**的答复，而不是 404。"""
    r = client.get("/videos/hok-video-auto/annotate")
    assert r.status_code == 410
    assert "自动出报告" in r.json()["detail"]


# ---------------------------------------------------------------- finalize


def test_finalize_creates_the_match_and_returns_the_report(client):
    match_id = "hok-video-auto"
    _fake_session(match_id)
    try:
        r = client.post("/api/videos/finalize", json={"match_id": match_id})
    finally:
        from app.collection import video_api

        video_api._SESSIONS.pop(match_id, None)

    assert r.status_code == 200, r.text[:400]
    data = r.json()
    assert data["match_id"]
    assert data["report_url"] == f"/api/matches/{data['match_id']}/report"

    # 报告真的打得开
    html = client.get(data["report_url"]).text
    assert "操作模式分析" in html


def test_finalize_is_idempotent(client):
    """刷新页面、重试提交都走这里 —— 不能建出第二局。"""
    match_id = "hok-video-auto"
    _fake_session(match_id)
    try:
        first = client.post("/api/videos/finalize", json={"match_id": match_id})
        second = client.post("/api/videos/finalize", json={"match_id": match_id})
    finally:
        from app.collection import video_api

        video_api._SESSIONS.pop(match_id, None)

    assert first.status_code == 200 and second.status_code == 200
    assert first.json()["match_id"] == second.json()["match_id"]
    assert first.json()["report_url"] == second.json()["report_url"]


def test_finalize_on_an_expired_session_says_so(client):
    """会话没了要明确说「重新上传」，而不是 404 掉在门口。"""
    r = client.post("/api/videos/finalize", json={"match_id": "nope"})
    assert r.status_code == 404
    assert "重新上传" in r.json()["detail"]


# ---------------------------------------------------------------- 报告的说法


def test_the_report_does_not_explain_where_the_conclusion_came_from(client):
    """报告**不摆**「结论是从画面里读出来的 · 读取模型 X」那类说明。

    用户看过之后说了一句：「这些描述性文字都去掉，我完全不需要。」

    理由也在这儿：标注页下线之后，动作**只有**一个来源 —— 模型从画面里读，
    没有第二个来源可对比，这句话每局都一样，说的都是废话。
    这一页要回答的是「哪一段做错了、该怎么做」。

    所以这里断言的是**它不在** —— 免得哪天又被加回来。
    """
    from app.collection import settings

    settings.save_slot("vlm", provider="deepseek", api_key="sk-x",
                        model="deepseek-v4-pro")
    match_id = "hok-video-auto"
    _fake_session(match_id)
    try:
        data = client.post("/api/videos/finalize",
                            json={"match_id": match_id}).json()
    finally:
        from app.collection import video_api

        video_api._SESSIONS.pop(match_id, None)

    html = client.get(data["report_url"]).text
    assert "从画面里读出来的" not in html, "报告又在解释结论的来源了"
    assert "不是玩家自述" not in html
    assert "读取模型" not in html
    assert "deepseek-v4-pro" not in html, "报告上不该摆模型名"
    assert "多模态模型" not in html
    # 该有的还在：结论本身
    assert "操作模式分析" in html


def test_without_a_model_the_report_still_comes_out(client):
    """**没配模型也得能出报告**。

    少一个可选配置就整个瘫掉，是最糟的降级 ——
    这时候报告照出，只是没有自动结论而已。
    """
    match_id = "hok-video-auto"
    _fake_session(match_id)
    try:
        r = client.post("/api/videos/finalize", json={"match_id": match_id})
    finally:
        from app.collection import video_api

        video_api._SESSIONS.pop(match_id, None)

    assert r.status_code == 200, r.text[:400]
    html = client.get(r.json()["report_url"]).text
    assert "操作模式分析" in html


def test_the_marks_come_from_the_model_not_from_the_player(client):
    """动作来源是 ``derived`` —— 不是玩家自述。

    这两种来源的结论可信度完全不同。报告不标出来，
    用户看到「视频中做了什么」会以为那是玩家自己说的 —— 那是替玩家编话。
    """
    match_id = "hok-video-auto"
    _fake_session(match_id)
    try:
        r = client.post("/api/videos/finalize", json={"match_id": match_id})
    finally:
        from app.collection import video_api

        video_api._SESSIONS.pop(match_id, None)

    payload = client.get(f"/api/matches/{r.json()['match_id']}/report.json").json()
    assert payload["meta"]["annotation_source"] == "derived"


# ---------------------------------------------------------------- 扫描 → 会话


def test_the_scan_job_carries_the_model_reading_into_the_session(client):
    """模型读出来的东西（分路/胜负/BP）要跟着摘要一路走到建对局。"""
    from app.collection import scan_jobs, video_api

    src = (APP_ROOT / "app" / "collection" / "scan_jobs.py").read_text("utf-8")
    assert "read_lane_and_result" in src
    assert "read_bp_steps" in src
    assert 'summary["prefill"] = prefill' in src
    assert '"prefill"' in (APP_ROOT / "app" / "collection" / "video_api.py").read_text("utf-8")


def test_the_session_is_ready_before_the_job_is_marked_done(client):
    """顺序要紧：先挂会话，再标完成。

    前端一看到 done 就去建对局。顺序反了的话那会儿会话还没挂上，
    建对局拿到的是「解析结果已失效」—— 用户白等一场。
    """
    src = (APP_ROOT / "app" / "collection" / "scan_jobs.py").read_text("utf-8")
    on_done = src.index("on_done(job)")
    done = src.index("job.set(status=DONE)")
    assert done > on_done, "先标完成再挂会话 —— 前端会撞一个空会话"
