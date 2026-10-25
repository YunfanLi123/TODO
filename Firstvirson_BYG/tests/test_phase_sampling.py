"""按阶段抽帧与「局内时间进内核」这条链路的测试。

盯的是两处**静默**的错位 —— 都不报错，只是让结论悄悄变空：

1. **局内时间没进事实**。视频链路早先把 ``time_min`` 漏在了骨架之外，
   适配器归一化时取不到，就把每一回合都当成第 0 分钟并标成缺失 ——
   所有带时间窗的模式一条都不触发，报告看起来只是「没结论」。
2. **``video_time`` 存成了游戏内分钟**。报告的「看这一段」拿它去定位
   播放器，于是永远跳到录屏开头，而页面上看不出任何异常。

这两条都靠真实数据才暴露得出来 —— 合成视频读不出时钟，也就没有阶段。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.adapters import registry as _registry  # noqa: F401  触发适配器注册
from app.collection import video_upload as vu
from app.collection.video import to_raw_match

APP_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- 工具


def _make_video(path: str, seconds: float = 30.0, fps: float = 10.0) -> str:
    """造一段有画面的视频：每帧底色随时间变化，门控才有东西可看。"""
    import cv2
    import numpy as np

    w, h = 320, 180
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not vw.isOpened():                      # pragma: no cover
        pytest.skip("mp4v 编码不可用")
    n = int(seconds * fps)
    for i in range(n):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :] = (30 + (i * 3) % 120, 40, 60)
        vw.write(img)
    vw.release()
    return path


def _clock_info(start_t: float = 10.0, end_t: float = 28.0) -> dict:
    """造一份「时钟读出来了」的信息。

    ``readable_times`` 是采集层用来确定「对局到哪一秒为止」的依据 ——
    录屏往往在对局结束后还继续着，那段不是对局。
    """
    times = [start_t + i * 0.5 for i in range(int((end_t - start_t) / 0.5) + 1)]
    return {
        "ok": True,
        "game_start_t": start_t,
        "readable_times": times,
        "slope": 1.0,
        "residual": 0.2,
    }


# ------------------------------------------------- 局内时间进内核


def test_skeleton_keeps_the_clock_minute_as_a_fact(tmp_path):
    """时钟读出的局内时间必须进事实，方法标成 clock。

    少了它，适配器归一化时取不到 time_min，就把每一回合都当成第 0 分钟
    并标成缺失 —— 所有带时间窗的模式在缺失事实上一律判不成立。
    表现是：接口正常、库是满的，报告却一条结论都没有。
    """
    video = vu.UploadedVideo(video_id="v1", path="x.mp4", original_name="x.mp4",
                             size=1, duration_sec=30.0)
    shots = [{
        "round_no": 1, "timestamp": 240.0, "game_minute": 0.7,
        "image": "phase-001.jpg", "confidence": 1.0,
    }]
    sk = vu.build_skeleton(video, shots, "hok", "S45", "m1")

    tm = [f for f in sk.rounds[0].facts if f.name == "time_min"]
    assert tm, "局内时间没有进事实 —— 带时间窗的模式会全部不触发"
    assert tm[0].value == 0
    assert tm[0].method == "clock", "这是读时钟读出来的，不是人填的"
    assert tm[0].confidence == 1.0


def test_video_time_is_the_video_second_not_the_game_minute(tmp_path):
    """``video_time`` 必须是视频里的秒数。

    报告的「看这一段」靠它定位播放器。早先它存的是游戏内分钟，
    于是「第 3 分钟」被当成「视频第 3 秒」—— 点开永远跳到录屏开头。
    """
    video = vu.UploadedVideo(video_id="v1", path="x.mp4", original_name="x.mp4",
                             size=1, duration_sec=30.0)
    shots = [{
        "round_no": 1, "timestamp": 240.0, "game_minute": 0.7,
        "image": "phase-001.jpg", "confidence": 1.0,
    }]
    sk = vu.build_skeleton(video, shots, "hok", "S45", "m1")
    assert sk.rounds[0].video_time == 240.0
    assert sk.rounds[0].time_index == pytest.approx(0.7)


def test_video_round_reaches_the_kernel_with_the_right_phase():
    """整条链：骨架 → 降级 → 适配器归一化，时间与阶段都要对。

    这是最值钱的一条断言 —— 它同时钉住「局内时间进了事实」和
    「阶段由局内时间推导」两件事。任何一环掉链子，报告就会静默变空。
    """
    from app.adapters.base import get_adapter

    adapter = get_adapter("hok")
    video = vu.UploadedVideo(video_id="v1", path="x.mp4", original_name="x.mp4",
                             size=1, duration_sec=900.0)
    shots = [
        {"round_no": 1, "timestamp": 240.0, "game_minute": 0.7,
         "image": "a.jpg", "confidence": 1.0},
        {"round_no": 2, "timestamp": 540.0, "game_minute": 5.7,
         "image": "b.jpg", "confidence": 1.0},
        {"round_no": 3, "timestamp": 900.0, "game_minute": 11.4,
         "image": "c.jpg", "confidence": 1.0},
    ]
    sk = vu.build_skeleton(video, shots, "hok", "S45", "m1")
    raw, _ = to_raw_match(sk, adapter.known_facts(), adapter.known_action_facts())

    ctx = adapter.build_context(raw)
    assert len(ctx.rounds) == 3
    assert ctx.rounds[0].facts["time_min"] == 0
    assert ctx.rounds[0].facts["phase"] == "early"
    assert ctx.rounds[1].facts["time_min"] == 5
    assert ctx.rounds[1].facts["phase"] == "mid"
    assert ctx.rounds[2].facts["time_min"] == 11
    assert ctx.rounds[2].facts["phase"] == "late"
    # 「看这一段」要落在视频里的真实位置
    assert ctx.rounds[1].video_time == 540.0
    # 局内时间是读出来的，不该被算成缺失
    assert "time_min" not in ctx.rounds[1].missing_facts


def test_a_round_without_economic_facts_does_not_crash():
    """没填经济差时不能抛异常。

    早先这里只判了 ``missing`` 集合，没判值本身是 None：
    ``facts.get("gold_diff")`` 返回 None，被原样传进派生函数，
    ``None <= -5000`` 直接 TypeError。

    而「用户不填经济差」恰恰是**默认状态** —— 标注页的数字框本来就
    是空的。所以这条链路早先必崩：上传扫描都正常，提交标注时报 500。
    """
    from app.adapters.base import get_adapter

    adapter = get_adapter("hok")
    video = vu.UploadedVideo(video_id="v1", path="x.mp4", original_name="x.mp4",
                             size=1, duration_sec=900.0)
    shots = [{
        "round_no": 1, "timestamp": 240.0, "game_minute": 0.7,
        "image": "a.jpg", "confidence": 1.0,
    }]
    sk = vu.build_skeleton(video, shots, "hok", "S45", "m1")
    raw, _ = to_raw_match(sk, adapter.known_facts(), adapter.known_action_facts())

    ctx = adapter.build_context(raw)          # 早先在这里抛 TypeError
    r = ctx.rounds[0]
    assert r.facts["time_min"] == 0
    # 经济差没填就是「没有这个事实」，不是 0
    assert "gold_diff" not in r.facts
    # 依赖经济差的派生事实因此判不了
    assert "pressure" in r.missing_facts
    assert r.facts["is_decision_point"] is True   # 第 0 分钟是时间片切换点


# ------------------------------------------------- 按阶段抽帧


def test_phase_sampling_returns_none_without_a_clock(tmp_path):
    """读不到时钟就不能按阶段抽 —— 返回 None，让调用方回退。

    按视频秒硬凑一个阶段是最容易犯也最难发现的错：每一条带时间窗的
    模式都会判到错误的阶段，而报告上只是「没触发」。
    """
    video = vu.UploadedVideo(video_id="v1", path="x.mp4", original_name="x.mp4",
                             size=1, duration_sec=30.0)
    assert vu.sample_by_phase(video, {"ok": False}, str(tmp_path),
                              {"early": [0, 4]}) is None
    assert vu.sample_by_phase(video, {"ok": True, "game_start_t": None},
                              str(tmp_path), {"early": [0, 4]}) is None


def test_phase_sampling_lands_inside_each_phase_window(tmp_path):
    """抽帧落点必须在该阶段的有效区间内。

    有效区间比规则里的窗口多一分钟：规则窗口是闭区间 ``between [0, 4]``，
    涵盖第 0~4 分钟；而 ``time_min`` 取整分，所以第 4 分钟实际覆盖
    4:00–4:59。窗口写 [0,4] 说的是「0–4 分这一段」，抽帧必须按 [0,5)
    来铺，否则第 4~5 分钟一个落点都没有。
    """
    path = _make_video(str(tmp_path / "s.mp4"), seconds=30.0, fps=10.0)
    video = vu.UploadedVideo(video_id="v1", path=path, original_name="s.mp4",
                             size=1, duration_sec=30.0, fps=10.0)

    # 对局从第 10 秒开始，到第 28 秒结束 —— 只有 18 秒，约 0.3 分钟
    result = vu.sample_by_phase(
        video, _clock_info(10.0, 28.0), str(tmp_path),
        {"early": [0, 4], "mid": [5, 10], "late": [11, 60]},
    )
    assert result is not None
    assert result["rounds"], "一个阶段都没抽到帧"

    for r in result["rounds"]:
        minute = r["game_minute"]
        assert minute is not None, "抽出来的帧必须有局内时间"
        if r["phase"] == "early":
            assert 0 <= minute < 5, f"前期帧落在了第 {minute} 分钟"
        elif r["phase"] == "mid":
            assert 5 <= minute < 11, f"中期帧落在了第 {minute} 分钟"
        else:
            assert minute >= 11, f"后期帧落在了第 {minute} 分钟"


def test_phase_sampling_also_produces_a_bp_sequence(tmp_path):
    """BP 帧序列是产物的一部分 —— 对局开始之前的那一段。

    它的价值是让用户看清双方各选了谁：英雄层模式要靠这个才触发，
    而对位优劣本来就只能由人判断。
    """
    path = _make_video(str(tmp_path / "s.mp4"), seconds=30.0, fps=10.0)
    video = vu.UploadedVideo(video_id="v1", path=path, original_name="s.mp4",
                             size=1, duration_sec=30.0, fps=10.0)

    result = vu.sample_by_phase(
        video, _clock_info(10.0, 28.0), str(tmp_path),
        {"early": [0, 4]}, bp_samples=4,
    )
    assert result is not None
    bp = result["bp"]
    assert len(bp) == 4, "BP 帧数量不对"
    assert all(f["timestamp"] <= 10.0 for f in bp), "BP 帧不该落在对局开始之后"
    assert all(f["image"].startswith("bp-") for f in bp)
    assert all(f["image_url"] for f in bp), "BP 帧必须有能打开的地址"


def test_phase_windows_come_from_the_adapter():
    """阶段窗口从适配器取，采集层不写死一份。

    两份窗口迟早会只改一处，然后出现「帧标中期、内核判前期」的错位。
    """
    from app.adapters.base import get_adapter

    ad = get_adapter("hok")
    assert ad.phase_time_windows, "MOBA 适配器必须声明阶段窗口"
    assert ad.phase_time_windows["early"] == [0, 4]
    assert ad.phase_time_windows["mid"] == [5, 10]
    assert ad.phase_time_windows["late"] == [11, 60]


# ------------------------------------------------- 标注页


def test_the_estimated_time_does_not_empty_the_report(tmp_path):
    """读不到时钟时：**时间**是推算的，但**帧**是确定的。

    这两件事必须分开记。记混了的后果是整局归零：
    推算的时间置信度只有 0.4，而回合门控的门槛是 0.50 ——
    一旦把「时间不准」记进门控置信度，每一个回合都会被判成
    「边界没数准」而整段丢弃，于是所有带时间窗的模式一条都不触发，
    报告上只是「没结论」，看不出任何异常。
    """
    from app.collection.video import (
        FACT_ACCEPT_THRESHOLD,
        FACT_FLAG_THRESHOLD,
        GATE_TRUST_THRESHOLD,
    )

    path = _make_video(str(tmp_path / "s.mp4"), seconds=30.0, fps=10.0)
    video = vu.UploadedVideo(video_id="v1", path=path, original_name="s.mp4",
                             size=1, duration_sec=30.0, fps=10.0)

    points = [(i * 0.3, 5.0 if 10 < i * 0.3 < 100 else 0.2) for i in range(400)]
    result = vu.moba_slice_product(
        video, points, str(tmp_path), {"early": [0, 4], "mid": [5, 10]},
        budget=4,
    )
    assert result and result["rounds"], "按信息密度切时间片没有产出"

    for r in result["rounds"]:
        # 帧是确定的 —— 门控置信度必须够，否则整段被丢
        assert r["confidence"] >= GATE_TRUST_THRESHOLD, (
            f"第 {r['round_no']} 片的门控置信度 {r['confidence']} 过低，"
            "这一局会被整段丢掉，报告只剩「没结论」"
        )
        assert r.get("time_source") == "estimate"

    # 推到内核里：时间事实要被采信（只是标成低置信度）
    sk = vu.build_skeleton(video, result["rounds"], "hok", "S45", "m1")
    tm = [f for f in sk.rounds[0].facts if f.name == "time_min"]
    assert tm, "推算的局内时间没有进事实"
    assert tm[0].confidence >= FACT_ACCEPT_THRESHOLD, "推算的时间被丢掉了"
    assert tm[0].confidence < FACT_FLAG_THRESHOLD, "推算的时间没有被标成低置信度"
    assert tm[0].method == "estimate"
    assert sk.rounds[0].gate_confidence >= GATE_TRUST_THRESHOLD


def test_the_phase_label_uses_the_whole_minute(tmp_path):
    """阶段按**整分钟**判，不能掉进两个窗口之间的空档。

    规则窗口是「前期 0~4 分、中期 5~10 分」这种整数分钟；
    拿 4.8 分去比，early 不含、mid 也不含 —— 阶段就成了空的，
    这一段在报告里归不到任何阶段。
    """
    path = _make_video(str(tmp_path / "s.mp4"), seconds=30.0, fps=10.0)
    video = vu.UploadedVideo(video_id="v1", path=path, original_name="s.mp4",
                             size=1, duration_sec=30.0, fps=10.0)

    points = [(i * 0.3, 5.0 if 10 < i * 0.3 < 100 else 0.2) for i in range(400)]
    result = vu.moba_slice_product(
        video, points, str(tmp_path), {"early": [0, 4], "mid": [5, 10]},
        budget=4,
    )
    assert result
    for r in result["rounds"]:
        assert r["phase"], f"第 {r['round_no']} 片没有阶段标签"


def test_bp_is_read_by_the_model_and_carried_into_the_match():
    """BP 不再由人看帧点选 —— 模型读，读出来的直接带进对局。

    以前 BP 帧序列摆在标注页最上面，让人一个一个点「禁谁、选谁」。
    现在那一页没了：BP 由模型从画面读（``read_bp_steps``），
    读不出来的记成缺失（不猜），读出来的跟着对局一起存下来。
    """
    from app.collection import scan_jobs, video_api

    src = (APP_ROOT / "app" / "collection" / "scan_jobs.py").read_text("utf-8")
    assert "read_bp_steps" in src, "扫描阶段没有让模型读 BP"
    assert '"prefill"' in src, "BP 读出来的东西没有跟着摘要走"

    api = (APP_ROOT / "app" / "collection" / "video_api.py").read_text("utf-8")
    assert 'raw["bp"] = bp' in api, "BP 没有带进对局"


def test_the_game_minute_is_filled_by_the_server():
    """局内时间仍然**自动给**，只是不再要玩家手打。

    让用户再手打一遍既费事又容易打错，而打错之后所有带时间窗的模式
    都会判到错误的阶段 —— 报告上只是「没触发」。

    现在页面上**没有**逐段时间的输入框了（玩家明确说不想标），
    时间由服务端按窗回填（见 app/collection/video_upload._window_rounds：
    每窗写一条 ``time_min = 窗的起始分钟``）。
    """
    # 时间仍然来自时钟：窗的起始分钟由服务端写进事实
    from app.collection import video_upload as vu

    src = (APP_ROOT / "app" / "collection" / "video_upload.py").read_text("utf-8")
    assert 'name="time_min"' in src, "服务端没有按窗回填局内时间"


# ------------------------------------------------- 标注页真的渲染得出来


@pytest.fixture
def client(tmp_path):
    """独立临时库的 TestClient（规则库要先灌上 —— 报告要有规则才判得动）。"""
    from fastapi.testclient import TestClient

    from app import db as dbmod
    from app.main import app

    dbmod._engine = None
    dbmod._SessionFactory = None
    dbmod.get_engine(str(tmp_path / "annotate.db"))
    from app.db import session_scope
    from app.rules.store import latest_version, load_seed_library

    with session_scope() as s:
        if not latest_version(s, "hok"):
            load_seed_library(s, "hok")

    yield TestClient(app)

    dbmod._engine = None
    dbmod._SessionFactory = None


def _fake_session(match_id: str = "hok-video-test") -> None:
    """往采集层挂一个假的扫描会话。

    标注页的数据来自进程内的 ``_SESSIONS``，不落库 —— 所以要直接造。
    """
    from app.adapters.base import get_adapter
    from app.collection import video_api

    video = vu.UploadedVideo(video_id="v1", path="x.mp4", original_name="x.mp4",
                             size=1, duration_sec=900.0)
    shots = [
        {"round_no": 1, "phase": "early", "game_minute": 0.5, "timestamp": 228.5,
         "image": "phase-001.jpg", "confidence": 1.0},
        {"round_no": 2, "phase": "early", "game_minute": 1.5, "timestamp": 288.5,
         "image": "phase-002.jpg", "confidence": 1.0},
        {"round_no": 3, "phase": "mid", "game_minute": 5.5, "timestamp": 528.5,
         "image": "phase-003.jpg", "confidence": 1.0},
    ]
    summary = {
        "video": video.to_dict(),
        "scan": {"note": "认出时钟", "fps": 89.7, "clock": {"ok": True}},
        "product": "phase",
        "game": "hok",
        "game_version": "S45",
        "round_count": len(shots),
        "rounds": shots,
        "bp": [
            {"label": "BP 1", "timestamp": 9.9, "image": "bp-001.jpg",
             "image_url": "/api/videos/v1/frames/bp-001.jpg"},
            {"label": "BP 2", "timestamp": 29.8, "image": "bp-002.jpg",
             "image_url": "/api/videos/v1/frames/bp-002.jpg"},
        ],
        "bp_count": 2,
        "match_minutes": 12.0,
    }
    video_api._SESSIONS[match_id] = {
        "summary": summary,
        "skeleton": vu.build_skeleton(video, shots, "hok", "S45", match_id),
        "adapter": get_adapter("hok"),
    }


def test_the_annotate_page_is_gone_but_says_why(client):
    """标注页下线了，但**别让人撞个 404**。

    有人可能还留着旧书签、或者从别处跳过来。返回一个说清来由的 410，
    比一个「找不到页面」有用得多 —— 人至少知道该重新走一遍上传。
    """
    match_id = "hok-video-render-test"
    _fake_session(match_id)
    try:
        r = client.get(f"/videos/{match_id}/annotate")
    finally:
        from app.collection import video_api

        video_api._SESSIONS.pop(match_id, None)

    assert r.status_code == 410, r.text[:200]
    assert "自动出报告" in r.json()["detail"]


def test_the_report_is_built_without_any_manual_step(client):
    """扫完 → 出报告，中间没有让人填的一步。

    这条链路上最要紧的不是「有没有页面」，而是**那条链路真的通**：
    扫描结果落进会话 → finalize 建对局 → 报告地址返回。
    """
    from app.collection import video_api

    match_id = "hok-video-render-test"
    _fake_session(match_id)
    try:
        r = client.post("/api/videos/finalize", json={"match_id": match_id})
    finally:
        video_api._SESSIONS.pop(match_id, None)

    assert r.status_code == 200, r.text[:400]
    data = r.json()
    assert data["report_url"].startswith("/api/matches/")
    assert data["match_id"]


# ------------------------------------------------- 扫描走哪条路


def _run_job(monkeypatch, tmp_path, game: str, clock_ok: bool):
    """跑一次扫描任务，返回产物的 kind。

    把真正耗时的部分（解视频）换掉，只测**分流**：
    读得到时钟且这游戏有阶段窗口 → 按阶段抽帧；否则 → 按回合边界。
    """
    from app.collection import scan_jobs
    from app.collection import video_upload as vu

    video = vu.UploadedVideo(video_id="v1", path="x.mp4", original_name="x.mp4",
                             size=1, duration_sec=900.0)
    job = scan_jobs.ScanJob("j1", "m1", video, game, "S45")

    called = {"probe": 0, "phase": 0, "activity": 0, "slice": 0}

    def fake_probe(video, sample_sec=2.0, progress=None, **kw):
        called["probe"] += 1
        info = {"ok": clock_ok, "game_start_t": 198.5 if clock_ok else None,
                "readable_times": [198.5, 918.0] if clock_ok else []}
        return info, {"note": "探时钟", "clock": info, "fps": 89.7}

    def fake_activity(video, config=None, progress=None, **kw):
        called["activity"] += 1
        points = [(10.0, 0.2), (30.0, 0.9), (60.0, 0.1)]
        return points, {"note": "密集扫了", "clock": {"ok": False}, "fps": 89.7}

    def fake_slice(video, points, base_dir, windows, budget=None, progress=None, **kw):
        called["slice"] += 1
        return {"rounds": [{"round_no": 1, "phase": "early", "game_minute": 0.5,
                            "timestamp": 228.5, "image": "a.jpg", "image_url": "",
                            "confidence": 1.0}],
                "bp": [], "match_start_t": 198.5, "match_end_t": 918.0,
                "match_minutes": 12.0}

    def fake_sample(video, clock_info, base_dir, windows, **kw):
        called["phase"] += 1
        return {"rounds": [{"round_no": 1, "phase": "early", "game_minute": 0.5,
                            "timestamp": 228.5, "image": "a.jpg", "image_url": "",
                            "confidence": 1.0}],
                "bp": [], "match_start_t": 198.5, "match_end_t": 918.0,
                "match_minutes": 12.0}

    monkeypatch.setattr(vu, "probe_clock", fake_probe)
    monkeypatch.setattr(vu, "activity_scan", fake_activity)
    monkeypatch.setattr(vu, "moba_slice_product", fake_slice)
    monkeypatch.setattr(vu, "sample_by_phase", fake_sample)

    runner = scan_jobs.JobRunner()
    runner._run(job, str(tmp_path), vu.ScanConfig(), None)
    assert job.status == "done", job.error
    return job.summary, called


def test_moba_with_a_readable_clock_uses_phase_sampling(monkeypatch, tmp_path):
    """MOBA + 读得到时钟 → 按阶段抽帧，不做密集扫描。

    密集扫描那一遍要 10 分钟（每秒 0.3 秒采一次 + 整片顺序解码），
    而 MOBA 根本不需要灰度差 —— 回合边界是自走棋的东西。
    """
    summary, called = _run_job(monkeypatch, tmp_path, "hok", clock_ok=True)
    assert summary["product"] == "phase"
    assert called["phase"] == 1
    assert called["activity"] == 0, "读得到时钟就不该再密集扫一遍"


def test_moba_without_a_clock_slices_by_information_density(monkeypatch, tmp_path):
    """读不到时钟 → 按信息密度切时间片，**不**退回自走棋的回合边界。

    按视频秒硬凑一个阶段是最容易犯也最难发现的错：每一条带时间窗的
    模式都会判到错误的阶段，而报告上只是「没触发」。

    而 MOBA 根本没有回合 —— 按灰度差切出来的「回合」全是镜头移动
    （实测一局王者切出 80 个「回合」，真正值得判断的时刻不到 30 个），
    所以这里改按信息密度切时间片。
    """
    summary, called = _run_job(monkeypatch, tmp_path, "hok", clock_ok=False)
    assert summary["product"] == "moba_slice"
    assert called["activity"] == 1, "读不到时钟时应该密集扫一遍拿信息密度"
    assert called["slice"] == 1, "没有按信息密度切时间片"
    assert called["phase"] == 0


def test_auto_chess_is_refused_outright(monkeypatch, tmp_path):
    """不做自走棋：别的游戏当场说清楚，别悄悄按回合切一遍。

    这个产品只服务王者荣耀的对局。自走棋那套「一回合一切」在王者里
    根本用不上（切出来的是镜头移动），留着只会让人以为视频也支持它。
    """
    from app.collection import scan_jobs
    from app.collection import video_upload as vu

    video = vu.UploadedVideo(video_id="v1", path="x.mp4", original_name="x.mp4",
                             size=1, duration_sec=900.0)
    job = scan_jobs.ScanJob("j1", "m1", video, "tft", "S45")
    runner = scan_jobs.JobRunner()
    runner._run(job, str(tmp_path), vu.ScanConfig(), None)

    assert job.status == "failed"
    assert "只支持王者荣耀" in job.error


# ------------------------------------------------- 报告要说得清「实际在做什么」


def test_report_says_what_the_player_actually_did():
    """玩家做了**别的事**时，报告的「实际在做啥」不能是空的。

    典型场景：这一帧去抓人了（团战动作），而这条模式要的是刷野。
    只统计期望那一类动作的话，报告会写「（未标）」——
    明明他做了很多事，报告却说成什么都没做，「怎么改」那一条也就
    给不出方向。
    """
    from app.core.attribution import analyze_match
    from app.core.context import ActionRecord, MatchContext, RoundRecord
    from app.core.dsl import Condition, Rule, Source, Trigger

    rule = Rule(
        id="T-1", game="fake", version="1",
        title="该刷野", category="打野",
        trigger=Trigger(all=[
            Condition("position", "==", "jungle"),
            Condition("phase", "==", "early"),
        ]),
        expect_fact="objective_action",
        expect_op="in",
        expect_value=["farm_jungle"],
        severity="high",
        source=Source(guide="测试"),
    )
    ctx = MatchContext(
        match_id="m", game="fake", game_version="1",
        rounds=[RoundRecord(
            stage=1, phase=1,
            facts={"phase": "early", "position": "jungle"},
            actions=[ActionRecord(kind="teamfight_action", value="engage",
                                  label="开团先手", confidence=1.0)],
            label_override="第 1 分钟",
        )],
    )
    report = analyze_match([rule], ctx)
    finding = report["lanes"]["findings"][0]
    # 逐回合判的是「偏离」，聚合成一段就是「整段没做」
    assert finding["verdict"] in ("deviated", "never")
    assert finding["actually_doing"], "「实际在做啥」是空的，玩家没法照着改"
    assert "开团先手" in finding["actually_doing"]
