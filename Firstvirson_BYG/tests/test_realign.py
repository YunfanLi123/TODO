"""对局起点校正：改一个数，要把每一段**重读一遍**。

背景。玩家在标注页上能改「对局从视频第几秒开始」。起点一改，
「第 6 分钟」对应的画面就换了 —— 原来那批帧是按旧起点摆的，
显示的局内时刻**本来就是错的**。

于是「把算出来的结果平移一下」是不行的：拿错时刻的画面说那一段的事，
方向全对、证据对不上，而且报告上看不出来哪里错了。所以校正必须
**重新抽帧 + 重新识别**：玩家给的数只决定「去看哪几秒」。

这里盯三件事：

1. 什么情况下才值得重读（差不到一秒就不读）；
2. 重读确实重新抽帧、重新识别，而不是挪数字；
3. 校正发生在标注**之前** —— 玩家的补充要落在重读后的结果上。
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest


# ------------------------------------------------------------------ 判定


def test_needs_realign_only_for_a_real_change():
    from app.collection import realign

    summary = {"match_start_t": 120.0}
    # 没给数 → 不重读
    assert not realign.needs_realign(summary, None)
    # 给的是同一个数 → 不重读
    assert not realign.needs_realign(summary, 120.0)
    # 差不到一秒 → 不重读（拿几十秒换一秒不划算）
    assert not realign.needs_realign(summary, 120.5)
    # 差得够多 → 重读
    assert realign.needs_realign(summary, 150.0)
    assert realign.needs_realign(summary, 90.0)
    # 连旧起点都没有 → 没有基准可改，不重读
    assert not realign.needs_realign({}, 150.0)


def test_shift_is_signed():
    from app.collection import realign

    summary = {"match_start_t": 174.9}
    assert realign.shift_of(summary, 150.0) == pytest.approx(-24.9)
    assert realign.shift_of(summary, 200.0) == pytest.approx(25.1)


# ------------------------------------------------------------------ 重读


def _summary() -> Dict[str, Any]:
    shots = [
        {"round_no": 1, "phase": "early", "game_minute": 0.5,
         "timestamp": 228.5, "image": "phase-001.jpg",
         "image_url": "/api/videos/v1/frames/phase-001.jpg"},
        {"round_no": 2, "phase": "early", "game_minute": 1.5,
         "timestamp": 288.5, "image": "phase-002.jpg",
         "image_url": "/api/videos/v1/frames/phase-002.jpg"},
    ]
    return {
        "match_start_t": 174.9,
        "product": "phase",
        "game": "hok",
        "game_version": "S45",
        "match_id": "hok-video-realign",
        "rounds": shots,
        "windows": [
            {
                "lo": 0, "hi": 1, "key": "0-1", "label": "0:00 ~ 2:00",
                "segment_key": "seg-0-1", "segment_label": "0:00 ~ 2:00",
                "slices": [
                    {"round_no": 1, "timestamp": 228.5, "game_minute": 0.5,
                     "image": "phase-001.jpg",
                     "image_url": "/api/videos/v1/frames/phase-001.jpg"},
                    {"round_no": 2, "timestamp": 288.5, "game_minute": 1.5,
                     "image": "phase-002.jpg",
                     "image_url": "/api/videos/v1/frames/phase-002.jpg"},
                ],
                # 旧识别结果：重读之后**不能**原样留着
                "facts": {"facts": {"hero_zone": {"value": "jungle",
                                                  "confidence": 0.8}},
                          "probe_actions": [{"kind": "objective_action",
                                             "value": "take_dragon"}],
                          "probe_moments": [], "absent": [], "missing": []},
            },
        ],
    }


def test_realign_reshots_and_rerecognizes(monkeypatch):
    """重读要真的重新抽帧、重新识别。"""
    from app.collection import realign, video_upload as vu

    asked: Dict[str, Any] = {}

    def fake_extract(video, times, base_dir, prefix, progress=None,
                     hud_layout=None):
        asked["times"] = list(times)
        asked["prefix"] = prefix
        return [
            {"timestamp": t, "image": f"{prefix}-{i + 1:03d}.jpg",
             "image_url": f"/new/{i}"}
            for i, (_label, t) in enumerate(times)
        ]

    def fake_regions(video, base_dir, rounds, sample=16):
        asked["regions_for"] = len(rounds)
        return {"profile": "std"}

    def fake_recognize(video, base_dir, bundle, segment_key, regions=None,
                       hero_track=None, t0=None, t1=None):
        asked["t0"] = t0
        asked["t1"] = t1
        asked["segment_key"] = segment_key
        asked["slice_times"] = [s.get("timestamp") for s in bundle.slices]
        asked["hero_track"] = hero_track
        return {"facts": {"hero_zone": {"value": "river", "confidence": 0.9}},
                "probe_actions": [{"kind": "objective_action",
                                   "value": "fight_baron"}],
                "probe_moments": [], "absent": [], "missing": []}

    monkeypatch.setattr(vu, "extract_frames_at", fake_extract)
    monkeypatch.setattr(vu, "_hud_regions_for", fake_regions)
    monkeypatch.setattr(vu, "_recognize_window", fake_recognize)

    windows, shots = realign.realign_windows(
        object(), "/tmp", _summary(), 150.0, [{"t": 10.0, "nx": 0.1}])

    # 帧是重新抽的：每一帧按新起点挪了位置
    assert asked["times"] == [("early", 203.6), ("early", 263.6)]
    assert asked["prefix"] == "phase"
    # 键位档位按新帧重挑
    assert asked["regions_for"] == 2
    # 识别按**新起点**划的窗区间重跑（窗 0~1 覆盖局内 0:00~2:00）
    assert asked["t0"] == pytest.approx(150.0)
    assert asked["t1"] == pytest.approx(270.0)
    assert asked["slice_times"] == [203.6, 263.6]
    assert asked["hero_track"] == [{"t": 10.0, "nx": 0.1}]
    # 结论来自新的识别，不是旧的
    assert windows[0]["facts"]["facts"] == {"hero_zone": {"value": "river",
                                                          "confidence": 0.9}}
    assert windows[0]["facts"]["probe_actions"][0]["value"] == "fight_baron"
    assert windows[0]["slices"][0]["timestamp"] == pytest.approx(203.6)
    assert shots[0]["timestamp"] == pytest.approx(203.6)


def test_realign_keeps_window_shape(monkeypatch):
    """重读后窗口的形状要和扫描时产出的完全一致 —— 否则骨架建不出来。"""
    from app.collection import realign, video_upload as vu

    monkeypatch.setattr(
        vu, "extract_frames_at",
        lambda video, times, base_dir, prefix, progress=None, hud_layout=None: [
            {"timestamp": t, "image": f"{prefix}-{i + 1:03d}.jpg",
             "image_url": f"/new/{i}"}
            for i, (_l, t) in enumerate(times)
        ])
    monkeypatch.setattr(vu, "_hud_regions_for",
                        lambda video, base_dir, rounds, sample=16: {})
    monkeypatch.setattr(
        vu, "_recognize_window",
        lambda *a, **k: {"facts": {}, "probe_actions": [], "probe_moments": [],
                         "absent": [], "missing": []})

    windows, _shots = realign.realign_windows(object(), "/tmp", _summary(), 150.0)
    w = windows[0]
    for key in ("lo", "hi", "key", "label", "segment_key", "segment_label",
                "slices", "facts"):
        assert key in w, f"重读后窗口少了 {key}"
    # 规则窗自己的时间轴不动：time_min 是规则的时间口径，不是测量值
    assert (w["lo"], w["hi"]) == (0, 1)


# ------------------------------------------------------------------ 接线


def test_apply_start_correction_rebuilds_skeleton(monkeypatch):
    """校正发生在标注之前：玩家的补充要落在重读后的结果上。"""
    from app.collection import video_api, video_upload as vu

    summary = _summary()
    video = vu.UploadedVideo(video_id="v1", path="x.mp4", original_name="x.mp4",
                             size=1, duration_sec=900.0)
    state = {
        "summary": summary,
        "skeleton": vu.build_skeleton(video, summary["rounds"], "hok", "S45",
                                       "hok-video-realign",
                                       windows=summary["windows"]),
        "adapter": None,
        "video": video,
    }

    called: Dict[str, Any] = {}

    def fake_realign(video, base_dir, summary, new_start, hero_track=None):
        called["new_start"] = new_start
        called["hero_track"] = hero_track
        windows = [dict(summary["windows"][0])]
        windows[0]["facts"] = {
            "facts": {"hero_zone": {"value": "river", "confidence": 0.9}},
            "probe_actions": [], "probe_moments": [],
            "absent": [], "missing": []}
        return windows, summary["rounds"]

    monkeypatch.setattr(video_api.realign, "realign_windows", fake_realign)

    out = video_api._apply_start_correction(state, "/tmp", 150.0)

    assert called["new_start"] == pytest.approx(150.0)
    assert out["summary"]["match_start_t"] == pytest.approx(150.0)
    assert out["summary"]["match_start_corrected"] is True
    assert "玩家手动校正" in (out["summary"]["match_start_reason"] or "")
    # 骨架换成重读后的那份
    facts = {f.name: f.value for f in out["skeleton"].rounds[0].facts}
    assert facts.get("hero_zone") == "river"


def test_apply_start_correction_skips_when_unchanged():
    from app.collection import video_api, video_upload as vu

    summary = _summary()
    video = vu.UploadedVideo(video_id="v1", path="x.mp4", original_name="x.mp4",
                             size=1, duration_sec=900.0)
    skeleton = vu.build_skeleton(video, summary["rounds"], "hok", "S45",
                                 "hok-video-realign",
                                 windows=summary["windows"])
    state = {"summary": summary, "skeleton": skeleton, "adapter": None,
             "video": video}

    out = video_api._apply_start_correction(state, "/tmp", 174.9)
    assert out["skeleton"] is skeleton
    assert "match_start_corrected" not in out["summary"]


def test_apply_start_correction_rejects_out_of_range():
    """起点落在视频之外要说清楚，不能默默夹到边界 —— 那是编。"""
    import pytest as _pytest
    from fastapi import HTTPException

    from app.collection import video_api, video_upload as vu

    summary = _summary()
    video = vu.UploadedVideo(video_id="v1", path="x.mp4", original_name="x.mp4",
                             size=1, duration_sec=900.0)
    state = {"summary": summary, "skeleton": None, "adapter": None,
             "video": video}

    with _pytest.raises(HTTPException) as exc:
        video_api._apply_start_correction(state, "/tmp", 5000.0)
    assert exc.value.status_code == 422
    assert "900" in str(exc.value.detail)


def test_apply_start_correction_without_video_does_nothing():
    """没有视频就没法重读 —— 宁可不动，也不平移旧结论。"""
    from app.collection import video_api

    summary = _summary()
    state = {"summary": summary, "skeleton": object(), "adapter": None}

    out = video_api._apply_start_correction(state, "/tmp", 150.0)
    assert out["skeleton"] is state["skeleton"]
    assert "match_start_corrected" not in out["summary"]
