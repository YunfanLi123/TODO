"""「出错的那一段」能不能被定位到原视频。

这条链路跨了三层，每一层都可能悄悄断掉而页面照样渲染成功：

    扫描 → 对局数据里存下 video_id / 回合秒数 / 截图名
         → 分路结论带出出问题的回合
         → 报告拼出可点的截图

任何一环断了，用户看到的是「结论在那儿，但点不了」—— 不会报错，
只会让功能看起来没做。所以每一环都单独断言。

这里走 :func:`save_match` 直接入库，而不是 ``POST /api/matches``：
后者是**手工 / API 录入**的口子，它的 pydantic 模型只认 stage/phase/facts/
actions，视频元数据（回合秒数、截图名）根本不在字段里 —— 那条路上本来
就没有视频，也不该有。视频来源的对局走 ``/api/videos/finalize``，
它经 :func:`to_raw_match` 组装，字段是全的。
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.adapters import registry as _registry  # noqa: F401  触发适配器注册


@pytest.fixture
def session(tmp_path):
    from app import db as dbmod

    dbmod._engine = None
    dbmod._SessionFactory = None
    dbmod.get_engine(str(tmp_path / "clip_test.db"))
    with dbmod.session_scope() as s:
        yield s
    dbmod._engine = None
    dbmod._SessionFactory = None


def _demo_with_video():
    """一局带「原视频坐标」的打野对局，刻意做成有对有错。

    第 8 分钟落后却分推牵制、第 15 分钟后期还在刷野 —— 这两回合会被判偏离，
    报告就该在它们下面给出可点的片段。

    事实与动作是照着 S45 分路决策树那套库挑的：

      - 第 2 分钟刷野 -> ``HOK-MODE-JGL-T0200-01``（两分钟：均势去反野）**做对了**；
      - 第 8 分钟落后却分推 -> ``HOK-MODE-JGL-T0600-04``（落后就只刷安全野区）
        要求只刷安全野区，而分推牵制不是 —— 判成偏离；
      - 第 15 分钟刷野 -> ``HOK-MODE-JGL-T1200-01``（后期带线拆塔或开龙）
        判成「后期还在刷野」。
    """
    return {
        "match_id": "clip-demo-001",
        "game": "hok",
        "game_version": "S45",
        "player": {"nickname": "示例玩家", "position": "打野"},
        "outcome": {"result": "lose"},
        "played_at": "2026-10-18 21:00",
        "rounds": [
            {"stage": 1, "phase": 1, "time_min": 2, "video_time": 5.0,
             "image": "round-001.jpg",
             "facts": {"gold": 1800, "gold_diff": 200, "level": 4,
                       "tower_diff": 0, "kill_diff": 0},
             "actions": [{"kind": "objective_action", "value": "farm_jungle",
                          "label": "刷野补经济", "confidence": 1.0}],
             "missing_facts": []},
            {"stage": 2, "phase": 1, "time_min": 8, "video_time": 640.0,
             "image": "round-003.jpg",
             "facts": {"gold": 8400, "gold_diff": -300, "level": 10,
                       "tower_diff": 2, "kill_diff": 0},
             "actions": [{"kind": "objective_action", "value": "split_push",
                          "label": "分推牵制", "confidence": 1.0}],
             "missing_facts": []},
            {"stage": 3, "phase": 1, "time_min": 15, "video_time": 910.0,
             "image": "round-004.jpg",
             "facts": {"gold": 15800, "gold_diff": -2100, "level": 14,
                       "tower_diff": -3, "kill_diff": -6},
             "actions": [
                 {"kind": "objective_action", "value": "farm_jungle",
                  "label": "刷野补经济", "confidence": 0.9},
                 {"kind": "teamfight_action", "value": "engage",
                  "label": "开团先手", "confidence": 0.7},
             ],
             "missing_facts": []},
        ],
        "recognition": {"video_id": "abc123", "degradation": {}},
    }


def _report_for(session, raw):
    """按视频链路的方式入库并出报告。"""
    from app.adapters.base import get_adapter
    from app.collection import match as match_col
    from app.report.builder import build_match_report
    from app.rules.store import get_rules

    adapter = get_adapter("hok")
    record, _ = match_col.save_match(session, adapter, raw)
    ctx = adapter.build_context(record.get_payload())
    rules = get_rules(session, "hok", record.game_version)
    return record, build_match_report(rules, ctx)


def _seed(session):
    from app.rules.store import latest_version, load_seed_library

    if not latest_version(session, "hok"):
        load_seed_library(session, "hok")


def test_video_metadata_survives_into_the_stored_match(session):
    """video_id、回合秒数、截图名都必须跟着对局一起落库。

    只写在扫描结果里是不够的 —— 报告是对局生成的，几小时后再去看，
    没人说得清那个视频还在不在、叫什么。
    """
    _seed(session)
    record, _ = _report_for(session, _demo_with_video())
    payload = record.get_payload()
    assert payload["recognition"]["video_id"] == "abc123"
    assert payload["rounds"][1]["video_time"] == 640.0
    assert payload["rounds"][1]["image"] == "round-003.jpg"


def test_findings_carry_the_offending_moments(session):
    """出问题的回合要带出「第几秒 + 哪张截图 + 可点的地址」。"""
    _seed(session)
    _, report = _report_for(session, _demo_with_video())
    lanes = report["lanes"]

    assert lanes["video_id"] == "abc123"
    assert lanes["video_url"] == "/api/videos/abc123/file"

    with_moments = [f for f in lanes["findings"] if f.get("moments")]
    assert with_moments, "没有一条结论带上了出错时刻"

    m = with_moments[0]["moments"][0]
    assert m["time"] > 0
    assert m["image"].startswith("round-")
    assert m["image_url"] == f"/api/videos/abc123/frames/{m['image']}"

    # 偏离的两回合都该有坐标 —— 少一个就意味着那段没法回看
    all_times = {m["time"] for f in with_moments for m in f["moments"]}
    assert 640.0 in all_times
    assert 910.0 in all_times


def test_manual_match_gets_no_play_entry(session):
    """手工录入的对局没有视频，就**不给**播放入口。

    给一个点开是 404 的按钮比没有更糟：它会让「结论指向了一段
    不存在的画面」，而那比没有依据更让人信以为真。
    """
    _seed(session)
    raw = _demo_with_video()
    raw.pop("recognition")
    for rd in raw["rounds"]:
        rd.pop("video_time", None)
        rd.pop("image", None)

    _, report = _report_for(session, raw)
    lanes = report["lanes"]
    assert lanes["video_id"] == ""
    assert lanes["video_url"] == ""
    assert all(not f["moments"] for f in lanes["findings"])


def test_report_renders_the_shot_buttons(session):
    """报告页要把原视频片段的播放器渲染出来。"""
    from app.report.views import render_report

    _seed(session)
    _, report = _report_for(session, _demo_with_video())
    html = render_report(report)

    assert 'class="clip"' in html, "报告里没有出错片段的播放器"
    assert "/api/videos/abc123/frames/" in html
    assert 'class="clip-vid"' in html
    assert "/api/videos/abc123/file" in html
    assert "vid.currentTime = start" in html, "播放器没有做定位跳转"
