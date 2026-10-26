"""「操作模式分析」必须**看全**这一局的偏离。

2026-10-22 改版把分路结论与逐回合差距合成一节时，留了个坑：
分路结论只覆盖「能归到某一条分路」的模式（``position`` 条件恰好一个值），
自走棋那种按阶段编号的规则、「全位置通用」的规则都归不进去 ——
它们的偏离如果没人收，报告就会对着一局有偏离的对局说
「没有需要改的地方」。

这里盯三件事：

1. 没有分路结论的偏离，**也要成条**，且四部分齐全；
2. 同一条规则不许在报告里出现两次（分路那一层和逐回合那一层重复）；
3. 低置信度标记只挂在**真的用到**那个低置信度事实的条目上 ——
   挂太宽等于没挂。
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.adapters import registry as _registry  # noqa: F401  触发适配器注册


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from app import db as dbmod
    from app.main import app

    dbmod._engine = None
    dbmod._SessionFactory = None
    dbmod.get_engine(str(tmp_path / "mode_analysis.db"))

    yield TestClient(app)

    dbmod._engine = None
    dbmod._SessionFactory = None


@pytest.fixture
def video_match(client):
    """一局**带低置信度事实**的自走棋对局。

    选它是因为它同时踩中三件事：规则不分路（归不进分路结论）、
    有一个中等置信度的事实（连胜场次 0.48）、还有一个识别失败的事实。
    """
    from tests.scenarios import VIDEO_RECOGNITION, seed_fixture_library

    assert client.post("/api/games/tft/rules/seed").status_code == 200
    seed_fixture_library()
    r = client.post("/api/matches/from-recognition", json=VIDEO_RECOGNITION)
    assert r.status_code == 200, r.text
    return client, VIDEO_RECOGNITION["match_id"]


def _entries(client, match_id):
    r = client.get(f"/api/matches/{match_id}/report.json")
    assert r.status_code == 200, r.text
    return r.json().get("mode_analysis") or []


def test_the_rules_without_a_lane_still_show_up(video_match):
    """归不进分路结论的偏离，也得在报告里立一条。"""
    client, match_id = video_match
    entries = _entries(client, match_id)

    assert entries, "有偏离的对局，报告却说没有需要改的地方"

    for e in entries:
        assert e["title"], "条目没有标题"
        assert e["time_range"], f"条目「{e['title']}」没有对局时间区间"
        assert e["actual_text"], f"条目「{e['title']}」没有实际操作的描述"
        assert e["advice_text"], f"条目「{e['title']}」没有建议操作描述"
        assert e["reason_text"], f"条目「{e['title']}」没有建议理由"


def test_a_rule_is_never_listed_twice(video_match):
    """分路结论与逐回合差距是同一件事的两个视角，不能各算一条。"""
    client, match_id = video_match
    entries = _entries(client, match_id)

    ids = [e["mode_id"] for e in entries]
    assert len(ids) == len(set(ids)), f"同一条模式出现了多次：{ids}"


def test_the_low_confidence_tag_only_marks_what_uses_that_fact(video_match):
    """低置信度标记只挂在真的用到那个事实的条目上。

    「这一回合有个低置信度的事实」不等于「这条结论建立在这个猜测上」。
    全都挂上等于没挂 —— 玩家看到的每条都写着「置信度不足」，
    反而看不出哪一条真该打问号。
    """
    client, match_id = video_match
    entries = _entries(client, match_id)

    flagged = [e for e in entries if e["low_confidence"]]
    assert flagged, "用到了低置信度事实的条目没有被标出来"

    for e in flagged:
        assert e["confidence"] < 1.0
        assert e["confidence"] > 0, "置信度不该是 0"

    # 被标出来的必须是「确实用到连胜场次那条事实」的规则
    # —— 站位那条跟连胜没关系，不该跟着沾。
    titles = [e["title"] for e in flagged]
    assert not any("站位" in t for t in titles), \
        f"跟低置信度事实无关的条目被标上了：{titles}"


def test_the_report_speaks_chinese_only(video_match):
    """布尔取值说成「是 / 否」，句子里不许蹦出 True / False。"""
    client, match_id = video_match
    html = client.get(f"/api/matches/{match_id}/report").text

    assert "True" not in html, "报告里出现了英文 True"
    assert "False" not in html, "报告里出现了英文 False"
    assert "是" in html, "布尔事实应该说成「是」"
