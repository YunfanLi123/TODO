"""即焚模式：关掉网页，报告和上传的视频就没了。

这个需求来自一句很直接的话：
「我不希望把它写在腾讯云上，应该在网页关闭后就清除用户的报告和上传的视频数据」。

所以这里要保证的是：
  1. 数据确实被删了 —— 对局、分析、视频文件一个都不留；
  2. 删的是**这个设备**的 —— A 走了不能把 B 的对局一起清掉；
  3. 删的时机是**真的走了之后** —— 多标签页只关了一个不能误删；
  4. beacon 丢了也能清 —— TTL 兜底；
  5. 页面关闭之前，用户有机会把报告存到自己电脑上（不主动存就没了，
     这正是「用完即焚」的含义，页面上也写明了）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.adapters import registry as _registry  # noqa: F401  触发适配器注册

APP_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app import db as dbmod
    from app.core import session as session_mod
    from app.main import app

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("MULTI_TENANT", raising=False)
    monkeypatch.setenv("EPHEMERAL_SESSION", "1")
    # 宽限期给 0.5 秒：既能测「马上回来还有」，也能 sleep 一下测「真删了」。
    # 0 是不行的 —— 那意味着任何时刻都算到期，
    # 「回来还有」这件事就没法测了。
    monkeypatch.setenv("EPHEMERAL_GRACE_SECONDS", "0.5")
    # TTL 不在这里设成 0 —— 那等于「所有设备都过期」，
    # 清扫会把别人的数据一起带走（隔离测试正好会撞上这个）。
    monkeypatch.setenv("EPHEMERAL_TTL_SECONDS", "3600")
    # 数据目录指到临时目录：视频文件写在真实的 data/ 里会污染用户的库
    monkeypatch.setenv("MATCH_DATA_DIR", str(tmp_path / "data"))

    session_mod.reset()
    dbmod._engine = None
    dbmod._SessionFactory = None
    dbmod.get_engine(str(tmp_path / "ephemeral.db"))

    TestClient(app).post("/api/games/hok/rules/seed")

    yield TestClient(app)

    session_mod.reset()
    dbmod._engine = None
    dbmod._SessionFactory = None


def _make_match(client, match_id=None):
    from demo_data import build_demo_match

    raw = build_demo_match("hok")
    if match_id:
        # 对局 id 是全局主键，不是按玩家分的 —— 两个设备要各录一局的话，
        # id 必须不同（这条也是真机上要记住的：id 撞了就是 UNIQUE 约束报错）。
        raw = dict(raw)
        raw["match_id"] = match_id
        raw["player"] = dict(raw.get("player") or {})
        raw["player"]["nickname"] = match_id

    r = client.post("/api/matches", json=raw)
    assert r.status_code == 200, r.text
    return r.json()["match_id"]


def _attach_video(match_id: str, video_id: str) -> str:
    """给一局挂上视频文件，并让对局 payload 记着这个 video_id。"""
    import json

    from app.db import Match, session_scope

    path = None
    with session_scope() as s:
        record = s.get(Match, match_id)
        payload = record.get_payload() or {}
        payload["video_id"] = video_id
        record.set_payload(payload)
        s.flush()

    from app.collection.video_upload import video_store_dir

    path = os.path.join(video_store_dir(), video_id + ".mp4")
    with open(path, "wb") as fh:
        fh.write(b"fake-video")
    return path


# ---------------------------------------------------------------- 基本开关


def test_off_by_default_the_service_keeps_data(client, monkeypatch):
    """没开即焚模式时，页面关闭不该删东西。"""
    from app.core import session as session_mod

    monkeypatch.setenv("EPHEMERAL_SESSION", "0")
    session_mod.reset()

    match_id = _make_match(client)
    client.post("/api/session/end")

    assert match_id in client.get("/matches").text
    assert client.get(f"/api/matches/{match_id}/report").status_code == 200


def test_policy_endpoint_tells_the_truth(client):
    """前端要知道自己面对的是不是即焚模式 —— 页面上的说明靠它。"""
    policy = client.get("/api/session/policy").json()
    assert policy["ephemeral"] is True
    assert policy["isolated"] is True


# ---------------------------------------------------------------- 真的删


def test_closing_the_page_deletes_the_match(client):
    """关掉网页 → 这局对局没了。"""
    import time

    from app.core import session as session_mod

    match_id = _make_match(client)
    assert match_id in client.get("/matches").text

    client.post("/api/session/end")
    time.sleep(0.7)          # 等过宽限期
    session_mod.sweep()

    assert match_id not in client.get("/matches").text
    assert client.get(f"/api/matches/{match_id}/report").status_code == 404


def test_closing_the_page_deletes_the_uploaded_video(client):
    """视频文件也得删 —— 它不在数据库里，忘了删就成了一地垃圾。"""
    import time

    from app.core import session as session_mod

    match_id = _make_match(client)
    video_path = _attach_video(match_id, "vid-ephemeral-1")
    assert os.path.exists(video_path)

    client.post("/api/session/end")
    time.sleep(0.7)
    session_mod.sweep()

    assert not os.path.exists(video_path)


def test_ttl_cleans_up_when_the_beacon_never_arrived(client, monkeypatch):
    """浏览器崩了、断网了 —— 没收到「我走了」，也得清。"""
    from app.core import session as session_mod

    monkeypatch.setenv("EPHEMERAL_TTL_SECONDS", "0")
    match_id = _make_match(client)
    # 不调 /api/session/end，只让 TTL 过期
    session_mod.sweep()

    assert match_id not in client.get("/matches").text


# ---------------------------------------------------------------- 隔离


def test_purging_one_device_leaves_the_other_alone(client):
    """A 走了不能把 B 的对局带走。"""
    from fastapi.testclient import TestClient

    from app.core import session as session_mod
    from app.main import app

    a = client
    b = TestClient(app)

    match_a = _make_match(a, "demo-device-a")
    match_b = _make_match(b, "demo-device-b")

    a.post("/api/session/end")
    import time

    time.sleep(0.7)
    session_mod.sweep()

    assert match_a not in a.get("/matches").text, "A 自己的对局应该被清掉"
    assert match_b in b.get("/matches").text, "B 的对局不该被牵连"


# ---------------------------------------------------------------- 时机


def test_a_device_that_comes_back_keeps_its_data(client):
    """关掉又马上回来 → 数据还在（清的是「走了」，不是「关过」）。"""
    from app.core import session as session_mod

    match_id = _make_match(client)
    client.post("/api/session/end")

    # 马上又访问了一次（另一个标签页还在用）
    client.get("/matches")
    session_mod.sweep()

    assert match_id in client.get("/matches").text


def test_another_tab_keeps_the_data_alive(client):
    """多标签页只关了一个 → 不该把另一个正在看的报告抽走。"""
    from app.core import session as session_mod

    match_id = _make_match(client)

    # 老标签页关闭（打标记），新标签页还在正常请求
    client.post("/api/session/end")
    client.get("/matches")
    client.get("/matches")
    session_mod.sweep()

    assert match_id in client.get("/matches").text
