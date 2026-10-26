"""设备隔离：公网部署时，各人的对局不能串。

这个开关（``MULTI_TENANT=1``）是给「部署到云上、别人也能用」的场景准备的：
数据层本来就按 ``player_id`` 分区，缺的只是「这个请求属于谁」。

默认关闭 —— 本地单人使用时行为一个字都不变，
所以这里既要有「开着能隔离」的断言，也要有「关着仍然共用」的断言：
不然哪天有人把默认值改了，谁都不会发现。
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
def app_and_db(tmp_path, monkeypatch):
    """每个测试一套干净数据库；``MULTI_TENANT`` 由各测试自己决定。"""
    from fastapi.testclient import TestClient

    from app import db as dbmod
    from app.main import app

    monkeypatch.delenv("DATABASE_URL", raising=False)
    dbmod._engine = None
    dbmod._SessionFactory = None
    dbmod.get_engine(str(tmp_path / "device_isolation.db"))

    # 先把规则库灌上 —— demo 对局的录入要校验规则库在不在。
    # 规则库是**全局**的（不分设备）：它属于产品，不属于某个人。
    TestClient(app).post("/api/games/hok/rules/seed")

    yield app

    dbmod._engine = None
    dbmod._SessionFactory = None


def _make_match(client):
    from demo_data import build_demo_match

    r = client.post("/api/matches", json=build_demo_match("hok"))
    assert r.status_code == 200
    return r.json()["match_id"]


# ---------------------------------------------------------------- 默认行为


def test_without_multi_tenant_everyone_shares_local(app_and_db):
    """隔离没开：两个浏览器看到的是同一批对局（本地单人使用就是这样）。"""
    from fastapi.testclient import TestClient

    app = app_and_db
    a = TestClient(app)
    b = TestClient(app)

    match_id = _make_match(a)
    assert match_id in b.get("/matches").text
    assert b.get(f"/api/matches/{match_id}/report").status_code == 200


# ---------------------------------------------------------------- 打开隔离


def test_two_devices_do_not_see_each_other(app_and_db, monkeypatch):
    """打开隔离：A 录的对局，B 看不见也拿不到。"""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MULTI_TENANT", "1")
    app = app_and_db
    a = TestClient(app)
    b = TestClient(app)

    match_id = _make_match(a)

    assert match_id in a.get("/matches").text, "A 应该看得到自己的对局"
    assert match_id not in b.get("/matches").text, "B 不该看得到 A 的对局"
    assert b.get(f"/api/matches/{match_id}/report").status_code == 404
    assert b.delete(f"/api/matches/{match_id}").status_code == 404


def test_the_device_cookie_is_set_on_first_visit(app_and_db, monkeypatch):
    """第一次访问就发一个设备标识 Cookie —— 不登录也要分得清是谁。"""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MULTI_TENANT", "1")
    client = TestClient(app_and_db)
    client.get("/matches")

    assert "ma_device" in client.cookies


def test_the_same_device_keeps_its_identity(app_and_db, monkeypatch):
    """同一个浏览器再来一次，还是同一个人（Cookie 有效期内）。"""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MULTI_TENANT", "1")
    app = app_and_db
    first = TestClient(app)
    first.get("/matches")
    device_cookie = first.cookies.get("ma_device")
    assert device_cookie

    # 换一个新的客户端，但把 Cookie 带过去 —— 就是同一台设备
    second = TestClient(app)
    second.cookies.set("ma_device", device_cookie)
    match_id = _make_match(second)

    assert match_id in first.get("/matches").text


def test_explicit_player_id_still_wins(app_and_db, monkeypatch):
    """显式指定 ?player_id= 时按它来 —— 脚本、测试、分享链接不受影响。"""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MULTI_TENANT", "1")
    app = app_and_db
    a = TestClient(app)
    b = TestClient(app)

    match_id = _make_match(a)
    # B 显式问 local 那个池子 —— 里面没有（隔离下 A 的对局在 A 的池子里）
    assert b.get(f"/api/matches/{match_id}/report?player_id=local").status_code == 404
