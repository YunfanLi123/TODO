"""报告导出：把网页报告变成一个能留在自己电脑上的文件。

2026-10-22 加的。起因是一个很实际的问题：
「部署到云上以后，用户关掉网页，他的报告会丢吗？」

答案是分两层的，这套测试盯的是第二层：
  * 报告本身在数据库里 —— 云上用 PostgreSQL 就不会丢（那是部署侧的事）；
  * 就算服务器哪天没了，用户还能把报告导成一个自包含单文件带回家。

所以断言盯四件事：
  1. 导出接口真的返回**文件**（attachment），而不是又一个网页；
  2. 导出的文件自包含：样式与脚本都内联进去，双击能开，不依赖服务器；
  3. 文件里写明它是导出件 —— 收到的人不会以为是自己刚截的图；
  4. 对局记录页每一项都有保存按钮，BP 那个只在录过 BP 时才出现。
"""

from __future__ import annotations

import json
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
    from app.main import app

    # 本地测试必须走 SQLite：环境里若留了云端的 DATABASE_URL，
    # 测试会跑去连生产库 —— 那不是「测试失败」，是「误伤生产数据」。
    monkeypatch.delenv("DATABASE_URL", raising=False)
    dbmod._engine = None
    dbmod._SessionFactory = None
    dbmod.get_engine(str(tmp_path / "report_export.db"))

    yield TestClient(app)

    dbmod._engine = None
    dbmod._SessionFactory = None


@pytest.fixture
def seeded(client):
    """灌一套王者规则库 + 一局 demo 对局，返回 (client, match_id)。"""
    from demo_data import build_demo_match

    assert client.post("/api/games/hok/rules/seed").status_code == 200
    r = client.post("/api/matches", json=build_demo_match("hok"))
    assert r.status_code == 200
    return client, r.json()["match_id"]


BP_PAYLOAD = {
    "tier": "ranked",
    "decisions": [
        {"side": "my", "phase": "ban_1", "action": "ban_opp_core", "hero": "典韦"},
        {"side": "my", "phase": "pick_1", "action": "pick_frontline", "hero": "廉颇"},
    ],
}


# ---------------------------------------------------------------- 局内报告


def test_export_returns_a_file_not_a_page(seeded):
    """导出要返回 attachment，而不是又一个网页。"""
    client, match_id = seeded
    r = client.get(f"/api/matches/{match_id}/report/export")

    assert r.status_code == 200
    disposition = r.headers.get("content-disposition", "")
    assert disposition.startswith("attachment"), disposition
    assert ".html" in disposition, disposition


def test_export_filename_is_readable_chinese(seeded):
    """文件名带中文时要用 RFC 5987 的 filename*，否则浏览器拿到乱码名。"""
    client, match_id = seeded
    r = client.get(f"/api/matches/{match_id}/report/export")

    disposition = r.headers.get("content-disposition", "")
    assert "filename*=UTF-8''" in disposition, disposition


def test_export_is_self_contained(seeded):
    """样式和脚本都内联 —— 双击能开，不依赖服务器。"""
    client, match_id = seeded
    html = client.get(f"/api/matches/{match_id}/report/export").text

    assert "<link" not in html, "导出文件里还有外链样式表，离线打开会没样式"
    assert "<style>" in html, "导出文件里没有内联样式"
    assert 'src="/static' not in html, "导出文件里还有外链脚本，离线打开会不工作"


def test_export_page_says_where_it_came_from(seeded):
    """导出件要写明自己是从网页导出的、什么时候导的。"""
    client, match_id = seeded
    html = client.get(f"/api/matches/{match_id}/report/export").text

    assert "导出" in html
    assert "离线" in html


def test_export_json_is_valid_json(seeded):
    """JSON 导出是存档用的：得能直接解析。"""
    client, match_id = seeded
    r = client.get(f"/api/matches/{match_id}/report/export?format=json")

    assert r.status_code == 200
    data = json.loads(r.text)
    # 局内报告的正文在 mode_analysis 里（BP 报告才是 entries）
    assert "meta" in data and "mode_analysis" in data


def test_export_of_unknown_match_is_404(seeded):
    """不存在的对局要说没有，不能导出一个空文件。"""
    client, _ = seeded
    assert client.get("/api/matches/nope/report/export").status_code == 404


# ---------------------------------------------------------------- BP 复盘


def test_bp_export_404_when_bp_not_recorded(seeded):
    """没录 BP 的对局没有 BP 复盘可导出。"""
    client, match_id = seeded
    assert client.get(f"/api/matches/{match_id}/bp/report/export").status_code == 404


def test_bp_export_is_self_contained(seeded):
    """录了 BP 之后，BP 复盘也能导出成自包含单文件。"""
    client, match_id = seeded
    assert client.post(f"/api/matches/{match_id}/bp", json=BP_PAYLOAD).status_code == 200

    r = client.get(f"/api/matches/{match_id}/bp/report/export")
    assert r.status_code == 200
    assert r.headers.get("content-disposition", "").startswith("attachment")

    html = r.text
    assert "<link" not in html, "BP 导出文件里还有外链样式表"
    assert "<style>" in html


# ---------------------------------------------------------------- 对局记录页


def test_matches_page_has_a_save_button_per_match(seeded):
    """每一项对局都有「保存报告」。"""
    client, _ = seeded
    html = client.get("/matches").text

    assert "保存报告" in html
    assert "/report/export" in html


def test_bp_save_button_only_shows_when_bp_exists(seeded):
    """没录 BP 就不给「保存BP复盘」—— 给一个必然 404 的按钮是骗人。"""
    client, match_id = seeded

    html = client.get("/matches").text
    assert "保存BP复盘" not in html

    assert client.post(f"/api/matches/{match_id}/bp", json=BP_PAYLOAD).status_code == 200
    html = client.get("/matches").text
    assert "保存BP复盘" in html
    assert "/bp/report/export" in html
