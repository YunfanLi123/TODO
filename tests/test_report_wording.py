"""报告页的措辞：胜负按游戏说话，不拿别的游戏的行话硬套。

盯的是三件事：

* **王者荣耀写「胜利 / 失败」**。「吃鸡」是自走棋、大逃杀的词 ——
  王者荣耀是推塔，赢了就是胜利，输了就是失败。拿别的游戏的行话写出来，
  玩家读着别扭，看着也不像是在讲他打的那一局。
* **没填胜负就不显示**。「没打」和「输了」是两件事，混成一个词就是编造。
* **标注来源横幅里不出现没渲染的 markdown 记号**。模板里没有 markdown
  渲染器，写上去的 ``**`` 会原样显示在页面上。

这三个都是「措辞」而不是「逻辑」，但措辞错了就是在替玩家说错话。
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
def client(tmp_path):
    from fastapi.testclient import TestClient

    from app import db as dbmod
    from app.main import app

    dbmod._engine = None
    dbmod._SessionFactory = None
    dbmod.get_engine(str(tmp_path / "report_wording.db"))

    client = TestClient(app)
    client.post("/api/games/hok/rules/seed")

    yield client

    dbmod._engine = None
    dbmod._SessionFactory = None


def _post_match(client, match_id, result=None, position="jungle", hero="赵云"):
    """录一条王者荣耀对局。``result=None`` 表示**不填胜负**。"""
    outcome = {"result": result} if result else {}
    r = client.post("/api/matches", json={
        "match_id": match_id, "game": "hok", "game_version": "S45",
        "player": {"nickname": "示例玩家", "position": position, "hero": hero},
        "outcome": outcome,
        "played_at": "2026-10-20 21:00",
        "rounds": [{
            "stage": 0, "phase": 1, "time_min": 1,
            "facts": {"gold_diff": 0, "kill_diff": 0, "tower_diff": 0},
            "actions": [{"kind": "positioning_action", "value": "push_wave",
                         "label": "推线压塔", "confidence": 1.0}],
            "missing_facts": [],
        }],
    })
    assert r.status_code == 200, r.text[:300]
    return r


def _insert_derived_match(match_id):
    """直接塞一条「动作来自读帧反推」的对局进库。

    走 POST 接口塞不进去 —— ``annotation_source`` 是标注流程写进去的字段，
    手工录对局时它不存在。这里要验的只是报告页怎么**说明**这个来源。
    """
    from app.db import Match, session_scope

    with session_scope() as s:
        rec = Match(id=match_id, game="hok", game_version="S45",
                    player_id="local")
        rec.set_payload({
            "player": {"nickname": "示例玩家", "position": "jungle", "hero": "赵云"},
            "outcome": {"result": "win"},
            "annotation_source": "derived",
            "rounds": [{
                "stage": 0, "phase": 1, "time_min": 1,
                "facts": {"gold_diff": 0, "kill_diff": 0, "tower_diff": 0},
                "actions": [{"kind": "positioning_action", "value": "push_wave",
                             "label": "推线压塔", "confidence": 1.0}],
            }],
        })
        rec.set_outcome({"result": "win"})
        s.add(rec)


# ------------------------------------------------------------------ 胜负措辞


def test_hok_report_says_victory_not_chicken_dinner(client):
    """王者荣耀赢了就是胜利，不是吃鸡。"""
    _post_match(client, "m-win", result="win")

    html = client.get("/api/matches/m-win/report").text
    assert "结果 <b>胜利</b>" in html
    assert "吃鸡" not in html, "报告页还在用自走棋的行话"


def test_hok_report_says_defeat(client):
    """输了就是失败。"""
    _post_match(client, "m-lose", result="lose")

    html = client.get("/api/matches/m-lose/report").text
    assert "结果 <b>失败</b>" in html
    assert "吃鸡" not in html


def test_missing_result_shows_no_result_at_all(client):
    """没填胜负就不显示这一项。

    写成「结果 未打」会让人以为他打了个平手或者被系统判负 ——
    那是**编造**。宁可空着。
    """
    _post_match(client, "m-noresult", result=None)

    html = client.get("/api/matches/m-noresult/report").text
    assert "结果 <b>" not in html, "没填胜负却显示出了结果"
    assert "吃鸡" not in html


# ------------------------------------------------------------ 标注来源横幅
#
# 横幅本身**已经删掉**（2026-10-25）：用户看过之后说
# 「这些描述性文字都去掉，我完全不需要」。
# 标注页下线之后动作只有一个来源 —— 模型从画面里读，
# 没有第二个来源可对比，这句话每局都一样，是废话。
# 这里留一条断言钉住「它别再回来」。


def test_the_report_has_no_derived_note(client):
    """报告上不该有「结论是从画面里读出来的」那类来源横幅。

    两种来源都不加：横幅没了，就是没了。
    """
    _insert_derived_match("m-derived")
    html = client.get("/api/matches/m-derived/report").text
    assert "derived-note" not in html
    assert "从画面里读出来的" not in html
    assert "不是玩家自述" not in html

    _post_match(client, "m-player", result="win")
    html = client.get("/api/matches/m-player/report").text
    assert "derived-note" not in html
