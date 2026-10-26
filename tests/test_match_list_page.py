"""对局记录页：名字、分组、检索。

盯的是三件事：

* **名字**：对局名是「分路·英雄·胜负」，不是内部 id。
  内部 id（``hok-video-5052bd0c136e``）不是人读的东西 —— 玩家要找的
  是「昨天那局打野」，不是一串十六进制。
* **分组**：按分路组织，五条分路 + 全部。
* **检索**：**只按对局名**找。早先还能按游戏、按版本筛，三个条件叠在
  一起时用户会怀疑「为什么搜不到」—— 因为他搜的是版本号而不是名字。
"""

from __future__ import annotations

import os
import re
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
    dbmod.get_engine(str(tmp_path / "match_list.db"))

    client = TestClient(app)
    # 录对局要有一条能对上的规则库，否则接口直接 409。
    # 这不是被测的东西，只是前置条件。
    client.post("/api/games/hok/rules/seed")

    yield client

    dbmod._engine = None
    dbmod._SessionFactory = None


def _post_match(client, match_id, position, hero, result, game="hok",
                version="S45", played_at="2026-10-20 21:00"):
    # 事实名按游戏给：自走棋没有王者那套事实，填了会被当成未注册事实拒掉。
    facts = ({"gold_diff": 0, "kill_diff": 0, "tower_diff": 0}
             if game == "hok" else {})
    r = client.post("/api/matches", json={
        "match_id": match_id, "game": game, "game_version": version,
        "player": {"nickname": "示例玩家", "position": position, "hero": hero},
        "outcome": {"result": result},
        "played_at": played_at,
        "rounds": [{
            "stage": 0, "phase": 1, "time_min": 1,
            "facts": facts,
            "actions": [{"kind": "positioning_action", "value": "push_wave",
                         "label": "推线压塔", "confidence": 1.0}]
            if game == "hok" else [],
            "missing_facts": [],
        }],
    })
    assert r.status_code == 200, r.text[:300]
    return r


def _insert_other_game_match(match_id, game="tft", version="13.1"):
    """直接塞一条**别的游戏**的对局进库。

    走 POST 接口录不进去 —— 那条路要求该游戏有规则库，而自走棋的
    种子资料已经删了（模块留着，资料后补）。这里要验的只是
    「对局列表只列王者荣耀」，不需要一条能分析的对局。
    """
    from app.db import Match, session_scope

    with session_scope() as s:
        rec = Match(id=match_id, game=game, game_version=version,
                    player_id="local")
        rec.set_payload({
            "player": {"nickname": "示例玩家", "position": "jungle", "hero": "赵云"},
            "outcome": {"result": "win"},
            "rounds": [],
        })
        rec.set_outcome({"result": "win"})
        s.add(rec)


def test_match_name_is_lane_hero_result(client):
    """对局名 = 分路 + 英雄 + 胜负。"""
    _post_match(client, "m-side-nz", "side", "哪吒", "lose")
    _post_match(client, "m-jungle-zy", "jungle", "赵云", "win")

    html = client.get("/matches").text
    assert "对抗路·哪吒·失败" in html
    assert "打野·赵云·胜利" in html
    # 内部 id 不该再当标题用（它只该出现在链接里）
    assert ">m-side-nz<" not in html
    assert ">m-jungle-zy<" not in html


def test_match_name_does_not_guess_missing_fields(client):
    """没填的就写没填 —— 宁可缺一段，也不编。

    补一段的后果是「未选英雄」被当成「这个英雄」，
    而两局看上去一模一样，用户根本分不出哪一局是哪一局。
    """
    _post_match(client, "m-no-hero", "mid", "", "win")
    html = client.get("/matches").text
    assert "中路·未选英雄·胜利" in html


def test_matches_are_grouped_by_lane(client):
    """五条分路 + 全部。"""
    _post_match(client, "m-j1", "jungle", "赵云", "win")
    _post_match(client, "m-j2", "jungle", "镜", "win")
    _post_match(client, "m-c1", "carry", "敖隐", "lose")

    html = client.get("/matches").text
    # 六个分类：全部 + 五条分路
    for label in ("全部", "打野", "中路", "对抗路", "发育路", "游走"):
        assert f">{label}</a>" in html, f"分类里少了 {label}"
    for code in ("jungle", "mid", "side", "carry", "roam"):
        assert f"lane={code}" in html, f"缺分路入口 {code}"

    # 有对局的分路各开一个小节；空的分路不开（空标题只是噪音）
    m = re.search(r'data-lane="jungle"(.*?)</section>', html, re.S)
    assert m, "找不到打野小节"
    assert m.group(1).count('class="item"') == 2
    m = re.search(r'data-lane="carry"(.*?)</section>', html, re.S)
    assert m and m.group(1).count('class="item"') == 1
    assert 'data-lane="mid"' not in html, "没有对局的中路不该开小节"

    # 分路筛选
    html = client.get("/matches", params={"lane": "jungle"}).text
    assert "打野·赵云·胜利" in html
    assert "发育路·敖隐·失败" not in html


def test_match_search_only_looks_at_the_name(client):
    """检索只认对局名。

    早先这里还能按游戏、按版本筛。用户搜「S45」却什么都搜不到，
    而页面上又没有任何地方说明「搜的是名字不是版本」——
    最后只能归因为「这功能坏了」。
    """
    _post_match(client, "m-j1", "jungle", "赵云", "win")
    _post_match(client, "m-c1", "carry", "敖隐", "lose")

    html = client.get("/matches", params={"q": "赵云"}).text
    assert "打野·赵云·胜利" in html
    assert "发育路·敖隐·失败" not in html

    # 版本号搜不到 —— 因为名字里没有版本号
    html = client.get("/matches", params={"q": "S45"}).text
    assert "打野·赵云·胜利" not in html
    assert "发育路·敖隐·失败" not in html


def test_only_honor_of_kings_matches_are_listed(client):
    """这个产品只做王者荣耀。另外两款的规则库都没有，列出来只是噪音。"""
    _post_match(client, "m-j1", "jungle", "赵云", "win")
    _insert_other_game_match("t-1")

    html = client.get("/matches").text
    assert html.count('class="item"') == 1, "非王者荣耀的对局不该出现"
    assert "云顶" not in html and "金铲铲" not in html


def test_match_page_has_no_game_pills(client):
    """游戏切换早就是多余的分叉了 —— 页面上不该再有它。"""
    html = client.get("/matches").text
    assert "?game=" not in html
    for dead in ("云顶", "金铲铲"):
        assert dead not in html
