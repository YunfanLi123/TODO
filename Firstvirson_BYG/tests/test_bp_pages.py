"""BP 录入页 / BP 复盘页：样式与按钮的回归测试。

这两页以前有三个毛病：
1. 复盘页不是内层页面的样子（没有左栏、没有主题开关）；
2. 录入页上的「查看复盘」在没录过 BP 时点进去是个 404 —— 按钮是死的；
3. 主题开关在两页上都点了没反应（引了按钮没引脚本）。

这里把「每个按钮都得有用」钉死。
"""
from __future__ import annotations

import json
import os
import re
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
    dbmod.get_engine(str(tmp_path / "bp_ui.db"))
    yield TestClient(app)
    dbmod._engine = None
    dbmod._SessionFactory = None


BP_DEMO = {
    "my_missing_frontline": True,
    "decisions": [
        {"side": "my", "phase": "ban_1", "action": "ban_tank", "hero": "廉颇"},
        {"side": "my", "phase": "pick_1", "action": "pick_core_output", "hero": "敖隐"},
        {"side": "opp", "phase": "pick_1", "action": "pick_counter", "hero": "兰陵王"},
    ],
}


def _make_match(client, with_bp: bool = True) -> str:
    from demo_data import build_demo_match

    payload = build_demo_match("hok")
    payload["match_id"] = "bp-ui-test"
    if not with_bp:
        payload.pop("bp", None)
    r = client.post("/api/matches", json=payload)
    assert r.status_code in (200, 201), r.text
    return "bp-ui-test"


def test_the_bp_pages_look_like_the_inner_pages(client):
    """BP 两页都要有内层页面的壳：左栏导航 + 主题开关。"""
    assert client.post("/api/games/hok/rules/seed").status_code == 200
    match_id = _make_match(client)
    client.post(f"/api/matches/{match_id}/bp", json=BP_DEMO)

    for url in (
        f"/api/matches/{match_id}/bp",
        f"/api/matches/{match_id}/bp/report",
    ):
        html = client.get(url).text
        assert 'class="app-header"' in html, f"{url} 没有左栏"
        assert 'id="themeBtn"' in html, f"{url} 没有主题开关"
        # 主题开关的脚本必须一起在 —— 只有按钮就是装饰
        assert "ma-theme" in html, f"{url} 的主题开关点了没反应"


def test_the_bp_review_page_has_working_way_back(client):
    """复盘页要能回得去：回对局记录、回录入页、回局内报告。"""
    assert client.post("/api/games/hok/rules/seed").status_code == 200
    match_id = _make_match(client)
    client.post(f"/api/matches/{match_id}/bp", json=BP_DEMO)

    html = client.get(f"/api/matches/{match_id}/bp/report").text
    assert 'href="/matches"' in html
    assert f'href="/api/matches/{match_id}/bp"' in html
    assert f'href="/api/matches/{match_id}/report"' in html


def test_the_bp_form_has_no_dead_view_button(client):
    """没录过 BP 时不摆「查看复盘」—— 那个链接点进去是个 404。"""
    assert client.post("/api/games/hok/rules/seed").status_code == 200
    match_id = _make_match(client, with_bp=False)

    html = client.get(f"/api/matches/{match_id}/bp").text
    assert f'href="/api/matches/{match_id}/bp/report"' not in html, \
        "没录 BP 却摆着「查看复盘」，点进去只有 404"

    # 录过之后才摆
    client.post(f"/api/matches/{match_id}/bp", json=BP_DEMO)
    html = client.get(f"/api/matches/{match_id}/bp").text
    assert f'href="/api/matches/{match_id}/bp/report"' in html


def test_the_bp_form_can_add_and_remove_a_hand(client):
    """录入页的「加一条」和每行的删除按钮是页面自带的脚本，要真的能增删。"""
    assert client.post("/api/games/hok/rules/seed").status_code == 200
    match_id = _make_match(client)

    html = client.get(f"/api/matches/{match_id}/bp").text
    assert "function addRow" in html
    assert "row.remove()" in html


def test_the_bp_pages_do_not_use_the_autochess_word_for_hand(client):
    """「第几手」是自走棋的说法，BP 里没有这个单位。

    坐标只说「哪个环节」（一抢 / 二抢 / 三抢…）—— 环节名自己就把先后说清楚了。
    """
    assert client.post("/api/games/hok/rules/seed").status_code == 200
    match_id = _make_match(client)
    client.post(f"/api/matches/{match_id}/bp", json=BP_DEMO)

    for url in (
        f"/api/matches/{match_id}/bp",
        f"/api/matches/{match_id}/bp/report",
    ):
        html = client.get(url).text
        assert not re.search(r"第\s*\d+\s*手", html), f"{url} 里还有「第几手」"
        assert "手次" not in html, f"{url} 里还有「手次」"

    # 坐标本身也干净
    data = client.get(f"/api/matches/{match_id}/bp/report.json").json()
    for e in data["entries"]:
        assert "手" not in e["step_label"], e["step_label"]


def test_the_bp_review_hero_keeps_only_the_essentials(client):
    """复盘页顶头只留「哪一局、哪个游戏」。

    BP 操作了几次、段位档是几档 —— 那是内部账，结论里该说的会说，
    摆在顶上是让人多读两行。
    """
    assert client.post("/api/games/hok/rules/seed").status_code == 200
    match_id = _make_match(client)
    client.post(f"/api/matches/{match_id}/bp", json=BP_DEMO)

    html = client.get(f"/api/matches/{match_id}/bp/report").text
    meta = re.search(r'<div class="hero-meta">(.*?)</div>', html, re.S)
    assert meta, "复盘页没有 hero-meta"
    meta_html = meta.group(1)
    assert "BP 操作" not in meta_html, "复盘页顶头还摆着 BP 操作次数"
    assert "手" not in meta_html, f"复盘页顶头还有「手」：{meta_html}"
    assert "段位" not in meta_html, "复盘页顶头还摆着段位档"
    assert "对局" in meta_html and "游戏" in meta_html


def test_the_offline_bp_export_has_no_links_to_the_server(client):
    """导出的单文件是拿来看的：里面不能有指向服务器的链接（离线点不动）。"""
    assert client.post("/api/games/hok/rules/seed").status_code == 200
    match_id = _make_match(client)
    client.post(f"/api/matches/{match_id}/bp", json=BP_DEMO)

    r = client.get(f"/api/matches/{match_id}/bp/report/export")
    assert r.status_code == 200, r.text
    html = r.text
    assert "<style>" in html and "report.css" not in html.split("<style>")[0], \
        "样式没有内联进去"
    assert 'href="/matches"' not in html, "离线导出里还摆着回服务器的链接"
    assert 'class="app-header"' not in html, "离线导出里还摆着左栏"


# ==========================================================================
# 记录 → 复盘：整条链路 + 两页的一致性
# ==========================================================================


def test_recording_then_reading_the_review_runs_end_to_end(client):
    """BP 记录 → BP 复盘：链路跑通，两页说的是同一件事。

    这两页是同一件事的两面 —— 记的东西要能被复盘读懂，
    复盘说的话要能在记录页上找得到对应。
    """
    from app.adapters.honor_of_kings import BP_PHASE_LABELS

    assert client.post("/api/games/hok/rules/seed").status_code == 200
    match_id = _make_match(client)

    # ---- 1. 记录页：表单的字段都在，且段位与预选已经拿掉 ----
    form = client.get(f"/api/matches/{match_id}/bp").text
    for name in ("my_missing_frontline", "my_missing_engage",
                 "my_missing_protect", "my_missing_damage",
                 "opp_missing_frontline"):
        assert f'name="{name}"' in form, f"记录页少了「{name}」这个字段"
    # 段位与预选：复盘页与记录页里都不该再出现
    assert "段位" not in form, "记录页还有段位"
    assert "预选" not in form, "记录页还有预选"
    assert 'name="tier"' not in form
    assert 'name="preselect_conflict"' not in form
    assert "加一条" in form

    # ---- 2. 按记录页的契约提交（页面脚本就是这么拼的） ----
    r = client.post(f"/api/matches/{match_id}/bp", json=BP_DEMO)
    assert r.status_code == 200, r.text
    assert r.json()["report_url"] == f"/api/matches/{match_id}/bp/report"
    assert r.json()["entries"] > 0, "录了 BP 却一条结论都没有"

    # ---- 3. 复盘页真的给得出结论 ----
    review = client.get(r.json()["report_url"]).text
    assert "BP 复盘" in review
    assert "BP 操作分析" in review
    assert "BP 环节" in review
    assert "没有发现 BP 阶段需要改的地方" not in review
    # 概览卡不再摆「己方阵容 / 对方阵容」两行
    assert "己方阵容" not in review, "复盘页还摆着「己方阵容」"
    assert "对方阵容" not in review, "复盘页还摆着「对方阵容」"
    # 段位与预选：两页都别再提
    for gone in ("段位", "预选"):
        assert gone not in review, f"复盘页还有「{gone}」"

    # ---- 4. 复盘的坐标就是记录页的环节名（两页同一套话） ----
    # 页面里的环节下拉是从 PHASES 渲染的（中文在 JSON 里是 \uXXXX 转义，
    # 所以要把那段 JSON 抠出来解析，不能直接找字面量）。
    raw = re.search(r"const PHASES = (\{.*?\});", form, re.S)
    assert raw, "记录页没有把环节清单交给页面脚本"
    page_phases = json.loads(raw.group(1))
    assert set(page_phases.values()) == set(BP_PHASE_LABELS.values()), \
        "记录页的环节下拉与规则库用的环节名对不上"

    data = client.get(f"/api/matches/{match_id}/bp/report.json").json()
    labels = set(BP_PHASE_LABELS.values())
    for e in data["entries"]:
        assert e["step_label"] in labels, \
            f"复盘里的坐标「{e['step_label']}」不在记录页的环节里"

    # ---- 5. 两页指的是同一局 ----
    assert match_id in form and match_id in review


def test_the_withdrawn_phase_is_gone_from_the_record_form(client):
    """「进入 BP」那一环撤了 —— 记录页的下拉里不该再有它。

    它那三条规则全依赖段位与预选，而页面不再问这两项 ——
    留着这一环，录了也不会有结论。
    """
    assert client.post("/api/games/hok/rules/seed").status_code == 200
    match_id = _make_match(client)

    html = client.get(f"/api/matches/{match_id}/bp").text
    raw = re.search(r"const PHASES = (\{.*?\});", html, re.S)
    assert raw, "记录页没有把环节清单交给页面脚本"
    page_phases = json.loads(raw.group(1))
    assert "进入 BP" not in page_phases.values(), "记录页还有「进入 BP」这一环"
    assert "bp_start" not in page_phases, "记录页还有 bp_start 这个编码"
    assert len(page_phases) == 7, f"记录页的环节数不对：{sorted(page_phases.values())}"

    # 模式库页也不该再有这一组
    bp_html = client.get("/rules?bp=1").text
    assert "进入 BP" not in bp_html, "模式库页还有「进入 BP」这一组"


def test_an_old_record_with_the_withdrawn_phase_is_kept_intact(client):
    """旧记录里录过「进入 BP」的那一步，不能在保存时被换成别的环节。

    这一环撤了，但老记录还在 —— 那一行必须原样摆出来：
    下拉里选不中它，保存时它就会变成「禁用（阶段一）」，
    老数据被改坏。
    """
    assert client.post("/api/games/hok/rules/seed").status_code == 200
    match_id = _make_match(client)

    old = {
        "my_missing_frontline": True,
        "decisions": [
            {"side": "my", "phase": "bp_start", "action": "plan_comp", "hero": "澜"},
            {"side": "my", "phase": "pick_1", "action": "pick_damage",
             "hero": "沈梦溪"},
        ],
    }
    assert client.post(f"/api/matches/{match_id}/bp", json=old).status_code == 200

    html = client.get(f"/api/matches/{match_id}/bp").text
    legacy = re.search(r"const LEGACY_PHASES = (\{.*?\});", html, re.S)
    assert legacy, "记录页没有带上已撤下的环节的名字"
    assert json.loads(legacy.group(1)).get("bp_start") == "进入 BP", \
        "旧记录里那一步会显示成裸编码"
    assert "phaseKeys.indexOf(data.phase)" in html, \
        "旧记录里的已撤下环节没被保留（保存时会被换成别的环节）"

    # 录过的内容确实回到了表单里
    assert "bp_start" in html, "旧记录里的那一步没回到表单"


def test_the_two_bp_pages_have_the_same_skeleton(client):
    """记录页与复盘页的骨架必须一模一样。

    概览卡 → 一节 → 卡片 → 去处 → 页脚。
    记录页曾经多出说明文字、也多出 02/03 两节 —— 那是「解释」，
    而复盘页一句解释都没有，两页摆在一起就像两个产品。
    """
    assert client.post("/api/games/hok/rules/seed").status_code == 200
    match_id = _make_match(client)
    client.post(f"/api/matches/{match_id}/bp", json=BP_DEMO)

    form = client.get(f"/api/matches/{match_id}/bp").text
    review = client.get(f"/api/matches/{match_id}/bp/report").text

    for html in (form, review):
        assert 'class="panel hero' in html, "两页都要有概览卡"
        assert html.count('class="section-head"') == 1, \
            "两页都只该有一节（编号 01）"
        assert 'class="footer"' in html, "两页都要有页脚"
        assert "生成于" in html
        assert 'class="duo"' in html, "两页右上角都要有那两位"

    # 解释性文字：复盘页一句都没有，记录页也不该有
    assert "BP 在比赛之前" not in form
    assert "勾不出来的可以不勾" not in form
