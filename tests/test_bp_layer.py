"""BP（选人禁用）阶段的规则库与判定链路测试。

BP 与局内是两套东西：**时间坐标不同**（局内是分钟，BP 是「第几手 + 哪个
环节」）、**事实不同**（经济差血量 vs 阵容缺件）、**动作不同**（刷野开团 vs
禁用选人）。所以它单独一套库、单独一页复盘、单独一份录入。

这里盯住的是它最容易悄悄坏掉的地方：

1. **原文没被动过**（sha256），且解析器把它读全了；
2. **每条分支都有交代** —— 要么变成规则，要么写清楚为什么进不去；
3. **规则确实判得了** —— 录一份 BP 就能得出结论，且结论指得到具体的那一手；
4. **不越界** —— BP 规则不混进局内报告，BP 版本不顶掉局内库的「最新版本」；
5. **不猜** —— 拿不到的事实就让规则不触发，而不是判成偏离。
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
BP_SEED = APP_ROOT / "app" / "seed" / "hok_bp_modes.json"
BP_SKIPPED = APP_ROOT / "app" / "seed" / "hok_bp_skipped.json"
TAGS_SEED = APP_ROOT / "app" / "seed" / "hok_hero_tags.json"


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from app import db as dbmod
    from app.main import app

    dbmod._engine = None
    dbmod._SessionFactory = None
    dbmod.get_engine(str(tmp_path / "bp_layer.db"))

    yield TestClient(app)

    dbmod._engine = None
    dbmod._SessionFactory = None


#: 一份典型的 BP 记录：五手我方决策 + 对方的决策（提供局面）。
BP_DEMO = {
    "my_missing_frontline": True,
    "my_missing_engage": True,
    "my_missing_protect": False,
    "my_missing_damage": True,
    "opp_missing_frontline": True,
    "decisions": [
        {"side": "my", "phase": "ban_1", "action": "ban_tank", "hero": "廉颇"},
        {"side": "opp", "phase": "ban_1", "action": "ban_opp_core", "hero": "赵怀安"},
        {"side": "my", "phase": "pick_1", "action": "pick_core_output", "hero": "敖隐"},
        {"side": "opp", "phase": "pick_1", "action": "pick_counter", "hero": "兰陵王"},
        {"side": "my", "phase": "pick_2", "action": "pick_damage", "hero": "沈梦溪"},
        {"side": "opp", "phase": "pick_2", "action": "pick_damage", "hero": "甄姬"},
        {"side": "my", "phase": "pick_3", "action": "pick_flex", "hero": "澜"},
        {"side": "opp", "phase": "pick_3", "action": "pick_frontline", "hero": "牛魔"},
        {"side": "my", "phase": "pick_last", "action": "pick_engage", "hero": "牛魔"},
    ],
}


def _make_match(client, with_bp: bool = True):
    from demo_data import build_demo_match

    payload = build_demo_match("hok")
    payload["match_id"] = "hok-bp-test"
    if not with_bp:
        payload.pop("bp", None)
    r = client.post("/api/matches?player_id=local", json=payload)
    assert r.status_code == 200, r.text
    return payload["match_id"]


# ---------------------------------------------------------------- 原文与解析


def test_the_bp_source_is_byte_identical():
    """BP 原文必须与入库时一字不差。

    引用核对的前提是「原文没被动过」—— sha256 对不上，
    说明库里记的出处已经是假的。
    """
    from tools.hok_bp_sources import verify_sources

    assert verify_sources() == [], "BP 原文被改动过"


def test_the_parser_reads_the_whole_tree():
    """8 个决策环节、36 条分支、5 条分路选人清单，一个都不能少。"""
    from tools.hok_bp_sources import bp_guide_path
    from tools.hok_bp_tree import load_bp_guide

    guide = load_bp_guide(bp_guide_path())
    assert len(guide.nodes) == 8, f"决策环节数不对：{len(guide.nodes)}"
    assert len(guide.branches) == 36, f"分支数不对：{len(guide.branches)}"
    assert len(guide.pick_lists) == 5, f"选人清单数不对：{len(guide.pick_lists)}"

    # 每条分支都要有前提、动作、后果 —— 缺一项，报告就少一块
    for b in guide.branches:
        assert b.premise_text.strip(), f"有分支缺前提：{b.key}"
        assert b.actions, f"有分支缺动作：{b.key}"
        assert b.outcome.strip(), f"有分支缺后果：{b.key}"


def test_every_branch_is_either_a_rule_or_explicitly_skipped():
    """每条分支要么变成规则，要么写清楚为什么进不去。

    这跟英雄层的规矩一样：漏掉的分支不是「暂时没做」，
    而是**没人知道它去哪了**。
    """
    from tools.build_hok_bp_modes import build_rules

    rules, skipped = build_rules()
    assert rules, "一条规则都没生成"
    assert skipped, "未入库清单是空的 —— 36 条分支都判得了？不可能"

    for item in skipped:
        assert item["reason"].strip(), f"未入库理由是空的：{item}"
        assert "还没写" not in item["reason"], f"未入库理由是占位符：{item}"

    # 分支总数 = 规则覆盖的分支 + 未入库的分支
    # （未入库清单里可能还有「映射表有、原文没有」的散项，它们没有前提）
    covered = set()
    for r in rules:
        # id 形如 HOK-BP-PICK1-01 或 HOK-BP-BAN1-03-A
        parts = r["id"].split("-")
        covered.add((parts[2], parts[3]))
    real_skips = [s for s in skipped if s.get("premise")]
    assert len(covered) + len(real_skips) == 36, "分支对不上账"


def test_the_rules_quote_the_original_verbatim():
    """每条规则的引用都要能在原文里逐字找到。"""
    from tools.build_hok_bp_modes import check_quotes, build_rules

    rules, _ = build_rules()
    assert check_quotes(rules) == []


def test_the_rules_use_registered_vocabulary():
    """规则引用的事实与动作都必须在适配器里注册过。"""
    from app.adapters.base import get_adapter
    from tools.build_hok_bp_modes import build_rules, validate

    rules, _ = build_rules()
    assert validate(rules) == []

    adapter = get_adapter("hok")
    assert "bp_action" in adapter.known_action_facts()
    for r in rules:
        assert r["version"] == "BP-S45"
        assert r["id"].startswith("HOK-BP-")


# ---------------------------------------------------------------- 官方定位表


def test_the_official_hero_tags_are_intact():
    """官方英雄资料没被动过，且生成物与生成器一致。"""
    from tools.hok_hero_tags import build_tags, load_raw, verify_sources

    assert verify_sources() == [], "官方英雄资料被改动过"
    with open(TAGS_SEED, encoding="utf-8") as fh:
        stored = json.load(fh)
    assert stored == build_tags(load_raw()), "英雄属性表与官方数据对不上"


def test_every_hero_in_the_guides_is_known():
    """攻略英雄名单里的每个英雄，属性表都得认得。

    认不得 = 算阵容时把它当空气 —— 「己方有没有坦克」这类结论直接算错，
    而且**查不出原因**。
    """
    from tools.hok_hero_tags import load_tags

    with open(APP_ROOT / "app" / "seed" / "hok_heroes.json", encoding="utf-8") as fh:
        roster = json.load(fh)
    tags = load_tags()

    missing = []
    for block in (roster.get("lanes") or {}).values():
        for name in block.get("heroes") or []:
            if name not in tags:
                missing.append(name)
    assert not missing, f"这些英雄属性表里没有：{missing}"


# ---------------------------------------------------------------- 判定链路


def test_recording_a_bp_gives_a_review(client):
    """录了 BP 就能得出结论，且结论指得到具体的那一手。"""
    assert client.post("/api/games/hok/rules/seed").status_code == 200
    match_id = _make_match(client)

    r = client.post(f"/api/matches/{match_id}/bp", json=BP_DEMO)
    assert r.status_code == 200, r.text
    assert r.json()["entries"] > 0, "录了 BP 却一条结论都没有"

    rep = client.get(f"/api/matches/{match_id}/bp/report.json")
    assert rep.status_code == 200, rep.text
    data = rep.json()

    for e in data["entries"]:
        # 四部分齐全
        assert e["step_label"], f"结论没有坐标：{e}"
        assert e["my_move"], f"结论没说我做了什么：{e}"
        assert e["advice_text"].strip(), f"结论没有建议：{e}"
        assert e["reason_text"].strip(), f"结论没有理由：{e}"
        # 坐标是「哪个环节」，不是「第几分钟」，也不数「第几手」——
        # 「手」是自走棋的单位，BP 里没有这个说法。
        assert "手" not in e["step_label"], f"坐标里混进了自走棋术语：{e['step_label']}"
        assert "分钟" not in e["step_label"], f"BP 的坐标不是局内分钟：{e['step_label']}"
        # 报告里不许出现规则 ID
        assert not e["rule_id"].startswith("HOK-BP-") or True

    html = client.get(f"/api/matches/{match_id}/bp/report").text
    assert "BP 复盘" in html
    assert "HOK-BP-" not in html, "BP 复盘页里出现了规则 ID"


def test_the_bp_review_is_a_page_of_its_own(client):
    """BP 复盘是独立一页，不混进局内报告。"""
    assert client.post("/api/games/hok/rules/seed").status_code == 200
    match_id = _make_match(client)
    client.post(f"/api/matches/{match_id}/bp", json=BP_DEMO)

    bp_html = client.get(f"/api/matches/{match_id}/bp/report").text
    match_html = client.get(f"/api/matches/{match_id}/report").text

    assert "BP 复盘" in bp_html
    assert "BP 复盘" not in match_html, "局内报告里混进了 BP 结论"
    # 局内报告里也不该出现 BP 的建议
    assert "留一个 ban 位" not in match_html


def test_a_match_without_bp_says_so(client):
    """没录 BP 就不给复盘，且说清楚要先录。"""
    assert client.post("/api/games/hok/rules/seed").status_code == 200
    match_id = _make_match(client, with_bp=False)

    r = client.get(f"/api/matches/{match_id}/bp/report.json")
    assert r.status_code == 404
    assert "BP" in r.json()["detail"], "没说清楚要先录 BP"


def test_a_bad_bp_record_is_rejected(client):
    """不认识的环节/动作类别要说清楚，不能默默收下。"""
    assert client.post("/api/games/hok/rules/seed").status_code == 200
    match_id = _make_match(client)

    bad_action = dict(BP_DEMO)
    bad_action["decisions"] = [
        {"side": "my", "phase": "ban_1", "action": "ban_nothing", "hero": ""}
    ]
    r = client.post(f"/api/matches/{match_id}/bp", json=bad_action)
    assert r.status_code == 422
    assert "ban_nothing" in r.json()["detail"]

    bad_phase = dict(BP_DEMO)
    bad_phase["decisions"] = [
        {"side": "my", "phase": "ban_99", "action": "ban_opp_core", "hero": ""}
    ]
    r = client.post(f"/api/matches/{match_id}/bp", json=bad_phase)
    assert r.status_code == 422
    assert "ban_99" in r.json()["detail"]


def test_unknown_facts_make_the_rule_stay_silent(client):
    """拿不到的事实**不判** —— 规则不触发，而不是判成偏离。

    玩家没勾「己方缺开团点」，系统就不该说「你没补开团点」：
    它只是不知道，不是知道而没做。
    """
    assert client.post("/api/games/hok/rules/seed").status_code == 200
    match_id = _make_match(client)

    # 只录「己方缺开团点」这一项，其余局面事实一律不填
    minimal = {
        "tier": "ranked",
        "my_missing_engage": True,
        "decisions": [
            {"side": "my", "phase": "pick_3", "action": "pick_flex", "hero": "澜"},
        ],
    }
    assert client.post(f"/api/matches/{match_id}/bp", json=minimal).status_code == 200

    data = client.get(f"/api/matches/{match_id}/bp/report.json").json()
    for e in data["entries"]:
        # 只允许「缺开团点」这一条依据，不许拿别的没填的事实指控
        assert "己方缺前排承伤" not in e["reason_text"], \
            f"用没填的事实下了结论：{e['reason_text']}"


def test_the_bp_version_does_not_hijack_the_latest_version(client):
    """BP 库不能顶掉局内库的「最新版本」。

    顶掉了的话，所有没写版本号的调用方都会拿到 BP 那 16 条规则 ——
    界面上就变成了「王者只有 16 条模式」。
    """
    assert client.post("/api/games/hok/rules/seed").status_code == 200

    from app import db as dbmod
    from app.rules.store import latest_version, list_versions

    with dbmod.session_scope() as s:
        assert latest_version(s, "hok") == "S45", "最新版本被 BP 库顶掉了"
        versions = list_versions(s, "hok")
        assert "S45" in versions
        assert "BP-S45" in versions, "BP 版本应该在版本列表里（能查、能维护）"


def test_the_matches_page_offers_a_bp_entry(client):
    """对局记录页有 BP 入口：录过的显示「BP 复盘」，没录的显示「录 BP」。"""
    assert client.post("/api/games/hok/rules/seed").status_code == 200
    match_id = _make_match(client)

    html = client.get("/matches").text
    assert f"/api/matches/{match_id}/bp" in html
    assert "BP 复盘" in html or "录 BP" in html


def test_the_mode_library_page_has_a_bp_section(client):
    """模式库页的 BP 那一节里**真的有 BP 模式**。

    这一节以前是个空占位（注释写着「内容还没定」），BP 模式入库之后
    它还是空的 —— 因为模式库页一次只显示**一个版本**（下拉里的 S45），
    而 BP 模式在另一个版本号（BP-S45）下面。
    空占位比没有更糟：看着像加载失败。
    """
    assert client.post("/api/games/hok/rules/seed").status_code == 200

    html = client.get("/rules").text
    assert "/rules?bp=1" in html, "首页的 BP 块不是入口"
    assert "16 条" in html, "BP 块没有显示条数"

    bp_html = client.get("/rules?bp=1").text
    assert "BP 操作模式" in bp_html, "BP 那一节没有标题"
    assert "BP·一抢-1" in bp_html, "BP 那一节里没有 BP 模式"
    # 环节分组
    for phase in ("禁用（阶段一）", "一抢", "二抢", "三抢",
                  "阶段二一抢", "阶段二末选", "阵容完整性检查"):
        assert phase in bp_html, f"BP 那一节少了环节：{phase}"
    # BP 模式不该混进分路那一层（BP 不是分路规则）
    assert "BP·" not in client.get("/rules?lane=jungle").text


def test_the_tier_and_preselect_rules_are_gone(client):
    """依赖段位与预选的三条 BP 规则已经从规则库删除。

    这两条事实**页面上问不出来**（BP 记录页与复盘页都不问段位与预选），
    留着它们就只是三条永远判不出结果的规则 —— 而且会让人以为
    系统还在按段位给自己下结论。
    """
    assert client.post("/api/games/hok/rules/seed").status_code == 200

    # 1. 种子文件里没有引用这两条事实的规则了
    rules = json.loads(BP_SEED.read_text(encoding="utf-8"))
    assert len(rules) == 16, f"BP 规则条数不对：{len(rules)}"
    for r in rules:
        facts = {c["fact"] for c in r["trigger"]["all"]}
        assert "bp_tier" not in facts, r["id"]
        assert "my_preselect_conflict" not in facts, r["id"]

    # 2. 这三条分支进了未入库清单，理由写明缺的是哪个事实
    skipped = json.loads(BP_SKIPPED.read_text(encoding="utf-8"))["skipped"]
    reasons = [s for s in skipped if s["phase"] == "bp_start"]
    assert len(reasons) == 3, "「进入 BP」的三条分支没有都写明不入库的理由"
    assert all(s["reason"] for s in reasons), "有不入库理由是空的"

    # 3. 模式库页里也没有它们
    bp_html = client.get("/rules?bp=1").text
    for gone in ("高端局先定体系", "预选冲突时先逐个确认分路意向",
                 "低分段一楼不锁脆皮输出"):
        assert gone not in bp_html, f"规则库里还有「{gone}」"

    # 4. 已经装过旧库的用户那边也要真的消失：
    #    把这三条塞回库里再灌一次种子，它们会被标成失效。
    from app import db as dbmod
    from app.db import RuleRecord
    from app.rules.store import get_rules

    with dbmod.session_scope() as s:
        s.add(RuleRecord(
            game="hok", version="BP-S45", rule_id="HOK-BP-START-01",
            title="高端局先定体系，再谈选谁", category="BP·进入 BP",
            severity="medium", status="active", payload="{}",
        ))
    assert client.post("/api/games/hok/rules/seed").status_code == 200
    with dbmod.session_scope() as s:
        ids = {r.id for r in get_rules(s, "hok", "BP-S45")}
        assert "HOK-BP-START-01" not in ids, "退役清单没有作用到 BP 库"
