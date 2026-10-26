"""规则库页面与 MOBA 抽取链路的**界面接线**测试。

这些断言盯的是「模板和接口对不对得上」，不是抽取算法本身
（算法在 :mod:`tests.test_moba_extract` 里）。

为什么值得单独测：这一层最容易坏的方式不是报错，而是**两边各自都还
「能用」**。抽取口写错成自走棋那个，页面照样渲染、按钮照样点得动，
只是永远返回 0 条候选；确认口写错成通用 bulk 口，规则照样入库，
只是没有位置的规则也被放行 —— 都没有异常，只有结果是错的。
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Dict, List

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.adapters import registry as _registry  # noqa: F401  触发适配器注册

APP_ROOT = Path(__file__).resolve().parent.parent
APP_TEMPLATES = APP_ROOT / "app" / "templates"

#: 一段典型的 MOBA 攻略：位置、阶段、动作三件齐备的有几句，
#: 缺件的也有几句 —— 用来确认界面上的抽取确实走的是 MOBA 抽取器。
GUIDE_TEXT = (
    "打野前期以刷野补经济为主，尽量别去gank。"
    "中期打野要抓单，抢先手。"
    "中路前期清线压塔。"
    "游走中期跟着打野入侵野区。"
    "辅助注意保护射手。"
)


@pytest.fixture
def client(tmp_path):
    """指向独立临时库的 TestClient。

    独立库是硬要求：和主库共享状态的话，这条规则「是否写入」会取决于
    上一轮跑过什么，同一份代码第二次跑就得出另一个结论。
    """
    from fastapi.testclient import TestClient

    from app import db as dbmod
    from app.main import app

    dbmod._engine = None
    dbmod._SessionFactory = None
    dbmod.get_engine(str(tmp_path / "rules_page.db"))

    yield TestClient(app)

    dbmod._engine = None
    dbmod._SessionFactory = None


# ---------------------------------------------------------------- 页面本身


def test_rules_page_defaults_to_honor_of_kings(client):
    r = client.get("/rules")
    assert r.status_code == 200
    # 探针必须是**页面内容**，不能是某个 JS 常量 ——
    # 常量只服务于按钮，删了按钮它就没了，而「这一页是王者的」这件事还在。
    for lane in ("打野", "中路", "对抗路", "发育路", "游走"):
        assert lane in r.text, f"规则库首页上没有 {lane}"


def test_rules_page_has_no_game_switcher(client):
    """页面上不该再有按游戏切换的入口。

    这个产品现在只做王者荣耀，另外两款连规则库都没有 ——
    留着切换，用户点过去只会看到空列表，还以为是规则库坏了。
    """
    html = client.get("/rules").text
    assert "/rules?game=" not in html, "规则库不该再有按游戏切换的链接"
    for dead in ("云顶", "金铲铲"):
        assert dead not in html, f"规则库页面上还出现 {dead}"


def test_rules_page_ignores_stale_game_query(client):
    """``?game=tft`` 是收敛前的旧链接。

    参数已经不接了，但老书签、老聊天记录里可能还带着它。表现应该是
    照常打开王者荣耀的规则库，而不是 500 —— 更不能默默显示云顶的规则，
    那些规则没有 position 维度，混进来分母就不对了。
    """
    r = client.get("/rules", params={"game": "tft"})
    assert r.status_code == 200
    for lane in ("打野", "中路", "对抗路", "发育路", "游走"):
        assert lane in r.text, f"老链接打开的不是王者的规则库（少了 {lane}）"
    for dead in ("云顶", "金铲铲", "TFT-"):
        assert dead not in r.text, f"老链接打开后混进了 {dead} 的规则"


def test_rules_page_has_no_rule_actions(client):
    """模式详情里不该再有「停用」和「修改历史」。

    这两件事都搬到了审核工作台：模式库页是**给玩家查怎么打的**，
    摆一排维护用的按钮在那儿，用户点「停用」把一条模式关掉之后，
    报告里那条建议就悄悄消失了 —— 而他根本不知道自己干了什么。
    """
    client.post("/api/games/hok/rules/seed")
    html = client.get("/rules?lane=jungle").text
    assert "HOK-MODE-" in html, "分路详情页上没有王者的模式"
    for dead in ("停用", "修改历史", "setSt(", "mode-acts"):
        assert dead not in html, f"模式库页面上还有 {dead}"
    # 接口本身还在（审核工作台要用），只是页面不再摆它
    assert client.get(
        "/api/games/hok/rules/S45/HOK-MODE-JGL-T0400-02/history"
    ).status_code == 200


def test_rules_page_has_no_authoring_panel(client):
    """填写与从攻略抽取这两个入口先不做，页面上就不能留着半截表单。

    留着比没有更糟：按钮在、样子也在，点下去才发现不能写规则，
    用户只会以为是自己用错了。
    """
    html = client.get("/rules").text
    for dead in ("/api/guides/moba/extract", "/api/guides/moba/confirm",
                 "pane-form", "pane-guide", "saveRule", "extractBtn",
                 "addTabs"):
        assert dead not in html, f"{dead} 还留在规则库页面上"

    # 攻略抽取的**接口**还在（那是另一条路），只是页面不再摆它
    assert client.post("/api/guides/moba/extract", json={
        "game": "hok", "version": "S38", "guide_title": "测试攻略",
        "text": GUIDE_TEXT,
    }).status_code == 200


def test_rules_page_has_no_stat_cards(client):
    """规则 / 生效 / 停用 / 从未触发 / 样本对局这几个统计卡去掉了。

    它们统计的是「库自己有多大规模」，而这个页面要做的是「查这一条
    模式怎么打的」。数字摆在这里只会让人以为那是重点。
    """
    client.post("/api/games/hok/rules/seed")
    html = client.get("/rules").text
    for gone in ("rb-top", "样本对局", "从未触发"):
        assert gone not in html, f"规则库页面上还留着 {gone}"


def test_rules_page_version_switch_only_when_needed(client):
    """只有一个版本时不该有版本下拉 —— 一个选项的下拉是纯噪音。

    真出现第二个版本时它必须回来，而且不能带 game 参数
    （游戏已经收敛成王者荣耀了）。
    """
    html = client.get("/rules").text
    assert 'id="verSel"' not in html, "只有一个版本时不该出现版本下拉"

    client.post("/api/games/hok/rules/seed")

    from app.core.dsl import Rule
    from app.db import session_scope
    from app.rules.store import upsert_rule

    with session_scope() as s:
        upsert_rule(s, Rule.from_dict({
            "id": "HOK-NEXT-1", "game": "hok", "version": "S46",
            "title": "下一版本的模式", "category": "打野·前期",
            "trigger": {"all": [
                {"fact": "position", "op": "==", "value": "jungle"},
                {"fact": "time_min", "op": "between", "value": [0, 4]},
            ]},
            "expect": {"fact": "objective_action", "op": "==",
                       "value": "farm_jungle"},
            "severity": "medium", "status": "active",
        }))

    html = client.get("/rules").text
    assert 'id="verSel"' in html, "有两个版本时该出现版本下拉"
    assert "game=" not in html, "版本切换不该再带 game 参数"


def test_rules_page_counts_use_latest_version(client):
    """分路卡片上的条数按最新版本统计。

    用别的版本统计的话，页面上写着 10 条而报告里查的是另一套，对不上。
    """
    client.post("/api/games/hok/rules/seed")
    seeded = client.get("/api/games/hok/rules").json()["count"]
    assert seeded["total"] > 0

    html = client.get("/rules").text
    counts = [int(x) for x in re.findall(r'<span class="lc-n">(\d+) 条</span>', html)]
    assert counts, "分路卡片上没有条数"
    assert sum(counts) == seeded["active"], \
        f"分路条数加起来对不上库里 active={seeded['active']}"


# ------------------------------------------------- 三层结构：分路 → 英雄 → 模式


def test_rules_page_starts_with_lane_choice(client):
    """点进来是**选分路**，不是一长串模式。"""
    client.post("/api/games/hok/rules/seed")
    html = client.get("/rules").text
    for code in ("jungle", "mid", "side", "carry", "roam"):
        assert f'href="/rules?lane={code}"' in html, f"分路 {code} 进不去"
    # 还没选分路时不该出现任何一条模式
    assert 'class="mode-name"' not in html, "还没选分路就渲染出模式了"


def test_rules_page_shows_no_coverage_and_no_source(client):
    """模式库里不摆「覆盖」与「来源」。

    覆盖（被检验几回合、偏离率）是**内部账**：它衡量的是「这条模式被触发过
    多少回」，摆在模式旁边会让人以为那是模式的质量分。
    来源那一行是给维护者看的溯源，正文里「前提 / 该做 / 后果」已经说清了。
    """
    client.post("/api/games/hok/rules/seed")
    html = client.get("/rules?lane=jungle&hero=赵云").text
    assert "覆盖：" not in html, "模式里还有「覆盖」"
    assert "来源：" not in html, "模式里还有「来源」"
    assert "被检验" not in html, "模式里还有覆盖率的读数"
    assert "偏离率" not in html, "模式里还有偏离率"


def test_the_library_page_has_no_sidebar_and_a_back_link(client):
    """库页不挂应用侧边栏，左上角是「返回」。

    那一列是给「上传 / 对局」用的，看库的时候它只是噪音；
    库从封面第三页进来，那儿是所有库的目录 —— 返回就回那儿。
    """
    client.post("/api/games/hok/rules/seed")
    for url in ("/rules", "/rules?bp=1", "/rules?lane=side",
                "/rules?lane=side&hero=哪吒"):
        html = client.get(url).text
        assert 'class="app-nav"' not in html, f"{url} 还挂着应用侧边栏"
        assert "全部分路" not in html, f"{url} 右上角还有「全部分路」"
        assert 'href="/#page-3"' in html, f"{url} 左上角没有返回"


def _container_inner(html: str, cls: str) -> str:
    """取 ``class=cls`` 那个容器里面的内容。

    用正则 ``(.*?)</div>`` 取不到 —— 容器里第一个 ``</div>`` 就把内容截断了
    （BP 块自己还套着一层 div）。这里按标签配平找真正的结束位置。
    """
    open_tag = f'<div class="{cls}">'
    start = html.index(open_tag)
    depth = 0
    for m in re.finditer(r"<div\b[^>]*>|</div>", html[start:]):
        if m.group(0).startswith("</"):
            depth -= 1
            if depth == 0:
                return html[start + len(open_tag): start + m.start()]
        else:
            depth += 1
    raise AssertionError(f"容器 {cls} 没有闭合")


def test_bp_block_matches_the_lane_cards(client):
    """BP 块要和分路块是同一套规格。

    两个地方曾经不一致：

    * 标题字号 —— BP 15px、分路名 19px。同一页两种字号，
      BP 那个看着像另一个产品的东西。
    * 块宽 —— BP 块是分路卡容器的**外面**的独立一块，
      宽度靠各自撑开，看着总比下面那排宽一点。

    现在 BP 块和五条分路卡是同一个 flex 容器里的兄弟，
    宽度由 stretch 统一决定；标题共用一条字号规则。
    """
    client.post("/api/games/hok/rules/seed")
    html = client.get("/rules").text

    # 1. 标题字号：BP 块现在**就是一张分路卡**（.lc-name），
    #    字号自然共用同一条规则 —— 不再有两处能各改各的。
    css = (APP_TEMPLATES / "rules.html").read_text("utf-8")
    m = re.search(r"\.lc-name\s*\{([^}]*)\}", css)
    assert m, "没有 .lc-name 的字号规则"
    rule = m.group(1)
    assert "font-size" in rule and "font-weight" in rule, \
        "共用的字号规则里没有字号/字重"
    # 旧的单点规则不该再留着（留着就等于又有一处能单独改）
    assert not re.search(r"^\.bp-slot-h\s*\{", css, re.M), "BP 标题还留着单独的字号规则"
    assert not re.search(r"^\.bp-slot\s*\{", css, re.M), "BP 占位块的样式还留着"

    # 2. 块宽：BP 块和分路卡在同一个容器里
    inner = _container_inner(html, "lane-cards")
    assert 'class="lane-card bp-card"' in inner, "BP 块不是分路卡（宽度就没法保证一致）"
    assert inner.index('class="lane-card bp-card"') < inner.index('class="lane-card"'), \
        "BP 块应该在分路卡前面（第一行）"
    assert inner.count('class="lane-card') == 6, "卡片数量不对（BP + 五条分路）"

    # 3. BP 块是**入口**：点进去是 BP 库那一节
    assert 'href="/rules?bp=1"' in inner, "BP 块点不进去"


def test_rules_lane_page_has_general_modes_and_heroes(client):
    """分路页：先给这条分路前中后期的总体模式，再给英雄。

    英雄层现在是**满的**（从 5 份英雄级攻略生成，123 个英雄），
    所以这一页除总体模式之外，还要列出这条分路下的英雄。
    """
    client.post("/api/games/hok/rules/seed")
    html = client.get("/rules", params={"lane": "side"}).text
    assert "该分路前中后期总体操作模式" in html
    assert "对抗路-前期-1" in html
    # 英雄这一层只列**确实有模式**的英雄
    hero_links = re.findall(r'href="/rules\?lane=side&hero=([^"]+)"', html)
    assert hero_links, "英雄层有内容，页面上却一个英雄都没列"
    assert 'class="hero-cards"' in html, "页面上没有英雄小节"
    # 别的分路的模式不该混进来
    assert "打野-前期-1" not in html


def test_rules_hero_page_has_only_that_hero(client):
    """英雄页：这个英雄前中后期的模式，且**只**有这个英雄的。"""
    client.post("/api/games/hok/rules/seed")
    r = client.get("/rules", params={"lane": "side", "hero": "哪吒"})
    assert r.status_code == 200
    html = r.text
    assert "该英雄前中后期操作模式" in html
    assert "对抗路-哪吒-前期-1" in html
    # 别人的模式不该出现在这一页
    assert "对抗路-吕布" not in html, "英雄页里混进了别的英雄的模式"
    # 通则那一节也不该重复出现（选了英雄就走英雄层）
    assert "对抗路-前期-1" not in html, \
        "选了英雄却还在给分路通则，同一条建议会被说两遍"


def test_mode_names_number_within_the_same_prefix(client):
    """序号只在**名字前面部分完全相同时**才加一。

    「对抗路-前期」下面有 12 条，就叫 1 到 12；
    「对抗路-中期」是另一个前缀组，序号重新从 1 开始。
    英雄层也一样：「对抗路-哪吒-前期」是一个前缀组。
    """
    client.post("/api/games/hok/rules/seed")

    html = client.get("/rules", params={"lane": "side"}).text
    names = re.findall(r'<span class="mode-name">([^<]+)</span>', html)
    assert names, "这一页没有模式"

    # 同一前缀组内，序号从 1 连续递增，不重不跳
    groups: Dict[str, List[int]] = {}
    for name in names:
        prefix, _, tail = name.rpartition("-")
        groups.setdefault(prefix, []).append(int(tail))
    for prefix, nums in groups.items():
        assert nums == list(range(1, len(nums) + 1)), \
            f"{prefix} 的序号不是从 1 连续递增：{nums}"

    # 英雄页：只有这个英雄的那一组
    hero_html = client.get("/rules", params={"lane": "side", "hero": "哪吒"}).text
    hero_names = re.findall(r'<span class="mode-name">([^<]+)</span>', hero_html)
    assert hero_names, "英雄页没有模式"
    assert all(n.startswith("对抗路-哪吒-") for n in hero_names), \
        f"英雄页里混进了别人的模式：{hero_names}"


# ------------------------------------------------- 界面 JS ↔ 接口 的实际契约


def test_extract_then_confirm_round_trip(client):
    """照着页面 JS 的顺序跑一遍：抽取 → 原样回传 → 入库。"""
    r = client.post("/api/guides/moba/extract", json={
        "game": "hok", "version": "S38",
        "guide_title": "测试攻略", "text": GUIDE_TEXT,
    })
    assert r.status_code == 200
    data = r.json()
    cands = data["candidates"]
    assert cands, "这段文本应当抽出至少一条候选"

    for c in cands:
        conds = c["trigger"]["all"]
        assert any(x["fact"] == "position" for x in conds), \
            f"候选 {c['title']} 没带位置条件"
        # 页面把 CANDS[i] 原样回传，所以这里必须不需要任何字段搬运
        assert {"title", "trigger", "expect", "position_label", "phase_label"} \
            <= set(c), "候选字典缺字段，页面原样回传会对不上"

    w = client.post("/api/guides/moba/confirm", json={
        "game": "hok", "version": "S38",
        "guide_title": "测试攻略", "candidates": cands,
    })
    assert w.status_code == 200
    body = w.json()
    assert body["imported"], body
    assert not body["failed"], body
    assert body.get("skipped_no_position", 0) == 0

    # 回到库里核对：每一条都真的挂上了位置，而不是只在候选阶段挂着
    stored = client.get("/api/games/hok/rules",
                        params={"version": "S38"}).json()["rules"]
    assert len(stored) == len(cands)
    for rule in stored:
        conds = rule["trigger"]["all"]
        pos = [x for x in conds if x["fact"] == "position"]
        assert pos, f"入库的规则没有位置条件：{rule['title']}"
        assert pos[0]["value"] != "unknown", \
            f"入库的规则位置是「未指定」：{rule['title']}"
        # 分类写成「打野·前期」这种形式，列表上才分得清是谁的活
        assert "·" in rule["category"], rule["category"]


def test_confirm_rejects_candidate_without_position(client):
    """页面上把 position 那行的 fact 改成别的，规则必须被拒。

    这条正是确认口不能用通用 bulk 口的原因：bulk 不看位置，
    改一个下拉框就能把「前期打野该做什么」写成一条谁都能触发的规则。
    """
    r = client.post("/api/guides/moba/extract", json={
        "game": "hok", "version": "S38",
        "guide_title": "测试攻略", "text": GUIDE_TEXT,
    })
    cands = r.json()["candidates"]
    assert cands

    tampered = []
    for c in cands:
        c = dict(c)
        c["trigger"] = {"all": [dict(x) for x in c["trigger"]["all"]],
                        "any": [], "none": []}
        for cond in c["trigger"]["all"]:
            if cond["fact"] == "position":
                cond["fact"] = "time_min"          # 抹掉位置，只留下时间
        tampered.append(c)

    w = client.post("/api/guides/moba/confirm", json={
        "game": "hok", "version": "S38",
        "guide_title": "测试攻略", "candidates": tampered,
    })
    assert w.status_code == 200
    body = w.json()
    assert body["skipped_no_position"] == len(tampered), body
    assert not body["imported"], "没有位置的规则不该入库"
    assert "未指定位置" in body["message"]

    listed = client.get("/api/games/hok/rules",
                        params={"version": "S38"}).json()["rules"]
    assert listed == [], "库里不该留下任何被拒的规则"


def test_extract_reports_why_sentences_were_skipped(client):
    """页面把 skipped 渲染成「缺什么｜原句」，所以格式得对得上。"""
    r = client.post("/api/guides/moba/extract", json={
        "game": "hok", "version": "S38", "text": GUIDE_TEXT,
    })
    skipped = r.json()["skipped"]
    assert skipped, "这段文本里有句子该被跳过却没给理由"
    for item in skipped:
        assert set(item) == {"sentence", "reasons"}
        assert item["reasons"], "跳过理由不能是空串"

    reasons = " ".join(i["reasons"] for i in skipped)
    assert "缺分路" in reasons or "缺阶段" in reasons or "主观" in reasons


def test_candidate_between_value_survives_a_round_trip(client):
    """``time_min between [0, 6]`` 的取值是数组。

    页面把候选原样回传，所以数组不能在中途变成字符串 —— 一旦变成
    "0,6"，DSL 会在入库时拒掉，用户填完整页才发现白填。
    """
    r = client.post("/api/guides/moba/extract", json={
        "game": "hok", "version": "S38", "text": GUIDE_TEXT,
    })
    saw_between = False
    for c in r.json()["candidates"]:
        for cond in c["trigger"]["all"]:
            if cond["op"] == "between":
                saw_between = True
                assert isinstance(cond["value"], (list, tuple))
                assert len(cond["value"]) == 2
                assert all(isinstance(v, (int, float)) for v in cond["value"])
    assert saw_between, "MOBA 候选本该带 time_min between 条件"


# ------------------------------------------------- 分路：规则与录对局两侧


def test_position_choices_exist_only_where_positions_do():
    """标注页只在该游戏有分路概念时才给下拉。

    自走棋没有分路，渲染一个空的下拉框比不渲染更糟：用户会以为
    「原来可以选」，选了之后发现没有任何区别。
    """
    from app.adapters.base import get_adapter
    from app.collection.video_api import _position_choices

    hok = _position_choices(get_adapter("hok"))
    assert [c["value"] for c in hok] == [
        "jungle", "mid", "side", "carry", "roam", "unknown"]
    assert [c["label"] for c in hok] == [
        "打野", "中路", "对抗路", "发育路", "游走", "未指定"]
    # 「未指定」留在列表里是有意的：它对应「这一局不评判」，不是一个位置
    assert hok[-1]["label"] == "未指定"

    for game in ("tft", "jcc"):
        assert _position_choices(get_adapter(game)) == []


# 「每条操作模式都必须挂位置」这条约束由 tests/test_mode_library.py 集中检查
# （那里连同「同一格子里两条模式不能互相指控」一起验）。这个文件只管界面接线，
# 不再重复一份 —— 重复的断言会在有人改一处时悄悄失效。
