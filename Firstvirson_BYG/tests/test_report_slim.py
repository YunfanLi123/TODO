"""报告与对局记录页的**瘦身**约束。

2026-10-22 这一轮把报告改成了「只有操作模式分析一节」。改版最容易出的
问题不是新功能坏了，而是**旧内容没删干净** —— 页面上多留一块，
读者就多看一块，注意力就分走一份。

所以这里盯的是「不该有的东西不许在」：

1. 报告正文只有「操作模式分析」，而且每条恰好四部分；
2. 那些统计口径（执行度、曲线、分类、最需改进、无法判定）不许再出现；
3. 顶部不许再出现玩家、段位、名次；
4. 报告里不许出现规则 ID —— 那是给程序看的，不是给人看的；
5. 对局记录页：原始数据按钮没了，删除按钮在；
6. 删除接口真的能删掉一局，删完再取是 404。
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
    dbmod.get_engine(str(tmp_path / "report_slim.db"))

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


# ---------------------------------------------------------------- 报告


def test_the_report_has_only_the_mode_analysis_section(seeded):
    """报告正文只有「操作模式分析」一节。"""
    client, match_id = seeded
    html = client.get(f"/api/matches/{match_id}/report").text

    assert "操作模式分析" in html, "报告没有「操作模式分析」这一节"
    assert 'id="analysis"' in html

    # 旧的那些区块，一个都不许再出现
    for gone in ('id="curve"', 'id="categories"', 'id="weakest"',
                 'id="undetermined"', 'id="recognition"',
                 'id="timeline"', 'id="lanes"'):
        assert gone not in html, f"报告里还留着 {gone}"


def test_the_report_has_no_execution_statistics(seeded):
    """执行度那套统计口径整个去掉。"""
    client, match_id = seeded
    html = client.get(f"/api/matches/{match_id}/report").text

    for gone in ("规则执行度", "较首次对局", "模式覆盖率",
                 "符合规范", "偏离规范", "分类执行度",
                 "最需要改进的规则", "无法判定的部分"):
        assert gone not in html, f"报告里还有「{gone}」"


def test_the_report_header_only_shows_game_and_result(seeded):
    """顶部只留标题、游戏与结果。"""
    client, match_id = seeded
    html = client.get(f"/api/matches/{match_id}/report").text

    assert "游戏" in html and "结果" in html
    for gone in ("玩家", "段位", "名次"):
        assert gone not in html, f"报告顶部还有「{gone}」"


def test_every_entry_has_the_four_parts(seeded):
    """每条四部分：对局时间区间 / 原视频片段 / 原视频操作描述 / 建议操作描述。"""
    client, match_id = seeded
    html = client.get(f"/api/matches/{match_id}/report").text

    for label in ("对局时间区间", "原视频被建议部分片段",
                  "原视频操作的文字描述", "根据模式库建议操作的描述"):
        assert label in html, f"报告里少了「{label}」这一部分"
    assert "建议理由" in html


def test_the_report_shows_no_rule_ids(seeded):
    """规则 ID 是给程序看的，页面上不许出现。

    「HOK-HERO-JGL-ZYUN-T1200-01」这种串出现在报告里，
    读者既看不懂也没法用 —— 它只是把内部结构漏到了外面。
    """
    client, match_id = seeded
    html = client.get(f"/api/matches/{match_id}/report").text

    assert "HOK-HERO-" not in html, "报告里出现了英雄层规则 ID"
    assert "HOK-MODE-" not in html, "报告里出现了分路层规则 ID"


def test_the_report_says_the_time_range_in_chinese(seeded):
    """时间区间写成「第 12 ~ 14 分钟」这种中文。"""
    client, match_id = seeded
    html = client.get(f"/api/matches/{match_id}/report").text

    assert "分钟" in html
    assert "第 " in html


# ---------------------------------------------------------------- 对局记录页


def test_the_matches_page_has_no_raw_data_button(client):
    """「原始数据」按钮去掉了 —— 报告页自己就是给人看的那份。"""
    html = client.get("/matches").text
    assert "原始数据" not in html
    assert "report.json" not in html


def test_the_matches_page_has_no_execution_ring(seeded):
    """执行度圆环去掉了 —— 列表页只回答「我打过哪些局」。

    那个环上写的是这一局的执行度百分比（40、67 这样的数）。
    它是**判定口径**不是对局事实：同一局换个规则版本数字就变，
    摆在列表上会让人以为这个分数是这局游戏本身打出来的。
    """
    client, _ = seeded
    html = client.get("/matches").text

    assert 'class="ring"' not in html, "对局记录页还有执行度圆环"
    assert "执行度" not in html, "对局记录页还有执行度的说法"


def test_the_matches_page_has_a_delete_button(client):
    """每一局后面都有删除按钮。"""
    html = client.get("/matches").text
    assert "btn-del" in html, "对局记录页没有删除按钮"
    assert 'method: \'DELETE\'' in html or 'method:"DELETE"' in html, \
        "删除按钮没有发 DELETE 请求"


def test_the_matches_page_has_the_two_men_at_the_top_right(client):
    """右上角站着狄仁杰与诸葛亮，注视着屏幕里的人。"""
    html = client.get("/matches").text

    assert "狄仁杰与诸葛亮" in html, "对局记录页没有那两位"
    assert 'class="duo"' in html
    # 视线跟着光标走：视线向量写进 CSS 变量，瞳孔消费它
    assert "--lx" in html and "--ly" in html
    assert "mousemove" in html, "他们的视线没有跟着光标"


def test_the_report_page_has_the_two_men_too(seeded):
    """报告页右上角也站着这两人 —— 报告是内层页面里最常看的一页。"""
    client, match_id = seeded
    html = client.get(f"/api/matches/{match_id}/report").text

    assert 'class="duo"' in html, "报告页没有那两位"
    assert "狄仁杰与诸葛亮" in html
    assert "--lx" in html and "--ly" in html


def test_the_brand_word_is_a_seal_in_the_rail(client):
    """左栏那四个字排成两行两列 —— 像一枚印章，而不是一句说明文字。"""
    html = client.get("/matches").text

    assert 'class="brand-word"' in html
    assert "<i>对</i><i>局</i><i>之</i><i>外</i>" in html, \
        "产品名没有排成两行两列"


def test_the_cover_title_is_left_alone(client):
    """封面那个「对局之外」是大字标题，不跟着左栏一起改。

    封面是产品自己的门面，跟内层页面的窄栏是两件事。
    """
    html = client.get("/").text

    assert "<h1>对局之外</h1>" in html, "封面的大标题没了"
    assert "brand-word" not in html, "封面的标题被左栏那套样式带走了"


# ---------------------------------------------------------------- 删除接口


def test_deleting_a_match_removes_it(seeded):
    """删掉一局之后，它就没了。"""
    client, match_id = seeded

    assert client.get(f"/api/matches/{match_id}/report").status_code == 200

    r = client.delete(f"/api/matches/{match_id}")
    assert r.status_code == 200, r.text
    assert r.json().get("match_id") == match_id

    assert client.get(f"/api/matches/{match_id}/report").status_code == 404
    assert match_id not in client.get("/matches").text


def test_deleting_a_missing_match_is_404(seeded):
    """删不存在的对局要说清楚，而不是假装删掉了。"""
    client, _ = seeded
    r = client.delete("/api/matches/根本没有这一局")
    assert r.status_code == 404
    assert "不存在" in r.text
