"""端到端集成测试：把四层串起来跑一遍真实链路。

这里刻意不 mock 任何一层 —— 数据库、适配器、规则库、内核、报告全部真跑。
单元测试保证每层正确，集成测试保证它们接得上，而接不上往往才是真问题。
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.adapters.base import get_adapter
from app.adapters.registry import REGISTRY
from app.collection import guide as guide_col
from app.collection import match as match_col
from app.core.dsl import Rule
from app.core.context import MatchContext
from app.db import Base, get_engine, session_scope
from app.report.builder import build_match_report
from app.report.views import render_report
from app.rules.store import (
    count_rules,
    get_rules,
    latest_version,
    list_versions,
    load_seed_library,
)

from demo_data import DEMO_GUIDE, build_demo_match

ALL_GAMES = sorted(REGISTRY)

#: 目前**填了操作模式库**的游戏。
#:
#: 云顶 / 金铲铲只保留适配器模块，资料还没做 —— 所以依赖「有库」的测试只对
#: 王者荣耀跑。给它们硬造一份库来跑测试，等于把没验证过的数据当成真的：
#: 库是空的本来就是当前的事实，测试应当如实反映它。
MODE_GAMES = ["hok"]

#: 只保留模块、还没填资料的游戏。
UNFILLED_GAMES = [g for g in ALL_GAMES if g not in MODE_GAMES]


@pytest.fixture(scope="module")
def seeded_engine():
    """建库并载入全部种子规则库，整个模块共用一个数据库文件。"""
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "test_pipeline.db",
    )
    if os.path.exists(db_path):
        os.remove(db_path)

    from app import db as dbmod

    dbmod._engine = None
    dbmod._SessionFactory = None
    get_engine(db_path)

    with session_scope() as s:
        for game in ALL_GAMES:
            if not latest_version(s, game):
                load_seed_library(s, game)

    yield db_path


@pytest.fixture
def session(seeded_engine):
    """每个测试一个 SAVEPOINT。

    没有这层隔离的话，任何一个测试失败都会让 session_scope 回滚整个模块的
    事务，连带把种子规则库也回滚掉 —— 后面的测试于是拿到一个空库，
    报出与真实缺陷无关的失败。这种级联假失败比直接报错更难查。
    """
    with session_scope() as s:
        nested = s.begin_nested()
        try:
            yield s
        finally:
            if nested.is_active:
                nested.rollback()


# --------------------------------------------------------------------------
# 规则库
# --------------------------------------------------------------------------


class TestRuleLibrary:
    def test_filled_games_have_a_library(self, session):
        for game in MODE_GAMES:
            if not latest_version(session, game):
                load_seed_library(session, game)
            assert latest_version(session, game), f"{game} 未载入操作模式库"

    @pytest.mark.parametrize("game", UNFILLED_GAMES)
    def test_unfilled_games_have_no_library_yet(self, session, game):
        """还没填资料的游戏，库里就应该是空的。

        断言这一点而不是跳过，是因为「空」和「漏载」在界面上长得一样 ——
        前者是计划内，后者是缺陷。测试必须能区分它们。
        """
        assert latest_version(session, game) is None, (
            f"{game} 本应还没有操作模式库（模块保留、资料后补），"
            f"却载入了 {latest_version(session, game)}"
        )
        # 空库不能报错，必须安静地返回空结果
        result = load_seed_library(session, game)
        assert result.accepted == []

    #: 每款游戏「生效模式条数」的合理区间。
    #:
    #: 王者的 84 条不是注水 —— 它们是《S45 分路决策树》31 个时间节点里
    #: 每一条分支（或一组可接受动作相同的分支）对应的一条，
    #: 每条都有逐字引用与明确时间窗。真正防注水的是
    #: ``tests/test_mode_library.py`` 里的「库必须与生成器输出逐字一致」，
    #: 那条测试挡住的是**手改库**。
    # 现行库 = 分路层 84 条 + 英雄层 1596 条（123 个英雄 × 6 个时间节点，
    # 从 5 份英雄级攻略生成）。上界给得宽一点，但下界要卡住 ——
    # 英雄层没加载进来时条数会掉到 84，报告上就只剩分路通则。
    LIBRARY_SIZE_BOUNDS = {"hok": (1500, 2200)}

    @pytest.mark.parametrize("game", MODE_GAMES)
    def test_library_size_is_credible(self, session, game):
        """模式宁可少也不能注水 —— 条数太少说明没积累，太说明没筛选。"""
        version = latest_version(session, game)
        stats = count_rules(session, game, version)
        assert stats["active"] >= 8, f"{game} 模式过少"
        lo, hi = self.LIBRARY_SIZE_BOUNDS.get(game, (8, 60))
        assert stats["active"] <= hi, f"{game} 模式过多，疑似注水"
        assert stats["deprecated"] == 0, "种子库不应有失效条目"

    @pytest.mark.parametrize("game", MODE_GAMES)
    def test_every_rule_is_decidable(self, session, game):
        """全部模式必须通过可判定性准入 —— 官方维护的库不允许混入水货。"""
        from app.core.dsl import DEFAULT_SUBJECTIVE_MARKERS, check_admission

        adapter = get_adapter(game)
        rules = get_rules(session, game, latest_version(session, game))
        assert rules, f"{game} 没有模式"

        for rule in rules:
            reasons = check_admission(
                rule,
                adapter.known_facts(),
                DEFAULT_SUBJECTIVE_MARKERS,
                adapter.known_action_facts(),
            )
            assert not reasons, f"{rule.id} 未通过准入：{reasons}"

    @pytest.mark.parametrize("game", MODE_GAMES)
    def test_rules_have_provenance(self, session, game):
        """每条模式都要能指回原始攻略，否则报告结论无法复查。"""
        for rule in get_rules(session, game, latest_version(session, game)):
            assert rule.source is not None, f"{rule.id} 缺溯源"
            assert rule.source.guide, f"{rule.id} 缺来源名称"


# --------------------------------------------------------------------------
# 对局分析
# --------------------------------------------------------------------------


class TestAnalysis:
    @pytest.mark.parametrize("game", MODE_GAMES)
    def test_demo_match_produces_located_gaps(self, session, game):
        """核心产品主张的回归测试：结论必须定位到具体回合与具体动作。"""
        adapter = get_adapter(game)
        raw = build_demo_match(game)

        record, issues = match_col.save_match(session, adapter, raw)
        rules = get_rules(session, game, record.game_version)
        ctx = adapter.build_context(record.get_payload())
        report = build_match_report(rules, ctx)

        stats = report["stats"]
        assert stats["triggered"] > 0, f"{game} 示例对局未触发任何模式"
        assert stats["coverage"] <= 1.0
        assert report["gaps"], f"{game} 示例对局未产生任何差距"

        for gap in report["gaps"]:
            # 每条差距都必须能被复查
            assert gap["round_label"], "差距未定位到回合"
            assert gap["rule_id"], "差距未指向具体模式"
            assert gap["expected_text"], "差距缺少应做动作"
            assert gap["actual_text"], "差距缺少实际动作"
            assert gap["why_text"], "差距缺少「为什么」"
            assert gap["rule_title"], "差距缺少模式标题"
            # 报告正文只有三件事：实际 / 该做 / 为什么
            assert "situation_text" not in gap, (
                "报告里不该再有「当时局面」—— 前提条件写进「为什么」里了"
            )

    def test_situation_text_speaks_chinese_and_skips_unfilled(self):
        """「当时局面」要能直接读，不能漏出内部代码。

        两种坏情况都堵住：

        1. 内部代码外泄 —— 打印 ``jungle`` / ``unknown`` 这种
           内部编码，用户看不懂那是什么意思。
        2. 空值占位 —— 玩家没填英雄，于是句子里出现
           「… · 我方英雄 · 对位优劣 …」，读起来像内容被截断。
        """
        from app.report.builder import describe_situation

        adapter = get_adapter("hok")
        text = describe_situation(adapter, {
            "time_min": 4,
            "position": "jungle",
            "matchup": "unknown",
            "has_ult_ready": True,
            "is_decision_point": False,
            "my_hero": "",
        })

        assert "jungle" not in text, f"局面摘要漏出了内部编码：{text}"
        assert "unknown" not in text, f"局面摘要漏出了内部编码：{text}"
        assert "打野" in text, f"局面摘要没把分路说成人话：{text}"
        assert "不确定" in text, f"局面摘要没把对位优劣说成人话：{text}"
        assert "是" in text and "否" in text, f"布尔取值没中文化：{text}"
        # 没填的英雄不该占一个位置
        assert "我方英雄" not in text, f"空取值占了位：{text}"

    def test_situation_text_falls_back_when_nothing_filled(self):
        """一个事实都没填时要说「数据缺失」，不能印出空句。"""
        from app.report.builder import describe_situation

        adapter = get_adapter("hok")
        assert describe_situation(adapter, {}) == "局面数据缺失"
        assert describe_situation(adapter, {"my_hero": ""}) == ""

    def test_hok_demo_is_judged_by_lane_and_phase(self, session):
        """示例对局必须按「分路 × 阶段」给出结论。

        这条是「按分路的系统思维」落地的回归测试：如果哪天退回到
        「一堆规则各判各的」，lanes 里就不会有分路和阶段，
        报告也就答不上「我这条分路哪一段做错了」。
        """
        adapter = get_adapter("hok")
        record, _ = match_col.save_match(
            session, adapter, build_demo_match("hok")
        )
        rules = get_rules(session, "hok", record.game_version)
        ctx = adapter.build_context(record.get_payload())
        report = build_match_report(rules, ctx)

        lanes = report["lanes"]
        assert lanes["findings"], "没有给出任何分路结论"
        assert lanes["lanes"], "没有按分路分组"
        assert lanes["lanes"][0]["lane"] == "jungle"

        for f in lanes["findings"]:
            assert f["lane"] == "jungle", f
            assert f["phase"] in ("early", "mid", "late", "endgame"), f
            assert f["should_do"], f"模式 {f['mode_id']} 没写应该做什么"
            assert f["how_to_fix"], f"模式 {f['mode_id']} 没写怎么改"
            assert f["verdict"] in (
                "ok", "occasional", "systemic", "never", "undetermined")

        # 三个阶段都该有结论 —— 少一段说明时间窗对齐出了问题
        phases = {f["phase"] for f in lanes["findings"]}
        assert {"early", "mid", "late"} <= phases, f"缺阶段：{phases}"

    def test_hok_demo_reports_what_to_do_about_it(self, session):
        """出问题的那几段，必须同时给出「你实际在做什么」和「怎么改」。"""
        adapter = get_adapter("hok")
        record, _ = match_col.save_match(
            session, adapter, build_demo_match("hok")
        )
        rules = get_rules(session, "hok", record.game_version)
        ctx = adapter.build_context(record.get_payload())
        report = build_match_report(rules, ctx)

        problems = [f for f in report["lanes"]["findings"] if f["verdict"]
                    in ("systemic", "never", "occasional")]
        assert problems, "示例对局里应至少有一条问题"
        for f in problems:
            assert f["actually_doing"], (
                f"{f['mode_id']} 判了问题却说不出玩家当时在做什么")
            assert f["how_to_fix"], f"{f['mode_id']} 判了问题却没给改法"

    @pytest.mark.parametrize("game", MODE_GAMES)
    def test_compliance_rate_is_none_or_in_range(self, session, game):
        adapter = get_adapter(game)
        record, _ = match_col.save_match(session, adapter, build_demo_match(game))
        rules = get_rules(session, game, record.game_version)
        ctx = adapter.build_context(record.get_payload())
        rate = build_match_report(rules, ctx)["stats"]["compliance_rate"]

        if rate is not None:
            assert 0.0 <= rate <= 1.0, f"{game} 执行度越界：{rate}"

    def test_reanalysis_is_deterministic(self, session):
        """同一份数据 + 同一版规则，必须得出同一结论。"""
        adapter = get_adapter("tft")
        raw = build_demo_match("tft")
        record, _ = match_col.save_match(session, adapter, raw)
        rules = get_rules(session, "tft", record.game_version)

        runs = []
        for _ in range(3):
            ctx = adapter.build_context(record.get_payload())
            rep = build_match_report(rules, ctx)
            runs.append((rep["stats"], [g["round_label"] for g in rep["gaps"]]))

        assert all(r == runs[0] for r in runs), "分析结果不可复现"

    def test_empty_rounds_rejected(self, session):
        from app.collection.match import MatchValidationError

        raw = build_demo_match("tft")
        raw["rounds"] = []
        with pytest.raises(MatchValidationError):
            match_col.save_match(session, get_adapter("tft"), raw)

    @pytest.mark.parametrize(
        "value,expect_date",
        [
            ("2026-09-10 21:30", "2026-09-10"),        # 系统自身产出的格式
            ("2026-09-10 21:30:00", "2026-09-10"),
            ("2026-09-10", "2026-09-10"),
            ("2026/09/10 21:30", "2026-09-10"),
        ],
    )
    def test_played_at_formats(self, session, value, expect_date):
        """对局时间解析。

        这不是可有可无的细节：解析失败会静默回退到「现在」，于是所有对局
        堆在同一天，成长曲线的时间轴塌成一条竖线 —— 而「执行度随时间怎么变」
        正是成长曲线存在的理由。
        """
        raw = build_demo_match("tft")
        raw["match_id"] = f"date-{expect_date}-{abs(hash(value)) % 9999}"
        raw["played_at"] = value
        record, _ = match_col.save_match(session, get_adapter("tft"), raw)
        assert record.played_at.strftime("%Y-%m-%d") == expect_date

    def test_moba_string_phase_is_accepted(self, session):
        """MOBA 的 phase 是字符串枚举，不能因为类型不同就拒绝录入。

        两款自走棋的小阶段是整数（1/2/3），MOBA 是 early/mid/late ——
        时间坐标本来就不同，接口不该逼调用方把两者硬塞进同一个 int。
        """
        from app.collection.match import MatchValidationError

        raw = build_demo_match("hok")
        assert isinstance(raw["rounds"][0]["phase"], str)

        adapter = get_adapter("hok")
        record, issues = match_col.save_match(session, adapter, raw)
        assert record.id
        rules = get_rules(session, "hok", record.game_version)
        ctx = adapter.build_context(record.get_payload())
        report = build_match_report(rules, ctx)
        assert report["stats"]["triggered"] > 0

        # 重建后 phase 仍在，且报告定位用的是可读时间标签
        ctx2 = MatchContext.from_dict(record.get_payload())
        assert ctx2.rounds[0].facts["phase"] in ("early", "mid", "late", "endgame")
        assert ctx2.rounds[0].label.startswith("第 ")

    def test_string_phase_flows_through_api_model(self):
        """API 层也必须接受字符串 phase —— 否则 hok 根本录不进来。"""
        from app.main import MatchInput

        payload = build_demo_match("hok")
        model = MatchInput(**{**payload, "rounds": payload["rounds"]})
        assert model.rounds[0].phase == payload["rounds"][0]["phase"]

    def test_played_at_not_all_today(self, session):
        """回归：多局历史必须落在不同日期，否则曲线横轴无意义。"""
        dates = set()
        for i in range(3):
            raw = build_demo_match("tft")
            raw["match_id"] = f"spread-{i}"
            raw["played_at"] = f"2026-0{7+i}-0{i+1} 20:00"
            record, _ = match_col.save_match(session, get_adapter("tft"), raw)
            dates.add(record.played_at.date())
        assert len(dates) == 3


# --------------------------------------------------------------------------
# 成长曲线
# --------------------------------------------------------------------------


class TestGrowthCurve:
    def test_curve_spans_multiple_matches(self, session):
        from app.core.curve import build_curve

        adapter = get_adapter("hok")
        analyses = []
        for i in range(4):
            raw = build_demo_match("hok")
            raw["match_id"] = f"curve-demo-{i}"
            raw["played_at"] = f"2026-10-0{i + 1}"
            record, _ = match_col.save_match(session, adapter, raw)
            rules = get_rules(session, "hok", record.game_version)
            ctx = adapter.build_context(record.get_payload())
            rep = build_match_report(rules, ctx)
            analyses.append(rep)

        curve = build_curve(analyses)
        assert curve["summary"]["match_count"] == 4
        assert len(curve["points"]) == 4
        assert curve["axis"] == {"min": 0.0, "max": 1.0}


# --------------------------------------------------------------------------
# 攻略抽取
# --------------------------------------------------------------------------


class TestGuideExtraction:
    def test_extraction_finds_decidable_candidates(self, session):
        """带显式阈值的攻略句应被抽成候选。"""
        result = guide_col.ingest_guide(
            session,
            text=DEMO_GUIDE,
            game="tft",
            version="13.1",
            guide_title="测试攻略",
            accept=None,               # 预览模式
        )
        assert result.candidates, "示例攻略未抽出任何候选"
        assert any(c.trigger.all for c in result.candidates), "候选均无触发条件"

    def test_subjective_lines_are_flagged_not_ingested(self, session):
        """主观判断句必须被标记且无法入库 —— 这是规则库质量的守门人。"""
        subjective_only = "这波该不该追三要看对手站位，尽量不要梭哈。"

        preview = guide_col.ingest_guide(
            session,
            text=subjective_only,
            game="tft",
            version="13.1",
            guide_title="主观攻略",
            accept=None,
        )
        assert preview.rejected_subjective, "主观句未被识别"
        # 低置信度候选不应进入候选列表
        assert all(c.confidence >= 0.4 for c in preview.candidates)
        # 标记了主观词的候选，其置信度必须被压到不可用区间
        for cand in preview.candidates:
            if cand.subjective_markers:
                assert cand.confidence <= 0.35

    def test_preview_mode_ingests_nothing(self, session):
        """预览模式绝不能写库 —— 未经人工确认的候选不得污染模式库。"""
        # 先造一个基线版本：预览模式的「不变」要相对于一个真实存在的版本说，
        # 否则「库里本来什么都没有」和「预览什么都没写」分不开
        good = [c.title for c in guide_col.ingest_guide(
            session, text=DEMO_GUIDE, game="tft", version="13.1",
            guide_title="基线", accept=None,
        ).candidates if not c.subjective_markers]
        guide_col.ingest_guide(
            session, text=DEMO_GUIDE, game="tft", version="13.1",
            guide_title="基线", accept=good,
        )
        before = len(get_rules(session, "tft", "13.1"))

        guide_col.ingest_guide(
            session,
            text=DEMO_GUIDE,
            game="tft",
            version="13.2",
            guide_title="仅预览",
            accept=None,
        )
        assert latest_version(session, "tft") == "13.1", "预览模式不应产生新版本"
        assert len(get_rules(session, "tft", "13.1")) == before

    def test_confirmed_guide_creates_versioned_rules(self, session):
        """人工确认放行的条目才会入库，且按版本隔离。"""
        preview = guide_col.ingest_guide(
            session,
            text=DEMO_GUIDE,
            game="tft",
            version="13.1",
            guide_title="确认导入测试",
            accept=None,
        )
        good = [c.title for c in preview.candidates if not c.subjective_markers]
        assert good, "没有可确认的候选"

        result = guide_col.ingest_guide(
            session,
            text=DEMO_GUIDE,
            game="tft",
            version="13.1",
            guide_title="确认导入测试",
            accept=good,
        )
        assert result.imported, "确认后未入库任何规则"
        assert "13.1" in list_versions(session, "tft")

        # 另一个版本互不干扰
        preview2 = guide_col.ingest_guide(
            session, text=DEMO_GUIDE, game="tft", version="13.2",
            guide_title="另一版本", accept=None,
        )
        good2 = [c.title for c in preview2.candidates if not c.subjective_markers]
        guide_col.ingest_guide(
            session, text=DEMO_GUIDE, game="tft", version="13.2",
            guide_title="另一版本", accept=good2,
        )
        assert latest_version(session, "tft") == "13.2"
        assert get_rules(session, "tft", "13.1"), "13.1 不该被 13.2 的导入清空"


# --------------------------------------------------------------------------
# 报告渲染
# --------------------------------------------------------------------------


class TestReportRendering:
    @pytest.mark.parametrize("game", MODE_GAMES)
    def test_report_renders(self, session, game):
        adapter = get_adapter(game)
        record, _ = match_col.save_match(session, adapter, build_demo_match(game))
        rules = get_rules(session, game, record.game_version)
        ctx = adapter.build_context(record.get_payload())
        report = build_match_report(rules, ctx)

        html = render_report(report)
        assert html.lstrip().startswith("<!") or "<html" in html.lower()
        assert len(html) > 3000, "报告过短，可能渲染失败"

    def test_report_has_no_placeholder_artifacts(self, session):
        adapter = get_adapter("tft")
        record, _ = match_col.save_match(session, adapter, build_demo_match("tft"))
        rules = get_rules(session, "tft", record.game_version)
        ctx = adapter.build_context(record.get_payload())
        html = render_report(build_match_report(rules, ctx))

        for bad in ("{{", "}}", "undefined", "NaN", "[object Object]", "None"):
            assert bad not in html, f"报告残留 {bad!r}"

    def test_report_survives_empty_gaps(self, session):
        """没有差距时不能崩 —— 那是一个正常且常见的局面。"""
        adapter = get_adapter("tft")
        raw = build_demo_match("tft")
        raw["rounds"] = [raw["rounds"][0]]      # 只留一个回合，大概率无偏差
        record, _ = match_col.save_match(session, adapter, raw)
        rules = get_rules(session, "tft", record.game_version)
        ctx = adapter.build_context(record.get_payload())
        report = build_match_report(rules, ctx)

        assert render_report(report)
        if report["stats"]["compliance_rate"] is None:
            assert "这一局没有发现偏离操作模式的地方" in render_report(report)


#: 报告正文（实际 / 该做 / 为什么 + 时间节点）的样例。
#:
#: 单独跑一遍完整链路（灌库 -> 录对局 -> 分析 -> 出页面），
#: 断言才能落在**用户看到的东西**上，而不是内核的中间结构。
@pytest.fixture(scope="module")
def report_text_sample(tmp_path_factory):
    from fastapi.testclient import TestClient

    from app import db as dbmod
    from app.main import app

    dbmod._engine = None
    dbmod._SessionFactory = None
    dbmod.get_engine(str(tmp_path_factory.mktemp("report-text") / "r.db"))
    client = TestClient(app)
    client.post("/api/games/hok/rules/seed")
    r = client.post("/api/matches", json=build_demo_match("hok"))
    assert r.status_code == 200, r.text[:400]
    match_id = r.json()["match_id"]
    page = client.get(f"/api/matches/{match_id}/report")
    assert page.status_code == 200
    return {
        "json": client.get(f"/api/matches/{match_id}/report.json").json(),
        "html": page.text,
    }


class TestReportText:
    """报告正文只有三件事：实际 / 该做 / 为什么，外加准确的时间节点。

    这组断言盯的是**报告怎么说话**：
    - 每条偏离都能定位到具体时间；
    - 三段话都是完整句子（主谓宾齐全、以句号收尾）；
    - 页面里不再有「原文前提：」「下回怎么做」这类元叙述和旧块。
    """

    def test_every_gap_has_time_and_three_parts(self, report_text_sample):
        gaps = report_text_sample["json"]["gaps"]
        assert gaps, "示例对局应当有偏离可查"
        for g in gaps:
            assert g.get("time_label"), f"{g.get('rule_id')} 没有时间节点"
            for key in ("actual_text", "expected_text", "why_text"):
                assert g.get(key), f"{g.get('rule_id')} 缺 {key}"

    def test_three_parts_are_complete_sentences(self, report_text_sample):
        """主谓宾补全：每段话都要能独立读懂，以句号收尾。"""
        for g in report_text_sample["json"]["gaps"]:
            for key in ("actual_text", "expected_text", "why_text"):
                text = (g.get(key) or "").strip()
                assert text.endswith("。"), f"{key} 不是完整句子：{text!r}"
                assert not text.startswith("（"), f"{key} 像是占位符：{text!r}"

    def test_why_carries_this_match_numbers(self, report_text_sample):
        """「为什么」要带这一局的实际取值，否则跟任何一局都成立。

        措辞按游戏走：王者是「这一段」，自走棋是「这一回合」——
        王者的一局是连续的一整场，根本没有「回合」这个东西。
        """
        whys = [g["why_text"] for g in report_text_sample["json"]["gaps"]]
        assert any("经济差 -300" in w for w in whys), (
            f"「为什么」没有带上这一局的实际取值：{whys}"
        )
        assert any(w.startswith("这一段") or w.startswith("这一回合")
                   for w in whys), (
            f"「为什么」没有指明说的是哪一段：{whys}"
        )

    def test_no_meta_narration_in_report(self, report_text_sample):
        """页面里不该再有元叙述和旧块。"""
        html = report_text_sample["html"]
        for bad in ("原文前提", "原文结果", "规则溯源", "判定证据",
                    "当时局面", "下回怎么做", "你在做的", "怎么改"):
            assert bad not in html, f"报告页面上还有 {bad}"

    def test_page_shows_the_four_parts_and_the_time_range(self, report_text_sample):
        """报告正文是「操作模式分析」，每条四部分。

        四部分固定是：时间区间 / 原视频片段 / 原视频操作描述 /
        建议操作描述（含建议理由）。少了任何一栏，读者就缺一块信息。
        """
        html = report_text_sample["html"]
        assert "mo-row" in html, "报告页没有四部分正文"
        for label in ("对局时间区间", "原视频被建议部分片段",
                      "原视频操作的文字描述", "根据模式库建议操作的描述"):
            assert label in html, f"报告页上少了「{label}」这一部分"
        assert "建议理由" in html, "报告页没有给出建议理由"
        assert "第 " in html and "分钟" in html, "报告页没有给出对局时间区间"

    def test_empty_report_dict_does_not_crash(self):
        """最坏情况：传一个空字典。渲染层不能因缺字段炸掉。"""
        assert render_report({})

    def test_json_export(self, session):
        adapter = get_adapter("tft")
        record, _ = match_col.save_match(session, adapter, build_demo_match("tft"))
        rules = get_rules(session, "tft", record.game_version)
        ctx = adapter.build_context(record.get_payload())
        payload = render_report(
            build_match_report(rules, ctx), as_json=True
        )
        parsed = json.loads(payload)
        assert parsed["meta"]["game"] == "tft"
