"""规则库审核的测试。

规则库是整个系统的资产 —— 报告里每一条结论都建立在它之上。这个模块
守的是两条底线：

  1. **人工改规则不等于绕过准入。** 走 ``import_rules`` 拦得住的非法规则，
     从审核接口改也必须被拦住。两边标准不一致就等于开了后门。
  2. **修改必须留痕。** 人工改过的东西和机器抽出来的可信度不同，
     而抽取器的方向性错误只有把原件和改后版本摆在一起才看得见。
"""

from __future__ import annotations

import json
import os

import pytest

from app.adapters import registry as _registry  # noqa: F401  触发适配器注册
from app.core.dsl import Rule
from app.rules.review import (
    EDITABLE_FIELDS,
    ReviewError,
    RuleCoverage,
    build_review_view,
    compute_coverage,
    coverage_summary,
    revision_history,
    set_status,
    update_rule,
)


def _seed_rule_count() -> int:
    """种子规则文件里的规则总数（分路层 + 英雄层）。"""
    seed_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "app", "seed",
    )
    total = 0
    for name in ("hok_modes.json", "hok_hero_modes.json"):
        path = os.path.join(seed_dir, name)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                total += len(json.load(fh))
    return total


@pytest.fixture
def session(tmp_path):
    from app import db as dbmod
    from app.db import get_engine, session_scope
    from app.rules.store import latest_version, load_seed_library

    dbmod._engine = None
    dbmod._SessionFactory = None
    get_engine(str(tmp_path / "review_test.db"))
    with session_scope() as s:
        if not latest_version(s, "hok"):
            load_seed_library(s, "hok")
    with session_scope() as s:
        nested = s.begin_nested()
        try:
            yield s
        finally:
            if nested.is_active:
                nested.rollback()


def _seed_match(session, match_id="rv-demo", action="take_dragon", **round_overrides):
    """造一局打野对局。

    动作默认选成「该做而没做」：前期该反野 / 守自家野区却去抢龙，
    这样覆盖率与偏离统计才有非零样本可查。

    动作取值是照着 S45 新库挑的，不能照抄旧库：

    - ``HOK-MODE-JGL-T0200-01``（两分钟：均势去反野，劣势守自家野区）
      期望「反野 / 刷野」，而它的**适用时机**只覆盖「刷野、反野、抢龙、拿龙」
      这一类目标动作。所以「回城」在新库里**根本不会被判**（适用时机对不上），
      拿它当样本会得到一条覆盖率为零的规则 —— 看着像「这条规则从没生效过」。
    - 第 7 分钟落在 ``HOK-MODE-JGL-T0600-02``（人齐不先开团，用龙和塔逼对方动），
      它要 ``kill_diff <= 0`` 才成立，所以这一回合的击杀差取 0。
    """
    from app.adapters.base import get_adapter
    from app.collection import match as match_col

    raw = {
        "match_id": match_id,
        "game": "hok",
        "game_version": "S45",
        # position 必须有 —— 操作模式全部按分路生效，缺了它一条都不触发
        "player": {"nickname": "示例玩家", "position": "打野"},
        "outcome": {"result": "lose"},
        "played_at": "2026-10-14 21:00",
        "rounds": [
            # stage 是必填的：没有它 save_match 会拒绝入库，标签也会退化成 ?-1
            {"stage": 2, "phase": 1, "time_min": 2,
             "facts": {"gold": 1800, "gold_diff": 200, "level": 4,
                       "tower_diff": 0, "kill_diff": 0},
             "actions": [{"kind": "objective_action", "value": action,
                          "label": "开小龙团", "confidence": 1.0}],
             "missing_facts": []},
            {"stage": 7, "phase": 2, "time_min": 7,
             "facts": {"gold": 5000, "gold_diff": 300, "level": 8,
                       "tower_diff": 0, "kill_diff": 0},
             "actions": [{"kind": "objective_action", "value": action,
                          "label": "开小龙团", "confidence": 1.0}],
             "missing_facts": []},
        ],
    }
    for rd in raw["rounds"]:
        rd["facts"].update(round_overrides)
    return match_col.save_match(session, get_adapter("hok"), raw)


# --------------------------------------------------------------------------
# 覆盖率
# --------------------------------------------------------------------------


class TestCoverage:
    def test_no_matches_all_never(self, session):
        cov = compute_coverage(session, "hok", "S45")
        assert cov
        assert all(c.checked == 0 for c in cov.values())

    def test_no_matches_no_fabricated_rates(self, session):
        """没有样本时不能编造偏离率。"""
        cov = compute_coverage(session, "hok", "S45")
        assert all(c.rate is None for c in cov.values())
        assert all(c.verdict == "never" for c in cov.values())

    def test_computed_from_matches(self, session):
        _seed_match(session)
        cov = compute_coverage(session, "hok", "S45")
        hit = cov["HOK-MODE-JGL-T0200-01"]
        assert hit.checked >= 1
        assert hit.deviated >= 1
        assert hit.matches == 1

    def test_counts_all_verdicts_not_only_deviations(self, session):
        """模式被触发但玩家做对了的情况也要计入 —— 快照里没存这个。"""
        _seed_match(session, "m1", action="take_dragon")  # 前期该反野却抢龙
        _seed_match(session, "m2", action="farm_jungle")   # 前期刷野，做对了
        cov = compute_coverage(session, "hok", "S45")
        hit = cov["HOK-MODE-JGL-T0200-01"]
        # 只记偏离会把「做对了」的那些悄悄丢掉，覆盖率于是失真
        assert hit.checked >= 2
        assert hit.deviated >= 1

    def test_verdict_states(self):
        assert RuleCoverage("r", 0, 0).verdict == "never"
        assert RuleCoverage("r", 1, 0).verdict == "thin"
        assert RuleCoverage("r", 5, 0).verdict == "always_ok"
        assert RuleCoverage("r", 5, 2).verdict == "active"

    def test_summary_lists_never_rules(self, session):
        cov = {
            "A": RuleCoverage("A", 0),
            "B": RuleCoverage("B", 3, 1),
        }
        s = coverage_summary(cov)
        assert s["never_count"] == 1
        assert [x["rule_id"] for x in s["never_triggered"]] == ["A"]

    def test_version_filter(self, session):
        """只统计对应版本的对局，跨版本混算没有意义。"""
        _seed_match(session, match_id="v131")
        cov = compute_coverage(session, "hok", "S45")
        assert all(c.matches == 1 for c in cov.values())


# --------------------------------------------------------------------------
# 修改规则
# --------------------------------------------------------------------------


class TestUpdateRule:
    def test_threshold_change(self, session):
        res = update_rule(
            session, "hok", "S45", "HOK-MODE-JGL-T0200-01",
            {"trigger": {"all": [
                {"fact": "gold_diff", "op": ">=", "value": 500},
                {"fact": "level", "op": ">=", "value": 5},
            ], "any": [], "none": []}},
            reviewer="示例玩家", note="经济差 500 以上就算顺风",
        )
        assert "trigger" in res.changed_fields

    def test_original_preserved(self, session):
        update_rule(
            session, "hok", "S45", "HOK-MODE-JGL-T0200-01",
            {"trigger": {"all": [
                {"fact": "gold_diff", "op": ">=", "value": 900},
            ], "any": [], "none": []}},
            reviewer="示例玩家",
        )
        view = build_review_view(session, "hok", "S45", with_coverage=False)
        rule = next(r for r in view["rules"] if r["rule_id"] == "HOK-MODE-JGL-T0200-01")
        assert rule["edited"] is True
        assert rule["original_payload"] is not None

        # 改前的原件必须还在，且是**改之前**的那一版 —— 这正是要复查的东西。
        # 原件里的时间窗是 S45 决策树那个节点的窗口（两分钟这一格是 2-3 分钟），
        # 不是旧库那种「前期 0-4」整段窗口。
        orig = next(
            c for c in rule["original_payload"]["trigger"]["all"]
            if c["fact"] == "time_min"
        )
        assert orig["value"] == [2, 3], "原件被覆盖了，改前的触发条件找不回来"
        cur = next(
            c for c in rule["payload"]["trigger"]["all"] if c["fact"] == "gold_diff"
        )
        assert cur["value"] == 900                # 人工改后

    def test_history_recorded(self, session):
        for value in (5, 6, 7):
            update_rule(
                session, "hok", "S45", "HOK-MODE-JGL-T0200-01",
                {"trigger": {"all": [
                    {"fact": "gold_diff", "op": ">=", "value": value},
                ], "any": [], "none": []}},
                reviewer="示例玩家",
            )
        hist = revision_history(session, "hok", "S45", "HOK-MODE-JGL-T0200-01")
        assert len(hist) == 3
        assert hist[0]["after"]["all"][0]["value"] == 5
        assert hist[-1]["after"]["all"][0]["value"] == 7

    def test_edited_fields_listed(self, session):
        update_rule(session, "hok", "S45", "HOK-MODE-JGL-T0200-01",
                    {"severity": "low"}, reviewer="示例玩家")
        view = build_review_view(session, "hok", "S45", with_coverage=False)
        rule = next(r for r in view["rules"] if r["rule_id"] == "HOK-MODE-JGL-T0200-01")
        assert "severity" in rule["edited_fields"]

    def test_no_op_change_writes_nothing(self, session):
        """把值改成它**已经是的**那个 —— 不该有任何写入。"""
        view = build_review_view(session, "hok", "S45", with_coverage=False)
        rule = next(r for r in view["rules"]
                    if r["rule_id"] == "HOK-MODE-JGL-T0200-01")
        res = update_rule(session, "hok", "S45", "HOK-MODE-JGL-T0200-01",
                          {"severity": rule["severity"]}, reviewer="示例玩家")
        assert res.changed_fields == []
        assert revision_history(session, "hok", "S45", "HOK-MODE-JGL-T0200-01") == []


class TestUpdateRejections:
    """人工改规则不等于绕过准入 —— 非法改动必须被拦住。"""

    def test_unregistered_fact(self, session):
        with pytest.raises(ReviewError):
            update_rule(session, "hok", "S45", "HOK-MODE-JGL-T0200-01",
                        {"trigger": {"all": [
                            {"fact": "根本不存在的事实", "op": ">=", "value": 1},
                        ], "any": [], "none": []}})

    def test_expect_must_be_action_fact(self, session):
        """把期望动作挂在观测事实上 —— 这条规则会恒判为偏离。"""
        with pytest.raises(ReviewError) as exc:
            update_rule(session, "hok", "S45", "HOK-MODE-JGL-T0200-01",
                        {"expect": {"fact": "pressure", "op": "==", "value": "high"}})
        assert "动作" in str(exc.value)

    def test_readonly_fields_rejected(self, session):
        for field in ("game", "version", "id"):
            with pytest.raises(ReviewError) as exc:
                update_rule(session, "hok", "S45", "HOK-MODE-JGL-T0200-01",
                            {field: "whatever"})
            assert "不可修改" in str(exc.value)

    def test_missing_rule(self, session):
        with pytest.raises(ReviewError):
            update_rule(session, "hok", "S45", "NO-SUCH-RULE",
                        {"severity": "low"})

    def test_rejected_edit_not_persisted(self, session):
        """被拒的修改绝不能部分写入 —— 校验在副本上做，失败就不碰库。"""
        before = len(revision_history(session, "hok", "S45", "HOK-MODE-JGL-T0200-01"))
        with pytest.raises(ReviewError):
            update_rule(session, "hok", "S45", "HOK-MODE-JGL-T0200-01",
                        {"trigger": {"all": [
                            {"fact": "不存在", "op": ">=", "value": 1},
                        ], "any": [], "none": []},
                         "severity": "low"})
        after = len(revision_history(session, "hok", "S45", "HOK-MODE-JGL-T0200-01"))
        assert after == before


class TestStatus:
    def test_deprecate_and_restore(self, session):
        set_status(session, "hok", "S45", "HOK-MODE-MID-T0600-01", "deprecated")
        view = build_review_view(session, "hok", "S45", with_coverage=False)
        rule = next(r for r in view["rules"] if r["rule_id"] == "HOK-MODE-MID-T0600-01")
        assert rule["status"] == "deprecated"

        set_status(session, "hok", "S45", "HOK-MODE-MID-T0600-01", "active")
        view = build_review_view(session, "hok", "S45", with_coverage=False)
        rule = next(r for r in view["rules"] if r["rule_id"] == "HOK-MODE-MID-T0600-01")
        assert rule["status"] == "active"

    def test_deprecate_does_not_count_as_content_edit(self, session):
        """停用是生命周期操作，不是内容修改。"""
        set_status(session, "hok", "S45", "HOK-MODE-MID-T0600-01", "deprecated")
        view = build_review_view(session, "hok", "S45", with_coverage=False)
        rule = next(r for r in view["rules"] if r["rule_id"] == "HOK-MODE-MID-T0600-01")
        assert rule["status"] == "deprecated"
        assert rule["edited"] is False
        assert view["summary"]["edited"] == 0

    def test_invalid_status(self, session):
        with pytest.raises(ReviewError):
            set_status(session, "hok", "S45", "HOK-MODE-MID-T0600-01", "deleted")

    def test_deprecation_not_deletion(self, session):
        """停用不删规则 —— 历史对局复盘仍需当时的规则状态。"""
        from app.rules.store import get_rules

        set_status(session, "hok", "S45", "HOK-MODE-MID-T0600-01", "deprecated")
        still_there = get_rules(
            session, "hok", "S45", include_deprecated=True
        )
        assert any(r.id == "HOK-MODE-MID-T0600-01" for r in still_there)
        active_only = get_rules(session, "hok", "S45")
        assert not any(r.id == "HOK-MODE-MID-T0600-01" for r in active_only)


# --------------------------------------------------------------------------
# 审核视图
# --------------------------------------------------------------------------


class TestReviewView:
    def test_structure(self, session):
        view = build_review_view(session, "hok", "S45", with_coverage=False)
        assert view["game"] == "hok"
        assert view["version"] == "S45"
        # 现行库 = 分路层 84 条（分路通则）+ 英雄层 1596 条（123 个英雄
        # 的专属打法，从 5 份英雄级攻略生成）。条数从种子文件数出来，
        # 不写死 —— 写死的数字每次改攻略都要跟着改，改漏了就变成一句
        # 「明明加了一堆模式，测试却说你只有 85 条」。
        assert len(view["rules"]) == _seed_rule_count()
        assert len(view["rules"]) > 1000, "英雄层没进来"
        assert view["fact_labels"]
        assert view["action_facts"]

    def test_summary_counts(self, session):
        set_status(session, "hok", "S45", "HOK-MODE-MID-T0600-01", "deprecated")
        update_rule(session, "hok", "S45", "HOK-MODE-JGL-T0200-01",
                    {"severity": "low"}, reviewer="示例玩家")
        view = build_review_view(session, "hok", "S45", with_coverage=False)
        s = view["summary"]
        total = _seed_rule_count()
        assert s["total"] == total
        assert s["active"] == total - 1
        assert s["deprecated"] == 1
        assert s["edited"] == 1

    def test_every_rule_carries_provenance(self, session):
        view = build_review_view(session, "hok", "S45", with_coverage=False)
        for r in view["rules"]:
            assert r["payload"]["source"] is not None, f"{r['rule_id']} 缺溯源"

    def test_empty_version(self, session):
        view = build_review_view(session, "hok", "99.9", with_coverage=False)
        assert view["rules"] == []
        assert view["summary"]["total"] == 0
