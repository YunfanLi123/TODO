"""版本适配与回测的测试。

回测是这个系统里最容易被做坏的部分 —— 它跨版本、跨数据来源，
一旦口径错了不会报错，只会安静地给出错误结论。所以这里重点断言
**统计口径**和**降级行为**，而不只是「函数返回了列表」。
"""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from app.core.context import ActionRecord, MatchContext, RoundRecord
from app.core.curve import (
    backtest_version_change,
    build_curve,
    diff_versions,
)
from app.core.dsl import Condition, Rule, Source, Trigger


# --------------------------------------------------------------------------
# 夹具
# --------------------------------------------------------------------------


def make_rule(rule_id="R1", gold=50, health=100, version="v1", **kw):
    base = dict(
        id=rule_id,
        game="g",
        version=version,
        title="金币足够时应存钱",
        category="经济",
        trigger=Trigger(
            all=[
                Condition("gold", ">=", gold),
                Condition("health", "<", health),
            ]
        ),
        expect_fact="econ_action",
        expect_op="==",
        expect_value="save",
        source=Source(guide="测试攻略"),
    )
    base.update(kw)
    return Rule(**base)


def make_ctx(match_id, rounds_spec):
    """rounds_spec: [(stage, phase, gold, health, action_value)]"""
    rounds = []
    for stage, phase, gold, hp, act in rounds_spec:
        actions = (
            [ActionRecord(kind="econ_action", value=act, label="")]
            if act
            else []
        )
        rounds.append(
            RoundRecord(
                stage=stage,
                phase=phase,
                facts={"gold": gold, "health": hp},
                actions=actions,
            )
        )
    return MatchContext(match_id=match_id, game="g", game_version="v1", rounds=rounds)


# --------------------------------------------------------------------------
# 序列化往返
# --------------------------------------------------------------------------


class TestContextRoundTrip:
    """版本回测依赖「从 payload 重建上下文」，往返必须无损。"""

    def test_round_trip_preserves_everything(self):
        rnd = RoundRecord(
            stage=3,
            phase=2,
            facts={"gold": 40, "health": 70},
            actions=[
                ActionRecord(kind="econ_action", value="save", label="存钱", ts=1.5,
                             confidence=0.8)
            ],
            missing_facts={"item_count"},
            label_override="第 12 分钟",
            time_index=12.0,
        )
        back = RoundRecord.from_dict(rnd.to_dict())
        assert back.facts == rnd.facts
        assert back.missing_facts == rnd.missing_facts
        assert back.label_override == "第 12 分钟"
        assert back.time_index == 12.0
        assert back.actions[0].value == "save"
        assert back.actions[0].confidence == 0.8

    def test_round_trip_preserves_label_semantics(self):
        """可读标签必须活着穿过序列化 —— 否则重建后 MOBA 报告退回「2-2」。"""
        rnd = RoundRecord(stage=2, phase=2, label_override="第 12 分钟")
        assert RoundRecord.from_dict(rnd.to_dict()).label == "第 12 分钟"

    def test_round_trip_preserves_time_ordering(self):
        rnd = RoundRecord(stage=2, phase=2, time_index=17.5)
        assert RoundRecord.from_dict(rnd.to_dict()).time_index == 17.5

    def test_legacy_payload_without_new_fields(self):
        """早期写入的 payload 没有新字段，必须能读出来而不是整局失败。"""
        legacy = {
            "label": "2-2",
            "stage": 2,
            "phase": 2,
            "facts": {"gold": 10},
            "actions": [],
            "missing_facts": [],
        }
        rnd = RoundRecord.from_dict(legacy)
        assert rnd.label == "2-2"          # 退回默认坐标
        assert rnd.time_index == 0.0
        assert rnd.label_override == ""

    def test_context_round_trip(self):
        ctx = make_ctx("m1", [(1, 1, 10, 90, "save"), (2, 1, 60, 80, "roll")])
        back = MatchContext.from_dict(ctx.to_dict())
        assert back.match_id == "m1"
        assert len(back.rounds) == 2
        assert back.rounds[1].facts["gold"] == 60


# --------------------------------------------------------------------------
# 版本差异
# --------------------------------------------------------------------------


class TestDiff:
    def test_version_alone_is_not_a_change(self):
        """最关键的一条：同一规则换个版本号不算「变更」。

        否则规则库里每条都会显示已变更，真正的阈值调整被噪声淹没。
        """
        old = make_rule(version="v1")
        new = make_rule(version="v2")
        diff = diff_versions([old], [new])
        assert diff["changed"] == []
        assert diff["added"] == [] and diff["removed"] == []

    def test_threshold_change_is_detected(self):
        old = make_rule(version="v1", gold=50)
        new = make_rule(version="v2", gold=60)
        diff = diff_versions([old], [new])
        assert len(diff["changed"]) == 1
        assert "触发条件变更" in diff["changed"][0]["changes"]

    def test_added_and_removed(self):
        old = [make_rule("R1", version="v1")]
        new = [make_rule("R1", version="v2"), make_rule("R2", version="v2")]
        diff = diff_versions(old, new)
        assert [a["rule_id"] for a in diff["added"]] == ["R2"]
        assert diff["removed"] == []
        assert diff["changed"] == []


# --------------------------------------------------------------------------
# 回测统计口径
# --------------------------------------------------------------------------


class TestBacktest:
    def test_counts_every_triggering_round_not_just_first(self):
        """一局里反复犯同一个错，应记多次 —— 这是衡量「习惯有多顽固」的关键。"""
        rule_old = make_rule(version="v1", gold=50)
        # 三个回合都满足触发条件，玩家每次都违规
        ctx = make_ctx(
            "m1",
            [
                (1, 1, 60, 90, "roll"),
                (2, 1, 70, 90, "roll"),
                (3, 1, 80, 90, "roll"),
            ],
        )
        rows = backtest_version_change([rule_old], [make_rule(version="v2", gold=50)], [ctx])
        # 新旧一致 → 不进回测，用不同阈值强制进入
        rows = backtest_version_change(
            [rule_old], [make_rule(version="v2", gold=50, title="改标题")], [ctx]
        )
        row = next(r for r in rows if r["rule_id"] == "R1")
        assert row["old_checked"] == 3
        assert row["old_deviated"] == 3
        assert row["old_rate"] == 1.0

    def test_unchanged_rules_are_not_backtested(self):
        old = make_rule(version="v1")
        new = make_rule(version="v2")
        ctx = make_ctx("m1", [(1, 1, 60, 90, "roll")])
        assert backtest_version_change([old], [new], [ctx]) == []

    def test_relaxed_threshold_narrows_coverage(self):
        old = make_rule(version="v1", gold=50)
        new = make_rule(version="v2", gold=100)
        ctx = make_ctx("m1", [(1, 1, 60, 90, "roll")])   # 60 < 100，新版不触发
        rows = backtest_version_change(
            [old], [new], [ctx, make_ctx("m2", [(1, 1, 60, 90, "roll")])]
        )
        row = next(r for r in rows if r["rule_id"] == "R1")
        assert row["old_checked"] == 2
        assert row["new_checked"] == 0
        assert row["verdict"] == "可以放松"
        assert "彻底不适用" in row["message"]

    def test_newly_binding_rule(self):
        """旧版从未触发、新版开始管你 —— 是「新增约束」而非「无法对比」。"""
        old = make_rule(version="v1", gold=500)
        new = make_rule(version="v2", gold=50)
        ctx = make_ctx("m1", [(1, 1, 60, 90, "roll")])
        rows = backtest_version_change([old], [new], [ctx])
        row = next(r for r in rows if r["rule_id"] == "R1")
        assert row["old_checked"] == 0
        assert row["new_checked"] == 1
        assert row["verdict"] == "新增约束"
        assert "新增生效" in row["message"]

    def test_stricter_rule_detected(self):
        """收紧触发门槛后，玩家在新版覆盖范围内偏离更多才算「更严格」。

        注意：trigger 变严本身不等于约束变严 —— 如果玩家在新覆盖的那局
        恰好做对了，偏离率反而会下降。所以数据必须构造成「被新规则管到的
        局次里玩家违规更多」。
        """
        old = make_rule(version="v1", gold=50)
        new = make_rule(version="v2", gold=55)
        # gold 52：旧版管、新版不管，且这局玩家做对了（save）
        # gold 58 / 60：两版都管，玩家都违规（roll）
        ctx = make_ctx(
            "m1",
            [
                (1, 1, 52, 90, "save"),
                (2, 1, 58, 90, "roll"),
                (3, 1, 60, 90, "roll"),
            ],
        )
        rows = backtest_version_change([old], [new], [ctx])
        row = next(r for r in rows if r["rule_id"] == "R1")
        assert row["old_checked"] == 3
        assert row["new_checked"] == 2
        assert row["old_rate"] == pytest.approx(2 / 3, abs=1e-3)
        assert row["new_rate"] == pytest.approx(1.0, abs=1e-3)
        assert row["verdict"] == "需要更严格"

    def test_deviation_examples_are_recorded(self):
        old = make_rule(version="v1", gold=50)
        new = make_rule(version="v2", gold=100)
        ctx = make_ctx("m1", [(1, 3, 60, 90, "roll")])
        rows = backtest_version_change([old], [new], [ctx])
        row = next(r for r in rows if r["rule_id"] == "R1")
        ex = row["old_examples"][0]
        assert ex["match_id"] == "m1"
        assert ex["round_label"] == "1-3"
        assert ex["expected"] == "save"
        assert ex["actual"] == "roll"

    def test_rates_never_invented_without_data(self):
        old = make_rule(version="v1", gold=50)
        new = make_rule(version="v2", gold=60)
        rows = backtest_version_change([old], [new], [])
        row = next(r for r in rows if r["rule_id"] == "R1")
        assert row["old_rate"] is None
        assert row["new_rate"] is None
        assert row["rate_delta"] is None
        assert row["verdict"] == "数据不足"
        assert "样本不足" in row["message"]

    def test_rates_within_bounds(self):
        old = make_rule(version="v1", gold=50)
        new = make_rule(version="v2", gold=55)
        ctx = make_ctx(
            "m0",
            [
                (1, 1, 50 + i, 90, "roll" if i % 2 else "save")
                for i in range(6)
            ],
        )
        row = next(
            r for r in backtest_version_change([old], [new], [ctx]) if r["rule_id"] == "R1"
        )
        for key in ("old_rate", "new_rate"):
            assert 0.0 <= row[key] <= 1.0
        assert row["old_deviated"] <= row["old_checked"]
        assert row["new_deviated"] <= row["new_checked"]

    def test_impact_sorted_first(self):
        old = make_rule("A", version="v1", gold=50)
        new_a = make_rule("A", version="v2", gold=55)      # 偏离率上升
        old_b = make_rule("B", version="v1", gold=50)
        new_b = make_rule("B", version="v2", gold=200)     # 放宽
        ctx = make_ctx("m1", [(1, 1, 60, 90, "roll")] * 3)
        rows = backtest_version_change([old, old_b], [new_a, new_b], [ctx])
        assert rows[0]["rule_id"] == "A"


# --------------------------------------------------------------------------
# 成长曲线的类型兼容
# --------------------------------------------------------------------------


class TestCurveTolerance:
    def test_accepts_verdict_objects_and_dicts(self):
        """曲线要同时吃实时 Verdict 对象和数据库重建的 dict。

        每条规则需出现两局以上 —— 曲线刻意过滤掉单点样本，
        否则用一局的胜负就说「改进了」，那正是这个项目要消灭的误导。
        """
        from app.core.matcher import Verdict

        obj = Verdict(
            rule_id="R1", rule_title="标题", category="经济",
            severity="medium", status="deviated",
        )
        as_dict = {
            "rule_id": "R2", "rule_title": "另一个", "category": "节奏",
            "status": "satisfied",
        }
        curve = build_curve(
            [
                {"match_id": "m1", "compliance_rate": 0.5, "verdicts": [obj, as_dict]},
                {"match_id": "m2", "compliance_rate": 1.0, "verdicts": [obj, as_dict]},
            ]
        )
        assert {r["rule_id"] for r in curve["rules"]} == {"R1", "R2"}

    def test_nested_stats_shape(self):
        """analyze_match 的输出把执行度放在 stats 里，也必须能读。"""
        curve = build_curve(
            [{"match_id": "m1", "stats": {"compliance_rate": 0.8, "triggered": 5}}]
        )
        assert curve["points"][0]["compliance_rate"] == 0.8
