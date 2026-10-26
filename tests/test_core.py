"""分析内核测试。

重点验证三件事：
  1. 规则 DSL 的判定语义正确（比较符、缺失事实、主观词拦截）
  2. 内核确实「游戏无关」—— 用一个虚构游戏跑通全流程
  3. 归因输出能定位到具体回合，且不伪造数据（缺失即标为无法判定）
"""

from __future__ import annotations

import pytest

from app.core.attribution import (
    STATUS_DEVIATED,
    STATUS_SATISFIED,
    STATUS_UNDETERMINED,
    analyze_match,
)
from app.core.context import ActionRecord, MatchContext, RoundRecord
from app.core.curve import build_curve
from app.core.dsl import (
    Condition,
    Rule,
    RuleValidationError,
    Source,
    Trigger,
    check_admission,
)
from app.core.matcher import match_rules


# --------------------------------------------------------------------------
# DSL
# --------------------------------------------------------------------------


class TestCondition:
    @pytest.mark.parametrize(
        "op,value,actual,expected",
        [
            (">=", 50, 58, True),
            (">=", 50, 40, False),
            ("==", 4, 4, True),
            ("in", ["a", "b"], "a", True),
            ("in", ["a", "b"], "c", False),
            ("between", [3, 6], 5, True),
            ("between", [3, 6], 7, False),
        ],
    )
    def test_evaluate(self, op, value, actual, expected):
        assert Condition("f", op, value).evaluate(actual) is expected

    def test_missing_fact_is_never_satisfied(self):
        """事实没采集到时必须判为不成立，绝不能当作 True。"""
        assert Condition("f", ">=", 1).evaluate(None, present=False) is False
        assert Condition("f", "exists").evaluate(None, present=False) is False

    def test_exists_operator(self):
        assert Condition("f", "exists").evaluate(0, present=True) is True

    def test_type_mismatch_does_not_raise(self):
        """阈值类型不符时返回 False 而不是抛异常 —— 一局数据的瑕疵不该毁掉整局分析。"""
        assert Condition("f", ">=", 50).evaluate("abc") is False

    def test_unknown_operator_rejected(self):
        with pytest.raises(RuleValidationError):
            Condition("f", "~=", 1).validate()

    def test_between_requires_ordered_bounds(self):
        with pytest.raises(RuleValidationError):
            Condition("f", "between", [10, 2]).validate()


class TestTrigger:
    def _cond(self, fact, op, value):
        return Condition(fact, op, value)

    def test_all_group_short_circuits(self):
        """all 组一旦不成立就应停止求值后续条件，省掉无谓的事实计算。"""
        t = Trigger(all=[self._cond("a", "==", 99), self._cond("b", "==", 2)])
        seen = []

        def resolve(f):
            seen.append(f)
            return (1, True)

        result = t.evaluate(resolve)
        assert result.fired is False
        assert seen == ["a"]              # b 未被求值
        assert len(result.evidence) == 1

    def test_any_group(self):
        t = Trigger(any=[self._cond("a", "==", 1), self._cond("b", "==", 2)])
        resolve = lambda f: (2, True) if f == "b" else (0, True)    # noqa: E731
        assert t.evaluate(resolve).fired is True

    def test_none_group_blocks(self):
        t = Trigger(none=[self._cond("a", "==", 1)])
        resolve = lambda f: (1, True)                                # noqa: E731
        assert t.evaluate(resolve).fired is False

    def test_evidence_records_threshold_and_actual(self):
        """证据必须能支撑复查：阈值、实际值、结论三者齐全。"""
        t = Trigger(all=[self._cond("health", ">=", 50)])
        result = t.evaluate(lambda f: (58, True))
        ev = result.evidence[0]
        assert ev["threshold"] == 50
        assert ev["actual"] == 58
        assert ev["result"] is True


class TestAdmission:
    """准入筛选是「规则宁可少也不能注水」的强制执行点。"""

    FACTS = {"health", "econ_action"}

    def _rule(self, **kw):
        base = dict(
            id="TFT-ECON-001",
            game="tft",
            version="13.1",
            title="测试规则",
            category="经济运营",
            trigger=Trigger(all=[Condition("health", ">=", 50)]),
            expect_fact="econ_action",
            expect_op="in",
            expect_value=["save"],
            source=Source(guide="测试攻略"),
        )
        base.update(kw)
        return Rule(**base)

    def test_valid_rule_passes(self):
        assert check_admission(self._rule(), self.FACTS) == []

    def test_unregistered_fact_rejected(self):
        """引用未注册事实 = 不可判定，必须拒收。"""
        rule = self._rule(
            trigger=Trigger(all=[Condition("winrate", ">=", 50)]),
            expect_fact="winrate",
        )
        reasons = check_admission(rule, self.FACTS)
        assert any("winrate" in r for r in reasons)

    def test_subjective_wording_rejected(self):
        rule = self._rule(deviation_hint="这波该不该追三要看情况")
        reasons = check_admission(rule, self.FACTS)
        assert any("主观判断" in r for r in reasons)

    def test_missing_source_rejected(self):
        rule = self._rule(source=None)
        reasons = check_admission(rule, self.FACTS)
        assert any("溯源" in r for r in reasons)

    def test_global_rule_rejected(self):
        """全局规则无法定位到具体回合，与产品主张冲突。"""
        rule = self._rule(trigger=Trigger())
        reasons = check_admission(rule, self.FACTS)
        assert any("触发条件" in r for r in reasons)


# --------------------------------------------------------------------------
# 内核游戏无关性
# --------------------------------------------------------------------------


class _FakeGameContext(MatchContext):
    """构造一个完全不存在的游戏上下文，验证内核不依赖任何具体游戏。"""

    def __init__(self, **kw):
        super().__init__(game="fake-game", game_version="1.0", **kw)


def _round(stage, phase, facts, actions):
    return RoundRecord(
        stage=stage,
        phase=phase,
        facts=facts,
        actions=[
            ActionRecord(kind=k, value=v, label=l, confidence=c)
            for k, v, l, c in actions
        ],
    )


@pytest.fixture
def fake_ctx():
    """覆盖四条判定路径：满足、违背、该做而没做、不触发。"""
    ctx = _FakeGameContext(match_id="fake-1")
    ctx.rounds = [
        _round(1, 1, {"mood": 8}, [("action", "wait", "等待", 1.0)]),    # 满足
        _round(2, 1, {"mood": 8}, [("action", "rush", "激进", 1.0)]),    # 违背
        _round(3, 1, {"mood": 9}, []),                                    # 该做而没做
        _round(4, 1, {"mood": 2}, [("action", "wait", "等待", 0.6)]),    # 不触发
    ]
    return ctx


@pytest.fixture
def fake_rule():
    return Rule(
        id="FAKE-001",
        game="fake-game",
        version="1.0",
        title="mood 高于 5 时应该等待",
        category="节奏",
        trigger=Trigger(all=[Condition("mood", ">", 5)]),
        expect_fact="action",
        expect_op="==",
        expect_value="wait",
        severity="high",
        source=Source(guide="虚构攻略"),
        deviation_hint="这一回合应当等待",
    )


class TestEngine:
    def test_deviation_is_located_to_round(self, fake_ctx, fake_rule):
        """核心产品主张：结论必须指向具体回合。"""
        report = analyze_match([fake_rule], fake_ctx)
        labels = [g.round_label for g in report["gaps"]]
        assert labels == ["2-1", "3-1"]

    def test_absent_action_counts_as_deviation(self, fake_ctx, fake_rule):
        """「该做而没做」也必须被算成偏离，而不是被静默跳过。"""
        report = analyze_match([fake_rule], fake_ctx)
        gap = next(g for g in report["gaps"] if g.round_label == "3-1")
        assert gap.actual["values"] == []

    def test_compliance_rate_counts_only_decided(self, fake_ctx, fake_rule):
        report = analyze_match([fake_rule], fake_ctx)
        stats = report["stats"]
        # mood>5 的三个回合被触发：1 满足、2 偏离；4-1 不触发故不计入分母
        assert stats.triggered == 3
        assert stats.satisfied == 1
        assert stats.deviated == 2
        assert stats.compliance_rate == pytest.approx(1 / 3)
        assert stats.coverage == pytest.approx(1.0)     # 唯一一条规则被检验过

    def test_missing_facts_yield_undetermined_not_guess(self):
        """关键事实缺失时必须标为无法判定，不能猜。"""
        ctx = _FakeGameContext(match_id="fake-2")
        ctx.rounds = [
            RoundRecord(
                stage=1, phase=1,
                facts={},
                actions=[],
                missing_facts={"mood"},
            )
        ]
        rule = Rule(
            id="FAKE-002", game="fake-game", version="1.0",
            title="需要 mood 的规则", category="节奏",
            trigger=Trigger(all=[Condition("mood", ">", 5)]),
            expect_fact="action", expect_op="==", expect_value="wait",
            source=Source(guide="虚构攻略"),
        )
        report = analyze_match([rule], ctx)
        assert report["stats"].triggered == 0
        assert report["stats"].compliance_rate is None   # 绝不编造数字

    def test_no_trigger_returns_none_rate(self, fake_ctx):
        """没有任何规则触发时，执行度必须是 None 而不是 0。"""
        rule = Rule(
            id="FAKE-003", game="fake-game", version="1.0",
            title="永不触发", category="节奏",
            trigger=Trigger(all=[Condition("mood", ">", 9999)]),
            expect_fact="action", expect_op="==", expect_value="wait",
            source=Source(guide="虚构攻略"),
        )
        assert analyze_match([rule], fake_ctx)["stats"].compliance_rate is None

    def test_low_confidence_is_flagged(self, fake_ctx, fake_rule):
        report = analyze_match([fake_rule], fake_ctx)
        # 4-1 mood=2 不触发；要触发得构造一个高 mood 低置信度的回合
        ctx = _FakeGameContext(match_id="fake-3")
        ctx.rounds = [_round(1, 1, {"mood": 9}, [("action", "rush", "激进", 0.55)])]
        rep = analyze_match([fake_rule], ctx)
        assert rep["gaps"][0].low_confidence is True

    def test_only_selected_rules(self, fake_ctx, fake_rule):
        report = analyze_match([fake_rule], fake_ctx, only_rules=["FAKE-001"])
        assert report["stats"].triggered == 3

        empty = analyze_match([fake_rule], fake_ctx, only_rules=["NOPE"])
        assert empty["stats"].triggered == 0


# --------------------------------------------------------------------------
# 成长曲线
# --------------------------------------------------------------------------


class TestCurve:
    def test_null_rates_preserved_as_gaps(self):
        """没有触发规则的对局在曲线上留空，不插值 —— 空洞本身就是信息。"""
        analyses = [
            {"match_id": "m1", "compliance_rate": 0.6, "verdicts": []},
            {"match_id": "m2", "compliance_rate": None, "verdicts": []},
            {"match_id": "m3", "compliance_rate": 0.8, "verdicts": []},
        ]
        curve = build_curve(analyses)
        rates = [p["compliance_rate"] for p in curve["points"]]
        assert rates == [0.6, None, 0.8]
        assert curve["summary"]["measured_count"] == 2

    def test_regression_detection(self):
        """玩家最容易自我欺骗的地方：段位涨了但老套路没改。"""
        analyses = [
            {"match_id": "m1", "compliance_rate": 1.0,
             "verdicts": [_v("R1", STATUS_SATISFIED)]},
            {"match_id": "m2", "compliance_rate": 0.0,
             "verdicts": [_v("R1", STATUS_DEVIATED)]},
        ]
        curve = build_curve(analyses)
        rule = curve["rules"][0]
        assert rule["trend"] == "退步了"
        assert curve["regressing_rules"]

    def test_improvement_detection(self):
        analyses = [
            {"match_id": "m1", "compliance_rate": 0.0,
             "verdicts": [_v("R1", STATUS_DEVIATED)]},
            {"match_id": "m2", "compliance_rate": 1.0,
             "verdicts": [_v("R1", STATUS_SATISFIED)]},
        ]
        assert build_curve(analyses)["rules"][0]["trend"] == "改进了"


def _v(rule_id, status):
    """构造最小 Verdict 形态。"""
    from app.core.matcher import Verdict

    return Verdict(
        rule_id=rule_id,
        rule_title=f"规则 {rule_id}",
        category="测试",
        severity="medium",
        status=status,
    )
