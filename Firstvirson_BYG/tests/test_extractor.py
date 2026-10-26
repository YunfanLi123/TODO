"""攻略抽取器的测试。

这一层出过几个「不报错、但结论彻底相反」的 bug，所以测试重点是
**方向和归属**：阈值到底修饰哪个事实、是大于还是小于。

最典型的一次是「血量不低于 50」被抽成 ``health < 50`` —— 因为
「低于」是「不低于」的子串，先匹配上了。规则方向一反，报告就会
言之凿凿地给出完全相反的建议，而表面上一切正常。
"""

from __future__ import annotations

import pytest

from app.adapters import registry as _registry  # noqa: F401  触发适配器注册
from app.rules.extractor import (
    _extract_numeric_condition,
    _extract_stage,
    _find_comparison,
    _match_action,
    extract_candidates,
    split_sentences,
)

ADAPTER = _registry.get_adapter("tft")
FACTS = ADAPTER.known_facts()


# --------------------------------------------------------------------------
# 比较词方向（回归重点）
# --------------------------------------------------------------------------


class TestComparisonDirection:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("不低于 50", (">=", 50)),
            ("不少于 50", (">=", 50)),
            ("至少 50", (">=", 50)),
            ("不超过 30", ("<=", 30)),
            ("不多于 30", ("<=", 30)),
            ("最多 30", ("<=", 30)),
            ("超过 50", (">", 50)),
            ("高于 50", (">", 50)),
            ("低于 50", ("<", 50)),
            ("少于 50", ("<", 50)),
            ("等于 50", ("==", 50)),
        ],
    )
    def test_direction(self, text, expected):
        assert _find_comparison(text) == expected

    def test_negation_beats_substring(self):
        """「不低于」必须先于「低于」匹配 —— 方向反了规则就全错。"""
        assert _find_comparison("不低于 50")[0] == ">="
        assert _find_comparison("不超过 30")[0] == "<="

    def test_integer_values_stay_integral(self):
        op, value = _find_comparison("达到 3 以上")
        assert op == ">="
        assert value == 3
        assert isinstance(value, int)

    def test_no_comparison(self):
        assert _find_comparison("应该停手存钱") is None


# --------------------------------------------------------------------------
# 阈值归属：紧邻而非全句搜索
# --------------------------------------------------------------------------


class TestNumericAttribution:
    def test_health_from_health_clause(self):
        conds = _extract_numeric_condition(
            "4-1 回合连胜达到 3 以上并且血量不低于 50 时，应该停手存钱", FACTS
        )
        by_fact = {c.fact: (c.op, c.value) for c, _ in conds}
        assert by_fact["health"] == (">=", 50)
        assert by_fact["streak"] == (">=", 3)

    def test_action_words_are_not_facts(self):
        """「存钱」是动作不是观测量，不能当金币别名吸走别的阈值。"""
        conds = _extract_numeric_condition(
            "2 阶段金币超过 50 就不要乱花钱，存钱吃利息", FACTS
        )
        facts_hit = {c.fact for c, _ in conds}
        assert "gold" in facts_hit
        # 不会凭空多出 income / level 之类
        assert facts_hit == {"gold"}

    def test_threshold_must_be_adjacent(self):
        """阈值不与事实相邻就不该配对 —— 宁可漏，不要错。"""
        conds = _extract_numeric_condition(
            "4-1 回合连胜达到 3 以上并且血量不低于 50 时，应该停手存钱", FACTS
        )
        gold = [c for c, _ in conds if c.fact == "gold"]
        assert not gold, "「连胜达到 3」不该被算成金币阈值"

    def test_stage_number_not_treated_as_threshold(self):
        """"2 阶段金币超过 50" 里的 2 是阶段号，不是金币阈值。"""
        conds = _extract_numeric_condition(
            "2 阶段金币超过 50 就不要乱花钱，存钱吃利息", FACTS
        )
        gold = next(c for c, _ in conds if c.fact == "gold")
        assert gold.value == 50


class TestStageExtraction:
    @pytest.mark.parametrize(
        "text,stage",
        [
            ("4-1 回合应该存钱", 4),
            ("第 4 阶段应该存钱", 4),
            ("4 阶段应该存钱", 4),
            ("2 阶段金币超过 50", 2),
            ("6-3 回合强化主 C", 6),
        ],
    )
    def test_stage_recognized(self, text, stage):
        cond = _extract_stage(text)
        assert cond is not None
        assert cond.fact == "stage"
        assert cond.value == stage

    def test_no_stage(self):
        assert _extract_stage("血量高于 50 时应该存钱") is None


# --------------------------------------------------------------------------
# 动作匹配
# --------------------------------------------------------------------------


class TestActionMatching:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("应该停手存钱", "save"),
            ("存钱吃利息", "save"),
            ("主动断连胜", "break_streak"),
            ("追三星", "all_in"),
            ("冲人口上等级", "level_up"),
            ("搜牌锁血", "reroll"),
            ("调整站位", "shift_lineup"),
        ],
    )
    def test_action_recognized(self, text, expected):
        assert expected in _match_action(text, "econ_action") or \
               expected in _match_action(text, "positioning_action")

    def test_no_action(self):
        assert _match_action("这一回合随便打打", "econ_action") == []


# --------------------------------------------------------------------------
# 端到端抽取
# --------------------------------------------------------------------------


class TestExtractCandidates:
    GUIDE = (
        "云顶之弈 13.2 运营要点\n"
        "4-1 回合连胜达到 3 以上并且血量不低于 50 时，应该停手存钱。\n"
        "2 阶段金币超过 50 就不要乱花钱，存钱吃利息。\n"
        "5 阶段装备达到 4 件并且血量不超过 30 时，应该搜牌锁血。\n"
        "这波该不该追三要看对手站位，尽量不要梭哈。\n"
    )

    def test_sentences_split(self):
        sents = split_sentences(self.GUIDE)
        assert len([s for s in sents if s.strip()]) >= 4

    def test_decidable_lines_produce_candidates(self):
        cands = extract_candidates(
            text=self.GUIDE, game="tft", version="13.2", known_facts=FACTS
        )
        assert cands, "示例攻略应抽出候选"
        assert all(c.trigger.all for c in cands), "候选都应有触发条件"
        assert all(c.expect_fact for c in cands), "候选都应有期望动作"

    def test_candidate_health_direction_is_correct(self):
        """回归：4-1 那条候选里的血量条件必须是 >=，不能是 <。"""
        cands = extract_candidates(
            text=self.GUIDE, game="tft", version="13.2", known_facts=FACTS
        )
        four_one = next(
            c for c in cands if c.trigger.all
            and any(cc.fact == "stage" and cc.value == 4 for cc in c.trigger.all)
        )
        health = next(cc for cc in four_one.trigger.all if cc.fact == "health")
        assert health.op == ">="
        assert health.value == 50

    def test_subjective_line_rejected(self):
        cands = extract_candidates(
            text=self.GUIDE, game="tft", version="13.2", known_facts=FACTS
        )
        assert not any(
            "该不该" in c.title or "尽量" in c.title for c in cands
        )

    def test_no_fabricated_facts(self):
        """抽取出的规则只能引用已注册事实。"""
        cands = extract_candidates(
            text=self.GUIDE, game="tft", version="13.2", known_facts=FACTS
        )
        for c in cands:
            for cc in c.trigger.all + c.trigger.any + c.trigger.none:
                assert cc.fact in FACTS
            assert c.expect_fact in FACTS

    def test_threshold_values_are_ints(self):
        cands = extract_candidates(
            text=self.GUIDE, game="tft", version="13.2", known_facts=FACTS
        )
        for c in cands:
            for cc in c.trigger.all + c.trigger.any + c.trigger.none:
                assert not isinstance(cc.value, float), \
                    f"{cc.fact} 的阈值 {cc.value} 应为整数"
