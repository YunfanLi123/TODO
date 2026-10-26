"""MOBA「阶段 × 位置 → 该做什么」的抽取测试。

重点守三件事：
  1. **维度必须齐** —— 缺位置或缺阶段的句子不进候选
  2. **位置要认主语** ——「游走位要跟着打野」是游走的规则，不是打野的
  3. **位置要真的参与判定** —— 同一种行为换个位置就不该命中同一条规则
"""

from __future__ import annotations

import pytest

from app.adapters import registry as _registry  # noqa: F401
from app.rules.moba_extract import PHASE_WINDOWS

FACTS = _registry.get_adapter("hok").known_facts()


def _extract(text):
    from app.rules.moba_extract import extract_moba

    return extract_moba(text, FACTS, version="S38", guide_title="t")


# --------------------------------------------------------------------------
# 维度完整性
# --------------------------------------------------------------------------


class TestDimensions:
    def test_full_triple_is_extracted(self):
        cands = _extract("前期打野以刷野为主。")
        assert len(cands) == 1
        c = cands[0]
        assert c.position == "jungle"
        assert c.phase == "early"
        assert c.action_value == "farm_jungle"
        assert c.extra_conditions == []

    def test_missing_position_dropped(self):
        assert _extract("中期视野很重要，开龙前先把河道点亮。") == []

    def test_missing_phase_dropped(self):
        assert _extract("打野记得刷野。") == []

    def test_missing_action_dropped(self):
        """「注意节奏」没有可判定动作 —— 收了也只是噪音。"""
        assert _extract("前期打野要注意节奏。") == []

    def test_subjective_dropped(self):
        assert _extract("一般来说前期打野要注意别被反野。") == []

    def test_parallel_position_dropped(self):
        """「打野和中路都要插眼」说不清给谁看。"""
        assert _extract("前期打野和中路都要插眼。") == []

    def test_time_window_attached(self):
        c = _extract("中期游走位要跟着打野入侵野区。")[0]
        assert PHASE_WINDOWS[c.phase] == [5, 10]


# --------------------------------------------------------------------------
# 中文主语优先
# --------------------------------------------------------------------------


class TestSubjectPriority:
    def test_first_position_wins(self):
        """中文主语在前：「游走位要跟着打野」讲的是游走。"""
        c = _extract("中期游走位要跟着打野入侵野区扩大优势。")
        assert len(c) == 1
        assert c[0].position == "roam"
        assert c[0].action_value == "invade_jungle"

    def test_alias_normalized(self):
        for word, expect in (("打野", "jungle"), ("游走", "roam"),
                             ("中单", "mid"), ("对抗路", "side"),
                             ("射手", "carry"), ("辅助", "support")):
            got = _extract(f"前期{word}应该刷野。")
            assert got and got[0].position == expect, f"{word} -> {expect}"


# --------------------------------------------------------------------------
# 多阶段拆分
# --------------------------------------------------------------------------


class TestMultiPhase:
    def test_two_phases_two_rules(self):
        """「前期要gank，中期带线」是两条规则，不是一条。"""
        cands = _extract("刺客前期要gank，中期带线。")
        assert len(cands) == 2
        assert {c.phase for c in cands} == {"early", "mid"}
        assert all(c.position == "jungle" for c in cands)

    def test_duplicate_phrase_deduped(self):
        cands = _extract("前期打野刷野。")
        assert len(cands) == 1


# --------------------------------------------------------------------------
# 落地为规则
# --------------------------------------------------------------------------


class TestRuleOutput:
    def test_rule_validates(self):
        from app.rules.moba_extract import stable_rule_id

        c = _extract("前期打野以刷野为主。")[0]
        rule = c.to_rule(stable_rule_id(c))
        rule.validate(FACTS)          # 不抛异常即通过

    def test_rule_has_position_and_time(self):
        from app.rules.moba_extract import stable_rule_id

        c = _extract("前期打野以刷野为主。")[0]
        facts = [x.fact for x in c.to_rule(stable_rule_id(c)).trigger.all]
        assert "position" in facts
        assert "time_min" in facts

    def test_category_is_position_phase(self):
        c = _extract("前期打野以刷野为主。")[0]
        rule = c.to_rule("X")
        assert rule.category == "打野·前期"

    def test_stable_id_same_input(self):
        from app.rules.moba_extract import stable_rule_id

        a = _extract("前期打野以刷野为主。")[0]
        b = _extract("前期打野以刷野为主。")[0]
        assert stable_rule_id(a) == stable_rule_id(b)

    def test_different_action_different_id(self):
        from app.rules.moba_extract import stable_rule_id

        a = _extract("前期打野以刷野为主。")[0]
        b = _extract("前期打野应该反野入侵。")[0]
        assert stable_rule_id(a) != stable_rule_id(b)


# --------------------------------------------------------------------------
# 位置真的参与判定（内核层）
# --------------------------------------------------------------------------


class TestPositionMattersAtRuntime:
    def _ctx(self, position, action):
        from app.core.context import ActionRecord, MatchContext, RoundRecord

        rnd = RoundRecord(
            stage=1, phase=1,
            facts={"position": position, "time_min": 3, "phase": "early"},
            actions=[ActionRecord(kind="objective_action", value=action)],
        )
        return MatchContext(
            match_id="t", game="hok", game_version="S38", rounds=[rnd]
        ), rnd

    def test_jungle_rule_fires_for_jungle(self):
        from app.core.matcher import evaluate_round
        from app.rules.moba_extract import stable_rule_id

        rule = _extract("前期打野以刷野为主。")[0].to_rule(
            stable_rule_id(_extract("前期打野以刷野为主。")[0]))
        ctx, rnd = self._ctx("jungle", "invade_jungle")
        v = evaluate_round(rule, ctx, rnd)
        assert v is not None and v.status == "deviated"

    def test_same_action_does_not_fire_for_carry(self):
        from app.core.matcher import evaluate_round
        from app.rules.moba_extract import stable_rule_id

        c = _extract("前期打野以刷野为主。")[0]
        rule = c.to_rule(stable_rule_id(c))
        ctx, rnd = self._ctx("carry", "invade_jungle")
        assert evaluate_round(rule, ctx, rnd) is None

    def test_same_position_correct_action_satisfied(self):
        from app.core.matcher import evaluate_round
        from app.rules.moba_extract import stable_rule_id

        c = _extract("前期打野以刷野为主。")[0]
        rule = c.to_rule(stable_rule_id(c))
        ctx, rnd = self._ctx("jungle", "farm_jungle")
        v = evaluate_round(rule, ctx, rnd)
        assert v is not None and v.status == "satisfied"

    def test_late_phase_outside_window_does_not_fire(self):
        """15 分钟的早期打野不适用「前期刷野」—— 时间窗是硬门槛。"""
        from app.core.context import ActionRecord, MatchContext, RoundRecord
        from app.core.matcher import evaluate_round
        from app.rules.moba_extract import stable_rule_id

        c = _extract("前期打野以刷野为主。")[0]
        rule = c.to_rule(stable_rule_id(c))
        rnd = RoundRecord(
            stage=3, phase=1,
            facts={"position": "jungle", "time_min": 18, "phase": "late"},
            actions=[ActionRecord(kind="objective_action", value="farm_jungle")],
        )
        ctx = MatchContext(
            match_id="t", game="hok", game_version="S38", rounds=[rnd])
        assert evaluate_round(rule, ctx, rnd) is None


# --------------------------------------------------------------------------
# 跳过原因可解释
# --------------------------------------------------------------------------


class TestSkipExplanation:
    def test_reasons_listed(self):
        from app.rules.moba_extract import explain_skips

        skips = explain_skips(
            "中期视野很重要。\n前期打野以刷野为主。\n记住发育节奏。", FACTS)
        by = {s["sentence"][:4]: s["reasons"] for s in skips}
        # 能抽成规则的那句不该出现在 skips 里
        assert "前期打野" not in by, "成功抽取的句子不该被列为跳过"
        assert "缺分路" in by["中期视野"]
        assert "缺分路" in by["记住发育"]

    def test_all_reasons_present_for_empty_line(self):
        from app.rules.moba_extract import explain_skips

        skips = explain_skips("记住节奏很重要。", FACTS)
        r = skips[0]["reasons"]
        assert "缺分路" in r and "缺阶段" in r and "缺可判定动作" in r
