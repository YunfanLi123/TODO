"""视频识别契约与降级转换的测试。

这一层是整个系统里最容易悄悄出错的地方：识别结果带着不确定性进入
系统，如果降级规则写松了，报告就会把猜测当成事实讲给玩家。所以测试
重点在**拒绝什么**，而不只是接收什么。
"""

from __future__ import annotations

import pytest

from app.adapters import registry as _registry  # noqa: F401  触发适配器注册
from app.collection.video import (
    ACTION_ACCEPT_THRESHOLD,
    FACT_ACCEPT_THRESHOLD,
    GATE_TRUST_THRESHOLD,
    DetectedAction,
    DetectedFact,
    GateFrame,
    RecognitionResult,
    RecognizedRound,
    VideoRecognizer,
    summarize_cost,
    to_raw_match,
)

FACTS = {"health", "gold", "stage", "phase", "streak", "level"}
ACTION_FACTS = {"econ_action", "positioning_action"}


def fact(name, value, conf, failure="", method="roi"):
    return DetectedFact(name=name, value=value, confidence=conf,
                        method=method, failure=failure)


def action(kind, value, conf):
    return DetectedAction(kind=kind, value=value, confidence=conf)


def rnd(stage=4, phase=1, facts=None, actions=None, gate=1.0):
    return RecognizedRound(
        stage=stage, phase=phase,
        facts=facts or [], actions=actions or [], gate_confidence=gate,
    )


def result(rounds, frames=None, **kw):
    return RecognitionResult(
        match_id=kw.pop("match_id", "v1"),
        game=kw.pop("game", "tft"),
        game_version=kw.pop("game_version", "13.1"),
        frames=frames or [],
        rounds=rounds,
        **kw,
    )


@pytest.fixture
def seeded_session(tmp_path):
    """每个测试一个独立库 + SAVEPOINT 隔离。

    视频识别测试要反复建对局、反复回滚，没有隔离的话一个断言失败就会
    连累后续测试报出与真实缺陷无关的错误。
    """
    from app import db as dbmod
    from app.db import get_engine, session_scope
    from app.rules.store import latest_version, load_seed_library

    dbmod._engine = None
    dbmod._SessionFactory = None
    get_engine(str(tmp_path / "video_test.db"))

    with session_scope() as s:
        if not latest_version(s, "tft"):
            load_seed_library(s, "tft")

    with session_scope() as s:
        nested = s.begin_nested()
        try:
            yield s
        finally:
            if nested.is_active:
                nested.rollback()


# --------------------------------------------------------------------------
# 置信度三态
# --------------------------------------------------------------------------


class TestConfidenceTiers:
    def test_high_confidence_accepted(self):
        raw, rep = to_raw_match(
            result([rnd(facts=[fact("health", 58, 0.98)])]), FACTS, ACTION_FACTS
        )
        assert raw["rounds"][0]["facts"]["health"] == 58
        assert raw["rounds"][0]["missing_facts"] == []

    def test_low_confidence_kept_but_recorded(self):
        """关键行为：0.45 置信度的「血量 58」保留使用，但必须被标记。

        直接丢弃会让识别质量中等的视频完全无法分析（用户半夜录屏、
        风扇响、画面糊一点就得不出任何结论）。保留 + 标记才能同时做到
        「系统能用」和「玩家知道该不该信」。
        """
        raw, rep = to_raw_match(
            result([rnd(facts=[fact("health", 58, 0.45)])]), FACTS, ACTION_FACTS
        )
        assert raw["rounds"][0]["facts"]["health"] == 58
        assert raw["rounds"][0]["missing_facts"] == []

        assert rep.low_confidence_facts
        item = rep.low_confidence_facts[0]
        assert item["fact"] == "health"
        assert item["value"] == 58
        assert item["confidence"] == pytest.approx(0.45)

    def test_very_low_confidence_is_rejected_with_reason(self):
        raw, rep = to_raw_match(
            result(
                [rnd(facts=[fact("health", 58, 0.1, failure="画面被遮挡")])]
            ),
            FACTS, ACTION_FACTS,
        )
        assert "health" not in raw["rounds"][0]["facts"]
        assert "health" in raw["rounds"][0]["missing_facts"]
        reasons = [r.get("reason") for r in rep.rejected_facts]
        assert any("画面被遮挡" in (r or "") for r in reasons)

    def test_threshold_boundaries(self):
        """恰好等于门槛时采信 —— 边界行为必须确定，不能看运气。"""
        at = to_raw_match(
            result([rnd(facts=[fact("gold", 10, FACT_ACCEPT_THRESHOLD)])]),
            FACTS, ACTION_FACTS,
        )[0]
        assert at["rounds"][0]["facts"]["gold"] == 10

        below = to_raw_match(
            result([rnd(facts=[fact("gold", 10, FACT_ACCEPT_THRESHOLD - 0.01)])]),
            FACTS, ACTION_FACTS,
        )[0]
        assert "gold" in below["rounds"][0]["missing_facts"]


# --------------------------------------------------------------------------
# 未注册事实/动作
# --------------------------------------------------------------------------


class TestUnregisteredRejection:
    def test_unregistered_fact_dropped_with_reason(self):
        """识别器多认出的事实必须丢弃并记录，否则会污染内核的事实集。"""
        raw, rep = to_raw_match(
            result([rnd(facts=[fact("combo_damage", 999, 0.99)])]),
            FACTS, ACTION_FACTS,
        )
        assert "combo_damage" not in raw["rounds"][0]["facts"]
        assert any("combo_damage" in (r.get("fact") or "") for r in rep.rejected_facts)

    def test_unregistered_action_dropped(self):
        raw, rep = to_raw_match(
            result([rnd(actions=[action("dance_action", "wave", 0.99)])]),
            FACTS, ACTION_FACTS,
        )
        assert raw["rounds"][0]["actions"] == []
        assert any("dance_action" in (d.get("kind") or "") for d in rep.dropped_actions)

    def test_low_confidence_action_dropped(self):
        raw, rep = to_raw_match(
            result([rnd(actions=[action("econ_action", "all_in", 0.2)])]),
            FACTS, ACTION_FACTS,
        )
        assert raw["rounds"][0]["actions"] == []
        assert rep.dropped_actions

    def test_flagged_action_is_kept_with_confidence(self):
        """介于两个门槛之间的动作要保留 —— 报告需要它来提示「建议确认」。"""
        conf = (ACTION_ACCEPT_THRESHOLD + 0.75) / 2
        raw, _ = to_raw_match(
            result([rnd(actions=[action("econ_action", "all_in", conf)])]),
            FACTS, ACTION_FACTS,
        )
        acts = raw["rounds"][0]["actions"]
        assert len(acts) == 1
        assert acts[0]["confidence"] == pytest.approx(conf, abs=1e-3)


# --------------------------------------------------------------------------
# 回合门控
# --------------------------------------------------------------------------


class TestRoundGate:
    def test_low_gate_confidence_drops_whole_round(self):
        """门控没数准边界，里面的 ROI 再准也没意义 —— 整段判缺失。"""
        raw, rep = to_raw_match(
            result([rnd(facts=[fact("health", 58, 0.99), fact("gold", 30, 0.99)],
                        gate=0.3)]),
            FACTS, ACTION_FACTS,
        )
        rd = raw["rounds"][0]
        assert rd["facts"] == {}
        assert set(rd["missing_facts"]) == {"health", "gold"}
        assert rep.dropped_rounds == 1

    def test_gate_threshold_is_inclusive(self):
        raw, _ = to_raw_match(
            result([rnd(facts=[fact("health", 58, 0.99)], gate=GATE_TRUST_THRESHOLD)]),
            FACTS, ACTION_FACTS,
        )
        assert raw["rounds"][0]["facts"]["health"] == 58

    def test_empty_round_is_kept_not_discarded(self):
        """什么都不知道的回合也要保留。

        丢掉它等于假装这回合不存在，玩家会以为系统漏检了；
        保留它，报告才能说「这一回合我们什么都不知道」。
        """
        raw, rep = to_raw_match(result([rnd(facts=[])]), FACTS, ACTION_FACTS)
        assert len(raw["rounds"]) == 1
        assert raw["rounds"][0]["facts"] == {}
        assert rep.usable_rounds == 0


# --------------------------------------------------------------------------
# 诊断留档
# --------------------------------------------------------------------------


class TestDiagnostics:
    def test_recognition_block_always_present(self):
        raw, _ = to_raw_match(
            result([rnd(facts=[fact("health", 58, 0.99)])]), FACTS, ACTION_FACTS
        )
        rec = raw["recognition"]
        assert "degradation" in rec
        assert rec["degradation"]["total_rounds"] == 1

    def test_warnings_propagate(self):
        raw, rep = to_raw_match(
            result([rnd()], warnings=["第 3 回合之后画面被遮挡"]), FACTS, ACTION_FACTS
        )
        assert "第 3 回合之后画面被遮挡" in raw["recognition"]["degradation"]["warnings"]
        assert rep.warnings

    def test_no_fabricated_facts(self):
        """识别器没给出的事实，绝不能被补默认值。"""
        raw, _ = to_raw_match(result([rnd(facts=[])]), FACTS, ACTION_FACTS)
        assert raw["rounds"][0]["facts"] == {}

    def test_missing_facts_deduplicated(self):
        raw, _ = to_raw_match(
            result([rnd(facts=[fact("health", 1, 0.1), fact("health", 2, 0.1)])]),
            FACTS, ACTION_FACTS,
        )
        assert raw["rounds"][0]["missing_facts"] == ["health"]


# --------------------------------------------------------------------------
# 识别器接口
# --------------------------------------------------------------------------


class TestRecognizerInterface:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            VideoRecognizer()

    def test_subclass_must_implement_both_steps(self):
        class Partial(VideoRecognizer):
            def detect_round_boundaries(self, video):
                return []

        with pytest.raises(TypeError):
            Partial()

    def test_two_step_pipeline_shape(self):
        class Stub(VideoRecognizer):
            recognizer_id = "stub"
            recognizer_version = "1.0"

            def detect_round_boundaries(self, video):
                return [
                    GateFrame(i, i * 0.5, is_boundary=(i == 0 or i == 20),
                              confidence=0.9)
                    for i in range(40)
                ]

            def recognize_rounds(self, video, boundaries):
                starts = [f for f in boundaries if f.is_boundary]
                return [rnd(stage=i + 1, facts=[fact("health", 90, 0.95)])
                        for i, _ in enumerate(starts)]

        rec = Stub()
        frames = rec.detect_round_boundaries(None)
        assert sum(1 for f in frames if f.is_boundary) == 2

        rounds = rec.recognize_rounds(None, frames)
        assert len(rounds) == 2

        res = result(rounds, frames=frames)
        cost = summarize_cost(res)
        assert cost["frames_scanned"] == 40
        assert cost["fine_recognitions"] == 2
        assert cost["fine_ratio"] < 0.1      # 昂贵识别确实被限制在边界上


# --------------------------------------------------------------------------
# 端到端：识别结果 → 报告
# --------------------------------------------------------------------------


class TestRecognitionToReport:
    def test_degraded_match_reports_honestly(self, seeded_session):
        """识别不可靠时，报告既不能用猜的值下结论，也不能假装没这回事。"""
        from app.adapters.base import get_adapter
        from app.collection import match as match_col
        from app.core.context import MatchContext
        from app.report.builder import build_match_report
        from app.rules.store import get_rules, latest_version

        adapter = get_adapter("tft")
        res = result(
            [
                rnd(
                    stage=4, phase=1,
                    facts=[
                        fact("stage", 4, 0.99), fact("phase", 1, 0.99),
                        fact("health", 58, 0.99), fact("streak", 4, 0.99),
                        fact("level", 7, 0.5), fact("pressure", "mid", 0.2),
                    ],
                    actions=[action("econ_action", "all_in", 0.95)],
                )
            ],
            match_id="video-demo-1",
        )
        raw, degradation = to_raw_match(
            res, adapter.known_facts(), adapter.known_action_facts()
        )
        raw["played_at"] = "2026-10-14 21:00"

        with seeded_session as session:
            record, _ = match_col.save_match(
                session, adapter, raw, source_kind="video"
            )
            rules = get_rules(session, "tft", latest_version(session, "tft"))
            ctx = MatchContext.from_dict(record.get_payload())
            report = build_match_report(rules, ctx)

        # 低置信度（level 0.5）保留使用但被记录；未识别（pressure 0.2）进缺失
        stored = ctx.round_by_label("4-1")
        assert stored.facts["health"] == 58
        assert stored.facts["level"] == 7          # 低置信度仍可用
        assert "pressure" in stored.missing_facts  # 0.2 未达采信门槛

        # 但诊断信息完整保留，且报告能指出哪些结论建立在低置信度之上
        assert report["recognition"]["rejected_facts"]
        assert report["recognition"]["low_confidence_facts"]
        assert degradation.accepted_facts >= 4

    def test_fully_recognized_match_has_no_diagnosis_block(self, seeded_session):
        from app.adapters.base import get_adapter
        from app.collection import match as match_col
        from app.core.context import MatchContext
        from app.report.builder import build_match_report
        from app.rules.store import get_rules, latest_version

        adapter = get_adapter("tft")
        res = result(
            [
                rnd(
                    stage=4, phase=1,
                    facts=[
                        fact("stage", 4, 0.99), fact("phase", 1, 0.99),
                        fact("health", 58, 0.99), fact("streak", 4, 0.99),
                        fact("level", 7, 0.99), fact("pressure", "mid", 0.99),
                    ],
                    actions=[action("econ_action", "all_in", 0.99)],
                )
            ],
            match_id="video-demo-clean",
        )
        raw, _ = to_raw_match(res, adapter.known_facts(), adapter.known_action_facts())
        raw["played_at"] = "2026-10-14 21:30"

        with seeded_session as session:
            record, _ = match_col.save_match(session, adapter, raw, source_kind="video")
            rules = get_rules(session, "tft", latest_version(session, "tft"))
            ctx = MatchContext.from_dict(record.get_payload())
            report = build_match_report(rules, ctx)

        rec = report["recognition"]
        assert rec["low_confidence_facts"] == []
        assert rec["rejected_facts"] == []
        assert rec["dropped_rounds"] == 0
        # 识别干净时不显示「识别质量」区块，模板会据此跳过
        assert report["low_confidence_facts"] == []
