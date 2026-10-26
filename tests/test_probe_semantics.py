"""探测器的三态 → 内核的三判。

这是整个方案的地基：
- 探测器 **found** → 内核判「做到了」；
- 探测器 **absent**（信号可读、明确没有）→ 内核判「该做而没做」；
- 探测器 **unknown**（读不出来）→ 内核判「无法判定」。

**「无法判定」和「该做而没做」是两件事** —— 混起来，报告里就会出现
一条玩家从没下过的结论，而界面上完全看不出来。
"""
from __future__ import annotations

import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from app.core.context import ActionRecord, MatchContext, RoundRecord  # noqa: E402
from app.core.dsl import Condition, Rule, Source, Subject, Trigger  # noqa: E402
from app.core.matcher import evaluate_round                  # noqa: E402
from app.collection.recognizer import HudRegions, WindowRecognizer  # noqa: E402
from app.collection import probes as pm                      # noqa: E402


def _rule():
    """一条「前期打野该刷野」的模式。"""
    return Rule(
        id="HOK-TEST-JGL-001",
        game="hok",
        version="S45",
        title="前期打野以刷野为主",
        category="打野·前期",
        trigger=Trigger(all=[Condition(fact="position", op="==", value="jungle"),
                             Condition(fact="time_min", op="between", value=[0, 4])]),
        expect_fact="objective_action",
        expect_op="in",
        expect_value=["farm_jungle", "invade_jungle"],
        subject=Subject(fact="objective_action",
                        values=["farm_jungle", "invade_jungle", "clear_vision_sprite"]),
        expect_scope="at_least_once",
        source=Source(guide="g", locator="l", quoted="q"),
    )


def _ctx(rnd):
    return MatchContext(match_id="t", game="hok", game_version="S45", rounds=[rnd])


def test_found_action_is_satisfied():
    """探测器说「刷野」→ 规则判「做到了」。"""
    rnd = RoundRecord(
        stage=0, phase=1,
        facts={"position": "jungle", "time_min": 2},
        actions=[ActionRecord(kind="objective_action", value="farm_jungle")],
    )
    v = evaluate_round(_rule(), _ctx(rnd), rnd)
    assert v is not None and v.status == "satisfied"


def test_absent_is_deviated():
    """信号可读、明确没有 → 判「该做而没做」。"""
    rnd = RoundRecord(
        stage=0, phase=1,
        facts={"position": "jungle", "time_min": 2, "objective_action": None},
    )
    v = evaluate_round(_rule(), _ctx(rnd), rnd)
    assert v is not None and v.status == "deviated", (
        "动作事实缺失时不能判「该做而没做」，那是替玩家编结论")


def test_unknown_is_undetermined():
    """读不出来 → 判「无法判定」。"""
    rnd = RoundRecord(
        stage=0, phase=1,
        facts={"position": "jungle", "time_min": 2},
        missing_facts={"objective_action"},
    )
    v = evaluate_round(_rule(), _ctx(rnd), rnd)
    assert v is not None and v.status == "undetermined"


def test_subject_guard_prevents_false_deviation():
    """这一段在做别的事 → 规则**不适用**（不进分母），不是「该做而没做」。"""
    rnd = RoundRecord(
        stage=0, phase=1,
        facts={"position": "jungle", "time_min": 2},
        actions=[ActionRecord(kind="objective_action", value="take_dragon")],
    )
    assert evaluate_round(_rule(), _ctx(rnd), rnd) is None


def test_recognizer_writes_the_three_states_as_facts():
    """识别器把三态写成「事实」，内核据此判三态。

    - found   → 事实有取值（置信度 ≥ 门槛）；
    - absent  → 事实**采信但为空**（置信度高、值为 None）；
    - unknown → 事实**未采信**（置信度 0）。
    """
    reg = HudRegions.load()
    r = WindowRecognizer(reg)
    # 造一个「没有帧但有轨迹」的窗：位置读不出来 → 全部 unknown
    out = r.analyze([], None)
    assert out.missing == ["画面"]

    out2 = r.analyze([__import__("numpy").zeros((574, 1280, 3), dtype="uint8")])
    # 一张纯黑帧：位置读不出、大招按钮读不出 → 事实缺失
    assert "大招就绪" in out2.missing


def test_probe_result_states_are_distinct():
    """三态互不相同，且 absent 只有强探测器能给。"""
    assert pm.FOUND != pm.ABSENT != pm.UNKNOWN
    res = pm.ProbeResult(kind="objective_action", value="farm_jungle")
    assert res.unknown
    res2 = pm.ProbeResult(kind="objective_action", value="farm_jungle",
                          status=pm.ABSENT)
    assert res2.absent and not res2.unknown


# ---------------------------------------------------------------------------
# 「玩家不标」这件事在服务端的表现
# ---------------------------------------------------------------------------

class _Ann:
    """一次标注提交（这里只用到几个字段）。"""

    def __init__(self, facts=None, actions=None, skipped=False):
        self.facts = facts or {}
        self.actions = actions or []
        self.skipped = skipped
        self.note = ""


def _hok_adapter():
    """拿王者适配器（导入模块会完成注册）。"""
    import app.adapters.honor_of_kings  # noqa: F401  触发注册
    from app.adapters.base import get_adapter

    return get_adapter("hok")


def _round_with_probe(adapter):
    """造一个「识别层已经认出刷野」的回合。"""
    from app.collection.video import DetectedAction, DetectedFact, RecognizedRound

    return RecognizedRound(
        stage=0, phase=1,
        facts=[
            DetectedFact(name="time_min", value=2, confidence=0.8, method="window"),
            DetectedFact(name="objective_action", value="farm_jungle",
                         confidence=0.8, method="auto"),
        ],
        actions=[
            DetectedAction(kind="objective_action", value="farm_jungle",
                           label="刷野补经济", confidence=0.8, source="probe"),
        ],
    )


def test_player_labels_nothing_and_the_detected_action_survives():
    """玩家一个字都没标 → 识别层认出的动作**照样**进对局。

    这是「我完全不想标」能成立的前提：动作是从画面里读出来的，
    不是从表单里来的。
    """
    from app.collection.video import RecognitionResult, to_raw_match
    from app.collection.video_api import _apply_annotation
    adapter = _hok_adapter()
    rnd = _round_with_probe(adapter)
    _apply_annotation(rnd, adapter, _Ann(facts={"position": "jungle"}))
    raw, _deg = to_raw_match(
        RecognitionResult(match_id="t", game="hok", game_version="S45",
                          rounds=[rnd]),
        adapter.known_facts(), adapter.known_action_facts())
    assert raw["rounds"][0]["actions"], "自动识别的动作被清空了"
    assert raw["rounds"][0]["actions"][0]["value"] == "farm_jungle"

    # 分路是**对局级**事实（适配器按 player.position 广播到每一段）
    raw["player"] = {"position": "jungle"}
    ctx = adapter.normalize(raw)
    r0 = ctx.rounds[0]
    assert any(a.value == "farm_jungle" for a in r0.actions), r0.actions
    v = evaluate_round(_rule(), ctx, r0)
    assert v is not None and v.status == "satisfied"


def test_probe_action_keeps_its_source_through_storage():
    """画面里看出来的动作，落库后**不能**变成玩家自述。

    来源这一项以前在 ``to_raw_match`` 里被丢了，适配器只能默认成
    ``player`` —— 于是「系统从画面里读出来的」在库里就成了玩家自己说的，
    报告据此写的「实际在做什么」就成了替玩家编话。
    """
    from app.collection.video import RecognitionResult, to_raw_match
    adapter = _hok_adapter()
    rnd = _round_with_probe(adapter)
    raw, _deg = to_raw_match(
        RecognitionResult(match_id="t", game="hok", game_version="S45",
                          rounds=[rnd]),
        adapter.known_facts(), adapter.known_action_facts())
    assert raw["rounds"][0]["actions"][0]["source"] == "probe"

    raw["player"] = {"position": "jungle"}
    ctx = adapter.normalize(raw)
    sources = {a.source for a in ctx.rounds[0].actions}
    assert sources == {"probe"}, sources


def test_skipped_window_is_undetermined_not_deviated():
    """玩家说「这段看不清」→ 判**无法判定**，不是「该做而没做」。

    两者的区别是这整个方案的底线：把「看不出来」当成「没做」，
    报告里就会出现一条玩家从没下过的结论。
    """
    from app.collection.video import RecognitionResult, to_raw_match
    from app.collection.video_api import _apply_annotation
    adapter = _hok_adapter()
    rnd = _round_with_probe(adapter)
    _apply_annotation(rnd, adapter, _Ann(facts={"position": "jungle"}, skipped=True))
    raw, _deg = to_raw_match(
        RecognitionResult(match_id="t", game="hok", game_version="S45",
                          rounds=[rnd]),
        adapter.known_facts(), adapter.known_action_facts())
    assert not raw["rounds"][0]["actions"], "看不清的段不该留着动作"
    assert "objective_action" in raw["rounds"][0]["missing_facts"]

    raw["player"] = {"position": "jungle"}
    ctx = adapter.normalize(raw)
    r0 = ctx.rounds[0]
    v = evaluate_round(_rule(), ctx, r0)
    assert v is not None and v.status == "undetermined", (
        "看不清的段被判成偏离，等于替玩家编了一条他没下过的结论")
