"""「节点级合法打法集」：做得没错的，不纠正；整段没做的，才纠正。

这是「零人工」能不能不冤枉人的关键。

问题
----
攻略在一个时段给好几套打法，各自有前提：

    4:00  塔的早期护盾已消失 + 敌方打野不在我方半区 → 带兵线推塔
    4:00  暴君倒计时进入最后一段 + 龙坑有我方视野     → 集合占位龙坑

这些前提（塔的护盾、倒计时、视野）**画面里读不出来**。
系统只知道玩家这会儿在做什么动作，照着规则库的字面判就会冤枉人：
玩家在 4:00 刷野，被说「该抢龙」—— 可他没抢龙是因为龙还没刷。

做法
----
每条规则带一个 ``node_actions``：**同一时段、同一类动作上，攻略给出的全部打法**。

- 玩家做了其中任何一件 → 他这会儿在按攻略做事，**不纠正他**；
- 只有整段一次都没做过节点里的任何一件事时才有话说 ——
  那才是「改了一定有进步」的纠正（原文给了后果）。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------- 工具


def _rule(expect_values, node_actions, *, fact="objective_action",
          subject_values=None, severity="medium"):
    from app.core.dsl import Condition, Rule, Subject, Trigger

    return Rule(
        id="HOK-MODE-TST-T0600-01",
        game="hok",
        version="S45",
        title="测试规则",
        category="打野·中期",
        trigger=Trigger(
            all=[
                Condition("position", "==", "jungle"),
                Condition("time_min", "between", [6, 9]),
            ],
            none=[Condition("my_hero", "in", ["赵云"])],
        ),
        expect_fact=fact,
        expect_op="in",
        expect_value=list(expect_values),
        subject=Subject(fact=fact, values=list(subject_values or expect_values)),
        node_actions=list(node_actions),
        severity=severity,
    )


def _ctx(action_value, *, fact="objective_action", time_min=6):
    from app.core.context import ActionRecord, MatchContext, RoundRecord

    rnd = RoundRecord(
        stage=1, phase=1,
        facts={"position": "jungle", "time_min": time_min, "phase": "mid"},
        actions=[ActionRecord(kind=fact, value=action_value)],
    )
    return MatchContext(
        match_id="t", game="hok", game_version="S45", rounds=[rnd]
    ), rnd


# ---------------------------------------------------------------- 判定


def test_a_legal_alternative_is_not_corrected():
    """玩家做了这个时段攻略认可的**另一件事** → 不纠正。

    原文 6:00 给的是「敌方减员就开团」和「己方减员就不接团」两套打法。
    分不清是哪种时，玩家做了任一件都说明他在按攻略做事。
    """
    from app.core.matcher import evaluate_round

    rule = _rule(["engage"], ["engage", "retreat"], fact="teamfight_action",
                 subject_values=["engage", "retreat", "focus_target"])
    ctx, rnd = _ctx("retreat", fact="teamfight_action")
    assert evaluate_round(rule, ctx, rnd) is None, "做了合法替代还被纠正"


def test_the_expected_action_is_satisfied():
    from app.core.matcher import evaluate_round

    rule = _rule(["engage"], ["engage", "retreat"], fact="teamfight_action",
                 subject_values=["engage", "retreat", "focus_target"])
    ctx, rnd = _ctx("engage", fact="teamfight_action")
    v = evaluate_round(rule, ctx, rnd)
    assert v is not None and v.status == "satisfied"


def test_something_outside_the_node_is_still_a_deviation():
    """节点里的打法一件都没做 → 判偏离（这是要纠正的那一种）。"""
    from app.core.matcher import evaluate_round

    rule = _rule(["engage"], ["engage", "retreat"], fact="teamfight_action",
                 subject_values=["engage", "retreat", "focus_target"])
    ctx, rnd = _ctx("focus_target", fact="teamfight_action")
    v = evaluate_round(rule, ctx, rnd)
    assert v is not None and v.status == "deviated"


def test_two_rules_in_the_same_cell_do_not_both_accuse():
    """同一格里的两条规则不会同时指控。

    「该开团」与「该避战」是同一场团战的两种打法。
    玩家做了其中一件，另一条就该闭嘴 —— 不能两条一起说他错。
    """
    from app.core.matcher import evaluate_round

    engage = _rule(["engage"], ["engage", "retreat"], fact="teamfight_action",
                   subject_values=["engage", "retreat", "focus_target"])
    retreat = _rule(["retreat"], ["engage", "retreat"], fact="teamfight_action",
                    subject_values=["engage", "retreat", "focus_target"])
    ctx, rnd = _ctx("engage", fact="teamfight_action")
    verdicts = [evaluate_round(r, ctx, rnd) for r in (engage, retreat)]
    fired = [v for v in verdicts if v is not None]
    assert len(fired) == 1, f"两条规则同时判了：{fired}"
    assert fired[0].status == "satisfied"


def test_a_rule_without_node_actions_keeps_the_old_behaviour():
    """没带 node_actions 的规则（老数据）行为不变。"""
    from app.core.matcher import evaluate_round

    rule = _rule(["engage"], [], fact="teamfight_action",
                 subject_values=["engage", "retreat", "focus_target"])
    ctx, rnd = _ctx("retreat", fact="teamfight_action")
    v = evaluate_round(rule, ctx, rnd)
    assert v is not None and v.status == "deviated"


# ---------------------------------------------------------------- 种子库


def _seed_rules():
    path = APP_ROOT / "app" / "seed" / "hok_modes.json"
    return json.loads(path.read_text("utf-8"))


def test_every_seed_rule_carries_its_node_actions():
    rules = _seed_rules()
    assert rules, "种子库是空的"
    for r in rules:
        assert r.get("node_actions"), f"{r['id']} 没有节点级合法打法集"


def test_the_node_actions_of_a_cell_are_identical():
    """同一格、同一动作事实的规则，合法打法集必须一样。

    不一样的话，玩家做了其中一条建议的动作时，另一条仍会指控他 ——
    「同一格里两条模式必须触发互斥」这条约束就破了。
    """
    from collections import defaultdict

    def window(r):
        for c in r["trigger"]["all"]:
            if c["fact"] == "time_min" and c["op"] == "between":
                return tuple(c["value"])
        return None

    cells = defaultdict(list)
    for r in _seed_rules():
        lane = r["trigger"]["all"][0]["value"]
        cells[(lane, window(r), r["expect"]["fact"])].append(r)
    for key, group in cells.items():
        sets = {tuple(sorted(r["node_actions"])) for r in group}
        assert len(sets) == 1, f"{key} 里的合法打法集不一致：{sets}"


def test_no_seed_rule_gates_on_a_number():
    """分路层的触发条件里不许再有「经济差 / 击杀差 / 等级」。

    那些数在视频里读不出来，条件永远不成立 → 规则一条都判不了。
    手工录入的对局仍然带着这些数，判定照样精确。
    """
    numeric = {"gold_diff", "kill_diff", "tower_diff", "level", "gold"}
    for r in _seed_rules():
        for group in ("all", "any", "none"):
            for c in r["trigger"].get(group) or []:
                assert c["fact"] not in numeric, (
                    f"{r['id']} 的触发条件还挂着数值 {c['fact']}"
                )


def test_the_rules_that_mention_the_ult_use_it_as_a_condition():
    """原文里明写「大招在手」「控制链就绪」的分支，用大招状态当条件。

    大招亮不亮在画面里读得出 —— 这是不依赖数值的触发条件。
    """
    rules = _seed_rules()
    hits = 0
    for r in rules:
        conds = [c for c in r["trigger"]["all"]
                 if c["fact"] == "has_ult_ready"]
        if conds:
            hits += 1
            assert r["expect"]["fact"] in ("teamfight_action", "positioning_action")
    assert hits >= 4, f"用大招状态当条件的规则太少（{hits} 条）"


def test_in_combat_is_a_registered_fact():
    """模型每窗都在答「在不在打团」，适配器得认这个事实。

    不认的话它会被当成「未注册的事实」丢掉 —— 读了等于没读。
    """
    from app.adapters import registry  # noqa: F401  触发适配器注册
    from app.adapters.base import get_adapter

    assert "in_combat" in get_adapter("hok").known_facts()
