"""模型裁决：清单供给 + 硬校验 + 报告落地。

这三层是「只指出改正之后一定会进步的模式」的实现处：

- 清单把攻略原文（前提 / 该做 / 后果）按时间窗送到模型手上；
- 裁决要过四道闸才作数（前提证实 / 没做节点认可的事 / 没做这一条要求的事 /
  帧号真实存在）；
- 报告里每一条都带着它据以判断的那几帧 —— 文字与片段同一时刻。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.collection import vlm_review
from app.rules import checklist


# ---------------------------------------------------------------- 清单


def test_checklist_is_taken_per_window():
    """清单按时间窗取 —— 4:00 那窗就该是 4:00 的那几条。"""
    items = checklist.window_checklist("hok", 4, 4)
    assert items
    assert all(win[0] <= 4 <= win[1] or win[0] <= 4 <= win[1]
               for win in (i["window"] for i in items))
    ids = {i["id"] for i in items}
    assert "HOK-MODE-JGL-T0400-01" in ids


def test_checklist_narrows_to_the_players_lane():
    """分路已知就只给这一条线 —— 清单短一大截，模型也不会乱指。"""
    items = checklist.window_checklist("hok", 4, 4, lane="jungle")
    assert items
    assert {i["lane"] for i in items} == {"jungle"}
    assert len(items) < len(checklist.window_checklist("hok", 4, 4))


def test_checklist_carries_the_guide_verbatim():
    """清单带的是**攻略原文**（前提 / 该做 / 后果），不是我们的转述。"""
    items = checklist.window_checklist("hok", 4, 4, lane="jungle")
    item = next(i for i in items if i["id"] == "HOK-MODE-JGL-T0400-01")
    assert "塔的早期护盾已消失" in item["premise"]
    assert "带兵线推塔" in item["action"]
    assert item["outcome"]
    assert item["advice"]


def test_node_actions_are_not_polluted_by_hero_rules():
    """节点合法打法集只能由**分路层**规则算。

    英雄层有 1596 条，全都挂在同一条分路上、时间窗也常常一样 ——
    一并进来会把集合撑成几百个取值，「玩家做过其中任何一件就不纠正」
    这条就废了。
    """
    node = checklist.node_action_map("hok")
    jungle_4 = node[("jungle", (4, 4))]
    assert set(jungle_4) == {"positioning_action", "objective_action",
                              "teamfight_action"}
    assert "split_push" not in jungle_4.get("objective_action", [])
    assert "farm_jungle" not in jungle_4.get("objective_action", [])


# ---------------------------------------------------------------- 校验


def _entry(**kw):
    base = {
        "id": "HOK-MODE-JGL-T0400-01",
        "lane": "jungle",
        "lane_label": "打野",
        "title": "打野 4:00：带线推塔",
        "premise": "塔的早期护盾已消失",
        "action": "带兵线推塔",
        "outcome": "塔的经济到手",
        "advice": "带兵线推塔。",
        "expect_fact": "positioning_action",
        "expect_values": ["push_wave"],
        "subject": {"fact": "positioning_action",
                    "values": ["push_wave", "ward_vision", "pull_back"]},
        "node_actions": {"positioning_action": ["push_wave", "ward_vision"]},
        "window": [4, 4],
    }
    base.update(kw)
    return base


def _review(**kw):
    base = {
        "mode_id": "HOK-MODE-JGL-T0400-01",
        "premise": "visible",
        "matched": "no",
        "frames": [1, 2],
        "did": "在刷野",
        "why": "第 1 帧小地图在红 buff 野区",
    }
    base.update(kw)
    return base


def test_a_confirmed_deviation_survives():
    """前提被画面证实、做的确实不是这一条 → 算偏离。"""
    items, _ = vlm_review.parse_review(
        {"review": [_review()]}, [_entry()], 4,
        [{"kind": "positioning_action", "value": "pull_back"}])
    assert len(items) == 1
    assert items[0]["verdict"] == vlm_review.VERDICT_DEVIATION
    assert items[0]["mode_id"] == "HOK-MODE-JGL-T0400-01"


def test_premise_not_visible_is_not_a_correction():
    """前提在画面里看不出来 → 那一刻玩家做的可能就是对的，不纠正。

    这是用户那条硬要求：「一定要指出改正之后一定会进步的操作模式」——
    前提都不成立，怎么知道改了会更好。
    """
    for premise in ("not_visible", "unknown"):
        items, _ = vlm_review.parse_review(
            {"review": [_review(premise=premise)]}, [_entry()], 4,
            [{"kind": "positioning_action", "value": "pull_back"}])
        assert items[0]["verdict"] == vlm_review.VERDICT_UNCLEAR


def test_doing_what_the_node_endorses_is_not_a_correction():
    """玩家做了这个节点认可的另一件事 → 不纠正他。

    攻略一个时段给好几套打法，玩家做了其中另一件时，
    他是在按攻略做事，只是跟规则库写的那一条不一样。
    """
    items, _ = vlm_review.parse_review(
        {"review": [_review()]}, [_entry()], 4,
        [{"kind": "positioning_action", "value": "ward_vision"}])
    assert items[0]["verdict"] != vlm_review.VERDICT_DEVIATION


def test_doing_what_this_rule_asks_is_never_a_correction():
    """这一条要求的事玩家做到了 → 这一条他没错。"""
    items, _ = vlm_review.parse_review(
        {"review": [_review()]}, [_entry()], 4,
        [{"kind": "positioning_action", "value": "push_wave"}])
    assert items[0]["verdict"] == vlm_review.VERDICT_OK


def test_subject_must_match():
    """适用时机不命中 → 玩家这一帧做的不是这条规则管的事，不纠正。"""
    items, _ = vlm_review.parse_review(
        {"review": [_review()]}, [_entry()], 4,
        [{"kind": "objective_action", "value": "farm_jungle"}])
    assert items[0]["verdict"] == vlm_review.VERDICT_UNCLEAR


def test_unknown_mode_is_dropped():
    """模型引用了清单里没有的模式 → 丢掉（不认识的东西不能进报告）。"""
    items, notes = vlm_review.parse_review(
        {"review": [_review(mode_id="HOK-MODE-NOPE")]}, [_entry()], 4,
        [{"kind": "positioning_action", "value": "pull_back"}])
    assert not items
    assert notes


def test_a_deviation_without_a_real_frame_is_dropped():
    """帧号对不上 → 这条结论没法在原片上验证，不给。"""
    items, notes = vlm_review.parse_review(
        {"review": [_review(frames=[9, 10])]}, [_entry()], 4,
        [{"kind": "positioning_action", "value": "pull_back"}])
    assert not items
    assert notes


def test_frames_out_of_range_are_dropped_but_the_finding_stands():
    """帧号超范围的那几个丢掉，剩下的是真的就留着。"""
    items, _ = vlm_review.parse_review(
        {"review": [_review(frames=[1, 99])]}, [_entry()], 4,
        [{"kind": "positioning_action", "value": "pull_back"}])
    assert items[0]["frames"] == [1]


def test_a_blind_window_cannot_produce_a_correction():
    """这一窗什么动作都没读出来 → 判不了，「看不出来」不等于「没做」。"""
    items, _ = vlm_review.parse_review(
        {"review": [_review()]}, [_entry()], 4, [])
    assert items[0]["verdict"] == vlm_review.VERDICT_UNCLEAR


def test_node_gap_only_when_nothing_endorsed_was_done():
    """「整段没做」只在玩家真的没做节点认可的事时才出。"""
    node = {"lane": "jungle", "frames": [1], "did": "整段在刷野",
            "why": "这一段做的不在这个时段的安排里"}
    options = [_entry(), _entry(id="HOK-MODE-JGL-T0400-02",
                                expect_fact="objective_action",
                                expect_values=["contest_dragon"],
                                subject={"fact": "objective_action",
                                         "values": ["contest_dragon"]})]
    # 玩家做的正是节点认可的事 → 不出「整段没做」。
    items, notes = vlm_review.parse_review(
        {"node_gap": node}, options, 4,
        [{"kind": "positioning_action", "value": "push_wave"}])
    assert not items
    assert notes
    # 玩家做的是节点不认可的事 → 出。
    items, _ = vlm_review.parse_review(
        {"node_gap": node}, options, 4,
        [{"kind": "objective_action", "value": "farm_jungle"}])
    assert len(items) == 1
    assert items[0]["kind"] == vlm_review.KIND_NODE
    assert items[0]["verdict"] == vlm_review.VERDICT_NEVER


def test_node_gap_needs_more_than_one_option():
    """只有一条路可选时不叫「菜单」。"""
    node = {"lane": "jungle", "frames": [1], "did": "在刷野", "why": "x"}
    items, notes = vlm_review.parse_review(
        {"node_gap": node}, [_entry()], 4,
        [{"kind": "objective_action", "value": "farm_jungle"}])
    assert not items
    assert notes
