"""机制体检的守卫：模式库里不许再有**已移除**的机制。

为什么要有这条测试
------------------
攻略原文是唯一真源，但原文可能记的是旧版本机制 —— 实测 S45 已经没有
「河道之灵」（S36 移除：中路删掉、对抗路换成空间之灵），
原文里却仍写着「1 分钟河蟹」「中路河道之灵」，
于是规则库照抄出 136 条规则，判一件游戏里不存在的东西：
玩家永远做不到，报告只会冤枉人「该控河蟹却没控」。

所以：改的是**映射表**（``tools/hok_mechanics.py`` 的修订表），
再重新生成种子库；这条测试守住结果。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(APP_ROOT / "tools"))

from hok_mechanics import (  # noqa: E402
    MECHANIC_FIXES,
    REMOVED_MECHANICS,
    S45_MECHANICS,
    apply,
    audit,
    audit_rules,
    revised_rules,
)

SEED = APP_ROOT / "app" / "seed"
FILES = ("hok_modes.json", "hok_hero_modes.json", "hok_bp_modes.json")


def _rules():
    for name in FILES:
        p = SEED / name
        if p.exists():
            for r in json.loads(p.read_text(encoding="utf-8")):
                yield name, r


def test_the_seed_library_has_no_removed_mechanics():
    """规则的可执行文本里不许再出现已移除的机制。"""
    hits = []
    for name, r in _rules():
        for rid, m, title in audit_rules([r]):
            hits.append(f"{name} {rid} {title} ← {m}")
    assert not hits, "模式库里还有已移除的机制：\n" + "\n".join(hits[:10])


def test_the_ground_truth_lists_the_current_mechanics():
    """地面真相不能空 —— 否则体检就是个摆设。"""
    assert S45_MECHANICS, "S45 机制表空了"
    for k in ("暴君", "主宰", "空间之灵", "暗影暴君", "风暴暴君"):
        assert k in S45_MECHANICS, f"机制表里少了 {k}"
    for k in ("河道之灵", "河蟹"):
        assert k in REMOVED_MECHANICS, f"已移除机制表里少了 {k}"


def test_apply_rewrites_the_old_mechanics():
    assert "河道之灵" not in apply("与中单、游走控下中路河道之灵")
    assert "河蟹" not in apply("拿到 1 分钟河蟹，建立前期经济与联动优势")
    assert "河道蟹" not in apply("往河道蟹或传送阵走")
    assert "蟹坑" not in apply("清完线走河道到蟹坑")


def test_apply_leaves_other_text_alone():
    src = "与中单、游走一起控中路河道之灵，中路河道之灵优先于常规野怪"
    out = apply(src)
    assert "河道目标" in out
    assert "常规野怪" in out


def test_audit_detects_the_old_words():
    assert audit("控中路河道之灵")
    assert audit("控河蟹")
    assert not audit("控暴君与主宰")


def test_the_revision_table_is_not_empty():
    assert len(MECHANIC_FIXES) >= 20


def test_some_rules_were_actually_revised():
    """修订得真的落到规则里了（不是表改了但没重新生成）。"""
    revised = []
    for _name, r in _rules():
        revised.extend(revised_rules([r]))
    assert len(revised) >= 100, f"只修订了 {len(revised)} 条，修订表可能没生效"


def test_rules_needing_human_review_are_flagged():
    """原文提到河道目标、但那个时间窗里当前版本没有目标的规则，必须被点名。

    打野 0:30~1:00 那条已经**整条删掉**了（它的决策对象是已移除的河道之灵），
    所以现在的库里不该再有需要人工复核的规则。
    """
    from hok_mechanics import needs_review

    review = []
    for _name, r in _rules():
        review.extend(needs_review([r]))
    assert not review, review


def test_needs_review_flags_a_window_without_any_objective():
    """机制本身要能识别「这个时间窗里没有河道目标」。"""
    from hok_mechanics import needs_review

    fake = {
        "id": "X-1",
        "title": "某条规则",
        "trigger": {"all": [{"fact": "time_min", "op": "between", "value": [1, 1]}]},
        "source": {"quoted": "动作：控中路河道之灵"},
    }
    assert needs_review([fake])
    ok = {
        "id": "X-2",
        "title": "某条规则",
        "trigger": {"all": [{"fact": "time_min", "op": "between", "value": [2, 3]}]},
        "source": {"quoted": "动作：控中路河道之灵"},
    }
    assert not needs_review([ok])
