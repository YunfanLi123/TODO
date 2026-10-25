"""内核按「时间窗内的若干帧一起判」的口径归因。

盯两件事：

1. **一次性口径的规则，每条只出一条归因记录**。
   切片是按规则密度切的（密的地方多切几刀），密的地方正是最容易
   重复判到同一条规则的地方。按帧记的话，同一个错会被放大成好几条，
   报告看着像玩家连犯了好几次。
2. **做到过就不记偏差** —— 报告不能一边说「做对了」一边挂着偏差。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.attribution import to_gaps  # noqa: E402
from app.core.matcher import Verdict  # noqa: E402


def _verdict(rule_id: str, status: str, label: str) -> Verdict:
    return Verdict(
        rule_id=rule_id,
        rule_title=f"规则 {rule_id}",
        category="打野·中期",
        severity="medium",
        status=status,
        round_label=label,
        evidence=[],
        expected={},
        actual={},
        context={},
        tags=[],
    )


def test_one_gap_per_once_rule():
    """同一条规则在多帧上都偏离 → 只出一条归因。"""
    verdicts = [
        _verdict("R1", "deviated", "第 4 分钟"),
        _verdict("R1", "deviated", "第 4 分钟"),
        _verdict("R1", "deviated", "第 4 分钟"),
    ]
    gaps = to_gaps(verdicts, once_rules={"R1"})
    assert len(gaps) == 1, f"同一个错记了 {len(gaps)} 条归因"


def test_per_frame_rules_keep_every_record():
    """按帧判的规则仍然逐帧记（那是它本来的口径）。"""
    verdicts = [
        _verdict("R2", "deviated", "第 4 分钟"),
        _verdict("R2", "deviated", "第 4 分钟"),
    ]
    gaps = to_gaps(verdicts, once_rules=set())
    assert len(gaps) == 2


def test_a_satisfied_rule_produces_no_gap():
    """做到过就不记偏差。"""
    verdicts = [
        _verdict("R3", "deviated", "第 4 分钟"),
        _verdict("R3", "satisfied", "第 4 分钟"),
    ]
    gaps = to_gaps(verdicts, skip_rules={"R3"}, once_rules={"R3"})
    assert not gaps
