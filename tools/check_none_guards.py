"""检查规则库里的 ``none`` 护栏用到了哪些事实。

为什么关心这个
--------------
匹配器一旦改成「这一帧里做了符合要求的动作就算符合」（而不是只看最后一个
动作），那么「同一格子里两条模式互相指控」的判定口径就会变：

    规则 A 期望 objective_action ∈ {take_dragon}
    规则 B 带 none 护栏 objective_action != take_dragon

A、B 若在同一格子里同时触发，B 的护栏必须**按这一帧里的全部动作**判断，
而不是只看最后一个动作 —— 否则 A 判「做到了」、B 判「触发」，
玩家被两条同时指控。

这个脚本把库里的 none 护栏事实列出来，看这件事在真实库里到底存不存在。
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

sys.stdout.reconfigure(encoding="utf-8")


def main(argv: List[str]) -> int:
    game = argv[0] if argv else "hok"
    version = argv[1] if len(argv) > 1 else "S45"

    from app.db import session_scope
    from app.rules.store import get_rules

    with session_scope() as s:
        rules = get_rules(s, game, version)

    none_facts: Dict[str, List[str]] = {}
    any_facts: Dict[str, List[str]] = {}
    for r in rules:
        tr = r.trigger
        for c in (getattr(tr, "none", []) or []):
            none_facts.setdefault(c.fact, []).append(r.id)
        for c in (getattr(tr, "any", []) or []):
            any_facts.setdefault(c.fact, []).append(r.id)

    print(f"{game} {version}：{len(rules)} 条")
    print(f"\nnone 护栏用到的事实：{sorted(none_facts) or '（无）'}")
    for fact, ids in sorted(none_facts.items()):
        print(f"  {fact}: {ids}")
    print(f"\nany 组用到的事实：{sorted(any_facts) or '（无）'}")
    for fact, ids in sorted(any_facts.items()):
        print(f"  {fact}: {ids}")

    # 同一格里期望同一动作事实的模式对
    from app.core.lane_analysis import mode_cell

    cells: Dict[tuple, List] = {}
    for r in rules:
        cell = mode_cell(r)
        if cell is None:
            continue
        lane, window = cell
        lo = window[0] if window else 0
        phase = "early" if lo < 5 else "mid" if lo < 11 else "late"
        cells.setdefault((lane, phase), []).append(r)

    print("\n同一格子里期望同一动作事实的模式对：")
    found = False
    for (lane, phase), group in sorted(cells.items()):
        by_fact: Dict[str, List] = {}
        for r in group:
            by_fact.setdefault(r.expect_fact, []).append(r)
        for fact, same in by_fact.items():
            if len(same) < 2:
                continue
            found = True
            print(f"  {lane}·{phase}  动作事实 {fact}")
            for r in same:
                v = r.expect_value
                v = v if isinstance(v, list) else [v]
                print(f"      {r.id:<22} 期望 {v}")
    if not found:
        print("  （没有）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
