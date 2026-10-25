"""列出某个游戏某个版本的全部规则：触发条件 + 期望动作。

给「造一局能触发规则的对局」用 —— 手工读 47 条规则的 JSON 太慢，
而且容易看漏 `none` 护栏。

用法::

    .\\.venv\\Scripts\\python.exe tools\\list_rules.py hok S45
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

sys.stdout.reconfigure(encoding="utf-8")


def main(argv: List[str]) -> int:
    game = argv[1] if len(argv) > 1 else "hok"
    version = argv[2] if len(argv) > 2 else "S45"

    from app.db import session_scope
    from app.rules.store import get_rules

    with session_scope() as s:
        rules = get_rules(s, game, version)

    print(f"{game} {version}：{len(rules)} 条\n")
    for r in rules:
        tr = r.trigger
        allc = list(getattr(tr, "all", []) or [])
        anyc = list(getattr(tr, "any", []) or [])
        nonec = list(getattr(tr, "none", []) or [])
        parts: List[str] = []
        for c in allc:
            parts.append(f"{c.fact}{c.op}{c.value}")
        for c in anyc:
            parts.append(f"ANY({c.fact}{c.op}{c.value})")
        for c in nonec:
            parts.append(f"NOT({c.fact}{c.op}{c.value})")
        exp = r.expect_value
        exp = exp if isinstance(exp, list) else [exp]
        print(f"{r.id}")
        print(f"  标题   {r.title}")
        print(f"  触发   {' AND '.join(parts)}")
        print(f"  期望   {r.expect_fact} = {exp}   严重度 {r.severity}")
        print(f"  改法   {r.deviation_hint}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
