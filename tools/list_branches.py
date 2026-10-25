"""列出决策树里某条分路的分支（前提链 / 动作 / 后果）。

用途：写映射表时需要分支的**精确**标识（前提链），
手抄容易抄错一个标点，而标识错了生成器会直接报「找不到这条分支」。

用法::

    python tools/list_branches.py mid
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from tools.hok_decision_tree import load_guide  # noqa: E402

GUIDE_PATH = os.path.join(
    ROOT, "app", "seed", "sources", "hok_s45_lane_decision_tree.txt"
)


def main() -> int:
    lane = sys.argv[1] if len(sys.argv) > 1 else ""
    tree = load_guide(GUIDE_PATH)
    for code, nodes in tree.lanes.items():
        if lane and code != lane:
            continue
        print(f"\n######## {code}")
        for node in nodes:
            print(f"  -- {node.header}")
            for b in node.branches:
                print(f"     key: {b.key}")
                print(f"     前提: {b.premise_text}")
                for a in b.actions:
                    print(f"     动作: {a}")
                print(f"     后果: {b.outcome_text()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
