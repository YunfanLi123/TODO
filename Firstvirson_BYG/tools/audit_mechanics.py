"""机制体检：扫一遍模式库，看还有没有**已移除**的机制混在规则里。

用法::

    python tools/audit_mechanics.py            # 扫种子库
    python tools/audit_mechanics.py --strict   # 有命中就退出码 1（给 CI 用）
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from hok_mechanics import (  # noqa: E402
    MECHANIC_FIXES,
    OBJECTIVE_MINUTES,
    REMOVED_MECHANICS,
    S45_MECHANICS,
    audit_rules,
    needs_review,
    revised_rules,
)

ROOT = Path(__file__).resolve().parent.parent
SEED = ROOT / "app" / "seed"
FILES = ("hok_modes.json", "hok_hero_modes.json", "hok_bp_modes.json")


def main() -> int:
    ap = argparse.ArgumentParser(description="模式库机制体检")
    ap.add_argument("--strict", action="store_true",
                    help="有命中就退出码 1")
    args = ap.parse_args()

    print("当前版本（S45）真实存在的地图目标：")
    for k, v in S45_MECHANICS.items():
        print(f"  {k:<16}{v}")
    print("\n已移除的机制：")
    for k, v in REMOVED_MECHANICS.items():
        print(f"  {k:<16}{v}")

    hits = []
    revised = []
    review = []
    per_file: Counter = Counter()
    for name in FILES:
        p = SEED / name
        if not p.exists():
            continue
        items = json.loads(p.read_text(encoding="utf-8"))
        h = audit_rules(items)
        per_file[name] = len(h)
        hits.extend(h)
        revised.extend(revised_rules(items))
        review.extend(needs_review(items))

    print(f"\n=== 体检结果：规则的可执行文本里还有 {len(hits)} 处已移除的机制")
    for k, v in per_file.items():
        print(f"  {k}: {v} 处")
    for rid, m, title in hits[:10]:
        print(f"  {rid}  {title}  ← {m}")

    print(f"\n=== 按当前版本修订过的规则：{len(revised)} 条")
    print("（原文引用保持逐字不变，供复核；可执行文本已换成当前版本的机制）")

    print(f"\n=== 需人工复核：{len(review)} 条")
    print("这些不是改词能救的 —— 原文的决策对象在那个时间窗里已经不存在了。")
    print(f"（当前版本河道目标刷新在第 {', '.join(str(m) for m in OBJECTIVE_MINUTES)} 分钟）")
    for rid, win, title in review:
        print(f"  {rid}  {win}  {title}")

    print(f"\n修订表 {len(MECHANIC_FIXES)} 条；"
          f"生成器会套用它（改映射表 → 重新生成，不手改生成产物）。")
    if hits and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
