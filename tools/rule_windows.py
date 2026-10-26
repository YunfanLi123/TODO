"""统计规则库里的**时间窗**与**用到的事实**——这决定「综合分析」的分组单元。"""
from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]

files = ["app/seed/hok_modes.json", "app/seed/hok_hero_modes.json",
         "app/seed/hok_bp_modes.json"]
wins = collections.Counter()
facts = collections.Counter()
n = 0
per_file = {}
for rel in files:
    p = ROOT / rel
    if not p.exists():
        continue
    items = json.loads(p.read_text(encoding="utf-8"))
    per_file[rel] = len(items)
    for m in items:
        n += 1
        trig = m.get("trigger") or {}
        conds = list(trig.get("all") or []) + list(trig.get("any") or [])
        for c in conds:
            f = c.get("fact")
            facts[f] += 1
            if f == "time_min" and c.get("op") == "between":
                v = c.get("value")
                wins[(v[0], v[1]) if isinstance(v, (list, tuple)) else str(v)] += 1

print("各文件条数：", per_file, " 合计", n)
print("\n触发条件里用到的事实：")
for k, v in facts.most_common():
    print(f"  {k:<24} {v}")
print(f"\n不同的时间窗：{len(wins)} 种")
for k, v in sorted(wins.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    print(f"  {k[0]:>3} ~ {k[1]:<3} 分钟   {v} 条规则")
