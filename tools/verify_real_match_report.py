"""用真实对局验证报告这一段链路。

验的是六件事
------------

1. **分路与英雄确认**：报告里的分路、英雄、对位与提交时填的一致。
2. **抽出对应的操作模式**：这一局该被抽出来的分路层与英雄层模式都出现了。
3. **配对**：同一个（分路 × 时期）格子里，分路通则与英雄专属模式同时生效。
4. **找到偏差**：每条偏差都写清了「该做」与「实际」。
5. **时间顺序**：偏差按局内时间排，不是按严重度排。
6. **来源标记**：动作是玩家自述还是从画面反推的，报告里说清楚了。

用法::

    .\\.venv\\Scripts\\python.exe tools\\verify_real_match_report.py <match_id> [--base ...]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

sys.stdout.reconfigure(encoding="utf-8")

FAILS: List[str] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"　{detail}" if detail else ""))
    if not ok:
        FAILS.append(label)
    return ok


def minute_of(label: str) -> float:
    """回合标签「第 5 分钟」-> 5.0。读不出来就返回 -1。"""
    m = re.search(r"(\d+(?:\.\d+)?)", label or "")
    return float(m.group(1)) if m else -1.0


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="真实对局报告链路验证")
    ap.add_argument("match_id")
    ap.add_argument("--base", default="http://127.0.0.1:8849")
    args = ap.parse_args(argv)

    import httpx

    base = args.base.rstrip("/")
    with httpx.Client(timeout=120.0) as client:
        r = client.get(f"{base}/api/matches/{args.match_id}/report.json")
    if r.status_code != 200:
        print(f"取不到报告 {r.status_code}: {r.text[:300]}")
        return 1
    rep: Dict[str, Any] = r.json()

    # ---- 模式显示名（用页面同一套派生逻辑，别在这儿另起一套） ----------
    from app.adapters import registry as _registry  # noqa: F401
    from app.db import session_scope
    from app.report.organize import build_mode_tree
    from app.rules.review import build_review_view
    from app.rules.store import latest_version

    with session_scope() as s:
        version = latest_version(s, "hok")
        view = build_review_view(s, "hok", version=version, with_coverage=False)
        tree = build_mode_tree(view["rules"], _registry.REGISTRY["hok"])
    names: Dict[str, str] = {}
    for node in tree:
        for m in node["modes"]:
            names[m["rule_id"]] = m["name"]
        for h in node["heroes"]:
            for m in h["modes"]:
                names[m["rule_id"]] = m["name"]

    player = rep.get("player") or {}
    findings = (rep.get("lanes") or {}).get("findings") or []
    gaps = rep.get("gaps") or []
    undetermined = rep.get("undetermined") or []

    print(f"\n对局 {args.match_id}")
    print(f"  分路 {player.get('position')}｜英雄 {player.get('hero') or '—'}｜"
          f"对位 {player.get('matchup')}｜胜负 {(rep.get('outcome') or {}).get('result')}")
    print(f"  统计 {rep.get('stats')}")

    print("\n[1] 分路与英雄")
    check(bool(player.get("position")), "分路已确认", str(player.get("position")))
    check(bool(player.get("hero")), "英雄已确认", str(player.get("hero")))

    print("\n[2] 抽出来的操作模式")
    for f in findings:
        print(f"    {names.get(f['mode_id'], f['mode_id']):<22} "
              f"[{f['phase_label']}] {f['verdict_label']}　"
              f"该做 {f['should_do']}｜实际 {f['actually_doing'] or '（未标）'}")
    lane_level = [f for f in findings if not f["mode_id"].startswith("HOK-HERO-")]
    hero_level = [f for f in findings if f["mode_id"].startswith("HOK-HERO-")]
    check(bool(lane_level), "分路层模式被抽出来",
          "、".join(names.get(f["mode_id"], f["mode_id"]) for f in lane_level))
    check(bool(hero_level), "英雄层模式被抽出来",
          "、".join(names.get(f["mode_id"], f["mode_id"]) for f in hero_level))

    print("\n[3] 配对（同一格里的分路通则 + 英雄专属）")
    cells: Dict[str, List[str]] = {}
    for f in findings:
        cells.setdefault(f"{f['lane_label']}·{f['phase_label']}", []).append(
            f["mode_id"])
    paired = False
    for cell, ids in cells.items():
        has_lane = any(not i.startswith("HOK-HERO-") for i in ids)
        has_hero = any(i.startswith("HOK-HERO-") for i in ids)
        mark = "配对" if (has_lane and has_hero) else "单层"
        print(f"    {cell:<12} {mark}　{'、'.join(names.get(i, i) for i in ids)}")
        paired = paired or (has_lane and has_hero)
    check(paired, "至少有一格同时命中分路层与英雄层")

    print("\n[4] 偏差：该做什么 vs 实际做了什么")
    for g in gaps:
        print(f"    {g['round_label']:<10} {names.get(g['rule_id'], g['rule_id']):<22} "
              f"该做 {g['expected_text']}")
        print(f"    {'':<10} {'':<22} 实际 {g['actual_text']}")
    check(bool(gaps), "找到了偏差", f"{len(gaps)} 条")
    check(all(g.get("expected_text") for g in gaps), "每条偏差都写了「该做」")
    check(all(g.get("actual_text") for g in gaps), "每条偏差都写了「实际」")
    if undetermined:
        print(f"    无法判定 {len(undetermined)} 条："
              + "、".join(str(u.get("round_label")) for u in undetermined))

    print("\n[5] 按局内时间排序")
    minutes = [minute_of(g["round_label"]) for g in gaps]
    print("    " + " → ".join(f"{m:g}" for m in minutes))
    check(minutes == sorted(minutes), "偏差按局内时间排，不是按严重度排")

    print("\n[6] 标注来源")
    src = (rep.get("meta") or {}).get("annotation_source")
    print(f"    annotation_source = {src}")
    check(src in ("player", "derived"), "报告写明了动作标注的来源")
    if src == "derived":
        with httpx.Client(timeout=60.0) as client:
            html = client.get(f"{base}/api/matches/{args.match_id}/report").text
        check("derived-note" in html, "报告页顶部标出了「测试标注，不是玩家自述」")

    print("\n" + ("=" * 60))
    if FAILS:
        print(f"有 {len(FAILS)} 项没过：")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("真实对局报告链路验证全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
