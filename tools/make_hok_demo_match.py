"""造一局 S45 的打野对局，用来验证报告页与标注页的渲染。

为什么需要它
------------
报告页只有**有结论**的时候才看得出配色对不对 —— 空报告是一片白，
看不出层级、间距、状态色。而真实录屏那一局要等扫描 + 标注十几分钟，
等不起也不方便反复调样式。

这局数据是**手工构造**的：分路、英雄、对位、每分钟的动作都是照着
S45 规则库里打野那几条模式填的，因此能同时造出「做对了」和「做反了」
两种结局，报告里两类卡片都会出现。

用法::

    .\\.venv\\Scripts\\python.exe tools\\make_hok_demo_match.py [--base http://127.0.0.1:8849]
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List

_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(
    __import__("os").path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

sys.stdout.reconfigure(encoding="utf-8")

#: 每分钟的动作。键是局内分钟，值是这一分钟做的事。
#:
#: 刻意做成「大部分对、偶尔反」—— 全对报告里没有红色状态卡，
#: 全反也没有绿色，两种都验不到。
ACTIONS: Dict[int, List[str]] = {
    0:  ["farm_jungle"],
    1:  ["farm_jungle"],
    2:  ["farm_jungle", "clear_vision_sprite"],
    3:  ["push_wave"],                 # 前期跑去推线 —— 偏离
    4:  ["farm_jungle"],
    5:  ["take_dragon"],
    6:  ["take_dragon"],
    7:  ["farm_jungle"],               # 中期不控龙去刷野 —— 偏离
    8:  ["contest_dragon"],
    9:  ["take_dragon"],
    10: ["farm_jungle"],               # 又一次偏离
    11: ["split_push"],
    12: ["take_dragon"],
    13: ["siege"],
}

#: 局面数值。经济差后期转正，触发「后期顺风」那几条模式。
FACTS: Dict[int, Dict[str, int]] = {
    m: {
        "gold": 2000 + m * 1400,
        "gold_diff": -600 + m * 260,
        "level": 2 + m,
        "tower_diff": -1 + m // 4,
        "kill_diff": -2 + m // 3,
    }
    for m in ACTIONS
}


def build_rounds() -> List[Dict[str, Any]]:
    rounds = []
    for m in sorted(ACTIONS):
        actions = [
            {"kind": "objective_action", "value": v, "label": "", "confidence": 1.0}
            for v in ACTIONS[m]
        ]
        rounds.append({
            # stage/phase 组成回合标签，必须互不相同 —— 视频链路用的是
            # 「第几个抽帧」，这里同理用分钟序号。
            "stage": m + 1,
            "phase": 1,
            "time_min": m,
            "facts": dict(FACTS[m]),
            "actions": actions,
        })
    return rounds


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="造一局 S45 打野对局")
    ap.add_argument("--base", default="http://127.0.0.1:8849")
    ap.add_argument("--version", default="S45")
    args = ap.parse_args(argv)

    import httpx

    payload = {
        "match_id": "demo-hok-jungle",
        "game": "hok",
        "game_version": args.version,
        "player": {
            "nickname": "示例玩家",
            "position": "jungle",
            "hero": "赵云",
            "matchup": "advantage",
        },
        "outcome": {"result": "win"},
        "played_at": "2026-10-19 21:30",
        "rounds": build_rounds(),
    }

    base = args.base.rstrip("/")
    with httpx.Client(timeout=120.0) as client:
        r = client.post(f"{base}/api/matches", json=payload)
    if r.status_code != 200:
        print(f"入库失败 {r.status_code}: {r.text[:600]}")
        return 1

    data = r.json()
    print(f"match_id   {data['match_id']}")
    print(f"统计       {data['stats']}")
    print(f"偏离条数   {data['gap_count']}")
    print(f"报告       {base}{data['report_url']}")
    lanes = data.get("lanes") or {}
    for lane in lanes.get("lanes") or []:
        print(f"  {lane['lane_label']}: {lane['problem_count']}/{lane['mode_count']} 条有问题")
    for f in lanes.get("findings") or []:
        if f["verdict"] == "ok":
            continue
        print(f"    [{f['verdict_label']}] {f['title']}")
        print(f"        该做 {f['should_do']}｜实际 {f['actually_doing'] or '（未标）'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
