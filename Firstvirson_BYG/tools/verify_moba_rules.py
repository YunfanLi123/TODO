import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient     # noqa: E402
from app.main import app                     # noqa: E402

FAIL = []


def check(ok, msg):
    print(("  [PASS] " if ok else "  [FAIL] ") + msg)
    if not ok:
        FAIL.append(msg)


GUIDE = """前期打野以刷野为主，尽快升到四级抓人。
前期中单优先清线，抢二级的节奏。
中期游走位要跟着打野入侵野区扩大优势。
中期要跟着节奏走，不要单独送。
后期视野最重要，开大龙团前必须把龙坑周围照亮。
刺客前期要gank，中期带线。
一般来说前期打野要注意别被反野。
"""

c = TestClient(app)


def _isolate():
    """独立数据库。

    之前这脚本直接用主库，于是上一轮跑剩下的脏规则（缺位置的那些）
    留在了库里，把后续断言全带偏了 —— 「射手也触发了规则」看着像
    位置过滤失效，其实是上一次的残留。验证脚本之间共享状态，
    失败原因会指向完全错误的地方。
    """
    import os

    from app import db as dbmod

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "verify_moba_rules.db",
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    dbmod._engine = None
    dbmod._SessionFactory = None
    dbmod.get_engine(path)


_isolate()
c = TestClient(app)

print("\n[1] MOBA 抽取")
r = c.post("/api/guides/moba/extract", json={
    "text": GUIDE, "game": "hok", "version": "S38",
    "guide_title": "我的运营笔记", "preview": True,
})
check(r.status_code == 200, f"接口 200（{r.status_code}）")
d = r.json()
print(f"      {d['message']}")
for x in d["candidates"]:
    print(f"      [{x['position_label']}·{x['phase_label']}] "
          f"{x['title'][:24]}  -> {x['time_window']} {x['action_value']}")
check(len(d["candidates"]) >= 4, f"抽出 {len(d['candidates'])} 条（≥4）")
check(len(d["skipped"]) >= 2, f"说明 {len(d['skipped'])} 条未抽及原因")

print("\n[2] 每条都带位置 + 阶段 + 时间窗")
for x in d["candidates"]:
    if not (x["position"] and x["phase"] and len(x["time_window"]) == 2):
        check(False, f"{x['title'][:20]} 缺维度")
        break
else:
    check(True, "全部候选维度完整")

print("\n[3] 确认入库")
sid = {x["title"]: x for x in d["candidates"]}
pick = list(sid.values())[:3]
r = c.post("/api/guides/moba/confirm", json={
    "game": "hok", "version": "S38", "guide_title": "我的运营笔记",
    "candidates": [x["trigger"] and {
        "title": x["title"],
        "position": x["position"], "position_label": x["position_label"],
        "phase": x["phase"], "phase_label": x["phase_label"],
        "action_fact": x["action_fact"], "action_value": x["action_value"],
        "severity": "medium",
        "trigger": x["trigger"], "expect": x["expect"],
        "source_quote": x["source_quote"],
    } for x in pick],
    "ids": d["ids"],
})
check(r.status_code == 200, f"确认接口 200（{r.status_code}）")
print("      " + json.dumps(r.json(), ensure_ascii=False)[:200])

print("\n[4] 缺位置的 MOBA 规则必须被拒")
r = c.post("/api/guides/moba/confirm", json={
    "game": "hok", "version": "S38",
    "candidates": [{
        "title": "坏规则：没有位置",
        "trigger": {"all": [{"fact": "gold_diff", "op": ">=", "value": 1}],
                     "any": [], "none": []},
        "expect": {"fact": "objective_action", "op": "in", "value": ["farm_jungle"]},
    }],
})
out = r.json()
check(out["imported"] == [] and out.get("skipped_no_position") == 1,
      f"无位置的规则被拒（skipped_no_position="
      f"{out.get('skipped_no_position')}）")

print("\n[5] 抽出来的规则真的会被对局触发")
base_rounds = [{
    "stage": 1, "phase": "early",
    "facts": {"time_min": 3, "gold": 800, "gold_diff": 200,
              "level": 6, "tower_diff": 0, "kill_diff": 1,
              "pressure": "low", "has_ult_ready": False,
              "is_decision_point": False},
    "actions": [{"kind": "objective_action", "value": "invade_jungle",
                 "confidence": 0.95}],
    "missing_facts": [],
}]


def _enter(match_id, position):
    r = c.post("/api/matches", json={
        "match_id": match_id, "game": "hok", "game_version": "S38",
        "player": {"nickname": "示例玩家", "position": position},
        "outcome": {"result": "win"}, "played_at": "2026-10-18 21:00",
        "rounds": base_rounds,
    })
    return r


r = _enter("hok-moba-rule", "打野")
check(r.status_code == 200, f"打野局录入（{r.status_code}）")
jungle_ids = set()
if r.status_code == 200:
    st = r.json()["stats"]
    print(f"      触发 {st['triggered']} 次 · 偏离 {st['deviated']} 次")
    check(st["triggered"] > 0, "前期打野规则在 3 分钟这局触发了")
    rep = c.get("/api/matches/hok-moba-rule/report.json").json()
    jungle_ids = {g["rule_id"] for g in rep["gaps"]}
    for g in rep["gaps"][:2]:
        print(f"      {g['round_label']}  {g['rule_id']}  {g['rule_title'][:28]}")
        print(f"        该做 {g['expected_text']} | 实际 {g['actual_text']}")

print("\n[6] 位置不同则同一行为不触发同一条规则")
r = _enter("hok-moba-carry", "射手")
check(r.status_code == 200, f"射手局录入（{r.status_code}）")
if r.status_code == 200:
    rep = c.get("/api/matches/hok-moba-carry/report.json").json()
    carry_ids = {g["rule_id"] for g in rep["gaps"]}
    shared = jungle_ids & carry_ids
    check(not shared,
          f"射手局没有触发打野专属规则"
          + (f"（重合 {shared}）" if shared else ""))

print("\n" + "=" * 60)
if FAIL:
    print(f"失败 {len(FAIL)} 项：")
    for f in FAIL:
        print("  - " + f)
    raise SystemExit(1)
print("MOBA 操作模式抽取验证全部通过")
raise SystemExit(0)
