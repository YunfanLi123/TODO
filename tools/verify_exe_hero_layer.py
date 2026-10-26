"""对**正在跑的 exe** 做一遍闭合链路验证。

验的是打包后的产物，不是源码跑起来的 dev 服 —— 打包会漏文件
（种子规则库漏进去，exe 照样起来，只是库里是空的），
所以这一遍必须打在 exe 上。

用法：python tools/verify_exe_hero_layer.py http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"

FAILS = []


def check(cond, msg):
    print(("  [PASS] " if cond else "  [FAIL] ") + msg)
    if not cond:
        FAILS.append(msg)


def get(path):
    with urllib.request.urlopen(f"{BASE}{path}", timeout=60) as r:
        return r.status, r.read().decode("utf-8", "replace")


def post(path, payload=None):
    body = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}{path}", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def section(title):
    print()
    print("-" * 58)
    print(title)
    print("-" * 58)


section("[1] 规则库")
code, body = post("/api/games/hok/rules/seed")
check(code == 200, f"载入规则库（{code}）")
check(body.get("accepted_count") == 1681,
      f"载入 1681 条（{body.get('accepted_count')}）")
check(not body.get("rejected"), "没有规则被拒")

code, txt = get("/api/games/hok/rules?version=S45")
rules = json.loads(txt)["rules"]
hero = [r for r in rules if r["id"].startswith("HOK-HERO-")]
lane = [r for r in rules if r["id"].startswith("HOK-MODE-")]
check(len(hero) == 1596, f"英雄层 1596 条（{len(hero)}）")
check(len(lane) == 85, f"分路层 85 条（{len(lane)}）")
heroes = {c["value"] for r in hero for c in r["trigger"]["all"] if c["fact"] == "my_hero"}
check(len(heroes) == 123, f"覆盖 123 个英雄（{len(heroes)}）")

section("[2] 模式库页面")
code, txt = get("/rules?lane=side")
check(code == 200, f"分路页可打开（{code}）")
check("对抗路" in txt, "分路页显示对抗路")
hero_links = re.findall(r'href="/rules\?lane=side&hero=([^"]+)"', txt)
check(len(hero_links) == 33, f"对抗路页列出 33 个英雄（{len(hero_links)}）")
code, txt = get("/rules?lane=jungle&hero=" + urllib.request.quote("赵云"))
check(code == 200, f"赵云页可打开（{code}）")
check("该英雄前中后期操作模式" in txt, "赵云页有英雄小节")
check("打野-赵云-前期-1" in txt, "赵云页模式名带英雄")
check("打野-澜-" not in txt, "赵云页里没有别的英雄")

section("[3] 灌两局 demo 并验证判定")
code, out = post("/api/matches?player_id=local", {
    "match_id": "exe-check-hero", "game": "hok", "game_version": "S45",
    "player": {"nickname": "示例玩家", "position": "jungle", "hero": "赵云"},
    "outcome": {"result": "win"}, "played_at": "2026-10-21 23:40",
    "rounds": [
        {"stage": 0, "phase": 1, "time_min": 2,
         "facts": {"gold": 1800, "gold_diff": 0, "level": 4, "tower_diff": 0,
                   "kill_diff": 0},
         "actions": [{"kind": "objective_action", "value": "farm_jungle",
                      "label": "刷野补经济", "confidence": 1.0}],
         "missing_facts": []},
    ],
})
check(code == 200, f"录入成功（{code}）")
ids = {f["mode_id"] for f in out.get("lanes", {}).get("findings", [])}
check(bool(ids), f"判出了结论（{len(ids)} 条）")
check(all(i.startswith("HOK-HERO-") for i in ids),
      f"选了赵云只判英雄层：{sorted(ids)}")
mid = out.get("match_id")

section("[4] 报告页四部分")
code, html = get(f"/api/matches/{mid}/report")
check(code == 200, f"报告页可打开（{code}）")
visible = re.sub(r"<script.*?</script>", "", html, flags=re.S)
visible = re.sub(r"<style.*?</style>", "", visible, flags=re.S)
check("{{" not in html, "无未渲染占位符")
# 报告正文只有「操作模式分析」一节，每条四部分：时间区间 / 原视频片段 /
# 原视频操作描述 / 建议操作描述 + 建议理由。标签跟着改版走，
# 这里断言的是**现在这一版的标签**，不是上一版的。
for word in ("对局时间区间", "原视频被建议部分片段",
             "原视频操作的文字描述", "根据模式库建议操作的描述",
             "建议理由"):
    check(word in visible, f"报告里有「{word}」")
check("原文前提：" not in visible, "没有「原文前提：」这种元叙述")
check("第 2 分钟" in visible, "有准确时间节点")
check("赵云" in visible, "出现英雄层模式")

section("[5] 对局列表页")
code, txt = get("/matches")
check(code == 200, f"对局列表可打开（{code}）")
check("exe-check-hero" in txt or "打野" in txt, "列表有对局")

print()
if not FAILS:
    print("exe 闭合链路全部通过")
    raise SystemExit(0)
print(f"失败 {len(FAILS)} 项：")
for f in FAILS:
    print("  -", f)
raise SystemExit(1)
