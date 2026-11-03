"""对局状态融合与出装方向建议。

这个模块不把看不清的 HUD 猜成事实：字段必须附带置信度，低于门槛只保留
“待确认”。输入既可来自 VLM，也可来自记分板/装备栏图标识别器。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

ACCEPT = 0.70


def normalize(raw: Dict[str, Any]) -> Dict[str, Any]:
    confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
    player = dict(raw.get("player") or {})
    matchup = dict(raw.get("matchup") or {})
    build = dict(raw.get("build") or {})
    items = [str(item).strip() for item in (player.get("items") or []) if str(item).strip()]
    roster = []
    for member in build.get("enemy_roster") or []:
        if not isinstance(member, dict):
            continue
        hero = str(member.get("hero") or "").strip()
        evidence = str(member.get("evidence") or "").strip()
        try:
            member_confidence = float(member.get("confidence", 0.0))
        except (TypeError, ValueError):
            member_confidence = 0.0
        if hero and hero.lower() != "unknown" and evidence and member_confidence >= ACCEPT:
            roster.append({"hero": hero, "evidence": evidence,
                           "confidence": min(1.0, member_confidence)})
    traits = build.get("enemy_traits") if isinstance(build.get("enemy_traits"), dict) else {}
    return {"time_min": max(0, int(raw.get("time_min", 0))), "confidence": confidence,
            "player": {"hero": str(player.get("hero") or ""), "level": player.get("level"), "gold": player.get("gold"), "hp_ratio": player.get("hp_ratio"), "mana_ratio": player.get("mana_ratio"), "ult_ready": player.get("ult_ready"), "items": items},
            "matchup": {"hero": str(matchup.get("hero") or ""), "level": matchup.get("level"), "gold": matchup.get("gold"), "kills": matchup.get("kills"), "deaths": matchup.get("deaths"), "assists": matchup.get("assists"), "damage_type": str(matchup.get("damage_type") or "unknown"), "control_heavy": bool(matchup.get("control_heavy", False))},
            "build": {"enemy_heal_heavy": bool(build.get("enemy_heal_heavy", traits.get("healing_heavy", False))), "enemy_tank_heavy": bool(build.get("enemy_tank_heavy", traits.get("tank_heavy", False))), "behind": bool(build.get("behind", False)), "enemy_roster": roster, "enemy_traits": {"control_heavy": bool(traits.get("control_heavy", False)), "magic_heavy": bool(traits.get("magic_heavy", False))}, "purchase_events": [dict(event) for event in (build.get("purchase_events") or []) if isinstance(event, dict)]}}


def consensus(snapshots: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """同一分钟多帧投票；只有至少两帧同意的英雄/装备才对外给结论。"""
    rows = [normalize(row) for row in snapshots if float(row.get("confidence", 0)) >= ACCEPT]
    if not rows:
        return {"status": "uncertain", "message": "画面证据不足，未对状态作判断。"}
    rows.sort(key=lambda row: row["confidence"], reverse=True)
    best = rows[0]
    # 专项购买扫描常只带物品；不能因为它置信度更高，就把常规 HUD 已读到的
    # 等级、金币等字段清空。每个字段独立取最可信的非空证据。
    for group in ("player", "matchup"):
        for field in best[group]:
            if best[group].get(field) not in (None, "", [], "unknown"):
                continue
            for row in rows[1:]:
                value = row[group].get(field)
                if value not in (None, "", [], "unknown"):
                    best[group][field] = value
                    break
    def vote(path: str) -> str:
        values = []
        for row in rows:
            value = row["player" if path == "player" else "matchup"].get("hero")
            if value:
                values.append(value)
        return max(set(values), key=values.count) if values else ""
    best["player"]["hero"] = vote("player")
    best["matchup"]["hero"] = vote("matchup")
    purchases: List[Dict[str, Any]] = []
    seen = set()
    for row in sorted(rows, key=lambda row: row["time_min"]):
        for event in row["build"].get("purchase_events") or []:
            item = str(event.get("item") or "").strip()
            key = (row["time_min"], item)
            if not item or key in seen:
                continue
            seen.add(key)
            purchases.append({"time_min": row["time_min"], "item": item,
                              "evidence": str(event.get("evidence") or ""),
                              "confidence": float(event.get("confidence", row["confidence"]))})
    best["build"]["purchase_events"] = purchases
    best["status"] = "ready"
    best["samples"] = len(rows)
    return best


def recommend(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    if snapshot.get("status") != "ready":
        return {"status": "uncertain", "direction": "待确认", "reason": "尚未读到可靠的英雄、对位或装备证据。"}
    matchup, build = snapshot["matchup"], snapshot["build"]
    if build.get("behind"):
        return {"status": "ready", "direction": "性价比与保命", "reason": "当前处于劣势，优先保证成型速度和团战存活。"}
    if matchup.get("control_heavy") or build.get("enemy_traits", {}).get("control_heavy"):
        return {"status": "ready", "direction": "韧性/解控", "reason": "对位控制较多，先提高被控后的存活与操作空间。"}
    if matchup.get("damage_type") == "magic":
        return {"status": "ready", "direction": "魔抗/法术防御", "reason": "对位以法术伤害为主，防御收益更直接。"}
    if build.get("enemy_heal_heavy"):
        return {"status": "ready", "direction": "重伤", "reason": "敌方续航明显，优先补能压制回复的装备。"}
    if build.get("enemy_tank_heavy"):
        return {"status": "ready", "direction": "穿透", "reason": "敌方前排较多，穿透类装备能提高有效伤害。"}
    return {"status": "ready", "direction": "核心输出", "reason": "暂无需要优先应对的对位信号，按英雄核心成装推进。"}
