"""把一局复盘中的偏离压成下一局可执行的训练清单。"""
from __future__ import annotations

from typing import Any, Dict

_WEIGHT = {"high": 3, "medium": 2, "low": 1}


def build_training_plan(match: Any, completed_rule_ids: set[str] | None = None) -> Dict[str, Any]:
    """一局最多三项练习，防止复盘变成无法执行的待办堆。"""
    completed_rule_ids = completed_rule_ids or set()
    snapshots = list(getattr(match, "analyses", []) or [])
    if not snapshots:
        return {"status": "waiting", "tasks": [], "message": "这局还没有可用复盘；上传并分析录屏后再生成训练清单。"}
    chosen: Dict[str, Dict[str, Any]] = {}
    for gap in list((snapshots[0].get_detail() or {}).get("gaps") or []):
        rule_id = str(gap.get("rule_id") or "")
        if not rule_id:
            continue
        score = _WEIGHT.get(str(gap.get("severity") or "medium"), 2)
        if rule_id not in chosen or score > chosen[rule_id]["score"]:
            chosen[rule_id] = {"rule_id": rule_id, "title": str(gap.get("rule_title") or "本局关键操作"), "when": str(gap.get("time_label") or gap.get("round_label") or "关键时段"), "action": str(gap.get("expected_text") or gap.get("action") or gap.get("hint") or "按复盘建议执行"), "why": str(gap.get("why_text") or gap.get("hint") or "本局在这里出现偏离"), "severity": str(gap.get("severity") or "medium"), "score": score}
    tasks = sorted(chosen.values(), key=lambda task: (-task["score"], task["title"]))[:3]
    for index, task in enumerate(tasks, 1):
        task["order"] = index
        task["completed"] = task["rule_id"] in completed_rule_ids
        task.pop("score", None)
    return {"status": "ready" if tasks else "clear", "tasks": tasks, "message": "下一局只练这几件事。完成一项比同时记住十项更重要。" if tasks else "本局没有可转成训练任务的偏离。"}
