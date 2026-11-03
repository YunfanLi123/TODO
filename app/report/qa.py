"""有边界的对局问答：只把本局已确认的复盘信息交给文本模型。"""
from __future__ import annotations

import json
from typing import Any, Dict


class MatchQaError(ValueError):
    pass


SYSTEM_PROMPT = """你是王者荣耀个人复盘教练。只能依据给出的对局摘要回答；不知道就明确说不知道，不能编造游戏事件或版本数据。回答用简洁中文，先给直接结论，再给最多三条可执行建议。不要索取账号、密码、战绩 ID 或其他个人信息。"""


def context_for(match: Any, plan: Dict[str, Any]) -> Dict[str, Any]:
    payload = match.get_payload() or {}
    player = payload.get("player") or {}
    outcome = payload.get("outcome") or {}
    gaps = []
    analyses = list(getattr(match, "analyses", []) or [])
    if analyses:
        for gap in list((analyses[0].get_detail() or {}).get("gaps") or [])[:12]:
            gaps.append({key: gap.get(key) for key in ("rule_title", "severity", "time_label", "expected_text", "actual_text", "why_text")})
    return {"hero": player.get("hero") or "未识别", "position": player.get("position") or player.get("role") or "未识别", "result": outcome.get("result") or "未知", "gaps": gaps, "training_tasks": [{key: task.get(key) for key in ("title", "when", "action", "completed")} for task in plan.get("tasks", [])]}


def ask(match: Any, plan: Dict[str, Any], question: str) -> str:
    from ..collection import settings, vlm_client

    text = question.strip()
    if not text:
        raise MatchQaError("请输入问题")
    creds = settings.credentials_for("text") or settings.credentials_for("vlm")
    if creds is None:
        raise MatchQaError("还没有配置文本模型；请先到“模型”页面配置文本模型或视觉模型")
    model = settings.model_name("text") or settings.model_name("vlm") or creds.model
    if not model:
        raise MatchQaError("模型配置缺少模型名称")
    prompt = "对局摘要：\n" + json.dumps(context_for(match, plan), ensure_ascii=False) + "\n\n玩家问题：\n" + text
    try:
        result = vlm_client.call_model(model, [vlm_client.text_block(prompt)], creds=creds, system=SYSTEM_PROMPT, max_tokens=700, temperature=0.2)
    except vlm_client.VlmError as exc:
        raise MatchQaError(f"模型暂时无法回答：{exc}") from exc
    answer = (result.text or "").strip()
    if not answer:
        raise MatchQaError("模型没有返回内容")
    return answer
