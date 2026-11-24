"""分析层：全局动作描述 + 规则库 → 不超过 300 字的操作模式分析报告。

这一层**不碰视频、不做识别**：
1. 接收「全局动作描述」（一段文字）；
2. 用现有的检索（retriever.search_modes）查规则库对应部分；
3. 交给文字模型自己组织语言，写三部分报告：
   一、本局摘要；二、操作模式分析；三、玩法建议。

报告不超过 300 字；写不出来也不抛异常，返回失败原因。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

SYSTEM_PROMPT = """你是《王者荣耀》复盘分析助手。你会收到：
① 这一局的「全局动作描述」；② 从操作模式库检索到的相关攻略规则。

只做一件事：自己组织语言，写一份**不超过 300 字**的操作模式分析报告。
**严格分三部分，每部分单独成段 —— 三部分不要挤在同一行里**：

一、本局摘要
（本部分正文另起一段）

二、操作模式分析
（本部分正文另起一段）

三、玩法建议
（本部分正文另起一段）

要求：
- 只写事实（来自描述）和规则（来自检索），不编技能、不编没发生的事；
- 不写时间数字；只用玩家英雄名，不写别的英雄名字；
- 语言简洁，像教练复盘，不像复述画面。
"""


def _rules_text(result: Any) -> str:
    hits = list(getattr(result, "hits", []) or [])
    if not hits:
        return "（没有检索到相关规则）"
    lines: List[str] = []
    for h in hits[:8]:
        title = str(getattr(h, "title", "") or "").strip()
        text = str(getattr(h, "text", "") or "").strip()
        lines.append("- %s：%s" % (title, text))
    return "\n".join(lines)


def build_analysis_report(description: str, *, game: str = "hok",
                          hero: str = "", position: str = "",
                          version: str = "S45", data_dir: str = "",
                          model: str = "", slot: str = "text",
                          timeout: float = 180.0,
                          retry: int = 1) -> Dict[str, Any]:
    """描述 + 规则 → 三部分报告。返回 {report, rules, notes}。"""
    from ..collection import settings, vlm_client
    from ..rag.retriever import search_modes

    notes: List[str] = []
    desc = (description or "").strip()

    # 1. 检索规则库对应部分（与模型凭据同目录：Qdrant 就在该数据目录下）
    try:
        with settings.use_data_dir(data_dir or None):
            result = search_modes(desc[:300] or hero or game, game=game,
                                  version=version, lane=position, hero=hero,
                                  top_k=8)
        rules = _rules_text(result)
    except Exception as exc:                              # noqa: BLE001
        rules = "（检索失败：%s）" % exc
        notes.append("检索规则失败：%s" % exc)

    # 2. 拿文字模型凭据（凭据与型号必须来自同一份配置目录）
    try:
        with settings.use_data_dir(data_dir or None):
            creds = vlm_client.find_credentials(slot, data_dir or None)
            if not model:
                model = settings.model_name(slot)
    except Exception as exc:                              # noqa: BLE001
        return {"report": "", "rules": rules,
                "notes": ["取不到文字模型配置：%s" % exc]}
    if not creds or not model:
        return {"report": "", "rules": rules,
                "notes": ["没有配置文字模型（slot=%s）" % slot]}

    # 3. 让模型写报告
    prompt = (
        "【全局动作描述】\n%s\n\n"
        "【检索到的攻略规则】\n%s\n\n"
        "请按三部分写，不超过 300 字。" % (desc or "（空）", rules)
    )
    try:
        resp = vlm_client.call_model(
            model, [vlm_client.text_block(prompt)], creds=creds, slot=slot,
            system=SYSTEM_PROMPT, max_tokens=700, timeout=timeout, retry=retry)
        text = (resp.text or "").strip()
    except Exception as exc:                              # noqa: BLE001
        return {"report": "", "rules": rules,
                "notes": ["文字模型调用失败：%s" % exc]}

    return {"report": text, "rules": rules, "notes": notes}
