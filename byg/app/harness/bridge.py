"""**桥接层**：把模型的裁决接到老内核的形状上。

为什么要有这一层
----------------
阶段 5 的目标是「**视频路径上老内核的判定调用归零**」：
视频来源的对局，判定权归模型；玩家录入的对局仍走老内核
（事实是玩家给的，规则字面就是唯一依据）。

但报告层还有两处要用老内核的**形状**：

* **分路 × 阶段的运营视图**（:func:`app.core.lane_analysis.analyze_lanes`）——
  它要的是「一组裁决」，不关心这组裁决是谁给的；
* 报告 JSON 的 ``verdicts`` / ``by_category`` 字段 —— 页面与接口认这个形状。

所以这里做一次**单向转换**：模型裁决 → 老形状的裁决。
转换之后，视频路径**不需要再调用**老内核的判定
（:func:`app.core.attribution.analyze_match`），
但报告的每一段仍然照常渲染。

转换只做**形状对齐**，不重新判断：
模型说「偏离」就是偏离，模型说「看不出来」就是无法判定 ——
这里一个字都不改。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..core.context import MatchContext
from ..core.matcher import (
    STATUS_DEVIATED,
    STATUS_SATISFIED,
    STATUS_UNDETERMINED,
    Verdict,
)

#: 模型裁决的状态 → 老内核的状态。
#:
#: 两边的取值本来就一样（``satisfied`` / ``deviated`` / ``undetermined``），
#: 这里显式写出来，是为了让「口径一致」这件事**看得见** ——
#: 万一哪天一边改了名字，这里会立刻炸，而不是让报告悄悄少一半结论。
_STATUS_MAP = {
    STATUS_SATISFIED: STATUS_SATISFIED,
    STATUS_DEVIATED: STATUS_DEVIATED,
    STATUS_UNDETERMINED: STATUS_UNDETERMINED,
}


def _round_of(ctx: MatchContext, label: str) -> Any:
    for rnd in (ctx.rounds or []):
        if getattr(rnd, "label", "") == label or \
                getattr(rnd, "label_override", "") == label:
            return rnd
    return None


def to_legacy_verdicts(ctx: MatchContext,
                        verdicts: Sequence[Any]) -> List[Verdict]:
    """模型裁决 → 老内核形状的裁决。

    只补「形状」，不补「判断」：
    ``status`` 直接沿用模型说的；``evidence`` 留空列表
    （模型的 ``evidence`` 是一句人话，不是老内核那种「事实/阈值/实际取值」
    的结构，塞进去会让人按错误的口径去读）。
    """
    out: List[Verdict] = []
    for v in verdicts or []:
        status = _STATUS_MAP.get(getattr(v, "status", ""), STATUS_UNDETERMINED)
        label = str(getattr(v, "round_label", "") or "")
        rnd = _round_of(ctx, label)
        out.append(Verdict(
            rule_id=str(getattr(v, "rule_id", "") or ""),
            rule_title=str(getattr(v, "rule_title", "") or ""),
            category=str(getattr(v, "category", "") or ""),
            severity=str(getattr(v, "severity", "medium") or "medium"),
            status=status,
            round_label=label or None,
            round_stage=int(getattr(rnd, "stage", 0) or 0),
            round_phase=int(getattr(rnd, "phase", 0) or 0),
            round_time=float(getattr(rnd, "time_index", 0.0) or 0.0),
            evidence=[],
            expected={},
            actual={},
            context={},
            rationale=str(getattr(v, "why", "") or ""),
            deviation_hint=str(getattr(v, "advice", "") or ""),
            premise=str(getattr(v, "premise", "") or ""),
            action=str(getattr(v, "action", "") or ""),
            outcome=str(getattr(v, "outcome", "") or ""),
            source=None,
            tags=[],
            # **静态可纠正性**留给报告层过滤；模型裁决本身不受那条限制
            # （模型是看着画面判断的，见 :func:`app.report.builder._corrective_ids`）。
            corrective=None,
        ))
    return out


def analysis_from_model(rules: Sequence[Any],
                         ctx: MatchContext,
                         harness: Any) -> Dict[str, Any]:
    """按老内核的返回形状，组装一份**由模型判定**的分析结果。

    与 :func:`app.core.attribution.analyze_match` 的返回**同形**，
    这样报告层不用写两套分支 —— 但里面装的全是模型的裁决：

    * ``verdicts`` —— 模型的裁决（老形状）；
    * ``lanes`` —— 分路 × 阶段的运营视图（由模型裁决算出）；
    * ``gaps`` —— **空的**：视频路径的结论正文来自模型逐条裁决
      （见 :func:`app.report.builder._review_entries`），
      老内核那套「规则命中 → 差距」的归因在视频路径上**不再参与**；
    * ``stats`` —— 模型裁决的统计（分母只算模型判得了的）。
    """
    from ..core.lane_analysis import analyze_lanes

    verdicts = to_legacy_verdicts(ctx, getattr(harness, "verdicts", []) or [])
    active = [r for r in (rules or []) if getattr(r, "status", "active") == "active"]
    try:
        lanes = analyze_lanes(active, ctx, verdicts)
    except Exception:                                  # noqa: BLE001
        lanes = {"findings": [], "lanes": [], "cells": []}

    #: 按分类统计（老内核的形状：``分类 → 计数``）。
    by_category: Dict[str, Dict[str, Any]] = {}
    try:
        from ..core.attribution import ExecutionStats

        by_category = ExecutionStats().by_category(verdicts)
    except Exception:                                  # noqa: BLE001
        by_category = {}

    stats = getattr(harness, "stats", None)
    return {
        "match_id": ctx.match_id,
        "game": ctx.game,
        "game_version": ctx.game_version,
        "player": ctx.player,
        "outcome": ctx.outcome,
        "round_count": len(ctx.rounds or []),
        "stats": stats if stats is not None else None,
        # **视频路径上没有老内核的差距** —— 结论正文来自模型裁决。
        "gaps": [],
        "lanes": lanes,
        "verdicts": verdicts,
        "by_category": by_category,
        "undetermined": [v for v in verdicts
                         if v.status == STATUS_UNDETERMINED],
        # 标记这份分析是**模型判的** —— 报告层要能说清口径。
        "source": "model",
    }
