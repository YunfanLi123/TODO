"""Harness 内核：**从模型的裁决得到这一局的判定与统计**。

这一层替代旧内核在「视频来源」对局上的判定权
（:mod:`app.core.attribution.analyze_match` 那条路）。

分工
----
* **模型**：逐条裁决攻略清单 —— 前提看不看得见成立、哪一支成立、
  玩家做的是不是这一件事、差在哪一步；
* **代码**：校验（帧号真实、模式在清单里）、聚合（计数与执行度）、
  落地（写报告）。

代码**不**替模型判对错 —— 那是这次重写内核的核心。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..core.context import MatchContext
from .verdict import (
    STATUS_DEVIATED,
    STATUS_SATISFIED,
    STATUS_UNDETERMINED,
    VERDICT_DEVIATION,
    VERDICT_NEVER,
    VERDICT_OK,
    VERDICT_UNCLEAR,
    HarnessStats,
    ModelVerdict,
)

#: 模型裁决 → 内核状态。
_VERDICT_TO_STATUS = {
    VERDICT_OK: STATUS_SATISFIED,
    VERDICT_DEVIATION: STATUS_DEVIATED,
    VERDICT_NEVER: STATUS_DEVIATED,
    VERDICT_UNCLEAR: STATUS_UNDETERMINED,
}


@dataclass
class HarnessAnalysis:
    """一局的模型裁决结果。"""

    verdicts: List[ModelVerdict] = field(default_factory=list)
    stats: HarnessStats = field(default_factory=HarnessStats)
    #: **没进结论的条目与原因** —— 报告要能说清「挡了几条、为什么」。
    #:
    #: 口径（用户拍板）：维持严闸（不写进结论区），但要写明。
    #: 空报告最容易被读成「打得没问题」，实际可能是「模型没判出来」。
    dropped: Dict[str, int] = field(default_factory=dict)
    #: 逐条原因（给人看的）。
    drop_reasons: List[str] = field(default_factory=list)
    #: **检索口径**：这一局模型自己查了几次打法库、召回了几条、用上了几条。
    #:
    #: 为什么要有这个字段
    #: ------------------
    #: RAG 之后「结论条数变多」有两种原因：**检索补齐了清单的缺口**，
    #: 或者**模型判得更细**。不分开说，「效果变好」就归因不清 ——
    #: 下次改检索还是改提示词，会完全改错方向。
    #:
    #: 口径：
    #: * ``searched`` —— 有多少窗**真的查了**；
    #: * ``recalled`` —— 查回来多少条（去重）；
    #: * ``used`` —— 多少条**真的进了结论**（判出了对错）。
    retrieval: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdicts": [v.to_dict() for v in self.verdicts],
            "stats": self.stats.to_dict(),
            "dropped": dict(self.dropped),
            "drop_reasons": list(self.drop_reasons),
            "retrieval": dict(self.retrieval),
        }


def _category_of(entry: Dict[str, Any]) -> str:
    """分组名。优先用分路标签，没有就按模式 id 的前缀猜。"""
    lane_label = str(entry.get("lane_label") or "").strip()
    if lane_label:
        return lane_label
    mode_id = str(entry.get("mode_id") or "")
    if "-HERO-" in mode_id:
        return "英雄专属打法"
    return "全线路"


def _span_of(entry: Dict[str, Any]) -> Tuple[float, float]:
    """从裁决条目里取回**检验区间**（局内秒）。

    取不到就返回 ``(0, 0)`` —— 那是「不知道这段是哪一段」，
    不是「这段是第 0 秒到第 0 秒」。
    """
    raw = entry.get("span")
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        try:
            return (float(raw[0]), float(raw[1]))
        except (TypeError, ValueError):
            return (0.0, 0.0)
    return (0.0, 0.0)


def _verdict_of(entry: Dict[str, Any]) -> ModelVerdict:
    """把一条模型裁决转成内核用的裁决对象。"""
    raw_verdict = str(entry.get("verdict") or VERDICT_UNCLEAR)
    status = _VERDICT_TO_STATUS.get(raw_verdict, STATUS_UNDETERMINED)
    frames = []
    for f in (entry.get("frames") or []):
        try:
            frames.append(int(f))
        except (TypeError, ValueError):
            continue
    moments = []
    for m in (entry.get("moments") or []):
        if isinstance(m, dict):
            moments.append(m)
    try:
        confidence = float(entry.get("confidence", 1.0))
    except (TypeError, ValueError):
        confidence = 1.0
    span = _span_of(entry)
    return ModelVerdict(
        rule_id=str(entry.get("mode_id") or ""),
        rule_title=str(entry.get("title") or ""),
        category=_category_of(entry),
        severity=str(entry.get("severity") or "medium"),
        round_label=str(entry.get("round_label") or ""),
        span=span,
        code_status=str(entry.get("code_status") or ""),
        status=status,
        branch_no=int(entry.get("branch_no") or 0),
        branch_premise=str(entry.get("branch_premise") or ""),
        branch_action=str(entry.get("branch_action") or ""),
        branch_outcome=str(entry.get("branch_outcome") or ""),
        premise=str(entry.get("premise") or ""),
        action=str(entry.get("action") or ""),
        outcome=str(entry.get("outcome") or ""),
        evidence=str(entry.get("evidence") or ""),
        did=str(entry.get("did") or ""),
        why=str(entry.get("why") or ""),
        advice=str(entry.get("advice") or ""),
        frames=frames,
        moments=moments,
        confidence=confidence,
        kind=str(entry.get("kind") or "mode"),
        # 这一条是不是**模型自己检索回来的**打法。
        retrieved=bool(entry.get("retrieved")),
    )


def analyze_from_model(ctx: MatchContext,
                       active_rules: Optional[int] = None) -> HarnessAnalysis:
    """把这一局里**模型的逐条裁决**聚合成判定与统计。

    ``active_rules`` 是这一局本该检验的条目总数（用来算覆盖率）。
    传 None 就只算已检验到的那些。
    """
    out = HarnessAnalysis()
    stats = HarnessStats()
    seen: Dict[str, ModelVerdict] = {}
    #: 每条模式**最后**留下的裁决的原始判定值（「没进结论」的判据）。
    final_raw: Dict[str, str] = {}
    #: 每条模式最后留下的裁决指认了第几支（0 = 没指认）。
    final_branch: Dict[str, int] = {}

    #: **检索口径**的累加器。
    searched = 0
    recalled_ids: List[str] = []
    asked_total = 0

    for rnd in (ctx.rounds or []):
        label = str(getattr(rnd, "label_override", "") or getattr(rnd, "label", "")
                    or "")
        # 这一窗模型自己查打法库的痕迹。
        retrieval = getattr(rnd, "retrieval", None) or {}
        if isinstance(retrieval, dict) and retrieval.get("asked"):
            searched += 1
            asked_total += len(retrieval.get("asked") or [])
            for rid in (retrieval.get("retrieved") or []):
                rid = str(rid or "")
                if rid and rid not in recalled_ids:
                    recalled_ids.append(rid)
        for entry in (getattr(rnd, "review", None) or []):
            if not isinstance(entry, dict):
                continue
            mode_id = str(entry.get("mode_id") or "").strip()
            if not mode_id:
                continue
            verdict = _verdict_of(entry)
            verdict.round_label = verdict.round_label or label

            # **同一条攻略只算一次**：模型可能对同一窗的不同帧
            # 重复给同一条裁决（一次回答里一条 mode 一条 review）。
            # 重复计数会把执行度算歪。
            raw = str(entry.get("verdict") or VERDICT_UNCLEAR)
            prev = seen.get(mode_id)
            if prev is not None:
                if prev.status == STATUS_UNDETERMINED and verdict.status != STATUS_UNDETERMINED:
                    # 换成了**有结论**的那条 —— 记账按最后留下的那条算，
                    # 所以这里只要把「原始判定」一起换掉就行。
                    seen[mode_id] = verdict
                    final_raw[mode_id] = raw
                    final_branch[mode_id] = int(entry.get("branch_no") or 0)
                continue
            seen[mode_id] = verdict
            final_raw[mode_id] = raw
            final_branch[mode_id] = int(entry.get("branch_no") or 0)

    # 统计「没进结论」的条目与原因（口径：严闸 + 写明）。
    #
    # **按每条模式最后留下的那个裁决算** —— 同一模式先「看不出来」
    # 后判定了，那是**进了结论**的，不能同时还说它「没进结论」。
    # 早先在循环里边记边算，于是同一条模式会同时说「做对了」
    # 和「没进结论」。
    for mode_id, raw in final_raw.items():
        if raw == VERDICT_UNCLEAR:
            out.dropped["unclear"] = out.dropped.get("unclear", 0) + 1
            reason = ("%s：模型说画面里看不出来（前提或做法读不清），"
                      "所以不判对错" % mode_id)
            if reason not in out.drop_reasons:
                out.drop_reasons.append(reason)
        elif not final_branch.get(mode_id):
            out.dropped["no_branch"] = out.dropped.get("no_branch", 0) + 1
            reason = ("%s：模型没能指认是第几支成立，"
                      "只能给整条做法" % mode_id)
            if reason not in out.drop_reasons:
                out.drop_reasons.append(reason)

    out.verdicts = list(seen.values())
    for v in out.verdicts:
        stats.triggered += 1
        stats.checked_rule_ids.add(v.rule_id)
        if v.status == STATUS_SATISFIED:
            stats.satisfied += 1
        elif v.status == STATUS_DEVIATED:
            stats.deviated += 1
        else:
            stats.undetermined += 1
    stats.active_rules = int(active_rules or len(stats.checked_rule_ids))
    out.stats = stats

    #: **检索口径**（报告要能说清「哪几条是查回来的」）。
    used_ids = [v.rule_id for v in out.verdicts if v.retrieved]
    out.retrieval = {
        "searched": searched,
        "asked": asked_total,
        "recalled": len(recalled_ids),
        "recalled_ids": list(recalled_ids),
        "used": len(used_ids),
        "used_ids": used_ids,
    }
    return out
