"""以 LLM 为核心的新内核：**裁决与统计**。

和旧内核的关系
--------------
旧内核（:mod:`app.core.matcher` / :mod:`app.core.attribution`）是
「规则引擎拿事实去比对」的：规则命中 → 收集证据 → 求值 → 判满足/偏离。

新内核把**判定权交回模型**：

* 模型逐条裁决攻略清单（前提是否成立、哪一支成立、玩家做的是什么）——
  这是**语义判断**，代码做不了；
* 代码只做三件事：**校验**（帧号真实吗、模式在清单里吗）、
  **聚合**（满足/偏离/无法判定各多少）、**落地**（写进报告）。

所以这里的 :class:`ModelVerdict` 不是「规则命中」，而是
「模型对一条攻略的裁决」——它带着**画面依据**，因此可回看、可核对。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

#: 裁决的三种状态（与旧内核同名，但**来源不同**）。
STATUS_SATISFIED = "satisfied"
STATUS_DEVIATED = "deviated"
STATUS_UNDETERMINED = "undetermined"

#: 模型裁决里的四种取值（见 :mod:`app.collection.vlm_review`）。
VERDICT_OK = "ok"
VERDICT_DEVIATION = "deviation"
VERDICT_NEVER = "never"
VERDICT_UNCLEAR = "unclear"

#: 模型裁决 → 内核状态。
_VERDICT_TO_STATUS = {
    VERDICT_OK: STATUS_SATISFIED,
    VERDICT_DEVIATION: STATUS_DEVIATED,
    VERDICT_NEVER: STATUS_DEVIATED,
    VERDICT_UNCLEAR: STATUS_UNDETERMINED,
}


@dataclass
class ModelVerdict:
    """模型对**一条攻略**的裁决。

    每一条都必须能指回画面（``frames`` / ``evidence``）——
    指不回去的条目在接地闸就被丢了，不会走到这里。
    """

    #: 攻略条目的 id（模式库里的 ``mode_id``）。
    rule_id: str
    rule_title: str = ""
    #: 分组名（报告里按它归类）。
    category: str = ""
    #: 严重度 —— 报告排序用。
    severity: str = "medium"
    #: 出现在哪一段（``第 3 分钟`` 这种）。
    round_label: str = ""
    #: 这一条**实际检验的局内秒区间** ``(起, 止)``。
    #:
    #: 为什么必须有它：裁决要落到报告的某一段时间上，靠的就是它跟
    #: 时间片的区间求交。没有它，裁决会落进**每一个**时间片 ——
    #: 报告上每条结论的「对局时间区间」都变成第一段，
    #: 而证据帧明明在第 8 分钟。
    #:
    #: 单位是**局内秒**（与 :class:`~app.recognition2.beat.Beat` 的
    #: ``game_time_start`` / ``game_time_end`` 同一套）。
    span: Tuple[float, float] = (0.0, 0.0)
    #: **纯代码口径**的状态（同一批事实用规则条件直接比对的结果）。
    #:
    #: 为什么留着：判定权交给模型之后，报告得能回答「模型和规则条件
    #: 算出来的不一样时，信哪个」。口径是**以模型为准**，代码这边只做
    #: 交叉验证 —— 分歧写进说明位，不进执行度分母。
    #:
    #: 空串 = 没跑过纯代码比对（不分歧）。
    code_status: str = ""
    #: 内核状态：满足 / 偏离 / 无法判定。
    status: str = STATUS_UNDETERMINED
    #: 模型指认的**第几支**（0 = 没指认）。
    branch_no: int = 0
    #: 那一支的前提 / 该做 / 后果（照攻略原文）。
    branch_premise: str = ""
    branch_action: str = ""
    branch_outcome: str = ""
    #: 攻略原文（前提 / 该做 / 后果）。
    premise: str = ""
    action: str = ""
    outcome: str = ""
    #: 模型写的「看到了什么」「玩家做的是什么」「差在哪一步」。
    evidence: str = ""
    did: str = ""
    why: str = ""
    #: 建议（带条件的具体做法）。
    advice: str = ""
    #: 证据帧号（窗内帧号，1 起）。
    frames: List[int] = field(default_factory=list)
    #: 证据时刻（可回看原片）。
    moments: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 1.0
    #: 这一条是**逐条模式**（kind=mode）还是**整段没做**（kind=node）。
    kind: str = "mode"
    #: 来源标记 —— 这条结论是模型说的，不是规则算的。
    source: str = "model"
    #: 这一条是**模型自己检索回来的**打法（不是随题给的清单里的）。
    #:
    #: 为什么要标：RAG 之后「结论变多」有两种原因 ——
    #: 检索补齐了清单的缺口，或者模型判得更细。
    #: 报告得能分开说，否则「效果变好」归因不清。
    retrieved: bool = False

    @property
    def decided(self) -> bool:
        """能不能进执行度的分母。看不清的不算。"""
        return self.status in (STATUS_SATISFIED, STATUS_DEVIATED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_title": self.rule_title,
            "category": self.category,
            "severity": self.severity,
            "round_label": self.round_label,
            "span": [float(self.span[0]), float(self.span[1])],
            "code_status": self.code_status,
            "status": self.status,
            "branch_no": self.branch_no,
            "branch_premise": self.branch_premise,
            "branch_action": self.branch_action,
            "branch_outcome": self.branch_outcome,
            "premise": self.premise,
            "action": self.action,
            "outcome": self.outcome,
            "evidence": self.evidence,
            "did": self.did,
            "why": self.why,
            "advice": self.advice,
            "frames": list(self.frames),
            "moments": list(self.moments),
            "confidence": round(self.confidence, 3),
            "kind": self.kind,
            "source": self.source,
            "retrieved": self.retrieved,
        }


@dataclass
class HarnessStats:
    """一局的执行度统计 —— **分母只算模型判得了的**。

    和旧内核同名同义，但**来源不同**：这里数的是模型裁决，
    不是规则命中。看不清的（unclear）不算进分母，
    否则执行度会因为「模型没看清」而虚低。
    """

    triggered: int = 0
    satisfied: int = 0
    deviated: int = 0
    undetermined: int = 0
    active_rules: int = 0
    #: 本局被真正检验过的条目 id。
    checked_rule_ids: Set[str] = field(default_factory=set)
    #: 统计口径说明 —— 报告页要能让读者知道这数字是谁算的。
    source: str = "model"

    @property
    def decided(self) -> int:
        return self.satisfied + self.deviated

    @property
    def compliance_rate(self) -> Optional[float]:
        """执行度。一条都判不了时返回 None —— 不编一个数字。"""
        if self.decided == 0:
            return None
        return self.satisfied / self.decided

    @property
    def coverage(self) -> float:
        """触发覆盖率：被检验过的条目占清单的比例。

        取值恒在 0~1（分子不会超过分母）—— 与
        :attr:`app.core.attribution.ExecutionStats.coverage` 同一口径。
        """
        if not self.active_rules:
            return 0.0
        return min(1.0, len(self.checked_rule_ids) / self.active_rules)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggered": self.triggered,
            "satisfied": self.satisfied,
            "deviated": self.deviated,
            "undetermined": self.undetermined,
            "decided": self.decided,
            "active_rules": self.active_rules,
            "compliance_rate": (
                round(self.compliance_rate, 4)
                if self.compliance_rate is not None else None
            ),
            "coverage": round(self.coverage, 4),
            "checked_rules": len(self.checked_rule_ids),
            # **口径要写在数字旁边** —— 读者得知道这是模型判的。
            "source": self.source,
        }
