"""从一局的裁决里造**偏好对**（chosen / rejected）。

为什么造这个
------------
GRPO 是**在线**算法：现场采样、现场打分、现场更新。它要能跑起来，
前提是「采样」这一步能产出**同一件事的不同写法**。

我们这里的「一件事」是：同一段画面、同一份攻略，报告条目该写成什么样。
于是采样空间就是**受控扰动**出来的候选写法：

* 去掉画面依据（文字还在，但指不回任何一帧）；
* 把具体建议换成整张菜单；
* 把对照式理由换成「照这个打法，X」；
* 把建议指向**别的**那一支（说的不是这一局适用的那支）；
* 把建议换成另一条线的打法（与原视频描述说的不是同一件事）。

每个候选都由 :mod:`app.report.rl_reward` 打分，
最高分当 chosen、最低分当 rejected —— **标签来自可验证的奖励**，
不来自人，也不来自另一个模型。

这既是训练数据，也是**回归基线**：
奖励分最高的写法，应该就是报告里该采用的那种。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from . import rl_reward

#: 扰动名 -> 说明（写进产物，便于复核每一对是怎么来的）。
PERTURBATIONS = (
    "drop_grounding",
    "menu_advice",
    "outcome_reason",
    "wrong_branch",
    "other_lane_advice",
)


@dataclass
class Pair:
    """一对偏好样本。"""

    prompt: str
    chosen: str
    rejected: str
    chosen_reward: float
    rejected_reward: float
    chosen_entry: Dict[str, Any] = field(default_factory=dict)
    rejected_entry: Dict[str, Any] = field(default_factory=dict)
    perturbation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "chosen_reward": round(self.chosen_reward, 4),
            "rejected_reward": round(self.rejected_reward, 4),
            "perturbation": self.perturbation,
            "chosen_entry": self.chosen_entry,
            "rejected_entry": self.rejected_entry,
        }


def _entry_text(entry: Dict[str, Any]) -> str:
    """把一条报告条目压成一段文本（训练时的「回答」）。"""
    parts = [
        "时间区间：%s" % (entry.get("time_range") or ""),
        "原视频操作：%s" % (entry.get("actual_text") or ""),
        "建议：%s" % (entry.get("advice_text") or ""),
        "理由：%s" % (entry.get("reason_text") or ""),
    ]
    return "\n".join(parts)


def _copy(entry: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(entry, ensure_ascii=False))


def perturb(entry: Dict[str, Any], kind: str) -> Dict[str, Any]:
    """做一种**受控扰动**，产出一个变体条目。"""
    out = _copy(entry)

    if kind == "drop_grounding":
        # 文字还在，但再也指不回任何一帧 —— 这就是「有结论没出处」。
        out.setdefault("clip", {})["moments"] = []
    elif kind == "menu_advice":
        # 建议换成整张菜单（玩家读不出该做哪一条）。
        out["advice_text"] = (
            "这一段攻略给了几种可选打法：推线到对方塔下、绕后断对方兵线、"
            "先到龙坑占位。可以结合当时的局面，挑适合的一条试试看。")
    elif kind == "outcome_reason":
        # 理由换成「照这个打法，X」—— 与玩家做的事没有对照关系。
        out["reason_text"] = "照这个打法，能把这一段的节奏带起来。"
    elif kind == "wrong_branch":
        # 建议指向**别的**那一支（说的不是这一局适用的那支）。
        advice = str(out.get("advice_text") or "")
        out["advice_text"] = ("改成完全相反的做法：" + advice) if advice else ""
    elif kind == "other_lane_advice":
        # 建议换成另一条线的打法 —— 与「原视频描述」说的不是同一件事。
        out["advice_text"] = "去中路推线到对方塔下，占住河道视野。"
    else:
        raise ValueError("未知的扰动：%s" % kind)
    return out


def build_pairs(
    entries: Sequence[Dict[str, Any]],
    frame_count: int = 0,
    perturbations: Sequence[str] = PERTURBATIONS,
) -> List[Pair]:
    """给一批条目造偏好对。

    每条**原始条目**先当成 chosen 的候选（它已经过了报告层的接地闸，
    通常就是好写法），再拿它的扰动版本当 rejected。
    """
    pairs: List[Pair] = []
    for entry in entries:
        base = rl_reward.score_entry(entry, frame_count=frame_count)
        base_text = _entry_text(entry)
        prompt = str(entry.get("prompt") or entry.get("title") or "")
        for kind in perturbations:
            bad = perturb(entry, kind)
            bad_reward = rl_reward.score_entry(bad, frame_count=frame_count)
            if bad_reward.total >= base.total:
                # 扰动没有让奖励变低 → 说明这一项抓不住这个问题，
                # 这一对没有学习价值。
                continue
            pairs.append(Pair(
                prompt=prompt,
                chosen=base_text,
                rejected=_entry_text(bad),
                chosen_reward=base.total,
                rejected_reward=bad_reward.total,
                chosen_entry=entry,
                rejected_entry=bad,
                perturbation=kind,
            ))
    return pairs


def to_jsonl(pairs: Iterable[Pair]) -> str:
    lines = []
    for p in pairs:
        lines.append(json.dumps(p.to_dict(), ensure_ascii=False))
    return "\n".join(lines)
