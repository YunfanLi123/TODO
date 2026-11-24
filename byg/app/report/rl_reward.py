"""报告条目的**可验证奖励**。

为什么要有它
------------
用户对报告的抱怨是：「跟视频之间没有一致性」「是胡编的」。
这类问题**不该**靠再调提示词解决 —— 提示词改一次就要重跑一局，
而且没人能保证下一版不引入新的毛病。

强化学习能用在这里，是因为这一层**有可验证的奖励**：
一条报告结论是不是有画面依据、是不是指认了攻略里的具体一条、
理由是不是对照式的、建议与描述说的是不是同一件事 ——
这些都能**从产物本身算出来**，不需要人打分，也不需要另一个模型打分。

奖励函数因此是**纯函数**：同样的条目永远得到同样的分。
这既是它能当训练信号的原因，也是它能被单测盖住的原因。

奖励项
------
============  ======  ====================================================
项             权重    判据
============  ======  ====================================================
接地           0.30   有画面依据，且帧号真实存在
具体           0.20   指认了攻略里的**具体一条**（不是一整张菜单）
对照           0.20   理由是「攻略要 X，你做的是 Y」
同场景         0.10   建议与「原视频描述」说的是同一片战场
唯一           0.05   与同局其它条目不重复
============  ======  ====================================================

罚分
----
* 出现「照这个打法，……」—— 那是攻略原文写给别人的结果，
  与玩家这一段做的事没有对照关系（用户明确不要这种写法）；
* 出现「几种可选打法」式的菜单 —— 玩家读不出自己该做哪一条。

**这些罚分是「减分」而不是「单项归零」**：一条结论可能接地很好、
只是写法不好，那它仍然比一条既没依据又给菜单的结论好。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

#: 各项权重（和为 1）。
WEIGHTS: Dict[str, float] = {
    "grounding": 0.30,
    "specific": 0.20,
    "contrast": 0.20,
    "same_target": 0.10,
    "unique": 0.05,
}

#: 罚分。
PENALTY_MENU = 0.20      # 建议是一整张菜单
PENALTY_OUTCOME = 0.15   # 理由是「照这个打法，X」

#: 「说的是同一件事吗」的粗判：文本 -> **场景对象**。
#:
#: 为什么判「场景」而不是判「动作」
#: ------------------------------
#: 一条**对照式**结论天生就是「你做的是 A，攻略要 B」—— A 与 B 本来就
#: 不同。按「动作是否一致」判分，等于惩罚所有正确的偏离。
#:
#: 用户当时抱怨的不是「建议和描述的动作不同」，而是**说的不是同一件事**：
#: 描述说「在刷野」，建议说「中路清线」—— 那是拿别的分路的打法来对，
#: 场景根本不是那个场景。
#:
#: 所以判据换成：双方提到的是不是**同一片战场**（塔 / 龙 / 兵线 / 野区 /
#: 草丛 / 河道 …）。判不准就保持中性，不扣分。
_TEXT_TO_CATEGORY: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("farm_jungle", ("野怪", "野区", "buff", "蓝buff", "红buff", "小野", "刷野",
                     "打野", "蓝区", "红区")),
    ("take_dragon", ("暴君", "主宰", "小龙", "大龙", "龙坑", "暗影暴君",
                     "风暴龙王", "开龙", "拿龙")),
    ("siege", ("推塔", "拆塔", "高地", "塔下", "一塔", "二塔", "水晶", "上高地")),
    ("clear_wave", ("兵线", "清线", "小兵", "兵", "补刀")),
    ("vision", ("视野", "眼位", "草丛", "探草", "照亮")),
    ("teamfight", ("团战", "开团", "多人交战", "技能特效", "混战", "交战")),
    ("recall", ("回城", "泉水")),
    ("rotate", ("转线", "支援", "换路", "转去")),
)

#: **场景对象**：判断「说的是不是同一片战场」用。
_SCENE_OBJECTS: Tuple[str, ...] = (
    "塔", "龙坑", "暴君", "主宰", "兵线", "野区", "野怪", "草丛", "河道",
    "水晶", "高地", "视野", "野区", "蓝区", "红区", "团战",
)

#: 对照式理由的标志。
#:
#: 「攻略要 X，你做的是 Y」有两种常见写法：
#: 「攻略这一段要的是……，你做的是……」与「……要的是 X，你做的是 Y」。
_CONTRAST_PATTERNS: Tuple[str, ...] = (
    r"要的是[^，,]{2,40}[，,]\s*你(?:做|实际)的",
    r"攻略[^。]{0,30}要[^。]{0,40}你[^。]{0,20}(?:做|实际)",
    r"你(?:做|实际)的[^。]{0,30}[，,]\s*攻略",
)

#: 「照这个打法」—— 用户明确不要的写法。
_OUTCOME_PHRASE = "照这个打法"

#: 菜单式建议的标志。
_MENU_PHRASE = "几种可选打法"


@dataclass
class Reward:
    """一条候选结论的奖励拆解。"""

    total: float = 0.0
    parts: Dict[str, float] = field(default_factory=dict)
    penalties: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": round(self.total, 4),
            "parts": {k: round(v, 4) for k, v in self.parts.items()},
            "penalties": {k: round(v, 4) for k, v in self.penalties.items()},
            "notes": list(self.notes),
        }


def categories_of(text: str) -> List[str]:
    """一段中文里出现过的动作类别。"""
    s = (text or "")
    out: List[str] = []
    for cat, words in _TEXT_TO_CATEGORY:
        if any(w in s for w in words):
            out.append(cat)
    return out


def _has_contrast(text: str) -> bool:
    for pat in _CONTRAST_PATTERNS:
        if re.search(pat, text or ""):
            return True
    return False


def _menu_like(text: str) -> bool:
    """这段建议像不像「一整张菜单」。

    判据：出现「几种可选打法」这类说法，或者同一句里出现多个
    「……时/就……」的条件分句（攻略的条件式建议挤在一起就是这样）。
    """
    s = (text or "")
    if _MENU_PHRASE in s:
        return True
    parts = re.split(r"[；;]", s)
    # **条件分句**（「……时，就……」这种）—— 挤了两条以上，
    # 读起来就是一张菜单，玩家不知道自己该做哪一条。
    conditional = [p for p in parts if ("时" in p and "就" in p)]
    return len(conditional) >= 2


def score_entry(
    entry: Dict[str, Any],
    frame_count: int = 0,
    seen_titles: Optional[Iterable[str]] = None,
) -> Reward:
    """给一条候选报告条目打分。

    ``entry`` 是报告条目（:func:`app.report.builder.build_mode_analysis`
    产出的那种形状）。``frame_count`` 是这一窗的帧数 —— 用来判断
    「帧号真实存在」。``seen_titles`` 是本局已经出现过的标题，
    用来判重复。
    """
    reward = Reward()
    title = str(entry.get("title") or "")
    advice = str(entry.get("advice_text") or "")
    reason = str(entry.get("reason_text") or "")
    actual = str(entry.get("actual_text") or "")
    moments = entry.get("clip", {}).get("moments") or []

    # -- 接地 ---------------------------------------------------------------
    if moments:
        # 帧号必须真实存在：报告里放的画面要真的抽到过。
        bad = [m for m in moments
               if not str(m.get("image_url") or "").strip()]
        if bad and len(bad) == len(moments):
            reward.parts["grounding"] = 0.2
            reward.notes.append("片段没有可回看的画面")
        else:
            reward.parts["grounding"] = 1.0
    else:
        reward.parts["grounding"] = 0.0
        reward.notes.append("没有画面依据")

    # -- 具体 ---------------------------------------------------------------
    if advice.strip():
        if _menu_like(advice):
            reward.parts["specific"] = 0.2
            reward.notes.append("建议是一整张菜单，不是具体一条")
        else:
            reward.parts["specific"] = 1.0
    else:
        reward.parts["specific"] = 0.0
        reward.notes.append("没有可执行的建议")

    # -- 对照 ---------------------------------------------------------------
    if _has_contrast(reason):
        reward.parts["contrast"] = 1.0
    elif reason.strip():
        reward.parts["contrast"] = 0.3
        reward.notes.append("理由不是对照式的")
    else:
        reward.parts["contrast"] = 0.0

    # -- 同场景 -------------------------------------------------------------
    # 「原视频描述」说在红区刷野，建议说缩到塔下 —— 说的是**同一片战场**
    # 上的两件事，这就是一条正常的对照式结论。
    #
    # 反过来，描述说「在红区刷野」、建议说「去中路推线」——
    # 那是**别的分路**的打法，拿来对这一局就是不对题。
    #
    # 判不准时保持中性：宁可不给信号，也不要给错的信号。
    objs_actual = {o for o in _SCENE_OBJECTS if o in actual}
    objs_advice = {o for o in _SCENE_OBJECTS if o in advice}
    if not objs_actual or not objs_advice:
        reward.parts["same_target"] = 0.6
    elif objs_actual & objs_advice:
        reward.parts["same_target"] = 1.0
    else:
        reward.parts["same_target"] = 0.3
        reward.notes.append("建议与原视频描述说的不是同一片战场")

    # -- 唯一 ---------------------------------------------------------------
    if title and title in set(seen_titles or []):
        reward.parts["unique"] = 0.0
        reward.notes.append("与本局其它条目重复")
    else:
        reward.parts["unique"] = 1.0

    total = sum(WEIGHTS[k] * reward.parts.get(k, 0.0) for k in WEIGHTS)

    # -- 罚分 ---------------------------------------------------------------
    if _OUTCOME_PHRASE in reason:
        reward.penalties["outcome_phrase"] = PENALTY_OUTCOME
        reward.notes.append("理由用了「照这个打法，…」")
    if _MENU_PHRASE in advice:
        reward.penalties["menu"] = PENALTY_MENU

    reward.total = max(0.0, min(1.0, total - sum(reward.penalties.values())))
    return reward


def score_entries(entries: Sequence[Dict[str, Any]],
                   frame_count: int = 0) -> List[Reward]:
    """给一组条目打分（顺便判重复）。"""
    out: List[Reward] = []
    seen: List[str] = []
    for e in entries:
        out.append(score_entry(e, frame_count=frame_count, seen_titles=seen))
        title = str(e.get("title") or "")
        if title:
            seen.append(title)
    return out
