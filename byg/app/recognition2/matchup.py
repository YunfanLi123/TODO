# -*- coding: utf-8 -*-
"""**对线优劣让模型看画面判**，不再查对位表（2026-11-14 用户要求）。

背景
----
规则里有 10 来条的前提是「对位优劣」（优势/劣势/均势）。以前这个事实只有
两个来源：玩家自己填，或者一张**英雄对位表** —— 而用户明确说：

> 不要对位表了，让模型去看

理由很实在：**对线劣势不是英雄决定的**。同一个对位，被反野三次、
第一波团没接、血量一直半条，就是劣势；反过来打得顺就是优势。
表只能回答「理论上这俩谁强」，回答不了「这一局他打得好不好」。
而模型**看得到**：谁在塔下补刀、谁在压线、血量谁低、有没有被单杀。

这一层就干一件事：拿对线期那几段的画面描述 + 血量/人头事实，
让模型判一句「这一局对线是优势、劣势、还是均势」，**并给出依据**。
判不出来就 ``unknown`` —— 那就按「这条规则不适用」处理，不硬套。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

#: 对线期的判据窗口（局内分钟）：前 4 分钟足够看出谁被压。
LANE_WINDOW_MIN = 4.0

#: 归一后的取值（与适配器的 ``MATCHUP_VALUES`` 一致）。
VALUES = ("advantage", "disadvantage", "unknown")

LABELS = {"advantage": "优势", "disadvantage": "劣势", "unknown": "看不出来"}

SYSTEM = (
    "你是《王者荣耀》的复盘助手。你的活只有一件："
    "判断这一局**玩家对线期的对位优劣**。\n"
    "只输出 JSON，不要解释。"
)

PROMPT = (
    "这一局玩家是 **{hero}**，走**{lane}**。\n"
    "下面是对线期（前 {minutes:.0f} 分钟）的画面描述，按时间顺序：\n"
    "{timeline}\n"
    "血量事实：{hp}\n"
    "死亡事实：{deaths}\n"
    "\n请判断**这一局的对线**是谁占优（不是英雄强弱，是这局打出来的样子）：\n"
    '{{"matchup": "advantage|disadvantage|even|unknown", '
    '"why": "一句话：凭什么这么判（谁被压、谁在塔下、血量/人头怎么走的）"}}\n'
    "\n怎么判（照这个口径）：\n"
    "· **advantage**：对面被压回塔下、对方吃不到线、我方先到 4 级或先游走；\n"
    "· **disadvantage**：我方被压在塔下、血量长期偏低、被单杀或被越塔；\n"
    "· **even**：互有往来，谁也没压住谁；\n"
    "· **unknown**：描述里看不出来。"
)


@dataclass
class MatchupJudgement:
    """对线优劣的判断结果。"""

    #: ``advantage`` / ``disadvantage`` / ``even`` / ``unknown``。
    matchup: str = "unknown"
    why: str = ""
    notes: List[str] = field(default_factory=list)
    calls: int = 0
    tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matchup": self.matchup,
            "why": self.why,
            "label": LABELS.get(self.matchup, "看不出来"),
            "notes": list(self.notes),
        }


def _norm(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in ("advantage", "adv", "优势", "优势对位"):
        return "advantage"
    if text in ("disadvantage", "dis", "劣势", "劣势对位"):
        return "disadvantage"
    if text in ("even", "均势", "打平"):
        return "even"
    return "unknown"


def build_timeline(beats: Sequence[Any],
                   window: float = LANE_WINDOW_MIN) -> str:
    """把对线期那几段的描述拼起来。"""
    lines: List[str] = []
    for beat in beats or []:
        start = getattr(beat, "game_time_start", None)
        try:
            start = float(start)
        except (TypeError, ValueError):
            continue
        if start > window * 60.0:
            continue
        text = str(getattr(beat, "narrative", "") or "").strip()
        if not text:
            continue
        lines.append("· [%d:%02d] %s" % (int(start) // 60, int(start) % 60,
                                         text[:200]))
    return "\n".join(lines) or "（对线期没有描述）"


def build_deaths(deaths: Sequence[Dict[str, Any]],
                 window: float = LANE_WINDOW_MIN) -> str:
    """对线期的死亡（局内分钟）。"""
    out: List[str] = []
    for item in deaths or []:
        try:
            minute = float(item.get("minute") or 0.0)
        except (TypeError, ValueError):
            continue
        if minute > window:
            continue
        out.append("· %s %s" % (item.get("clock") or "?", item.get("cause") or "死因没读到"))
    return "\n".join(out) or "（对线期没有死亡）"


def build_hp(hp: Dict[str, Any], window: float = LANE_WINDOW_MIN) -> str:
    """对线期的血量低点。"""
    episodes = (hp or {}).get("episodes") or []
    lows = []
    for item in episodes:
        try:
            minute = float(item.get("minute") or 0.0)
        except (TypeError, ValueError):
            continue
        if minute > window:
            continue
        lows.append("%s 最低 %.0f%%" % (item.get("clock") or "?", float(item.get("min_pct") or 0)))
    return "；".join(lows) or "（对线期没有残血段）"


def judge(*, beats: Sequence[Any], deaths: Sequence[Dict[str, Any]] = (),
          hp: Optional[Dict[str, Any]] = None, hero: str = "", lane: str = "",
          slot: str = "vlm", model: str = "", data_dir: str = "",
          max_tokens: int = 600,
          timeout: float = 180.0) -> MatchupJudgement:
    """让模型判对线优劣。失败只留痕（返回 ``unknown``）。"""
    from app.collection import vlm_client

    out = MatchupJudgement()
    timeline = build_timeline(beats)
    if not timeline or timeline.startswith("（对线期没有描述"):
        out.notes.append("对线期没有画面描述 —— 没判对线优劣。")
        return out
    try:
        creds = vlm_client.find_credentials(slot, data_dir or None)
        if not model:
            from app.collection import settings
            model = settings.model_name(slot)
    except Exception as exc:                           # noqa: BLE001
        out.notes.append("取不到模型配置：%s" % exc)
        return out
    if not creds or not model:
        out.notes.append("没有配置「%s」槽位的模型 —— 没判对线优劣。" % slot)
        return out

    lane_label = {"jungle": "打野", "mid": "中路", "side": "对抗路",
                  "carry": "发育路", "roam": "游走"}.get(lane, lane or "不知道哪条路")
    prompt = PROMPT.format(
        hero=hero or "（没认出来）", lane=lane_label,
        minutes=LANE_WINDOW_MIN, timeline=timeline,
        hp=build_hp(hp or {}), deaths=build_deaths(deaths))
    try:
        answer = vlm_client.call_model(
            model, [vlm_client.text_block(prompt)], creds=creds,
            system=SYSTEM, max_tokens=max_tokens, timeout=timeout)
    except vlm_client.VlmError as exc:
        out.notes.append("对线优劣没判成：%s" % vlm_client.explain_error(exc))
        return out
    out.calls += 1
    try:
        usage = answer.usage or {}
        out.tokens += sum(int(usage.get(k) or 0) for k in
                          ("input_tokens", "output_tokens",
                           "prompt_tokens", "completion_tokens"))
    except Exception:                                  # noqa: BLE001
        pass
    try:
        raw = answer.json()
    except vlm_client.VlmError as exc:
        out.notes.append("对线优劣的回答不是合法 JSON（%s）" % exc)
        return out
    if not isinstance(raw, dict):
        out.notes.append("对线优劣的回答不是 JSON 对象。")
        return out
    out.matchup = _norm(raw.get("matchup"))
    out.why = str(raw.get("why") or "").strip()
    if out.matchup == "unknown":
        out.notes.append("模型说这局对线看不出来 —— 相关规则按不适用处理。")
    return out
