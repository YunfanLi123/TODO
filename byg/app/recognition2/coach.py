# -*- coding: utf-8 -*-
"""**死亡复盘**：这次死亡是赚还是亏，下次该怎么做（2026-11-13）。

为什么要单独一层
---------------
用户对报告的要求很具体：

> 针对局内死亡，模型可以参考模式库的基础上，并自己搜索攻略，
> 给出死亡分析，分析对局内有益还是有害。

而原来那一层（``death_review.py``）只做到「怎么死的 + 从攻略库检索一句建议」：
它不问**这一次死值不值**。于是报告里两次死亡都是一模一样的句式，
玩家看不出「一换一换掉打野」和「残血白送」的区别 —— 那是两件事。

这一层怎么判「有益 / 有害」
---------------------------
**不打分、不编数字**，只按局面说清楚三件事：

* ``what``：这一次换到了什么、丢了什么（换掉谁、丢了龙还是塔）；
* ``trade``：``beneficial`` / ``harmful`` / ``even`` / ``unknown``；
* ``advice``：下次这种情况该怎么做（具体到动作）。

判据写在提示词里（换掉关键位 / 换来资源 / 白送 / 死了丢龙），
**看不出来就写 unknown** —— 「不知道」和「血亏」在报告上必须是两句话。

**攻略由模型自己查**：它先写想查什么（``search``），我们拿它的话去模式库检索，
把结果再喂回去让它定稿。代码不替它选攻略 —— 死在不同局面该看哪条打法，
是它看着这一局决定的。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import frames_watch, watcher

#: 一次死亡最多查几次攻略。
MAX_SEARCH = 2

#: 每条死亡最多看几张画面（死前 / 死后各一张就够了）。
EVIDENCE_FRAMES = 2

#: 最多复盘几次死亡（和 pipeline 里落库的上限一致）。
MAX_DEATHS = 8

SYSTEM_PROMPT = (
    "你是《王者荣耀》的对局复盘助手，正在看**一次具体的死亡**。\n"
    "你的活只有一件：判断**这次死亡对局势是有益还是有害**，并给出下次怎么做。\n"
    "硬规矩：\n"
    "1. **只根据给你的画面与描述**判断；看不出来就写 unknown / 不写，不要猜。\n"
    "2. **不许编数字**（经济差、击杀数）—— 没给的就不写。\n"
    "3. 判断要落到**具体换到了什么**：人头？关键位？资源？还是什么都没换到。\n"
    "4. 只输出那一个 JSON 对象。\n"
)

PROMPT = (
    "这是一局《王者荣耀》里玩家的一次死亡，发生在**局内 {clock}**（{where}）。\n"
    "· 怎么死的：{cause}\n"
    "· 模型看到的自身问题：{fault}\n"
    "· 这一局当时在干什么（模型看整局时写的描述）：\n{context}"
    "\n下面给你{shots}张画面（这一带）。\n"
    "\n请判断这次死亡**对局势是有益还是有害**，并选一个说法：\n"
    # ⚠️ 这是 ``str.format`` 的模板 —— JSON 的花括号要写两遍。
    # 2026-11-14 真跑踩过：单括号被当成占位符，抛 KeyError 'trade'，
    # 一整局的死亡复盘全废（正文只剩「怎么死的」）。
    '{{"trade": "beneficial|harmful|even|unknown", '
    '"what": "这次死亡换到了什么 / 丢了什么（一句话，落到具体的人或资源）", '
    '"why": "凭什么这么判（画面上/描述里看到的）", '
    '"advice": "下次这种情况该怎么做（具体到动作，别写「注意站位」这种）", '
    '"search": [{{"query": "你想查的那条打法", "k": 3}}]}}\n'
    "\n怎么判（照着这个口径，不要自己另编一套）：\n"
    "· **beneficial（有益）**：换掉了对方关键位（打野/射手/核心）、换来了资源"
    "（龙、塔、高地）、拖住多人让队友拿到东西、终结对手节奏；\n"
    "· **harmful（有害）**：什么都没换到（白给）、死了导致丢龙/丢塔/丢高地、"
    "团战还没开就送、残血硬拼；\n"
    "· **even（打平）**：一换一但两边都没换到关键位；\n"
    "· **unknown**：画面看不出来，或当时局势本来就看不明白。\n"
    "\n⚠️ search 最多两条，写你**真的想查**的局面（比如「残血被追该不该回城」）——\n"
    "查回来的攻略会再给你，你可以据此改 advice。"
)

#: 第二轮（拿到攻略之后）的追加提问。
FOLLOWUP = (
    "下面是按你的请求从**打法库**里查回来的原文（可能为空）：\n{guides}\n"
    "\n现在给最终答案：仍然只输出那个 JSON（trade / what / why / advice）。"
    "advice 要结合查到的打法，但**只写和这次死亡有关的**；查不到就按画面判。"
)


@dataclass
class DeathInsight:
    """一次死亡的复盘结论。"""

    #: 原视频里的秒数 / 局内 MM:SS（由调用方给，模型不碰时间）。
    video_start: float = 0.0
    clock: str = ""
    where: str = ""
    cause: str = ""
    #: ``beneficial`` / ``harmful`` / ``even`` / ``unknown``。
    trade: str = "unknown"
    what: str = ""
    why: str = ""
    advice: str = ""
    #: 查回来的攻略（``[{id,title,advice}]``）。
    guides: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    calls: int = 0
    tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_start": round(float(self.video_start), 2),
            "clock": self.clock,
            "where": self.where,
            "cause": self.cause,
            "trade": self.trade,
            "what": self.what,
            "why": self.why,
            "advice": self.advice,
            "guides": [dict(g) for g in self.guides],
            "notes": list(self.notes),
        }


#: ``trade`` 的说法（进报告标题）。
TRADE_LABELS = {
    "beneficial": "这波不亏",
    "harmful": "这波亏了",
    "even": "一换一，打平",
    "unknown": "看不出来值不值",
}


def _norm_trade(value: Any) -> str:
    """把模型写的说法归一成四档。**认不出就是 unknown**（不猜）。"""
    text = str(value or "").strip().lower()
    if text in ("beneficial", "benefit", "good", "worth", "有益", "赚"):
        return "beneficial"
    if text in ("harmful", "bad", "waste", "有害", "亏", "亏了"):
        return "harmful"
    if text in ("even", "neutral", "trade", "打平", "一换一"):
        return "even"
    # 模型有时写中文短句（「这波不亏」/「白给」），按关键词兜一层。
    if "不亏" in text or "赚" in text or "换掉" in text:
        return "beneficial"
    if "亏" in text or "白给" in text or "送" in text:
        return "harmful"
    return "unknown"


def _context_lines(rounds: Sequence[Any], minute_sec: float,
                   limit: int = 3) -> str:
    """死亡前后那几段的画面描述（模型自己写的）。

    ``rounds`` 可以是 :class:`app.recognition2.beat.Beat`（v2 时间线），
    也可以是报告层的 ``RoundRecord`` —— 两种都有 ``narrative``，
    只是时间字段不同（``game_time_start`` vs ``label``）。
    """
    picked: List[str] = []
    for rnd in rounds or []:
        span = _round_span(rnd)
        if span is None:
            continue
        lo, hi = span
        if hi < minute_sec - 240 or lo > minute_sec + 120:
            continue
        text = str(getattr(rnd, "narrative", "") or "").strip()
        if not text:
            continue
        label = str(getattr(rnd, "label_override", "")
                    or getattr(rnd, "label", "") or "")
        if not label:
            try:
                label = "%.0f:%.0f" % (lo // 60, lo % 60)
            except Exception:                          # noqa: BLE001
                label = ""
        picked.append("  · [%s] %s" % (label, text[:220]))
    return "\n".join(picked[-limit:]) or "  （这一段没读到描述）"


def _round_span(rnd: Any) -> Optional[Tuple[float, float]]:
    """段的时间区间（秒）。

    两种形状都认：v2 的 ``Beat``（``game_time_start/end``）与
    报告层的 ``RoundRecord``（``label_override`` 里的 ``M:SS ~ M:SS``）。
    """
    start = getattr(rnd, "game_time_start", None)
    end = getattr(rnd, "game_time_end", None)
    if start is not None and end is not None:
        try:
            return float(start), float(end)
        except (TypeError, ValueError):
            pass
    from .death_review import _round_span as _span

    return _span(rnd)


def _search(service: Any, query: str, *, lane: str, hero: str,
            minute: float, k: int = 3) -> List[Dict[str, Any]]:
    """拿模型自己写的话去打法库检索（不替它选攻略）。"""
    lo = max(0, int(minute) - 2)
    hi = max(lo + 1, int(minute) + 3)
    try:
        result = service.search(query, lane=lane or "", hero=hero or "",
                                lo=lo, hi=hi, top_k=max(1, min(int(k or 3), 5)))
        entries = service.entries(result) or []
    except Exception:                                  # noqa: BLE001
        return []
    out: List[Dict[str, Any]] = []
    for entry in entries[:5]:
        entry = entry or {}
        out.append({
            "id": str(entry.get("id") or ""),
            "title": str(entry.get("title") or ""),
            "premise": str(entry.get("premise") or ""),
            "action": str(entry.get("action") or ""),
            "advice": str(entry.get("deviation_hint") or entry.get("action")
                          or ""),
        })
    return out


def _guide_lines(guides: Sequence[Dict[str, Any]], limit: int = 3) -> str:
    if not guides:
        return "  （这次没查到）"
    lines = []
    for g in guides[:limit]:
        text = "  · %s" % (g.get("title") or g.get("id") or "（无标题）")
        if g.get("premise"):
            text += "｜前提：%s" % str(g.get("premise"))[:80]
        if g.get("action"):
            text += "｜做法：%s" % str(g.get("action"))[:100]
        lines.append(text)
    return "\n".join(lines)


def review_death(*, death: Dict[str, Any], rounds: Sequence[Any],
                 video_path: str, frames_dir: str = "",
                 facts: Any = None, lane: str = "", hero: str = "",
                 slot: str = "vlm", model: str = "", data_dir: str = "",
                 transform: Any = None,
                 kda_events: Sequence[Dict[str, Any]] = (),
                 max_tokens: int = 1200,
                 timeout: float = 240.0) -> DeathInsight:
    """复盘一次死亡：**赚还是亏** + 下次怎么做。任何一步失败都只留痕。"""
    from app.collection import vlm_client

    out = DeathInsight(
        video_start=float(death.get("video_start") or 0.0),
        clock=str(death.get("clock") or ""),
        where=str(death.get("where") or "").strip(),
        cause=str(death.get("cause") or "").strip())
    try:
        creds = vlm_client.find_credentials(slot, data_dir or None)
        if not model:
            from app.collection import settings
            model = settings.model_name(slot)
    except Exception as exc:                           # noqa: BLE001
        out.notes.append("取不到模型配置：%s" % exc)
        return out
    if not creds or not model:
        out.notes.append("没有配置「%s」槽位的模型 —— 这次死亡没复盘。" % slot)
        return out

    minute = float(death.get("minute") or 0.0) * 60.0
    context = _context_lines(rounds, minute)
    # **事实**：顶栏 K/D/A 推出来的「死前死后我方拿到几个人头」。
    # 这是判「赚还是亏」唯一硬依据 —— 没有它，模型只能凭画面猜。
    fact_line = ""
    try:
        from .kda_read import window_facts

        around = window_facts(kda_events, out.video_start)
        before = window_facts(kda_events, out.video_start, before=45.0, after=0.0)
        after = window_facts(kda_events, out.video_start, before=0.0, after=45.0)
        fact_line = ("· 顶栏 K/D/A 推出来的事实：这次死亡**前后各 45 秒**里"
                     "我方拿到 %d 个人头（死前 %d 个、死后 %d 个）。\n"
                     % (around.get("kills", 0), before.get("kills", 0),
                        after.get("kills", 0)))
    except Exception:                                  # noqa: BLE001
        fact_line = ""
    # 死前 5 秒 + 死后 3 秒：一张能看出「当时在哪、身边有谁」，
    # 一张能看出「换到了没有」。
    times = [max(0.0, out.video_start - 5.0), out.video_start + 3.0]
    shots = frames_watch.read_frames(video_path, times, transform) \
        if video_path and os.path.isfile(video_path) else []

    prompt = PROMPT.format(
        clock=out.clock or "（读不到）",
        where=out.where or "（不知道死在哪）",
        cause=out.cause or "（没读到）",
        fault=str(death.get("fault") or "（没读到）"),
        context=context,
        shots=len(shots)) + ("\n" + fact_line if fact_line else "")
    blocks: List[Dict[str, Any]] = [vlm_client.text_block(prompt)]
    for frame in shots:
        blocks.append(vlm_client.image_block(frame.jpeg))

    def _call(extra: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        payload = list(blocks)
        if extra:
            payload = [vlm_client.text_block(extra.get("text") or "")] + payload
        try:
            result = vlm_client.call_model(
                model, payload, creds=creds, system=SYSTEM_PROMPT,
                max_tokens=max_tokens, timeout=timeout)
        except vlm_client.VlmError as exc:
            out.notes.append("这次死亡没复盘成：%s"
                             % vlm_client.explain_error(exc))
            return None
        out.calls += 1
        try:
            usage = result.usage or {}
            out.tokens += int(usage.get("output_tokens") or 0)
        except Exception:                              # noqa: BLE001
            pass
        try:
            raw = result.json()
        except vlm_client.VlmError as exc:
            out.notes.append("模型回答不是合法 JSON（%s）" % exc)
            return None
        return raw if isinstance(raw, dict) else None

    raw = _call()
    if raw is None:
        return out

    # -- 模型自己要求查攻略 ----------------------------------------------
    wanted = raw.get("search")
    if isinstance(wanted, list) and wanted:
        try:
            from ..rag.context import ModeSearchService

            service = ModeSearchService(game=getattr(facts, "game", "") or "hok")
        except Exception:                              # noqa: BLE001
            service = None
        if service is not None:
            for item in wanted[:MAX_SEARCH]:
                if isinstance(item, str):
                    query, k = item, 3
                elif isinstance(item, dict):
                    query = str(item.get("query") or "").strip()
                    try:
                        k = int(item.get("k") or 3)
                    except (TypeError, ValueError):
                        k = 3
                else:
                    continue
                if not query:
                    continue
                found = _search(service, query, lane=lane, hero=hero,
                                minute=minute / 60.0, k=k)
                if found:
                    out.guides = found
                    break
            if out.guides:
                follow = _call({"text": FOLLOWUP.format(
                    guides=_guide_lines(out.guides))})
                if follow:
                    raw = follow

    out.trade = _norm_trade(raw.get("trade"))
    out.what = str(raw.get("what") or "").strip()
    out.why = str(raw.get("why") or "").strip()
    out.advice = str(raw.get("advice") or "").strip()
    if not out.advice:
        out.notes.append("这次死亡模型没给具体做法。")
    return out


def review_deaths(deaths: Sequence[Dict[str, Any]], *, rounds: Sequence[Any],
                  video_path: str, facts: Any = None, lane: str = "",
                  hero: str = "", slot: str = "vlm", model: str = "",
                  data_dir: str = "", transform: Any = None,
                  kda_events: Sequence[Dict[str, Any]] = (),
                  limit: int = MAX_DEATHS) -> Tuple[List[DeathInsight], List[str]]:
    """把落库的死亡清单逐条复盘。返回 ``(结论列表, 留痕)``。"""
    out: List[DeathInsight] = []
    notes: List[str] = []
    items = [d for d in (deaths or []) if isinstance(d, dict)][:max(0, limit)]
    for death in items:
        insight = review_death(death=death, rounds=rounds,
                               video_path=video_path, facts=facts,
                               lane=lane, hero=hero, slot=slot, model=model,
                               data_dir=data_dir, transform=transform,
                               kda_events=kda_events)
        notes.extend(insight.notes or [])
        out.append(insight)
    return out, notes


# ---------------------------------------------------------------- 整局复盘
#
# 用户 2026-11-13 的要求（报告正文改成三段式）：
#
#   第一部分：总结，指出玩家错在哪了；
#   第二部分：**让模型当教官给分析**，不要再考虑命中规则的事情。
#
# 所以这一步**不看规则命中**：它拿的是这一局的事实（时间线描述、每次死亡、
# 有问题的操作），产出两样东西 —— ``wrong_points``（错在哪，3~5 条）
# 和 ``analysis``（教官怎么讲：为什么错、下次怎么做）。
# 规则库在这里只是背景知识，不进结论链。

MATCH_SYSTEM = (
    "你是一名《王者荣耀》职业队的复盘教官，正在给一名玩家做**赛后复盘**。\n"
    "你只谈**这一局真实发生的事**：他做了什么、错在哪、下次该怎么做。\n"
    "硬规矩：\n"
    "1. **不提规则库、不提命中、不提模式** —— 玩家要听的是教练的话，不是系统日志。\n"
    "2. **只根据给你的材料**；材料里没有的，不要编（尤其是画面细节）。\n"
    "3. 说人话：**具体到动作和时间**，不说「注意意识」「多观察」这种空话。\n"
    "4. 只输出那一个 JSON 对象。\n"
)

MATCH_PROMPT = (
    "这一局的材料如下。\n\n"
    "【基本盘】英雄：{hero}｜分路：{lane}｜结果：{result}｜时长：约 {minutes} 分钟\n"
    "【整局过程（另一名助手看完整局录像后写的描述，按时间顺序）】\n{timeline}\n"
    "【每一次死亡（另一名助手逐次复盘过）】\n{deaths}\n"
    "【判过「有问题」的操作】\n{operations}\n"
    "{rules}"
    "\n请产出两样东西：\n"
    "1. ``wrong_points``：**这一局他错在哪**，3~5 条，按严重程度排。\n"
    "   每条只要三样：什么时候（at）、做错了什么（what，一句话）、错在哪（why，一句话）。\n"
    "2. ``advice``：**该怎么做**，3~5 条，和 wrong_points 一一对应。\n"
    "   每条讲清楚：那个问题是什么（title）、换成一个会打的人会怎么处理"
    "（how，具体到动作、用自己的话讲，不要照抄参考里的句子）、"
    "凭什么这么改（basis：参照了哪条打法要点、或者哪一段对局描述）。\n"
    "\n只输出一个 JSON：\n"
    '{{"wrong_points": [{{"at": "M:SS", "what": "…", "why": "…"}}], '
    '"advice": [{{"title": "…", "how": "…", "basis": "…"}}]}}'
)


@dataclass
class MatchReview:
    """整局复盘：错在哪 + 教官分析。"""

    wrong_points: List[Dict[str, str]] = field(default_factory=list)
    #: **该怎么做**（2026-11-14 用户定：报告第二段）。
    #: 每条 ``{title, how, basis}`` —— 模型参照打法要点与对局描述自己润色。
    advice: List[Dict[str, str]] = field(default_factory=list)
    #: 旧字段（教官分析）：保留给老数据/老页面，新的只用 ``advice``。
    analysis: List[Dict[str, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    calls: int = 0
    tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wrong_points": [dict(p) for p in self.wrong_points],
            "advice": [dict(a) for a in self.advice],
            "analysis": [dict(a) for a in self.analysis],
            "notes": list(self.notes),
        }


def _compact_timeline(rounds: Sequence[Any], limit: int = 900) -> str:
    """把整局描述压成一段（每段一行，最多 ``limit`` 字）。"""
    lines: List[str] = []
    for rnd in rounds or []:
        text = str(getattr(rnd, "narrative", "") or "").strip()
        if not text:
            continue
        start = getattr(rnd, "game_time_start", None)
        label = ""
        if start is not None:
            try:
                label = "%d:%02d" % (int(start) // 60, int(start) % 60)
            except (TypeError, ValueError):
                label = ""
        lines.append("· [%s] %s" % (label or "?", text[:240]))
    out = "\n".join(lines)
    return out[:limit] if len(out) > limit else out


def _deaths_digest(deaths: Sequence[Dict[str, Any]]) -> str:
    """死亡清单（已被逐次复盘过的那份）。"""
    lines: List[str] = []
    for item in deaths or []:
        clock = str(item.get("clock") or "")
        trade = str(item.get("trade") or "unknown")
        what = str(item.get("what") or "").strip()
        label = {"beneficial": "这波不亏", "harmful": "这波亏了",
                 "even": "一打一", "unknown": "看不出来"}.get(trade, trade)
        lines.append("· %s %s%s" % (clock or "?", label,
                                     ("：" + what) if what else ""))
    return "\n".join(lines) or "（这一局没有死亡，或者没读到）"


def _operations_digest(operations: Sequence[Dict[str, Any]]) -> str:
    """判过「有问题」的操作。"""
    lines: List[str] = []
    for item in operations or []:
        if str(item.get("verdict") or "") != "problem":
            continue
        at = str(item.get("at") or "")
        what = str(item.get("what") or "").strip()
        why = str(item.get("why") or "").strip()
        line = "· %s %s" % (at or "?", what)
        if why:
            line += "（%s）" % why
        lines.append(line)
    return "\n".join(lines) or "（没有单独标出来的问题操作）"


def _rules_block(rules: Sequence[str], limit: int = 12) -> str:
    """把候选打法压成一段「可参照的要点」—— 只是参考，不是结论。"""
    lines = [r.strip() for r in (rules or []) if str(r).strip()][:limit]
    if not lines:
        return "【可参照的打法】\n（这次没有检索到可参照的打法）\n"
    out = ["【可参照的打法（只是参考，不是结论）】"]
    for text in lines:
        # 一行一条：去掉换行、截断，免得把提示词撑爆。
        one = " ".join(text.split())
        out.append("· " + (one[:180] + "…" if len(one) > 180 else one))
    return "\n".join(out) + "\n"


def review_match(*, rounds: Sequence[Any], deaths: Sequence[Dict[str, Any]],
                 operations: Sequence[Dict[str, Any]] = (),
                 situation: str = "", rules: Sequence[str] = (),
                 hero: str = "", lane: str = "", result: str = "",
                 minutes: float = 0.0, slot: str = "vlm", model: str = "",
                 data_dir: str = "", max_tokens: int = 2200,
                 timeout: float = 240.0) -> MatchReview:
    """整局复盘：**错在哪 + 教官分析**（不看规则命中）。失败只留痕。"""
    from app.collection import vlm_client

    out = MatchReview()
    try:
        creds = vlm_client.find_credentials(slot, data_dir or None)
        if not model:
            from app.collection import settings
            model = settings.model_name(slot)
    except Exception as exc:                           # noqa: BLE001
        out.notes.append("取不到模型配置：%s" % exc)
        return out
    if not creds or not model:
        out.notes.append("没有配置「%s」槽位的模型 —— 这一局没做整局复盘。" % slot)
        return out

    timeline = _compact_timeline(rounds)
    if not timeline:
        out.notes.append("整局描述是空的 —— 没做整局复盘。")
        return out
    prompt = MATCH_PROMPT.format(
        hero=hero or "（没认出来）",
        lane={"jungle": "打野", "mid": "中路", "side": "对抗路",
              "carry": "发育路", "roam": "游走"}.get(lane, lane or "（不知道）"),
        result={"lose": "输了", "win": "赢了"}.get(result, result or "（不知道）"),
        minutes="%.0f" % (minutes or 0.0),
        timeline=timeline,
        rules=_rules_block(rules),
        deaths=_deaths_digest(deaths),
        operations=_operations_digest(operations)
        + (("\n【前面判过的局势】\n" + situation) if situation else ""))
    try:
        answer = vlm_client.call_model(
            model, [vlm_client.text_block(prompt)], creds=creds,
            system=MATCH_SYSTEM, max_tokens=max_tokens, timeout=timeout)
    except vlm_client.VlmError as exc:
        out.notes.append("整局复盘没做成：%s" % vlm_client.explain_error(exc))
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
        out.notes.append("整局复盘的回答不是合法 JSON（%s）" % exc)
        return out
    if not isinstance(raw, dict):
        out.notes.append("整局复盘的回答不是 JSON 对象。")
        return out

    for item in (raw.get("wrong_points") or []):
        if not isinstance(item, dict):
            continue
        what = str(item.get("what") or "").strip()
        why = str(item.get("why") or "").strip()
        if not (what or why):
            continue
        out.wrong_points.append({
            "at": str(item.get("at") or "").strip(),
            "what": what,
            "why": why,
        })
        if len(out.wrong_points) >= 6:
            break
    for item in (raw.get("advice") or []):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        how = str(item.get("how") or item.get("fix") or "").strip()
        basis = str(item.get("basis") or "").strip()
        if not (title or how):
            continue
        out.advice.append({"title": title or how[:20], "how": how,
                           "basis": basis})
        if len(out.advice) >= 6:
            break
    for item in (raw.get("analysis") or []):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        detail = str(item.get("detail") or "").strip()
        fix = str(item.get("fix") or "").strip()
        if not (title or detail):
            continue
        out.analysis.append({"title": title or detail[:20],
                             "detail": detail, "fix": fix})
        if len(out.analysis) >= 6:
            break
    if not out.wrong_points and not out.advice and not out.analysis:
        out.notes.append("整局复盘没有可用内容（错在哪、分析都是空的）。")
    return out
