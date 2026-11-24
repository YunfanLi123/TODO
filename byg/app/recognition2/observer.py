"""**全局综述层**：让模型看到这一局的**全貌**，而不只是一个个切片。

为什么要有它
------------
逐段识别有一个结构性的盲点：每段都是**孤立的**。
「第 3 分钟在刷红 buff」这句话单独看没什么意义；放在全局里，
它连着「开局红开、4 级抓下路、反打一波」才构成一局。

而且 —— 逐段识别只能回答「玩家做了什么」，回答不了
「**这局有没有攻略没覆盖到的问题**」。那需要**同时**看到两样东西：
这一局实际发生了什么，和攻略都写了些什么。

所以这一层做两件事：

1. **全局综述**：把整局的段落叙述 + 若干张跨整局的画面交给模型，
   让它写出这局的节奏（几个阶段、每段在干什么、整体评价）；
2. **自由发现**：把**攻略摘要**一起给它，让它回答
   「除了攻略覆盖到的这些，你在画面里还看到了什么问题」。

第 2 条就是「人拿着书做报告」与「只会念书」的分界：
拿着书的人会一边看现场一边看书写下**书里没有的东西**。

这层的输出**不是判定** —— 它不下「做错了」的结论（那是
:mod:`app.recognition2.adjudicator` 的事），它给的是**复盘的开场**。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .beat import Beat, Timeline

#: 跨整局取多少张画面给模型看。
DEFAULT_MAX_FRAMES = 16


# ---------------------------------------------------------------- 配置


@dataclass
class ObserverConfig:
    """综述层的配置。"""

    slot: str = "vlm"
    model: str = ""
    data_dir: str = ""
    max_tokens: int = 4096
    timeout: float = 180.0
    #: 证据帧的目录（``Beat.frames`` 里存的是文件名）。
    frames_dir: str = ""
    #: 跨整局取多少张画面。
    max_frames: int = DEFAULT_MAX_FRAMES


# ---------------------------------------------------------------- 结果


@dataclass
class Overview:
    """一局的全局综述。"""

    #: 整局的整体评价（几句人话）。
    summary: str = ""
    #: 阶段划分：``[{"name": ..., "from": ..., "to": ..., "what": ...}]``。
    phases: List[Dict[str, Any]] = field(default_factory=list)
    #: **攻略没覆盖、但画面里看到了**的问题。
    findings: List[Dict[str, Any]] = field(default_factory=list)
    #: **逐段的操作复盘**：这一段在干什么、有没有问题、该怎么做。
    #:
    #: 与 ``findings`` 的差别：``findings`` 只挑「最要紧的几条」，
    #: 而这里是**按时间把整局过一遍**（每 2 分钟一段一条），
    #: 每条都要给出「有没有问题」的判断（``verdict``）。
    #: 用户的要求就是这个（2026-11-13）：
    #: 「针对局内的操作，在参考模式库的基础上，判断是否存在问题」。
    operations: List[Dict[str, Any]] = field(default_factory=list)
    #: 玩家的风格特征。
    style: str = ""
    #: 留痕（失败原因、被丢掉的回答…）。
    notes: List[str] = field(default_factory=list)
    calls: int = 0
    tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "phases": [dict(p) for p in self.phases],
            "findings": [dict(f) for f in self.findings],
            "operations": [dict(o) for o in self.operations],
            "style": self.style,
            "notes": list(self.notes),
            "calls": self.calls,
            "tokens": self.tokens,
        }


# ---------------------------------------------------------------- 提示词

SYSTEM_PROMPT = (
    "你是一个王者荣耀对局复盘助手，正在做**整局复盘**。\n"
    "你会拿到：这一局各段时间的**自由描述**（按时间顺序，那是另一个助手"
    "**看完整局录像**写下来的），可能还有若干张跨整局的**关键画面**，"
    "以及**攻略覆盖到的要点**。\n"
    "你的活有三件：\n"
    "1. **串起全貌**：这局分几个阶段、每段在干什么、整体是什么走势。\n"
    "2. **找出攻略没写的问题**：拿着攻略当**参考**，看看画面里有没有"
    "**攻略里没有、但确实存在**的问题 —— 这一条最重要："
    "你是复盘助手，不是复读机。\n"
    "3. **概括风格**：这个玩家打游戏是什么风格。\n"
    "硬规矩：\n"
    "1. **只写画面和描述里有的**。没看到的不写，不要靠常识补全。\n"
    "2. **用自己的话说**，不要从任何词表里挑词。\n"
    "3. **看不清就说看不清**。宁可少写一条，也不要编。\n"
    "4. **只输出那一个 JSON 对象**，不要写分析过程、不要写前后说明。\n"
)


def build_prompt(beats: Sequence[Beat], guide: Sequence[str],
                 frame_names: Sequence[str]) -> str:
    """造综述这一次的用户提问。

    **没有画面也能写** —— 新链路里模型已经**看完整局录像**了，
    段落描述就是它自己讲的，这时再塞几张抽帧反而添乱。
    """
    lines: List[str] = []
    if frame_names:
        lines.append("这一局的画面（按时间先后编号，跨整局均匀取样）：")
        for i in range(1, len(frame_names) + 1):
            lines.append("  图%d" % i)
        lines.append("")
        lines.append("这一局各段时间的描述（按时间顺序）：")
    else:
        lines.append("下面是这一局各段时间的描述 —— 那是另一个助手**看完整局录像**"
                     "写下来的（可能漏、可能有错，仅作线索）：")
    for b in beats:
        text = (b.narrative or "").strip()
        when = "?" if b.game_time < 0 else "%d:%02d" % (
            int(b.game_time) // 60, int(b.game_time) % 60)
        lines.append("  · [%s] %s" % (when, text or "（这一段没读出来）"))
    lines.append("")
    if guide:
        lines.append("攻略覆盖到的要点（**只是参考**，不是这局的结论）：")
        for item in guide:
            lines.append("  - %s" % item)
        lines.append("")
        lines.append("请对照着看：**除了上面这些，画面里还有什么攻略没写到的问题？**")
    else:
        lines.append("（这次没有提供攻略要点，只讲你在画面里看到的。）")
    lines.append("")
    lines.append("只回答一个 JSON 对象：")
    lines.append('{"summary": "3~5 句：这局整体怎么样", '
                 '"phases": [{"name": "阶段名", "from": "M:SS", "to": "M:SS", '
                 '"what": "这一阶段在干什么"}], '
                 '"findings": [{"title": "问题的一句话标题", '
                 '"at": "发生在局内什么时候（M:SS，看不出来写空）", '
                 '"detail": "具体是什么、为什么算问题", '
                 '"evidence": "画面上的依据（哪一段、看到了什么）", '
                 '"advice": "下次这种情况该怎么做（一句能照着做的话）"}], '
                 '"operations": [{"at": "M:SS~M:SS", '
                 '"what": "这一段玩家在干什么（一句话）", '
                 '"verdict": "problem|ok|unclear", '
                 '"why": "凭什么判有问题/没问题", '
                 '"advice": "有问题的怎么写，没问题的留空", '
                 '"guide": "参照了哪条打法要点（没有就留空）"}], '
                 '"style": "这个玩家的风格特征"}')
    lines.append("")
    lines.append("findings 的要求（**这一份是报告正文，别写套话**）：")
    lines.append("1. 3~6 条，**按「该改的优先级」排**，最要紧的放第一条。")
    lines.append("2. 每条都要**指得出是哪一段**（at 尽量填），"
                 "说的是**这一局真发生的事**，不是通用建议。")
    lines.append("3. advice 要具体到动作：「先回城补满再出来」而不是"
                 "「注意血量」；不说「多观察」「加强意识」这种空话。")
    lines.append("4. 攻略要点只是**参考**：攻略没写、但这一局确实有问题的地方，"
                 "照样写，那就是最有价值的一条。")
    lines.append("5. 拿不准、画面里没依据的，**别写**。")
    lines.append("")
    lines.append("operations 的要求（**逐段过一遍，这段有没有问题**）：")
    lines.append("1. 按时间把整局**每 2 分钟一条**写下来（有几段写几条），"
                 "不要只挑有问题的写。")
    lines.append("2. 每条都要给 verdict：``problem``（有问题）/ ``ok``（没问题）"
                 "/ ``unclear``（看不清、判不了）。**没问题也要写**，"
                 "否则读者不知道你到底是看过还是漏了。")
    lines.append("3. 有问题的，advice 要给到动作；没问题的，advice 留空。")
    lines.append("4. ``guide`` 写这条判断参照了哪条**打法要点**（照抄上面清单里的"
                 "一句）；没有对应攻略就写空 —— 那就说明你是在按这一局的情况判。")
    return "\n".join(lines)


# ---------------------------------------------------------------- 解析


def parse_overview(raw: Any) -> tuple:
    """把模型回答解析成综述。返回 ``(Overview, 丢掉了什么)``。

    硬校验：阶段与发现都必须**有内容**（空标题的直接丢），
    阶段的时间范围要能解析。
    """
    drops: List[str] = []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return Overview(), ["模型回答不是合法 JSON。"]
    if not isinstance(raw, dict):
        return Overview(), ["模型回答不是 JSON 对象。"]

    out = Overview()
    out.summary = str(raw.get("summary") or "").strip()
    out.style = str(raw.get("style") or "").strip()

    for item in (raw.get("phases") or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        what = str(item.get("what") or "").strip()
        if not (name or what):
            continue
        out.phases.append({
            "name": name or "未命名阶段",
            "from": str(item.get("from") or "").strip(),
            "to": str(item.get("to") or "").strip(),
            "what": what,
        })

    for item in (raw.get("findings") or []):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        detail = str(item.get("detail") or "").strip()
        if not (title or detail):
            continue
        out.findings.append({
            "title": title or "（未命名）",
            "detail": detail,
            "evidence": str(item.get("evidence") or "").strip(),
            # **下次怎么做** —— 没有它，这条「发现」就只是点评，
            # 报告要回答的是「我该怎么改」（2026-11-13）。
            "advice": str(item.get("advice") or "").strip(),
            # 大致发生在局内什么时候（``M:SS`` 或 ``M:SS~M:SS``，可空）。
            "at": str(item.get("at") or item.get("when") or "").strip(),
        })

    for item in (raw.get("operations") or []):
        if not isinstance(item, dict):
            continue
        what = str(item.get("what") or "").strip()
        at = str(item.get("at") or item.get("when") or "").strip()
        if not (what or str(item.get("why") or "").strip()):
            continue
        verdict = str(item.get("verdict") or "").strip().lower()
        if verdict not in ("problem", "ok", "unclear", "问题", "没问题",
                           "看不清"):
            verdict = "problem" if str(item.get("problem") or "").strip() else "unclear"
        if verdict == "问题":
            verdict = "problem"
        elif verdict == "没问题":
            verdict = "ok"
        elif verdict == "看不清":
            verdict = "unclear"
        out.operations.append({
            "at": at,
            "what": what,
            "verdict": verdict,
            "why": str(item.get("why") or "").strip(),
            "advice": str(item.get("advice") or "").strip(),
            #: 这条判断参照了哪条打法（模型的引用，可为空 = 按通用判断）。
            "guide": str(item.get("guide") or "").strip(),
        })

    if not out.summary and not out.phases and not out.findings \
            and not out.operations:
        drops.append("模型回答里没有可用的内容（综述、阶段、发现、操作都是空的）。")
    return out, drops


# ---------------------------------------------------------------- 编排


def guide_digest(patterns, lane: str = "", hero: str = "",
                 limit: int = 60) -> List[str]:
    """把规则库压成**要点清单**（一行一条），给模型当参考。

    这是「书」在复盘里的正当用法：**拿着书对照现场**，而不是
    **照着书念**。所以这里给的是**要点**（标题 + 一句话要求），
    不是全文 —— 模型要的是「书覆盖了哪些点」，
    好判断「画面里还有什么书没写」。

    ``lane`` 留空 = 不按分路筛。``hero`` 给了就**这个英雄的专属规则排前面**
    —— 专属规则才是他这局真正被要求的东西，排在后面会被淹掉。
    ``limit`` 是给模型看的条数上限 —— 太多反而让它抓不住重点。
    """
    picked: List[str] = []
    rest: List[str] = []
    # **统一走规则卡**（2026-11-13）：裁决层和综述层必须看到同一个形状，
    # 否则同一份规则两处讲得不一样，模型的口径就飘。
    try:
        from ..rules import cards as _cards
    except Exception:                                  # noqa: BLE001
        _cards = None
    for pattern in patterns or []:
        if lane:
            pat_lane = str(getattr(pattern, "lane", "") or "")
            applies = getattr(pattern, "applies_to", None)
            pat_lane = pat_lane or str(getattr(applies, "lane", "") or "")
            if pat_lane and pat_lane != lane:
                continue
        if _cards is not None:
            card = _cards.rule_card(pattern)
            if not (card.get("title") or card.get("should")):
                continue
            line = _cards.card_block(card, compact=True)
            if hero and hero in (card.get("title") or ""):
                picked.append(line)
            else:
                rest.append(line)
            continue
        title = str(getattr(pattern, "title", "") or "").strip()
        action = str(getattr(pattern, "action", "") or "").strip()
        if not (title or action):
            continue
        line = ("%s —— %s" % (title, action[:60])) if (title and action) \
            else (title or action[:60])
        if hero and hero in title:
            picked.append(line)
        else:
            rest.append(line)
    return (picked + rest)[:limit]


def _pick_frames(beats: Sequence[Beat], limit: int) -> List[str]:
    """跨整局**均匀**取若干张帧。

    每段先取一张（那段最有代表性的），再在段之间均匀拉开 ——
    全部挤在开局就等于没给全貌。
    """
    names: List[str] = []
    for b in beats:
        for name in b.frames:
            if name not in names:
                names.append(name)
    if not names:
        return []
    if len(names) <= limit:
        return names
    step = len(names) / float(limit)
    picked: List[str] = []
    for i in range(limit):
        idx = min(int(i * step), len(names) - 1)
        if names[idx] not in picked:
            picked.append(names[idx])
    return picked


def observe(timeline: Timeline, guide: Optional[Sequence[str]] = None,
            config: Optional[ObserverConfig] = None) -> Overview:
    """让模型看整局，写出全局综述与**攻略之外**的发现。

    ``guide`` 是**攻略摘要**（一行一条要点）。给它，它才能回答
    「攻略没覆盖到的还有什么」；不给它，就只剩「这局发生了什么」。
    """
    from app.collection import settings, vlm_client

    cfg = config or ObserverConfig()
    out = Overview()
    beats = [b for b in timeline.sorted_beats()
             if (b.narrative or "").strip()]
    if not beats:
        out.notes.append("这一局没有读出任何描述 —— 没有综述可写。")
        return out

    creds = None
    model = cfg.model or ""
    try:
        # **凭据与型号来自同一份配置** —— 同一个数据目录，
        # 否则一个来自这里、一个来自进程环境变量，两边可能不是一份。
        with settings.use_data_dir(cfg.data_dir or None):
            creds = vlm_client.find_credentials(cfg.slot)
            model = model or settings.model_name(cfg.slot)
    except Exception as exc:                       # noqa: BLE001
        out.notes.append("取不到「%s」槽位的模型配置：%s" % (cfg.slot, exc))
        return out
    if not creds:
        out.notes.append("没有配置「%s」这个槽位的模型 —— 没有综述。" % cfg.slot)
        return out
    if not model:
        out.notes.append("没有配置「%s」这个槽位的模型 —— 没有综述。" % cfg.slot)
        return out

    frame_names = _pick_frames(beats, cfg.max_frames)
    loaded: List[bytes] = []
    for name in frame_names:
        path = os.path.join(cfg.frames_dir, name) if cfg.frames_dir else ""
        data = b""
        if path:
            try:
                with open(path, "rb") as fh:
                    data = fh.read()
            except OSError:
                data = b""
        if data:
            loaded.append(data)
    if loaded:
        frame_names = frame_names[:len(loaded)]
    else:
        frame_names = []

    content = [vlm_client.text_block(
        build_prompt(beats, list(guide or []), frame_names))]
    for data in loaded:
        content.append(vlm_client.image_block(data))

    try:
        result = vlm_client.call_model(
            model, content, creds=creds, system=SYSTEM_PROMPT,
            max_tokens=cfg.max_tokens, timeout=cfg.timeout)
    except vlm_client.VlmError as exc:
        out.notes.append("全局综述没写成：%s" % vlm_client.explain_error(exc))
        return out

    out.calls = 1
    out.tokens = _token_count(result.usage or {})
    raw = None
    try:
        raw = result.json()
    except vlm_client.VlmError as exc:
        out.notes.append("全局综述没写成：模型回答不是合法 JSON（%s）" % exc)
    if raw is None:
        return out

    parsed, drops = parse_overview(raw)
    out.notes.extend(drops)
    parsed.notes.extend(out.notes)
    parsed.calls = out.calls
    parsed.tokens = out.tokens
    return parsed


def _token_count(usage: Dict[str, Any]) -> int:
    """从各种形状的 usage 里取 token 数。"""
    total = 0
    for key in ("input_tokens", "output_tokens", "prompt_tokens",
                "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, (int, float)):
            total += int(value)
    return total
