"""**模型裁决层**：把「这一段玩家在干什么」和「攻略要求什么」放在一起，让模型自己判。

为什么要有它
------------
纯代码判定（:mod:`app.recognition2.judge`）拿**动作枚举**去比规则里的
**期望动作**。这有两处把模型绑住了：

1. **词表**：模型的自由描述被压进 25 个动作码里，选不出来的段一律
   ``unclear``。实测 70 段里 18 段归类不到（查看战绩面板、逛装备商店、
   阵亡等复活、停在预选界面……）—— 模型明明读出来了，却因为
   「不许发明新词」被记成「没读到」。
2. **判官**：攻略的前提（「对方位移或闪现刚交」「首条暴君倒计时进入
   最后一段」）是**局面条件**，画面上能不能确认，只有看画面才知道。
   代码确认不了就判「判不了」—— 宁可少判一条，也不冤枉人。

这里把顺序反过来：**要查哪几条由模型自己定**
（它看完整局录像后写下「我想查什么」，RAG 去取攻略原文 ——
见 :mod:`app.recognition2.selector`），**判官也是模型** ——
它拿到这一段的画面、模型自己写的段落叙述、以及规则的原文，
然后自己回答「做没做到、凭什么」。

三条铁律仍然成立
----------------
1. **看不清 ≠ 没做** → 判 ``undetermined``；
2. **不是给这个玩家的 ≠ 卡点** → 筛子已经排掉，模型不用管；
3. **前提不成立 ≠ 没做** → 模型要能说清「前提没成立」；前提**看不出来**
   也是 ``undetermined``，绝不是 ``deviated``。

硬校验
------
模型的回答**不可信到不能用**：它可能给一条没问的规则下结论、可能把
看不清判成「没做到」、可能引用不存在的帧号。所以每条回答都要过
:func:`parse_verdicts`：规则号对不上就**丢掉**、理由缺失就降级成
``undetermined``、帧号不在给它的帧里就**去掉**。宁可少一条结论，
不可多一条冤枉人的话。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .beat import Beat, Timeline
from .game_knowledge import knowledge_for
from .judge import (STATUS_DEVIATED, STATUS_NOT_APPLICABLE,
                    STATUS_SATISFIED, STATUS_UNDETERMINED, Candidate)

#: 模型可以给的状态。``not_applicable`` 不计入分母。
MODEL_STATUSES = (STATUS_SATISFIED, STATUS_DEVIATED,
                   STATUS_UNDETERMINED, STATUS_NOT_APPLICABLE)


# ---------------------------------------------------------------- 配置


@dataclass
class AdjudicatorConfig:
    """裁决层的配置。"""

    #: 走哪个模型槽位（默认「vlm」—— 看得见画面的那个）。
    slot: str = "vlm"
    #: 模型名。留空 = 从配置里读当前槽位的模型。
    model: str = ""
    #: 数据目录（配置与用户数据在这）。
    data_dir: str = ""
    #: 单次调用的 token 上限。
    max_tokens: int = 4096
    #: 单次调用的超时（秒）。
    timeout: float = 120.0
    #: 一次调用最多问几条规则。
    max_rules_per_call: int = 6
    #: 一次调用最多给多少张画面（``evidence="frames"`` 时用）。
    max_frames_per_call: int = 10
    #: 整局最多调多少次。0 = 不限。
    max_calls: int = 0
    #: 整局最多用多少 token。0 = 不限。
    budget_tokens: int = 0
    #: 证据帧的目录（``Beat.frames`` 里存的是文件名）。
    frames_dir: str = ""
    #: **这一局的局势摘要**（2026-11-14 用户定序之后新增）。
    #:
    #: 裁决要「综合以上所有信息」：对线优劣怎么判的、死过几次各是赚是亏。
    #: 这些是前面几步（逐段叙述 → 对线 → 死亡复盘）的结论，
    #: 裁决层把它们当**上下文**用，而不是从头再猜一遍。
    situation: str = ""
    #: 证据用**视频片段**还是**抽帧**。
    #:
    #: ``video``（新链路）：给模型看这一段的**录像** —— 它能看出连续动作，
    #: 「是不是打到了」「是不是交了位移」这类只有连续画面才看得出来。
    #: ``frames``（旧链路）：抽几张静止画面给模型看。
    evidence: str = "video"
    #: 原视频路径（``evidence="video"`` 时用来切片段）。
    video_path: str = ""
    #: 切片落盘目录。
    video_dir: str = ""
    #: 切片帧率（每秒几帧）。片段短，可以比整局密。
    clip_fps: float = 4.0
    #: 切片前后各留多少秒（前提可能刚好发生在窗口边上）。
    clip_padding: float = 15.0
    #: 一个切片最长多少秒。太长模型会看晕，也烧钱。
    clip_max_seconds: float = 120.0
    #: 给模型看片段时的采样帧率与细节档。
    clip_model_fps: float = 3.0
    clip_detail: str = "default"


# ---------------------------------------------------------------- 结果


@dataclass
class ModelVerdict:
    """模型对一条规则的裁决。"""

    rule_id: str
    status: str = STATUS_UNDETERMINED
    #: **模型原话**：画面里玩家实际做了什么。
    did: str = ""
    #: **模型原话**：凭什么这么判。
    why: str = ""
    #: 画面上的客观状态。
    evidence: str = ""
    #: 模型给的建议。
    advice: str = ""
    #: 引用的证据帧文件名。
    frames: List[str] = field(default_factory=list)
    #: 证据在**原视频**的第几秒（模型报的）。
    #:
    #: 为什么要有：视频证据模式下模型看的是**一段录像**，它报的是
    #: 「这段录像里的第几秒」；报告要靠它把结论定位回原视频的那一秒
    #: （截图、跳转都指着那个时刻）。
    evidence_seconds: List[float] = field(default_factory=list)
    confidence: float = 0.0
    #: 人话：为什么是这个状态。
    reason: str = ""
    #: 这一条**实际检验的局内秒区间** ``(起, 止)``。
    #:
    #: 为什么必须有它：报告要靠它把这条落到某一段时间上。缺了它，
    #: 每条裁决都会落进**每一个**时间片 —— 报告上每条结论的
    #: 「对局时间区间」都变成第一段，而证据帧明明在第 8 分钟。
    #: （这是 2026-11-01 真跑抓出来的：字段根本不存在，
    #: 分桶过滤成了死代码。）
    span: Tuple[float, float] = (0.0, 0.0)
    #: **纯代码比对**的口径（满足 / 偏离 / 判不了 / 不适用）。
    #:
    #: 它不参与判定 —— 判定以模型为准。这个字段只用来**对账**：
    #: 模型说「做到」而代码比对说「没做到」（或反过来），
    #: 说明两边看到的不是同一件事，报告要把分歧写出来，
    #: 而不是只挑一边说、让玩家猜另一边是不是漏了。
    code_status: str = ""

    @property
    def decided(self) -> bool:
        """算不算「判得了」（进入执行度的分母）。"""
        return self.status in (STATUS_SATISFIED, STATUS_DEVIATED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "status": self.status,
            "did": self.did,
            "why": self.why,
            "evidence": self.evidence,
            "advice": self.advice,
            "frames": list(self.frames),
            "evidence_seconds": [round(float(s), 2)
                                 for s in self.evidence_seconds],
            "confidence": round(float(self.confidence), 3),
            "reason": self.reason,
            "span": [float(self.span[0]), float(self.span[1])],
            "code_status": self.code_status,
        }


@dataclass
class ModelJudgement:
    """整局的模型裁决。"""

    verdicts: List[ModelVerdict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    calls: int = 0
    tokens: int = 0

    def by_status(self, status: str) -> List[ModelVerdict]:
        return [v for v in self.verdicts if v.status == status]

    @property
    def decided(self) -> List[ModelVerdict]:
        return [v for v in self.verdicts if v.decided]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdicts": [v.to_dict() for v in self.verdicts],
            "notes": list(self.notes),
            "calls": self.calls,
            "tokens": self.tokens,
            "stats": {
                "total": len(self.verdicts),
                "satisfied": len(self.by_status(STATUS_SATISFIED)),
                "deviated": len(self.by_status(STATUS_DEVIATED)),
                "undetermined": len(self.by_status(STATUS_UNDETERMINED)),
                "not_applicable": len(self.by_status(STATUS_NOT_APPLICABLE)),
            },
        }


# ---------------------------------------------------------------- 提示词

SYSTEM_PROMPT = knowledge_for("hok") + (
    "\n【这一层的活】\n"
    "你是一个王者荣耀对局复盘助手，正在做**对照检查**。\n"
    "你会拿到：一段时间里的几张画面（按时间先后编号）、这一段玩家动作的"
    "**自由描述**（那是另一个助手看画面写下来的，可能有错、也可能有漏）、"
    "以及**一条或多条攻略规则**。\n"
    "你的活只有一件：**对着画面，判断玩家做到了规则要求的事没有**。\n"
    "硬规矩：\n"
    "1. **以画面为准**。自由描述只是线索 —— 它和画面不一致时信画面；"
    "它没提到但画面上有的，也要写出来。\n"
    "2. **用自己的话说**。不要从任何词表里挑词：写「他在红 buff 野区打小野怪」，"
    "不要写「farm_jungle」。\n"
    "3. **看不清就判不了**：画面里看不出来、或你不敢确定，写 "
    "status=\"undetermined\"，并在 why 里说清**哪一点看不出来**。"
    "绝不要因为看不清就判 \"deviated\"。\n"
    "4. **前提没成立 ≠ 没做到**：规则的「前提：…」如果这段画面里**没有发生**"
    "（比如要求「对方位移或闪现刚交」，但这段没交），那是**这条不适用他** —— "
    "写 status=\"not_applicable\"，并在 why 里说清前提为什么没成立。"
    "前提**看不出来**（敌方打野在哪、暴君倒计时到哪了）写 \"undetermined\"。\n"
    "5. **做到了哪些就算做到了**：做了规则要求的那件事、或等价的事、"
    "或其中关键的一步，判 \"satisfied\"；做的是完全另一件事，判 \"deviated\"。\n"
    "6. **每条都要说清凭什么**：did 写「画面里他实际做了什么」，"
    "why 写「凭什么这么判」，evidence 写「画面上的客观状态」。\n"
    "7. **只输出那一个 JSON 对象**，不要写分析过程、不要写前后说明。\n"
)


def build_prompt(rules: Sequence[Candidate], beats: Sequence[Beat],
                 frame_names: Sequence[str], situation: str = "") -> str:
    """造一次调用的用户提问（**抽帧**证据时用）。

    **给规则的原文，不给规则的标题** —— 模型的判官身份全靠原文：
    前提、要求、预期结果、建议都摆出来，它才知道「做到」的标准是什么。
    """
    lines: List[str] = []
    lines.append("下面这一段时间的画面（按时间先后编号）：")
    for i, name in enumerate(frame_names, start=1):
        lines.append("  图%d" % i)
    lines.append("")
    lines.append("这一段玩家动作的自由描述（可能漏、可能有错，仅作线索）：")
    for b in beats:
        text = (b.narrative or "").strip()
        if not text:
            continue
        lines.append("  · %s" % text)
    if not any((b.narrative or "").strip() for b in beats):
        lines.append("  （这一段没有描述，请你只看画面）")
    if situation:
        lines.append("")
        lines.append("这一局的局势（前面几步已经判过的，**当上下文用**）：")
        for line in situation.splitlines():
            if line.strip():
                lines.append("  %s" % line.strip())
    if situation:
        lines.append("")
        lines.append("这一局的局势（前面几步已经判过的，**当上下文用**）：")
        for line in situation.splitlines():
            if line.strip():
                lines.append("  %s" % line.strip())
    lines.append("")
    lines.append("要对照检查的攻略规则：")
    for i, cand in enumerate(rules, start=1):
        lines.append("")
        lines.append("【规则 %d】 id=%s" % (i, cand.rule_id))
        lines.append(cand.rule_text())
        win = cand.window
        lines.append("  时间窗：第 %g~%g 分钟" % (win[0], win[1]))
        # **前提交给模型判**（2026-11-14）：代码只能按手上那几个字段说
        # 「对不上」，而那些字段是粗的。把线索交出去，由模型看着画面判
        # 这条打法适不适用这一段 —— 不适用就写 not_applicable。
        _unknown = list(getattr(cand.judgement, "unknown_facts", None) or [])
        if _unknown:
            lines.append("  这局读不到、或与本局对不上的前提项：%s"
                         " —— 请你自己判断这条适不适用这一段。"
                         % "、".join(dict.fromkeys(_unknown)))
    lines.append("")
    lines.append("只回答一个 JSON 对象：")
    lines.append('{"verdicts": [{"rule_id": "上面给过的 id", '
                 '"status": "satisfied|deviated|undetermined|not_applicable", '
                 '"did": "画面里他实际做了什么", '
                 '"why": "凭什么这么判", '
                 '"evidence": "画面上的客观状态", '
                 '"advice": "给他的建议（可选）", '
                 '"frames": [1, 2], '
                 '"confidence": 0.0~1.0}]}')
    lines.append("每条规则都要有且只有一个条目，frames 只能是上面给过的图号。")
    lines.append("**先判这条打法适不适用这一段**：前提不成立就写 "
                 "not_applicable（并说明为什么），不要硬判成「没做到」。")
    return "\n".join(lines)


def build_video_prompt(rules: Sequence[Candidate], beats: Sequence[Beat],
                       video_lo: float, video_hi: float,
                       game_lo: float, game_hi: float,
                       situation: str = "") -> str:
    """造一次调用的用户提问（**视频片段**证据时用）。

    跟抽帧那一版的区别只有一处，但很要紧：**模型看到的是连续画面**。
    所以提示词里要明确「这是一段录像」，让它按「看视频」的方式判断 ——
    「是不是打到了」「是不是交了位移」这类问题，抽帧根本看不出来。
    """
    lines: List[str] = []
    lines.append("下面这段**录像**对应原视频第 %.0f~%.0f 秒"
                 "（局内第 %g~%g 分）。" % (video_lo, video_hi,
                                          game_lo / 60.0, game_hi / 60.0))
    lines.append("这是**连续画面** —— 动作的起承转合都在里面，"
                 "别只盯着某一帧。")
    lines.append("")
    lines.append("这一段玩家动作的自由描述（可能漏、可能有错，仅作线索）：")
    for b in beats:
        text = (b.narrative or "").strip()
        if not text:
            continue
        lines.append("  · %s" % text)
    if not any((b.narrative or "").strip() for b in beats):
        lines.append("  （这一段没有描述，请你只看录像）")
    lines.append("")
    lines.append("要对照检查的攻略规则：")
    for i, cand in enumerate(rules, start=1):
        lines.append("")
        lines.append("【规则 %d】 id=%s" % (i, cand.rule_id))
        lines.append(cand.rule_text())
        win = cand.window
        lines.append("  时间窗：第 %g~%g 分钟" % (win[0], win[1]))
        # **前提交给模型判**（2026-11-14）：代码只能按手上那几个字段说
        # 「对不上」，而那些字段是粗的。把线索交出去，由模型看着画面判
        # 这条打法适不适用这一段 —— 不适用就写 not_applicable。
        _unknown = list(getattr(cand.judgement, "unknown_facts", None) or [])
        if _unknown:
            lines.append("  这局读不到、或与本局对不上的前提项：%s"
                         " —— 请你自己判断这条适不适用这一段。"
                         % "、".join(dict.fromkeys(_unknown)))
    lines.append("")
    lines.append("只回答一个 JSON 对象：")
    lines.append('{"verdicts": [{"rule_id": "上面给过的 id", '
                 '"status": "satisfied|deviated|undetermined|not_applicable", '
                 '"did": "录像里他实际做了什么", '
                 '"why": "凭什么这么判", '
                 '"evidence": "录像上的客观状态", '
                 '"advice": "给他的建议（可选）", '
                 '"at": 录像里第几秒（数字，取不到写 null）, '
                 '"confidence": 0.0~1.0}]}')
    lines.append("每条规则都要有且只有一个条目；at 是**这段录像里**的第几秒。")
    lines.append("**先判这条打法适不适用这一段**：前提不成立就写 "
                 "not_applicable（并说明为什么），不要硬判成「没做到」。")
    return "\n".join(lines)


# ---------------------------------------------------------------- 解析


def _verdict_items(raw: Any) -> Tuple[List[Any], List[str]]:
    """把模型回答拆成「一条条裁决」。返回 ``(条目, 丢掉了什么)``。"""
    drops: List[str] = []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return [], ["模型回答不是合法 JSON。"]
    if not isinstance(raw, dict):
        return [], ["模型回答不是 JSON 对象。"]
    items = raw.get("verdicts")
    if not isinstance(items, list):
        return [], ["模型回答里没有 verdicts 数组。"]
    return items, drops


def _verdict_from_item(item: Dict[str, Any], allowed: set, seen: set,
                       drops: List[str]) -> Optional[ModelVerdict]:
    """把模型的一条回答变成一条裁决（**硬校验**都在这里）。

    - 规则号不在被问的范围里 → **整条丢掉**（防止张冠李戴）；
    - 状态不认识 → 降级成 ``undetermined``；
    - 「做到了 / 没做到」却没有理由 → 降级成 ``undetermined``。
    """
    rule_id = str(item.get("rule_id") or "").strip()
    if not rule_id or rule_id not in allowed:
        drops.append("模型给了一条没问的规则（%s），已丢掉。"
                     % (rule_id or "空 id"))
        return None
    if rule_id in seen:
        drops.append("模型对 %s 说了两次，只留第一条。" % rule_id)
        return None
    seen.add(rule_id)

    status = str(item.get("status") or "").strip()
    did = str(item.get("did") or "").strip()
    why = str(item.get("why") or "").strip()
    evidence = str(item.get("evidence") or "").strip()
    advice = str(item.get("advice") or "").strip()
    try:
        confidence = float(item.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    if status not in MODEL_STATUSES:
        drops.append("%s 的状态「%s」不认识，降级成「判不了」。"
                     % (rule_id, status or "空"))
        status = STATUS_UNDETERMINED
    if status in (STATUS_SATISFIED, STATUS_DEVIATED) and not (did or why):
        drops.append("%s 判了「%s」却没说凭什么，降级成「判不了」。"
                     % (rule_id, status))
        status = STATUS_UNDETERMINED

    if status == STATUS_SATISFIED:
        reason = why or did
    elif status == STATUS_DEVIATED:
        reason = why or did
    elif status == STATUS_NOT_APPLICABLE:
        reason = why or "模型认为这条的前提没成立。"
    else:
        reason = why or "模型没给出可判的依据。"

    return ModelVerdict(
        rule_id=rule_id, status=status, did=did, why=why,
        evidence=evidence, advice=advice,
        confidence=confidence, reason=reason)


def parse_verdicts(raw: Any, asked: Sequence[str],
                    frame_names: Sequence[str]) -> Tuple[List[ModelVerdict], List[str]]:
    """把模型回答解析成裁决（**抽帧**证据）。返回 ``(裁决, 丢掉了什么)``。

    硬校验：模型说的每一条都要过这一关 ——
    规则号不在被问的范围里 → **整条丢掉**（防止张冠李戴）；
    状态不认识 → 降级成 ``undetermined``；
    「做到了 / 没做到」却没有理由 → 降级成 ``undetermined``；
    引用的图号不存在 → **去掉那个图号**（引用不存在的证据比不引用更糟）。
    """
    items, drops = _verdict_items(raw)
    allowed = set(asked)
    out: List[ModelVerdict] = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        verdict = _verdict_from_item(item, allowed, seen, drops)
        if verdict is None:
            continue
        frames: List[str] = []
        for num in (item.get("frames") or []):
            try:
                idx = int(num)
            except (TypeError, ValueError):
                continue
            if idx < 1 or idx > len(frame_names):
                continue
            name = frame_names[idx - 1]
            if name not in frames:
                frames.append(name)
        verdict.frames = frames
        out.append(verdict)
    return out, drops


def parse_verdicts_video(raw: Any, asked: Sequence[str], clip_start: float,
                          clip_scale: float = 1.0
                          ) -> Tuple[List[ModelVerdict], List[str]]:
    """把模型回答解析成裁决（**视频片段**证据）。

    和抽帧那一版的差别只有一处：模型报的是「这段录像里的第几秒」
    （字段 ``at``），这里换算成**原视频**的秒。

    换算按比例：片段重编码过，实际时长跟原视频那几秒可能差一点点，
    所以用 ``clip_start + at * clip_scale``。
    ``clip_scale`` = 原片段长 / 编码后长（一般接近 1）。
    """
    items, drops = _verdict_items(raw)
    allowed = set(asked)
    out: List[ModelVerdict] = []
    seen = set()
    scale = float(clip_scale or 1.0)
    for item in items:
        if not isinstance(item, dict):
            continue
        verdict = _verdict_from_item(item, allowed, seen, drops)
        if verdict is None:
            continue
        seconds: List[float] = []
        raw_at = item.get("at")
        if raw_at is not None:
            try:
                at = float(raw_at)
            except (TypeError, ValueError):
                at = None
            if at is not None:
                seconds.append(float(clip_start) + at * scale)
        verdict.evidence_seconds = seconds
        out.append(verdict)
    return out, drops


# ---------------------------------------------------------------- 编排


def _load_frames(frames_dir: str, names: Iterable[str]) -> List[Tuple[str, bytes]]:
    """读证据帧。读不到就**跳过**（绝不拿一张别的图去替）。"""
    out: List[Tuple[str, bytes]] = []
    if not frames_dir:
        return out
    for name in names:
        path = os.path.join(frames_dir, name)
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError:
            continue
        if data:
            out.append((name, data))
    return out


def _group_candidates(candidates: Sequence[Candidate], beats: Sequence[Beat],
                      max_rules: int, max_frames: int
                      ) -> List[List[Candidate]]:
    """把候选分批：**时间上挨着的放一批**。

    一批里的规则共用一次调用 —— 共用画面，token 省。
    但一批的规则必须在时间上挨着，否则要给模型看的画面就太杂了
    （一次调用 10 张图，要覆盖 3 分钟和 13 分钟两处，模型分不清谁是谁的）。
    """
    ordered = sorted(candidates, key=lambda c: (c.span[0], c.rule_id))
    groups: List[List[Candidate]] = []
    current: List[Candidate] = []
    for cand in ordered:
        trial = current + [cand]
        lo = min(c.span[0] for c in trial)
        hi = max(c.span[1] for c in trial)
        n_frames = sum(len(b.frames) for b in beats
                       if b.game_time_end >= lo and b.game_time_start <= hi)
        if current and (len(trial) > max_rules or n_frames > max_frames):
            groups.append(current)
            current = [cand]
        else:
            current = trial
    if current:
        groups.append(current)
    return groups


def _beats_in_span(beats: Sequence[Beat], lo: float, hi: float) -> List[Beat]:
    out = [b for b in beats
           if b.game_time_end >= lo and b.game_time_start <= hi]
    out.sort(key=lambda b: b.game_time)
    return out


def _pick_frames(beats: Sequence[Beat], limit: int) -> List[str]:
    """从一段里**均匀**取若干张帧。

    均匀取而不是取前 N 张 —— 前 N 张全是同一个动作，后面的就看不到了。
    """
    names: List[str] = []
    for b in beats:
        for name in b.frames:
            if name not in names:
                names.append(name)
    if len(names) <= limit:
        return names
    step = len(names) / float(limit)
    picked: List[str] = []
    for i in range(limit):
        idx = int(i * step)
        idx = min(idx, len(names) - 1)
        if names[idx] not in picked:
            picked.append(names[idx])
    return picked


def _clip_range(beats: Sequence[Beat], cfg: AdjudicatorConfig
                ) -> Tuple[float, float]:
    """这一批候选要看**原视频的哪一段**。

    规则的时间窗是**局内**秒，而录像是**原视频**秒 —— 两边要对上，
    靠的是每段 Beat 上的 ``video_time_start/end``（模型看视频时
    报的就是原视频秒）。

    前后各留一点余量：前提可能刚好发生在窗口边上。
    """
    los = [float(b.video_time_start) for b in beats
           if float(b.video_time_start or 0) > 0]
    his = [float(b.video_time_end) for b in beats
           if float(b.video_time_end or 0) > 0]
    lo = min(los) if los else 0.0
    hi = max(his) if his else 0.0
    if hi <= lo:
        pts = [float(b.video_time) for b in beats if float(b.video_time or 0) > 0]
        if pts:
            lo = min(pts)
            hi = max(pts)
    lo = max(0.0, lo - float(cfg.clip_padding))
    hi = float(hi) + float(cfg.clip_padding)
    span = float(cfg.clip_max_seconds)
    if span > 0 and hi - lo > span:
        mid = (lo + hi) / 2.0
        lo = max(0.0, mid - span / 2.0)
        hi = lo + span
    return lo, hi


def _attach_thumbnails(verdicts: Sequence[ModelVerdict],
                       span_beats: Sequence[Beat], cfg: AdjudicatorConfig,
                       transform=None, counter: Optional[List[int]] = None,
                       fallback_second: float = 0.0) -> None:
    """给每条裁决配一张**代表帧**（报告里的缩略图）。

    模型判读用的是**录像**，但报告上要有一张能一眼看到「说的是哪一秒」
    的图 —— 所以在模型指出的那一刻抽一张，**挂到对应的那一段上**
    （``Beat.frames`` 与 ``detail["frame_times"]`` 必须一一对应）。

    **模型没指出时刻怎么办**
    ----------------------
    字段 ``at`` 是「取不到就写 null」—— 所以它经常是空的。但**空不能
    等于没有证据**：报告那边看到「这条结论没有对应的画面依据」就会把
    这条**丢掉**（实测：8 条结论全部被丢）。所以拿**片段的中点**兜底 ——
    结论本来就是关于这一整段的，指向中点不算撒谎。
    """
    if not cfg.frames_dir or not cfg.video_path:
        return
    from . import videopack

    for verdict in verdicts:
        second = 0.0
        if verdict.evidence_seconds:
            second = float(verdict.evidence_seconds[0])
        elif fallback_second > 0:
            second = float(fallback_second)
        if second <= 0:
            continue
        beat = None
        for b in span_beats:
            lo = float(b.video_time_start or 0)
            hi = float(b.video_time_end or 0)
            if lo <= second <= hi or (hi <= lo and abs(lo - second) < 5.0):
                beat = b
                break
        index = counter[0] if counter else 0
        if counter is not None:
            counter[0] += 1
        thumbs = videopack.save_thumbnails(
            cfg.video_path, [second], cfg.frames_dir,
            prefix="adj", index=index, transform=transform)
        # 存不下来时那个位置是 None（列表与时刻一一对应）——
        # 别把它当成「有一张图」。
        if not thumbs or not thumbs[0]:
            continue
        name, actual = thumbs[0]
        verdict.frames = [name]
        if beat is not None:
            beat.frames.append(name)
            detail = beat.detail or {}
            times = list(detail.get("frame_times") or [])
            times.append(float(actual))
            detail["frame_times"] = times
            beat.detail = detail


def adjudicate(timeline: Timeline, candidates: Sequence[Candidate],
               config: Optional[AdjudicatorConfig] = None,
               transform=None) -> ModelJudgement:
    """让模型逐条裁决候选规则。

    ``candidates`` 是**候选检索 skill** 的产物
    （:func:`app.recognition2.selector.select_candidates` ——
    模型自己写检索请求，RAG 去取攻略原文；退路是整库筛一遍）——
    这里只判「这一局要检验的那几条」，不判整库。

    证据默认是**视频片段**（模型看连续画面）；切不出片段时退回抽帧，
    并把这件事写进留痕。

    每一批候选一次调用；一次调用失败**不吞** —— 错误原因写进 ``notes``，
    那批候选降级成「判不了」，绝不假装分析过了。
    """
    from app.collection import settings, vlm_client
    from . import videopack

    cfg = config or AdjudicatorConfig()
    out = ModelJudgement()
    if not candidates:
        out.notes.append("这一局没有要检验的规则（筛子全筛掉了）—— 没有裁决。")
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
        out.notes.append("没有配置「%s」这个槽位的模型 —— 没有裁决。" % cfg.slot)
        return out
    if not model:
        out.notes.append("没有配置「%s」这个槽位的模型 —— 没有裁决。" % cfg.slot)
        return out

    beats = [b for b in timeline.sorted_beats()
             if b.game_time_start >= 0 and b.game_time_end >= 0]
    groups = _group_candidates(candidates, beats,
                               cfg.max_rules_per_call, cfg.max_frames_per_call)
    thumb_counter = [0]

    for group_no, group in enumerate(groups):
        if cfg.max_calls and out.calls >= cfg.max_calls:
            out.notes.append("达到调用上限（%d 次），剩下 %d 条候选没裁决。"
                             % (cfg.max_calls, len(groups) - out.calls))
            break
        if cfg.budget_tokens and out.tokens >= cfg.budget_tokens:
            out.notes.append("达到 token 预算（%d），剩下 %d 条候选没裁决。"
                             % (cfg.budget_tokens, len(groups) - out.calls))
            break

        lo = min(c.span[0] for c in group)
        hi = max(c.span[1] for c in group)
        span_beats = _beats_in_span(beats, lo, hi)

        # -- 证据：优先给**视频片段** --
        use_video = bool(cfg.evidence == "video" and cfg.video_path)
        clip_path = ""
        clip_lo = clip_hi = 0.0
        clip_scale = 1.0
        if use_video:
            clip_lo, clip_hi = _clip_range(span_beats, cfg)
            if clip_hi > clip_lo and cfg.video_dir:
                os.makedirs(cfg.video_dir, exist_ok=True)
                clip_path = os.path.join(cfg.video_dir,
                                         "clip_%03d.mp4" % group_no)
                info = videopack.cut_clip(
                    cfg.video_path, clip_lo, clip_hi, clip_path,
                    fps=cfg.clip_fps, transform=transform)
                if info.size <= 0:
                    clip_path = ""
                else:
                    encoded = float(info.duration or 0.0)
                    if encoded > 0:
                        clip_scale = (clip_hi - clip_lo) / encoded
            if not clip_path:
                use_video = False
                out.notes.append("切不出这一段的录像 —— 改用抽帧判读。")

        frame_names: List[str] = []
        loaded: List[Tuple[str, bytes]] = []
        if not use_video:
            frame_names = _pick_frames(span_beats, cfg.max_frames_per_call)
            loaded = _load_frames(cfg.frames_dir, frame_names)
            if loaded:
                frame_names = [name for name, _ in loaded]
            else:
                loaded = []

        asked = [c.rule_id for c in group]
        if use_video:
            content = [vlm_client.text_block(build_video_prompt(
                group, span_beats, clip_lo, clip_hi, lo, hi,
                situation=cfg.situation))]
            try:
                with open(clip_path, "rb") as fh:
                    data = fh.read()
                content.append(vlm_client.video_block(
                    data, fps=cfg.clip_model_fps, detail=cfg.clip_detail))
            except OSError as exc:
                out.notes.append("读不出切好的片段：%s —— 改用抽帧判读。" % exc)
                use_video = False
                frame_names = _pick_frames(span_beats, cfg.max_frames_per_call)
                loaded = _load_frames(cfg.frames_dir, frame_names)
                if loaded:
                    frame_names = [name for name, _ in loaded]
                else:
                    loaded = []
                content = [vlm_client.text_block(build_prompt(
                    group, span_beats, frame_names))]
                for _name, blob in loaded:
                    content.append(vlm_client.image_block(blob))
        else:
            content = [vlm_client.text_block(
                build_prompt(group, span_beats, frame_names,
                             situation=cfg.situation))]
            for _name, blob in loaded:
                content.append(vlm_client.image_block(blob))

        try:
            result = vlm_client.call_model(
                model, content, creds=creds, system=SYSTEM_PROMPT,
                max_tokens=cfg.max_tokens, timeout=cfg.timeout)
        except vlm_client.VlmError as exc:
            out.notes.append("%d 条候选没裁决：%s"
                             % (len(group), vlm_client.explain_error(exc)))
            for cand in group:
                out.verdicts.append(ModelVerdict(
                    rule_id=cand.rule_id, status=STATUS_UNDETERMINED,
                    span=tuple(cand.span),
                    code_status=cand.code_status,
                    reason="模型没调用成功（%s）—— 判不了，不算他没做。"
                           % vlm_client.explain_error(exc)))
            continue

        out.calls += 1
        used = _token_count(result.usage or {})
        out.tokens += used

        raw = None
        try:
            raw = result.json()
        except vlm_client.VlmError as exc:
            out.notes.append("%d 条候选没裁决：模型回答不是合法 JSON（%s）"
                             % (len(group), exc))
        if raw is None:
            for cand in group:
                out.verdicts.append(ModelVerdict(
                    rule_id=cand.rule_id, status=STATUS_UNDETERMINED,
                    span=tuple(cand.span),
                    code_status=cand.code_status,
                    reason="模型这次没给出可解析的回答 —— 判不了，不算他没做。"))
            continue

        if use_video:
            verdicts, drops = parse_verdicts_video(raw, asked, clip_lo,
                                                   clip_scale)
            _attach_thumbnails(verdicts, span_beats, cfg, transform,
                               thumb_counter,
                               fallback_second=(clip_lo + clip_hi) / 2.0)
        else:
            verdicts, drops = parse_verdicts(raw, asked, frame_names)
        out.notes.extend(drops)
        # **把检验区间带在裁决上** —— 报告要靠它把这条落到某一段时间上。
        # 没有它，裁决会落进每一个时间片，报告上每条结论的「对局时间区间」
        # 都变成第一段，而证据帧明明在第 8 分钟。
        span_by_rule = {c.rule_id: tuple(c.span) for c in group}
        code_by_rule = {c.rule_id: c.code_status for c in group}
        for v in verdicts:
            if not v.span or not any(v.span):
                v.span = span_by_rule.get(v.rule_id, (0.0, 0.0))
            if not v.code_status:
                v.code_status = code_by_rule.get(v.rule_id, "")
        got = {v.rule_id for v in verdicts}
        for cand in group:
            if cand.rule_id not in got:
                out.verdicts.append(ModelVerdict(
                    rule_id=cand.rule_id, status=STATUS_UNDETERMINED,
                    span=tuple(cand.span),
                    code_status=cand.code_status,
                    reason="模型这次没回答这一条 —— 判不了，不算他没做。"))
        out.verdicts.extend(verdicts)

    out.verdicts.sort(key=lambda v: v.rule_id)
    return out


def _token_count(usage: Dict[str, Any]) -> int:
    """从各种形状的 usage 里取 token 数。"""
    total = 0
    for key in ("input_tokens", "output_tokens", "prompt_tokens",
                "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, (int, float)):
            total += int(value)
    return total
