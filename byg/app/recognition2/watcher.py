"""**看视频**那一层：让模型**直接看录像**，自己讲这一局发生了什么。

这一层替代了什么
----------------
旧链路是「切段 → 抽帧 → 逐段问模型」。它有三个结构性问题：

1. **模型看不见连续动作**。抽出来的几帧是静止的、彼此无关的 ——
   「这一下是不是打到了」「是不是交了位移」「是不是在被打」这些
   **只有在连续画面里才看得出来**。
2. **每段都是孤立的**。模型第 3 段不知道第 2 段发生了什么，
   于是只能描述局部，串不成一局。
3. **一次调用一句结论**，整局要几十次调用，既贵又碎。

现在改成：**把整局录像给模型看一次**，让它自己讲 —— 从头到尾的
节奏、每个阶段在干什么、这个玩家是什么风格。这一层出来的就是
**模型自己的语言**，不是从词表里挑的标签。

两条纪律
--------
1. **看不清就写 ``unclear``**，不许编（``unclear`` 的事件照样进时间线，
   但判定时算「无法判定」，**永远不算「没做」**）。
2. **时间以视频秒为准**。模型报的是「这段录像的第几秒」，
   换算回原视频由代码做（见 :mod:`app.recognition2.videopack`）。

时间轴怎么来的
--------------
模型看不到顶栏那个 35×12 像素的计时器（视频一压更看不清），
所以**局内时刻不靠它读钟**，靠两样东西：

- **对局起点**：模型说「进到游戏画面」是在第几秒；
- **对照读数**：模型要是能读出顶栏时刻，就报上来，代码取**中位数**
  反推起点（多个读数比模型嘴里的一个数字准）。

两者都没有 → 局内时刻读不出来 → 带时间窗的规则一条都判不了，
这件事要**写进留痕**，不能装作分析过了。
"""

from __future__ import annotations

import json
import os
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .beat import PHASE_IN_GAME, PHASE_POST_GAME, PHASE_PRE_GAME, PHASE_UNKNOWN
from .watch_prompt import (
    FRAMES_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_prompt,
)
from .lane_evidence import LANE_VALUES, normalize_lane, normalize_position
from .selector import RetrievalRequest, parse_requests

#: 模型可以给的阶段。
PHASES = (PHASE_IN_GAME, PHASE_PRE_GAME, PHASE_POST_GAME, PHASE_UNKNOWN)


# ---------------------------------------------------------------- 配置


@dataclass
class WatchConfig:
    """看视频这一层的参数。"""

    #: 走哪个模型槽位（默认「vlm」）。
    slot: str = "vlm"
    #: 模型名。留空 = 从配置里读当前槽位的模型。
    model: str = ""
    #: 数据目录（配置与用户数据在这）。
    data_dir: str = ""
    #: 单次调用的 token 上限。
    #:
    #: **8192 不够**（2026-11-02 18:24 实测）：模型给一局 19 分钟的录像
    #: 认真写了 **104 段**（每段带叙述/动作/位置/时刻），正正好好写到
    #: ``output_tokens=8192`` 上限就被**截断在 JSON 中间** ——
    #: 整局一条都没读出来，报告只能是空的。
    #: 现在给到 16384；真再截断，:func:`salvage_events` 会把
    #: **已经写完**的那几段捞出来用（并写清楚只读到这么多）。
    max_tokens: int = 16384
    #: 单次调用的超时（秒）。视频比图片慢得多。
    timeout: float = 900.0
    #: **模型看片的采样帧率**（每秒给模型看几帧）。0.2~5。
    #:
    #: 越小越省、越容易漏动作；越大越贵。整局看用 0.5，
    #: 裁决单条规则时看片段用 3~4。
    fps: float = 0.5
    #: 细节档：``low`` / ``default`` / ``high``。
    detail: str = "default"
    #: 打包目录（整局视频、切片都落这儿）。
    video_dir: str = ""
    #: 证据帧目录（报告里的缩略图）。
    frames_dir: str = ""
    #: 打包出来的体积上限。超过就走 Files 上传。
    max_upload_bytes: int = 480 * 1024 * 1024
    #: 允许先上传到对方的 Files（>45 MB 的视频只能走这条）。
    allow_upload: bool = True
    #: 整局最多调多少次模型。
    max_calls: int = 4
    #: 整局最多花多少 token（0 = 不限）。
    budget_tokens: int = 0
    #: 打包时用的帧率（模型看片的采样率与它无关）。
    pack_fps: float = 2.0
    #: **玩家自己填的英雄 / 分路** —— 填了就以它为准，模型不必再猜。
    hero: str = ""
    position: str = ""
    #: 用顶栏 K/D/A 校验击杀/阵亡（队友击杀不能算成玩家的）。
    verify_events: bool = True
    #: 从小地图读「在哪条路」，替掉模型猜的位置。
    location_fact: bool = True
    #: 关键时刻要不要「±30 秒、一秒一帧」再细看一遍（费调用，但过程清楚）。
    detail_events: bool = True
    #: 细看窗口的半径（秒）。
    detail_span: float = 30.0


# ---------------------------------------------------------------- 结果


@dataclass
class WatchEvent:
    """模型讲的一个事件。"""

    #: 原视频里的秒数（区间）。
    video_start: float = 0.0
    video_end: float = 0.0
    #: **模型自己的话** —— 这一段的正文。
    narrative: str = ""
    #: 动作码（顺手归的类）；看不清就是 ``unclear``。
    action: str = "unclear"
    action_label: str = ""
    #: 位置码（规范值）或 ``unknown``。
    position: str = "unknown"
    #: **这一段玩家在哪条路**（结构化标注）。
    #:
    #: 为什么单独要一个字段：顶层那个 ``position`` 只是模型**随口说的一个词**，
    #: 错了没人拦（2026-11-07 实测：它说「中路」，叙述里却全是「上路河道」
    #: 「上路一塔」）。逐段标路，裁决层才有几十条证据可对账。
    lane: str = "unknown"
    #: 具体地点（自由描述）。
    where: str = ""
    #: 阶段：对局中 / 对局前 / 结算。
    phase: str = PHASE_UNKNOWN
    #: 置信度 0~1。
    confidence: float = 0.0
    #: 模型读到的局内时刻（秒）。读不到就是 None。
    clock: Optional[float] = None
    #: 三段中的哪一段：early / mid / late（按**局内时间**分）。
    stage: str = ""
    #: 四类事件之一：death / tower_lost / kill / tower_taken。
    event_type: str = ""
    #: 判断这条事件依据的那几帧（录像秒数）。
    frames: List[float] = field(default_factory=list)
    #: 细看有没有给出可验证证据（拆塔/丢塔：塔血归零 + 归属 + 路）。
    detail_verified: bool = True

    @property
    def is_unclear(self) -> bool:
        """这一段是不是**没读出来**。"""
        return not (self.narrative or "").strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_start": round(float(self.video_start), 2),
            "video_end": round(float(self.video_end), 2),
            "narrative": self.narrative,
            "action": self.action,
            "action_label": self.action_label,
            "position": self.position,
            "lane": self.lane,
            "where": self.where,
            "phase": self.phase,
            "confidence": round(float(self.confidence), 3),
            "clock": (round(float(self.clock), 2)
                      if self.clock is not None else None),
            "stage": self.stage,
            "event_type": self.event_type,
            "frames": [round(float(x), 2) for x in self.frames],
            "detail_verified": bool(self.detail_verified),
        }


@dataclass
class Death:
    """玩家自己的一次阵亡。

    为什么单独要这个
    ----------------
    失败对局最该回答的是「**怎么死的，下次怎么办**」。而「怎么死的」
    只有模型看完整局才拿得到：事件叙述里写的是「他被包夹了」，
    不是「因为视野没做、站位靠前才被包夹」。

    它跟 :class:`WatchEvent` 分开，是因为**一次死亡横跨好几段** ——
    事件是时间片，死亡是一个时刻。
    """

    #: 原视频里的秒数（死亡那一刻）。
    video_start: float = 0.0
    #: 死在哪儿（自由描述）。
    where: str = ""
    #: **怎么死的** —— 模型看到的情况（被谁先手、几个人、在哪被打的）。
    cause: str = ""
    #: 玩家自己哪一步出的问题。**看不出就空**（不猜）。
    fault: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_start": round(float(self.video_start), 2),
            "where": self.where,
            "cause": self.cause,
            "fault": self.fault,
        }


@dataclass
class WatchResult:
    """整局「看」的结果。"""

    events: List[WatchEvent] = field(default_factory=list)
    #: **这局玩家自己死了几次** —— 每次一条（模型看整局时数的）。
    #:
    #: 失败对局的复盘是从这里起头的：从最后一次死亡往前推，
    #: 看每次是怎么死的、该怎么做。读不到就空数组（**不猜**）。
    deaths: List[Death] = field(default_factory=list)
    #: 玩家操控的英雄（模型认的）。认不出就是空。
    hero: str = ""
    #: 玩家的分路（规范值）。定不出就是 ``unknown``。
    position: str = "unknown"
    #: 这一局的胜负（``win`` / ``lose`` / ``unknown``）。
    #:
    #: 为什么在这里读：``prefill.read_result`` 单独抽**最后几帧**去问胜负，
    #: 而结算画面出现后录屏往往还录了十几秒（玩家点掉结算、回到主界面）——
    #: 那几帧就落在结算画面**之前**，于是「明明画面上写着失败」却读成 unknown
    #: （2026-11-07 实测：叙述里已经写「水晶爆裂并出现失败画面」，结果栏却空着）。
    #:
    #: 看整局这次**把整局都看过了**，结算画面就在里面 —— 顺手读一次，
    #: 不多花一次调用，两处读数还能对账。
    result: str = "unknown"
    #: 对局在**原视频**里从第几秒开始。读不到就是 None。
    match_start: Optional[float] = None
    #: 模型写的整局综述（几句话）。
    summary: str = ""
    #: 模型自己写的**检索请求**（「我想查什么」）。
    #:
    #: 这一步以前是代码按作用域/英雄/时间窗整条筛（1 680 条）——
    #: 筛出来的与这一局发生了什么无关。现在由模型自己决定查什么，
    #: 代码只负责去 RAG 库里把攻略取回来（见 :mod:`app.recognition2.selector`）。
    requests: List[RetrievalRequest] = field(default_factory=list)
    #: 留痕（失败原因、丢掉的回答…）。
    notes: List[str] = field(default_factory=list)
    calls: int = 0
    tokens: int = 0
    #: 打包信息（体积、编码、有没有上传）。
    pack: Dict[str, Any] = field(default_factory=dict)
    uploaded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "events": [e.to_dict() for e in self.events],
            "deaths": [d.to_dict() for d in self.deaths],
            "hero": self.hero,
            "position": self.position,
            "result": self.result,
            "match_start": (round(float(self.match_start), 2)
                            if self.match_start is not None else None),
            "summary": self.summary,
            "requests": [r.to_dict() for r in self.requests],
            "notes": list(self.notes),
            "calls": self.calls,
            "tokens": self.tokens,
            "pack": dict(self.pack),
            "uploaded": bool(self.uploaded),
        }


# 提示词见 app/recognition2/watch_prompt.py（这一层只负责调用与解析）。


# ---------------------------------------------------------------- 解析


def parse_clock(value: Any) -> Optional[float]:
    """把模型给的时刻解析成秒。解析不了返回 None（不猜）。"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in ("null", "none", "unknown", "n/a", "-"):
        return None
    if ":" in text:
        head, _, tail = text.partition(":")
        try:
            return int(head) * 60 + int(tail)
        except ValueError:
            pass
    try:
        return float(text)
    except ValueError:
        return None


def _norm_result(value: Any) -> str:
    """把模型写的胜负**归一成三个值**（``win`` / ``lose`` / ``unknown``）。

    为什么只收三个
    --------------
    下游（报告顶上的结果栏）是 ``{'失败' if result == 'lose' else '胜利'}``
    这种**判别式渲染** —— 任何没被认出来的非空值都会被渲染成「胜利」。
    所以这里必须**白名单**：认不出来就是 ``unknown``，不猜、不透传。
    """
    text = str(value or "").strip().lower()
    if text in ("win", "victory", "胜利", "胜"):
        return "win"
    if text in ("lose", "loss", "defeat", "失败", "输"):
        return "lose"
    return "unknown"


def _first_float(data: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        if key not in data:
            continue
        value = data.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            continue
    return None


def salvage_head(text: str) -> Dict[str, Any]:
    """从**被截断**的回答里，捞出 ``events`` 之前的那些顶层字段。

    为什么要有这个
    --------------
    提示词里 JSON 的字段顺序是**我们定的**，那就得按它的意思用：
    ``match_start`` / ``hero`` / ``position`` / ``queries`` / ``summary``
    这些**短的**写在前面，``events``（长的）写在最后。
    服务端把回答截断时，断点几乎总是落在 ``events`` 数组中间 ——
    于是前面那几个字段**模型明明已经写了**，却跟着整份丢掉。

    2026-11-03 实测：那一局的回答在第 17 432 字符处断掉，兜底捞回了 86 段，
    但**英雄与分路是空的** —— 其实模型早就答了「敖隐 / carry」。
    英雄层因此一条都不参与，报告只好拿分路层的打法凑。

    捞法
    ----
    只在 ``"events"`` **之前**那一段里找各个键的值（后面那些是事件内容，
    里面有同名的字样，会认错）。
    值写的是 ``null`` 就当**没读到**（不猜）；值没写完（字符串没闭合）也丢掉。

    这样一来，回答就算被截断，**模型的判断仍然算数** ——
    这正是「看完整局就由模型决定」能成立的前提。
    """
    if not text:
        return {}
    cut = text.find('"events"')
    head = text if cut < 0 else text[:cut]
    out: Dict[str, Any] = {}
    for key in ("match_start", "hero", "position", "result"):
        value = _head_value(head, key)
        if value is not None:
            out[key] = value
    # ``queries`` 是数组：只有**整个数组写完了**才捞（断在半截的不要）。
    m = re.search(r'"queries"\s*:\s*(\[.*?\])', head, re.S)
    if m:
        try:
            arr = json.loads(m.group(1))
        except ValueError:
            arr = None
        if isinstance(arr, list):
            out["queries"] = arr
    # ``deaths`` 同理 —— 死亡清单也写在 ``events`` 之前，断在后面就捞不到。
    m = re.search(r'"deaths"\s*:\s*(\[.*?\])', head, re.S)
    if m:
        try:
            arr = json.loads(m.group(1))
        except ValueError:
            arr = None
        if isinstance(arr, list):
            out["deaths"] = arr
    return out


def _parse_deaths(raw: Any, scale: float = 1.0, offset: float = 0.0) -> List[Death]:
    """把模型给的 ``deaths`` 解析成死亡清单。**读不出来就是空**（不猜）。

    模型可能把时间写在 ``at`` 里（提示词里 ``at`` 是「死在哪」）——
    数值型的 ``at`` 当时间用，文本型的当地点用。

    **时刻用同一套换算**（``scale`` / ``offset``）—— 与事件时刻一致，
    否则死亡点会落到别的段落上，接不上画面依据。
    """
    if not isinstance(raw, dict):
        return []
    items = raw.get("deaths")
    if not isinstance(items, list):
        items = []
    out: List[Death] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        t = _first_float(item, ("t", "time", "at_seconds"))
        where = str(item.get("at") or "").strip()
        if t is None and where:
            # at 写成了秒数 —— 当时间用，地点留空。
            t = _first_float({"t": where}, ("t",))
            if t is not None:
                where = ""
        if t is None:
            continue
        out.append(Death(
            video_start=float(t) * float(scale or 1.0) + float(offset or 0.0),
            where=clean_words(where),
            cause=clean_words(str(item.get("cause") or "").strip()),
            fault=clean_words(str(item.get("fault") or "").strip()),
        ))
    # 新提示词把死亡写在 stages 里（type == death）—— 一并收进来，
    # 否则「怎么死的」这一层就空了。
    seen = [d.video_start for d in out]
    for _stage, item in _iter_raw_events(raw):
        if str(item.get("type") or "").strip().lower() != "death":
            continue
        t = _first_float(item, ("t", "time"))
        if t is None:
            continue
        when = float(t) * float(scale or 1.0) + float(offset or 0.0)
        if any(abs(when - x) <= 2.0 for x in seen):
            continue
        seen.append(when)
        out.append(Death(
            video_start=when,
            where=clean_words(str(item.get("at") or item.get("where") or "").strip()),
            cause=clean_words(str(item.get("cause")
                                  or item.get("narrative") or "").strip()),
            fault=clean_words(str(item.get("fault") or "").strip()),
        ))
    return out


def _head_value(head: str, key: str) -> Any:
    """在 ``head`` 里找 ``"key": 值`` 的值（写完的才算）。"""
    m = re.search(
        r'"%s"\s*:\s*("(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?|null)' % re.escape(key),
        head)
    if not m:
        return None
    raw = m.group(1)
    if raw == "null":
        return None
    if raw.startswith('"'):
        try:
            value = json.loads(raw)
        except ValueError:
            return None
        return value if isinstance(value, str) and value.strip() else None
    try:
        return float(raw)
    except ValueError:
        return None


def salvage_events(text: str) -> List[Dict[str, Any]]:
    """从**被截断**的回答里，捞出已经写完的那些事件。

    为什么要有这个
    --------------
    模型看整局录像，回答可能**超上限被截断**（2026-11-02 18:24 实测：
    20 422 字符处 JSON 断在半截，于是整局一条都没读出来，
    报告只能是空的）。整份丢掉太浪费 —— 断掉之前写完的那十几段
    是**完整的**，能用。

    怎么捞
    ------
    找到 ``"events"`` 后面那个 ``[``，然后**逐字符**扫：
    记住自己在不在字符串里（字符串里的 ``{`` ``}`` 不算），
    每凑齐一个完整的 ``{...}`` 就单独 ``json.loads`` 一次。
    断在半截的那个对象自然凑不齐 —— 直接丢掉，不猜。

    捞到的东西**只是这一局的残片**：报告里必须说清楚读到了几段、
    后面没读到（调用方把这件事写进留痕）。
    """
    if not text:
        return []
    key = text.find('"events"')
    if key < 0:
        return []
    bracket = text.find("[", key)
    if bracket < 0:
        return []
    out: List[Dict[str, Any]] = []
    depth = 0
    in_str = False
    escape = False
    obj_start = -1
    for i in range(bracket + 1, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and obj_start >= 0:
                    try:
                        obj = json.loads(text[obj_start:i + 1])
                    except ValueError:
                        obj = None
                    if isinstance(obj, dict):
                        out.append(obj)
                    obj_start = -1
        elif ch == "]" and depth == 0:
            break
    return out


#: 三段（与 app/recognition2/watch_prompt.py 的分法一致）。
STAGE_NAMES = ("early", "mid", "late")

#: 三段的中文名（拼整局摘要用）。
STAGE_LABELS = {"early": "前期", "mid": "中期", "late": "后期"}

#: 敌方阵营的写法统一成「敌方英雄」（模型常写红名/红方/对面）。
_ENEMY_WORDS = (
    "红名英雄", "红方英雄", "蓝方英雄", "对面英雄",
    "红方射手", "敌方射手", "对面射手", "红名", "红方英雄",
)


def stage_of_game_seconds(seconds: Optional[float]) -> str:
    """局内秒数 -> early / mid / late（0-5 / 5-10 / 10+）。"""
    if seconds is None:
        return ""
    minute = float(seconds) / 60.0
    if minute < 5.0:
        return "early"
    if minute < 10.0:
        return "mid"
    return "late"


_HERO_NAMES_CACHE: Optional[set] = None


def known_hero_names() -> set:
    """王者英雄名清单（来自适配器的种子表）。认不出就空集，不挡主流程。"""
    global _HERO_NAMES_CACHE
    if _HERO_NAMES_CACHE is None:
        try:
            from ..adapters.honor_of_kings import HEROES_BY_LANE
            _HERO_NAMES_CACHE = {name for names in HEROES_BY_LANE.values()
                                 for name in names if name}
        except Exception:                              # noqa: BLE001
            _HERO_NAMES_CACHE = set()
    return _HERO_NAMES_CACHE


def clean_words(text: str, hero: str = "") -> str:
    """把称呼统一：敌方 -> 「敌方英雄」，玩家 -> 英雄名。

    提示词已经禁止这些写法，但模型偶尔照旧 —— 报告上的措辞必须一致，
    所以这里再兜一层（只改称呼，不改事实）。
    """
    out = str(text or "")
    for word in _ENEMY_WORDS:
        out = out.replace(word, "敌方英雄")
    out = out.replace("敌方的敌方英雄", "敌方英雄")
    out = out.replace("敌方英雄英雄", "敌方英雄")
    if hero:
        # 玩家统一用英雄名；「其他」「他们」不动。
        out = re.sub(r"(?<!其)他(?!们)", hero, out)
        # 除了玩家自己，别的英雄名一律不写（敌方写「敌方英雄」）。
        # 单字英雄名（镜 / 澜 / 曜…）不替换 —— 「镜头」会被改成「敌方英雄头」。
        for name in sorted(known_hero_names(), key=len, reverse=True):
            if len(name) >= 2 and name != hero:
                out = out.replace(name, "敌方英雄")
    return out


_PLACE_LANES = ("上路", "中路", "下路", "对抗路", "发育路")


#: 叙述里不许出现时间数字（第 550 秒 / 7 分 20 秒）。
_TIME_REF = re.compile(
    r"第?\s*\d{1,3}\s*秒(?:钟)?|\d{1,2}\s*分\s*\d{1,2}\s*秒")


#: 叙述里不写路名（位置由系统按小地图标在标题上），避免「标题写上路、正文写下路」。
_LANE_WORDS = re.compile(
    r"在?(?:己方|敌方)?(?:上路|中路|下路)(?:附近|一带|区域|侧)?")


#: 模型认不出技能 —— 叙述里不写具体技能（大招/技能名/位移/闪现）。
def strip_skills(text: str) -> str:
    out = str(text or "")
    out = out.replace("技能连招", "输出").replace("控制技能", "控制")
    for word in ("大招", "一技能", "二技能", "三技能", "位移", "闪现",
                 "技能", "施放", "释放"):
        out = out.replace(word, "")
    return out


def strip_lane_words(text: str) -> str:
    """把叙述里的路名抹掉（保留河道、野区这类区域词）。"""
    return _LANE_WORDS.sub("", str(text or ""))


def strip_time_refs(text: str) -> str:
    """把叙述里的时间数字抹掉（展示层已经统一标局内时间）。"""
    return _TIME_REF.sub("", str(text or ""))


def clean_place(text: str) -> str:
    """地点只说「河道 / 河道草丛 / 野区 / 野区草丛」；阵营说 我方/敌方。

    「中路河道」这种带分路的写法一律抹掉路名（用户 2026-11-14 定的口径），
    红方 / 蓝方换成敌方 / 我方。
    """
    out = str(text or "")
    for lane in _PLACE_LANES:
        out = out.replace(lane + "河道", "河道")
        out = out.replace(lane + "野区", "野区")
        out = out.replace(lane + "草丛", "草丛")
    out = out.replace("红方", "敌方").replace("蓝方", "我方")
    return out


#: where 只允许这几种写法（2026-11-14 用户口径）。
PLACE_VALUES = ("河道草丛", "河道", "野区草丛", "野区", "防御塔")


def canonical_place(text: str) -> str:
    """where 归一成「己方/敌方 + 上路/中路/下路」，或「河道/野区」。

    不知道归属就只写路名；只有「塔」没有路就留空（不编）。
    """
    raw = str(text or "")
    side = ""
    if "敌方" in raw:
        side = "敌方"
    elif "己方" in raw or "我方" in raw:
        side = "己方"
    for key in ("上路", "中路", "下路"):
        if key in raw:
            return side + key if side else key
    if "河道" in raw:
        return "河道"
    if "野区" in raw:
        return "野区"
    return ""


def apply_stages(events: List["WatchEvent"],
                 match_start: Optional[float]) -> None:
    """按**局内时间**重新标段（clock 优先，其次 录像秒 - 对局起点）。

    模型按批次标段经常错（第 3 批被标成中期，其实局内才 2 分钟）——
    段由代码算，模型只负责给 clock。
    """
    for event in events:
        # **录像秒 - 对局起点**优先（模型读顶栏经常读错）；
        # 没有起点才退到它报的 clock。
        game_seconds = None
        if match_start is not None:
            game_seconds = float(event.video_start) - float(match_start)
        if game_seconds is None:
            game_seconds = event.clock
        stage = stage_of_game_seconds(game_seconds)
        if stage:
            event.stage = stage


def _stages_summary(raw: Dict[str, Any]) -> str:
    """把三段各自的 summary 拼成整局摘要。

    新提示词不再要顶层 summary —— 走势本来就是按三段讲的，
    代码把三句拼起来即可，省得模型重复写一遍。
    """
    stages = raw.get("stages")
    if not isinstance(stages, list):
        return ""
    parts: List[str] = []
    for st in stages:
        if not isinstance(st, dict):
            continue
        text = str(st.get("summary") or "").strip()
        if not text:
            continue
        label = STAGE_LABELS.get(str(st.get("stage") or "").strip().lower(), "")
        parts.append(("%s：%s" % (label, text)) if label else text)
    return "；".join(parts)

#: 每段要找的四类事件。
EVENT_TYPES = ("death", "kill", "tower_taken", "dragon")


def _iter_raw_events(raw: Dict[str, Any]):
    """按出现顺序吐出 (stage, item)。

    新提示词的形状是 stages[].events[]；旧的 events[] 仍然认 ——
    换提示词不必把历史快照一起改掉。
    """
    stages = raw.get("stages")
    if isinstance(stages, list):
        for st in stages:
            if not isinstance(st, dict):
                continue
            stage = str(st.get("stage") or "").strip().lower()
            if stage not in STAGE_NAMES:
                stage = ""
            for item in (st.get("events") or []):
                if isinstance(item, dict):
                    yield stage, item
    items = raw.get("events")
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            stage = str(item.get("stage") or "").strip().lower()
            yield (stage if stage in STAGE_NAMES else ""), item


def parse_events(raw: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    """把模型回答里的 stages[].events[] 解析成规整的事件列表。

    硬校验：不是字典的丢掉；时间读不出来的**保留但标出来**
    （它至少还带一句话，不能整条丢 —— 丢多了报告就成空表）。
    """
    drops: List[str] = []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return [], ["模型回答不是合法 JSON。"]
    if not isinstance(raw, dict):
        return [], ["模型回答不是 JSON 对象。"]
    if not isinstance(raw.get("stages"), list) and not isinstance(
            raw.get("events"), list):
        return [], ["模型回答里没有 stages / events 数组。"]
    # 记下「不是对象」的条目 —— 模型偶尔会夹带解释性文字。
    stages = raw.get("stages")
    for stage in (stages if isinstance(stages, list) else []):
        if not isinstance(stage, dict):
            continue
        inner = stage.get("events")
        for item in (inner if isinstance(inner, list) else []):
            if not isinstance(item, dict):
                drops.append("模型给了一条不是对象的 event，已丢掉。")
    top = raw.get("events")
    for item in (top if isinstance(top, list) else []):
        if not isinstance(item, dict):
            drops.append("模型给了一条不是对象的 event，已丢掉。")
    out: List[Dict[str, Any]] = []
    for stage, item in _iter_raw_events(raw):
        narrative = str(item.get("narrative") or item.get("text") or "").strip()
        t = _first_float(item, ("t", "t_start", "start", "time", "video_start"))
        t_end = _first_float(item, ("t_end", "end", "video_end"))
        if t is None and t_end is None:
            t = t_end = None
        elif t is None:
            t = t_end
        if t_end is None:
            t_end = t
        action = str(item.get("action") or item.get("primary_action")
                     or "unclear").strip() or "unclear"
        label = str(item.get("action_label") or "").strip()
        position = str(item.get("position") or "unknown").strip() or "unknown"
        phase = str(item.get("phase") or PHASE_UNKNOWN).strip() or PHASE_UNKNOWN
        if phase not in PHASES:
            phase = PHASE_UNKNOWN
        event_type = str(item.get("type") or item.get("event_type")
                         or "").strip().lower()
        if event_type not in EVENT_TYPES:
            event_type = ""
        frames: List[float] = []
        for value in (item.get("frames") or item.get("moments") or []):
            try:
                frames.append(float(value))
            except (TypeError, ValueError):
                continue
        try:
            confidence = float(item.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        out.append({
            "t": t,
            "t_end": t_end,
            "narrative": narrative,
            "action": action,
            "action_label": label,
            "position": position,
            "lane": normalize_lane(item.get("lane")),
            "where": str(item.get("where") or "").strip(),
            "phase": phase,
            "clock": parse_clock(item.get("clock")),
            "confidence": max(0.0, min(1.0, confidence)),
            "stage": stage,
            "event_type": event_type,
            "frames": frames,
        })
    return out, drops


# ---------------------------------------------------------------- 编排


def _token_count(usage: Dict[str, Any]) -> int:
    total = 0
    for key in ("input_tokens", "output_tokens", "prompt_tokens",
                "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, (int, float)):
            total += int(value)
    return total


def _resolve_match_start(events: Sequence[WatchEvent], guessed: Optional[float]
                         ) -> Tuple[Optional[float], str]:
    """定「对局从原视频第几秒开始」。

    优先用**对照读数**：模型读到的顶栏时刻 ``c`` 出现在原视频第 ``v`` 秒，
    那么对局起点就是 ``v - c``。多个读数取**中位数** —— 比模型嘴里的
    一个数字准（那个数字是它看一眼估的）。

    没有读数就用模型给的 ``match_start``。
    """
    diffs: List[float] = []
    for event in events:
        if event.clock is None:
            continue
        diffs.append(float(event.video_start) - float(event.clock))
    if len(diffs) >= 1:
        start = statistics.median(diffs)
        if abs(start) < 1e-9:
            start = 0.0        # 别写成「-0.0 秒」
        return start, "对照读数（%d 个）" % len(diffs)
    if guessed is not None:
        start = float(guessed)
        if abs(start) < 1e-9:
            start = 0.0
        return start, "模型口报"
    return None, ""


def watch_video(video_path: str,
                config: Optional[WatchConfig] = None,
                transform: Optional[Callable[[Any], Any]] = None,
                pack: Optional[Any] = None) -> WatchResult:
    """让模型**看一遍这局录像**，讲清楚这一局发生了什么。

    ``pack`` 可以传一个已经打好的包（:func:`app.recognition2.videopack.pack_video`
    的结果）；不传就现打一个。
    """
    from app.collection import settings, vlm_client
    from . import videopack

    cfg = config or WatchConfig()
    out = WatchResult()

    creds = None
    model = cfg.model or ""
    try:
        # **凭据与型号来自同一份配置** —— 同一个数据目录，
        # 否则一个来自这里、一个来自进程环境变量，两边可能不是一份。
        with settings.use_data_dir(cfg.data_dir or None):
            creds = vlm_client.find_credentials(cfg.slot)
            model = model or settings.model_name(cfg.slot)
    except Exception as exc:                           # noqa: BLE001
        out.notes.append("取不到「%s」槽位的模型配置：%s" % (cfg.slot, exc))
        return out
    if not creds or not model:
        out.notes.append("没有配置「%s」这个槽位的模型 —— 整局没有分析。"
                         % cfg.slot)
        return out

    # -- 这个模型收不收**视频** -------------------------------------------
    #
    # **先问这一声，再去打包**（2026-11-13 实测踩到）：整局看片发的是
    # ``video`` / ``video_url`` 内容块，而 OpenAI 兼容的接口没有这种块。
    # 拿 DeepSeek 发视频，服务端回 422 ``unknown variant video_url``，
    # 于是整局一条事件都没有；而那句真正的报错被「视频重编码过」这条
    # 无关留痕盖住，用户看到的是「这一局没有分析出任何内容：
    # 视频重编码过（h264 → mp4v）……」—— 完全指不到该改哪里。
    #
    # 所以：**认得出来这家不收视频时**，改走**抽帧**那条路
    # （每 5 分钟一批看图，见 :mod:`app.recognition2.frames_watch`）——
    # 能看图就不该退化成「没分析」，更不该白打一遍包。
    # 认不出来（比如用户把 MiniMax 的兼容地址填进「自定义」）就照老路试一次。
    takes_video: Optional[bool] = None
    who = ""
    try:
        from app.collection import model_presets

        with settings.use_data_dir(cfg.data_dir or None):
            _slot_cfg = settings.effective_slot(cfg.slot)
        _provider = str(_slot_cfg.get("provider") or "")
        _endpoint = str(_slot_cfg.get("endpoint") or creds.endpoint or "")
        _via = model_presets.provider_of_endpoint(_endpoint)
        takes_video = model_presets.video_support(_provider, _endpoint)
        # 本机编不出 H.264 时，视频块会被模型拒收 —— 改走抽帧。
        if takes_video is not False and not videopack.h264_available():
            out.notes.append(
                "本机没有 H.264 编码器，视频块会被模型拒收 —— 改走抽帧。")
            takes_video = False
        who = _provider if _provider and _provider != "custom" else (
            _via or _provider or "这个供应方")
    except Exception:                                  # noqa: BLE001
        # 能力判断失败不该挡住主流程 —— 大不了照老路发一次给它。
        takes_video = None

    if takes_video is False:
        from . import frames_watch

        out.notes.append(
            "整局看片：%s 不收视频输入（OpenAI 兼容的接口没有 video 内容块）"
            "—— 改用抽帧：按画面变化抽帧、每 %.0f 分钟一批，看清楚整局。"
            % (who, frames_watch.WINDOW_SEC / 60.0))
        sub = frames_watch.watch_by_frames(
            video_path, cfg=cfg, model=model, creds=creds,
            transform=transform, hero=cfg.hero, position=cfg.position)
        sub.notes = list(out.notes) + list(sub.notes or [])
        return sub

    # -- 打包 --
    if pack is None:
        if not cfg.video_dir:
            out.notes.append("没给打包目录 —— 整局没有分析。")
            return out
        pack = videopack.pack_video(video_path, cfg.video_dir,
                                    max_bytes=cfg.max_upload_bytes,
                                    fps=cfg.pack_fps, transform=transform,
                                    stem="match")
    out.pack = pack.to_dict()
    out.notes.extend(pack.notes or [])
    if not pack.path or pack.size <= 0:
        out.notes.append("视频没打成能传的形态 —— 整局没有分析。")
        return out

    # -- 视频怎么送过去 --
    #
    # 两条路：**内联**（≤45 MB）或 **Files 上传**（≤512 MB）。
    # 没有 Files 地址的供应方只能内联 —— 这时候打包必须**压到能内联**，
    # 否则整局直接没分析（实测踩过：打包走了「原样复制」，
    # 84 MB 的片子原封不动送不出去，整局一句话都没讲）。
    can_upload = bool(cfg.allow_upload and creds.files_base)
    inline_limit = vlm_client.MAX_VIDEO_INLINE_RAW_BYTES
    if pack.size > inline_limit and not can_upload:
        out.notes.append(
            "视频 %d MB 超过直传上限，而这个供应方没有 Files 上传地址 —— "
            "重新压到能内联的体积。" % (pack.size // 1024 // 1024))
        smaller = videopack.pack_video(
            video_path, cfg.video_dir, max_bytes=inline_limit,
            fps=cfg.pack_fps, transform=transform, stem="match_inline")
        if smaller.path and smaller.size <= inline_limit:
            pack = smaller
            out.pack = pack.to_dict()
            out.notes.extend(pack.notes or [])
        else:
            out.notes.append(
                "压到最小还是 %d MB，超过直传上限 %d MB —— 整局没有分析。"
                % ((smaller.size or pack.size) // 1024 // 1024,
                   inline_limit // 1024 // 1024))
            return out

    blocks: List[Dict[str, Any]] = []
    if pack.size <= inline_limit:
        try:
            with open(pack.path, "rb") as fh:
                data = fh.read()
            blocks.append(vlm_client.video_block(data, fps=cfg.fps,
                                                 detail=cfg.detail))
        except OSError as exc:
            out.notes.append("读打包好的视频失败：%s" % exc)
            return out
    else:
        if not can_upload:
            out.notes.append(
                "视频 %d MB 超过直传上限，又没地方上传 —— 整局没有分析。"
                % (pack.size // 1024 // 1024))
            return out
        try:
            file_id = vlm_client.upload_file(pack.path, creds.api_key,
                                             creds.files_base)
        except vlm_client.VlmError as exc:
            out.notes.append("上传视频失败：%s" % vlm_client.explain_error(exc))
            return out
        out.uploaded = True
        out.notes.append("整局录像已上传到对方的文件服务（按对方的政策最多留 7 天）。")
        blocks.append(vlm_client.video_ref_block(file_id, fps=cfg.fps,
                                                  detail=cfg.detail))

    # -- 问模型 --
    blocks.insert(0, vlm_client.text_block(
        build_prompt(pack.duration, cfg.hero, cfg.position)))
    try:
        result = vlm_client.call_model(
            model, blocks, creds=creds, system=SYSTEM_PROMPT,
            max_tokens=cfg.max_tokens, timeout=cfg.timeout)
    except vlm_client.VlmError as exc:
        out.notes.append("整局没看成：%s" % vlm_client.explain_error(exc))
        return out
    out.calls = 1
    out.tokens = _token_count(result.usage or {})

    try:
        raw = result.json()
    except vlm_client.VlmError as exc:
        # **回答被截断**（超上限）—— 前面写完的那几段是**完整的**，
        # 整份丢掉太浪费（2026-11-02 18:24 实测：20 422 字符处断掉，
        # 于是整局一条都没读出来，报告只能是空的）。
        salvaged = salvage_events(result.text or "")
        if salvaged:
            out.notes.append(
                "模型回答超长被截断（%s）—— 只读到了前面写完的 %d 段，"
                "后面的没读到。" % (exc, len(salvaged)))
            raw = {"events": salvaged}
            # **前面那几个字段模型已经写了** —— 一并捞回来。
            # 英雄、分路、对局起点、它自己想查的那些时刻，都在这段里；
            # 丢掉它们等于「模型答了却当它没答」（2026-11-03 实测踩到：
            # 那一局模型明明说了「敖隐」，英雄层却一条都不参与）。
            head = salvage_head(result.text or "")
            if head:
                raw.update(head)
                got = "、".join(sorted(head.keys()))
                out.notes.append(
                    "截断前模型已经写好的那几个字段已取回：%s。" % got)
        else:
            out.notes.append("整局没看成：模型回答不是合法 JSON（%s）" % exc)
            return out

    events, drops = parse_events(raw)
    out.notes.extend(drops)

    scale = float(pack.time_scale or 1.0)
    offset = float(getattr(pack, "source_start", 0.0) or 0.0)
    for item in events:
        t = item.get("t")
        t_end = item.get("t_end")
        event = WatchEvent(
            video_start=(float(t) * scale + offset) if t is not None else 0.0,
            video_end=(float(t_end) * scale + offset) if t_end is not None else 0.0,
            narrative=clean_words(item["narrative"]),
            action=item["action"],
            action_label=item["action_label"],
            position=item["position"],
            lane=item.get("lane") or "unknown",
            where=clean_words(item["where"]),
            phase=item["phase"],
            confidence=item["confidence"],
            clock=item["clock"],
            stage=str(item.get("stage") or ""),
            event_type=str(item.get("event_type") or ""),
            frames=[float(x) for x in (item.get("frames") or [])],
        )
        if event.video_end < event.video_start:
            event.video_end = event.video_start
        out.events.append(event)

    out.summary = str((raw.get("summary") or "") if isinstance(raw, dict)
                       else "").strip()
    if not out.summary and isinstance(raw, dict):
        out.summary = _stages_summary(raw)
    out.hero = str((raw.get("hero") or "") if isinstance(raw, dict) else "").strip()
    if not out.hero and cfg.hero:
        out.hero = str(cfg.hero).strip()
        out.notes.append("英雄用你上传前填的「%s」（模型这次没读出来）。"
                         % out.hero)
    # **顶层分路要归一** —— 模型常写「中路」「上路」。
    #
    # 不归一的后果很安静：分路成了「中路」而不是 ``mid``，于是每一条
    # 带分路条件的规则都被判成「不是给他写的」，整局一条结论都不出，
    # 而报告上**看不出哪里出了岔**（归一在这里做一次就够，
    # 后面每一层拿到的都是规范码）。
    raw_position = str((raw.get("position") or "")
                       if isinstance(raw, dict) else "").strip()
    out.position = normalize_position(raw_position)
    if raw_position and raw_position != "unknown" and out.position == "unknown":
        out.notes.append("模型说的分路「%s」认不出是哪条路 —— 分路这次没定出来。"
                         % raw_position)
    if (not out.position or out.position == "unknown") and cfg.position:
        out.position = str(cfg.position).strip()
        out.notes.append("分路用你上传前填的「%s」。" % out.position)
    out.result = _norm_result(raw.get("result") if isinstance(raw, dict) else None)
    out.deaths = _parse_deaths(raw, scale=scale, offset=offset)
    if out.hero:
        # 称呼统一（玩家 -> 英雄名，敌方 -> 敌方英雄）。
        for event in out.events:
            event.narrative = strip_skills(strip_lane_words(strip_time_refs(
                clean_place(clean_words(event.narrative, out.hero)))))
            event.where = canonical_place(clean_words(event.where, out.hero))
        for death in out.deaths:
            death.cause = clean_place(clean_words(death.cause, out.hero))
            death.fault = clean_place(clean_words(death.fault, out.hero))
            death.where = canonical_place(clean_words(death.where, out.hero))

    # -- 模型自己写的检索请求（「我想查什么」） --
    #
    # 时刻用**同一套**换算（scale / offset）—— 与事件时刻一致，
    # 否则「第几秒」会差一个比例，检索到的窗口就落到别的分钟上。
    if isinstance(raw, dict):
        out.requests = parse_requests(raw.get("queries"), scale=scale,
                                      offset=offset)
    if not out.requests:
        out.notes.append("模型没写要对照攻略的地方 —— 改用它讲的事件叙述当查询。")
    else:
        out.notes.append("模型自己提了 %d 个要对照攻略的地方。" % len(out.requests))

    guessed = None
    if isinstance(raw, dict):
        guessed = _first_float(raw, ("match_start", "game_start", "start"))
    if guessed is not None:
        # **换算成原视频秒** —— 模型看的是**打包片**，报的是打包片里的秒数；
        # 事件时刻（``video_start``）走的是同一套 ``scale / offset``。
        # 不换算，起点就会整体偏 ``start × (1/scale − 1)``。
        guessed = float(guessed) * scale + offset
    start, how = _resolve_match_start(out.events, guessed)
    out.match_start = start
    # 段由代码按局内时间算 —— 模型只负责给 clock（见 apply_stages）。
    apply_stages(out.events, out.match_start)
    if start is None:
        out.notes.append(
            "没定出这一局从哪一秒开始 —— 带时间窗的规则一条都判不了"
            "（模型没看出对局起点，也没读出顶栏时刻）。")
    else:
        out.notes.append("对局起点：原视频第 %.1f 秒（%s）。"
                         % (start, how or "模型口报"))

    if not out.events:
        out.notes.append("模型看完这局，一个事件都没讲出来。")
        return out

    # -- K/D/A 校验：模型说的「击杀」不算，数字 +1 才算 --------------------
    if cfg.verify_events and out.events:
        from . import event_verify

        verified = event_verify.verify_events(
            out.events, video_path, model=model, data_dir=cfg.data_dir,
            transform=transform)
        out.calls += int(verified.get("calls") or 0)
        out.tokens += int(verified.get("tokens") or 0)
        out.notes.extend(verified.get("notes") or [])
        apply_stages(out.events, out.match_start)
        out.events.sort(key=lambda e: e.video_start)

    # -- 位置不出现在报告里：清空 where（小地图只作内部校验，不外显）-------
    for event in out.events:
        event.where = ""
    for death in out.deaths:
        death.where = ""

    # -- 关键时刻再细看一遍：±30 秒、一秒一帧 ----------------------------
    if cfg.detail_events:
        from . import event_detail

        detail = event_detail.detail_events(
            out.events, video_path, model=model, creds=creds,
            transform=transform, hero=out.hero or cfg.hero,
            span=cfg.detail_span)
        out.calls += int(detail.get("calls") or 0)
        out.tokens += int(detail.get("tokens") or 0)
        out.notes.extend(detail.get("notes") or [])
        if detail.get("calls"):
            out.notes.append("关键时刻按 1 秒 1 帧细看了 %d 段（每段前后各 %.0f 秒）。"
                             % (detail["calls"], cfg.detail_span))
        if detail.get("fixed"):
            out.notes.append("细看时按画面提示帧校正了 %d 个事件时刻。"
                             % detail["fixed"])
        # 时刻校正后，段与顺序都要重算。
        apply_stages(out.events, out.match_start)
        out.events.sort(key=lambda e: e.video_start)
        # 细看的文字也要过一遍称呼/地点归一。
        if out.hero:
            for event in out.events:
                event.narrative = clean_place(clean_words(event.narrative, out.hero))

        # 丢塔本版不输出（用户口径）；拆塔要留下，交给细看按证据卡。
        lost = [e for e in out.events
                if str(getattr(e, "event_type", "")) == "tower_lost"]
        if lost:
            out.events = [e for e in out.events if e not in lost]
            out.notes.append("本版不输出丢塔：去掉 %d 条。" % len(lost))

    return out
