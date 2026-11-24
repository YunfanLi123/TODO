"""**时间线上的一段行为** —— 新识别子系统的唯一真源。

为什么要有这个结构
------------------
旧系统有**四套**并行的数据描述同一件事：

============  ==========================================
``facts``    一个事实**一个值**（``position=野区``）
``actions``  一个回合**多个动作**（刷野 + 转线 + 开团）
``probe_moments``  带时间点与画面描述的观测证据
``review``   模型对攻略清单的逐条裁决
============  ==========================================

它们互相不打架地对不上：判定层读 ``actions``，触发条件里的动作事实
读 ``facts`` 里**镜像进去的最后一个值**，报告覆盖表又读一遍 ``actions``。
「玩家做了什么」这件事**没有一个统一来源** —— 这就是重写的起点。

现在只有一个真源：:class:`Beat`。

一份 Beat 说清四件事
--------------------
* **什么时候**（局内时刻 + 视频秒数）
* **做了什么**（动作码 + 中文名）
* **看到了什么**（一句话叙述 —— 给人看的直接答案）
* **凭什么这么说**（证据帧 + 置信度）

事实（位置、是否在团战…）由 Beat **派生**，不再是独立真源。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

#: 置信度：模型明确说「看不清」时的取值。
#:
#: 为什么要有这个特殊值
#: ------------------
#: 「看不清」与「没做」在报告里的意义**完全相反**：
#: 前者是「我们没看见」，后者是「玩家没做」。
#: 混为一谈就是冤枉人 —— 所以 ``unclear`` 的段照样进时间线，
#: 但判定时算「无法判定」，**永远不算「没做」**。
UNCLEAR = "unclear"

#: 这一段在整局的哪个阶段。
#:
#: 为什么要有它
#: ----------
#: 录屏里**对局之外**的片段（选人、加载、结算）本来就没有「玩家在干什么」
#: 可说。把它们混进覆盖率，会让「这一局读懂了 91%」这种数字失真 ——
#: 那 9% 里有一大半根本不是对局。
PHASE_PRE_GAME = "pre_game"        # 开局前：选人 / 禁用 / 加载
PHASE_IN_GAME = "in_game"          # 对局中
PHASE_POST_GAME = "post_game"      # 结算
PHASE_UNKNOWN = "unknown"


@dataclass
class Beat:
    """时间线上的一段行为。"""

    # -- 时间 -------------------------------------------------------------
    #: 局内秒数（**锚点帧那一刻**的时刻）。读不到就是 -1。
    game_time: float = -1.0
    #: 原视频里的秒数。用来回看原画面 —— 指的就是锚点帧。
    video_time: float = 0.0
    #: 局内时刻区间（秒）：这一段**覆盖**的局内时间范围。
    #: 判定层拿它跟规则的时间窗求交，所以它是「一段」而不是「一点」。
    game_time_start: float = -1.0
    game_time_end: float = -1.0
    #: 原视频里的秒数区间。
    video_time_start: float = 0.0
    video_time_end: float = 0.0
    #: 锚点帧在 :attr:`frames` 里的下标（0 起）。
    #:
    #: 为什么要有锚点
    #: --------------
    #: ``game_time`` 必须是**某一帧自己的时刻**，不能是段内各帧的平均 ——
    #: 平均值离最近的那张帧有 L/2(n-1) 秒（17.8 秒的段 ≈ 3 秒），
    #: 拿它去比那张帧画面上的钟，必然差 3 秒。
    #: 锚点帧一旦定死，「这一段发生在 03:12」和「03:12 那张画面上的钟」
    #: 就是同一个瞬间，可以逐帧核对。
    anchor_frame: int = 0

    # -- 内容 -------------------------------------------------------------
    #: 动作码（``farm_jungle`` / ``engage`` / …）。看不清就是 ``unclear``。
    action: str = UNCLEAR
    #: 动作的中文名（刷野补经济 / 开团先手 / …）。
    action_label: str = ""
    #: 模型用一句话说清「看到了什么」—— **这是给人看的直接答案**。
    narrative: str = ""

    # -- 证据 -------------------------------------------------------------
    #: 证据帧文件名（相对抽帧目录）。每一条都要能回看到原画面。
    frames: List[str] = field(default_factory=list)
    #: 置信度 0~1。
    confidence: float = 0.0

    # -- 补充 -------------------------------------------------------------
    #: 补充信息（位置、目标、持续时长…）。**不参与判定**，只给人看。
    detail: Dict[str, Any] = field(default_factory=dict)

    #: 为什么没读出来（``unclear`` 时才有值）。
    #: 例如「模型说看不清」「没调用模型」「超出预算」「额度用尽」。
    why_not_read: str = ""

    @property
    def is_unclear(self) -> bool:
        """这一段是不是**没读出来**。

        **和 ``action == UNCLEAR`` 是两件事，别混**
        --------------------------------------
        - ``action == UNCLEAR``：没归类出对局动作 ——
          判定层按「无法判定」处理（**永远不算「没做」**）；
        - ``is_unclear``：**连一句话都没说出来** —— 覆盖率按这个算。

        模型说「玩家打开战绩面板查看双方数据」而「查看战绩」不在
        25 个动作词表里 —— 这是**读出来了**，只是归类不到。
        归类不到就说归类不到（「动作码未归类」记在 ``detail`` 里），
        不该算成「没读出来」，否则覆盖率会**虚低**。
        """
        if self.action != UNCLEAR:
            return self.confidence <= 0.0
        return not (self.narrative or "").strip()

    @property
    def label(self) -> str:
        """报告里显示的动作名。"""
        return self.action_label or self.action

    @property
    def game_span(self) -> float:
        """这一段覆盖的局内时长（秒）。没有区间就是 0。"""
        if self.game_time_start < 0 or self.game_time_end < 0:
            return 0.0
        return max(0.0, float(self.game_time_end) - float(self.game_time_start))

    @property
    def video_span(self) -> float:
        """这一段在原视频里跨了多久（秒）。"""
        if self.video_time_end <= 0 or self.video_time_start <= 0:
            return 0.0
        return max(0.0, float(self.video_time_end) - float(self.video_time_start))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_time": round(float(self.game_time), 2),
            "game_time_start": round(float(self.game_time_start), 2),
            "game_time_end": round(float(self.game_time_end), 2),
            "video_time": round(float(self.video_time), 2),
            "video_time_start": round(float(self.video_time_start), 2),
            "video_time_end": round(float(self.video_time_end), 2),
            "anchor_frame": int(self.anchor_frame),
            "action": self.action,
            "action_label": self.action_label,
            "narrative": self.narrative,
            "frames": list(self.frames),
            "confidence": round(float(self.confidence), 3),
            "detail": dict(self.detail),
            "why_not_read": self.why_not_read,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Beat":
        return Beat(
            game_time=float(data.get("game_time") or -1.0),
            game_time_start=float(data.get("game_time_start")
                                 if data.get("game_time_start") is not None else -1.0),
            game_time_end=float(data.get("game_time_end")
                               if data.get("game_time_end") is not None else -1.0),
            video_time=float(data.get("video_time") or 0.0),
            video_time_start=float(data.get("video_time_start") or 0.0),
            video_time_end=float(data.get("video_time_end") or 0.0),
            anchor_frame=int(data.get("anchor_frame") or 0),
            action=str(data.get("action") or UNCLEAR),
            action_label=str(data.get("action_label") or ""),
            narrative=str(data.get("narrative") or ""),
            frames=[str(x) for x in (data.get("frames") or [])],
            confidence=float(data.get("confidence") or 0.0),
            detail=dict(data.get("detail") or {}),
            why_not_read=str(data.get("why_not_read") or ""),
        )


@dataclass
class Timeline:
    """一局的行为时间线。"""

    game: str = "hok"
    beats: List[Beat] = field(default_factory=list)
    #: 整局的元信息（局内时长、模型、抽帧目录…）。
    meta: Dict[str, Any] = field(default_factory=dict)
    #: 整局级别的留痕（哪些段没读出来、为什么）。
    notes: List[str] = field(default_factory=list)

    def add(self, beat: Beat) -> None:
        self.beats.append(beat)

    def sorted_beats(self) -> List[Beat]:
        """按局内时间排序。读不到时间的排在最后（但保持原有相对顺序）。"""
        return sorted(
            self.beats,
            key=lambda b: (b.game_time < 0, b.game_time, b.video_time),
        )

    def sorted_by_video(self) -> List[Beat]:
        """按**视频时间**排序 —— 「每一段时间区间」的天然顺序。

        报告给人看用这个；判定层用 :meth:`sorted_beats`（按局内时刻，
        规则的时间窗靠它对齐）。
        """
        return sorted(self.beats, key=lambda b: b.video_time)

    def coverage(self) -> Dict[str, Any]:
        """覆盖率统计 —— 验收标准之一。

        **两个率分开报**，不能混成一个：

        - ``read_rate``：有多少段**读出了**「玩家做了什么」（有叙述）——
          这才是「每一段时间区间都知道玩家做了什么」那个指标；
        - ``classified_rate`：有多少段的**动作码能归类**到动作词表 ——
          归类不了的段，判定层按「动作不明」处理。

        **对局内、对局外分开算**：选人/结算那几段本来就没有「玩家在干什么」
        可说，混在一起会让这个数字失真。
        """
        total = len(self.beats)
        read = sum(1 for b in self.beats if not b.is_unclear)
        classified = sum(1 for b in self.beats
                         if b.action != UNCLEAR and b.confidence > 0)
        in_game = [b for b in self.beats
                   if (b.detail or {}).get("phase") == PHASE_IN_GAME]
        in_game_read = sum(1 for b in in_game if not b.is_unclear)
        in_game_classified = sum(1 for b in in_game
                                 if b.action != UNCLEAR and b.confidence > 0)
        return {
            "segments": total,
            "read": read,
            "unclear": total - read,
            "coverage": round(read / total, 3) if total else 0.0,
            "classified": classified,
            "classified_rate": (round(classified / total, 3) if total else 0.0),
            "in_game_segments": len(in_game),
            "in_game_read": in_game_read,
            "in_game_coverage": (round(in_game_read / len(in_game), 3)
                                 if in_game else 0.0),
            "in_game_classified": in_game_classified,
            "in_game_classified_rate": (round(in_game_classified / len(in_game), 3)
                                        if in_game else 0.0),
            "out_of_game_segments": total - len(in_game),
            "with_frames": sum(1 for b in self.beats if b.frames),
            "with_time": sum(1 for b in self.beats if b.game_time >= 0),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game": self.game,
            "beats": [b.to_dict() for b in self.sorted_beats()],
            "meta": dict(self.meta),
            "notes": list(self.notes),
            "coverage": self.coverage(),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Timeline":
        tl = Timeline(
            game=str(data.get("game") or "hok"),
            meta=dict(data.get("meta") or {}),
            notes=[str(x) for x in (data.get("notes") or [])],
        )
        tl.beats = [Beat.from_dict(b) for b in (data.get("beats") or [])]
        return tl

    def save(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=1)
        return path
