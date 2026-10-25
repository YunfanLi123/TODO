"""对局上下文：游戏无关的数据契约。

立项设计要求「新增一款游戏只需要写规则层和分析映射，不动内核」。
这条要求的实现方式就是本文件：内核只认 :class:`MatchContext` 和
:class:`RoundRecord`，完全不认识云顶之弈的金币/连胜，也不认识王者荣耀的
经济/等级。游戏适配器负责把各自的原始数据翻译成这里的事实字典。

一条事实在内核里就是一个 ``str -> 数值/字符串`` 的映射，
由适配器注册并负责计算（可能来自视频识别，也可能来自用户手工录入）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

#: 事实解析器签名：(原始对局数据, 回合记录) -> (取值, 是否采集到)
FactResolver = Callable[[Dict[str, Any], "RoundRecord"], Tuple[Any, bool]]


@dataclass
class ActionRecord:
    """回合内玩家实际做的一件事。

    规则判定的是「在这个回合，玩家做的是不是期望的那类动作」。
    """

    kind: str                    # 事实名，如 econ_action / positioning_action
    value: Any                   # 取值，如 "save" / "rush" / "split_push"
    label: str = ""              # 人类可读名称，如 "存钱不追星"
    ts: float = 0.0              # 回合内相对时间
    confidence: float = 1.0      # 识别置信度；低置信度会在报告中显式标记
    #: 这个动作**从哪来**：``player``（玩家自己勾的）/ ``probe``（画面里看出来的）。
    #:
    #: 两种来源的可靠性完全不同 —— 报告必须标出来，否则用户打开报告会以为
    #: 「视频里做了什么」是玩家自己说的，那就是替玩家编话。
    source: str = "player"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "label": self.label,
            "ts": self.ts,
            "confidence": round(self.confidence, 3),
            "source": self.source,
        }


@dataclass
class RoundRecord:
    """一个回合的结构化快照。

    ``facts`` 保存该回合观测到的全部事实取值。
    ``actions`` 保存玩家在该回合的实际操作序列 —— 判定时只关心
    是否存在一条满足期望的动作。
    """

    stage: int                                    # 阶段，如 4（对应 4-1）
    phase: int = 1                                # 小阶段，如 1
    facts: Dict[str, Any] = field(default_factory=dict)
    actions: List[ActionRecord] = field(default_factory=list)
    #: 哪些事实在这一回合没有被采集到（视频识别失败 / 用户未录入）
    missing_facts: set = field(default_factory=set)
    #: 覆盖默认的 ``stage-phase`` 标签（见 :attr:`label`）
    label_override: str = ""
    #: 时间轴上的连续位置。仅在同一 stage/phase 内部起排序作用 ——
    #: MOBA 的多个时刻会落进同一个粗粒度时间片，若不额外带上精确时间，
    #: 报告里的差距就会按严重度而非先后顺序排列，时间线被打乱。
    time_index: float = 0.0
    #: 这一回合在**原视频**里的位置（秒）。0 表示不是视频来源。
    #: 报告的「看这一段」靠它把播放器定位到出错的那一段 ——
    #: 手工录入的对局没有视频，所以是 0，界面上也就不显示那个入口。
    video_time: float = 0.0
    #: 探测器找到的**证据时刻**：``[{"t": 视频秒, "nx": .., "ny": ..,
    #: "why": "看到了什么"}]``。
    #:
    #: 报告里「这段在 12:34 出现开龙迹象」靠它 —— 光给结论，
    #: 玩家没法回看原画面确认。
    probe_moments: List[Dict[str, Any]] = field(default_factory=list)
    #: 明确**没做**的动作类（信号可读、画面里没找到）。
    probe_absent: List[str] = field(default_factory=list)
    #: 模型对**操作模式清单**的裁决（逐条偏离 / 整段没做）。
    #:
    #: 报告正文的主要来源：模型看着画面，对照攻略原文判断哪一条的前提成立、
    #: 玩家做的不是那一件事。每一条都带它据以判断的那几帧。
    review: List[Dict[str, Any]] = field(default_factory=list)
    #: 该回合的抽帧文件名，报告里配在「看这一段」旁边当缩略图
    image: str = ""

    @property
    def label(self) -> str:
        """回合标签，如 ``4-1``。报告里所有定位都指向这个标签。

        适配器可通过 ``label_override`` 给出更贴合该游戏习惯的写法：
        自走棋用阶段坐标 ``4-1``，而 MOBA 没有阶段概念，
        王者荣耀适配器给出「第 12 分钟」—— 技术上定位到同一回合，
        但玩家必须一眼看懂，否则「定位到具体回合」这个卖点是空的。
        """
        return self.label_override or f"{self.stage}-{self.phase}"

    def resolve(self, fact: str) -> Tuple[Any, bool]:
        """供 :class:`~app.core.dsl.Condition` 求值使用。"""
        if fact in self.missing_facts:
            return (None, False)
        if fact in self.facts:
            return (self.facts[fact], True)
        return (None, False)

    def action_values(self, kind: str) -> List[Any]:
        return [a.value for a in self.actions if a.kind == kind]

    def last_action(self, kind: str) -> Optional[ActionRecord]:
        """该回合内最后一类动作。

        这是**触发条件**那一侧的口径：适配器把动作镜像进 ``facts`` 时
        保留的就是这一个值，所以 trigger 里的动作事实条件看的是它。

        期望动作那一侧不是这个口径 —— 那里是「这一回合做过符合要求的
        动作就算符合」，见 :func:`app.core.matcher.evaluate_round`。
        """
        candidates = [a for a in self.actions if a.kind == kind]
        return candidates[-1] if candidates else None

    def lowest_confidence(self, kind: Optional[str] = None) -> float:
        """本回合相关动作的最低置信度，用于报告中的不确定性标记。"""
        pool = [a for a in self.actions if kind is None or a.kind == kind]
        return min((a.confidence for a in pool), default=1.0)

    def to_dict(self) -> Dict[str, Any]:
        # label_override / time_index 必须一并序列化：版本回测要靠
        # payload 重建对局上下文，漏掉它们会让重建结果丢失可读标签
        # 与时间轴精度，且这种缺失在正常分析路径上完全看不出来。
        return {
            "label": self.label,
            "label_override": self.label_override,
            "time_index": self.time_index,
            "video_time": self.video_time,
            "probe_moments": list(self.probe_moments),
            "probe_absent": list(self.probe_absent),
            "review": list(self.review),
            "image": self.image,
            "stage": self.stage,
            "phase": self.phase,
            "facts": dict(self.facts),
            "actions": [a.to_dict() for a in self.actions],
            "missing_facts": sorted(self.missing_facts),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "RoundRecord":
        """从 :meth:`to_dict` 的产物重建。

        兼容早期写入的 payload：那两个字段可能不存在，此时退回默认值，
        而不是让整局重建失败 —— 历史数据不该因为新增字段而读不出来。
        """
        return RoundRecord(
            stage=int(data.get("stage") or 0),
            phase=int(data.get("phase") or 0),
            facts=dict(data.get("facts") or {}),
            actions=[
                ActionRecord(
                    kind=a.get("kind", ""),
                    value=a.get("value"),
                    label=a.get("label", ""),
                    ts=float(a.get("ts") or 0.0),
                    confidence=float(a.get("confidence", 1.0)),
                    source=str(a.get("source") or "player"),
                )
                for a in (data.get("actions") or [])
            ],
            missing_facts=set(data.get("missing_facts") or []),
            label_override=data.get("label_override", "") or "",
            time_index=float(data.get("time_index") or 0.0),
            video_time=float(data.get("video_time") or 0.0),
            probe_moments=list(data.get("probe_moments") or []),
            probe_absent=list(data.get("probe_absent") or []),
            review=list(data.get("review") or []),
            image=data.get("image", "") or "",
        )


@dataclass
class MatchContext:
    """一局对局的完整上下文。

    ``raw`` 保存原始对局数据（视频识别结果或用户手工录入的原样）。
    事实不在这里预先算好，而是由适配器通过 fact_resolver 按需计算 ——
    这样同一份原始数据可以喂给不同版本、不同口径的规则库。
    """

    match_id: str
    game: str
    game_version: str
    rounds: List[RoundRecord] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)
    player: Dict[str, Any] = field(default_factory=dict)
    #: 事实名 -> 解析器。由 GameAdapter 注入。
    fact_resolvers: Dict[str, FactResolver] = field(default_factory=dict)
    #: 整局级结果（名次 / 胜负）
    outcome: Dict[str, Any] = field(default_factory=dict)
    #: 这一局的动作标注从哪来。
    #:
    #: ``player``  = 玩家自己在标注页上勾的
    #: ``derived`` = 从画面反推出来的（只有验证链路会这么用）
    #:
    #: 两者可信度不同：后者是「看着帧猜的」。报告必须把它标出来 ——
    #: 否则用户看到「视频中做了什么」会以为那是玩家自己说的。
    annotation_source: str = "player"

    def round_by_label(self, label: str) -> Optional[RoundRecord]:
        for r in self.rounds:
            if r.label == label:
                return r
        return None

    def rounds_in_stage(self, stage: int) -> List[RoundRecord]:
        return [r for r in self.rounds if r.stage == stage]

    def first_round(self) -> Optional[RoundRecord]:
        return self.rounds[0] if self.rounds else None

    def last_round(self) -> Optional[RoundRecord]:
        return self.rounds[-1] if self.rounds else None

    def iter_rounds(self) -> Iterable[RoundRecord]:
        return iter(sorted(self.rounds, key=lambda r: (r.stage, r.phase)))

    def resolve_in_round(self, rnd: RoundRecord) -> Callable[[str], Tuple[Any, bool]]:
        """返回一个解析函数：先查预计算事实，再走适配器注册的解析器。"""

        def _resolve(fact: str) -> Tuple[Any, bool]:
            if fact in rnd.facts:
                return (rnd.facts[fact], fact not in rnd.missing_facts)
            if fact in rnd.missing_facts:
                return (None, False)
            resolver = self.fact_resolvers.get(fact)
            if resolver is None:
                return (None, False)
            try:
                value, present = resolver(self.raw, rnd)
            except Exception:
                # 解析器异常等价于「该事实未采集到」，绝不打断整局分析
                return (None, False)
            if present:
                rnd.facts[fact] = value
            else:
                rnd.missing_facts.add(fact)
            return (value, present)

        return _resolve

    def precompute(self) -> None:
        """按适配器注册的事实全量预计算，写入各回合 facts 字典。"""
        for rnd in self.iter_rounds():
            resolver = self.resolve_in_round(rnd)
            for fact in self.fact_resolvers:
                resolver(fact)

    def to_dict(self) -> Dict[str, Any]:
        """序列化对局数据。

        ``recognition`` 单独带出而不是把 ``raw`` 整体塞进去：
        识别诊断（置信度、识别器版本、被丢弃的事实及原因）是结论可信度的
        一部分，必须随对局一起保存 —— 否则历史对局只剩数值而没了「这个数
        是怎么来的」，报告就再也无法解释自己的不确定性。
        而 ``raw`` 里可能塞着原始视频帧等大体积内容，不该进库。
        """
        out = {
            "match_id": self.match_id,
            "game": self.game,
            "game_version": self.game_version,
            "player": self.player,
            "outcome": self.outcome,
            "annotation_source": self.annotation_source,
            "rounds": [r.to_dict() for r in self.iter_rounds()],
        }
        recognition = (self.raw or {}).get("recognition")
        if recognition:
            out["recognition"] = recognition
        return out

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "MatchContext":
        """从 :meth:`to_dict` 的产物重建上下文。

        重建出的上下文**不带** fact_resolvers —— 那是适配器的运行时能力，
        不该被序列化进数据层。所有事实都已经在 ``facts`` 里定值，
        规则匹配只需要取值，不需要再算。这一点是版本回测成立的前提：
        用新规则重判历史对局时，不能依赖任何适配器行为。
        """
        ctx = MatchContext(
            match_id=data.get("match_id", ""),
            game=data.get("game", ""),
            game_version=data.get("game_version", ""),
            player=dict(data.get("player") or {}),
            outcome=dict(data.get("outcome") or {}),
            annotation_source=data.get("annotation_source") or "player",
            rounds=[RoundRecord.from_dict(r) for r in (data.get("rounds") or [])],
        )
        recognition = data.get("recognition")
        if recognition:
            ctx.raw = {"recognition": recognition}
        return ctx
