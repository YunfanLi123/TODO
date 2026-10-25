"""视频解析的数据契约（扩展阶段）。

这个模块**不实现识别算法**，只定义识别结果必须长什么样、如何降级成
内核能消费的对局数据。理由很实际：识别算法会反复更换（OCR 换模型、
ROI 裁剪策略调参、切分从帧差换成事件检测），但只要这个契约不变，
内核一行都不用改。反过来，没有契约就先写识别代码，注定返工。

技术路线在这里落地为接口：

    扫描切时间片（廉价）    识别器 A：detect_round_boundaries()
        ↓ 只在这些时刻
    精细识别 / 人工标注      识别器 B：recognize_rounds()

（接口名沿用早先的命名，含义已按王者的「时间片」重写：王者的一局是
**连续的一整场**，没有「一回合」可切，所以切分单位是时间片。）

置信度三态（这是本模块最重要的部分）
------------------------------
视频识别的输出天然带不确定性，而**「不确定」有两种完全不同的含义**，
压成一种就会骗人：

    高置信度  conf >= 0.75   正常使用
    低置信度  0.30 ~ 0.75   取值保留，但报告必须显式标记「这是识别的」
    未识别    conf < 0.30    取值不可信 → 事实进 missing_facts

为什么低置信度要保留而不是一丢了之：直接丢弃会让识别质量中等的视频
完全无法分析 —— 系统在半夜录的屏、风扇转不停、画面糊一点，就得出
「本局无可用数据」，那是把问题推给用户。而保留并标记，玩家看到的是
「这条结论建立在识别之上，建议你确认一下」，信息是完整的、责任是清晰的。

但为什么未识别必须丢弃：内核的判定是布尔值，一旦把 0.2 置信度的
「血量 58」当成事实用，报告会给出「你剩 58 血所以应该停手」这种斩钉截铁
的结论，而那个 58 基本是猜的。宁可诚实地说「这一刻判定不了」。

原始置信度全部保留在 :attr:`MatchContext.raw` 的诊断里，报告层可展开复查。
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# --------------------------------------------------------------------------
# 置信度阈值
# --------------------------------------------------------------------------

#: 取值可直接采信的门槛。低于此值的识别结果不进 facts。
FACT_ACCEPT_THRESHOLD = 0.30

#: 低于此值但高于上者时，取值仍然保留（在 raw 诊断里），报告须标记。
FACT_FLAG_THRESHOLD = 0.75

#: 动作采信门槛。动作比事实更主观（「玩家做了什么」需要理解意图），
#: 所以要求更高的把握。
ACTION_ACCEPT_THRESHOLD = 0.40

#: 低于此值的动作仍会采信但打低置信度标记。
ACTION_FLAG_THRESHOLD = 0.75

#: 这一段的置信度：低于此值说明「这一刻没切准 / 没看清」，
#: 该区间的识别结果整体不可信，应整段标记为缺失。
GATE_TRUST_THRESHOLD = 0.50


# --------------------------------------------------------------------------
# 数据契约
# --------------------------------------------------------------------------


@dataclass
class DetectedFact:
    """单个事实的识别结果。

    ``method`` 记录它是怎么来的：``gate``（回合门控顺带识别的粗粒度值）、
    ``roi``（ROI 精细识别）、``manual``（用户手工补录）。报告里用它说明
    「这个数字是怎么来的」，玩家才知道该不该信。
    """

    name: str
    value: Any
    confidence: float = 1.0
    method: str = "roi"
    #: 识别失败时的原因，便于排查（画面遮挡 / ROI 越界 / 模型无输出）
    failure: str = ""

    @property
    def accepted(self) -> bool:
        return self.confidence >= FACT_ACCEPT_THRESHOLD

    @property
    def flagged(self) -> bool:
        return FACT_ACCEPT_THRESHOLD <= self.confidence < FACT_FLAG_THRESHOLD

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "confidence": round(self.confidence, 3),
            "method": self.method,
            "failure": self.failure,
        }


@dataclass
class DetectedAction:
    """玩家实际操作的识别结果。"""

    kind: str
    value: Any
    label: str = ""
    confidence: float = 1.0
    ts: float = 0.0
    #: 这个动作**从哪来**：``probe``（画面里看出来的）/ ``player``（玩家勾的）。
    source: str = "probe"

    @property
    def accepted(self) -> bool:
        return self.confidence >= ACTION_ACCEPT_THRESHOLD

    @property
    def flagged(self) -> bool:
        return ACTION_ACCEPT_THRESHOLD <= self.confidence < ACTION_FLAG_THRESHOLD

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "ts": self.ts,
            "source": self.source,
        }


@dataclass
class GateFrame:
    """扫描阶段的一帧输出。

    扫描是廉价的：只看画面差异 / 时钟区域，不做精细识别。
    它的唯一任务是回答「哪一刻值得细看」。
    """

    index: int
    timestamp: float                       # 视频内秒数
    is_boundary: bool = False             # 是否为一段的起点
    confidence: float = 1.0               # 边界检测置信度
    frame_ref: str = ""                   # 抽取帧的存放位置（可选）

    @property
    def trustworthy(self) -> bool:
        return self.confidence >= GATE_TRUST_THRESHOLD

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": round(self.timestamp, 3),
            "is_boundary": self.is_boundary,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class RecognizedRound:
    """精细识别 / 人工标注的输出，对应一段完整的时间片。"""

    stage: int
    phase: int = 1
    label_override: str = ""
    time_index: float = 0.0
    #: 起止帧号，指向扫描阶段的结果，用于报告里回看原画面
    frame_start: int = 0
    frame_end: int = 0
    #: 该回合的抽帧文件名（相对 videos\<video_id>_frames\）。
    #: 报告里「看这一段」旁边的截图就是它。
    image: str = ""
    #: 该回合在**原视频**里的秒数。0 表示不是视频来源。
    #:
    #: 刻意与 :attr:`time_index` 分开：``time_index`` 是**游戏内**的时间轴位置
    #: （分钟），``video_time`` 是**视频文件**里的位置（秒）。早先两者共用一个
    #: 字段，结果报告的「看这一段」把游戏内第 3 分钟当成视频第 3 秒去定位 ——
    #: 点开永远跳到录屏的开头，而那种错误在报告页面上完全看不出来。
    video_time: float = 0.0
    facts: List[DetectedFact] = field(default_factory=list)
    actions: List[DetectedAction] = field(default_factory=list)
    #: **探测器**找到的证据时刻：``[{"t": 视频秒, "nx": .., "ny": ..,
    #: "why": "看到了什么", "kind": .., "value": ..}]``。
    #:
    #: 报告里「这段在 12:34 出现开龙迹象」靠它 ——
    #: 光给一个「做对了/没做对」，玩家没法回看原画面确认。
    probe_moments: List[Dict[str, Any]] = field(default_factory=list)
    #: 明确**没做**的动作类（信号可读，但画面里没找到）。
    #:
    #: 报告里要写「这一段画面里没看到开龙」，而不是「没记录到操作」——
    #: 后者听起来像是玩家没勾，而前者才是事实。
    probe_absent: List[str] = field(default_factory=list)
    #: 模型对**操作模式清单**的裁决（逐条偏离 / 整段没做）。
    #:
    #: 这是报告正文的**主要来源** —— 模型看着画面，对照攻略原文
    #: 判断哪一条的前提成立、玩家做的不是那一件事。
    #:
    #: 每一条都已经过硬校验（见 :mod:`app.collection.vlm_review`）：
    #: 模式得在清单里、前提得被画面证实、玩家做过节点认可的事就不纠正、
    #: 引用的帧必须真实存在。
    review: List[Dict[str, Any]] = field(default_factory=list)
    #: 这一窗里的**时间片**（证据）。
    #:
    #: 分析单元是时间窗 —— 窗内做一次综合分析，结论给窗里的所有规则共用。
    #: 时间片只是证据：报告里「看这一段」定位到窗内最贴切的那一片。
    slices: List[Dict[str, Any]] = field(default_factory=list)
    #: 所属**分析段**（相邻判定单元合并而成）。
    #:
    #: 判定单元是规则原窗（「2 ~ 3 分钟」），分析单元是合并后的段
    #: （玩家/识别器对着这一段的帧做一次综合分析）。
    #: 段里的判定单元共用这份结论。
    segment_key: str = ""
    #: 该回合对应的门控帧置信度；过低表示回合边界没数准
    gate_confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "phase": self.phase,
            "label_override": self.label_override,
            "time_index": self.time_index,
            "video_time": self.video_time,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "image": self.image,
            "gate_confidence": round(self.gate_confidence, 3),
            "facts": [f.to_dict() for f in self.facts],
            "actions": [a.to_dict() for a in self.actions],
            "probe_moments": list(self.probe_moments),
            "probe_absent": list(self.probe_absent),
            "review": list(self.review),
            "slices": list(self.slices),
            "segment_key": self.segment_key,
        }


@dataclass
class RecognitionResult:
    """一次视频解析的完整产物。"""

    match_id: str
    game: str
    game_version: str
    frames: List[GateFrame] = field(default_factory=list)
    rounds: List[RecognizedRound] = field(default_factory=list)
    #: 解析过程的统计（门控帧数、精细识别次数、耗时等），用于评估方案性价比
    stats: Dict[str, Any] = field(default_factory=dict)
    #: 整体告警，例如「第 3 回合之后画面被遮挡」
    warnings: List[str] = field(default_factory=list)
    #: 原视频在 videos\ 里的 id。
    #:
    #: 必须跟着对局一起存下来 —— 报告里「看这一段」要靠它定位到原视频，
    #: 而报告是对局生成的，几小时后就没人说得清视频还在不在。
    video_id: str = ""
    #: 用的哪套识别方案、哪一版。**必须跟着结论一起存** ——
    #: 换一版识别器，结论就可能不一样，报告得说清这局是哪套得出的。
    recognizer_id: str = ""
    recognizer_version: str = ""

    def boundary_count(self) -> int:
        return sum(1 for f in self.frames if f.is_boundary)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "game": self.game,
            "game_version": self.game_version,
            "frames": [f.to_dict() for f in self.frames],
            "rounds": [r.to_dict() for r in self.rounds],
            "stats": dict(self.stats),
            "warnings": list(self.warnings),
            "recognizer_id": self.recognizer_id,
            "recognizer_version": self.recognizer_version,
        }


# --------------------------------------------------------------------------
# 识别器接口
# --------------------------------------------------------------------------


class VideoRecognizer(abc.ABC):
    """视频识别器。算法实现方只需满足这两个方法。

    拆成两步不是洁癖：门控要廉价所以能全片跑，ROI 精细识别昂贵所以
    只在门控找到的回合边界上跑。这个成本结构是立项里「团队现有设备即可
    完成、不需要大模型算力」的前提，必须体现在接口上才能被后续实现遵守。
    """

    #: 稳定标识，写入分析结果便于追溯用了哪套识别方案
    recognizer_id: str = ""
    #: 版本号。识别质量变化时递增，历史对局仍能说清当时用的是哪版。
    recognizer_version: str = ""

    @abc.abstractmethod
    def detect_round_boundaries(self, video: Any) -> List[GateFrame]:
        """第一步（廉价，全片）：找出回合边界。

        禁止在这里做精细识别 —— 一旦在门控里塞进昂贵的操作，
        「轻量方案」的成本优势就没了，而那是整个技术选型的依据。
        """

    @abc.abstractmethod
    def recognize_rounds(
        self, video: Any, boundaries: Sequence[GateFrame]
    ) -> List[RecognizedRound]:
        """第二步（昂贵，仅边界处）：对每个回合做 ROI 精细识别。"""


# --------------------------------------------------------------------------
# 降级转换
# --------------------------------------------------------------------------


@dataclass
class DegradationReport:
    """降级过程的自述。采集层把「为什么这条结论不可信」如实记下来。"""

    total_rounds: int = 0
    usable_rounds: int = 0
    dropped_rounds: int = 0
    accepted_facts: int = 0
    low_confidence_facts: List[Dict[str, Any]] = field(default_factory=list)
    rejected_facts: List[Dict[str, Any]] = field(default_factory=list)
    dropped_actions: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_rounds": self.total_rounds,
            "usable_rounds": self.usable_rounds,
            "dropped_rounds": self.dropped_rounds,
            "accepted_facts": self.accepted_facts,
            "low_confidence_facts": self.low_confidence_facts,
            "rejected_facts": self.rejected_facts,
            "dropped_actions": self.dropped_actions,
            "warnings": self.warnings,
        }


def to_raw_match(
    result: RecognitionResult,
    known_facts: set,
    known_action_facts: set,
) -> tuple:
    """把识别结果降级成内核可直接消费的 ``raw`` 对局字典。

    返回 ``(raw, DegradationReport)``。

    降级规则（全部原因为记录，不静默丢弃）：

    1. 门控置信度过低的回合 —— 整段判为缺失。边界都没数准，
       里面的 ROI 再精细也没意义。
    2. 未注册的事实 —— 丢弃并记录。识别器多认出的事实不该污染内核。
    3. 采信但置信度低的事实 —— 照常进 ``facts``，同时单独留档，
       报告层据此把「建立在其上的结论」标记出来。
    4. 未采信的事实（conf < 0.30）与未采信的动作 —— 丢弃并记录原因。
    5. 全部信息都不足的回合 —— 保留该回合，只是不带任何事实和动作。
       保留它是为了让报告能说「这一回合我们什么都不知道」，
       而不是假装这回合不存在。
    """
    report = DegradationReport()
    report.warnings.extend(result.warnings)

    rounds_out: List[Dict[str, Any]] = []
    low_conf: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    dropped_actions: List[Dict[str, Any]] = []

    for rnd in result.rounds:
        report.total_rounds += 1
        label = f"{rnd.stage}-{rnd.phase}"

        facts: Dict[str, Any] = {}
        missing: List[str] = []

        if rnd.gate_confidence < GATE_TRUST_THRESHOLD:
            # 回合边界不可信 → 这一回合的所有结论都不成立
            missing.extend(f.name for f in rnd.facts)
            rejected.append(
                {
                    "round_label": label,
                    "reason": f"回合门控置信度 {rnd.gate_confidence:.2f} 过低",
                    "facts": [f.name for f in rnd.facts],
                }
            )
            report.dropped_rounds += 1
        else:
            for det in rnd.facts:
                if det.name not in known_facts:
                    rejected.append(
                        {
                            "round_label": label,
                            "reason": f"事实 {det.name!r} 未在 {result.game} 注册",
                            "fact": det.name,
                        }
                    )
                    continue

                if det.accepted:
                    facts[det.name] = det.value
                    report.accepted_facts += 1
                    if det.flagged:
                        low_conf.append(
                            {
                                "round_label": label,
                                "fact": det.name,
                                "value": det.value,
                                "confidence": round(det.confidence, 3),
                                "method": det.method,
                            }
                        )
                else:
                    missing.append(det.name)
                    rejected.append(
                        {
                            "round_label": label,
                            "reason": det.failure or "识别置信度低于采信门槛",
                            "fact": det.name,
                            "confidence": round(det.confidence, 3),
                        }
                    )

        actions_out: List[Dict[str, Any]] = []
        for act in rnd.actions:
            if act.kind not in known_action_facts:
                dropped_actions.append(
                    {
                        "round_label": label,
                        "reason": f"动作类型 {act.kind!r} 未注册",
                        "kind": act.kind,
                    }
                )
                continue
            if not act.accepted:
                dropped_actions.append(
                    {
                        "round_label": label,
                        "reason": "识别置信度低于动作采信门槛",
                        "kind": act.kind,
                        "confidence": round(act.confidence, 3),
                    }
                )
                continue
            # 采信但置信度低的动作照常入库 —— matcher 会据此打低置信度标记，
            # 保留它才能让报告说「这一步是识别出来的，建议你确认」
            actions_out.append(
                {
                    "kind": act.kind,
                    "value": act.value,
                    "label": act.label,
                    "confidence": round(act.confidence, 3),
                    "ts": round(act.ts, 3),
                    # 来源不能丢：**画面里看出来的** ≠ **玩家自己勾的**。
                    # 丢了这一项，适配器只能默认成 player ——
                    # 于是「系统从画面里读出来的」在库里就成了玩家自述，
                    # 而报告据此说的「实际在做什么」就成了替玩家编话。
                    "source": act.source,
                }
            )

        rounds_out.append(
            {
                "stage": rnd.stage,
                "phase": rnd.phase,
                "facts": facts,
                "actions": actions_out,
                "missing_facts": sorted(set(missing)),
                "label_override": rnd.label_override,
                "time_index": rnd.time_index,
                # 回合在原视频里的位置与截图。报告的「看这一段」靠这两个
                # 字段定位 —— 少了它们，结论就只剩文字，玩家没法回看原画面。
                #
                # 用的是 video_time（视频里的秒数），不是 time_index（游戏内分钟）。
                # 两者混用会让播放器定位到录屏开头，而界面上看不出任何异常。
                "video_time": rnd.video_time,
                "image": rnd.image,
                # 探测器找到的证据时刻 —— 报告里「这段在 12:34 出现开龙迹象」
                # 靠它；玩家要能回看原画面确认，光给结论是不够的。
                "probe_moments": list(rnd.probe_moments),
                "probe_absent": list(rnd.probe_absent),
                # 模型对攻略清单的裁决。报告正文的「操作模式分析」
                # 主要靠它 —— 逐回合判定只覆盖前提能从数值判出来的那几条。
                "review": list(rnd.review),
            }
        )
        if facts or actions_out:
            report.usable_rounds += 1

    report.low_confidence_facts = low_conf
    report.rejected_facts = rejected
    report.dropped_actions = dropped_actions

    raw = {
        "match_id": result.match_id,
        "game": result.game,
        "game_version": result.game_version,
        "player": {},
        "outcome": {},
        "played_at": "",
        "rounds": rounds_out,
        # 识别诊断完整留档。报告层据此展示「为什么这条结论不可信」，
        # 同时未来重算历史对局时也能知道当时的识别质量。
        "recognition": {
            "recognizer_id": getattr(result, "recognizer_id", ""),
            "recognizer_version": getattr(result, "recognizer_version", ""),
            "stats": dict(result.stats),
            "frames": [f.to_dict() for f in result.frames],
            "degradation": report.to_dict(),
            # 原视频 id：报告里「看这一段」靠它找视频文件
            "video_id": getattr(result, "video_id", ""),
        },
    }

    return raw, report


def summarize_cost(result: RecognitionResult) -> Dict[str, Any]:
    """门控 + ROI 的成本结构，用于验证「轻量方案」确实省了算力。

    立项里说这套方案不依赖视觉大模型、现有设备即可完成。这个说法需要
    能被数据支撑 —— 不是靠嘴说，而是靠「全片廉价、只在边界处精细」这个
    比例本身可被测量。
    """
    total_frames = len(result.frames)
    boundaries = result.boundary_count()
    fine = len(result.rounds)
    ratio = (fine / total_frames) if total_frames else 0.0
    return {
        "frames_scanned": total_frames,
        "round_boundaries": boundaries,
        "fine_recognitions": fine,
        "fine_ratio": round(ratio, 4),
        "note": (
            f"门控扫描 {total_frames} 帧，精细识别 {fine} 次，"
            f"占比 {ratio:.1%} —— 昂贵识别被限制在回合边界上"
        ),
    }
