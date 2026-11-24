"""新识别子系统（第二版）。

**它是唯一的识别链路** —— 旧的「按规则窗口切段 + 逐窗枚举动作」那套
（``app/collection/vlm_recognizer.py`` / ``vlm_review.py`` / ``probes.py`` /
``window_analysis.py``）已经删掉了。「上传一个视频 → 出报告」这条路
只跑这里。

模块分工
--------
``beat``       统一真源结构：``Beat`` / ``Timeline``
``videopack``  视频打包：整局压成能上传的形态 / 切裁决片段
``watcher``    **模型直接看整局录像** → 事件流（当前链路的第一步）
``sampler``    按画面变化切段（**已不在链路上**，留作旧链路）
``reader``     让模型逐段读画面（**已不在链路上**，留作旧链路）
``clockaxis``  时间轴：把「视频第几秒」对到「局内第几秒」（旧链路）
``timefit``    抗离群拟合（Theil–Sen 起步）
``prefill``    对局级事实：胜负 / 地图位置 / BP 记录（英雄与分路由裁决层定）
``judge``      判定层：把时间线对齐到规则库，算「该做而没做」
``selector``   候选检索 skill：模型自己决定「这一局该查哪几条攻略」
``adjudicator`` 模型裁决层：代码只当筛子，判官是模型
``observer``   全局综述层：让模型看到整局全貌 + 攻略之外的问题
``pipeline``   整条链路串起来
``bridge``     桥：把 v2 的时间线与判定结果转成报告层认识的形状

三层职责的分工
--------------
- **观察**：模型说「这段在干什么」（reader + observer）；
- **对照**：拿攻略当**参考**，问「做到了没有」（adjudicator）；
- **判定**：纯代码的机械比对（judge），可复现、便宜、可交叉验证。

「书是参考，不是提纲」—— 模式库永远只在**对照**这一层出现，
不进入**观察**这一层：观察层的输出是模型自己的话。
"""

from . import prefill, selector, videopack, watcher
from .adjudicator import (AdjudicatorConfig, ModelJudgement, ModelVerdict,
                           adjudicate)
from .beat import (Beat, Timeline,
                   UNCLEAR, PHASE_IN_GAME, PHASE_POST_GAME, PHASE_PRE_GAME,
                   PHASE_UNKNOWN)
from .bridge import (beat_sentences, bucket_beats, coverage_summary,
                     coverage_windows, rule_entries, to_legacy_rounds)
from .clockaxis import ClockAxis, FrameClock, pick_anchor, resolve_segment
from .judge import (MatchFacts, RuleJudgement, Judgement, judge,
                    Candidate, screen_candidates,
                    STATUS_SATISFIED, STATUS_DEVIATED, STATUS_UNDETERMINED,
                    STATUS_NOT_APPLICABLE)
from .observer import ObserverConfig, Overview, guide_digest, observe
from .pipeline import PipelineConfig, match_start_t, run_pipeline
from .sampler import ContentSampler, SamplerConfig, Segment
from .selector import (RetrievalRequest, SelectionResult, SelectorConfig,
                       parse_requests, select_candidates)

__all__ = [
    "Beat",
    "Timeline",
    "UNCLEAR",
    "PHASE_PRE_GAME",
    "PHASE_IN_GAME",
    "PHASE_POST_GAME",
    "PHASE_UNKNOWN",
    "ContentSampler",
    "SamplerConfig",
    "Segment",
    "ClockAxis",
    "FrameClock",
    "pick_anchor",
    "resolve_segment",
    "MatchFacts",
    "RuleJudgement",
    "Judgement",
    "judge",
    "Candidate",
    "screen_candidates",
    "AdjudicatorConfig",
    "ModelJudgement",
    "ModelVerdict",
    "adjudicate",
    "bucket_beats",
    "beat_sentences",
    "coverage_windows",
    "coverage_summary",
    "rule_entries",
    "to_legacy_rounds",
    "ObserverConfig",
    "Overview",
    "guide_digest",
    "observe",
    "PipelineConfig",
    "run_pipeline",
    "match_start_t",
    "RetrievalRequest",
    "SelectionResult",
    "SelectorConfig",
    "parse_requests",
    "select_candidates",
    "selector",
    "prefill",
    "videopack",
    "watcher",
    "STATUS_SATISFIED",
    "STATUS_DEVIATED",
    "STATUS_UNDETERMINED",
    "STATUS_NOT_APPLICABLE",
]
