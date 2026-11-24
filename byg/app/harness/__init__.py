"""以 LLM 为核心的 harness 内核。

这一包是「重写内核」的第一步：**判定权从规则引擎交回模型**。

四层的分工
----------
1. **感知**（确定性）：抽帧、时钟、轨迹、英雄、分路 —— 在
   :mod:`app.collection` 里，已经是可复核的。
2. **上下文构建**：为每一窗组装模型要看的全部东西（帧、边界帧、
   轨迹图、时间线、位置、身份、攻略清单）—— 在
   :mod:`app.collection.vlm_recognizer`。
3. **模型循环**：模型判读 + 指出看不清、要再看哪 —— 在
   :mod:`app.collection.vlm_recognizer`。
4. **接地与落地**：校验模型输出、聚合成统计、写报告 ——
   在 :mod:`app.harness.kernel` 与 :mod:`app.report.builder`。

旧内核（:mod:`app.core.matcher` / :mod:`app.core.attribution`）
保留给**玩家手工录入**的对局 —— 那条路上没有画面、没有模型，
规则引擎是唯一的判断来源。视频来源的对局改由本包判定。
"""

from .kernel import HarnessAnalysis, analyze_from_model
from .verdict import (
    STATUS_DEVIATED,
    STATUS_SATISFIED,
    STATUS_UNDETERMINED,
    HarnessStats,
    ModelVerdict,
)

__all__ = [
    "HarnessAnalysis",
    "analyze_from_model",
    "HarnessStats",
    "ModelVerdict",
    "STATUS_SATISFIED",
    "STATUS_DEVIATED",
    "STATUS_UNDETERMINED",
]
