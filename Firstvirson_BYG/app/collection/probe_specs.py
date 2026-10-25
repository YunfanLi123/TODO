"""探测器规格：**标定值**（阈值、区域名、判据参数都在这里）。

为什么单独一个文件
------------------
探测器的判据是「**在画面上找证据**」，里面的数字（停留几秒算开龙、
折返率多高算刷野）全是**经验值**。把它们写在代码里，改一次就要翻文件、
改代码、跑回归；写成标定文件，改完重启就生效 —— 跟
``hud_regions.json`` / ``map_geometry.json`` 是同一套做法。

三态约定
--------
每个探测器返回三种状态里的一种：

- ``found``   —— 看到了证据；
- ``absent``  —— 这一类信号**可读**，但明确没有（只有强探测器能给）；
- ``unknown`` —— 读不出来。

**只有 ``absent`` 才允许判「该做而没做」**；``unknown`` 一律「无法判定」。
这是「读错比读不到危险」的落地形式。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

#: 规格文件（随包打包）。
_SPECS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "probe_specs.json")


@dataclass
class ProbeSpec:
    """一个探测器的规格。"""

    #: 键：``<动作种类>:<动作取值>``，例如 ``objective_action:farm_jungle``
    key: str
    #: 动作种类（``objective_action`` / ``positioning_action`` / …）
    kind: str
    #: 动作取值（``farm_jungle`` / ``take_dragon`` / …）
    value: str
    #: 人话说明（报告里用）
    desc: str = ""
    #: 强探测器 = 它的信号足够干净，读不出「有」就能说「没有」。
    strong: bool = False
    #: 允许给 ``absent``（弱探测器必须为 False）
    allow_absent: bool = False
    #: 判据参数（各探测器自己解释）
    params: Dict[str, Any] = field(default_factory=dict)
    #: 这个探测器在本轮**是否实现**了（没实现的 → 永远 unknown）
    enabled: bool = True

    @property
    def label(self) -> str:
        return self.desc or self.value


def load_specs(path: Optional[str] = None) -> Dict[str, ProbeSpec]:
    """读规格文件。读不到就返回空 dict（识别层据此「什么都不报」）。"""
    p = path or _SPECS_JSON
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    out: Dict[str, ProbeSpec] = {}
    for key, spec in (data.get("规格") or {}).items():
        if not isinstance(spec, dict):
            continue
        kind, _, value = key.partition(":")
        out[key] = ProbeSpec(
            key=key,
            kind=kind or spec.get("kind", ""),
            value=value or spec.get("value", ""),
            desc=spec.get("说明", ""),
            strong=bool(spec.get("强", False)),
            allow_absent=bool(spec.get("允许判没做", spec.get("强", False))),
            params=dict(spec.get("参数") or {}),
            enabled=bool(spec.get("启用", True)),
        )
    return out


def spec_key(kind: str, value: str) -> str:
    """探测器键。"""
    return f"{kind}:{value}"
