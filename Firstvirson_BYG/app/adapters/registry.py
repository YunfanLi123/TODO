"""适配器注册入口。

``app/adapters/__init__.py`` 留空由分层骨架约定，这里单独开一个模块承载
「导入即注册」的副作用：谁需要拿适配器就 ``from app.adapters.registry import
REGISTRY``，导入顺序不再影响 :data:`app.adapters.base.REGISTRY` 的内容。

注意 :data:`REGISTRY` 本身是从 base 模块 re-export 的同一个字典对象 ——
base 里注册的任何适配器，这里都能看到。
"""

from __future__ import annotations

from typing import Dict, List

from .base import REGISTRY, GameAdapter, get_adapter, register, registered_games
from . import tft as _tft  # noqa: F401  —— 导入即注册
from . import jcc as _jcc  # noqa: F401  —— 导入即注册
from . import honor_of_kings as _hok  # noqa: F401  —— 导入即注册

#: 三款游戏的适配器实例。未注册任何一款时为 []，测试据此保证导入链路完好。
ADAPTERS: Dict[str, GameAdapter] = dict(REGISTRY)

#: 本项目收录的游戏 id，顺序固定 —— 报告与界面按这个顺序展示。
GAME_IDS: List[str] = ["tft", "jcc", "hok"]

__all__ = [
    "REGISTRY",
    "ADAPTERS",
    "GAME_IDS",
    "GameAdapter",
    "get_adapter",
    "register",
    "registered_games",
]
