"""适配器注册入口。

``app/adapters/__init__.py`` 留空由分层骨架约定，这里单独开一个模块承载
「导入即注册」的副作用：谁需要拿适配器就 ``from app.adapters.registry import
REGISTRY``，导入顺序不再影响 :data:`app.adapters.base.REGISTRY` 的内容。

注意 :data:`REGISTRY` 本身是从 base 模块 re-export 的同一个字典对象 ——
base 里注册的任何适配器，这里都能看到。

**这个产品现在只做王者荣耀**（2026-11-01）
-------------------------------------
自走棋的适配器（``tft.py`` / ``jcc.py``）**已整个删掉**：

- 它们的产品资料早就删了（模式库、种子、攻略全都只剩王者那一份），
  留着适配器模块只是留着一条「切过去看到的是另一套东西」的路；
- 而那两套东西的数据口径并不一致 —— 自走棋有「回合」、王者只有时间片，
  两边各写一份量词迟早只改一处（这轮就是这么暴露的）。

**扩展位还留着**：:class:`~app.adapters.base.GameAdapter` 的抽象不动 ——
将来真要加第二款游戏，写一个适配器注册进来就行，内核不用改。
"""

from __future__ import annotations

from typing import Dict, List

from .base import REGISTRY, GameAdapter, get_adapter, register, registered_games
from . import honor_of_kings as _hok  # noqa: F401  —— 导入即注册

#: 本项目的适配器实例。未注册任何一款时为 []，测试据此保证导入链路完好。
ADAPTERS: Dict[str, GameAdapter] = dict(REGISTRY)

#: 本项目收录的游戏 id，顺序固定 —— 报告与界面按这个顺序展示。
GAME_IDS: List[str] = ["hok"]

__all__ = [
    "REGISTRY",
    "ADAPTERS",
    "GAME_IDS",
    "GameAdapter",
    "get_adapter",
    "register",
    "registered_games",
]
