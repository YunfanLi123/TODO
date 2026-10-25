"""操作模式库从**哪来**。

为什么要可配置
--------------
用户原话：**「操作模式库，不要那么死板了」**。

模式库是**知识**，不是设置：它有版本、有人 review、要能 diff、要能回滚。
把它锁在代码目录里，用户就只能「看着报告生气，却改不了那一条」。

所以它可以来自**一个独立的 git 仓库**（用户自己的 GitHub），
改完推上去，应用下一次载入就用新的。内置那份只当**兜底** ——
仓库拉不到时不能让产品开天窗。

查找顺序
--------
1. 环境变量 ``MODE_LIBRARY_PATH``（部署时最方便）；
2. ``<数据目录>/mode-library`` —— exe 旁边，用户 clone 下来就生效；
3. ``<项目>/mode-library`` —— 开发机上那份可编辑副本；
4. ``app/seed/`` —— **内置兜底**。

注意：只有**操作模式**那几个文件走这条路（``*_modes.json`` /
``*_rules.json`` / ``*_retired.json`` / ``*_skipped.json``）。
英雄与标签数据（``hok_heroes.json`` / ``hok_hero_tags.json``）是**素材**，
不是知识库，仍从内置读。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

#: 内置兜底目录。
BUILTIN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "seed")

#: 走「可配置来源」的文件后缀。
LIBRARY_SUFFIXES = ("_modes.json", "_rules.json", "_retired.json", "_skipped.json")

#: 素材文件（不走可配置来源）。
ASSET_SUFFIXES = ("_heroes.json", "_hero_tags.json")


def _data_dir() -> str:
    """数据目录（与 :mod:`app.db` 同一个）。"""
    return os.environ.get("MATCH_DATA_DIR") or os.path.join(
        _project_dir(), "data")


def _project_dir() -> str:
    """项目根目录。``app/rules/xxx.py`` 往上退**两级**。"""
    return os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))


def candidate_dirs() -> List[str]:
    """可能的来源目录，按优先级排。"""
    out: List[str] = []
    env = os.environ.get("MODE_LIBRARY_PATH", "").strip()
    if env:
        out.append(env)
    out.append(os.path.join(_data_dir(), "mode-library"))
    out.append(os.path.join(_project_dir(), "mode-library"))
    out.append(BUILTIN_DIR)
    return out


def _has_library_files(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for name in os.listdir(path):
        if name.endswith(LIBRARY_SUFFIXES):
            return True
    return False


def library_dir() -> str:
    """模式库文件所在的目录。找不到就用内置兜底。"""
    for path in candidate_dirs():
        if _has_library_files(path):
            return path
    return BUILTIN_DIR


def asset_dir() -> str:
    """素材（英雄、标签）所在目录 —— 始终是内置那份。"""
    return BUILTIN_DIR


def describe() -> Dict[str, Any]:
    """当前用的是哪一份（给界面显示）。"""
    path = library_dir()
    builtin = os.path.abspath(BUILTIN_DIR) == os.path.abspath(path)
    return {
        "path": path,
        "is_builtin": builtin,
        "candidates": candidate_dirs(),
        "files": sorted(
            f for f in (os.listdir(path) if os.path.isdir(path) else [])
            if f.endswith(LIBRARY_SUFFIXES)
        ) if os.path.isdir(path) else [],
    }


def library_files(game: str, *, skip_bp: bool = False,
                  only_bp: bool = False) -> List[str]:
    """该游戏的**模式库**文件（可配置来源优先）。"""
    return _match(game, library_dir(), skip_bp=skip_bp, only_bp=only_bp)


def asset_files(game: str) -> List[str]:
    """该游戏的**素材**文件（内置）。"""
    return _match(game, asset_dir(), skip_bp=False, only_bp=False)


def _match(game: str, root: str, *, skip_bp: bool = False,
           only_bp: bool = False) -> List[str]:
    if not os.path.isdir(root):
        return []
    prefix = f"{game}_"
    suffixes = ("_modes.json", "_rules.json")
    out: List[str] = []
    for name in sorted(os.listdir(root)):
        if not (name.startswith(prefix) and name.endswith(suffixes)):
            continue
        is_bp = name.endswith("_bp_modes.json")
        if only_bp and not is_bp:
            continue
        if skip_bp and is_bp:
            continue
        out.append(os.path.join(root, name))
    return out
