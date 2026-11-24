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
LIBRARY_SUFFIXES = ("_modes.json", "_rules.json", "_retired.json", "_skipped.json",
                    "_patch_notes.json")

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
    """当前用的是哪一份（给界面显示）。

    2026-10-31 起，**数据库优先**（见 :func:`app.rules.density._seed_rules`）：
    库里有这个游戏的模式时用库里的；库里没有才回退文件。
    这里把两边的状态都报出来 —— 用户要看得见「现在到底在用哪一份」。
    """
    path = library_dir()
    builtin = os.path.abspath(BUILTIN_DIR) == os.path.abspath(path)
    info = {
        "path": path,
        "is_builtin": builtin,
        "candidates": candidate_dirs(),
        "files": sorted(
            f for f in (os.listdir(path) if os.path.isdir(path) else [])
            if f.endswith(LIBRARY_SUFFIXES)
        ) if os.path.isdir(path) else [],
    }
    info["database"] = describe_database()
    return info


def describe_database() -> Dict[str, Any]:
    """RAG 库的状态（有没有这个游戏的模式、多少条）。

    读不出来就说读不出来 —— **不假装库是空的**。
    """
    try:
        from ..rag import qdrant_client as qc
        from ..rag.store import ModeStore

        store = ModeStore()
        stats = store.stats()
        return {
            "available": bool(stats.get("documents")),
            "documents": int(stats.get("documents") or 0),
            "chunks": int(stats.get("chunks") or 0),
            "with_embedding": int(stats.get("with_embedding") or 0),
            "layers": list(stats.get("layers") or []),
            # 库住在**数据目录**里（``<数据目录>/qdrant``）。
            "path": qc.qdrant_path(),
            "mode": "server" if qc.server_url() else "local",
        }
    except Exception as exc:                          # noqa: BLE001
        return {"available": False, "error": "%s: %s" % (type(exc).__name__, exc)}


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
