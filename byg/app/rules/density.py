"""规则密度：每个局内分钟能判出多少条规则。

为什么需要它
------------
报告的价值 = 切出来的那些时刻里，**有多少条规则真的能判**。
而规则的时间窗**不是均匀的**。以王者 S45 为例（局内 1681 条）：

    分钟      0   1   2   3   4   5   6   7   8   9  10  11  12+
    能判规则 237 237 299 299 327   0 314 314 314 314 254   0  249

第 4 分钟最密（327 条：龙坑、带线、惩击、视野全在这一分钟），
而**第 5 分钟和第 11 分钟一条都没有** —— 攻略的节点在 4:00 与 6:00、
10:00 与 12:00 之间留了空档。

所以切片预算该按**规则密度**分，而不是按时间平均分：
密度大的分钟多切几刀，一条规则都没有的分钟切了也判不出任何东西。

从**种子库**读，不碰数据库
------------------------
扫描跑在后台线程里，按约定不碰数据库（见 ``app/collection/scan_jobs.py``）。
而密度只是个形状提示，用种子库算与用库里的版本算相差无几 ——
库里的版本本来就是种子灌进去的。

真要严格对齐某个版本，把这个函数的入参换成 ``get_rules`` 的结果即可。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

from . import library_source

#: 种子目录（内置兜底；可配置来源见 :mod:`app.rules.library_source`）。
SEED_DIR = library_source.BUILTIN_DIR

#: 局内规则的时间事实名。
_TIME_FACT = "time_min"

#: 数据库里的规则缓存：``(game, layer) -> (指纹, 规则列表)``。
#:
#: 指纹变了就重读 —— 这样另一个进程导入新库后，本进程不用重启。
_DB_CACHE: Dict[Any, Any] = {}


def _time_windows(rule: Dict[str, Any]) -> Optional[List[int]]:
    """取一条规则的时间窗 ``[lo, hi]``（分钟，含端点）。"""
    for cond in (rule.get("trigger") or {}).get("all", []):
        if cond.get("fact") == _TIME_FACT and cond.get("op") == "between":
            value = cond.get("value")
            if isinstance(value, (list, tuple)) and len(value) == 2:
                return [int(value[0]), int(value[1])]
    return None


def minute_density(rules: Iterable[Dict[str, Any]],
                   max_minute: int = 60) -> Dict[int, int]:
    """每个游戏分钟能判出的规则条数。

    规则的时间窗 ``[lo, hi]`` 覆盖的每一个分钟都算它一次 ——
    窗口 ``[6, 9]`` 的规则在第 6~9 分钟都可能触发，四个分钟都要算。
    """
    density: Dict[int, int] = {}
    for rule in rules:
        win = _time_windows(rule)
        if not win:
            continue
        lo, hi = win
        for minute in range(max(0, lo), min(hi, max_minute) + 1):
            density[minute] = density.get(minute, 0) + 1
    return density


def _read_rules(paths: List[str]) -> List[Dict[str, Any]]:
    rules: List[Dict[str, Any]] = []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(data, list):
            rules.extend(r for r in data if isinstance(r, dict))
    return rules


def seed_density(game: str, *, skip_bp: bool = True,
                 max_minute: int = 60) -> Dict[int, int]:
    """从该游戏的**模式库**算密度（局内规则）。

    与 :func:`app.rules.store._seed_files` 用同一套文件匹配规则 ——
    按后缀匹配（``hok_modes.json`` / ``hok_hero_modes.json`` 都收），
    同一个可配置来源（见 :mod:`app.rules.library_source`）。
    """
    return minute_density(
        _read_rules(library_source.library_files(
            game, skip_bp=skip_bp, only_bp=not skip_bp)),
        max_minute=max_minute)


def density_weights(density: Dict[int, int]) -> Dict[int, float]:
    """把密度归一化成 0~1 的权重。

    没有规则的分钟权重为 0 —— 那里切了也判不出东西。
    """
    if not density:
        return {}
    top = max(density.values())
    if top <= 0:
        return {m: 0.0 for m in density}
    return {m: (n / top) for m, n in density.items()}


def _seed_rules(game: str, *, skip_bp: bool = True) -> List[Dict[str, Any]]:
    """读该游戏的模式库规则（局内 / BP 分开）。

    **数据库优先**（2026-10-31 起，2026-11-07 换成 Qdrant）
    -------------------------------------------------------
    模式库已经迁进向量库（Qdrant 的 ``mode_documents`` 集合），这里先查库；
    库里没有这个游戏的记录时，才回退到文件
    （用户自己的 git 仓库 → 内置兜底，见 :mod:`app.rules.library_source`）。

    两边的内容**形状完全一样**（都是原始规则记录），
    所以判定口径一字不变 —— 这只是「同一份内容从哪儿读」的差别。

    进程内缓存按**库指纹**（条数 + 索引版本）失效：
    另一个进程刚重建索引时，本进程也能发现（不重启就能生效）。
    """
    layer = "bp" if not skip_bp else None
    try:
        from ..rag.store import ModeStore

        store = ModeStore()
        cached = _DB_CACHE.get((game, layer))
        fingerprint = store.fingerprint(game=game, layer=layer or "")
        if cached is not None and cached[0] == fingerprint and fingerprint:
            return cached[1]
        rules = store.rules(game=game, layer=layer or "")
        if rules:
            _DB_CACHE[(game, layer)] = (fingerprint, rules)
            return rules
    except Exception:                                  # noqa: BLE001
        # 库不可用（老库没这张表 / 数据库连不上）—— 静默回退文件。
        pass
    return _read_rules(library_source.library_files(
        game, skip_bp=skip_bp, only_bp=not skip_bp))
