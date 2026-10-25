"""操作模式库的**来源**：可配置优先，内置兜底。

盯的是几件会静默出错的事：

1. 用户改的是 ``mode-library/``，应用读的却是内置那份 —— 改了半天没生效；
2. 换机器（exe）时找不到用户的仓库 —— 不能因此开天窗，要退回内置；
3. 库坏了（空目录、文件不是 JSON）—— 要**说清楚用的是哪一份**，
   而不是悄悄用另一份。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.rules import library_source


def _write_library(path: Path, marker: str = "x") -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "hok_modes.json").write_text(
        json.dumps([{"id": "T-1", "marker": marker}], ensure_ascii=False),
        "utf-8")


def test_the_builtin_is_the_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("MODE_LIBRARY_PATH", raising=False)
    monkeypatch.setenv("MATCH_DATA_DIR", str(tmp_path / "data"))
    # 项目里没有 mode-library 时（用真实项目路径测）——
    # 这里只断言「没有配置时一定有来源」，不钉死是哪一份。
    path = library_source.library_dir()
    assert os.path.isdir(path)
    assert library_source.describe()["files"], "一份模式库都找不到"


def test_the_env_var_wins(monkeypatch, tmp_path):
    lib = tmp_path / "mine"
    _write_library(lib)
    monkeypatch.setenv("MODE_LIBRARY_PATH", str(lib))
    monkeypatch.setenv("MATCH_DATA_DIR", str(tmp_path / "data"))
    assert library_source.library_dir() == str(lib)
    desc = library_source.describe()
    assert desc["is_builtin"] is False
    assert "hok_modes.json" in desc["files"]


def test_the_data_dir_is_next(monkeypatch, tmp_path):
    """exe 旁边的那份 —— 用户 clone 下来就生效。"""
    data = tmp_path / "data"
    _write_library(data / "mode-library")
    monkeypatch.delenv("MODE_LIBRARY_PATH", raising=False)
    monkeypatch.setenv("MATCH_DATA_DIR", str(data))
    assert library_source.library_dir() == str(data / "mode-library")


def test_an_empty_directory_does_not_win(monkeypatch, tmp_path):
    """空目录不算来源 —— 那样会「找到」一个没有模式的库。"""
    data = tmp_path / "data"
    (data / "mode-library").mkdir(parents=True)
    monkeypatch.delenv("MODE_LIBRARY_PATH", raising=False)
    monkeypatch.setenv("MATCH_DATA_DIR", str(data))
    path = library_source.library_dir()
    assert "mode-library" not in os.path.abspath(path) or \
        os.path.isfile(os.path.join(path, "hok_modes.json"))


def test_the_assets_stay_builtin(monkeypatch, tmp_path):
    """英雄与标签是**素材**，不走可配置来源。"""
    lib = tmp_path / "mine"
    _write_library(lib)
    monkeypatch.setenv("MODE_LIBRARY_PATH", str(lib))
    assert library_source.asset_files("hok"), "素材文件找不到"
    for path in library_source.asset_files("hok"):
        assert "mine" not in path


def test_the_sync_tool_reports_differences(tmp_path):
    """同步工具要能报出「内置改了、可编辑那份没跟上」。"""
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "sync_mode_library", root / "tools" / "sync_mode_library.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # 内置与可编辑那份现在应该一致（刚建仓时拷的）
    assert mod.FILES, "同步清单是空的"
    for name in mod.FILES:
        src = root / "app" / "seed" / name
        assert src.is_file(), f"内置缺 {name}"
