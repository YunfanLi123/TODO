"""exe 第一次跑起来要灌**两个**规则库。

局内库（S45）与 BP 库（BP-S45）是两个版本号，strict 导入不许一次跨版本，
所以得分两次灌。启动器曾经只灌了局内库 —— BP 规则页在 exe 里就一直是
「0 条」，而且**不报错**：页面只是空着，最难发现的那种问题。

所以这里盯的是「首启之后两个库都在」：
1. 局内库有规则；
2. BP 库也有规则；
3. 已经在库里时，重复启动不会重复灌、也不会把库清空。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _bootstrap_into(tmp_path, monkeypatch):
    """把启动器的首启流程跑在一个临时数据库上。"""
    from app import db as dbmod
    import scripts.launcher as launcher

    monkeypatch.setenv("MATCH_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(dbmod, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(dbmod, "DB_PATH", str(tmp_path / "app.db"))
    # 源码模式下 resource_dir() 指的是 scripts/，那里没有 app 包。
    monkeypatch.setattr(launcher, "resource_dir", lambda: str(ROOT))

    dbmod._engine = None
    dbmod._SessionFactory = None
    try:
        out = launcher.bootstrap(str(tmp_path))
        with dbmod.session_scope() as s:
            from app.rules.store import BP_VERSION, get_rules

            main_rules = get_rules(s, "hok", "S45")
            bp_rules = get_rules(s, "hok", BP_VERSION)
        return out, main_rules, bp_rules
    finally:
        dbmod._engine = None
        dbmod._SessionFactory = None


def test_the_first_run_seeds_both_libraries(tmp_path, monkeypatch):
    """首启之后，局内库和 BP 库都要有规则。"""
    out, main_rules, bp_rules = _bootstrap_into(tmp_path, monkeypatch)

    assert main_rules, "局内库是空的"
    assert bp_rules, "BP 规则库是空的 —— 启动器只灌了局内库"
    assert "BP 规则" in out, f"控制台没有报告 BP 库已灌入：\n{out}"


def test_a_second_run_keeps_the_bp_library(tmp_path, monkeypatch):
    """再启动一次，BP 库还在（不会被重复灌成空，也不会消失）。"""
    _bootstrap_into(tmp_path, monkeypatch)
    out, main_rules, bp_rules = _bootstrap_into(tmp_path, monkeypatch)

    assert main_rules and bp_rules
    assert "载入 hok  BP 规则" not in out, "第二次启动又把 BP 库灌了一遍"
