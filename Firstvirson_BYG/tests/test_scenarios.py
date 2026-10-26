"""端到端场景的 pytest 包装。

**这些场景的断言逻辑只有一份** —— 在 :mod:`tests.scenarios` 里。
``verify_*.py`` 薄壳调的是同一个函数，所以不存在「脚本里改了、pytest 里
还是旧的」这种分叉，也不会出现 CI 跑 pytest 时把最该跑的检查漏掉。

每个场景跑在自己的临时数据库里，互不影响且可重复运行。
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.adapters import registry as _registry  # noqa: F401  触发适配器注册
from tests.scenarios import SCENARIOS, Recorder


@pytest.fixture
def client_for(tmp_path):
    """给一个指向独立临时库的 TestClient。

    独立库是关键：场景之间共享主库会让结果取决于「上一轮跑过什么」，
    同样的代码第二次跑出不同的失败。
    """
    from fastapi.testclient import TestClient

    from app import db as dbmod
    from app.main import app

    path = str(tmp_path / "scenario.db")
    dbmod._engine = None
    dbmod._SessionFactory = None
    dbmod.get_engine(path)

    yield TestClient(app)

    dbmod._engine = None
    dbmod._SessionFactory = None


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario(client_for, name):
    rec = Recorder()
    SCENARIOS[name](client_for, rec)

    failed = rec.failed
    detail = "\n".join(
        f"  [{c.section}] {c.name}" for c in failed
    )
    assert not failed, (
        f"场景 {name} 有 {len(failed)} 项断言失败：\n{detail}"
    )

    passed = len([c for c in rec.items
                  if c.ok and not c.name.startswith("─")])
    assert passed >= 20, (
        f"场景 {name} 只跑了 {passed} 项断言 —— "
        f"是不是中途 return 了？"
    )


# --------------------------------------------------------------------------
# 查询次数护栏
#
# 「遍历全部对局」的路径（成长曲线、版本回测、覆盖率统计）很容易在不知情
# 地写出 N+1：ORM 的 lazy load 逐条回表，代码读起来完全正常，对局量小的
# 时候也看不出问题，直到某天报告页开始卡。
#
# 这里不比较耗时（机器差异太大），只断言「查询次数不随对局数线性增长」——
# 那才是真正的退化信号。
# --------------------------------------------------------------------------

from sqlalchemy import event  # noqa: E402
from sqlalchemy.engine import Engine  # noqa: E402


@pytest.fixture
def query_counter():
    counter = {"n": 0}

    @event.listens_for(Engine, "before_cursor_execute")
    def _count(conn, cursor, statement, params, context, executemany):
        counter["n"] += 1

    yield counter

    event.remove(Engine, "before_cursor_execute", _count)


def _seed_matches(n: int) -> None:
    import copy

    from app.adapters.base import get_adapter
    from app.collection import match as match_col
    from app.db import session_scope

    from demo_data import build_demo_match

    adapter = get_adapter("tft")
    base = build_demo_match("tft")
    with session_scope() as s:
        for i in range(n):
            raw = copy.deepcopy(base)
            raw["match_id"] = f"nplus1-{i:03d}"
            raw["played_at"] = f"2026-01-{1 + i:02d} 20:00"
            match_col.save_match(s, adapter, raw)


def test_list_matches_does_not_n_plus_one(tmp_path, query_counter):
    """列出 20 局 + 访问每局的 analyses，查询次数必须是常数级。"""
    from app import db as dbmod
    from app.collection.match import list_matches
    from app.db import get_engine, session_scope

    path = str(tmp_path / "nplus1.db")
    dbmod._engine = None
    dbmod._SessionFactory = None
    get_engine(path)

    from app.rules.store import load_seed_library

    with session_scope() as s:
        load_seed_library(s, "tft")
    _seed_matches(20)

    query_counter["n"] = 0
    with session_scope() as s:
        matches = list_matches(s, player_id="local", game="tft")
        for m in matches:
            _ = m.analyses          # 正是这一行曾触发 N+1

    assert query_counter["n"] <= 5, (
        f"20 局产生了 {query_counter['n']} 条查询 —— "
        f"N+1 回来了（预加载被去掉了？）"
    )


def test_growth_endpoint_query_count_is_flat(tmp_path, query_counter):
    """成长曲线遍历全部对局，查询次数不应随局数增长。"""
    from fastapi.testclient import TestClient

    from app import db as dbmod
    from app.db import get_engine, session_scope
    from app.main import app as fastapi_app
    from app.rules.store import load_seed_library

    path = str(tmp_path / "growth.db")
    dbmod._engine = None
    dbmod._SessionFactory = None
    get_engine(path)

    with session_scope() as s:
        load_seed_library(s, "tft")

    client = TestClient(fastapi_app)
    _seed_matches(15)

    query_counter["n"] = 0
    resp = client.get("/api/growth/tft")
    assert resp.status_code == 200
    assert query_counter["n"] <= 12, (
        f"15 局产生了 {query_counter['n']} 条查询 —— 成长曲线出现 N+1"
    )
