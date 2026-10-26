"""英雄层闭合链路：规则库 → 录入 → 判定 → 报告 → 标注。

断言逻辑不在这里 —— 它在 ``tests/scenarios.py``，pytest 与本脚本共用同一份。
本文件只负责：切到独立数据库 → 跑场景 → 打印结果。

为什么要共享。早先把断言写在这个脚本里，结果 CI 跑 pytest 时这几个
场景全被漏掉 —— 最该跑的检查最容易丢。现在 ``pytest tests/`` 会把它们
一并跑掉，而本脚本仍可单独执行来看逐条输出。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from fastapi.testclient import TestClient          # noqa: E402

from app.adapters import registry                  # noqa: F401,E402  触发注册
from app.main import app                          # noqa: E402
from tests.scenarios import Recorder, SCENARIOS   # noqa: E402

SCENARIO = "hero_layer"
TITLE = "英雄层闭合链路"
DBNAME = "verify_verify_hero_layer"


def isolate() -> str:
    """指向本脚本专属数据库，保证可重复运行。"""
    from app import db as dbmod

    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", DBNAME + ".db"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    dbmod._engine = None
    dbmod._SessionFactory = None
    dbmod.get_engine(path)
    return path


def main() -> int:
    isolate()
    rec = Recorder()
    SCENARIOS[SCENARIO](TestClient(app), rec)

    printed = [c for c in rec.items if not c.name.startswith("─")]
    passed = len([c for c in printed if c.ok])
    failed = [c for c in printed if not c.ok]

    print()
    print("=" * 60)
    print(TITLE)
    print("=" * 60)
    for c in printed:
        print("  " + str(c))
    print()
    if not failed:
        print("全部通过（%d 项断言）" % passed)
        return 0
    print("失败 %d / %d 项：" % (len(failed), len(printed)))
    for c in failed:
        print("  - " + c.name)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
