"""英雄层的「建议」必须**按条件写**，不能退化成一张备选清单。

背景
----
同一动作族里的多条分支会合并成一条规则 —— 因为它们的前提
（「对方有主宰」「我方先到龙坑」）对局数据判不了，两条可能同时成立。
合并的是**判定**，不是**建议**：

- 判定合并 → 期望动作取并集，做到其中之一就算做到；
- 建议也合并 → 报告里读到的就是「打法一；打法二；打法三」，
  玩家不知道哪条适用，只会觉得这是一张清单。

所以 ``deviation_hint`` 保留分支级条件，写成「什么情况下：做什么」。
这里钉住三件事：

1. 每条建议都带条件；
2. **合并前的每一条打法都在建议里** —— 条件化不能以漏掉某个分支为代价；
3. 条件确实来自攻略原文的前提（归一化后逐字对得上）。

外加一条：状态标记（``【反向开局完成】``）不许漏进建议 ——
那是攻略给下一个节点做引用输入的内部标记，读者读不懂。
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.adapters import registry as _registry  # noqa: F401  触发适配器注册
from app.core.dsl import DEFAULT_SUBJECTIVE_MARKERS

APP_ROOT = Path(__file__).resolve().parent.parent
HERO_MODES_PATH = APP_ROOT / "app" / "seed" / "hok_hero_modes.json"


@pytest.fixture(scope="module")
def hero_modes() -> List[Dict[str, Any]]:
    return json.loads(HERO_MODES_PATH.read_text("utf-8"))


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from app import db as dbmod
    from app.main import app

    dbmod._engine = None
    dbmod._SessionFactory = None
    dbmod.get_engine(str(tmp_path / "hero_hint.db"))

    yield TestClient(app)

    dbmod._engine = None
    dbmod._SessionFactory = None


def _norm(text: str) -> str:
    """归一化：把标点、状态标记、分隔符都去掉，只比字符本身。"""
    return re.sub(r"[\s【】\[\]（）()、,，。；;：:+＋\-—＞>→/]", "", text or "")


def _clauses(hint: str) -> List[str]:
    """建议按句拆开。"""
    return [c for c in re.split(r"(?<=。)", hint or "") if c.strip()]


def test_every_hint_states_its_condition(hero_modes):
    """每条建议都要说清楚「什么情况下」。"""
    missing = [m["id"] for m in hero_modes if "时：" not in m["deviation_hint"]]
    assert not missing, f"这些建议没写条件：{missing[:10]}"


def test_no_branch_playstyle_is_dropped(hero_modes):
    """合并前的每一条打法都得在建议里。

    条件化最容易出的错是「顺手把没写条件的分支丢了」——
    报告读起来很顺，但攻略里的某个打法就没了。
    """
    problems: List[str] = []
    for m in hero_modes:
        hint = m["deviation_hint"]
        # ``action`` 是合并后的原文动作，按「；」拆开就是每条分支的动作行。
        for part in (m.get("action") or "").split("；"):
            part = part.strip()
            if not part:
                continue
            if part not in hint:
                problems.append(f"{m['id']} 少了动作「{part[:30]}」")
    assert not problems, "建议漏了打法：\n  " + "\n  ".join(problems[:20])


def test_the_conditions_come_from_the_guide_premise(hero_modes):
    """条件短语必须来自这条模式的前提（归一化后逐字对得上）。"""
    problems: List[str] = []
    for m in hero_modes:
        premise = _norm(m.get("premise") or "")
        for clause in _clauses(m["deviation_hint"]):
            if "时：" not in clause:
                continue
            cond = _norm(clause.split("时：")[0])
            if not cond:
                continue
            if cond not in premise:
                problems.append(f"{m['id']} 的条件「{clause.split('时：')[0]}」不在前提里")
    assert not problems, "条件对不上原文前提：\n  " + "\n  ".join(problems[:20])


def test_no_state_marker_leaks_into_the_hint(hero_modes):
    """``【反向开局完成】`` 这类内部标记不许出现在给玩家看的建议里。"""
    bad = [m["id"] for m in hero_modes
           if "【" in m["deviation_hint"] or "】" in m["deviation_hint"]]
    assert not bad, f"建议里还有状态标记：{bad[:10]}"


def test_the_hint_stays_free_of_subjective_wording(hero_modes):
    """建议里不能出现主观判断词。

    建议现在含原文的前提，前提也是攻略的话术 ——
    所以这条校验从「动作」扩到了「前提 + 建议」。
    """
    bad: List[str] = []
    for m in hero_modes:
        for marker in DEFAULT_SUBJECTIVE_MARKERS:
            if marker in m["deviation_hint"]:
                bad.append(f"{m['id']} 含「{marker}」")
    assert not bad, "建议里有主观判断词：\n  " + "\n  ".join(bad[:20])


def test_the_report_prints_the_advice_one_clause_per_line(client):
    """报告里一句一行 —— 条件与做法的对应关系才对得上。"""
    from demo_data import build_demo_match

    assert client.post("/api/games/hok/rules/seed").status_code == 200
    r = client.post("/api/matches", json=build_demo_match("hok"))
    assert r.status_code == 200, r.text
    match_id = r.json()["match_id"]

    html = client.get(f"/api/matches/{match_id}/report").text
    assert "advice-line" in html, "报告没有按句分行渲染建议"
