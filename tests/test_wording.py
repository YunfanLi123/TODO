"""模式库与报告的**文案体检**。

四件事，缺一件报告就会开始说空话：

1. **时间口径统一** —— 规则覆盖一段就是一段，覆盖一个点就是一个点。
   标题写「两分钟」而规则管的是第 2 到第 3 分钟，读者会以为只管第 2 分钟。
   统一成时钟写法：单点「4:00」、范围「2:00 ~ 4:00」、之后「12:00 之后」。
2. **状态记号不进报告** —— 攻略里的「状态【反向开局完成】」是它内部记状态
   用的写法，读者看到「状态【…】」只知道有个状态，不知道是什么。
3. **模糊词不进库** —— 「尽量」「必要时」「可能」「正常清线」「稳扎稳打」
   这类词一进来，结论就落不了地：玩家不知道自己该做什么。
4. **一句话只留一个时刻** —— 标题已经写明这条规则管哪一段，正文再冒一个
   裸时刻（「赵云 2:00 ~ 4:00：1:00 与中单…」），读者会以为规则管错了时间。

这个文件把四条都钉成测试：谁改回去，测试立刻红。
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.adapters import registry as _registry  # noqa: F401  触发适配器注册
from tools.hok_wording import lint, normalize  # noqa: E402

APP_ROOT = Path(__file__).resolve().parent.parent
SEED_DIR = APP_ROOT / "app" / "seed"

#: 三个库的文件名。
LIBRARIES = ("hok_modes.json", "hok_hero_modes.json", "hok_bp_modes.json")

#: 要体检的字段。
FIELDS = ("title", "premise", "action", "outcome", "deviation_hint")

#: 裸时刻。
CLOCK_RE = re.compile(r"\d{1,2}:\d{2}")

#: 时刻区间：「1:30 至 2:00」。
RANGE_RE = re.compile(r"(\d{1,2}:\d{2})\s*至\s*(\d{1,2}:\d{2})")

#: 旧的分钟写法：标题里不该再出现。
OLD_MINUTE_FORMS = (
    re.compile(r"第\s*\d+\s*分钟"),
    re.compile(r"\d+\s*~\s*\d+\s*分钟"),
)


def _load(name: str):
    return json.loads((SEED_DIR / name).read_text("utf-8"))


def _window(rule):
    for c in (rule.get("trigger") or {}).get("all", []):
        if c.get("fact") == "time_min" and c.get("op") == "between":
            return c.get("value")
    return None


def _expected_label(lo: int, hi: int) -> str:
    """窗口唯一允许的写法。"""
    if lo == hi:
        return f"{lo}:00"
    if hi >= 60:
        return f"{lo}:00 之后"
    return f"{lo}:00 ~ {hi + 1}:00"


@pytest.mark.parametrize("name", LIBRARIES)
def test_no_rule_carries_a_state_marker(name):
    """攻略内部的状态记号不进库。"""
    problems = []
    for rule in _load(name):
        for field in FIELDS:
            text = rule.get(field) or ""
            if "【" in text or "】" in text:
                problems.append(f"{rule['id']} 的 {field}：{text[:60]}")
    assert not problems, "这些字段里还留着状态记号：\n  " + "\n  ".join(problems[:15])


@pytest.mark.parametrize("name", LIBRARIES)
def test_no_rule_uses_vague_wording(name):
    """模糊词与空话不进库。"""
    problems = []
    for rule in _load(name):
        for field in FIELDS:
            issues = lint(rule.get(field) or "")
            if issues:
                problems.append(f"{rule['id']} 的 {field}：{issues} "
                                f"→ {(rule.get(field) or '')[:50]}")
    assert not problems, "这些字段里有模糊表述：\n  " + "\n  ".join(problems[:15])


@pytest.mark.parametrize("name", LIBRARIES)
def test_the_title_time_matches_the_rule_window(name):
    """标题里的时间与规则实际覆盖的窗口一致，且只用时钟写法。"""
    problems = []
    for rule in _load(name):
        win = _window(rule)
        if not win:
            continue
        lo, hi = win
        title = rule.get("title") or ""
        label = _expected_label(lo, hi)
        if label not in title:
            problems.append(f"{rule['id']} 窗口={win} 应写「{label}」"
                            f" 标题=「{title}」")
        for old in OLD_MINUTE_FORMS:
            if old.search(title):
                problems.append(f"{rule['id']} 标题还留着旧的分钟写法：{title}")
                break
    assert not problems, "这些标题的时间与窗口对不上：\n  " + "\n  ".join(problems[:15])


@pytest.mark.parametrize("name", LIBRARIES)
def test_the_text_keeps_only_one_moment(name):
    """正文里的时刻必须说清楚是哪一种。

    合法的写法：
    - 「1:00 时河道之灵刷新」—— 时刻加了「时」；
    - 「1:30 至 2:00」—— 区间；
    - 「4:00 之后的河道团」—— 时段；
    - 「等 20:00 风暴龙王」—— 等一个东西。

    不合法的：
    - 「1:00 河道之灵刷新」—— 裸时刻，读起来像标题而不是句子；
    - 攻略里那几处「1:00 与中单、游走控下中路河道之灵」更糟 ——
      它跟标题的时间对不上，读者会以为规则管错了时间。
    """
    problems = []
    for rule in _load(name):
        for field in ("premise", "action", "outcome", "deviation_hint"):
            text = rule.get(field) or ""
            if not text:
                continue
            # 区间两端不算裸时刻
            ranges = [(m.start(1), m.end(1)) for m in RANGE_RE.finditer(text)]
            ranges += [(m.start(2), m.end(2)) for m in RANGE_RE.finditer(text)]
            for m in CLOCK_RE.finditer(text):
                if any(s <= m.start() < e for s, e in ranges):
                    continue
                tail = text[m.end():].lstrip()
                head = text[:m.start()].rstrip()
                if tail.startswith(("时", "至 ", "之后", "之前")):
                    continue
                if head.endswith(("至", "等")):
                    continue
                problems.append(f"{rule['id']} 的 {field}：裸时刻「{m.group(0)}」"
                                f" → {text[:50]}")
                break
    assert not problems, "这些字段里有裸时刻：\n  " + "\n  ".join(problems[:15])


def test_moment_wording_rules():
    """时刻的三条处理规则。"""
    # 条件里的时刻补「时」
    assert normalize("1:00 河道之灵刷新") == "1:00 时河道之灵刷新"
    # 区间不动
    assert normalize("1:30 至 2:00 用这个窗口抓人") == "1:30 至 2:00 用这个窗口抓人"
    # 截止时间不动
    assert normalize("1:50 前到 4 级") == "1:50 前到 4 级"
    # 动作开头的时刻去掉
    assert normalize("1:00 与中单、游走控下中路河道之灵",
                     strip_clause_time=True) == "与中单、游走控下中路河道之灵"
    # 状态记号与加号
    assert normalize("【反向开局完成】+ 已到 4 级") == "反向开局完成、已到 4 级"


def test_the_report_itself_says_no_vague_wording():
    """报告渲染出来的文字也不能有模糊表述与状态记号。"""
    from fastapi.testclient import TestClient

    from app import db as dbmod
    from app.main import app
    from demo_data import build_demo_match

    dbmod._engine = None
    dbmod._SessionFactory = None
    dbmod.get_engine(str(APP_ROOT / ".tmp" / "wording_report.db"))
    try:
        client = TestClient(app)
        assert client.post("/api/games/hok/rules/seed").status_code == 200
        payload = build_demo_match("hok")
        r = client.post("/api/matches?player_id=local", json=payload)
        assert r.status_code == 200, r.text
        match_id = r.json()["match_id"]

        html = client.get(f"/api/matches/{match_id}/report").text
        # 报告正文里出现这些，说明文案规范化漏了
        for bad in ("【", "】", "状态【"):
            assert bad not in html, f"报告里还有「{bad}」"
        for word in ("尽量", "必要时", "正常清线", "稳扎稳打"):
            assert word not in html, f"报告里还有「{word}」"
    finally:
        dbmod._engine = None
        dbmod._SessionFactory = None
