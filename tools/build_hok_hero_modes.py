"""从英雄级决策树生成英雄层的操作模式库。

用法::

    python tools/build_hok_hero_modes.py            # 生成 app/seed/hok_hero_modes.json
    python tools/build_hok_hero_modes.py --check    # 只校验，不写文件

生成链路
--------
    app/seed/sources/hok_s45_hero_guides/*.txt      5 份英雄攻略原文（唯一真源）
        ↓ tools/hok_hero_tree.py 解析
    英雄 × 时间节点 × 分支（前提 / 动作 / 后果，逐字）
        ↓ tools/hok_hero_actions.py 分类
    这条分支要求玩家做哪个**已注册的动作**
        ↓ 本模块合并 + 校验
    app/seed/hok_hero_modes.json

为什么要单独一层
----------------
分路层那 85 条是**分路通则**（「前期打野该刷野」）。
英雄级攻略给的是**每个英雄自己的决策树**（「赵云 4:00 该占龙坑视野，
惩击留到暴君最后一刻」）。两者不是同一件事，也不能互相替代：
玩家玩赵云时，通则那条规则不该再重复指控他一遍。

所以英雄层规则带 ``my_hero`` 触发条件，与分路层并存但互斥：
分路层那 85 条会加一条 ``my_hero in (全部英雄)`` 的护栏，
玩家选了自己的英雄就只走英雄层。

合并口径
--------
一个节点里的多条分支，前提都是**当场能看见的地图事实**
（对方有主宰 / 我方先到龙坑 / 对方位移刚交），
这些事实对局数据里判不了 —— 两条分支可能同时成立。
所以同一节点、同一动作族的分支**合并成一条规则**，
期望动作取并集：这一段里做到其中之一就算做到了。
这跟分路层「前提判不了的分支按可接受动作集合合并」是同一套口径。

合并的是**判定**，不是**建议**
--------------------------
同一动作族里的各条分支，打法并不相同：攻略写「敌方打野红开就蓝开反向开局」
和「敌方打野蓝开就红开反向开局」，判的时候可以合成一条
（做到其中之一就算做到），但「怎么改」不能合成一句 ——
把两个打法并排放，玩家读完不知道哪条适用，只会觉得这是一张备选清单。

所以 ``deviation_hint`` 保留**分支级**的条件与动作，写成
「什么情况下：做什么」：

    敌方打野红开（从预选推测）时：己方蓝开反向开局，蓝 BUFF → 蜥蜴 → 狼 → 红 BUFF → 小鸟。
    敌方打野蓝开时：己方红开反向开局，红 BUFF → 小鸟 → 猪 → 蓝 BUFF。

``action`` 字段仍然是合并后的原文动作（那是溯源用的），
``premise`` / ``outcome`` 同理。

判不了的分支
------------
动作全是沟通（「通知边路集结」）、留技能、买装备这类系统记不下来的事，
一律**不入库**，并在 ``skipped.json`` 里写明理由 ——
宁可少，也不能编一条永远判不出结果的规则。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from tools.build_hok_modes import (            # noqa: E402
    PHASE_LABELS,
    _mutually_exclusive,
    _windows_overlap,
    lane_windows,
    node_slug,
    phase_of_window,
    strip_ws,
)
from tools.hok_hero_actions import (          # noqa: E402
    CATEGORIES,
    classify_branch,
    subject_for,
)
from tools.hok_hero_codes import hero_code, verify_hero_codes  # noqa: E402
from tools.hok_hero_sources import hero_guide_paths, verify_sources  # noqa: E402
from tools.hok_hero_tree import load_hero_guide  # noqa: E402
from tools.hok_wording import (                  # noqa: E402
    apply_fixes,
    normalize,
    window_label,
)

OUT_PATH = os.path.join(ROOT, "app", "seed", "hok_hero_modes.json")
SKIP_PATH = os.path.join(ROOT, "app", "seed", "hok_hero_skipped.json")
HEROES_PATH = os.path.join(ROOT, "app", "seed", "hok_heroes.json")

#: 人工改写的英雄打法。攻略原文里没有这些子树（例如亚连在打野位的打法），
#: 但库里需要它们 —— 放在这里当**声明式源**，生成器每次都一起写进库。
#:
#: 为什么不直接手改 ``hok_hero_modes.json``：那是生成产物，下次重跑生成器
#: 就被覆盖了（这件事真发生过一次）。手写的内容必须落在一个生成器会读的地方，
#: 否则「重跑得到一模一样的库」这条保证就不成立。
REWRITE_PATH = os.path.join(
    ROOT, "app", "seed", "sources", "hok_hero_rewrites.json"
)

GAME = "hok"
VERSION = "S45"
GUIDE_NAME = "王者荣耀 S45「月照长安」英雄级决策树"
COLLECTED_AT = "2026-10-21"

#: 分路编码 -> 规则 id 里的简写（与分路层一致）。
LANE_SLUGS = {
    "jungle": "JGL",
    "mid": "MID",
    "side": "SID",
    "carry": "CAR",
    "roam": "ROA",
}

LANE_LABELS = {
    "jungle": "打野",
    "mid": "中路",
    "side": "对抗路",
    "carry": "发育路",
    "roam": "游走",
}

SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


class BuildError(RuntimeError):
    pass


def _join_unique(items: List[str]) -> str:
    """去重后用「；」连接，保持出现顺序。"""
    out: List[str] = []
    for item in items:
        text = (item or "").strip()
        if text and text not in out:
            out.append(text)
    return "；".join(out)


def _summary(text: str, limit: int = 34) -> str:
    """给标题用的一句话摘要。"""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _title_text(branch: Any) -> str:
    """标题里那句摘要的原文。

    取分支的第一条动作，去掉裸时刻（「1:00 与中单、游走控下中路河道之灵」
    → 「与中单、游走控下中路河道之灵」），也去掉模糊表述。
    标题前面已经写了这条规则管哪一段时间，正文再带一个时刻，
    读者会以为规则管错了时间。
    """
    raw = branch.actions[0] if getattr(branch, "actions", None) else ""
    return apply_fixes(normalize(raw, strip_clause_time=True))


def _rationale(premise: str, outcome: str) -> str:
    parts = []
    if premise:
        parts.append(f"原文前提：{premise}")
    if outcome:
        parts.append(f"原文结果：{outcome}")
    return "；".join(parts) + "。"


#: 攻略用「+」并列同一个前提里的多个条件，用「；」表示前提链的上下层。
#: 两种都拆开，再合成一句可读的条件短语。
#:
#: 加号**后面一定跟空格**（原文写「【反向开局完成】+ 已到 4 级」），
#: 前面不一定有空格 —— 所以只要求后置空格，避免把词里的「+」误当分隔符。
_COND_SPLIT_RE = re.compile(r"[；;]|\s*\+\s+")

#: 攻略里的状态标记（``【反向开局完成】``）是给下一个节点做引用输入的，
#: 写进「建议」里读者读不懂，也不该看到 —— 只留里面的文字。
_STATE_MARK_RE = re.compile(r"[【】]")


def _condition_text(premise: str) -> str:
    """把攻略原文的前提写成一句可读的条件短语。

    「【反向开局完成】+ 已到 4 级 + 敌方对抗路过线过深」
    → 「反向开局完成、已到 4 级、敌方对抗路过线过深」
    """
    text = _STATE_MARK_RE.sub("", premise or "")
    parts: List[str] = []
    for chunk in _COND_SPLIT_RE.split(text):
        chunk = chunk.strip()
        if chunk and chunk not in parts:
            parts.append(chunk)
    return "、".join(parts)


def _hint_from_branches(branches: List[Any]) -> str:
    """把同一动作族里的各条分支写成**条件式**的建议。

    「什么情况下：做什么」—— 一条一句，句号分隔。
    分支之间的动作不混在一起，玩家不用自己猜哪条适用。

    动作里的裸时刻（「1:00 与中单、游走控下中路河道之灵」）要去掉：
    标题已经写明这条规则管哪一段时间，正文再带一个时刻，
    读者会以为规则管错了时间。
    """
    clauses: List[str] = []
    for b in branches:
        acts = [a.strip() for a in b.actions if a and a.strip()]
        if not acts:
            continue
        cond = apply_fixes(normalize(_condition_text(b.premise_text)))
        action = apply_fixes(
            normalize("；".join(acts), strip_clause_time=True)
        )
        clauses.append(f"{cond}时：{action}" if cond else action)
    return "。".join(clauses)


def load_rewrites() -> List[Dict[str, Any]]:
    """人工改写的英雄打法（攻略原文里没有对应子树，见 ``REWRITE_PATH``）。"""
    with open(REWRITE_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return list(data.get("rules") or [])


def build_rules() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """生成英雄层规则。返回 ``(规则列表, 未入库分支清单)``。"""
    rules: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for lane, lane_label, path in hero_guide_paths():
        guide = load_hero_guide(path, lane)
        for hb in guide.heroes:
            code = hero_code(hb.hero)
            windows = lane_windows(hb.nodes)
            for node in hb.nodes:
                lo, hi = windows[node.time_label]
                phase = phase_of_window([lo, hi])

                # 按动作族把分支分组：前提判不了，所以同一族里取并集。
                groups: Dict[str, List[Tuple[Any, List[Any]]]] = {}
                for b in node.branches:
                    cats = classify_branch(b.premise_text, b.actions)
                    if not cats:
                        skipped.append({
                            "lane": lane,
                            "lane_label": lane_label,
                            "hero": hb.hero,
                            "node": node.header,
                            "premise": b.premise_text,
                            "action": b.action_text,
                            "reason": "这条分支的动作在系统的动作词表里找不到对应项"
                                      "（多是沟通、留技能、买装备这类系统记不下来的事），"
                                      "判不出「做到了没有」，不入库。",
                        })
                        continue
                    fact = cats[0].fact
                    groups.setdefault(fact, []).append((b, cats))

                for idx, (fact, items) in enumerate(groups.items(), start=1):
                    branches = [b for b, _ in items]
                    all_cats = [c for _, cats in items for c in cats]

                    values: List[str] = []
                    for c in all_cats:
                        if c.value not in values:
                            values.append(c.value)
                    severity = sorted(
                        (c.severity for c in all_cats),
                        key=lambda s: SEVERITY_RANK.get(s, 3),
                    )[0]

                    premise = _join_unique([b.premise_text for b in branches])
                    action = _join_unique(
                        [a for b in branches for a in b.actions]
                    )
                    outcome = _join_unique([b.outcome_text() for b in branches])

                    # 文案规范化：去状态记号（「【反向开局完成】」→ 「反向开局完成」）、
                    # 把「+」改成顿号、去掉动作开头的裸时刻、替换剩下的模糊表述。
                    # 引用（source.quoted）保持逐字不变 —— 那是给复核用的。
                    premise = apply_fixes(normalize(premise))
                    action = apply_fixes(normalize(action, strip_clause_time=True))
                    outcome = apply_fixes(normalize(outcome))

                    subject = subject_for(all_cats)

                    rule: Dict[str, Any] = {
                        "id": (
                            f"HOK-HERO-{LANE_SLUGS[lane]}-{code}-"
                            f"{node_slug(node.time_label)}-{idx:02d}"
                        ),
                        "game": GAME,
                        "version": VERSION,
                        # 标题里的时间按**规则实际覆盖的窗口**写：
                        # 窗口是范围就写范围，写成一个时点会让人以为只管那一分钟。
                        # 摘要取自攻略原文的动作，所以裸时刻要去掉 ——
                        # 「赵云 2:00 ~ 4:00：1:00 与中单…」看着像自相矛盾。
                        "title": (
                            f"{hb.hero} {window_label(lo, hi)}："
                            f"{_summary(_title_text(branches[0]))}"
                        ),
                        "category": f"{lane_label}·{hb.hero}·{PHASE_LABELS[phase]}",
                        "trigger": {
                            "all": [
                                {"fact": "position", "op": "==", "value": lane},
                                {"fact": "my_hero", "op": "==", "value": hb.hero},
                                {"fact": "time_min", "op": "between", "value": [lo, hi]},
                            ],
                            "any": [],
                            "none": [],
                        },
                        "expect": {
                            "fact": fact,
                            "op": "in",
                            "value": values,
                        },
                        "subject": {"fact": fact, "values": subject},
                        "expect_scope": "at_least_once",
                        # 期望动作是「做视野」的那些，现有识别链路产不出来 ——
                        # 硬判只会每局挂一条「判不了」，标成不可判
                        # （见 app/core/dsl.py 的 Rule.unjudgeable）。
                        **({"unjudgeable": True}
                           if "ward_vision" in values else {}),
                        "severity": severity,
                        "status": "active",
                        "premise": premise,
                        "action": action,
                        "outcome": outcome,
                        "rationale": _rationale(premise, outcome),
                        # 建议按分支写成条件式（见模块文档）：
                        # 合并的是判定，「怎么改」不能退化成一张备选清单。
                        "deviation_hint": _hint_from_branches(branches),
                        "tags": [lane_label, hb.hero, PHASE_LABELS[phase]]
                                + sorted({c.label for c in all_cats if c.label}),
                        "source": {
                            "guide": GUIDE_NAME,
                            "locator": (
                                f"{lane_label} / {hb.hero} / {node.header} / "
                                f"{branches[0].premise}"
                            ),
                            "url": "",
                            "quoted": branches[0].quote(),
                            "collected_at": COLLECTED_AT,
                        },
                    }
                    rules.append(rule)
    # 生成完再把手写的那批并进来：``build_rules()`` 是**唯一**的产出口，
    # 种子库与它逐条相等（tests/test_hero_mode_library.py 盯着这条）。
    rules.extend(load_rewrites())
    return rules, skipped


# --------------------------------------------------------------------------
# 校验
# --------------------------------------------------------------------------


def _window_of(rule: Dict[str, Any]) -> List[int]:
    for c in rule["trigger"]["all"]:
        if c["fact"] == "time_min" and c["op"] == "between":
            return list(c["value"])
    return [0, 60]


def _hero_of(rule: Dict[str, Any]) -> str:
    for c in rule["trigger"]["all"]:
        if c["fact"] == "my_hero":
            return str(c["value"])
    return ""


def validate(rules: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    from app.adapters import registry as _registry  # noqa: F401
    from app.adapters.base import get_adapter
    from app.core.dsl import DEFAULT_SUBJECTIVE_MARKERS, Rule, check_admission

    adapter = get_adapter(GAME)
    known_facts = adapter.known_facts()
    action_facts = adapter.known_action_facts()
    action_labels = dict(adapter.action_labels)

    # 1. id 唯一
    seen = set()
    for r in rules:
        if r["id"] in seen:
            errors.append(f"规则 id 重复：{r['id']}")
        seen.add(r["id"])
        if not r["id"].startswith("HOK-HERO-"):
            errors.append(f"规则 id 前缀不对：{r['id']}")

    # 2. 期望动作必须可勾，subject 必须是已注册事实
    for r in rules:
        fact = r["expect"]["fact"]
        if fact not in action_facts:
            errors.append(f"{r['id']} 的期望事实不是动作事实：{fact}")
        for v in r["expect"]["value"]:
            if v not in action_labels:
                errors.append(f"{r['id']} 用了未注册的动作：{v}")
        subject = r.get("subject")
        if subject:
            if subject["fact"] not in known_facts:
                errors.append(
                    f"{r['id']} 的适用时机引用了未注册事实：{subject['fact']}"
                )
            missing = set(r["expect"]["value"]) - set(subject["values"])
            if missing:
                errors.append(
                    f"{r['id']} 的适用时机比期望动作窄：{sorted(missing)} 不在里面，"
                    "这样玩家做到位了也不会被判为符合"
                )

    # 3. 引用必须逐字来自原文
    flat = {lane: strip_ws(text) for lane, _, text in
            ((l, lb, open(p, encoding="utf-8").read())
             for l, lb, p in hero_guide_paths())}

    # 4. 同一格（英雄 × 阶段）里不能互相指控
    by_cell: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for r in rules:
        window = _window_of(r)
        phase = phase_of_window(window)
        lane = r["trigger"]["all"][0]["value"]
        by_cell.setdefault((lane, _hero_of(r), phase), []).append(r)

    for (lane, hero, phase), group in by_cell.items():
        by_fact: Dict[str, List[Dict[str, Any]]] = {}
        for r in group:
            by_fact.setdefault(r["expect"]["fact"], []).append(r)
        for fact, same in by_fact.items():
            for i in range(len(same)):
                for j in range(i + 1, len(same)):
                    a, b = same[i], same[j]
                    va = set(a["expect"]["value"])
                    vb = set(b["expect"]["value"])
                    if va & vb:
                        continue
                    if not _windows_overlap(a, b):
                        continue
                    if _mutually_exclusive(a, b):
                        continue
                    errors.append(
                        f"{hero}·{PHASE_LABELS[phase]} 里 {a['id']} 与 {b['id']} "
                        f"要求同一动作事实 {fact} 但取值没有交集，"
                        "玩家会被两条同时指控"
                    )

    # 5. 准入校验
    for r in rules:
        rule = Rule.from_dict(r)
        for reason in check_admission(rule, known_facts, DEFAULT_SUBJECTIVE_MARKERS,
                                     action_facts):
            errors.append(f"{r['id']} 未通过准入：{reason}")

    return errors


def check_quotes(rules: List[Dict[str, Any]]) -> List[str]:
    """引用必须能在**对应那份**原文里逐字找到。

    人工改写的规则没有攻略原文可对，改对 ``REWRITE_PATH`` ——
    要求引用与改写源里的那条一致，防止两边各改一半。
    """
    problems: List[str] = []
    texts: Dict[str, str] = {}
    for lane, _, path in hero_guide_paths():
        with open(path, "r", encoding="utf-8") as fh:
            texts[lane] = strip_ws(fh.read())
    rewrite_ids = {r["id"] for r in load_rewrites()}
    for r in rules:
        lane = r["trigger"]["all"][0]["value"]
        quoted = r["source"]["quoted"]
        if not quoted.strip():
            problems.append(f"{r['id']} 缺引用")
        elif r["id"] in rewrite_ids:
            # 人工改写的规则没有攻略原文可对，对照改写源即可。
            continue
        elif strip_ws(quoted) not in texts.get(lane, ""):
            problems.append(f"{r['id']} 的引用在原文里找不到")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验，不写文件")
    args = ap.parse_args()

    issues = verify_sources()
    issues += verify_hero_codes()
    if issues:
        print("✗ 源文件自检失败：")
        for i in issues:
            print("  -", i)
        return 1

    rules, skipped = build_rules()
    errors = validate(rules)
    errors += check_quotes(rules)

    if errors:
        print(f"✗ {len(errors)} 处问题：")
        for e in errors:
            print("  -", e)
        return 1

    if not args.check:
        with open(OUT_PATH, "w", encoding="utf-8") as fh:
            json.dump(rules, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        with open(SKIP_PATH, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "guide": GUIDE_NAME,
                    "collected_at": COLLECTED_AT,
                    "note": (
                        "这些分支在原文里写了，但动作在系统的动作词表里找不到对应项"
                        "（多是沟通、留技能、买装备这类系统记不下来的事），"
                        "判不出「做到了没有」，所以不入库。"
                    ),
                    "skipped": skipped,
                },
                fh, ensure_ascii=False, indent=2,
            )
            fh.write("\n")
        # 英雄名单单独存一份：标注页的英雄下拉要按分路列出全部英雄，
        # 而适配器不该在 import 时去解析 5 份攻略原文。
        roster = {}
        for lane, lane_label, path in hero_guide_paths():
            guide = load_hero_guide(path, lane)
            names: List[str] = []
            for hb in guide.heroes:
                if hb.hero not in names:
                    names.append(hb.hero)
            roster[lane] = {"label": lane_label, "heroes": names}
        with open(HEROES_PATH, "w", encoding="utf-8") as fh:
            json.dump(
                {"version": VERSION, "collected_at": COLLECTED_AT, "lanes": roster},
                fh, ensure_ascii=False, indent=2,
            )
            fh.write("\n")
        total = sum(len(v["heroes"]) for v in roster.values())
        print(f"✓ 写入 {OUT_PATH}（{len(rules)} 条），"
              f"未入库分支 {len(skipped)} 条 -> {SKIP_PATH}")
        print(f"✓ 英雄名单 {total} 个（去重 {len(set(h for v in roster.values() for h in v['heroes']))}）"
              f" -> {HEROES_PATH}")
    else:
        print(f"✓ 校验通过（{len(rules)} 条，未入库分支 {len(skipped)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
