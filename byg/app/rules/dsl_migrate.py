"""把 v1 规则库迁到 v2 DSL（模式 = 时点 + 一组可判定项）。

为什么需要这一步
----------------
v1 的规则把「什么时候」写在触发条件里、把「该做什么」压成一个值、
把「能不能纠正」定成入库时的布尔。这三件事让 MOBA 攻略里最具体的话
（「刷野 ≥4 次」「先清精灵再推线」「红开就蓝开」）全都表达不出来。

本模块做的是**结构迁移**，不重新发明攻略内容：

* 触发条件里的 ``time_min between [a, b]`` 提成 :class:`~app.core.dsl2.Node`；
* ``my_hero in [全部英雄]`` 从「前提」移到「适用范围」（它本来就不是
  筛选条件，而是「必须知道你在玩谁」）；
* 期望值变成一项 :class:`~app.core.dsl2.Check`，带计数与时间窗；
* ``node_actions`` 变成 ``accept``（本节点认可的全部打法）；
* ``corrective`` 布尔变成**运行时可解**的 :class:`~app.core.dsl2.Corrective`
  —— 分支前提被指认的那一局，规则自动变成可纠正的。

迁移是**可复现**的：同样的输入文件永远产出同样的输出文件
（除时间戳外），并记下输入文件的 sha256，改了输入能看出来。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.dsl2 import (
    CORRECTIVE_ALWAYS,
    CORRECTIVE_NEVER,
    CORRECTIVE_ON_BRANCH,
    SCOPE_AT_LEAST_ONCE,
    SCOPE_PER_FRAME,
    AppliesTo,
    Check,
    Corrective,
    Node,
    Pattern,
    Subject,
)

#: 分类名（``打野·前期``）-> (分路, 阶段)。
CATEGORY_PARTS: Dict[str, Tuple[str, str]] = {
    "打野": ("jungle", "early"),
    "中路": ("mid", "early"),
    "对抗路": ("side", "early"),
    "发育路": ("carry", "early"),
    "游走": ("roam", "early"),
}

PHASE_PART: Dict[str, str] = {
    "前期": "early",
    "中期": "mid",
    "后期": "late",
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def split_category(category: str) -> Tuple[str, str]:
    """``打野·中期`` -> ``("jungle", "mid")``。分不开就留空。"""
    if not category:
        return "", ""
    parts = category.split("·")
    lane_part = parts[0].strip()
    phase_part = parts[-1].strip() if len(parts) > 1 else ""
    lane = CATEGORY_PARTS.get(lane_part, ("", ""))[0]
    phase = PHASE_PART.get(phase_part, "")
    return lane, phase


def node_label(from_min: float, to_min: float) -> str:
    if float(from_min) == float(to_min):
        return f"第 {from_min:g} 分钟"
    return f"{from_min:g}–{to_min:g} 分钟"


def _as_check(raw: Dict[str, Any]) -> Check:
    return Check(
        fact=str(raw.get("fact") or ""),
        op=str(raw.get("op") or "=="),
        value=raw.get("value"),
    )


def migrate_rule(
    raw: Dict[str, Any],
    *,
    all_heroes: Optional[set] = None,
) -> Tuple[Pattern, List[str]]:
    """把一条 v1 规则迁成 v2 模式。返回 (模式, 迁移说明)。"""
    notes: List[str] = []

    trigger = raw.get("trigger") or {}
    when: List[Check] = []
    when_any: List[Check] = []
    when_none: List[Check] = []
    node_from: Optional[float] = None
    node_to: Optional[float] = None
    requires: List[str] = []
    heroes: List[str] = []

    for group, bucket in (("all", when), ("any", when_any), ("none", when_none)):
        for cond in (trigger.get(group) or []):
            fact = str(cond.get("fact") or "")
            op = str(cond.get("op") or "==")
            value = cond.get("value")
            if fact == "time_min" and op == "between" and isinstance(value, (list, tuple)):
                # 时点 → Node
                node_from = float(value[0])
                node_to = float(value[1])
                notes.append(
                    f"触发条件里的 time_min between {list(value)} 提成节点"
                )
                continue
            if (group == "all" and fact == "my_hero" and op == "in"
                    and isinstance(value, (list, tuple))):
                # 英雄白名单 → 适用范围（**只在 all 组**）。
                #
                # 这里有个坑：84 条分路规则的 ``my_hero in (全部英雄)``
                # 并不在 ``all`` 组，而在 ``none`` 组 —— 那是**让路护栏**
                # （玩家填了英雄就走英雄层，同一句建议不说两遍）。
                # 把它当「必须知道你在玩谁」搬过去，
                # 等于把护栏删了：填了英雄，分路通则照样判。
                heroes = [str(h) for h in value]
                requires.append("my_hero")
                if all_heroes is not None and set(heroes) >= all_heroes:
                    notes.append(
                        "my_hero 白名单覆盖全部英雄 → 降级为「必须知道你在玩谁」"
                    )
                    heroes = []
                continue
            if (group == "all" and fact == "my_hero" and op == "=="
                    and isinstance(value, str) and value.strip()):
                # 英雄专属模式（英雄层 1596 条）→ 适用范围白名单。
                #
                # 为什么不留在 ``when`` 里当筛选条件
                # -----------------------------------
                # 「你在玩赵云」不是**这一局成立的条件**，而是**这条模式
                # 是给谁看的**。两种写法的判定结果一样，但只有前者能回答
                # 「这条建议是哪个英雄的」，也只有前者能在诊断里点名
                # 「我方英雄没认出来，英雄层一条都没生效」。
                heroes = [value.strip()]
                requires.append("my_hero")
                notes.append(f"英雄专属模式 → 适用范围：{value.strip()}")
                continue
            bucket.append(Check(fact=fact, op=op, value=value))

    if node_from is None:
        # 没有时间窗 → 整局（0 到 999 分钟）。这类规则在 v1 里是「每帧都判」。
        node_from, node_to = 0.0, 999.0
        notes.append("触发条件里没有时间窗 → 节点覆盖整局")

    lane, phase = split_category(str(raw.get("category") or ""))

    exp = raw.get("expect") or {}
    expect = [
        Check(
            fact=str(exp.get("fact") or ""),
            op=str(exp.get("op") or "=="),
            value=exp.get("value"),
            min_count=1,
        )
    ]

    scope = str(raw.get("expect_scope") or SCOPE_AT_LEAST_ONCE)
    if scope not in (SCOPE_AT_LEAST_ONCE, SCOPE_PER_FRAME):
        scope = SCOPE_AT_LEAST_ONCE

    branches = [dict(b) for b in (raw.get("branches") or []) if isinstance(b, dict)]
    corrective_raw = bool(raw.get("corrective", True))
    if corrective_raw:
        mode = CORRECTIVE_ALWAYS
        notes.append("可纠正性：恒可纠正")
    elif branches:
        mode = CORRECTIVE_ON_BRANCH
        notes.append(
            f"可纠正性：改为运行时判定（{len(branches)} 支前提，"
            "指认了哪支才纠正）"
        )
    else:
        mode = CORRECTIVE_NEVER
        notes.append("可纠正性：不用来纠正")

    subject_raw = raw.get("subject") or {}
    subject = None
    if subject_raw.get("fact"):
        subject = Subject(
            fact=str(subject_raw.get("fact") or ""),
            values=tuple(str(v) for v in (subject_raw.get("values") or [])),
        )
        notes.append(
            f"适用时机（subject）：{subject.fact}"
            + (f" / {list(subject.values)}" if subject.values else "")
        )

    pattern = Pattern(
        id=str(raw.get("id") or ""),
        game=str(raw.get("game") or "hok"),
        version=str(raw.get("version") or ""),
        title=str(raw.get("title") or ""),
        category=str(raw.get("category") or ""),
        lane=lane,
        phase=phase,
        node=Node(
            label=node_label(node_from, node_to),
            from_min=node_from,
            to_min=node_to,
        ),
        applies_to=AppliesTo(
            lane=lane,
            heroes=tuple(heroes),
            requires=tuple(requires),
        ),
        subject=subject,
        when=when,
        when_any=when_any,
        when_none=when_none,
        expect=expect,
        accept=[str(a) for a in (raw.get("node_actions") or [])],
        scope=scope,
        severity=str(raw.get("severity") or "medium"),
        status=str(raw.get("status") or "active"),
        corrective=Corrective(mode=mode, branches=tuple(branches)),
        source=dict(raw["source"]) if raw.get("source") else None,
        rationale=str(raw.get("rationale") or ""),
        deviation_hint=str(raw.get("deviation_hint") or ""),
        premise=str(raw.get("premise") or ""),
        action=str(raw.get("action") or ""),
        outcome=str(raw.get("outcome") or ""),
        tags=[str(t) for t in (raw.get("tags") or [])],
    )
    return pattern, notes


def migrate_library(
    rules: List[Dict[str, Any]],
) -> Tuple[List[Pattern], List[Dict[str, Any]]]:
    """迁移整库。返回 (模式列表, 迁移报告条目)。"""
    # 全部英雄 = 所有规则的 my_hero 白名单的并集。
    # 用并集而不是外部英雄表：少一个外部依赖，也不会因为表变了
    # 导致「白名单覆盖全部」的判断漂移。
    all_heroes: set = set()
    for raw in rules:
        for cond in ((raw.get("trigger") or {}).get("all") or []):
            if cond.get("fact") == "my_hero" and cond.get("op") == "in":
                value = cond.get("value")
                if isinstance(value, (list, tuple)):
                    all_heroes |= {str(h) for h in value}

    patterns: List[Pattern] = []
    report: List[Dict[str, Any]] = []
    for raw in rules:
        pat, notes = migrate_rule(raw, all_heroes=all_heroes)
        patterns.append(pat)
        report.append({"id": pat.id, "title": pat.title, "notes": notes})
    return patterns, report


def build_document(
    patterns: List[Pattern],
    *,
    source_path: Path,
    source_sha: str,
    schema: str = "hok-patterns-v2",
) -> Dict[str, Any]:
    versions = sorted({p.version for p in patterns if p.version})
    games = sorted({p.game for p in patterns if p.game})
    return {
        "schema": schema,
        "game": games[0] if len(games) == 1 else "",
        "version": versions[0] if len(versions) == 1 else "",
        "generated_from": source_path.name,
        "source_sha256": source_sha,
        "count": len(patterns),
        "patterns": [p.to_dict() for p in patterns],
    }


def write_report(report: List[Dict[str, Any]], out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# 规则库 v1 → v2 迁移报告\n")
    lines.append(f"共迁移 {len(report)} 条。\n")
    lines.append("## 逐条说明\n")
    for item in report:
        lines.append(f"### {item['id']} {item['title']}\n")
        for note in item["notes"]:
            lines.append(f"- {note}")
        if not item["notes"]:
            lines.append("- （无结构变化）")
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="把 v1 规则库迁到 v2 DSL")
    ap.add_argument("--in", dest="inp", default=None, help="输入的 v1 规则库")
    ap.add_argument("--out", dest="out", default=None, help="输出的 v2 模式库")
    ap.add_argument("--report", dest="report", default=None, help="迁移报告（Markdown）")
    ap.add_argument(
        "--kind", choices=("lane", "hero"), default="lane",
        help="迁哪一层：lane = 分路层 84 条，hero = 英雄层 1596 条",
    )
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    if args.kind == "hero":
        src = Path(args.inp) if args.inp else root / "app" / "seed" / "hok_hero_modes.json"
        dst = Path(args.out) if args.out else root / "app" / "seed" / "hok_hero_patterns.json"
        schema = "hok-hero-patterns-v2"
    else:
        src = Path(args.inp) if args.inp else root / "app" / "seed" / "hok_modes.json"
        dst = Path(args.out) if args.out else root / "app" / "seed" / "hok_patterns.json"
        schema = "hok-patterns-v2"
    rep = (
        Path(args.report)
        if args.report
        else root / ".tmp" / (
            "migration_report.md" if args.kind == "lane"
            else "migration_report_hero.md"
        )
    )

    rules = json.loads(src.read_text(encoding="utf-8"))
    if isinstance(rules, dict):
        rules = rules.get("modes") or rules.get("rules") or []
    patterns, report = migrate_library(rules)

    doc = build_document(
        patterns,
        source_path=src,
        source_sha=sha256_of(src),
        schema=schema,
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(report, rep)

    print(f"migrated {len(patterns)} patterns -> {dst}")
    print(f"report -> {rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
