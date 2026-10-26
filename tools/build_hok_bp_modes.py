"""从 BP 决策树原文生成 BP 阶段的操作模式库。

用法::

    python tools/build_hok_bp_modes.py            # 生成
    python tools/build_hok_bp_modes.py --check    # 只校验，不写文件

生成链路
--------
    app/seed/sources/hok_s45_bp/*.txt        BP 决策树原文（唯一真源）
        ↓ tools/hok_bp_tree.py 解析
    决策环节 × 分支（前提 / 动作 / 后果，逐字）
        ↓ tools/hok_bp_spec.py 映射
    这条分支要求玩家做哪一类 BP 操作、依赖哪些局面事实
        ↓ 本模块合并 + 校验
    app/seed/hok_bp_modes.json

为什么 BP 单独一份库
--------------------
BP 在**比赛之前**：没有局内分钟、没有经济差，判的是「这一手该禁谁、该选谁」。
它与局内库（分路 85 + 英雄 1596）的时间坐标、事实、动作全不一样，
混在一起会让两套判定的口径互相污染。

所以 BP 库用**单独的版本号** ``S45-BP``：同一个游戏下并存两个版本，
局内报告取 ``S45``，BP 复盘取 ``S45-BP``，互不干扰。

未入库的分支
------------
需要「版本 Ban 率」「选手熟练度」「对手意图」这类事实的分支，
系统拿不到事实 —— 硬编一张表就是替玩家下判断。
这些分支进 ``hok_bp_skipped.json``，逐条写明**缺的是哪个事实**。
规矩跟英雄层一样：**每条分支要么变成规则，要么出现在未入库清单里**。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from tools.hok_bp_sources import (           # noqa: E402
    BP_GUIDE_NAME,
    bp_guide_path,
    verify_sources,
)
from tools.hok_bp_spec import (              # noqa: E402
    BP_ACTIONS,
    BP_FACTS,
    PHASE_LABELS,
    RULES,
    SKIPPED,
)
from tools.hok_bp_tree import load_bp_guide   # noqa: E402
from tools.hok_wording import apply_fixes, normalize  # noqa: E402

OUT_PATH = os.path.join(ROOT, "app", "seed", "hok_bp_modes.json")
SKIP_PATH = os.path.join(ROOT, "app", "seed", "hok_bp_skipped.json")

GAME = "hok"
#: BP 库单独一个版本号 —— 局内库是 S45，两者并存互不干扰。
#:
#: 用 ``BP-`` 开头还有第二个原因：它按字符串排序时排在 ``S45`` 前面，
#: 不会顶掉局内库的「最新版本」。
VERSION = "BP-S45"
COLLECTED_AT = "2026-10-22"

#: 环节编码 -> 规则 id 里的简写。
#:
#: 「进入 BP」那一环已经撤下（记录页不再有这一环），
#: 但它的编码留着 —— 历史 id 里出现过 ``HOK-BP-START-*``，
#: 编码不漂，退役清单里的 id 才认得出来。
PHASE_SLUGS: Dict[str, str] = {
    "bp_start": "START",
    "ban_1": "BAN1",
    "pick_1": "PICK1",
    "pick_2": "PICK2",
    "pick_3": "PICK3",
    "pick_late_1": "PICKL1",
    "pick_last": "PICKLAST",
    "final_check": "FINAL",
}


def _rationale(premise: str, outcome: str) -> str:
    parts = []
    if premise:
        parts.append(f"原文前提：{premise}")
    if outcome:
        parts.append(f"原文结果：{outcome}")
    return "；".join(parts) + "。"


def build_rules() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """生成 BP 规则。返回 ``(规则列表, 未入库分支清单)``。"""
    guide = load_bp_guide(bp_guide_path())
    rules: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    # 映射表按 (环节, 序号) 索引；同一键可以有多条规则。
    spec_by_key: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for spec in RULES:
        spec_by_key.setdefault(tuple(spec["key"]), []).append(spec)

    used_keys: set = set()
    for node in guide.nodes:
        for branch in node.branches:
            key = (node.phase, branch.index)
            specs = spec_by_key.get(key)
            if not specs:
                skipped.append({
                    "phase": node.phase,
                    "phase_label": node.phase_label,
                    "index": branch.index,
                    "premise": branch.premise_text,
                    "action": "；".join(branch.actions),
                    "reason": SKIPPED.get(
                        key,
                        "这条分支既没有映射表条目，也没写明不入库的理由 —— "
                        "映射表漏了，补上。",
                    ),
                })
                continue
            used_keys.add(key)

            premise = branch.premise_text
            action = "；".join(branch.actions)
            outcome = branch.outcome_text() if hasattr(branch, "outcome_text") else branch.outcome
            # 建议按原文逐字给；一句话一行，读的时候才对得上。
            hint = "。".join(a for a in branch.actions if a.strip())
            if hint and not hint.endswith(("。", "！", "？")):
                hint += "。"

            # 文案规范化：去攻略内部的状态记号（「状态【BP 目标已定】」→
            # 「BP 目标已定」）、把「+」改成顿号、替换模糊表述。
            # 引用（source.quoted）保持逐字不变 —— 那是给复核用的。
            premise = apply_fixes(normalize(premise))
            action = apply_fixes(normalize(action))
            outcome = apply_fixes(normalize(outcome))
            hint = apply_fixes(normalize(hint))

            for spec in specs:
                variant = str(spec.get("variant") or "")
                rule_id = (
                    f"HOK-BP-{PHASE_SLUGS[node.phase]}-{branch.index:02d}"
                    + (f"-{variant}" if variant else "")
                )
                rules.append({
                    "id": rule_id,
                    "game": GAME,
                    "version": VERSION,
                    "title": apply_fixes(normalize(spec["title"])),
                    "category": f"BP·{node.phase_label}",
                    "trigger": {
                        "all": [
                            {"fact": f, "op": op, "value": v}
                            for f, op, v in spec["trigger"]
                        ],
                        "any": [],
                        "none": [],
                    },
                    "expect": {
                        "fact": "bp_action",
                        "op": "in",
                        "value": list(spec["expect"]),
                    },
                    # 适用时机：这一手必须是 BP 动作，否则这条规则不适用。
                    "subject": {
                        "fact": "bp_action",
                        "values": sorted(BP_ACTIONS),
                    },
                    "expect_scope": spec.get("scope", "per_frame"),
                    "severity": spec.get("severity", "medium"),
                    "status": "active",
                    "premise": premise,
                    "action": action,
                    "outcome": outcome,
                    "rationale": _rationale(premise, outcome),
                    "deviation_hint": hint,
                    "tags": ["BP", node.phase_label]
                            + (["说明"] if spec.get("note") else []),
                    "source": {
                        "guide": BP_GUIDE_NAME,
                        "locator": f"BP / {node.phase_label} / {premise}",
                        "url": "",
                        "quoted": branch.quote(),
                        "collected_at": COLLECTED_AT,
                    },
                    # 口径说明：这条规则用哪个事实、怎么算的。规则库页看得见。
                    "note": spec.get("note", ""),
                })

    # 映射表里写了、原文里却没有的键 —— 攻略改版后会出这种事，必须报出来。
    stray = sorted(set(spec_by_key) - used_keys)
    for phase, index in stray:
        skipped.append({
            "phase": phase,
            "phase_label": PHASE_LABELS.get(phase, phase),
            "index": index,
            "premise": "",
            "action": "",
            "reason": "映射表里有这一条，但原文里找不到对应的分支 —— "
                      "攻略改版了，映射表要跟着改。",
        })

    return rules, skipped


# --------------------------------------------------------------------------
# 校验
# --------------------------------------------------------------------------


def validate(rules: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    from app.adapters import registry as _registry  # noqa: F401
    from app.adapters.base import get_adapter
    from app.core.dsl import DEFAULT_SUBJECTIVE_MARKERS, Rule, check_admission

    adapter = get_adapter(GAME)
    known_facts = adapter.known_facts()
    action_facts = adapter.known_action_facts()

    seen = set()
    for r in rules:
        if r["id"] in seen:
            errors.append(f"规则 id 重复：{r['id']}")
        seen.add(r["id"])
        if not r["id"].startswith("HOK-BP-"):
            errors.append(f"规则 id 前缀不对：{r['id']}")
        if r["version"] != VERSION:
            errors.append(f"{r['id']} 的版本不对：{r['version']}")

        for cond in r["trigger"]["all"]:
            if cond["fact"] not in known_facts:
                errors.append(
                    f"{r['id']} 引用了未注册的事实 {cond['fact']!r}"
                )
        for value in r["expect"]["value"]:
            if value not in BP_ACTIONS:
                errors.append(f"{r['id']} 用了未注册的 BP 动作 {value!r}")

        rule = Rule.from_dict(r)
        for reason in check_admission(rule, known_facts, DEFAULT_SUBJECTIVE_MARKERS,
                                     action_facts):
            errors.append(f"{r['id']} 未通过准入：{reason}")

    return errors


def check_quotes(rules: List[Dict[str, Any]]) -> List[str]:
    """引用必须能在原文里逐字找到。"""
    from tools.build_hok_modes import strip_ws

    problems: List[str] = []
    text = strip_ws(open(bp_guide_path(), encoding="utf-8").read())
    for r in rules:
        quoted = r["source"]["quoted"]
        if not quoted.strip():
            problems.append(f"{r['id']} 缺引用")
        elif strip_ws(quoted) not in text:
            problems.append(f"{r['id']} 的引用在原文里找不到")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验，不写文件")
    args = ap.parse_args()

    issues = verify_sources()
    if issues:
        print("✗ BP 原文自检失败：")
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
                    "guide": BP_GUIDE_NAME,
                    "collected_at": COLLECTED_AT,
                    "note": (
                        "这些分支在原文里写了，但判定要用的事实系统拿不到"
                        "（版本 Ban 率、选手熟练度、对手意图、英雄定位的定性描述等），"
                        "判不出「做到了没有」，所以不入库。理由逐条写明缺的是哪个事实。"
                    ),
                    "skipped": skipped,
                },
                fh, ensure_ascii=False, indent=2,
            )
            fh.write("\n")
        print(f"✓ 写入 {OUT_PATH}（{len(rules)} 条），"
              f"未入库分支 {len(skipped)} 条 -> {SKIP_PATH}")
    else:
        print(f"✓ 校验通过（{len(rules)} 条，未入库分支 {len(skipped)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
