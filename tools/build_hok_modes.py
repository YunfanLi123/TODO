"""从决策树攻略生成王者荣耀的操作模式库。

用法::

    python tools/build_hok_modes.py            # 生成 app/seed/hok_modes.json
    python tools/build_hok_modes.py --check    # 只校验，不写文件

生成链路
--------
    app/seed/sources/hok_s45_lane_decision_tree.txt   攻略原文（唯一真源）
        ↓ tools/hok_decision_tree.py 解析
    节点 × 分支（前提 / 动作 / 后果，逐字）
        ↓ tools/hok_spec/*.py 映射表（这条分支要求玩家做什么）
    app/seed/hok_modes.json                          规则库

映射表只回答**语义**问题：这条分支对应哪一个动作事实、哪些取值、
在什么时刻才适用（``subject``）、按什么口径算做到了（``expect_scope``）。
引用、标题里的原文片段、时间窗、阶段归属都由解析器和生成器决定 ——
所以「引用抄错」「阶段标错」「窗口和阶段对不上」这三类错误不可能发生。

校验
----
生成前会跑一整套检查，任何一条不过就**不写文件**：

- 攻略里的每条分支都必须有归宿（映射成规则，或显式标注「不可判定，不入库」）
- 期望动作必须在王者动作词表里，否则标注页勾不出来
- 同一「分路 × 阶段」里两条规则若要求同一动作事实却取值不重叠 → 玩家会被两条同时指控
- 引用必须能在原文里逐字找到
- 必须通过系统的可判定性准入（``check_admission``）
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

#: Windows 控制台默认 GBK，打印 "✓" 会抛 UnicodeEncodeError。
#: 后果很阴险：校验其实已经跑完并通过了，**只有最后那行打印**炸掉，
#: 退出码变成 1 —— 于是「校验通过」被误报成「校验失败」，
#: 而真相藏在一行栈回溯里。这里统一把输出换成 UTF-8 并对不可编码字符降级。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from tools.hok_decision_tree import (  # noqa: E402
    DecisionTree,
    load_guide,
    normalize_for_compare,
)
from tools.hok_wording import (  # noqa: E402
    apply_fixes,
    normalize,
    retime_title,
)

GUIDE_PATH = os.path.join(
    ROOT, "app", "seed", "sources", "hok_s45_lane_decision_tree.txt"
)
OUT_PATH = os.path.join(ROOT, "app", "seed", "hok_modes.json")

GAME = "hok"
VERSION = "S45"
GUIDE_NAME = "王者荣耀 S45「月照长安」分路决策树"
COLLECTED_AT = "2026-10-20"

#: **画面里读得出**的事实（不依赖任何数值）。
#:
#: 规则拿这些当触发条件，说明它的前提能被画面证实 ——
#: 于是这条规则可以拿来纠正玩家（前提成立与否，判得出来）。
#:
#: - ``has_ult_ready``：大招亮不亮（HUD 面板上就有）
#: - ``in_combat``：此刻是否在团战
#:
#: 注意 ``matchup``（对位优劣）**不在这里**：它是「换血之后才知道」的事，
#: 画面里读不出来 —— 拿它当纠正依据就是冤枉人。
_READABLE_FACTS = ("has_ult_ready", "in_combat")

#: 映射表模块。顺序只影响输出顺序，不影响内容。
SPEC_MODULES = ["jungle", "mid", "side", "carry", "roam"]

#: 分路编码 -> 规则 id 里的简写。
LANE_SLUGS = {
    "jungle": "JGL",
    "mid": "MID",
    "side": "SID",
    "carry": "CAR",
    "roam": "ROA",
}

#: 分路编码 -> 中文名（与适配器一致，但这里独立写一份：
#: 生成器要在**不导入适配器**的情况下也能跑，适配器的导入会拖起整套依赖）。
LANE_LABELS = {
    "jungle": "打野",
    "mid": "中路",
    "side": "对抗路",
    "carry": "发育路",
    "roam": "游走",
}

#: 阶段窗口。与适配器 ``PHASE_TIME_WINDOWS`` 必须一致 ——
#: 规则的时间窗要落在阶段窗口里，否则规则会在「阶段标签说前期、
#: 实际落在中期」的回合上触发。
PHASE_WINDOWS = {
    "early": [0, 4],
    "mid": [5, 10],
    "late": [11, 60],
}
PHASE_LABELS = {"early": "前期", "mid": "中期", "late": "后期"}


def _hero_roster() -> List[str]:
    """英雄级攻略里出现的全部英雄名。

    分路通则要给「玩家选了自己的英雄」让路，护栏里就得列出全部英雄。
    清单从英雄级原文里读 —— 那 5 份攻略是唯一真源，英雄名单不另维护一份。
    """
    from tools.hok_hero_sources import all_hero_names

    return all_hero_names()


#: 全部英雄名。生成时读一次，读不到就退化成空列表（护栏不生效，
#: 但不会让生成器崩掉 —— 崩在生成阶段的话，库里会留 half 半套规则）。
try:
    HERO_ROSTER: List[str] = _hero_roster()
except Exception:  # noqa: BLE001
    HERO_ROSTER = []


def phase_of_window(window: List[int]) -> str:
    lo = window[0]
    if lo < 5:
        return "early"
    if lo < 11:
        return "mid"
    return "late"


def node_slug(time_label: str) -> str:
    """节点时间 -> 规则 id 里的片段。``0:30-1:00`` -> ``T0030``。"""
    head = time_label.split("-")[0].strip()
    mm, _, ss = head.partition(":")
    return f"T{int(mm):02d}{int(ss):02d}"


def node_start_minute(time_label: str, header: str) -> int:
    """节点起始分钟。

    ``0:30-1:00`` 这种带区间的节点取**区间下界向上取整到分钟**（0:30 -> 1 分）——
    局内时间只精确到分钟，若取 0 分，它会和上一个节点（0:00）挤在同一分钟上，
    两条规则同时判到那一帧。
    ``12:00 之后`` 取 12。
    """
    head = time_label.split("-")[0].strip()
    mm, _, ss = head.partition(":")
    total = int(mm) * 60 + int(ss or 0)
    return (total + 59) // 60


def lane_windows(nodes) -> Dict[str, List[int]]:
    """给一条分路的每个节点算出时间窗。

    规则：节点 ``i`` 的窗 = [自己起始分钟, 下一个节点起始分钟 - 1]，
    再夹到所属阶段窗口内。这样同一分路里两个节点的窗**不重叠**，
    玩家在第 3 分钟做的动作不会被两个节点同时指控。
    """
    starts = [node_start_minute(n.time_label, n.header) for n in nodes]
    windows: Dict[str, List[int]] = {}
    for i, n in enumerate(nodes):
        lo = starts[i]
        hi = starts[i + 1] - 1 if i + 1 < len(starts) else 60
        # 夹进阶段窗口
        for phase, (p_lo, p_hi) in PHASE_WINDOWS.items():
            if p_lo <= lo <= p_hi:
                hi = min(hi, p_hi)
                break
        windows[n.time_label] = [lo, max(lo, hi)]
    return windows


def strip_ws(text: str) -> str:
    """逐字比对用的归一化：去排版字符（空白 / markdown 强调 / 树形符号）。"""
    return normalize_for_compare(text)


class BuildError(RuntimeError):
    pass


def load_specs(lane: Optional[str] = None) -> Dict[str, Any]:
    specs = {}
    for name in SPEC_MODULES:
        if lane and name != lane:
            continue
        mod = importlib.import_module(f"tools.hok_spec.{name}")
        specs[mod.LANE] = mod
    return specs


def _resolve_branches(node, entry) -> Tuple[List[Any], List[str]]:
    """按映射表找到对应的分支。找不到就报错 —— 静默跳过等于丢规则。"""
    key = entry["key"]
    branch = node.find(key)
    if branch is None:
        raise BuildError(
            f"{node.lane_label} / {node.header}：映射表里的前提在原文里找不到：{key!r}"
        )
    branches = [branch]
    missing = []
    for extra in entry.get("merge", []):
        b = node.find(extra)
        if b is None:
            missing.append(extra)
        else:
            branches.append(b)
    if missing:
        raise BuildError(
            f"{node.lane_label} / {node.header}：要合并的分支在原文里找不到：{missing}"
        )
    return branches, missing


def build_rules(tree: DecisionTree, specs: Dict[str, Any]) -> List[Dict[str, Any]]:
    rules: List[Dict[str, Any]] = []
    for lane_code, nodes in tree.lanes.items():
        spec = specs.get(lane_code)
        if spec is None:
            # ``--lane`` 只处理一条分路时，其余分路没有映射表 —— 跳过。
            continue
        windows = lane_windows(nodes)
        for node in nodes:
            lo, hi = windows[node.time_label]
            phase = phase_of_window([lo, hi])
            entries = [e for e in spec.RULES if e["node"] == node.time_label]
            if not entries:
                continue
            # **节点级合法打法集**：这个时段、这一类动作上，攻略给出的全部打法。
            #
            # 为什么需要它
            # ------------
            # 攻略在一个时段往往给好几套打法，各自有前提
            # （「敌方减员就开团」「人齐就别开团」「落后就只刷野区」）。
            # 这些前提**画面里读不出来** —— 系统只知道玩家这会儿在做什么动作。
            #
            # 照着字面判就会冤枉人：玩家在 6 分钟刷野，规则说「该开团」，
            # 可他没开团是因为他没减员 —— 他做得没错，只是跟规则库
            # 写的那一支不一样。
            #
            # 所以每条规则都带着这个集合：玩家做了其中**任何一件**，
            # 说明他这会儿在按攻略做事，就**不纠正他**；
            # 只有整段一次都没做过节点里的任何一件事时才有话说 ——
            # 那才是「改了一定有进步」的纠正。
            node_actions: Dict[str, List[str]] = {}
            for e in entries:
                fact = e["expect"]["fact"]
                bucket = node_actions.setdefault(fact, [])
                for v in e["expect"]["values"]:
                    if v not in bucket:
                        bucket.append(v)
            # **这一条能不能用来纠正玩家**。
            #
            # 判据（按攻略的结构算，不靠人拍）：
            #
            # 1. 触发条件里已经有**画面读得出**的事实（大招就绪、是否在团战）
            #    → 前提能判 → 能纠正。
            # 2. 这个时段里攻略给的**只有一类打法**（都在野区做事 / 都在
            #    中路占视野），而这条规则的期望动作又**覆盖了那一类的全部**
            #    → 前提只决定「哪种打法」，不决定「该不该做这类事」→ 能纠正。
            # 3. 其余（同一时段里有好几类打法，或者同类里还有几套互相取舍的
            #    打法，而取舍由读不出来的前提决定）→ **不能纠正**。
            #    原文说「该推塔」，可当时塔还有护盾、敌方打野就在我方半区，
            #    他退塔下才是对的 —— 拿这种前提去纠正人就是冤枉。
            #
            # 不能纠正的规则仍然在模式库里（攻略内容照常可查），
            # 只是不会拿它去指责玩家。
            _one_kind = len(node_actions) == 1
            for entry in entries:
                _readable_here = any(
                    c.get("fact") in _READABLE_FACTS
                    for c in entry.get("extra", [])
                )
                _whole = set(entry["expect"]["values"]) == set(
                    node_actions.get(entry["expect"]["fact"], [])
                )
                entry["_corrective"] = bool(_readable_here or (_one_kind and _whole))
            for entry in entries:
                branches, _ = _resolve_branches(node, entry)
                main = branches[0]
                premise_text = main.premise_text
                for b in branches[1:]:
                    premise_text += f"；{b.premise}"
                action_text = main.action_text
                for b in branches[1:]:
                    action_text += f"；{b.action_text}"
                outcome = main.outcome_text()

                # 文案规范化：去状态记号、把「+」改成顿号、替换模糊表述。
                # 引用（source.quoted）保持逐字不变 —— 那是给复核用的。
                premise_text = apply_fixes(normalize(premise_text))
                action_text = apply_fixes(normalize(action_text))
                outcome = apply_fixes(normalize(outcome))

                expect = entry["expect"]
                rule: Dict[str, Any] = {
                    "id": entry.get(
                        "id",
                        f"HOK-MODE-{LANE_SLUGS[lane_code]}-"
                        f"{node_slug(node.time_label)}-{main.index:02d}",
                    ),
                    "game": GAME,
                    "version": VERSION,
                    # 标题里的时间按**规则实际覆盖的窗口**重写：
                    # 「打野两分钟」在窗口是第 2 到第 3 分钟时要改成
                    # 「打野 2:00 ~ 4:00」—— 写点会让人以为只管第 2 分钟。
                    # 阶段词（「开局」「后期」）也换掉：它说的是一段，
                    # 却没说清是哪一段，而窗口知道答案。
                    "title": apply_fixes(retime_title(entry["title"], lo, hi)),
                    "category": f"{LANE_LABELS[lane_code]}·{PHASE_LABELS[phase]}",
                    # **分支**（原文一支一条）。
                    #
                    # 为什么要单独存：``premise`` / ``action`` 是把同一节点里
                    # 好几支用「；」拼起来的一句话 —— 玩家读到的是一张**菜单**
                    # （「敌方红开就己方蓝开，敌方蓝开就己方红开…」），
                    # 读不出这一局适用哪一支，于是「我做的正是菜单里的一支，
                    # 却说我做错了」—— 报告从此不敢信。
                    #
                    # 分支拆开之后，模型只要指认**第几支成立**，
                    # 报告就能给出那一支的具体动作，而不是整张菜单。
                    "branches": [
                        {
                            "premise": apply_fixes(normalize(b.premise_text)),
                            "action": apply_fixes(normalize(b.action_text)),
                            "outcome": apply_fixes(normalize(b.outcome_text())),
                        }
                        for b in branches
                    ],
                    "trigger": {
                        "all": [
                            {"fact": "position", "op": "==", "value": lane_code},
                            {"fact": "time_min", "op": "between", "value": [lo, hi]},
                        ] + list(entry.get("extra", [])),
                        "any": list(entry.get("any", [])),
                        # 分路通则与英雄专属**互斥**。
                        #
                        # 英雄级攻略给的是每个英雄自己的决策树，玩家选了自己的
                        # 英雄之后就只该按那一套判。两条都判的话，同一件事
                        # （「这一分钟该刷野」）会被分路通则和英雄专属各指控一次，
                        # 报告里出现两条互相重复的结论。
                        #
                        # 护栏用 ``my_hero in (全部英雄)``：玩家填了英雄就被挡住，
                        # 没填（空串）时条件不成立，通则照常生效 —— 没填英雄的局
                        # 仍然有话可说。
                        "none": list(entry.get("guards", [])) + [
                            {"fact": "my_hero", "op": "in", "value": HERO_ROSTER},
                        ],
                    },
                    "expect": {
                        "fact": expect["fact"],
                        "op": "in",
                        "value": list(expect["values"]),
                    },
                    "severity": entry.get("severity", "medium"),
                    "status": "active",
                    # 前提 / 动作 / 后果单独成字段，不塞在 rationale 里。
                    #
                    # 报告要写「结合前提条件，为什么该这么做」，
                    # 而 rationale 是给规则库页做**引文**用的 ——
                    # 那里出现「原文前提：」「原文结果：」是对的，
                    # 放到报告里就是一串元叙述，读者读不到任何关于这局的信息。
                    # 拆成三个字段之后，两边各取所需，不用互相迁就。
                    "premise": premise_text,
                    "action": action_text,
                    "outcome": outcome,
                    "rationale": _rationale(entry, premise_text, outcome),
                    "deviation_hint": apply_fixes(
                        normalize(entry.get("hint") or action_text)
                    ),
                    "tags": list(entry.get("tags", [])),
                    # 节点级合法打法集（见上面那段说明）。
                    "node_actions": list(
                        node_actions.get(entry["expect"]["fact"], [])
                    ),
                    # 这一条能不能用来纠正玩家（判据见上面）。
                    "corrective": bool(entry.get("_corrective", True)),
                    "source": {
                        "guide": GUIDE_NAME,
                        "locator": f"{LANE_LABELS[lane_code]} / {node.header} / "
                                   f"{main.premise}",
                        "url": "",
                        "quoted": main.quote(),
                        "collected_at": COLLECTED_AT,
                    },
                }
                if entry.get("scope") == "at_least_once":
                    rule["expect_scope"] = "at_least_once"
                subject = entry.get("subject")
                if subject:
                    rule["subject"] = {
                        "fact": subject["fact"],
                        "values": list(subject["values"]),
                    }
                rules.append(rule)
    return rules


def _default_rationale(premise: str, outcome: str) -> str:
    parts = []
    if premise:
        parts.append(f"原文前提：{premise}")
    if outcome:
        parts.append(f"原文结果：{outcome}")
    return "；".join(parts) + "。"


def _rationale(entry: Dict[str, Any], premise: str, outcome: str) -> str:
    """依据 = 原文前提与后果 + 映射表补充的说明。"""
    text = _default_rationale(premise, outcome)
    extra = (entry.get("why") or "").strip()
    if extra:
        text += extra
    return text


# --------------------------------------------------------------------------
# 互斥判断
# --------------------------------------------------------------------------


def _window_of(rule: Dict[str, Any]) -> List[int]:
    for c in rule["trigger"]["all"]:
        if c["fact"] == "time_min" and c["op"] == "between":
            return list(c["value"])
    return [0, 60]


def _windows_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    a_lo, a_hi = _window_of(a)
    b_lo, b_hi = _window_of(b)
    return a_lo <= b_hi and b_lo <= a_hi


def _cond_ok(cond: Dict[str, Any], value: Any) -> bool:
    op, v = cond["op"], cond.get("value")
    try:
        if op == "==":
            return value == v
        if op == "!=":
            return value != v
        if op == ">":
            return value > v
        if op == ">=":
            return value >= v
        if op == "<":
            return value < v
        if op == "<=":
            return value <= v
        if op == "in":
            return value in v
        if op == "not_in":
            return value not in v
        if op == "between":
            return v[0] <= value <= v[1]
    except TypeError:
        return False
    return False


def _constraints(rule: Dict[str, Any]) -> Dict[str, Tuple[List[Any], List[Any]]]:
    """一条规则在每个标量事实上 impose 的约束：``(all 条件, none 条件)``。"""
    out: Dict[str, Tuple[List[Any], List[Any]]] = {}
    for c in rule["trigger"]["all"]:
        out.setdefault(c["fact"], ([], []))[0].append(c)
    for c in rule["trigger"]["none"]:
        out.setdefault(c["fact"], ([], []))[1].append(c)
    return out


def _candidates(conds: List[Dict[str, Any]]) -> List[Any]:
    vals: List[Any] = []
    for c in conds:
        v = c.get("value")
        if isinstance(v, (list, tuple, set)):
            for x in v:
                vals.append(x)
                if isinstance(x, (int, float)) and not isinstance(x, bool):
                    vals.extend([x - 1, x + 1])
        else:
            vals.append(v)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                vals.extend([v - 1, v + 1])
    return vals or [None]


def _mutually_exclusive(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """两条规则是否**不可能同时成立**。

    判据：存在某个标量事实，两条规则对它的要求无法同时满足 ——
    比如一条要求 ``kill_diff > 0``、另一条要求 ``kill_diff < 0``。

    这种情况下两条规则要求不同动作不算「互相指控」：玩家只会被其中一条判，
    不会被两条同时判。这正是库里用护栏隔开互斥分支的做法。
    """
    ca, cb = _constraints(a), _constraints(b)
    for fact in set(ca) & set(cb):
        all_a, none_a = ca[fact]
        all_b, none_b = cb[fact]
        cands = _candidates(all_a + none_a + all_b + none_b)
        for v in cands:
            ok_a = all(_cond_ok(c, v) for c in all_a) and not any(
                _cond_ok(c, v) for c in none_a
            )
            ok_b = all(_cond_ok(c, v) for c in all_b) and not any(
                _cond_ok(c, v) for c in none_b
            )
            if ok_a and ok_b:
                break
        else:
            return True
    return False


# --------------------------------------------------------------------------
# 校验
# --------------------------------------------------------------------------


def validate(rules: List[Dict[str, Any]], tree: DecisionTree,
             specs: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    from app.adapters import registry as _registry  # noqa: F401  触发适配器注册
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
        if not r["id"].startswith("HOK-MODE-"):
            errors.append(f"规则 id 前缀不对：{r['id']}")

    # 2. 期望动作必须可勾
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
                errors.append(f"{r['id']} 的适用时机引用了未注册事实：{subject['fact']}")

    # 3. 引用必须逐字来自原文
    flat = strip_ws(tree.text)
    for r in rules:
        quoted = r["source"]["quoted"]
        if not quoted.strip():
            errors.append(f"{r['id']} 缺引用")
        elif strip_ws(quoted) not in flat:
            errors.append(f"{r['id']} 的引用在原文里找不到：{quoted!r}")

    # 4. 同一格里不能互相指控
    by_cell: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rules:
        window = _window_of(r)
        phase = phase_of_window(window)
        lane = r["trigger"]["all"][0]["value"]
        by_cell.setdefault((lane, phase), []).append(r)
    for (lane, phase), group in by_cell.items():
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
                    # 时间窗不重叠时，两条规则不可能同时判到同一回合 ——
                    # 「玩家被两条同时指控」这件事不会发生。
                    if not _windows_overlap(a, b):
                        continue
                    if _mutually_exclusive(a, b):
                        continue
                    errors.append(
                        f"{lane}·{PHASE_LABELS[phase]} 里 {a['id']} 与 {b['id']} "
                        f"要求同一动作事实 {fact} 但取值没有交集，"
                        "玩家会被两条同时指控"
                    )

    # 5. 动作事实上的 none 护栏不能和同一格的期望撞车
    for (lane, phase), group in by_cell.items():
        guarded: Dict[str, List[str]] = {}
        expecting: Dict[str, List[str]] = {}
        for r in group:
            for c in r["trigger"]["none"]:
                if c["fact"] in action_facts:
                    guarded.setdefault(c["fact"], []).append(r["id"])
            if r["expect"]["fact"] in action_facts:
                expecting.setdefault(r["expect"]["fact"], []).append(r["id"])
        for fact in set(guarded) & set(expecting):
            if set(guarded[fact]) & set(expecting[fact]):
                errors.append(
                    f"{lane}·{PHASE_LABELS[phase]} 里 {guarded[fact]} 与 "
                    f"{expecting[fact]} 在动作事实 {fact} 上既写护栏又期望"
                )

    # 5b. 同一格里，适用时机相同的两条规则不能给出**相反**的建议。
    #
    # 攻略的互斥分支（「对方有主宰 → 守高地」对「我方有主宰 → 带线拆塔」）
    # 前提是判不了的，所以两条规则都会在同一段生效。它们的判据又在同一个
    # 动作事实上（都是站位 / 都是目标），于是玩家推线的那一帧可能同时命中
    # 两条：一条说「该拆塔」，一条说「该守塔」。
    #
    # 这不是「取值不重叠」那种机械冲突 —— 两者期望的事实不同，
    # 但**同时判到同一帧**就意味着报告自相矛盾。解法是把其中一条的
    # 适用时机收窄到它真正管的那一类动作上（守 → 只在防守类动作上判）。
    for (lane, phase), group in by_cell.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                sa, sb = a.get("subject"), b.get("subject")
                if not sa or not sb or sa["fact"] != sb["fact"]:
                    continue
                if a["expect"]["fact"] == b["expect"]["fact"]:
                    continue          # 同一个事实的冲突上面已经查过了
                if not _windows_overlap(a, b):
                    continue
                if _mutually_exclusive(a, b):
                    continue
                va = set(a["expect"]["value"])
                vb = set(b["expect"]["value"])
                if va & vb:
                    continue
                errors.append(
                    f"{lane}·{PHASE_LABELS[phase]} 里 {a['id']}（期望 "
                    f"{a['expect']['fact']} {sorted(va)}）与 {b['id']}（期望 "
                    f"{b['expect']['fact']} {sorted(vb)}）适用时机相同却是相反建议："
                    "同一帧可能被两条同时判，报告自相矛盾。"
                    "把其中一条的 subject 收窄到它真正管的那类动作上。"
                )

    # 6. 分路 × 阶段不能有空格（只校验本次参与生成的分路）
    cells = set(by_cell.keys())
    for lane in LANE_LABELS:
        if lane not in specs:
            continue
        for phase in PHASE_LABELS:
            if (lane, phase) not in cells:
                errors.append(
                    f"{LANE_LABELS[lane]}·{PHASE_LABELS[phase]} 没有模式"
                )

    # 7. 攻略里的每条分支都要有归宿
    mapped = set()
    for lane_code, nodes in tree.lanes.items():
        spec = specs.get(lane_code)
        if spec is None:
            continue
        skipped = {
            (s["node"], s["key"]) for s in getattr(spec, "SKIPPED", [])
        }
        for node in nodes:
            for b in node.branches:
                hit = False
                for entry in spec.RULES:
                    if entry["node"] != node.time_label:
                        continue
                    keys = [entry["key"]] + list(entry.get("merge", []))
                    if b.key in keys:
                        hit = True
                        break
                if not hit and (node.time_label, b.key) not in skipped:
                    errors.append(
                        f"攻略分支没归宿：{LANE_LABELS[lane_code]} / {node.header} / "
                        f"{b.key}（要么映射成规则，要么在 SKIPPED 里写明不可判定的理由）"
                    )
                if hit:
                    mapped.add((node.time_label, b.key))

    # 8. 准入校验
    for r in rules:
        rule = Rule.from_dict(r)
        reasons = check_admission(rule, known_facts, DEFAULT_SUBJECTIVE_MARKERS,
                                  action_facts)
        for reason in reasons:
            errors.append(f"{r['id']} 未通过准入：{reason}")

    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验，不写文件")
    ap.add_argument(
        "--lane", default="",
        help="只处理某一条分路（jungle/mid/side/carry/roam），用于单独调试",
    )
    args = ap.parse_args()

    tree = load_guide(GUIDE_PATH)
    specs = load_specs(args.lane or None)
    rules = build_rules(tree, specs)
    errors = validate(rules, tree, specs)

    if errors:
        print(f"✗ {len(errors)} 处问题：")
        for e in errors:
            print("  -", e)
        return 1

    if not args.check:
        if args.lane:
            print(f"✓ {args.lane} 校验通过（{len(rules)} 条），未写文件")
            return 0
        with open(OUT_PATH, "w", encoding="utf-8") as fh:
            json.dump(rules, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        print(f"✓ 写入 {OUT_PATH}（{len(rules)} 条）")
    else:
        print(f"✓ 校验通过（{len(rules)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
