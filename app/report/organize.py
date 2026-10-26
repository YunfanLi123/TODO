"""展示层：把库里的东西组织成人能读的样子。

这里只做**派生** —— 名字、层级、顺序都不写进数据库，页面每次现算。

理由：这些是「怎么叫它」的问题，不是「它是什么」的问题。写进库就意味着
改名要洗数据，而规则库洗数据的代价很高（历史对局要靠规则 id 回溯，
id 一变，历史结论就接不上了）。

两套命名
--------

**对局名**：``分路·英雄·胜负``

    打野·赵云·胜利

    分路是这局最主要的信息 —— 操作模式本来就按分路生效，英雄决定
    哪一层模式适用，胜负是这局的结果。三个都缺得了，所以缺哪个就
    写哪个缺，不猜。

**模式名**：``分路-英雄-时期-序号``

    打野-前期-1
    对抗路-哪吒-前期-1
    对抗路-哪吒-前期-2

    序号**只在名字前面部分完全相同时**才加一。也就是说
    「对抗路-哪吒-前期」下面有两条，就叫 1 和 2；而
    「对抗路-哪吒-中期」重新从 1 开始 —— 它的前缀和前面那条不一样。

    没有英雄的（分路通则）不写英雄那一段：``打野-前期-1``。

这两套命名都是**派生显示名**，不进库。库里存的是规则 id 与条件，
名字随时可以改，改名不会动到任何历史数据。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..core.dsl import Rule
from ..core.lane_analysis import mode_cell, phase_of_window

#: 局势阶段的展开顺序。报告、模式库、标注页共用这一份。
PHASE_ORDER: List[str] = ["early", "mid", "late"]

#: 胜负取值 -> 中文名。没打过 / 没填的**不在这里** ——
#: 「没填」和「输了」是两件事，混成一个词就是编造。
RESULT_LABELS: Dict[str, str] = {
    "win": "胜利",
    "lose": "失败",
}

#: 缺的那一段在名字里怎么写。宁可写「没填」，也不要悄悄省掉 ——
#: 省掉的话「打野·赵云」看上去像完整名字，其实少了胜负。
HERO_MISSING = "未选英雄"
RESULT_MISSING = "结果未填"

#: 对局名里各段之间的分隔符。用中点而不是短横：短横和模式名里的
#: 分隔符一样，两套名字混在一个页面上会看错。
MATCH_SEP = "·"

#: 模式名里各段之间的分隔符。
MODE_SEP = "-"


# --------------------------------------------------------------------------
# 对局
# --------------------------------------------------------------------------


def match_identity(payload: Dict[str, Any], adapter: Any) -> Dict[str, Any]:
    """从一局对局的原始数据里读出「它是谁」。

    读的是**已经录进去的字段**，不推断：分路取玩家填的值（规整过），
    英雄取玩家选的值，胜负取 ``outcome.result``。没填的就是没填。

    ``player.role`` 是早期手工录入留下的键（那批对局写的是 role 而不是
    position）。它同样是被录进去的事实，读它不算猜 —— 但只在
    ``position`` 缺失时才用，两个键都写时以 ``position`` 为准。
    """
    player = payload.get("player") or {}
    raw_lane = player.get("position") or player.get("role") or ""
    lane = adapter.normalize_position(raw_lane) if raw_lane else "unknown"
    lane_label = adapter.position_labels.get(lane) or lane or "未指定"

    hero = str(player.get("hero") or "").strip()

    outcome = payload.get("outcome") or {}
    result = str(outcome.get("result") or "").strip().lower()
    result_label = RESULT_LABELS.get(result, "")

    name = MATCH_SEP.join([
        lane_label,
        hero or HERO_MISSING,
        result_label or RESULT_MISSING,
    ])

    return {
        "lane": lane,
        "lane_label": lane_label,
        "hero": hero,
        "result": result,
        "result_label": result_label,
        "name": name,
    }


def match_display_name(payload: Dict[str, Any], adapter: Any) -> str:
    """页面上显示的「这是哪一局」。

    以前这些地方直接摆内部 id（``hok-video-69481d223ed9``）——
    那是机器认的东西，人不认。用户要找的是「昨天那局打野」，
    或者更直接的：**那个视频文件叫什么名字**。

    所以：视频对局用**原始文件名**打头（录屏时的名字最认得出来），
    后面跟分路与胜负；手工对局没有文件名，就用「分路·英雄·胜负」。
    """
    ident = match_identity(payload, adapter)
    video_name = str(payload.get("video_name") or "").strip()
    if not video_name:
        return ident["name"]
    parts = [video_name]
    if ident["lane_label"] and ident["lane_label"] != "未指定":
        parts.append(ident["lane_label"])
    if ident["result_label"]:
        parts.append(ident["result_label"])
    return MATCH_SEP.join(parts)


def group_matches_by_lane(
    entries: List[Dict[str, Any]],
    lane_order: Iterable[str],
    lane_labels: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """按分路分组。顺序按适配器给的 ``position_order``。

    归不到任何一条分路的（玩家没填分路）单开一组「未指定」放在最后 ——
    不能因为归不进去就不显示，那一局就凭空消失了。
    """
    order = list(lane_order)
    labels = dict(lane_labels or {})
    groups: Dict[str, Dict[str, Any]] = {
        code: {"code": code, "label": labels.get(code, code), "matches": []}
        for code in order
    }
    loose: List[Dict[str, Any]] = []

    for entry in entries:
        code = entry.get("lane") or "unknown"
        if code in groups:
            groups[code]["matches"].append(entry)
        else:
            loose.append(entry)

    out: List[Dict[str, Any]] = []
    for code in order:
        g = groups[code]
        g["matches"].sort(key=lambda m: m.get("played_sort") or "", reverse=True)
        out.append(g)
    if loose:
        loose.sort(key=lambda m: m.get("played_sort") or "", reverse=True)
        out.append({"code": "unknown", "label": "未指定", "matches": loose})
    return out


# --------------------------------------------------------------------------
# 操作模式
# --------------------------------------------------------------------------


def _trigger_value(rule: Rule, fact: str) -> Optional[str]:
    """读触发条件里的某个事实取值（只取单值条件）。

    英雄与对位优劣都是这样挂上去的：``my_hero == "哪吒"``、
    ``matchup == "advantage"``。取不到就是取不到 —— 不去标题里猜。
    """
    for cond in rule.trigger.all:
        if cond.fact != fact:
            continue
        if cond.op not in ("==", "in"):
            continue
        values = cond.value if isinstance(cond.value, list) else [cond.value]
        real = [v for v in values if v not in (None, "", "unknown")]
        if len(real) == 1:
            return str(real[0])
    return None


def mode_phase(rule: Rule) -> str:
    """一条模式属于哪个时期。

    从它自己的时间窗算 —— 时间窗就是「这条模式在什么时候成立」，
    比 category 里的措辞可靠。
    """
    cell = mode_cell(rule)
    window = cell[1] if cell else None
    return phase_of_window(window)


def mode_name(lane_label: str, hero: str, phase_label: str, index: int) -> str:
    """拼一条模式的显示名。

    ``index`` 是**同一个前缀组内**的序号，前缀 = 分路 + 英雄 + 时期。
    英雄为空（分路通则）时不写英雄那一段。
    """
    parts = [lane_label]
    if hero:
        parts.append(hero)
    parts.append(phase_label)
    parts.append(str(index))
    return MODE_SEP.join(parts)


def _expect_label(rule: Rule, adapter: Any) -> str:
    """期望动作翻成人话。

    取值是动作码（``push_wave``），页面上要给人看的是「推线压塔」。
    拿不到中文名就原样给出 —— 宁可显示一个看不懂的码，
    也不要显示一个编出来的名字。
    """
    labels = dict(getattr(adapter, "action_labels", {}) or {})
    values = rule.expect_value
    values = values if isinstance(values, list) else [values]
    names = [labels.get(v, str(v)) for v in values]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]}或{names[1]}"
    return "、".join(names[:-1]) + f"或{names[-1]}"


def build_mode_tree(
    rules: List[Dict[str, Any]],
    adapter: Any,
    only_active: bool = True,
) -> List[Dict[str, Any]]:
    """把规则列表组织成「分路 → 英雄 → 模式」的三层树。

    ``rules`` 是 :func:`app.rules.review.build_review_view` 给出的条目
    （每条带 ``rule_id`` / ``title`` / ``payload`` / ``status``）。

    树里每一层都带好了显示名与条数，模板只管排版，不再自己数。

    只有**确实有模式**的英雄才会出现在树上 —— 英雄这一层是给「查这个
    英雄怎么打」用的，给一个没有模式的英雄开一节，点进去是空的。

    ``only_active=True``（默认）时**失效的规则不进树**。
    模式库页展示的是「当前生效的操作模式」，把退役规则和被停用的规则
    混在同一份清单里，用户看到的就是「旧的还在」—— 那正是用户要求
    「旧的通通删掉」时最不想看到的样子。它们并没有丢：审核工作台
    （``/api/rules/review``）仍然完整列出，历史对局复盘也仍然按当时的
    规则状态计算。
    """
    phase_labels = dict(getattr(adapter, "phase_labels", {}) or {})
    lane_labels = dict(getattr(adapter, "position_labels", {}) or {})
    lane_order = list(getattr(adapter, "position_order", []) or [])

    # 分路 -> 英雄 -> 时期 -> [条目]
    tree: Dict[str, Dict[str, Any]] = {
        code: {"code": code, "label": lane_labels.get(code, code),
               "modes": [], "heroes": {}}
        for code in lane_order
    }

    # 同一前缀组内的序号计数器
    counters: Dict[Tuple[str, str, str], int] = {}

    def sorted_rules() -> List[Dict[str, Any]]:
        # 排序固定下来，序号才不会因为数据库返回顺序变了而漂移
        return sorted(rules, key=lambda r: str(r.get("rule_id") or ""))

    for item in sorted_rules():
        if only_active and (item.get("status") or "active") != "active":
            continue
        try:
            rule = Rule.from_dict(item["payload"])
        except Exception:                       # noqa: BLE001
            continue
        cell = mode_cell(rule)
        if cell is None:
            # 没有分路的模式不属于任何一条分路，按分路组织时它无处安放
            continue
        lane = cell[0]
        node = tree.get(lane)
        if node is None:
            # 库里出现了适配器不认的分路 —— 新增一条，别把整条模式丢掉
            node = {"code": lane, "label": lane_labels.get(lane, lane),
                    "modes": [], "heroes": {}}
            tree[lane] = node

        hero = _trigger_value(rule, "my_hero") or ""
        matchup = _trigger_value(rule, "matchup") or ""
        phase = mode_phase(rule)
        phase_label = phase_labels.get(phase, phase)

        key = (lane, hero, phase)
        counters[key] = counters.get(key, 0) + 1
        name = mode_name(node["label"], hero, phase_label, counters[key])

        entry = {
            "rule_id": item.get("rule_id"),
            "name": name,
            "title": item.get("title") or "",
            "phase": phase,
            "phase_label": phase_label,
            "phase_order": PHASE_ORDER.index(phase) if phase in PHASE_ORDER else 9,
            "hero": hero,
            "expect_label": _expect_label(rule, adapter),
            "lane": lane,
            "lane_label": node["label"],
            "matchup": matchup,
            "status": item.get("status") or "active",
            "category": item.get("category") or "",
            "payload": item.get("payload") or {},
        }

        if hero:
            node["heroes"].setdefault(hero, []).append(entry)
        else:
            node["modes"].append(entry)

    out: List[Dict[str, Any]] = []
    for code in lane_order:
        node = tree.get(code)
        if node is None:
            continue
        heroes = [
            {
                "name": name,
                "modes": sorted(
                    entries, key=lambda e: (PHASE_ORDER.index(e["phase"])
                                            if e["phase"] in PHASE_ORDER else 9)
                ),
                "count": len(entries),
            }
            for name, entries in node["heroes"].items()
        ]
        heroes.sort(key=lambda h: h["name"])
        out.append({
            "code": node["code"],
            "label": node["label"],
            "modes": sorted(
                node["modes"], key=lambda e: (PHASE_ORDER.index(e["phase"])
                                             if e["phase"] in PHASE_ORDER else 9)
            ),
            "heroes": heroes,
            "mode_count": len(node["modes"]) + sum(h["count"] for h in heroes),
            "hero_count": len(heroes),
        })
    return out


def find_lane_node(tree: List[Dict[str, Any]], code: str) -> Optional[Dict[str, Any]]:
    for node in tree:
        if node["code"] == code:
            return node
    return None


# --------------------------------------------------------------------------
# BP 阶段
# --------------------------------------------------------------------------

#: BP 的决策环节顺序 —— 按 BP 实际发生的先后排，不是按字码。
BP_PHASE_ORDER: Tuple[str, ...] = (
    "ban_1", "pick_1", "pick_2", "pick_3",
    "pick_late_1", "pick_last", "final_check",
)


def build_bp_modes(rules: List[Dict[str, Any]], adapter: Any) -> List[Dict[str, Any]]:
    """把 BP 规则摆成和局内模式**一样的形状**。

    摆成同一个形状，是为了共用同一套渲染 —— 模式库页里 BP 那一节
    与五条分路长得一样，不是另起一套样式。
    """
    phase_labels = dict(getattr(adapter, "bp_phase_labels", {}) or {})

    out: List[Dict[str, Any]] = []
    counters: Dict[str, int] = {}
    for item in sorted(rules, key=lambda r: str(r.get("rule_id") or "")):
        if (item.get("status") or "active") != "active":
            continue
        try:
            rule = Rule.from_dict(item["payload"])
        except Exception:                       # noqa: BLE001
            continue

        phase = _trigger_value(rule, "bp_phase") or ""
        label = phase_labels.get(phase, phase or "BP")
        counters[phase] = counters.get(phase, 0) + 1

        out.append({
            "rule_id": item.get("rule_id"),
            # BP 模式名：BP·环节-序号（同一环节里从 1 数）
            "name": f"BP·{label}-{counters[phase]}",
            "title": item.get("title") or "",
            "phase": phase,
            "phase_label": label,
            "phase_order": (BP_PHASE_ORDER.index(phase)
                            if phase in BP_PHASE_ORDER else 9),
            "hero": "",
            "expect_label": _expect_label(rule, adapter),
            "lane": "bp",
            "lane_label": "BP",
            "matchup": "",
            "status": item.get("status") or "active",
            "category": item.get("category") or "",
            "payload": item.get("payload") or {},
            "coverage": None,
        })
    return out
