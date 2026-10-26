"""给模型用的「这一时段该做什么」清单。

为什么要有这一层
----------------
早先的流程是：模型只负责读出原子事实（刷野 / 推塔 / 大招亮着 / 是否团战），
然后由规则引擎拿**硬触发条件**去套攻略。

问题是攻略的分支前提（「塔的早期护盾已消失」「暴君倒计时进入最后一段」
「对方有人阵亡未复活」）**画面里读不出来**。前提读不出来，
引擎就不敢判 —— 王者 S45 分路层 84 条里只有 9 条敢拿来纠正玩家，
报告于是几乎是空的。

而模型真正擅长的事恰恰是：**看着画面，自己权衡这几套打法哪个成立**。
所以清单得先送到模型手上，裁决权才在模型那边。

这一层只做一件事：把模式库里的规则整理成「按时间窗取用」的清单，
并把**同一个节点里攻略认可的全部打法**并起来（``node_actions``）。

``node_actions`` 是「不冤枉人」的关键
------------------------------------
攻略在一个时段往往给好几套打法，各自有前提。玩家做了其中**另一件**时，
他这会儿是在按攻略做事，只是跟规则库写的那一条不一样 —— **不纠正他**。
只有当玩家**整段一次都没做过**这个节点里的任何一件事时，才有话说。

清单从**种子库**读，不碰数据库（扫描跑在后台线程，见
:mod:`app.collection.scan_jobs` 的约定）。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from . import density

#: 分路编码 -> 中文名。与适配器一致，但这里独立写一份：
#: 清单模块要在**不导入适配器**的情况下也能跑（适配器的导入会拖起整套依赖）。
LANE_LABELS: Dict[str, str] = {
    "jungle": "打野",
    "mid": "中路",
    "side": "对抗路",
    "carry": "发育路",
    "roam": "游走",
}

#: 分路层规则的 id 前缀。英雄层是 ``HOK-HERO-``。
_LANE_PREFIX = "HOK-MODE-"
_HERO_PREFIX = "HOK-HERO-"


def _trigger_of(rule: Dict[str, Any]) -> Dict[str, Any]:
    return rule.get("trigger") or {}


def lane_of(rule: Dict[str, Any]) -> Optional[str]:
    """这条规则属于哪条分路。

    从**触发条件**读，不解析分类字符串 —— 分类名是给人看的，
    改了措辞数据就跟着漂；而 ``position == jungle`` 是数据本身。

    写了全部分路（「全位置通用」）的规则返回 ``None``：
    它不属于任何单一分路，按分路给清单时不该挂在某一路名下。
    """
    for cond in _trigger_of(rule).get("all") or []:
        if cond.get("fact") != "position":
            continue
        if cond.get("op") not in ("==", "in"):
            continue
        values = cond.get("value")
        values = values if isinstance(values, list) else [values]
        real = [v for v in values if v != "unknown"]
        if len(real) == 1:
            return str(real[0])
    return None


def hero_of(rule: Dict[str, Any]) -> Optional[str]:
    """这条规则是给哪个英雄的（英雄层才有）。"""
    for group in ("all", "any"):
        for cond in _trigger_of(rule).get(group) or []:
            if cond.get("fact") != "my_hero":
                continue
            if cond.get("op") == "==":
                return str(cond.get("value") or "")
            if cond.get("op") == "in":
                values = cond.get("value")
                if isinstance(values, list) and len(values) == 1:
                    return str(values[0])
    return None


def window_of(rule: Dict[str, Any]) -> Optional[List[int]]:
    """这条规则管哪几分钟。"""
    return density._time_windows(rule)


def _expect_of(rule: Dict[str, Any]) -> Tuple[str, List[str]]:
    expect = rule.get("expect") or {}
    fact = str(expect.get("fact") or "")
    values = expect.get("value")
    values = values if isinstance(values, list) else [values]
    return fact, [str(v) for v in values if v is not None]


#: 种子规则（按游戏）。读一次缓存在进程里 ——
#: 清单是每个分析窗都要取一次的东西，每次都重读 1680 条 JSON 太浪费。
_SEED_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def seed_rules(game: str) -> List[Dict[str, Any]]:
    """这个游戏的模式库规则（局内，BP 除外）。"""
    if game not in _SEED_CACHE:
        _SEED_CACHE[game] = density._seed_rules(game, skip_bp=True)
    return _SEED_CACHE[game]


def clear_cache() -> None:
    """清掉缓存（模式库换了内容时用）。"""
    _SEED_CACHE.clear()


def _node_key(rule: Dict[str, Any]) -> Optional[Tuple[str, Tuple[int, int]]]:
    """节点标识：``(分路, 时间窗)``。

    只认**分路层**规则（``HOK-MODE-``）。英雄层的规则（``HOK-HERO-``）
    同样挂在 ``position == jungle`` 上、时间窗也常常一样 ——
    一并进来会把节点的合法打法集撑成几百个取值（40 个打野英雄各算一遍），
    「玩家做过其中任何一件就不纠正」这条就废了。
    英雄层是**个性化**的那一层，不参与节点通则。
    """
    if not str(rule.get("id") or "").startswith(_LANE_PREFIX):
        return None
    lane = lane_of(rule)
    win = window_of(rule)
    if lane is None or not win:
        return None
    return lane, (int(win[0]), int(win[1]))


def node_action_map(game: str) -> Dict[Tuple[str, Tuple[int, int]], Dict[str, List[str]]]:
    """**节点 → 这个时段攻略要求玩家做的事**（按动作事实分组）。

    键是 ``(分路, 时间窗)``，值是 ``{动作事实: [取值, ...]}``。

    为什么要并起来
    --------------
    攻略在一个节点里给好几套打法（「塔的护盾没了 → 推塔」
    「暴君进倒计时 → 占龙坑」「对面先手 → 不进团」）。玩家做了其中
    **任何一件**，说明他这会儿在按攻略做事，就不该被纠正 ——
    他做得没错，只是跟规则库写的那一条不一样。

    这个并集从**规则本身**算（同一节点里所有规则的期望动作取并集），
    不依赖种子里的 ``node_actions`` 字段 —— 英雄层的规则那个字段是空的，
    但机制上等价，两边用同一套算法就不会出现「分路层算了、英雄层没算」。
    """
    out: Dict[Tuple[str, Tuple[int, int]], Dict[str, List[str]]] = {}
    for rule in seed_rules(game):
        key = _node_key(rule)
        if key is None:
            continue
        fact, values = _expect_of(rule)
        if not fact or not values:
            continue
        bucket = out.setdefault(key, {}).setdefault(fact, [])
        for v in values:
            if v not in bucket:
                bucket.append(v)
    return out


def window_checklist(
    game: str,
    lo: int,
    hi: int,
    lane: str = "",
    hero: str = "",
) -> List[Dict[str, Any]]:
    """取 ``[lo, hi]`` 这一分钟段的「该做什么」清单。

    参数
    ----
    game : 游戏编码（``hok``）。
    lo, hi : 局内分钟，闭区间。
    lane : 玩家的分路。**已知就只给这一条线的** ——
        清单小一大截，模型也不会对着别的线的打法乱指。
        不知道就返回全部线（每窗十几次，仍然放得进提示词）。
    hero : 玩家的英雄。英雄层的规则**只有知道英雄才给** ——
        全部英雄的规则有 1600 条，一次塞进提示词会把模型淹掉。

    返回的每一项
    ------------
    ``id`` / ``lane`` / ``hero`` / ``title`` / ``premise`` / ``action`` /
    ``outcome`` / ``expect_fact`` / ``expect_values`` / ``node_actions`` /
    ``window``。

    ``premise`` / ``action`` / ``outcome`` 是**攻略原文**，逐字取的 ——
    送给模型的是真源，不是我们的转述。
    """
    nodes = node_action_map(game)
    out: List[Dict[str, Any]] = []
    for rule in seed_rules(game):
        rid = str(rule.get("id") or "")
        win = window_of(rule)
        if not win or rule.get("status") == "deprecated":
            continue
        # 时间窗**相交**就算相关 —— 分析窗比规则窗大时
        # （规则管 6~9 分，窗是 6~10 分），相交的那一段仍要判。
        if win[1] < lo or win[0] > hi:
            continue

        is_hero_rule = rid.startswith(_HERO_PREFIX)
        if not rid.startswith(_LANE_PREFIX) and not is_hero_rule:
            continue

        rule_lane = lane_of(rule)
        rule_hero = hero_of(rule)
        if lane and rule_lane != lane:
            continue
        if is_hero_rule:
            if not hero:
                continue
            if rule_hero != hero:
                continue

        fact, values = _expect_of(rule)
        if not fact or not values:
            continue

        key = _node_key(rule)
        node = nodes.get(key, {}) if key else {}
        subject = rule.get("subject") or None
        out.append({
            "id": rid,
            "lane": rule_lane or "",
            "lane_label": LANE_LABELS.get(rule_lane or "", rule_lane or ""),
            "hero": rule_hero or "",
            "title": str(rule.get("title") or ""),
            "premise": str(rule.get("premise") or ""),
            "action": str(rule.get("action") or ""),
            "outcome": str(rule.get("outcome") or ""),
            "expect_fact": fact,
            "expect_values": values,
            # **分支**（原文一支一条）。模型指认「第几支成立」，
            # 报告就能给出那一支的具体动作 —— 而不是一张
            # 「敌方红开就蓝开、敌方蓝开就红开」的菜单。
            "branches": [
                {
                    "premise": str(b.get("premise") or ""),
                    "action": str(b.get("action") or ""),
                    "outcome": str(b.get("outcome") or ""),
                }
                for b in (rule.get("branches") or [])
                if isinstance(b, dict)
            ],
            # 攻略给的**具体改法**（人话）。报告里「建议操作描述」用它 ——
            # 只说「该刷野」等于没说，玩家要的是能照着做的句子。
            "advice": str(rule.get("deviation_hint") or ""),
            # **适用时机**：这条规则只在玩家做这些动作时才拿来判。
            # 攻略写的是「在做 A 还是做 B 的时候该选 A」——
            # 玩家这一帧在刷野、跟这条规则没关系，就不该拿它去纠正他。
            "subject": {
                "fact": str(subject.get("fact") or ""),
                "values": [str(v) for v in (subject.get("values") or [])],
            } if subject else None,
            # 这个节点里攻略**要求**做的全部打法（跨动作事实）。
            "node_actions": node,
            "window": [int(win[0]), int(win[1])],
        })

    out.sort(key=lambda r: (r["window"][0], r["window"][1], r["lane"], r["id"]))
    return out


def checklist_prompt_block(items: List[Dict[str, Any]]) -> str:
    """把清单渲染成提示词里的一段（给模型看的「该做什么」）。

    刻意**逐字**带上原文的前提 / 动作 / 后果：模型要判断的是
    「这个前提在画面里成立吗」，前提得让它看到原话。
    """
    if not items:
        return ""
    lines: List[str] = []
    for item in items:
        head = f"【{item['id']}】"
        if item.get("lane_label"):
            head += item["lane_label"] + " · "
        head += item["title"]
        lines.append(head)
        branches = item.get("branches") or []
        if branches:
            # **分支一支一条**。模型要指认「哪一支成立」，
            # 得先看得清有几支、每支的前提与做法各是什么。
            for i, b in enumerate(branches, start=1):
                lines.append(f"  第 {i} 支 前提：{b['premise']}")
                lines.append(f"  第 {i} 支 该做：{b['action']}")
                if b.get("outcome"):
                    lines.append(f"  第 {i} 支 做了的后果：{b['outcome']}")
        else:
            if item.get("premise"):
                lines.append(f"  前提：{item['premise']}")
            if item.get("action"):
                lines.append(f"  该做：{item['action']}")
            if item.get("outcome"):
                lines.append(f"  做了的后果：{item['outcome']}")
    return "\n".join(lines)
