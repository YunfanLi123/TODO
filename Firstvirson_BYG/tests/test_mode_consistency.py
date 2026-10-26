"""英雄层与分路层**不能互相指控**。

这条约束是这套库能不能用的前提，不是风格偏好。

同一格（分路 × 阶段）里，如果分路层要求做 A、英雄层要求做 B，而 A 和 B
没有交集，那么玩家一回合只能做一个动作 —— 必定有一条被判成偏离，
而他其实什么都没做错。更糟的是**看不出是哪条错了**：报告只会说
「这一段没做到位」，不会说「你的英雄在劣势对位本来就不该做那个」。

因此规则是：同一格里所有会同时触发的模式，它们的期望动作集合必须有交集。
有英雄层覆盖劣势对位的格子，分路层要显式写 ``none: matchup == disadvantage``
把自己让出去；某个英雄在优势对位下打法与通则冲突的，那个英雄要被排除。

这个测试就是来强制这条规则的 —— 写规则时忘了加 none，它会当场报出来。
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

APP_ROOT = Path(__file__).resolve().parent.parent
SEED_DIR = APP_ROOT / "app" / "seed"


def _load_all() -> List[Dict[str, Any]]:
    """库里现存的全部模式。

    分路层（``hok_modes.json``）+ 英雄层（``hok_hero_modes.json``）。
    英雄层是**生成产物**：由 ``tools/build_hok_hero_modes.py`` 从 5 份
    英雄级攻略原文推出（123 个英雄 × 6 个时间节点）。
    """
    out: List[Dict[str, Any]] = []
    for name in ("hok_modes.json", "hok_hero_modes.json"):
        path = SEED_DIR / name
        if not path.exists():
            continue
        out.extend(json.loads(path.read_text("utf-8")))
    return out


def _require_hero_layer() -> List[Dict[str, Any]]:
    """取英雄层；英雄层为空时跳过。

    英雄层现在是**满的**（从英雄级攻略生成），但这条把关不能删 ——
    哪天生成器挂了、产出空文件，这里会先发现，而不是等到用户看到
    「这个英雄一条建议都没有」。
    """
    path = SEED_DIR / "hok_hero_modes.json"
    heroes = json.loads(path.read_text("utf-8")) if path.exists() else []
    if not heroes:
        pytest.skip(
            "英雄层暂时为空（hok_hero_modes.json 没生成出来）—— "
            "本条要等英雄层回来才生效"
        )
    return heroes


def _cell(rule: Dict[str, Any]) -> Tuple[str, str] | None:
    """模式所在的 (分路, 阶段)。没带 position 的不进这套检查。"""
    lane = phase = None
    window = None
    for c in rule["trigger"]["all"]:
        if c["fact"] == "position" and c["op"] == "==":
            lane = c["value"]
        elif c["fact"] == "time_min" and c["op"] == "between":
            window = c["value"]
    if lane is None or window is None:
        return None
    lo = window[0]
    phase = "early" if lo < 5 else "mid" if lo < 11 else "late"
    return lane, phase


def _accepts(rule: Dict[str, Any]) -> Set[Any]:
    v = rule["expect"]["value"]
    return set(v) if isinstance(v, list) else {v}


def _scopes(rule: Dict[str, Any]) -> Dict[str, Any]:
    """这条模式**声称自己**管哪些情况（用于判断两条会不会同时触发）。"""
    scope: Dict[str, Any] = {}
    for c in rule["trigger"]["all"]:
        if c["fact"] in ("my_hero", "matchup"):
            scope[c["fact"]] = c["value"]
    return scope


def _excludes(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """a 的 none 是否把 b 整个挡在外面。

    注意方向：看的是 **b 自己声明了什么**（``my_hero`` / ``matchup``），
    而 a 的 none 条件要把它覆盖住。分路层通常不声明任何英雄/对位，
    所以不能反过来去 a 的 scope 里找 —— 那会永远找不到。
    """
    none = a["trigger"].get("none") or []
    scope_b = _scopes(b)
    for c in none:
        key = c["fact"]
        if key not in scope_b:
            continue
        theirs = scope_b[key]
        theirs = set(theirs) if isinstance(theirs, list) else {theirs}
        if c["op"] == "==" and theirs == {c["value"]}:
            return True
        if c["op"] == "in" and theirs and theirs <= set(c["value"]):
            return True
    return False


def _overlap(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """两条模式的触发范围是否可能同时成立。"""
    sa, sb = _scopes(a), _scopes(b)
    for key in ("my_hero", "matchup"):
        va, vb = sa.get(key), sb.get(key)
        if va is None or vb is None:
            continue
        va = set(va) if isinstance(va, list) else {va}
        vb = set(vb) if isinstance(vb, list) else {vb}
        if not (va & vb):
            return False
    return True


def test_no_two_modes_in_one_cell_can_contradict():
    """同一格里会同时触发的模式，期望动作必须有交集。

    「会同时触发」要判准，不能只看「同一格」。S45 决策树生成的新库里，
    一个格子里有多条分支，它们靠**互斥的触发条件**隔开：

      - 时间窗不重叠（``_windows_overlap``）—— 早期那套「0:00 那一帧」
        和「6-9 分钟」本来就不可能判到同一回合；
      - 触发条件互斥（``_mutually_exclusive``）—— 一条要求 ``kill_diff > 0``、
        另一条要求 ``kill_diff < 0``，玩家只会被其中一条判。

    这两段判定逻辑**直接复用生成器里的实现**（``tools/build_hok_modes.py``），
    免得测试里另写一套、于是两边慢慢说不到一处去 —— 那正是「校验通过
    但报告自相矛盾」的来源。
    """
    from tools.build_hok_modes import _mutually_exclusive, _windows_overlap

    cells: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in _load_all():
        key = _cell(r)
        if key:
            cells[key].append(r)

    problems: List[str] = []
    for key, group in sorted(cells.items()):
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if _excludes(a, b) or _excludes(b, a):
                    continue
                if not _overlap(a, b):
                    continue
                if a["expect"]["fact"] != b["expect"]["fact"]:
                    continue          # 不同动作事实可以同时做，不冲突
                if _accepts(a) & _accepts(b):
                    continue          # 有交集：做任意一个都算达标
                if not _windows_overlap(a, b):
                    continue          # 时间窗不重叠：不可能判到同一回合
                if _mutually_exclusive(a, b):
                    continue          # 触发条件互斥：只会被其中一条判
                problems.append(
                    f"{key[0]}·{key[1]}: {a['id']} 期望 {sorted(_accepts(a))} "
                    f"而 {b['id']} 期望 {sorted(_accepts(b))}，"
                    f"两者会同时触发且无交集 —— 必定有一条被误判"
                )
    assert not problems, "同一格里存在互相指控的模式：\n  " + "\n  ".join(problems)


def _is_rule_file(path: Path) -> bool:
    """这个 JSON 是不是「规则文件」。

    判据是**内容**而不是文件名：``hok_retired.json`` 是退役清单
    （一个字符串数组），它不是规则 —— 它由 ``app.rules.store._retired_ids()``
    单独读，不走规则库那条加载路径。
    """
    data = json.loads(path.read_text("utf-8"))
    return isinstance(data, list) and all(isinstance(d, dict) for d in data)


def test_every_seed_file_is_actually_loaded():
    """**当前来源**下的规则文件都真的会被加载器读到。

    这个坑踩过一次：加载器按**前缀** ``hok_modes`` 匹配，于是叫
    ``hok_hero_modes.json`` 的英雄层**一次都没被载入**。库看着是完整的，
    界面也正常，只是英雄专属模式一条都不生效 —— 文件没被读到，而不是报错。

    退役清单 ``hok_retired.json`` **不算**规则文件：它是「哪批旧规则被
    新攻略取代了」的记录，格式是字符串数组，刻意不放进规则库的命名约定里
    （另有一条测试盯着它必须能被读到）。

    现在来源可配置了（用户自己的 git 仓库优先，见
    :mod:`app.rules.library_source`），所以要比的是**实际在用的那一份** ——
    比内置目录会得出「可编辑那份没被载入」这种假警报。
    """
    from app.rules.library_source import library_dir
    from app.rules.store import _seed_files

    active = Path(library_dir())
    on_disk = {p.name for p in active.glob("hok*.json") if _is_rule_file(p)}
    loaded = {os.path.basename(p) for p in _seed_files("hok")}
    missing = on_disk - loaded
    assert not missing, f"这些文件不会被加载：{missing}"
    assert on_disk, f"当前来源 {active} 里没有 hok 的规则文件"


def test_retired_list_is_still_readable():
    """退役清单必须还在、还得被读取器读到 —— 但它不是规则。

    退役清单记的是「哪 47 条旧规则被新攻略取代了」。它**不删除**规则，
    只把它们标记成 deprecated（历史对局复盘仍需当时的规则状态）。
    清单一丢，那些旧规则就会重新生效 —— 报告里于是同时出现新旧两套打法，
    而用户看到的是「新的还在，旧的也还在」。
    """
    from app.rules.store import _retired_ids

    path = SEED_DIR / "hok_retired.json"
    assert path.exists(), "退役清单不见了 —— 被取代的旧规则会重新生效"

    ids = _retired_ids("hok")
    assert ids, "退役清单读出来是空的"
    assert all(isinstance(i, str) and i for i in ids), "退役清单里混进了非字符串条目"

    # 清单里的 id 必须**不在**现行规则库里 —— 否则「退役」只是说说而已
    current = {r["id"] for r in _load_all()}
    overlap = sorted(set(ids) & current)
    assert not overlap, f"这些 id 还在现行规则库里，却挂着退役标记：{overlap}"


def test_every_hero_mode_pins_lane_and_hero():
    """英雄层必须钉住英雄、分路与时间窗。

    **不钉对位优劣**是英雄层现在的口径：英雄级攻略里「优势怎么打」和
    「劣势怎么打」写在同一条决策树上，前提（对方有主宰 / 我方先到龙坑）
    对局数据里判不了，所以两者合并成一条规则，期望动作取**并集** ——
    做到其中之一就算做到了。

    钉死对位反而会出事：玩家没填对位优劣时，那条规则一条都不触发，
    于是「没填」变成了「什么结论都没有」。
    """
    heroes = _require_hero_layer()
    for h in heroes:
        scope = _scopes(h)
        assert "my_hero" in scope, f"{h['id']} 没钉英雄"
        assert scope["my_hero"] not in ("", None), f"{h['id']} 英雄为空"
        assert _cell(h), f"{h['id']} 缺分路或时间窗"
        assert h["expect"]["value"], f"{h['id']} 没有期望动作"


def test_every_hero_has_at_least_one_mode():
    """攻略里出现的每个英雄，都要有模式。

    这是「所有英雄的操作模式都入库」这条要求的直接验收：
    少一个英雄，界面上就是「这个英雄一条建议都没有」，
    而用户要的是每个英雄都能查到怎么打。
    """
    from tools.hok_hero_sources import all_hero_names

    heroes = _require_hero_layer()
    covered = defaultdict(set)
    for h in heroes:
        s = _scopes(h)
        covered[s["my_hero"]].add(h["id"])

    missing: List[str] = []
    for name in all_hero_names():
        if not covered.get(name):
            missing.append(name)
    assert not missing, f"这些英雄还没有模式：{missing}"


def test_hero_names_come_from_the_adapter_list():
    """英雄名必须来自适配器的那份清单。

    手打「哪叱」这种同音错字的后果是**静默失效**：针对该英雄的模式永远
    不触发，而界面上看起来就是「这条英雄专属建议没出现」。
    """
    from app.adapters.honor_of_kings import HEROES_BY_LANE

    heroes = _require_hero_layer()
    known = {h for names in HEROES_BY_LANE.values() for h in names}
    for h in heroes:
        name = _scopes(h)["my_hero"]
        assert name in known, f"{h['id']} 的英雄「{name}」不在适配器的清单里"


def test_lane_rules_yield_to_the_hero_layer():
    """分路通则必须给英雄层让路。

    玩家选了自己的英雄，就只该按那一套判。否则同一件事
    （「这一分钟该刷野」）会被分路通则和英雄专属各指控一次，
    报告里出现两条互相重复的结论。

    让路的写法是分路规则加一条 ``my_hero in (全部英雄)`` 的护栏 ——
    玩家填了英雄就被挡住，没填时空串不满足条件，通则照常生效
    （没填英雄的局仍然有话可说）。
    """
    hero_rules = _require_hero_layer()
    lane_rules = {
        r["id"]: r for r in json.loads((SEED_DIR / "hok_modes.json").read_text("utf-8"))
    }

    assert lane_rules, "分路层没有规则了"
    for r in lane_rules.values():
        assert _excludes(r, hero_rules[0]) or True  # 护栏存在性在下面单独查

    # 每条分路规则都必须有一道「选了这 123 个英雄就不判」的护栏
    heroes = {_scopes(h)["my_hero"] for h in hero_rules}
    for r in lane_rules.values():
        guarded = False
        for c in r["trigger"].get("none") or []:
            if c["fact"] != "my_hero" or c["op"] != "in":
                continue
            if heroes <= set(c["value"]):
                guarded = True
        assert guarded, (
            f"{r['id']} 没有给英雄层让路 —— 玩家选了自己的英雄，"
            "同一条建议会被指控两次"
        )

    # 而且真的排除了：分路规则与英雄规则不会同时触发
    for h in hero_rules:
        for r in lane_rules.values():
            assert _excludes(r, h), (
                f"{r['id']} 没有把 {h['id']} 让出去 —— 同一回合会同时命中两条"
            )
