"""操作模式库与攻略原文的**一致性**测试。

这一组测试存在的理由：操作模式库是**照着一份攻略写的**，那这条约束就得
被机器检查，而不是靠「我记得是照着写的」。

它挡住三类悄悄跑偏的情况：

1. **引用被改写或编造** —— 报告里的 `quoted` 不是原文了，
   用户按这个去改操作会被带偏，而且完全看不出来。
2. **同一格子里两条模式互相指控** —— 两条模式挂在同一个「分路 × 阶段」上、
   要求的是**同一个动作事实**但取值不重叠时，玩家一回合只能做一个动作，
   于是必定有一条判成偏离，而玩家其实什么都没做错。
3. **分类出现空格** —— 某个「分路 × 阶段」格子没有模式，
   报告上就少一段，而缺的那段看起来和「这段做得好」长得一样。
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
from app.adapters.base import get_adapter
from app.adapters.honor_of_kings import (
    PHASE_CODES,
    PHASE_LABELS,
    PHASE_TIME_WINDOWS,
    POSITION_ORDER,
    POSITION_VALUES,
    normalize_position,
)

APP_ROOT = Path(__file__).resolve().parent.parent
MODES_PATH = APP_ROOT / "app" / "seed" / "hok_modes.json"
#: 攻略原文（仓库内唯一真源）。操作模式库由它生成，引用必须逐字来自它。
SOURCE_PATH = APP_ROOT / "app" / "seed" / "sources" / "hok_s45_lane_decision_tree.txt"
#: 退役清单：来自旧一版攻略、已被新攻略取代的规则 id。
RETIRED_PATH = APP_ROOT / "app" / "seed" / "hok_retired.json"

#: 需要有模式的格子。攻略分了三条时间窗，库必须三段都有。
EXPECTED_CELLS = [
    (lane, phase)
    for lane in POSITION_ORDER
    for phase in ("early", "mid", "late")
]


def _strip_md(text: str) -> str:
    """去掉排版字符，只留字符本身。

    攻略正文里强调用的是 ``**加粗**``，决策树还带 ``│ ├ └ ─`` 这些
    树形绘图符号 —— 它们不是空白，只去空白的话，两行「动作」之间
    夹着的 ``│`` 就会让逐字比对失败。

    中文正文里数字两侧还有排版空格（「升 4 级」「5 分钟后」），
    那些也不是引用的一部分。所以两边都压成「只剩字符」再比对。
    """
    from tools.hok_decision_tree import normalize_for_compare

    return normalize_for_compare(text)


@pytest.fixture(scope="module")
def source_text() -> str:
    assert SOURCE_PATH.exists(), (
        f"操作模式的原文不在仓库里：{SOURCE_PATH}。"
        "引用无法校验，等于没有溯源"
    )
    return _strip_md(SOURCE_PATH.read_text("utf-8"))


@pytest.fixture(scope="module")
def modes() -> List[Dict[str, Any]]:
    return json.loads(MODES_PATH.read_text("utf-8"))


def _trigger(m: Dict[str, Any], fact: str) -> List[Dict[str, Any]]:
    return [c for c in m["trigger"]["all"] if c["fact"] == fact]


def _lane_of(m: Dict[str, Any]) -> str:
    conds = _trigger(m, "position")
    assert conds, f"{m['id']} 没有 position 条件"
    value = conds[0]["value"]
    values = value if isinstance(value, list) else [value]
    real = [v for v in values if v != "unknown"]
    assert len(real) == 1, f"{m['id']} 的 position 必须指向单一分路：{values}"
    return real[0]


def _window_of(m: Dict[str, Any]) -> List[int]:
    conds = _trigger(m, "time_min")
    assert conds and conds[0]["op"] == "between", f"{m['id']} 没有时间窗"
    return list(conds[0]["value"])


def _phase_of_window(window: List[int]) -> str:
    lo = window[0]
    if lo < 5:
        return "early"
    if lo < 11:
        return "mid"
    if lo < 25:
        return "late"
    return "endgame"


# ---------------------------------------------------------------- 溯源


def test_every_quote_is_verbatim_from_the_guide(modes, source_text):
    """引用必须是原文的**逐字片段**。

    报告里把攻略结论转述成自己的话，是最容易悄悄发生的失真 ——
    「清完自己野区升 4 级 > 任何一次 gank」被转成「前期要刷野」之后，
    用户就失去了判断这条结论该不该采信的依据。
    """
    for m in modes:
        quoted = m["source"]["quoted"]
        assert quoted.strip(), f"{m['id']} 缺引用"
        assert _strip_md(quoted) in source_text, (
            f"{m['id']} 的引用在攻略原文里找不到：{quoted!r}\n"
            "要么抄错了，要么是编的 —— 两种都意味着用户没法复核这条模式"
        )


def test_every_mode_declares_its_source_section(modes):
    for m in modes:
        src = m["source"]
        assert src["guide"].strip(), f"{m['id']} 缺攻略名"
        assert src["locator"].strip(), f"{m['id']} 缺章节定位"
        assert src["collected_at"].strip(), f"{m['id']} 缺采集日期"


def test_five_lanes_only(modes):
    """只有五条分路，游走与辅助是同一个位置。"""
    lanes = {_lane_of(m) for m in modes}
    assert lanes == set(POSITION_ORDER), f"分路集合不对：{lanes}"
    for lane in lanes:
        assert lane in POSITION_VALUES
        assert normalize_position(lane) == lane


def test_helper_alias_maps_into_a_lane():
    """「辅助」必须能归到游走 —— 写错的表现是规则全部不触发。"""
    assert normalize_position("辅助") == "roam"
    assert normalize_position("游走") == "roam"
    assert normalize_position("打野") == "jungle"
    assert normalize_position("发育路") == "carry"
    assert normalize_position("边路") == "side"


# ---------------------------------------------------------------- 分类完整性


def test_every_lane_has_all_three_phases(modes):
    """五条分路 × 前中后期，一个格子都不能空。"""
    cells = {(_lane_of(m), _phase_of_window(_window_of(m))) for m in modes}
    missing = [f"{lane}·{PHASE_LABELS[p]}" for lane, p in EXPECTED_CELLS
               if (lane, p) not in cells]
    assert not missing, f"这些格子没有模式：{missing}"


def test_phase_windows_are_disjoint_and_cover_the_match(modes):
    """相位窗口不许重叠，且必须**不重不漏**地盖住整局。

    ``between`` 是闭区间，早先按 [0,4] / [4,10] 写，第 4 分钟会同时命中
    「前期要刷野」和「中期要控龙」，两条必定有一条判偏离。

    「不漏」同样要紧：窗口之间留一道缝，那些分钟不属于任何阶段，
    模式一条都不触发 —— 而界面上看不出那是「没命中」还是「本来就没规则」。
    """
    windows = [(tuple(v), k) for k, v in PHASE_TIME_WINDOWS.items()]
    ordered = sorted(windows)
    assert ordered[0][0][0] == 0, "第一个窗口没从 0 分钟开始"

    for (a_lo, a_hi), a in ordered:
        assert a_lo <= a_hi, f"{a} 窗口反了"
        for (b_lo, b_hi), b in ordered:
            if a >= b:
                continue
            assert min(a_hi, b_hi) < max(a_lo, b_lo), (
                f"相位窗口 {a}{[a_lo, a_hi]} 与 {b}{[b_lo, b_hi]} 重叠"
            )

    # 逐段检查既不重叠也不漏
    for i in range(len(ordered) - 1):
        hi, next_lo = ordered[i][0][1], ordered[i + 1][0][0]
        assert next_lo == hi + 1, (
            f"{ordered[i][1]} 止于 {hi}，{ordered[i + 1][1]} 起于 {next_lo}，"
            f"第 {hi + 1} 分钟不属于任何阶段"
        )

    assert ordered[-1][0][1] >= 60, "最后一个窗口没盖到 60 分钟"

    # 攻略只给了三段时间窗，多切一刀不会更准
    assert set(PHASE_TIME_WINDOWS) == {"early", "mid", "late"}


def test_mode_windows_match_the_phase_they_claim(modes):
    """模式分类里写的分路与阶段，必须和它的触发条件对得上。

    分类名是给人看的，但它一旦和触发条件脱节，报告就会把一条 8 分钟才生效的
    规则挂在「前期」下面 —— 用户照着报告改，改的是另一段时间的事。
    """
    labels = get_adapter("hok").position_labels
    for m in modes:
        lane, window = _lane_of(m), _window_of(m)
        phase = _phase_of_window(window)
        lane_label = labels[lane]
        phase_label = PHASE_LABELS[phase]
        assert m["category"].startswith(lane_label), (
            f"{m['id']} 分类 {m['category']} 的分路与触发条件（{lane_label}）不符"
        )
        assert phase_label in m["category"], (
            f"{m['id']} 分类 {m['category']} 的阶段与时间窗 {window} "
            f"推出的阶段（{phase_label}）不符"
        )


def test_expected_actions_are_registered(modes):
    """期望动作必须在适配器的动作词表里，否则永远判不出来。"""
    labels = get_adapter("hok").action_labels
    for m in modes:
        exp = m["expect"]
        assert exp["fact"] in get_adapter("hok").known_action_facts(), \
            f"{m['id']} 的 expect 不是动作事实：{exp['fact']}"
        values = exp["value"] if isinstance(exp["value"], list) else [exp["value"]]
        for v in values:
            assert v in labels, f"{m['id']} 用了未注册的动作：{v}"


# ------------------------------------------------- 同一格子里不能互相指控


def test_same_cell_modes_never_contradict_each_other(modes):
    """同一「分路 × 阶段」里，两条模式若要求**同一个动作事实**且取值不重叠，
    玩家必然被误判。

    判定口径是「这一回合做过符合要求的动作就算符合」，所以两条模式期望的
    取值必须**有交集** —— 有交集时玩家做了其中一件，两条都算做到。

    要求不同动作事实（比如一个要 positioning、一个要 teamfight）则互不干扰，
    因为玩家一回合可以同时做。

    两条例外
    --------
    1. **时间窗不重叠**：窗口不重叠时，两条规则不可能判到同一回合，
       玩家不会被两条同时指控。
    2. **互斥**：两条规则的触发条件互相排斥时（例如一条要求
       ``kill_diff > 0``、另一条要求 ``kill_diff < 0``），
       玩家只会被其中一条判，也不算互相指控。
    """
    from tools.build_hok_modes import _mutually_exclusive, _windows_overlap

    by_cell: Dict[tuple, List[Dict[str, Any]]] = {}
    for m in modes:
        key = (_lane_of(m), _phase_of_window(_window_of(m)))
        by_cell.setdefault(key, []).append(m)

    for (lane, phase), group in by_cell.items():
        by_fact: Dict[str, List[Dict[str, Any]]] = {}
        for m in group:
            by_fact.setdefault(m["expect"]["fact"], []).append(m)
        for fact, same in by_fact.items():
            for i in range(len(same)):
                for j in range(i + 1, len(same)):
                    a, b = same[i], same[j]
                    va = set(a["expect"]["value"] if isinstance(a["expect"]["value"], list)
                             else [a["expect"]["value"]])
                    vb = set(b["expect"]["value"] if isinstance(b["expect"]["value"], list)
                             else [b["expect"]["value"]])
                    if a["expect"]["op"] == "==" and b["expect"]["op"] == "==":
                        assert va == vb, (
                            f"{lane}·{PHASE_LABELS[phase]} 里 {a['id']} 与 {b['id']} "
                            f"要求同一个动作事实 {fact} 但取值不同，玩家会两条都违背"
                        )
                        continue
                    if a["expect"]["op"] == "==" or b["expect"]["op"] == "==":
                        eq, other = (a, b) if a["expect"]["op"] == "==" else (b, a)
                        ov = other["expect"]["value"]
                        ov = set(ov if isinstance(ov, list) else [ov])
                        assert eq["expect"]["value"] in ov, (
                            f"{lane}·{PHASE_LABELS[phase]} 里 {eq['id']} 与 "
                            f"{other['id']} 在动作事实 {fact} 上互斥"
                        )
                        continue
                    if va & vb:
                        continue
                    if not _windows_overlap(a, b):
                        continue
                    assert _mutually_exclusive(a, b), (
                        f"{lane}·{PHASE_LABELS[phase]} 里 {a['id']} 与 {b['id']} "
                        f"在动作事实 {fact} 上没有任何交集，必定有一条被误判"
                    )


def test_action_fact_guards_never_collide_with_a_sibling_expectation(modes):
    """同一格里，**动作事实**上的 none 护栏不能与另一条模式的期望撞车。

    期望判定是「这一回合做过符合要求的动作就算符合」（看全部同类动作），
    而触发条件看的是镜像出来的**最后一个**动作。两边口径不同，于是
    「none 护栏 + 另一条期望同一动作事实」这个组合会出事：

        规则 A  期望 objective_action ∈ {take_dragon}
        规则 B  none 护栏 objective_action != take_dragon

    玩家这一回合做了 [take_dragon, farm_jungle] 时 ——
    A 判「做到了」（take_dragon 做过），B 也触发（最后一个动作是 farm_jungle，
    不等于 take_dragon）。两条同时下结论，玩家被指控两次，而其实没做错。

    所以这个组合在库里直接禁掉。护栏写在 matchup 这类**标量**事实上不受影响，
    那种两边口径一致。
    """
    action_facts = set(get_adapter("hok").action_fact_names)

    by_cell: Dict[tuple, List[Dict[str, Any]]] = {}
    for m in modes:
        key = (_lane_of(m), _phase_of_window(_window_of(m)))
        by_cell.setdefault(key, []).append(m)

    for (lane, phase), group in by_cell.items():
        guarded: Dict[str, List[str]] = {}
        expecting: Dict[str, List[str]] = {}
        for m in group:
            for c in (m["trigger"].get("none") or []):
                if c["fact"] in action_facts:
                    guarded.setdefault(c["fact"], []).append(m["id"])
            if m["expect"]["fact"] in action_facts:
                expecting.setdefault(m["expect"]["fact"], []).append(m["id"])
        for fact in set(guarded) & set(expecting):
            assert not set(guarded[fact]) & set(expecting[fact]), (
                f"{lane}·{PHASE_LABELS[phase]} 里 {guarded[fact]} 与 "
                f"{expecting[fact]} 在动作事实 {fact} 上既写护栏又期望，"
                "玩家会被两条同时指控"
            )


def test_every_expected_action_is_selectable_in_the_action_options(modes):
    """每条模式期望的每个动作，都得在**这个游戏的动作词表**里。

    选不到就意味着这条模式**永远判不出「做到了」** —— 用户明明做了，
    报告却说「没做」，而词表里压根没那个值。

    早先用共享词表（自走棋时代的 take_baron / push_tower…），
    于是「清视野精灵」「补刀」这些 S45 才有的动作压根列不出来。

    标注页下线后这份词表由**模型读画面**时用：模型报回来的动作值
    必须落在同一份词表里，否则内核认不出来。
    """
    from app.collection.video_api import _action_options

    adapter = get_adapter("hok")
    options = _action_options(adapter)
    for kind, opts in options.items():
        values = {o["value"] for o in opts}
        assert values, f"{kind} 一个可选动作都没有"

    for m in modes:
        fact = m["expect"]["fact"]
        assert fact in options, f"{m['id']} 期望的事实 {fact} 不在动作词表里"
        values = {o["value"] for o in options[fact]}
        want = m["expect"]["value"]
        want = want if isinstance(want, list) else [want]
        for v in want:
            assert v in values, (
                f"{m['id']} 期望的动作 {v} 在词表里选不到 —— "
                "这条模式永远判不出「做到了」"
            )


def test_the_action_options_offer_no_actions_from_another_game(modes):
    """动作词表不能出现**这个游戏没有**的动作。

    共享词表里那些自走棋动作（take_baron / push_tower / adjust…）对王者
    没有意义：不对应任何一条王者模式，只是噪音；而且显示成英文
    原始码，玩家根本不知道那是什么。
    """
    from app.adapters.honor_of_kings import MOBA_ACTION_VALUES
    from app.collection.video_api import _action_options

    adapter = get_adapter("hok")
    options = _action_options(adapter)
    expected = {kind: set(values) for kind, values in MOBA_ACTION_VALUES.items()}

    assert set(options) == set(expected), (
        f"动作类别对不上：{sorted(options)} vs {sorted(expected)}"
    )
    for kind, opts in options.items():
        assert {o["value"] for o in opts} == expected[kind], (
            f"{kind} 的可选动作与王者词表不一致"
        )
        for o in opts:
            assert o["label"] and o["label"] != o["value"], (
                f"{kind} 里的 {o['value']} 没有中文名 —— "
                "说明它来自别的游戏的词表"
            )


def test_mode_ids_are_unique_and_prefixed(modes):
    ids = [m["id"] for m in modes]
    assert len(ids) == len(set(ids)), "模式 id 有重复"
    for mid in ids:
        assert mid.startswith("HOK-MODE-"), mid


def test_every_mode_says_how_to_fix_it(modes):
    """「应该怎么做」是这份报告的主要产出，不能是空的。"""
    for m in modes:
        assert m["deviation_hint"].strip(), f"{m['id']} 缺改法"
        assert m["rationale"].strip(), f"{m['id']} 缺依据"
        assert len(m["rationale"]) > 40, f"{m['id']} 的依据太短，多半是敷衍"


# ---------------------------------------------------------------- 同步 / 退役


def test_seed_library_is_exactly_what_the_generator_produces(modes):
    """种子库必须**逐条等于**生成器输出，禁止手改库注水。

    库的体积下限可以靠"多写几条"刷上去，但那些规则没有溯源、没有互斥论证。
    所以这里不比较条数，而是把生成器重新跑一遍，要求与落盘的 JSON 完全一致：
    手改一条，测试就红。
    """
    from tools.build_hok_modes import (
        GUIDE_PATH,
        build_rules,
        load_guide,
        load_specs,
    )

    tree = load_guide(GUIDE_PATH)
    rebuilt = build_rules(tree, load_specs(None))
    assert rebuilt == modes, (
        "app/seed/hok_modes.json 与生成器输出不一致 —— "
        "种子库是生成产物，不要手改；"
        "要改规则请改 app/seed/sources/ 里的攻略原文或 tools/hok_spec/ 里的映射表，"
        "再跑 tools/build_hok_modes.py 重新生成"
    )


def test_retired_list_is_disjoint_from_active_library(modes):
    """退役清单不能和新库撞 id，否则历史规则会被误标。"""
    retired = json.loads(RETIRED_PATH.read_text("utf-8"))
    assert isinstance(retired, list) and retired, "退役清单为空"
    active_ids = {m["id"] for m in modes}
    overlap = active_ids & set(retired)
    assert not overlap, f"退役清单与在用规则撞 id：{sorted(overlap)}"
    for rid in retired:
        assert isinstance(rid, str) and rid.strip(), f"退役清单里有非字符串：{rid!r}"
        assert rid not in active_ids, f"{rid} 同时在用和退役"


def test_retired_rules_are_marked_instead_of_deleted():
    """旧规则只能退役（deprecated），不能从库里消失。

    删掉的话，已经复盘过的老对局会突然"没查过这些规则"，
    历史结论就断了。

    退役清单跨两个库：局内的（``S45``）是**旧一版攻略里的同名条目** ——
    它们还得留在库里，只是标成失效；BP 的（``BP-S45``）是**整条撤下**
    （依赖段位与预选，页面已经不问这两项了），所以新装环境里根本不存在，
    只要求「不得仍在生效」。
    """
    from app import db as dbmod
    from app.rules.store import (
        get_rules,
        load_bp_seed_library,
        load_seed_library,
    )

    retired = json.loads(RETIRED_PATH.read_text("utf-8"))
    in_game = [rid for rid in retired if not rid.startswith("HOK-BP-")]
    bp = [rid for rid in retired if rid.startswith("HOK-BP-")]

    session = dbmod.get_session_factory()()
    try:
        load_seed_library(session, "hok")
        load_bp_seed_library(session, "hok")

        # ---- 局内：必须还在库里，且是失效状态 ----
        rules = get_rules(session, "hok", "S45", include_deprecated=True)
        by_id = {r.id: r for r in rules}
        missing = [rid for rid in in_game if rid not in by_id]
        assert not missing, (
            f"退役清单里的规则在库里找不到：{missing[:5]} —— "
            "退役是靠标 deprecated 实现的，历史规则必须还在库里"
        )
        for rid in in_game:
            assert by_id[rid].status == "deprecated", (
                f"{rid} 在退役清单里，但库里状态是 {by_id[rid].status}"
            )
        active = get_rules(session, "hok", "S45")
        assert not (set(in_game) & {r.id for r in active}), (
            "退役规则仍出现在在用库里"
        )

        # ---- BP：撤下的规则不得仍在生效 ----
        bp_records = get_rules(session, "hok", "BP-S45",
                               include_deprecated=True)
        bp_status = {r.id: r.status for r in bp_records}
        bp_active = {r.id for r in get_rules(session, "hok", "BP-S45")}
        for rid in bp:
            assert rid not in bp_active, f"{rid} 撤下了却仍在生效"
            # 老库里还留着的那条，必须已经被标成失效
            if rid in bp_status:
                assert bp_status[rid] == "deprecated", (
                    f"{rid} 已撤下，但老库里还是 {bp_status[rid]}"
                )
    finally:
        session.close()
