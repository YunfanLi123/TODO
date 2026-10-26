"""英雄层操作模式库与**英雄级攻略原文**的一致性测试。

分路层那 84 条的一致性由 ``test_mode_library.py`` 守着；这一组守的是
英雄层的同一套约束，源码换了地方：

    app/seed/sources/hok_s45_hero_guides/*.txt   5 份英雄攻略（唯一真源）
        ↓ tools/hok_hero_tree.py + tools/hok_hero_actions.py
    app/seed/hok_hero_modes.json                 1596 条英雄专属模式

它挡住四类悄悄跑偏：

1. **原文被改动** —— sha256 对不上，说明引用可能已经不是原文了；
2. **引用被改写或编造** —— ``quoted`` 不是原文的字；
3. **种子库被手改** —— 库里多一条没有溯源的规则；
4. **英雄名打错** —— 「哪叱」这种同音错字会让那条模式**静默失效**：
   界面上看起来就是「这个英雄的建议没出现」，没有任何报错。
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.adapters import registry as _registry  # noqa: F401  触发适配器注册
from app.adapters.base import get_adapter

APP_ROOT = Path(__file__).resolve().parent.parent
SEED_DIR = APP_ROOT / "app" / "seed"
HERO_MODES_PATH = SEED_DIR / "hok_hero_modes.json"
HERO_SKIPPED_PATH = SEED_DIR / "hok_hero_skipped.json"
HEROES_PATH = SEED_DIR / "hok_heroes.json"
SOURCE_DIR = SEED_DIR / "sources" / "hok_s45_hero_guides"


@pytest.fixture(scope="module")
def hero_modes() -> List[Dict[str, Any]]:
    return json.loads(HERO_MODES_PATH.read_text("utf-8"))


@pytest.fixture(scope="module")
def skipped() -> Dict[str, Any]:
    return json.loads(HERO_SKIPPED_PATH.read_text("utf-8"))


# ---------------------------------------------------------------- 原文


def test_hero_guides_are_byte_identical_to_the_manifest():
    """5 份英雄攻略必须与入库时一字不差。

    引用核对的前提是「原文没被动过」。sha256 对不上，说明要么原文被改了
    （那引用可能已经过期），要么文件被换过（那库里记的出处就是假的）。
    """
    from tools.hok_hero_sources import verify_sources

    problems = verify_sources()
    assert not problems, "英雄攻略原文被改动：\n  " + "\n  ".join(problems)


def test_repo_copies_match_the_original_uploaded_assets():
    """仓库里的原文副本必须与**用户上传的原件**逐字节一致。

    清单里的 sha256 是入库当时写下的，仓库自己跟自己比只能证明
    「入库之后没人动过」。真正要堵的是入库**那一刻**取错文件 ——
    比如路径拼错、拷了半截、拷成了别的英雄那份。

    所以这里把原件从附件目录拉回来再比一次 sha256。
    附件目录会被清理，所以原件不在时**跳过**而不是报错：
    仓库里的副本才是长期真源，原件只是用来复核「当初拿的是不是它」。
    """
    from tools.import_hok_hero_guides import SOURCES, sha256_of

    present = [(src, lane, label) for src, lane, label in SOURCES if os.path.exists(src)]
    if not present:
        pytest.skip("用户上传的原件已不在附件目录，无法复核入库时取的是不是同一份")

    mismatched: List[str] = []
    for src, lane, label in present:
        dst = os.path.join(str(SOURCE_DIR), os.path.basename(src))
        if not os.path.exists(dst):
            mismatched.append(f"{label}：仓库里没有副本 {os.path.basename(dst)}")
            continue
        if sha256_of(src) != sha256_of(dst):
            mismatched.append(f"{label}：副本与原件不一致 {os.path.basename(src)}")

    assert not mismatched, "入库的原文不是用户上传的那份：\n  " + "\n  ".join(mismatched)


def test_every_hero_guide_is_parsed():
    """5 份攻略都要能解析出英雄块，且覆盖到 5 条分路。"""
    from tools.hok_hero_sources import hero_guide_paths
    from tools.hok_hero_tree import load_hero_guide

    guides = hero_guide_paths()
    assert len(guides) == 5
    heroes: List[str] = []
    for lane, label, path in guides:
        guide = load_hero_guide(path, lane)
        assert guide.heroes, f"{label} 那份攻略一个英雄都没解析出来"
        for hb in guide.heroes:
            assert hb.nodes, f"{lane}/{hb.hero} 没有时间节点"
            heroes.append(hb.hero)
    # 123 个去重英雄（有些英雄在多条分路里都有）
    assert len(set(heroes)) == 123
    assert len(heroes) == 147


def test_raw_text_scan_reconciles_with_the_parser():
    """用**另一套实现**直接扫原文，和解析器逐份对账。

    上面那个测试把「882 节点 / 2910 分支 / 147 英雄块」钉死了，但那些数字
    是解析器的**输出**。解析器漏读一条，数字跟着变小，测试期望值也跟着
    变小 —— 测试照样绿，等于没测。

    这里用一套独立代码扫原文（``tools/hok_source_coverage.py``），
    同一个事实数两遍，两边必须完全一致。
    """
    from tools.hok_source_coverage import parser_scan, raw_scan
    from tools.hok_hero_sources import hero_guide_paths

    problems: List[str] = []
    tot_raw = [0, 0, 0]
    tot_parser = [0, 0, 0]
    for lane, label, path in hero_guide_paths():
        r_heroes, r_nodes, r_branches = raw_scan(path)
        p_heroes, p_nodes, p_branches = parser_scan(path, lane)
        tot_raw[0] += len(r_heroes)
        tot_raw[1] += r_nodes
        tot_raw[2] += r_branches
        tot_parser[0] += len(p_heroes)
        tot_parser[1] += p_nodes
        tot_parser[2] += p_branches
        if r_nodes != p_nodes:
            problems.append(f"{label}：节点数 原文扫描 {r_nodes} vs 解析器 {p_nodes}")
        if r_branches != p_branches:
            problems.append(f"{label}：分支数 原文扫描 {r_branches} vs 解析器 {p_branches}")
        if set(r_heroes) != set(p_heroes):
            only_raw = sorted(set(r_heroes) - set(p_heroes))
            only_parser = sorted(set(p_heroes) - set(r_heroes))
            problems.append(f"{label}：英雄名不一致 原文多 {only_raw} / 解析器多 {only_parser}")

    assert not problems, "原文扫描与解析器对不上（解析器可能漏读）：\n  " + "\n  ".join(problems)
    assert tot_raw == tot_parser, (tot_raw, tot_parser)
    assert tot_raw == [147, 882, 2910], f"合计对不上：{tot_raw}"


# ---------------------------------------------------------------- 同步


def test_seed_hero_library_is_exactly_what_the_generator_produces(hero_modes):
    """种子库必须**逐条等于**生成器输出，禁止手改库注水。"""
    from tools.build_hok_hero_modes import build_rules

    rebuilt, _skipped = build_rules()
    assert rebuilt == hero_modes, (
        "app/seed/hok_hero_modes.json 与生成器输出不一致 —— "
        "种子库是生成产物，不要手改；"
        "要改请改 app/seed/sources/hok_s45_hero_guides/ 里的原文"
        "或 tools/hok_hero_actions.py 里的动作分类表，"
        "再跑 tools/build_hok_hero_modes.py 重新生成"
    )


# ---------------------------------------------------------------- 引用


def test_every_quote_is_verbatim_from_its_guide(hero_modes):
    """引用必须能在**对应那份**攻略里逐字找到。"""
    from tools.build_hok_modes import strip_ws
    from tools.hok_hero_sources import hero_guide_paths

    texts: Dict[str, str] = {}
    for lane, _label, path in hero_guide_paths():
        texts[lane] = strip_ws(Path(path).read_text("utf-8"))

    problems: List[str] = []
    for m in hero_modes:
        lane = m["trigger"]["all"][0]["value"]
        quoted = (m["source"] or {}).get("quoted") or ""
        if not quoted.strip():
            problems.append(f"{m['id']} 缺引用")
        elif strip_ws(quoted) not in texts.get(lane, ""):
            problems.append(f"{m['id']} 的引用在原文里找不到")
    assert not problems, "引用有问题：\n  " + "\n  ".join(problems[:20])


def test_every_rule_carries_premise_action_and_outcome(hero_modes):
    """前提 / 动作 / 后果三个字段都要有值。

    报告的「为什么」要靠它们，少一个字段报告里就少一句话。
    """
    for m in hero_modes:
        assert m["premise"].strip(), f"{m['id']} 缺前提"
        assert m["action"].strip(), f"{m['id']} 缺动作"
        assert m["outcome"].strip(), f"{m['id']} 缺后果"
        assert m["deviation_hint"].strip(), f"{m['id']} 缺改法"


# ---------------------------------------------------------------- 结构


def test_every_rule_pins_hero_lane_and_window(hero_modes):
    """每条规则都要钉住英雄、分路、时间窗。"""
    for m in hero_modes:
        conds = {c["fact"]: c for c in m["trigger"]["all"]}
        assert conds.get("position"), f"{m['id']} 没钉分路"
        assert conds.get("my_hero"), f"{m['id']} 没钉英雄"
        assert conds.get("time_min"), f"{m['id']} 没钉时间窗"
        assert m["id"].startswith("HOK-HERO-"), m["id"]
        assert m["status"] == "active"
        assert m["expect"]["value"], f"{m['id']} 没有期望动作"
        assert m["expect_scope"] == "at_least_once", \
            f"{m['id']} 的满足口径不对：攻略讲的是一段里的要求，不是每帧"


def test_expect_values_are_registered_actions(hero_modes):
    """期望动作必须能在王者动作词表里勾出来，否则标注页勾不出来。"""
    adapter = get_adapter("hok")
    action_facts = adapter.known_action_facts()
    labels = dict(adapter.action_labels)
    for m in hero_modes:
        fact = m["expect"]["fact"]
        assert fact in action_facts, f"{m['id']} 的期望事实不是动作事实：{fact}"
        for v in m["expect"]["value"]:
            assert v in labels, f"{m['id']} 用了未注册的动作：{v}"
        subject = m.get("subject") or {}
        assert subject.get("fact") == fact, \
            f"{m['id']} 的适用时机和期望动作不是同一类事实"
        assert set(m["expect"]["value"]) <= set(subject.get("values") or []), \
            f"{m['id']} 的适用时机比期望动作窄，玩家做到位了也不会被判符合"


def test_hero_names_match_the_guide_exactly(hero_modes):
    """规则里的英雄名必须与攻略里的写法**一字不差**。

    错一个字的后果是**静默失效**：针对那个英雄的模式永远不触发，
    界面上看起来只是「这个英雄的建议没出现」，没有任何报错。
    """
    from tools.hok_hero_sources import hero_guide_paths
    from tools.hok_hero_tree import load_hero_guide

    known: Dict[str, set] = {}
    for lane, _label, path in hero_guide_paths():
        guide = load_hero_guide(path, lane)
        known[lane] = {hb.hero for hb in guide.heroes}

    for m in hero_modes:
        lane = m["trigger"]["all"][0]["value"]
        hero = [c for c in m["trigger"]["all"] if c["fact"] == "my_hero"][0]["value"]
        assert hero in known[lane], f"{m['id']} 的英雄「{hero}」不在{lane}那份攻略里"


def test_every_hero_in_the_guides_has_modes(hero_modes):
    """攻略里出现的每个英雄都要有模式。"""
    from tools.hok_hero_sources import all_hero_names

    covered = {
        c["value"] for m in hero_modes for c in m["trigger"]["all"]
        if c["fact"] == "my_hero"
    }
    missing = [h for h in all_hero_names() if h not in covered]
    assert not missing, f"这些英雄没有模式：{missing}"


def test_every_branch_is_mapped_or_explicitly_skipped(hero_modes, skipped):
    """攻略里的每条分支都要有归宿：映射成规则，或写明不可判定的理由。

    「漏写一条」的后果是攻略里明明写了，库里却查不到 ——
    而没人会知道该去哪查。

    这条是**精确**校验（不是数比例）：每条分支要么能在某条规则的
    前提/动作里逐字找到，要么出现在未入库清单里。两者都不是的话，
    就是分支凭空消失了。
    """
    from tools.hok_hero_sources import hero_guide_paths
    from tools.hok_hero_tree import load_hero_guide

    # 规则按英雄索引：前提与动作是「；」拼起来的，
    # 分支的原文必然是其中一段
    by_hero: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for m in hero_modes:
        hero = [c for c in m["trigger"]["all"] if c["fact"] == "my_hero"][0]["value"]
        by_hero[hero].append(m)

    skipped_keys = {
        (s["hero"], s["premise"], s["action"]) for s in (skipped.get("skipped") or [])
    }

    lost: List[str] = []
    total = 0
    # 库里存的是**规范化之后**的文本（去状态记号、去动作开头的裸时刻、
    # 替换模糊表述），所以比对的两边都要走同一条规范化流水线。
    from tools.hok_wording import apply_fixes, normalize

    for lane, _label, path in hero_guide_paths():
        guide = load_hero_guide(path, lane)
        for hb in guide.heroes:
            for node in hb.nodes:
                for b in node.branches:
                    total += 1
                    premise = b.premise_text
                    action = b.action_text
                    if (hb.hero, premise, action) in skipped_keys:
                        continue
                    premise_n = apply_fixes(normalize(premise))
                    action_n = apply_fixes(
                        normalize(action, strip_clause_time=True)
                    )
                    hit = any(
                        premise_n in (m["premise"] or "")
                        and action_n in (m["action"] or "")
                        for m in by_hero.get(hb.hero, [])
                    )
                    if not hit:
                        lost.append(f"{hb.hero} / {node.header} / {premise[:40]}")

    assert not lost, "这些分支既没有变成规则，也没写明不可入库的理由：\n  " + \
                     "\n  ".join(lost[:20])
    assert total == 2910, f"攻略里的分支数变了（{total}），解析器或原文要出问题"


# ---------------------------------------------------------------- 互斥


def test_no_two_hero_modes_in_one_cell_contradict(hero_modes):
    """同一格（英雄 × 阶段）里会同时触发的模式，期望动作必须有交集。

    判据与生成器里那套**同一份实现**，免得两边慢慢说不到一处去。
    """
    from tools.build_hok_modes import _mutually_exclusive, _windows_overlap

    def cell(m):
        conds = {c["fact"]: c for c in m["trigger"]["all"]}
        lane = conds["position"]["value"]
        hero = conds["my_hero"]["value"]
        lo = conds["time_min"]["value"][0]
        phase = "early" if lo < 5 else "mid" if lo < 11 else "late"
        return lane, hero, phase

    groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for m in hero_modes:
        groups[cell(m)].append(m)

    problems: List[str] = []
    for key, group in groups.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a["expect"]["fact"] != b["expect"]["fact"]:
                    continue
                va = set(a["expect"]["value"])
                vb = set(b["expect"]["value"])
                if va & vb:
                    continue
                if not _windows_overlap(a, b):
                    continue
                if _mutually_exclusive(a, b):
                    continue
                problems.append(
                    f"{key}: {a['id']} 期望 {sorted(va)} 而 {b['id']} 期望 {sorted(vb)}，"
                    "两者会同时触发且无交集 —— 必定有一条被误判"
                )
    assert not problems, "英雄层存在互相指控的模式：\n  " + "\n  ".join(problems[:20])


# ---------------------------------------------------------------- 界面数据


def test_hero_dropdown_covers_every_hero_in_the_library():
    """标注页的英雄下拉必须列全。

    下拉里没有的英雄，玩家就选不了 —— 选了别的英雄，英雄层的模式一条都不触发，
    而报告上只是「没有结论」。
    """
    from app.adapters.honor_of_kings import HEROES_BY_LANE

    data = json.loads(HEROES_PATH.read_text("utf-8"))
    offered = {h for v in HEROES_BY_LANE.values() for h in v}
    for lane, entry in (data.get("lanes") or {}).items():
        for name in entry["heroes"]:
            assert name in offered, f"下拉里没有「{name}」（{lane}）"
    assert len(offered) == 123, f"下拉只有 {len(offered)} 个英雄"
