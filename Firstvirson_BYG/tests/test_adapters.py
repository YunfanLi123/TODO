"""三款游戏适配器与种子规则库的对齐测试。

这个测试文件同时承担两个角色：
1. 契约测试 —— 适配器注册、事实集、规则准入必须全部通过
2. 回归样本 —— 下面的模拟对局是人工构造的「标准考卷」，用来验证规则库
   真的能命中真实形态的数据，而不是只在纸面上成立

模拟数据放在这里而不是 tests/fixtures/，是因为端到端验证脚本需要复用同一份
数据，改一处两边同步，不存在两份样本漂移的风险。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from app.adapters.registry import ADAPTERS, GAME_IDS, REGISTRY, get_adapter
from app.core.attribution import analyze_match
from app.core.context import MatchContext, RoundRecord
from app.core.dsl import (
    DEFAULT_SUBJECTIVE_MARKERS,
    Rule,
    check_admission,
)

SEED_DIR = Path(__file__).resolve().parents[1] / "app" / "seed"

#: 目前填了操作模式库的游戏。云顶 / 金铲铲只保留适配器模块，资料后补 ——
#: 所以依赖「有库」的测试只对王者荣耀跑，见 MODE_GAMES。
MODE_GAMES = ["hok"]

#: 交付要求的事实集。少一个事实就意味着有规则会引用不到事实，多一个事实
#: 则说明它没有任何规则在用 —— 两种情况都要暴露出来。
REQUIRED_AUTECHESS_FACTS = {
    "stage", "phase", "health", "level", "gold", "income", "streak",
    "econ_action", "positioning_action", "item_count", "has_main_carry",
    "is_decision_point", "pressure", "round_tempo",
}
REQUIRED_MOBA_FACTS = {
    "time_min", "phase", "gold", "gold_diff", "level", "tower_diff",
    "kill_diff", "objective_action", "positioning_action",
    "teamfight_action", "has_ult_ready", "is_decision_point", "pressure",
}

#: 种子文件按 ``<game>_modes*.json`` / ``<game>_rules*.json`` 命名，
#: 与 :func:`app.rules.store._seed_files` 同一套约定。
#: 不写死每款游戏的文件名：库还没做的游戏压根没有文件，写死会变成
#: FileNotFoundError，而不是一条能读懂的失败信息。
SEED_FILES = {
    "hok": "hok_modes.json",
}


# --------------------------------------------------------------------------
# 模拟对局数据
# --------------------------------------------------------------------------

#: 云顶对局：故意做出「4-1 连胜期梭哈追三 + 人口没提 + 没调站位」，
#: 以及 5-2 人口落后仍存钱两处偏离，另有一个 econ_action 采集失败的回合。
TFT_RAW: Dict[str, Any] = {
    "match_id": "tft-demo-001",
    "game_version": "13.1",
    "player": {"nickname": "示例玩家", "rank": "大师"},
    "outcome": {"placement": 4, "result": "lose"},
    "rounds": [
        {
            "stage": 1, "phase": 1,
            "facts": {"health": 100, "gold": 12, "streak": 0, "level": 4,
                      "item_count": 0, "has_main_carry": False},
            "actions": [{"kind": "econ_action", "value": "save", "label": "存钱吃利息",
                         "confidence": 0.95}],
            "missing_facts": [],
        },
        {
            # 2-1 攒满 50 利息却去搜牌 —— TFT-ECON-002 命中
            "stage": 2, "phase": 1,
            "facts": {"health": 92, "gold": 50, "streak": 2, "level": 5,
                      "item_count": 0, "has_main_carry": False},
            "actions": [{"kind": "econ_action", "value": "roll", "label": "搜牌",
                         "confidence": 0.88}],
            "missing_facts": [],
        },
        {
            "stage": 3, "phase": 1,
            "facts": {"health": 75, "gold": 14, "streak": -2, "level": 6,
                      "item_count": 1, "has_main_carry": False},
            "actions": [{"kind": "econ_action", "value": "save", "label": "存钱吃利息",
                         "confidence": 0.91}],
            "missing_facts": [],
        },
        {
            # 核心失误回合：4 连胜 + 58 血却梭哈追三，人口仍 6，也没调站位
            "stage": 4, "phase": 1,
            "facts": {"health": 58, "gold": 20, "streak": 4, "level": 6,
                      "item_count": 1, "has_main_carry": True},
            "actions": [{"kind": "econ_action", "value": "all_in", "label": "梭哈追三星",
                         "confidence": 0.92}],
            "missing_facts": [],
        },
        {
            "stage": 4, "phase": 2,
            "facts": {"health": 58, "gold": 35, "streak": 5, "level": 7,
                      "item_count": 2, "has_main_carry": True},
            "actions": [
                {"kind": "econ_action", "value": "save", "label": "存钱吃利息",
                 "confidence": 0.9},
                {"kind": "positioning_action", "value": "shift_lineup",
                 "label": "调整站位", "confidence": 0.82},
            ],
            "missing_facts": [],
        },
        {
            # econ_action 采集失败：该回合判定必须是 undetermined 而不是猜测
            "stage": 4, "phase": 3,
            "facts": {"health": 50, "gold": 30, "streak": 5, "level": 7,
                      "item_count": 2, "has_main_carry": True},
            "actions": [{"kind": "positioning_action", "value": "hold_lineup",
                         "label": "保持站位", "confidence": 0.8}],
            "missing_facts": ["econ_action"],
        },
        {
            # 装备成型 + 28 血 → 追星锁血，识别置信度偏低
            "stage": 5, "phase": 1,
            "facts": {"health": 28, "gold": 30, "streak": -4, "level": 8,
                      "item_count": 4, "has_main_carry": True},
            "actions": [{"kind": "econ_action", "value": "roll", "label": "搜牌锁血",
                         "confidence": 0.55}],
            "missing_facts": [],
        },
        {
            # 20 血：经济断档（TFT-ECON-003 满足）但人口仍落后（TFT-POP-003 偏离）
            "stage": 5, "phase": 2,
            "facts": {"health": 20, "gold": 5, "streak": -5, "level": 7,
                      "item_count": 4, "has_main_carry": True},
            "actions": [{"kind": "econ_action", "value": "save", "label": "存钱吃利息",
                         "confidence": 0.87}],
            "missing_facts": [],
        },
        {
            # 6-1 九人口还在升人口 —— TFT-POP-002 偏离
            "stage": 6, "phase": 1,
            "facts": {"health": 18, "gold": 40, "streak": -6, "level": 9,
                      "item_count": 5, "has_main_carry": True},
            "actions": [
                {"kind": "econ_action", "value": "level_up", "label": "升人口",
                 "confidence": 0.9},
                {"kind": "positioning_action", "value": "protect_carry",
                 "label": "主C缩角保护", "confidence": 0.78},
            ],
            "missing_facts": [],
        },
        {
            "stage": 6, "phase": 2,
            "facts": {"health": 15, "gold": 60, "streak": -6, "level": 9,
                      "item_count": 5, "has_main_carry": True},
            "actions": [{"kind": "econ_action", "value": "roll", "label": "搜牌锁血",
                         "confidence": 0.7}],
            "missing_facts": [],
        },
    ],
}

#: 金铲铲对局：开局赛季强化、3-1 人口没提、5 阶段主C仅 3 件装备就追星。
JCC_RAW: Dict[str, Any] = {
    "match_id": "jcc-demo-001",
    "game_version": "S14",
    "player": {"nickname": "示例玩家", "rank": "钻石 II"},
    "outcome": {"placement": 5, "result": "lose"},
    "rounds": [
        {
            "stage": 1, "phase": 1,
            "facts": {"health": 88, "gold": 12, "streak": 1, "level": 4,
                      "item_count": 0, "has_main_carry": False},
            "actions": [{"kind": "econ_action", "value": "activate_season_buff",
                         "label": "激活赛季强化", "confidence": 0.9}],
            "missing_facts": [],
        },
        {
            # 铲铲的利息门槛是 60：攒到 62 却去搜牌 —— JCC-ECON-001 偏离
            "stage": 2, "phase": 3,
            "facts": {"health": 84, "gold": 62, "streak": 2, "level": 5,
                      "item_count": 0, "has_main_carry": False},
            "actions": [{"kind": "econ_action", "value": "roll", "label": "搜牌",
                         "confidence": 0.85}],
            "missing_facts": [],
        },
        {
            # 金币又回到 64，但本回合经济动作未采集到：
            # JCC-ECON-001 触发但必须判成「无法判定」，不能算成偏离
            "stage": 2, "phase": 5,
            "facts": {"health": 82, "gold": 64, "streak": 1, "level": 5,
                      "item_count": 0, "has_main_carry": False},
            "actions": [],
            "missing_facts": ["econ_action"],
        },
        {
            # 3-1 人口仍 5 —— 金铲铲要求 3-1 到 6 人口，JCC-POP-001 偏离
            "stage": 3, "phase": 1,
            "facts": {"health": 80, "gold": 30, "streak": 3, "level": 5,
                      "item_count": 1, "has_main_carry": False},
            "actions": [{"kind": "econ_action", "value": "save", "label": "存钱吃利息",
                         "confidence": 0.88}],
            "missing_facts": [],
        },
        {
            "stage": 4, "phase": 1,
            "facts": {"health": 62, "gold": 40, "streak": 1, "level": 7,
                      "item_count": 2, "has_main_carry": True},
            "actions": [
                {"kind": "econ_action", "value": "level_up", "label": "升人口",
                 "confidence": 0.93},
                {"kind": "positioning_action", "value": "shift_lineup",
                 "label": "调整站位", "confidence": 0.84},
            ],
            "missing_facts": [],
        },
        {
            "stage": 4, "phase": 2,
            "facts": {"health": 46, "gold": 20, "streak": -1, "level": 7,
                      "item_count": 2, "has_main_carry": True},
            "actions": [{"kind": "econ_action", "value": "save", "label": "存钱吃利息",
                         "confidence": 0.9}],
            "missing_facts": [],
        },
        {
            # 5 阶段主 C 只有 3 件装备却追星 —— JCC-CARRY-001 偏离
            "stage": 5, "phase": 1,
            "facts": {"health": 34, "gold": 18, "streak": -3, "level": 8,
                      "item_count": 3, "has_main_carry": True},
            "actions": [{"kind": "econ_action", "value": "roll", "label": "搜牌追星",
                         "confidence": 0.6}],
            "missing_facts": [],
        },
        {
            "stage": 5, "phase": 2,
            "facts": {"health": 30, "gold": 8, "streak": -4, "level": 9,
                      "item_count": 3, "has_main_carry": True},
            "actions": [{"kind": "econ_action", "value": "save", "label": "存钱吃利息",
                         "confidence": 0.86}],
            "missing_facts": [],
        },
        {
            # 6-1 九人口仍升人口，且站位没按规则调整
            "stage": 6, "phase": 1,
            "facts": {"health": 28, "gold": 30, "streak": -5, "level": 9,
                      "item_count": 4, "has_main_carry": True},
            "actions": [
                {"kind": "econ_action", "value": "level_up", "label": "升人口",
                 "confidence": 0.89},
                {"kind": "positioning_action", "value": "protect_carry",
                 "label": "主C缩角保护", "confidence": 0.8},
            ],
            "missing_facts": [],
        },
        {
            "stage": 6, "phase": 2,
            "facts": {"health": 26, "gold": 45, "streak": -5, "level": 9,
                      "item_count": 4, "has_main_carry": True},
            "actions": [{"kind": "econ_action", "value": "roll", "label": "搜牌锁血",
                         "confidence": 0.7}],
            "missing_facts": [],
        },
    ],
}

#: 王者荣耀样例局：一局**打野**，刻意做成「前中期有对有错、后期顺风做对、
#: 后期逆风做错」。回合时间照着操作模式库的三条时间窗排（0-4 / 5-10 /
#: 11 分钟以后），所以这个样例演示的就是按分路 × 阶段的判定。
#:
#: player.position 必须填：position 是对局级事实，不填就是 unknown，
#: 所有带位置条件的模式一条都不触发 —— 而报告上完全看不出原因。
HOK_RAW: Dict[str, Any] = {
    "match_id": "hok-demo-001",
    "game_version": "S45",
    "player": {"nickname": "示例玩家", "rank": "王者 40 星", "position": "打野"},
    "outcome": {"result": "lose"},
    "rounds": [
        {
            # 前期 2 分钟：反向开局刷野 —— 做对了
            "time_min": 2,
            "facts": {"gold": 1800, "gold_diff": 200, "level": 4, "tower_diff": 0,
                      "kill_diff": 0, "has_ult_ready": False},
            "actions": [{"kind": "objective_action", "value": "farm_jungle",
                         "label": "刷野补经济", "confidence": 0.9}],
            "missing_facts": [],
        },
        {
            # 前期 3 分钟：动作没标出来 -> 期望 objective_action 的模式
            # 必须判成「无法判定」，不能混进偏离率里
            "time_min": 3,
            "facts": {"gold": 2400, "gold_diff": 260, "level": 4, "tower_diff": 0,
                      "kill_diff": 0, "has_ult_ready": False},
            "actions": [],
            "missing_facts": ["objective_action"],
        },
        {
            # 前期 4 分钟：龙坑最后一刻该抢龙，却继续刷野 —— 置信度故意压低，
            # 报告里应当显式标出来
            "time_min": 4,
            "facts": {"gold": 3600, "gold_diff": 320, "level": 5, "tower_diff": 0,
                      "kill_diff": 1, "has_ult_ready": False},
            "actions": [{"kind": "objective_action", "value": "farm_jungle",
                         "label": "刷野补经济", "confidence": 0.62}],
            "missing_facts": [],
        },
        {
            # 中期 6 分钟：人齐，用龙逼团 —— 拿下了小龙，做对了
            "time_min": 6,
            "facts": {"gold": 6800, "gold_diff": 900, "level": 9, "tower_diff": 1,
                      "kill_diff": 0, "has_ult_ready": True},
            "actions": [{"kind": "objective_action", "value": "take_dragon",
                         "label": "拿下小龙", "confidence": 0.91}],
            "missing_facts": [],
        },
        {
            # 中期 8 分钟：经济落后却还在分推牵制 —— 该缩塔只刷安全野区
            "time_min": 8,
            "facts": {"gold": 8400, "gold_diff": -300, "level": 10, "tower_diff": 0,
                      "kill_diff": 2, "has_ult_ready": True},
            "actions": [{"kind": "objective_action", "value": "split_push",
                         "label": "分推牵制", "confidence": 0.84}],
            "missing_facts": [],
        },
        {
            # 后期 12 分钟顺风：把经济差转成塔经济，分推牵制 —— 做对了
            "time_min": 12,
            "facts": {"gold": 12000, "gold_diff": 2600, "level": 12, "tower_diff": 3,
                      "kill_diff": 4, "has_ult_ready": True},
            "actions": [{"kind": "objective_action", "value": "split_push",
                         "label": "分推牵制", "confidence": 0.92}],
            "missing_facts": [],
        },
        {
            # 后期 15 分钟：经济领先 2100 还在刷野 —— 该开龙或带线拆塔
            # 置信度同样压低，报告里要显式标出来
            "time_min": 15,
            "facts": {"gold": 15800, "gold_diff": 2100, "level": 14, "tower_diff": 3,
                      "kill_diff": 6, "has_ult_ready": True},
            "actions": [{"kind": "objective_action", "value": "farm_jungle",
                         "label": "刷野补经济", "confidence": 0.66}],
            "missing_facts": [],
        },
        {
            # 后期 18 分钟：用拉扯消耗，没硬接团
            "time_min": 18,
            "facts": {"gold": 18200, "gold_diff": -1800, "level": 15, "tower_diff": -2,
                      "kill_diff": -5, "has_ult_ready": False},
            "actions": [{"kind": "teamfight_action", "value": "kite",
                         "label": "拉扯消耗", "confidence": 0.85}],
            "missing_facts": [],
        },
    ],
}

MOCK_RAW: Dict[str, Dict[str, Any]] = {
    "tft": TFT_RAW,
    "jcc": JCC_RAW,
    "hok": HOK_RAW,
}


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------


def load_rules(game: str) -> List[Rule]:
    """从种子 JSON 读回规则对象，顺带验证 JSON 与 Rule.to_dict 结构一致。"""
    path = SEED_DIR / SEED_FILES[game]
    return [Rule.from_dict(item) for item in json.loads(path.read_text("utf-8"))]


def load_raw_rules(game: str) -> List[Dict[str, Any]]:
    return json.loads((SEED_DIR / SEED_FILES[game]).read_text("utf-8"))


# --------------------------------------------------------------------------
# 注册与事实集
# --------------------------------------------------------------------------


def test_three_adapters_registered():
    """import registry 之后三款游戏必须都在注册表里。"""
    assert sorted(REGISTRY) == ["hok", "jcc", "tft"]
    assert GAME_IDS == ["tft", "jcc", "hok"]
    for gid in GAME_IDS:
        assert get_adapter(gid) is REGISTRY[gid]
        assert get_adapter(gid).display_name


def test_autochess_adapters_are_independent_objects():
    """两款自走棋必须是两个独立实例，且 fact_names 不是同一个 list 对象。"""
    tft, jcc = get_adapter("tft"), get_adapter("jcc")
    assert tft is not jcc
    assert tft.fact_names is not jcc.fact_names
    assert tft.fact_names == jcc.fact_names  # 事实名同源
    assert tft.game_id != jcc.game_id
    # 经济口径必须真的不同，否则两套规则库会指向同一套判定
    assert tft.profile.gold_cap == 50
    assert jcc.profile.gold_cap == 60
    assert tft.profile.level_milestones[4] == 8
    assert jcc.profile.level_milestones[3] == 6


@pytest.mark.parametrize("game", GAME_IDS)
def test_required_facts_registered(game):
    """交付要求的事实必须全部注册，且不允许重复。"""
    adapter = get_adapter(game)
    required = REQUIRED_AUTECHESS_FACTS if game in ("tft", "jcc") else REQUIRED_MOBA_FACTS
    assert required.issubset(set(adapter.fact_names))
    assert len(adapter.fact_names) == len(set(adapter.fact_names))


@pytest.mark.parametrize("game", GAME_IDS)
def test_every_fact_has_chinese_label(game):
    adapter = get_adapter(game)
    for fact in adapter.fact_names:
        assert adapter.fact_labels.get(fact), f"{game}.{fact} 缺少中文名"


def test_autochess_and_moba_factsets_differ():
    """自走棋与 MOBA 的事实集只有六个名字重合，其余语义完全不同。

    值得强调的是：``gold`` / ``level`` / ``positioning_action`` 这些重合的
    名字并不代表语义重合 —— 自走棋的 gold 是「当前金币」，MOBA 的 gold 是
    「本方总经济」；自走棋的 positioning_action 是「本回合棋盘站位」，
    MOBA 的是「本段时间的走位与视野」。名字重合会让人误以为可以共用规则，
    所以这里把交集显式钉住，避免后续有人「顺手合并」两个事实集。
    """
    tft = set(get_adapter("tft").fact_names)
    hok = set(get_adapter("hok").fact_names)
    assert tft & hok == {
        "phase", "level", "gold", "positioning_action",
        "is_decision_point", "pressure",
    }
    # 自走棋独有的离散经济/装备口径不应出现在 MOBA
    for fact in ("health", "income", "streak", "econ_action", "item_count",
                 "has_main_carry", "round_tempo"):
        assert fact in tft and fact not in hok, fact
    # MOBA 独有的连续差值不应出现在自走棋
    for fact in ("time_min", "gold_diff", "tower_diff", "kill_diff",
                 "objective_action", "teamfight_action", "has_ult_ready"):
        assert fact in hok and fact not in tft, fact
    assert get_adapter("hok").round_unit != get_adapter("tft").round_unit


# --------------------------------------------------------------------------
# 规则库与事实集的对齐
# --------------------------------------------------------------------------


@pytest.mark.parametrize("game", MODE_GAMES)
def test_rule_facts_all_registered(game):
    """硬性质量要求 1：规则不得引用未注册事实。"""
    adapter = get_adapter(game)
    known = adapter.known_facts()
    for rule in load_rules(game):
        unknown = rule.referenced_facts() - known
        assert not unknown, f"{rule.id} 引用了未注册事实 {sorted(unknown)}"


@pytest.mark.parametrize("game", MODE_GAMES)
def test_seed_rules_pass_admission(game):
    """硬性质量要求 2：全部规则通过可判定性准入。"""
    known = get_adapter(game).known_facts()
    for rule in load_rules(game):
        reasons = check_admission(rule, known, DEFAULT_SUBJECTIVE_MARKERS)
        assert not reasons, f"{rule.id} 未通过准入：{reasons}"


@pytest.mark.parametrize("game", MODE_GAMES)
def test_no_subjective_wording_in_rules(game):
    """硬性质量要求 2 的显式复查：主观判断词一个都不许出现。"""
    for rule in load_rules(game):
        blob = " ".join([rule.title, rule.category, rule.rationale, rule.deviation_hint])
        for marker in DEFAULT_SUBJECTIVE_MARKERS:
            assert marker not in blob, f"{rule.id} 含主观判断词「{marker}」"


@pytest.mark.parametrize("game", MODE_GAMES)
def test_every_rule_is_traceable(game):
    """硬性质量要求 3：每条规则都要有攻略出处与原文式引用。"""
    for rule in load_rules(game):
        assert rule.source is not None, f"{rule.id} 缺少溯源"
        assert rule.source.guide.strip(), f"{rule.id} 溯源缺攻略名"
        assert rule.source.quoted.strip(), f"{rule.id} 溯源缺原文引用"
        assert rule.source.collected_at.strip(), f"{rule.id} 溯源缺采集日期"
        assert rule.rationale.strip() and rule.deviation_hint.strip()


#: 每款游戏的种子规则条数，**写死**而不是给区间。
#:
#: 区间只能卡上限，卡不住「为了凑够位置变体而注水」—— 而 MOBA 的规则天然
#: 会变多：一条团队判断在不同位置上要求的动作本来就不同，拆开才是对的。
#: 写死数字之后，多一条少一条都会在测试里暴露出来。
#:
#: 王者的 84 条来自《S45 分路决策树》：31 个时间节点 × 每节点若干分支，
#: 前提判不了的分支按「可接受动作集合」合并，所以条数由攻略本身决定。
#: 改了攻略就必须重跑 ``tools/build_hok_modes.py``，条数变了这里会先炸。
SEED_RULE_COUNTS = {"hok": 84}


@pytest.mark.parametrize("game", MODE_GAMES)
def test_rule_count_is_pinned(game):
    """规则条数必须与上面写死的一致 —— 宁可少也不能注水。"""
    rules = load_rules(game)
    assert len(rules) == SEED_RULE_COUNTS[game], (
        f"{game} 规则数 {len(rules)}，应为 {SEED_RULE_COUNTS[game]}"
    )


@pytest.mark.parametrize("game", MODE_GAMES)
def test_rule_ids_unique_and_prefixed(game):
    rules = load_rules(game)
    ids = [r.id for r in rules]
    assert len(ids) == len(set(ids))
    prefix = game.upper()
    for rid in ids:
        assert rid.startswith(prefix + "-"), rid


@pytest.mark.parametrize("game", MODE_GAMES)
def test_rule_game_and_version_match_adapter(game):
    """规则的 game 与 version 必须与适配器声明一致，否则规则会挂错口径。"""
    adapter = get_adapter(game)
    expected_version = (
        adapter.default_version
    )
    for rule in load_rules(game):
        assert rule.game == game
        assert rule.version == expected_version


def test_rule_serialization_roundtrip():
    """JSON 结构必须与 Rule.to_dict() 完全一致，可无损往返。"""
    for game in MODE_GAMES:
        raw_items = load_raw_rules(game)
        for item, rule in zip(raw_items, load_rules(game)):
            assert rule.to_dict() == item, f"{rule.id} 的 JSON 结构与 to_dict 不一致"


def test_expect_is_always_an_action_fact():
    """期望动作必须是动作事实。

    matcher 在回合里找不到该类动作时直接判 deviated，所以把 expect 挂在
    非动作事实上，会让这条模式在任何数据上都恒为偏离 —— 那不是模式，
    是一条永远成立的假结论。
    """
    for game in MODE_GAMES:
        adapter = get_adapter(game)
        action_facts = {"econ_action", "positioning_action", "objective_action",
                        "teamfight_action"}
        for rule in load_rules(game):
            assert rule.expect_fact in adapter.known_facts()
            assert rule.expect_fact in action_facts, f"{rule.id} 的 expect 不是动作事实"


# --------------------------------------------------------------------------
# normalize
# --------------------------------------------------------------------------


def test_tft_normalize_builds_context():
    ctx = get_adapter("tft").build_context(TFT_RAW)
    assert isinstance(ctx, MatchContext)
    assert ctx.match_id == "tft-demo-001"
    assert ctx.game == "tft"
    assert ctx.game_version == "13.1"
    assert ctx.outcome["placement"] == 4
    assert ctx.player["nickname"] == "示例玩家"
    assert len(ctx.rounds) == len(TFT_RAW["rounds"])
    # 回合按 (stage, phase) 排序输出
    labels = [r.label for r in ctx.iter_rounds()]
    assert labels == sorted(labels, key=lambda x: tuple(int(i) for i in x.split("-")))


def test_tft_derived_facts():
    """自走棋必须推导出收入 / 人口节奏 / 压力 / 决策点四个派生事实。"""
    ctx = get_adapter("tft").build_context(TFT_RAW)
    r41 = ctx.round_by_label("4-1")
    # 5 基础 + 2 利息(20 金) + 1 连胜奖励(4 连) = 8
    assert r41.facts["income"] == 8
    # 4 阶段里程碑为 8 人口，此时只有 6 -> slow
    assert r41.facts["round_tempo"] == "slow"
    # 58 血落在中压区间
    assert r41.facts["pressure"] == "mid"
    assert r41.facts["is_decision_point"] is True
    # 20 金币不吃满利息
    r42 = ctx.round_by_label("4-2")
    assert r42.facts["income"] == 5 + 3 + 2   # 35 金 -> 3 利息，5 连 -> +2


def test_jcc_economy_profile_changes_derived_facts():
    """同一份 raw 在两款自走棋下必须得到不同的收入与节奏口径。"""
    raw = {
        "match_id": "profile-check",
        "rounds": [{"stage": 4, "phase": 1,
                    "facts": {"health": 50, "gold": 60, "streak": 4, "level": 7},
                    "actions": [], "missing_facts": []}],
    }
    tft_rnd = get_adapter("tft").build_context(raw).rounds[0]
    jcc_rnd = get_adapter("jcc").build_context(raw).rounds[0]
    # 云顶：60 金吃满 5 利息，4 连未达翻倍阈值(+1) -> 11
    assert tft_rnd.facts["income"] == 5 + 5 + 1
    # 金铲铲：60 金吃满 6 利息，4 连已达翻倍阈值(+2) -> 13
    assert jcc_rnd.facts["income"] == 5 + 6 + 2
    # 2 阶段人口里程碑不同：金铲铲 2 阶段只要 5 人口，云顶要 6 人口
    assert get_adapter("jcc").profile.level_milestones[2] == 5
    assert get_adapter("tft").profile.level_milestones[2] == 6
    # 同一份 level=7 的数据在 4 阶段：两款游戏都判 slow（里程碑都是 8）
    assert tft_rnd.facts["round_tempo"] == jcc_rnd.facts["round_tempo"] == "slow"


def test_jcc_normalize_builds_context():
    ctx = get_adapter("jcc").build_context(JCC_RAW)
    assert ctx.game == "jcc"
    assert ctx.game_version == "S14"
    assert len(ctx.rounds) == len(JCC_RAW["rounds"])
    # 3 阶段里程碑为 6，3-1 只有 5 人口 -> slow
    assert ctx.round_by_label("3-1").facts["round_tempo"] == "slow"
    # 34 血低于金铲铲 35 血高压线
    assert ctx.round_by_label("5-1").facts["pressure"] == "high"


def test_hok_normalize_builds_context():
    ctx = get_adapter("hok").build_context(HOK_RAW)
    assert ctx.game == "hok"
    assert ctx.game_version == "S45"
    assert len(ctx.rounds) == len(HOK_RAW["rounds"])
    # MOBA 没有阶段概念，标签直接给分钟数（见 RoundRecord.label_override）
    r = ctx.round_by_label("第 15 分钟")
    assert r is not None
    assert r.label == "第 15 分钟"
    assert r.facts["time_min"] == 15
    # 相位窗口取自 S45 攻略的三条时间窗：0-4 前期 / 5-10 中期 / 11+ 后期
    assert r.facts["phase"] == "late"
    # 局势压力：三项差值投票。第 18 分钟三项全落后 -> 高压
    late = ctx.round_by_label("第 18 分钟")
    assert late.facts["gold_diff"] == -1800
    assert late.facts["kill_diff"] == -5
    assert late.facts["tower_diff"] == -2
    assert late.facts["pressure"] == "high"
    # position 是对局级事实，会广播到每一个回合
    assert r.facts["position"] == "jungle"


def test_hok_round_labels_are_human_readable():
    """MOBA 报告里的定位必须是「第 N 分钟」，不能是玩家读不懂的「2-2」。"""
    ctx = get_adapter("hok").build_context(HOK_RAW)
    for rnd in ctx.iter_rounds():
        assert rnd.label.startswith("第 "), f"标签不可读：{rnd.label}"
        assert rnd.label.endswith(" 分钟"), f"标签不可读：{rnd.label}"

    # 排序仍按数值时间轴，不受可读标签影响
    times = [r.facts["time_min"] for r in ctx.iter_rounds()]
    assert times == sorted(times)


def _hok_fired_rules(position: str) -> List[str]:
    """同一局对局，只改分路，返回触发的规则 id 集合。"""
    raw = json.loads(json.dumps(HOK_RAW))
    if position is None:
        raw["player"].pop("position", None)
    else:
        raw["player"]["position"] = position
    adapter = get_adapter("hok")
    report = analyze_match(load_rules("hok"), adapter.build_context(raw))
    return sorted({v.rule_id for v in report["verdicts"]})


def test_hok_seed_rules_all_require_a_position():
    """每条 MOBA 种子规则都必须带 position 条件。

    不带位置的条件对每个位置都成立 —— 那是一条没有分工的规则：
    它不会告诉玩家「你这个位置该做什么」，只会把局面复述一遍。
    """
    for r in load_rules("hok"):
        conds = [c for c in r.trigger.all if c.fact == "position"]
        assert conds, f"{r.id} 没有 position 条件：{r.title}"
        values = conds[0].value
        values = values if isinstance(values, list) else [values]
        assert "unknown" not in values, \
            f"{r.id} 把「未指定」当成一个位置来判了：{r.title}"


def test_hok_position_gates_which_rules_fire():
    """换分路就该换一批模式 —— 两条分路的结论不该有一丝重叠。

    有重叠就说明模式里没真正区分位置：那条模式对每条分路都成立，
    等于没有分工。
    """
    jungle = set(_hok_fired_rules("打野"))
    carry = set(_hok_fired_rules("射手"))

    assert jungle, "打野这一局没触发任何模式"
    assert carry, "射手这一局没触发任何模式"
    assert not (jungle & carry), f"两条分路触发了同一批模式：{jungle & carry}"
    assert all(r.startswith("HOK-MODE-JGL") for r in jungle), jungle
    assert all(r.startswith("HOK-MODE-CAR") for r in carry), carry


def test_hok_without_position_judges_nothing():
    """没填分路就一条都不判 —— 而不是猜一个位置。

    猜错位置会让整局每一条带位置的规则都判错，而那种错误在报告里
    看不出来：它只是安静地一条都不触发。
    """
    assert _hok_fired_rules(None) == []


def test_hok_phase_boundaries():
    """相位边界必须与 PHASE_TIME_WINDOWS 逐一对齐。

    取的六个点覆盖三段窗口的两端：0-4 前期、5-10 中期、11 分钟以后后期。
    ``between`` 是闭区间，所以「上一段的最后一分钟」和「下一段的第一分钟」
    必须是两个不同的整数 —— 早先 [0,4] / [4,10] 就在这里重叠了，
    第 4 分钟会同时落进两条模式，必定有一条被判偏离。
    """
    adapter = get_adapter("hok")
    minutes = (0, 4, 5, 10, 11, 25)
    raw = {"match_id": "phase-check", "rounds": [
        {"time_min": t, "facts": {"gold": 1000, "gold_diff": 0, "level": 10,
                                  "tower_diff": 0, "kill_diff": 0},
         "actions": [], "missing_facts": []}
        for t in minutes
    ]}
    ctx = adapter.build_context(raw)
    phases = [r.facts["phase"] for r in ctx.iter_rounds()]
    assert phases == ["early", "early", "mid", "mid", "late", "late"]

    # 窗口不许留缝：每一分钟都得落在某一段里
    from app.adapters.honor_of_kings import PHASE_TIME_WINDOWS

    covered = set()
    for lo, hi in PHASE_TIME_WINDOWS.values():
        covered.update(range(lo, hi + 1))
    assert covered == set(range(0, 61))


def test_actions_are_mirrored_into_facts():
    """动作事实必须镜像进 facts，trigger 才能对动作下条件。"""
    ctx = get_adapter("tft").build_context(TFT_RAW)
    r41 = ctx.round_by_label("4-1")
    assert r41.facts["econ_action"] == "all_in"
    r42 = ctx.round_by_label("4-2")
    assert r42.facts["econ_action"] == "save"
    assert r42.facts["positioning_action"] == "shift_lineup"
    # 镜像取回合内最后一次动作，与 matcher 的 last_action 口径一致
    r61 = ctx.round_by_label("6-1")
    assert r61.facts["econ_action"] == "level_up"


@pytest.mark.parametrize("game", GAME_IDS)
def test_confidence_is_preserved(game):
    """低置信度是报告里要显式标记的东西，归一化过程不能抹平。"""
    ctx = get_adapter(game).build_context(MOCK_RAW[game])
    confs = [
        a.confidence
        for r in ctx.rounds
        for a in r.actions
        if a.confidence < 1.0
    ]
    assert confs, f"{game} 模拟数据里没有任何非满置信度动作"
    assert all(0.0 <= c <= 1.0 for c in confs)
    # 具体值：云顶 5-1 的追星动作置信度 0.55 必须原样到达
    if game == "tft":
        assert ctx.round_by_label("5-1").lowest_confidence("econ_action") == pytest.approx(0.55)


@pytest.mark.parametrize("game", GAME_IDS)
def test_missing_facts_pass_through(game):
    ctx = get_adapter(game).build_context(MOCK_RAW[game])
    marked = [
        (r.label, sorted(r.missing_facts))
        for r in ctx.rounds
        if r.missing_facts
    ]
    assert marked, f"{game} 模拟数据里没有采集缺失的回合"


def test_missing_fact_does_not_produce_derived_value():
    """原料缺失时派生事实必须一起标缺失，而不是拿半截数据算一个值。"""
    raw = {
        "match_id": "missing-check",
        "rounds": [{"stage": 4, "phase": 1,
                    "facts": {"health": 50, "level": 7, "streak": 2},
                    "actions": [], "missing_facts": ["gold"]}],
    }
    ctx = get_adapter("tft").build_context(raw)
    r = ctx.rounds[0]
    assert "income" in r.missing_facts
    assert "income" not in r.facts
    # resolve 对缺失事实必须返回 present=False，规则才不会拿它下判定
    assert r.resolve("income") == (None, False)
    assert r.resolve("pressure")[1] is True  # pressure 有 streak 原料，可正常计算


def test_explicit_missing_marker_blocks_derivation():
    """raw 显式标记 pressure 缺失时，适配器不得再把派生值写回去。"""
    raw = {
        "match_id": "missing-pressure",
        "rounds": [{"stage": 4, "phase": 1,
                    "facts": {"health": 50, "gold": 10, "streak": 1, "level": 7},
                    "actions": [], "missing_facts": ["pressure"]}],
    }
    r = get_adapter("tft").build_context(raw).rounds[0]
    assert "pressure" in r.missing_facts
    assert "pressure" not in r.facts


@pytest.mark.parametrize("game", GAME_IDS)
def test_validate_round_reports_no_problem(game):
    """归一化之后不应留下未注册的事实或动作类型。"""
    adapter = get_adapter(game)
    ctx = adapter.build_context(MOCK_RAW[game])
    problems: List[str] = []
    for r in ctx.rounds:
        problems.extend(adapter.validate_round(r))
    assert not problems, problems


def test_unknown_action_kind_is_flagged_not_swallowed():
    """未注册的动作类型必须被 validate_round 抓出来，而不是被静默丢弃。"""
    raw = {
        "match_id": "unknown-action",
        "rounds": [{"stage": 4, "phase": 1,
                    "facts": {"health": 50, "gold": 10, "streak": 1, "level": 7},
                    "actions": [{"kind": "shop_action", "value": "buy", "confidence": 1.0}],
                    "missing_facts": []}],
    }
    adapter = get_adapter("tft")
    r = adapter.build_context(raw).rounds[0]
    # 动作本身保留（供人工复查），但不进入事实字典
    assert r.actions[0].kind == "shop_action"
    assert "shop_action" not in r.facts
    problems = adapter.validate_round(r)
    assert any("shop_action" in p for p in problems)


# --------------------------------------------------------------------------
# 端到端：规则库 -> build_context -> analyze_match
# --------------------------------------------------------------------------


@pytest.mark.parametrize("game", MODE_GAMES)
def test_seed_rules_fire_on_mock_match(game):
    """硬性质量要求 4：规则必须真的能命中模拟数据。"""
    adapter = get_adapter(game)
    rules = load_rules(game)
    ctx = adapter.build_context(MOCK_RAW[game])
    report = analyze_match(rules, ctx)

    assert report["match_id"] == MOCK_RAW[game]["match_id"]
    assert report["game"] == game
    assert report["stats"].triggered > 0, f"{game} 没有任何规则被触发"
    # 每条结论都必须能定位到具体回合，且回合确实存在
    labels = {r.label for r in ctx.rounds}
    for v in report["verdicts"]:
        assert v.round_label in labels, f"{v.rule_id} 的回合 {v.round_label} 不存在"
    # 同时要出现「满足」与「偏离」，只有偏离说明期望动作定错了
    assert report["stats"].satisfied > 0, f"{game} 没有任何规则被满足"
    assert report["stats"].deviated > 0, f"{game} 没有任何规则被偏离"


@pytest.mark.parametrize("game", MODE_GAMES)
def test_gaps_carry_locatable_evidence(game):
    """归因记录必须同时带着回合、局面事实、期望动作与实际动作。"""
    adapter = get_adapter(game)
    ctx = adapter.build_context(MOCK_RAW[game])
    report = analyze_match(load_rules(game), ctx)

    assert report["gaps"], f"{game} 没有产出任何差距"
    for gap in report["gaps"]:
        assert gap.headline().startswith(gap.round_label)
        assert gap.situation, "局面快照为空，无法复查"
        assert gap.expected["fact"]
        assert "values" in gap.actual
        assert gap.hint and gap.rationale
        assert gap.source and gap.source["guide"]


@pytest.mark.parametrize("game", MODE_GAMES)
def test_compliance_rate_is_wellformed(game):
    """执行度必须来自真实判定，且落在 [0, 1]。"""
    adapter = get_adapter(game)
    report = analyze_match(load_rules(game), adapter.build_context(MOCK_RAW[game]))
    stats = report["stats"]
    rate = stats.compliance_rate
    assert rate is not None, "执行度不应为 None：模拟数据必然触发规则"
    assert 0.0 <= rate <= 1.0
    assert stats.decided == stats.satisfied + stats.deviated
    assert stats.decided > 0
    assert stats.to_dict()["active_rules"] == len(load_rules(game))
    assert report["by_category"]


@pytest.mark.parametrize("game", MODE_GAMES)
def test_low_confidence_deviations_are_flagged(game):
    """识别置信度低的偏离必须在报告里被显式标记。"""
    adapter = get_adapter(game)
    report = analyze_match(load_rules(game), adapter.build_context(MOCK_RAW[game]))
    flagged = [g for g in report["gaps"] if g.low_confidence]
    assert flagged, f"{game} 没有被标记的低置信度判定"
    for g in flagged:
        assert g.confidence < 0.75


@pytest.mark.parametrize("game", MODE_GAMES)
def test_missing_action_yields_undetermined_not_deviation(game):
    """采集缺失的回合必须判成「无法判定」，不能混进偏离率里。"""
    adapter = get_adapter(game)
    rules = load_rules(game)
    ctx = adapter.build_context(MOCK_RAW[game])
    report = analyze_match(rules, ctx)
    undetermined = report["undetermined"]
    assert undetermined, f"{game} 没有出现无法判定的回合"
    for v in undetermined:
        assert v.status == "undetermined"
    # 无法判定不得被算成偏离。
    # 注意比对粒度必须是 (rule_id, round_label)：同一条规则在别的回合
    # 完全可能是真实偏离（比如主 C 规则在 4-1 偏离、4-3 采集失败）。
    deviated_at = {(g.rule_id, g.round_label) for g in report["gaps"]}
    for v in undetermined:
        assert (v.rule_id, v.round_label) not in deviated_at


def test_hok_key_moment_is_located():
    """产品主张的直接验收：失误必须能定位到**具体那一个回合**。

    逐回合判定是复查用的底座 —— 上面那层「按分路收拢」只是把同一段的
    结论合上，落到哪一回合必须仍然可查。
    """
    ctx = get_adapter("hok").build_context(HOK_RAW)
    report = analyze_match(load_rules("hok"), ctx)
    # 第 4 分钟该抢龙却继续刷野 —— 打野四分钟那条模式应当命中
    ids = {(g.rule_id, g.round_label) for g in report["gaps"]}
    assert ("HOK-MODE-JGL-T0400-02", "第 4 分钟") in ids, ids
    gap = next(g for g in report["gaps"]
               if g.rule_id == "HOK-MODE-JGL-T0400-02")
    # 局面快照里必须带上复查所需的原始事实
    assert gap.situation["gold_diff"] == 320
    assert gap.actual["values"] == ["farm_jungle"]


def test_analyze_match_without_trigger_returns_no_rate():
    """没有任何模式触发时不编造执行度数字。"""
    empty = {"match_id": "empty", "rounds": [
        {"time_min": 3,
         "facts": {"gold": 1800, "gold_diff": 0, "level": 4,
                   "tower_diff": 0, "kill_diff": 0},
         "actions": [{"kind": "objective_action", "value": "farm_jungle",
                      "label": "", "confidence": 1.0}],
         "missing_facts": []}]}
    ctx = get_adapter("hok").build_context(empty)
    report = analyze_match(load_rules("hok"), ctx)
    assert report["stats"].compliance_rate is None
    assert report["gaps"] == []
