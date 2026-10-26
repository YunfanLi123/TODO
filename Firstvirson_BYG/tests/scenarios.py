"""端到端场景断言。

这些场景是「跨模块走一遍真实链路」的检查 —— 单元测试保证每层正确，
集成测试保证接得上，但接错时两者都未必会红（比如 API 返回 200 却什么都没做）。
这里守的就是最后这一类。

**同一份断言只有这一处定义。** ``verify_*.py`` 和 pytest 都调用这里的函数 ——
早先把断言写在独立脚本里，结果 CI 跑 pytest 时这些场景全被漏掉了，
等于最该跑的检查最容易丢。

每个场景接收一个 ``TestClient``，按顺序执行若干检查并返回结果；
场景内部的前置状态是后续检查的前提，所以必须整体按顺序跑，不能挑着跑。
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass
from typing import Callable, List

# --------------------------------------------------------------------------
# 检查结果
# --------------------------------------------------------------------------


@dataclass
class Check:
    """一次断言的结果。"""

    name: str
    ok: bool
    section: str = ""

    def __str__(self) -> str:
        return ("[PASS] " if self.ok else "[FAIL] ") + self.name


class Recorder:
    """收集断言。不直接打印，方便 pytest 复用。"""

    def __init__(self) -> None:
        self.items: List[Check] = []
        self._section = ""

    def section(self, title: str) -> None:
        self._section = title
        self.items.append(Check(f"── {title} ──", True, title))

    def check(self, cond: bool, name: str) -> bool:
        self.items.append(Check(name, bool(cond), self._section))
        return bool(cond)

    def fatal(self, name: str, detail: str = "") -> None:
        self.items.append(Check(f"{name}{'：' + detail if detail else ''}", False,
                                self._section))

    @property
    def failed(self) -> List[Check]:
        return [c for c in self.items if not c.ok]

    @property
    def total(self) -> int:
        return len([c for c in self.items if c.ok or not c.section.startswith("──")])

    @property
    def ok(self) -> bool:
        return not self.failed


def strip_script(html: str) -> str:
    """剥掉 script/style 后再检查渲染内容。

    内联的 JS/CSS 本来就含有 ``undefined`` / ``isNaN`` 这类词，直接扫源码
    必然误报 —— 这个坑踩过：断言一直红，代码其实没问题。
    """
    html = re.sub(r"<script.*?</script>", "", html, flags=re.S)
    return re.sub(r"<style.*?</style>", "", html, flags=re.S)


# --------------------------------------------------------------------------
# 共享的前置
# --------------------------------------------------------------------------


#: 只填了操作模式库的游戏 —— 目前只有王者荣耀。
#: 云顶 / 金铲铲保留适配器模块，资料后补，所以它们的产品库里是空的。
MODE_GAMES = ("hok",)

#: 端到端场景要验的是**机器**（版本差异、回测、覆盖率、抽取确认），
#: 不是某款游戏的具体资料。为了不假装云顶已经有产品库，场景改用
#: tests/fixtures/tft_test_library.json 这份**测试夹具库** —— 它只在
#: 测试里存在，产品不加载。
FIXTURE_LIBRARY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures", "tft_test_library.json"
)


def seed_fixture_library() -> None:
    """把夹具库的**基线版本**（13.1）导入当前会话指向的数据库。

    只导基线：13.2 是 version_flow 场景用 :func:`build_next_version` 现造的，
    提前塞进去会让「单次导入不允许跨版本」这道校验直接抛错 ——
    那条校验本身是对的，不该为了迁就夹具而放宽。
    """
    import json

    from app.core.dsl import Rule
    from app.rules.store import import_rules, latest_version
    from app.adapters.base import get_adapter

    with _session() as s:
        if latest_version(s, "tft"):
            return
        data = json.loads(open(FIXTURE_LIBRARY, encoding="utf-8").read())
        rules = [Rule.from_dict(d) for d in data if d.get("version") == "13.1"]
        adapter = get_adapter("tft")
        import_rules(
            s,
            rules,
            adapter.known_facts(),
            strict=True,
            action_facts=adapter.known_action_facts(),
        )


def seed_all(client) -> None:
    """载产品库 + 夹具库。

    ``/api/games/{game}/rules/seed`` 对没有资料的游戏返回空结果而不是报错，
    所以这一步对三款游戏都安全。
    """
    for game in ("tft", "jcc", "hok"):
        client.post(f"/api/games/{game}/rules/seed")
    seed_fixture_library()


# --------------------------------------------------------------------------
# 版本迭代构造：造一个真实的下一版本
# --------------------------------------------------------------------------


def build_next_version(rules_v131: list) -> list:
    """在 13.1 基础上造 13.2，刻意覆盖四种真实变更。

    这些构造器属于**场景**而不是脚本 —— pytest 也要用它们，
    所以必须住在 scenarios 里，而不是住在 verify_*.py 那个薄壳里
    （薄壳 import 场景、场景再 import 薄壳，是循环）。
    """
    from dataclasses import replace

    from app.core.dsl import Condition, Rule, Source, Trigger

    out = []
    for r in copy.deepcopy(rules_v131):
        rid = r.id

        # 放宽：金币门槛 50 -> 60
        # 现实依据：更新后利息上限下调，50 金币不再触发额外收益。
        if rid == "TFT-ECON-002":
            r.trigger = replace(
                r.trigger,
                all=[replace(c, op=">=", value=60) if c.fact == "gold" else c
                     for c in r.trigger.all],
            )
            r.title = "2 阶段金币达到 60 时才需要保持满利息"
            r.rationale = (
                "版本 13.2 起金币利息上限由 5 降至 4，50 金币已无额外收益，"
                "提前消费不再有明显代价。"
            )
            r.version = "13.2"
            out.append(r)
            continue

        # 收紧：血量阈值 30 -> 45
        # 现实依据：更新后排加强，血量低时再拖装备成型会被快速打穿。
        if rid == "TFT-ITEM-002":
            r.trigger = replace(
                r.trigger,
                all=[replace(c, value=45)
                     if c.fact == "health" and c.op == "<=" else c
                     for c in r.trigger.all],
            )
            r.title = "5 阶段装备 4 件以上且血量不超过 45 时必须搜牌"
            r.rationale = (
                "版本 13.2 起敌方伤害提升，血量 45 以下已处于被快速打穿区间，"
                "此时继续攒经济会让阵容来不及成型。"
            )
            r.version = "13.2"
            out.append(r)
            continue

        # 失效：该机制在新版本被移除，原建议彻底不适用
        if rid == "TFT-RHYTHM-001":
            continue

        r.version = "13.2"
        out.append(r)

    # 新增：只在新版本存在的打法
    out.append(
        Rule(
            id="TFT-SEASON-001", game="tft", version="13.2",
            title="3 阶段中后期金币低于 30 且连胜 3 以上时应主动断连",
            category="节奏运营",
            trigger=Trigger(all=[
                Condition("stage", "==", 3),
                Condition("gold", "<", 30),
                Condition("streak", ">=", 3),
            ]),
            expect_fact="econ_action", expect_op="in",
            expect_value=["save", "break_streak"],
            severity="medium",
            rationale=(
                "版本 13.2 新增连胜中断奖励机制，3 阶段中后期主动断连可换取额外金币，"
                "这是旧版本攻略里不存在的打法。"
            ),
            deviation_hint="这一回合即使在连胜也应停下来断连，换取新版本的奖励金币",
            tags=["版本新增", "连胜"],
            source=Source(
                guide="云顶之弈 13.2 版本更新说明",
                locator="赛季机制 / 连胜奖励",
                quoted="13.2 起，主动中断 3 连胜以上可获得 3 金币奖励。",
                collected_at="2026-10-15",
            ),
        )
    )
    return out


def make_history() -> list:
    """四局历史，金币刻意跨过新旧门槛（60）。

    - 局 1、3（62 / 65）：新旧规则都触发 —— 检验「偏离率是否变化」
    - 局 2、4（52 / 55）：新版不再触发 —— 检验「是否彻底不适用」

    玩家行为高度一致：2 阶段金币一到位就搜牌锁血（老习惯），
    这样回测结论稳定、可断言，而不是被随机波动带偏。
    """
    def one(idx: int, gold_at_2_3: int) -> dict:
        return {
            "match_id": f"hist-tft-{idx:02d}",
            "game": "tft", "game_version": "13.1",
            "played_at": f"2026-10-0{idx} 21:00",
            "player": {"nickname": "示例玩家", "rank": "钻石 II"},
            "outcome": {"placement": idx + 2, "result": "lose"},
            "rounds": [
                {"stage": 2, "phase": 3,
                 "facts": {"health": 100, "level": 5, "gold": gold_at_2_3,
                           "streak": 0, "income": 5, "item_count": 0,
                           "has_main_carry": False, "is_decision_point": False,
                           "pressure": "low", "round_tempo": "normal"},
                 "actions": [{"kind": "econ_action", "value": "reroll",
                              "confidence": 0.95}], "missing_facts": []},
                {"stage": 3, "phase": 4,
                 "facts": {"health": 82, "level": 6, "gold": 34,
                           "streak": 3, "income": 5, "item_count": 1,
                           "has_main_carry": True, "is_decision_point": True,
                           "pressure": "mid", "round_tempo": "fast"},
                 "actions": [{"kind": "econ_action", "value": "reroll",
                              "confidence": 0.92}], "missing_facts": []},
                {"stage": 5, "phase": 4,
                 "facts": {"health": 40, "level": 8, "gold": 40,
                           "streak": 0, "income": 5, "item_count": 4,
                           "has_main_carry": True, "is_decision_point": True,
                           "pressure": "high", "round_tempo": "normal"},
                 "actions": [{"kind": "econ_action", "value": "save",
                              "confidence": 0.9}], "missing_facts": []},
                {"stage": 6, "phase": 3,
                 "facts": {"health": 20, "level": 9, "gold": 25,
                           "streak": 1, "income": 5, "item_count": 6,
                           "has_main_carry": True, "is_decision_point": True,
                           "pressure": "high", "round_tempo": "normal"},
                 "actions": [{"kind": "econ_action", "value": "reroll",
                              "confidence": 0.93}], "missing_facts": []},
            ],
        }

    return [one(i, g) for i, g in enumerate([62, 52, 65, 55], start=1)]


# ==========================================================================
# 场景 1：版本适配与回测
# ==========================================================================


def scenario_version_flow(client, rec: Recorder) -> None:
    """版本切换回测：同一批历史对局，新旧两套规则各判一次。"""
    from app.rules.store import load_seed_library, upsert_rule

    rec.section("[0] 前置")
    client.post("/api/games/tft/rules/seed")
    seed_fixture_library()
    if not rec.check(True, "夹具库已载入"):
        return

    # 在 13.1 之上造一个 13.2，模拟真实版本迭代
    old_rules = _load_rules(client, "13.1")
    new_rules = build_next_version(old_rules)

    rec.section("[1] 建立新版本")
    with _session() as s:
        for r in new_rules:
            upsert_rule(s, r)
        s.flush()
    rec.check(True, f"构造 13.2 规则库（{len(new_rules)} 条）")

    rec.section("[2] 灌入历史对局")
    posted = 0
    for raw in make_history():
        if client.post("/api/matches", json=raw).status_code == 200:
            posted += 1
    rec.check(posted == 4, f"录入 {posted}/4 局历史对局")

    rec.section("[3] 版本差异识别")
    r = client.post("/api/versions/tft/diff", json={
        "game": "tft", "old_version": "13.1", "new_version": "13.2",
    })
    if not rec.check(r.status_code == 200, f"差异接口可用（{r.status_code}）"):
        return
    diff = r.json()["version_diff"]
    added = {a["rule_id"] for a in diff["added"]}
    removed = {x["rule_id"] for x in diff["removed"]}
    changed = {c["rule_id"] for c in diff["changed"]}
    rec.check("TFT-SEASON-001" in added, "识别出新增规则")
    rec.check("TFT-RHYTHM-001" in removed, "识别出移除规则")
    rec.check({"TFT-ECON-002", "TFT-ITEM-002"} <= changed, "识别出阈值变更")

    rec.section("[4] 回测结论")
    r = client.get("/api/versions/tft/backtest",
                   params={"old_version": "13.1", "new_version": "13.2"})
    if not rec.check(r.status_code == 200, f"回测接口可用（{r.status_code}）"):
        return
    rows = r.json()["backtest"]
    by_id = {x["rule_id"]: x for x in rows}

    e2 = by_id.get("TFT-ECON-002")
    rec.check(e2 is not None, "放宽的规则进入回测")
    if e2:
        rec.check(e2["old_rate"] is not None and e2["old_rate"] > 0.5,
                  f"旧版偏离率高（{e2['old_rate']}）")
        rec.check(e2["old_checked"] == 4, f"旧版检验 4 局（{e2['old_checked']}）")
        rec.check(e2["new_checked"] == 2, f"新版检验 2 局（{e2['new_checked']}）")
        rec.check(e2["verdict"] == "基本一致",
                  f"跨门槛仍偏离 → 基本一致（{e2['verdict']}）")

    i2 = by_id.get("TFT-ITEM-002")
    rec.check(i2 is not None, "收紧的规则进入回测")
    if i2:
        rec.check(i2["old_checked"] == 0, f"旧版从未触发（{i2['old_checked']}）")
        rec.check(i2["new_checked"] == 4, f"新版覆盖 4 回合（{i2['new_checked']}）")
        rec.check(i2["verdict"] == "新增约束",
                  f"结论为新增约束（{i2['verdict']}）")

    untouched = {"TFT-ECON-001", "TFT-POP-001", "TFT-D-ROLL-001",
                 "TFT-ITEM-001", "TFT-POS-001", "TFT-POS-002",
                 "TFT-POP-002", "TFT-ECON-003"}
    rec.check(not (untouched & set(by_id)),
              f"未变更的规则不参与回测（回测 {len(rows)} 条）")

    rec.section("[5] 报告与降级")
    r = client.get("/api/versions/tft/backtest",
                   params={"old_version": "13.1", "new_version": "13.2",
                           "as_html": "true"})
    html = r.text
    visible = strip_script(html)
    rec.check('id="version"' in html, "版本区块已渲染")
    rec.check('id="timeline"' not in html, "单局专属区块已隐藏")
    for bad in ("{{", "undefined", "NaN", "[object Object]"):
        rec.check(bad not in visible, f"渲染无残留 {bad!r}")

    r = client.get("/api/versions/tft/backtest",
                   params={"old_version": "13.1", "new_version": "13.2"})
    s = r.json()["backtest_summary"]
    rec.check(s["total"] == len(rows), "报告统计与回测一致")
    rec.check(bool(s.get("newly_binding")), "「新增约束」分桶非空")

    rec.section("[6] 切换预览与执行")
    r = client.post("/api/versions/tft/activate", json={
        "game": "tft", "from_version": "13.1", "to_version": "13.2",
        "dry_run": True,
    })
    rec.check(r.status_code == 200 and r.json().get("dry_run") is True,
              "dry_run 预览可用")


def _load_rules(client, version):
    """通过 API 取不到 Rule 对象，这里直接从库里读。"""
    from app.rules.store import get_rules

    with _session() as s:
        return get_rules(s, "tft", version)


# ==========================================================================
# 场景 2：视频识别契约
# ==========================================================================


VIDEO_RECOGNITION = {
    "match_id": "tft-video-20261014",
    "game": "tft", "game_version": "13.1",
    "player": {"nickname": "示例玩家", "rank": "钻石 II"},
    "outcome": {"placement": 4, "result": "lose"},
    "played_at": "2026-10-14 21:45",
    "recognizer_id": "gate-roi-v1", "recognizer_version": "0.3.0",
    "stats": {"elapsed_sec": 42.5, "frames_scanned": 5400},
    "warnings": ["第 2 回合起画面左上角有弹窗遮挡，持续到第 6 回合"],
    "frames": [{"index": i, "timestamp": i * 0.5,
                "is_boundary": i in (0, 900, 1800), "confidence": 0.93}
               for i in range(5400)],
    "rounds": [
        {"stage": 4, "phase": 1,
         "facts": [
             {"name": "stage", "value": 4, "confidence": 0.82, "method": "gate"},
             {"name": "phase", "value": 1, "confidence": 0.80, "method": "gate"},
             {"name": "health", "value": 58, "confidence": 0.96, "method": "roi"},
             {"name": "streak", "value": 4, "confidence": 0.48, "method": "roi"},
             {"name": "gold", "value": 20, "confidence": 0.12,
              "method": "roi", "failure": "金币栏被技能特效遮挡"},
         ],
         "actions": [{"kind": "econ_action", "value": "all_in",
                      "label": "梭哈追三星", "confidence": 0.91}],
         "gate_confidence": 0.93},
        {"stage": 4, "phase": 4,
         "facts": [
             {"name": "stage", "value": 4, "confidence": 0.81, "method": "gate"},
             {"name": "health", "value": 44, "confidence": 0.94, "method": "roi"},
         ],
         "actions": [{"kind": "econ_action", "value": "level_up",
                      "confidence": 0.55}],
         "gate_confidence": 0.22},
    ],
}


def scenario_video_contract(client, rec: Recorder) -> None:
    """视频识别：降级是否如实发生、诊断是否留档、成本结构是否成立。"""
    rec.section("[0] 前置")
    client.post("/api/games/tft/rules/seed")
    seed_fixture_library()
    if not rec.check(True, "夹具库已载入"):
        return

    rec.section("[1] 接收识别结果")
    r = client.post("/api/matches/from-recognition", json=VIDEO_RECOGNITION)
    if not rec.check(r.status_code == 200, f"录入成功（{r.status_code}）"):
        rec.fatal("录入失败", r.text[:200])
        return
    body = r.json()
    deg = body["degradation"]

    rec.check(body["source_kind"] == "video", "来源标记为 video")
    rec.check(deg["total_rounds"] == 2, f"共 2 个回合（{deg['total_rounds']}）")
    rec.check(deg["dropped_rounds"] == 1,
              f"门控低置信度回合整段丢弃（{deg['dropped_rounds']}）")

    rec.section("[2] 降级正确性")
    flagged = {f["fact"] for f in deg["low_confidence_facts"]}
    rejected = {f.get("fact") for f in deg["rejected_facts"]}
    rec.check("streak" in flagged, "中等置信度的 streak 被保留并标记")
    rec.check("gold" in rejected, "识别失败的 gold 被拒绝")
    reasons = [f.get("reason", "") for f in deg["rejected_facts"]]
    rec.check(any("遮挡" in x for x in reasons), "拒绝原因记录了具体成因")

    rec.section("[3] 成本结构")
    cost = body["cost"]
    rec.check(cost["frames_scanned"] == 5400, f"门控扫描 {cost['frames_scanned']} 帧")
    rec.check(cost["fine_recognitions"] == 2,
              f"精细识别仅 {cost['fine_recognitions']} 次")
    rec.check(cost["fine_ratio"] < 0.01,
              f"昂贵识别占比 {cost['fine_ratio']:.3%}（应 <1%）")

    rec.section("[4] 报告如实呈现")
    r = client.get(f"/api/matches/{VIDEO_RECOGNITION['match_id']}/report.json")
    if not rec.check(r.status_code == 200, "报告可取"):
        return
    rep = r.json()
    rc = rep.get("recognition") or {}
    rec.check(rc.get("recognizer_id") == "gate-roi-v1", "报告记录识别方案标识")
    rec.check(bool(rc.get("low_confidence_facts")), "报告列出低置信度事实")
    rec.check(bool(rc.get("rejected_facts")), "报告列出识别失败项")

    lc = rep.get("low_confidence_facts") or []
    rec.check(bool(lc), "低置信度事实已关联到具体差距")
    if lc:
        rec.check(bool(lc[0].get("rule_id")),
                  f"关联到规则 {lc[0].get('rule_id')}")
        rec.check(bool(lc[0].get("round_label")),
                  f"定位到回合 {lc[0].get('round_label')}")

    rec.section("[5] HTML 渲染")
    r = client.get(f"/api/matches/{VIDEO_RECOGNITION['match_id']}/report")
    html = r.text
    visible = strip_script(html)
    for bad in ("{{", "undefined", "NaN", "[object Object]"):
        rec.check(bad not in visible, f"渲染无残留 {bad!r}")
    # 报告正文只有「操作模式分析」一节，低置信度标记挂在条目上，
    # 不另起一节 —— 但它必须看得见。
    rec.check('id="analysis"' in html, "操作模式分析章节已渲染")
    rec.check("低置信度" in visible, "页面出现低置信度提示")
    rec.check("识别置信度" in visible, "低置信度标记没有挂在结论上")

    rec.section("[6] 非视频来源不显示该标记")
    from demo_data import build_demo_match

    client.post("/api/matches", json=build_demo_match("tft"))
    r = client.get("/api/matches/tft-demo-20261010/report")
    rec.check("识别置信度" not in strip_script(r.text),
              "手工录入的对局不该有识别置信度标记")


# ==========================================================================
# 场景 3：攻略采集合规
# ==========================================================================

GUIDE_HTML = """<html><head><title>云顶 13.2 版本运营要点</title>
<script>track();</script></head><body>
<nav>首页 攻略库</nav>
<div class="content">
<p>4-1 回合连胜达到 3 以上并且血量不低于 50 时，应该停手存钱。</p>
<p>2 阶段金币超过 50 就不要乱花钱，存钱吃利息。</p>
<p>5 阶段装备达到 4 件并且血量不超过 30 时，应该搜牌锁血。</p>
<p>6 阶段 2 到 4 回合已经确定主 C 时，应该搜牌或者强化主 C。</p>
<p>这波该不该追三要看对手站位，尽量不要梭哈。</p>
<p>局势危急的时候酌情撤退，看情况处理。</p>
</div>
<footer>版权所有</footer></body></html>"""


def scenario_guide_collect(client, rec: Recorder) -> None:
    """攻略采集：白名单边界、采集不写库、原文不落库。"""
    from app.collection.collector import HttpGuideCollector, SourceNotAllowed

    rec.section("[0] 载入基线")
    client.post("/api/games/tft/rules/seed")
    seed_fixture_library()
    if not rec.check(True, "夹具库已载入"):
        return

    rec.section("[1] 白名单边界")
    collector = HttpGuideCollector()
    rec.check(not collector.is_allowed("https://any-site.example/guide"),
              "默认配置拒绝任意站点")
    try:
        collector.fetch("https://any-site.example/guide")
        rec.check(False, "未拦截 —— 严重问题")
    except SourceNotAllowed:
        rec.check(True, "拦截异常类型正确")

    strict = HttpGuideCollector(allowed_hosts=["official.example.com"])
    rec.check(strict.is_allowed("https://official.example.com/g"), "白名单内放行")
    rec.check(not strict.is_allowed("https://notofficial.example.com/g"),
              "相似域名被拒")

    rec.section("[2] API 要求显式声明来源")
    r = client.post("/api/guides/collect", json={
        "url": "https://official.example.com/g", "game": "tft",
        "version": "13.2", "allowed_hosts": [],
    })
    rec.check(r.status_code == 422, f"未声明 allowed_hosts 被拒（{r.status_code}）")

    rec.section("[3] 正文提取与候选")
    r = client.post("/api/guides/text", json={
        "text": GUIDE_HTML, "game": "tft", "version": "13.2",
        "guide_title": "云顶 13.2 版本运营要点", "min_confidence": 0.4,
    })
    if not rec.check(r.status_code == 200, f"文本抽取成功（{r.status_code}）"):
        return
    body = r.json()
    cands = body["candidates"]
    rec.check(len(cands) > 0, f"抽出 {len(cands)} 条候选")
    rec.check(len(body["rejected_subjective"]) >= 2,
              f"拦下 {len(body['rejected_subjective'])} 条主观表述")
    rec.check(all("该不该" not in c["title"] for c in cands), "主观句未进入候选")
    rec.check(all(not c["subjective_markers"] for c in cands), "候选无主观标记")
    rec.check(all("<p>" not in c["title"] for c in cands), "HTML 标签已剥离")

    rec.section("[4] 采集阶段绝不写规则")
    from app.rules.store import get_rules, latest_version

    with _session() as s:
        rec.check(latest_version(s, "tft") == "13.1", "未产生新版本")
        rec.check(len(get_rules(s, "tft", "13.2")) == 0, "13.2 规则库仍为空")

    rec.section("[5] 原文不落库")
    from app.db import GuideSource

    with _session() as s:
        rows = s.query(GuideSource).all()
        rec.check(any(x.url for x in rows), "来源元信息已登记")
        blob = " ".join((x.note or "") + (x.title or "") for x in rows)
        rec.check("连胜达到 3 以上" not in blob, "攻略原文片段未入库")
        rec.check("版权所有" not in blob, "噪音未入库")

    rec.section("[6] 确认入库")
    sid = body["session_id"]
    r = client.post("/api/guides/collect/confirm",
                    json={"session_id": sid, "accept": []})
    rec.check(r.json().get("imported") == [], "空列表 = 一条都不要")

    r = client.post("/api/guides/collect/confirm",
                    json={"session_id": "nonexistent", "accept": ["x"]})
    rec.check(r.status_code == 404, "不存在的会话被拒")

    r = client.post("/api/guides/text", json={
        "text": GUIDE_HTML, "game": "tft", "version": "13.2",
        "guide_title": "云顶 13.2 版本运营要点", "min_confidence": 0.4,
    })
    sid = r.json()["session_id"]
    titles = [c["title"] for c in r.json()["candidates"]][:2]
    r = client.post("/api/guides/collect/confirm",
                    json={"session_id": sid, "accept": titles})
    if rec.check(r.status_code == 200, f"确认入库成功（{r.status_code}）"):
        out = r.json()
        rec.check(len(out["imported"]) == 2, f"入库 {len(out['imported'])} 条")
        rec.check(out["skipped"] > 0, f"跳过 {out['skipped']} 条未确认的")

    with _session() as s:
        rules = get_rules(s, "tft", "13.2")
        rec.check(len(rules) == 2, f"13.2 规则库现有 {len(rules)} 条")
        rec.check(all(r.source for r in rules), "每条规则都有溯源")
        rec.check(all(r.source.url for r in rules), "溯源含 URL")

    rec.section("[7] 入库后能被分析链路使用")
    r = client.post("/api/matches", json={
        "match_id": "tft-after-collect", "game": "tft", "game_version": "13.2",
        "player": {"nickname": "示例玩家"}, "outcome": {"placement": 4},
        "played_at": "2026-10-15 20:00",
        "rounds": [{"stage": 2, "phase": 1,
                    "facts": {"health": 100, "level": 5, "gold": 60, "streak": 0,
                              "income": 5, "item_count": 0,
                              "has_main_carry": False, "is_decision_point": False,
                              "pressure": "low", "round_tempo": "normal"},
                    "actions": [{"kind": "econ_action", "value": "reroll",
                                 "confidence": 0.95}],
                    "missing_facts": []}],
    })
    if rec.check(r.status_code == 200, f"新版本对局可分析（{r.status_code}）"):
        rec.check(r.json().get("stats", {}).get("triggered", 0) > 0,
                  "采集来的规则触发了判定")


# ==========================================================================
# 场景 4：规则库审核
# ==========================================================================


def scenario_rule_review(client, rec: Recorder) -> None:
    """审核工作台：覆盖率现算、改动留痕、人工改也走准入。"""
    rec.section("[0] 前置")
    client.post("/api/games/tft/rules/seed")
    seed_fixture_library()
    if not rec.check(True, "夹具库已载入"):
        return

    rec.section("[1] 审核视图")
    r = client.get("/api/games/tft/rules/review")
    if not rec.check(r.status_code == 200, f"数据可取（{r.status_code}）"):
        return
    d = r.json()
    rec.check(d["version"] == "13.1", f"版本 {d['version']}")
    rec.check(d["summary"]["total"] == 12, f"规则数 {d['summary']['total']}")
    rec.check(bool(d["fact_labels"]), "带事实中文名")
    rec.check(d["summary"]["matched_matches"] == 0, "无样本时不谎报覆盖")
    rec.check(d["summary"]["never_count"] == 12, "无样本时全部标记为从未触发")

    rec.section("[2] 灌历史后覆盖率现算")
    from demo_data import build_demo_match

    client.post("/api/matches", json=build_demo_match("tft"))
    d = client.get("/api/games/tft/rules/review").json()
    cov = d["summary"]
    rec.check(cov["matched_matches"] == 1, f"样本对局 {cov['matched_matches']}")
    rec.check(cov["never_count"] < 12, f"仍有 {cov['never_count']} 条从未触发")
    items = {x["rule_id"]: x for x in cov["items"]}
    e1 = items.get("TFT-ECON-001")
    rec.check(e1 and e1["checked"] > 0, f"ECON-001 被检验 {e1 and e1['checked']} 次")

    rec.section("[3] 人工修改留痕")
    r = client.patch("/api/games/tft/rules/13.1/TFT-ECON-001", json={
        "changes": {"trigger": {"all": [
            {"fact": "stage", "op": "==", "value": 4},
            {"fact": "phase", "op": "==", "value": 1},
            {"fact": "streak", "op": ">=", "value": 5},
            {"fact": "health", "op": ">=", "value": 50},
        ], "any": [], "none": []}},
        "reviewer": "示例玩家", "note": "攻略原文是连胜 5 以上",
    })
    rec.check(r.status_code == 200, f"修改成功（{r.status_code}）")

    rule = next(x for x in client.get("/api/games/tft/rules/review").json()["rules"]
                if x["rule_id"] == "TFT-ECON-001")
    rec.check(rule["edited"] is True, "标记为人工改过")
    rec.check(rule["original_payload"] is not None, "保留抽取器原始产出")
    rec.check(rule["reviewed_by"] == "示例玩家", "记录审核人")
    rec.check("trigger" in rule["edited_fields"], "标明改了哪个字段")
    rec.check(len(rule["history"]) >= 1, f"修改历史 {len(rule['history'])} 条")
    os_ = next(c for c in rule["original_payload"]["trigger"]["all"]
               if c["fact"] == "streak")
    cs_ = next(c for c in rule["payload"]["trigger"]["all"] if c["fact"] == "streak")
    rec.check(os_["value"] == 3 and cs_["value"] == 5,
              f"原件 streak={os_['value']} 改后 {cs_['value']}")

    rec.section("[4] 人工改也走准入")
    r = client.patch("/api/games/tft/rules/13.1/TFT-ECON-001", json={
        "changes": {"trigger": {"all": [
            {"fact": "根本不存在的事实", "op": ">=", "value": 1},
        ], "any": [], "none": []}}, "reviewer": "示例玩家",
    })
    rec.check(r.status_code == 422, f"未注册事实被拒（{r.status_code}）")
    rec.check("未在" in str(r.json().get("detail", "")), "错误信息说明原因")

    r = client.patch("/api/games/tft/rules/13.1/TFT-ECON-001", json={
        "changes": {"expect": {"fact": "pressure", "op": "==", "value": "high"}},
        "reviewer": "示例玩家",
    })
    rec.check(r.status_code == 422, f"expect 挂在观测事实上被拒（{r.status_code}）")

    r = client.patch("/api/games/tft/rules/13.1/TFT-ECON-001", json={
        "changes": {"game": "jcc"}, "reviewer": "示例玩家",
    })
    rec.check(r.status_code == 422, "只读字段被拒")

    rec.section("[5] 停用与恢复")
    r = client.post("/api/games/tft/rules/13.1/TFT-RHYTHM-001/status",
                    json={"status": "deprecated", "reviewer": "示例玩家",
                          "note": "样本不足"})
    rec.check(r.status_code == 200, f"停用成功（{r.status_code}）")
    d = client.get("/api/games/tft/rules/review").json()
    rec.check(d["summary"]["deprecated"] == 1, f"已停用 {d['summary']['deprecated']} 条")
    rule = next(x for x in d["rules"] if x["rule_id"] == "TFT-RHYTHM-001")
    rec.check(rule["edited"] is False, "停用不计入「人工改过」")

    client.post("/api/games/tft/rules/13.1/TFT-RHYTHM-001/status",
                json={"status": "active", "reviewer": "示例玩家"})
    rec.check(client.get("/api/games/tft/rules/review").json()["summary"]["active"]
              == 12, "恢复后重新生效")

    r = client.post("/api/games/tft/rules/13.1/TFT-RHYTHM-001/status",
                    json={"status": "deleted"})
    rec.check(r.status_code == 422, "非 active/deprecated 被拒")

    rec.section("[6] 工作台页面")
    r = client.get("/api/rules/review", params={"game": "tft"})
    html = r.text
    visible = strip_script(html)
    rec.check(r.status_code == 200, f"页面可打开（{r.status_code}）")
    rec.check("{{" not in html, "无未渲染占位符")
    for bad in ("undefined", "NaN", "[object Object]"):
        rec.check(bad not in visible, f"可见内容无 {bad!r}")
    rec.check("从未触发" in visible, "页面出现从未触发提示")
    rec.check("修改阈值" in visible, "页面提供修改入口")

    rec.section("[7] 修改真的影响分析")
    client.post("/api/matches", json={**build_demo_match("tft"),
                                      "match_id": "after-edit"})
    r = client.get("/api/matches/after-edit/report.json")
    gaps = [g for g in r.json()["gaps"] if g["rule_id"] == "TFT-ECON-001"]
    rec.check(not gaps, "改后 ECON-001 不再触发（阈值已抬高）")


# --------------------------------------------------------------------------
# 英雄层闭合链路
# --------------------------------------------------------------------------


def _hok_round(minute: int, actions: List[Dict[str, Any]], **facts) -> Dict[str, Any]:
    """造一个王者回合。

    时间必须给准 —— 英雄层的时间窗是按攻略的节点切的
    （0:00 / 2:00 / 4:00 / 6:00 / 10:00 / 12:00 之后），
    时间不对就一条都触发不了，而报告上只是「没结论」。
    """
    base = {"gold": 1200, "gold_diff": 0, "level": 3, "tower_diff": 0,
            "kill_diff": 0}
    base.update(facts)
    return {
        "stage": minute // 5, "phase": 1, "time_min": minute,
        "facts": base,
        "actions": actions,
        "missing_facts": [],
    }


def scenario_hero_layer(client, rec: Recorder) -> None:
    """英雄层闭合链路：规则库 → 录入 → 判定 → 报告 → 标注。

    这一组盯的是**整条链**通不通，以及两件事：

    1. 选了英雄就走英雄层，分路通则让路 —— 同一句建议不被说两遍；
    2. 没选英雄时仍然有话可说（回落到分路通则）——
       英雄层是加法，不是把通则删掉。
    """
    rec.section("[1] 规则库：分路层 + 英雄层一起载入")
    r = client.post("/api/games/hok/rules/seed")
    rec.check(r.status_code == 200, f"载入规则库（{r.status_code}）")
    body = r.json()
    rec.check(body.get("accepted_count", 0) > 1500,
              f"载入的规则数够（{body.get('accepted_count')}）")
    rec.check(not body.get("rejected"), "没有规则被准入拦下")

    rules = client.get("/api/games/hok/rules",
                        params={"version": "S45"}).json()["rules"]
    lane_rules = [x for x in rules if x["id"].startswith("HOK-MODE-")]
    hero_rules = [x for x in rules if x["id"].startswith("HOK-HERO-")]
    rec.check(len(lane_rules) == 84, f"分路层 84 条（{len(lane_rules)}）")
    rec.check(len(hero_rules) > 1500, f"英雄层 {len(hero_rules)} 条")
    rec.check(all(x["trigger"]["all"][0]["fact"] == "position" for x in lane_rules),
              "分路层规则都带分路条件")
    rec.check(all(any(c["fact"] == "my_hero" for c in x["trigger"]["all"])
                  for x in hero_rules), "英雄层规则都钉了英雄")

    rec.section("[2] 模式库页面：分路 → 英雄 → 模式")
    html = client.get("/rules", params={"lane": "side"}).text
    rec.check("该分路前中后期总体操作模式" in html, "分路页有总体模式那一节")
    rec.check('class="hero-cards"' in html, "分路页有英雄那一节")
    hero_html = client.get("/rules", params={"lane": "side", "hero": "哪吒"}).text
    rec.check("该英雄前中后期操作模式" in hero_html, "英雄页有该英雄的模式")
    rec.check("对抗路-哪吒-前期-1" in hero_html, "英雄页的模式名带英雄")
    rec.check("对抗路-吕布" not in hero_html, "英雄页里没有别人的模式")

    rec.section("[3] 录入：选了哪吒 → 只走英雄层")
    r = client.post("/api/matches", json={
        "match_id": "hero-nz-1", "game": "hok", "game_version": "S45",
        "player": {"nickname": "示例玩家", "position": "side",
                   "hero": "哪吒", "matchup": "advantage"},
        "outcome": {"result": "win"}, "played_at": "2026-10-21 21:00",
        "rounds": [
            _hok_round(2, [{"kind": "positioning_action", "value": "push_wave",
                            "label": "推线压塔", "confidence": 1.0}]),
            _hok_round(6, [{"kind": "teamfight_action", "value": "engage",
                            "label": "开团先手", "confidence": 1.0}]),
            _hok_round(11, [{"kind": "objective_action", "value": "take_dragon",
                             "label": "拿下小龙", "confidence": 1.0}]),
        ],
    })
    rec.check(r.status_code == 200, f"录入成功（{r.status_code}）")
    ids = {f["mode_id"] for f in r.json()["lanes"]["findings"]}
    rec.check(bool(ids), f"判出了结论（{len(ids)} 条）")
    rec.check(all(i.startswith("HOK-HERO-") for i in ids),
              f"选了英雄就只判英雄层：{sorted(i for i in ids if not i.startswith('HOK-HERO-'))}")
    rec.check(any(i.startswith("HOK-HERO-SID-NZ-") for i in ids),
              "结论来自哪吒自己的对抗路打法")

    rec.section("[4] 报告：三段式 + 准确时间节点")
    mid = r.json()["match_id"]
    rj = client.get(f"/api/matches/{mid}/report.json").json()
    gaps = rj.get("gaps") or []
    rec.check(bool(gaps), f"报告里有偏离（{len(gaps)} 条）")
    for g in gaps[:5]:
        rec.check(bool(g.get("time_label")), f"偏离带时间节点：{g.get('rule_id')}")
        rec.check(bool(g.get("actual_text")), f"偏离写了实际：{g.get('rule_id')}")
        rec.check(bool(g.get("expected_text")), f"偏离写了该做：{g.get('rule_id')}")
        rec.check(bool(g.get("why_text")), f"偏离写了为什么：{g.get('rule_id')}")
        why = g.get("why_text") or ""
        for bad in ("原文前提：", "原文结果：", "应该", "尽量"):
            rec.check(bad not in why, f"「为什么」里没有元叙述 {bad!r}：{why[:40]}")

    html = client.get(f"/api/matches/{mid}/report").text
    visible = strip_script(html)
    rec.check("{{" not in html, "报告无未渲染占位符")
    # 英雄层规则的标题带英雄名（「哪吒 2:00：…」）——
    # 派生模式名（对抗路-哪吒-前期-1）是模式库页用的，报告不显示。
    rec.check("哪吒" in visible, "报告里出现英雄层模式（标题带英雄名）")
    rec.check("原文前提：" not in visible, "报告里没有「原文前提：」这种元叙述")
    rec.check("第 2 分钟" in visible or "第 6 分钟" in visible or
              "第 11 分钟" in visible, "报告里出现准确时间节点")

    rec.section("[5] 没选英雄：回落到分路通则")
    r = client.post("/api/matches", json={
        "match_id": "no-hero-1", "game": "hok", "game_version": "S45",
        "player": {"nickname": "示例玩家", "position": "side", "matchup": "advantage"},
        "outcome": {"result": "lose"}, "played_at": "2026-10-21 21:05",
        "rounds": [
            _hok_round(2, [{"kind": "positioning_action", "value": "push_wave",
                            "label": "推线压塔", "confidence": 1.0}]),
        ],
    })
    rec.check(r.status_code == 200, f"录入成功（{r.status_code}）")
    ids = {f["mode_id"] for f in r.json()["lanes"]["findings"]}
    rec.check(any(i.startswith("HOK-MODE-") for i in ids),
              f"没填英雄时仍判分路通则：{sorted(ids)}")
    rec.check(not any(i.startswith("HOK-HERO-") for i in ids),
              "没填英雄就不该判英雄层")

    rec.section("[6] 标注：英雄下拉列全")
    from app.adapters.honor_of_kings import HEROES_BY_LANE

    offered = {h for v in HEROES_BY_LANE.values() for h in v}
    rec.check(len(offered) == 123, f"下拉里有 123 个英雄（{len(offered)}）")
    rec.check("哪吒" in offered, "下拉里有哪吒")
    rec.check("周瑜" in offered, "下拉里有周瑜（跨分路英雄）")

    rec.section("[7] 审核工作台：英雄层也在库里")
    r = client.get("/api/games/hok/rules/review",
                   params={"game": "hok", "version": "S45"})
    rec.check(r.status_code == 200, f"工作台可打开（{r.status_code}）")
    view = r.json()
    rec.check(view["summary"]["total"] > 1500,
              f"工作台列出了全部规则（{view['summary']['total']}）")
    hero_ids = [x["rule_id"] for x in view["rules"] if x["rule_id"].startswith("HOK-HERO-")]
    rec.check(bool(hero_ids), "工作台里有英雄层规则")


# --------------------------------------------------------------------------
# 会话辅助
# --------------------------------------------------------------------------


def _session():
    from app.db import session_scope

    return session_scope()


# --------------------------------------------------------------------------
# 场景 6：BP（选人禁用）阶段
# --------------------------------------------------------------------------

#: 一局典型的 BP 记录。刻意留了三处偏离：
#:   禁用阶段禁了坦边（而己方正缺前排）
#:   一抢抢了核心输出（攻略说一抢不锁射手法师野核）
#:   三抢拿摇摆位（而己方还缺开团点）
BP_DEMO_RECORD = {
    "my_missing_frontline": True,
    "my_missing_engage": True,
    "my_missing_protect": False,
    "my_missing_damage": True,
    "opp_missing_frontline": True,
    "decisions": [
        {"side": "my", "phase": "ban_1", "action": "ban_tank", "hero": "廉颇"},
        {"side": "opp", "phase": "ban_1", "action": "ban_opp_core", "hero": "赵怀安"},
        {"side": "my", "phase": "pick_1", "action": "pick_core_output", "hero": "敖隐"},
        {"side": "opp", "phase": "pick_1", "action": "pick_counter", "hero": "兰陵王"},
        {"side": "my", "phase": "pick_2", "action": "pick_damage", "hero": "沈梦溪"},
        {"side": "opp", "phase": "pick_2", "action": "pick_damage", "hero": "甄姬"},
        {"side": "my", "phase": "pick_3", "action": "pick_flex", "hero": "澜"},
        {"side": "opp", "phase": "pick_3", "action": "pick_frontline", "hero": "牛魔"},
        {"side": "my", "phase": "pick_last", "action": "pick_engage", "hero": "牛魔"},
    ],
}


def scenario_bp_layer(client, rec: Recorder) -> None:
    """BP 阶段：规则库、录入、判定、复盘。

    BP 在比赛之前，判的是「这一步该禁谁、该选谁」——
    所以它的坐标是「哪个环节」，结论也落在单独一页。
    """
    rec.section("[1] BP 规则库")
    r = client.post("/api/games/hok/rules/seed")
    rec.check(r.status_code == 200, f"载入规则库（{r.status_code}）")
    body = r.json()
    rec.check(body.get("accepted_count") == 1680,
              f"局内库 1680 条（{body.get('accepted_count')}）")
    bp = body.get("bp") or {}
    rec.check(bp.get("accepted_count", 0) > 0,
              f"BP 库也载入了（{bp.get('accepted_count')} 条）")
    rec.check(bp.get("version") == "BP-S45",
              f"BP 库是独立版本（{bp.get('version')}）")

    rec.section("[2] BP 原文与解析")
    from tools.hok_bp_sources import bp_guide_path, verify_sources
    from tools.hok_bp_tree import load_bp_guide

    rec.check(verify_sources() == [], "BP 原文 sha256 一致")
    guide = load_bp_guide(bp_guide_path())
    rec.check(len(guide.nodes) == 8, f"8 个决策环节（{len(guide.nodes)}）")
    rec.check(len(guide.branches) == 36, f"36 条分支（{len(guide.branches)}）")

    rec.section("[3] 官方英雄定位表")
    from tools.hok_hero_tags import build_tags, load_raw, verify_sources as vt

    rec.check(vt() == [], "官方英雄资料 sha256 一致")
    rec.check(len(build_tags(load_raw())) > 100, "英雄属性表有内容")

    rec.section("[4] 录一局带 BP 的对局")
    from demo_data import build_demo_match

    payload = build_demo_match("hok")
    payload["match_id"] = "hok-bp-scenario"
    r = client.post("/api/matches", json=payload)
    rec.check(r.status_code == 200, f"录对局（{r.status_code}）")
    mid = r.json()["match_id"]

    r = client.post(f"/api/matches/{mid}/bp", json=BP_DEMO_RECORD)
    rec.check(r.status_code == 200, f"录 BP（{r.status_code}）")
    if r.status_code == 200:
        rec.check(r.json()["entries"] > 0,
                  f"BP 复盘有结论（{r.json()['entries']} 条）")

    rec.section("[5] BP 复盘页")
    r = client.get(f"/api/matches/{mid}/bp/report.json")
    rec.check(r.status_code == 200, f"复盘 JSON 可取（{r.status_code}）")
    if r.status_code == 200:
        rep = r.json()
        entries = rep.get("entries") or []
        rec.check(bool(entries), f"有结论（{len(entries)} 条）")
        for e in entries[:6]:
            rec.check(bool(e.get("step_label")),
                      f"结论带坐标：{e.get('rule_id')}")
            rec.check(bool(e.get("my_move")),
                      f"结论写了这一手做了什么：{e.get('rule_id')}")
            rec.check(bool(e.get("advice_text")),
                      f"结论写了该怎么做：{e.get('rule_id')}")
            rec.check(bool(e.get("reason_text")),
                      f"结论写了为什么：{e.get('rule_id')}")
        summary = rep.get("summary") or {}
        rec.check(bool(summary.get("my_heroes")), "概要里有己方阵容")

    html = client.get(f"/api/matches/{mid}/bp/report").text
    visible = strip_script(html)
    rec.check("{{" not in html, "BP 复盘页无未渲染占位符")
    rec.check("BP 复盘" in visible, "页面标题是 BP 复盘")
    rec.check("HOK-BP-" not in visible, "BP 复盘页里没有规则 ID")

    rec.section("[6] 局内报告不受 BP 规则影响")
    match_html = client.get(f"/api/matches/{mid}/report").text
    rec.check("BP 复盘" not in match_html, "局内报告里没有 BP 结论")
    rec.check("留一个 ban 位" not in match_html, "局内报告里没有 BP 的建议")

    rec.section("[7] 没录 BP 就不给复盘")
    payload2 = build_demo_match("hok")
    payload2["match_id"] = "hok-bp-none"
    payload2.pop("bp", None)
    r = client.post("/api/matches", json=payload2)
    rec.check(r.status_code == 200, f"录第二局（{r.status_code}）")
    r = client.get("/api/matches/hok-bp-none/bp/report.json")
    rec.check(r.status_code == 404, "没录 BP 的对局给 404")
    rec.check("BP" in r.json().get("detail", ""), "说清楚要先录 BP")

    rec.section("[8] 非法的 BP 记录要报错")
    r = client.post(f"/api/matches/{mid}/bp", json={
        "decisions": [{"side": "my", "phase": "ban_1",
                       "action": "ban_nothing", "hero": ""}]})
    rec.check(r.status_code == 422, f"不认识的动作类别报 422（{r.status_code}）")
    rec.check("ban_nothing" in r.json().get("detail", ""),
              "错误里点名了不认识的类别")

    rec.section("[9] 对局记录页有 BP 入口")
    html = client.get("/matches").text
    rec.check("/bp" in html, "对局记录页里有 BP 入口")


#: 场景注册表。verify_*.py 与 pytest 都从这里取，避免两处各写一份。
SCENARIOS = {
    "version_flow": scenario_version_flow,
    "video_contract": scenario_video_contract,
    "guide_collect": scenario_guide_collect,
    "rule_review": scenario_rule_review,
    "hero_layer": scenario_hero_layer,
    "bp_layer": scenario_bp_layer,
}
