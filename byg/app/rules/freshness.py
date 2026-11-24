"""操作模式库的**时效治理**：这条模式现在还适用吗？

借鉴自哪里
----------
`github.com/lanking-hub/game_assistant`
（「峡谷智囊 - 王者荣耀 AI 游戏助手」，RAG + Tool Agent）的
``app/knowledge/freshness.py``。

**那个仓库没有 LICENSE 文件**（GitHub API 的 ``license`` 字段为 ``null``，
文件树里也没有 ``LICENSE``）→ 默认「保留所有权利」。
所以这里**只借鉴它的机制与约定，不复制它的代码**。

它解决的问题
------------
王者每赛季都改。攻略写于 S45 的打法，到了 S46 可能就不适用了 ——
而报告里那条「建议」是照着攻略写的，玩家照着改可能改错方向。

它的做法
--------
1. 每条知识带**时效等级**（``stable`` / ``versioned`` / ``time_sensitive`` /
   ``deprecated``）；
2. 另有一份「版本更新记录」（patch notes）—— 那是**证据**，不是结论；
3. 拿证据去判每条知识，给出三样东西：
   ``expire_status``（状态）、``freshness_penalty``（惩罚）、``freshness_reason``（原因）；
4. **缺发布时间就明说**（「缺少发布时间信息，无法确认时效性。」），不装作完整；
5. ``stable`` 的知识**不是永不改变** —— 一旦有相关补丁证据，照样按 ``versioned`` 判。

我们已有什么、缺什么
--------------------
已有：``version``（S45）、``status``（active / deprecated）、显式的退役清单、
按内容哈希的种子指纹。

缺的是**中间那一层**：这条模式**可能**过时了，但还没到退役。
这一层补上，报告就能对「可能过时的建议」给出提示，而不是照着旧攻略让玩家改。

四条约定
--------
1. **证据先行** —— 没有版本更新记录，就只能判「无法确认」，不能判「没问题」；
2. **缺什么明说** —— 判不出来时把缺的那一项写出来；
3. **惩罚分档** —— 轻微调整 / 核心调整 / 机制重做，对应不同的处置；
4. **不自动作废** —— 本模块**只判状态**，不删规则；退役仍走显式的退役清单。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

#: 时效等级。
#:
#: - ``stable`` —— 不随版本变的（地图机制、兵线、塔的规则）；
#: - ``versioned`` —— 跟着版本走的（开局顺序、出装、英雄打法）；
#: - ``time_sensitive`` —— 强时效（临时 BUG、特定活动的打法）；
#: - ``deprecated`` —— 已退役，不再使用。
LEVELS = ("stable", "versioned", "time_sensitive", "deprecated")

#: 判定结果。
EXPIRE_STATUSES = (
    "active",                 # 确认还适用
    "active_with_warning",    # 有轻微调整，仍可用但要核对
    "potentially_outdated",   # 有核心调整，可能已经不适用
    "deprecated",             # 应当作废
    "unknown",                # 判不出来（缺证据）
)

#: 改动级别 -> (判定状态, 惩罚分)。
#:
#: 惩罚分用来**排序**：同样两条建议，一条只是要核对、另一条可能已经过时，
#: 报告里就该把后者排前面、也标得更重。
CHANGE_LEVELS: Dict[str, tuple] = {
    "low": ("active_with_warning", 10),
    "medium": ("potentially_outdated", 35),
    "high": ("deprecated", 90),
    "rework": ("deprecated", 90),
}

#: 判不出来时的惩罚分 —— 比「确认没问题」高，比「可能过时」低。
UNKNOWN_PENALTY = 15


@dataclass(frozen=True)
class FreshnessVerdict:
    """一条模式的时效判定。"""

    #: 时效等级
    level: str = "versioned"
    #: 判定状态（见 :data:`EXPIRE_STATUSES`）
    expire_status: str = "unknown"
    #: 惩罚分（越大越严重）
    penalty: int = 0
    #: 中文原因（空 = 没有要提醒的）
    reason: str = ""
    #: 这条模式写于哪个版本
    rule_version: str = ""
    #: 当前对照的版本
    current_version: str = ""
    #: 支撑这个判定的版本更新记录
    evidence: tuple = ()

    @property
    def ok(self) -> bool:
        """这条模式可以直接拿去用（不用在报告里额外提醒）。"""
        return self.expire_status == "active"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "expire_status": self.expire_status,
            "penalty": self.penalty,
            "reason": self.reason,
            "rule_version": self.rule_version,
            "current_version": self.current_version,
            "evidence": [dict(e) for e in self.evidence],
        }


# --------------------------------------------------------------------------
# 默认等级
# --------------------------------------------------------------------------

#: 类别 -> 时效等级。
#:
#: 默认是 ``versioned``：我们库里的绝大多数规则是**打法**（开局顺序、
#: 带线时机、反野判断），它们跟着版本走。
#: 只有少数是地图/兵线这类硬机制，才算 ``stable``。
CATEGORY_LEVELS: Dict[str, str] = {
    "lane": "versioned",
    "jungle": "versioned",
    "mid": "versioned",
    "roam": "versioned",
    "marksman": "versioned",
    "hero": "versioned",
    "bp": "versioned",
    "mechanics": "stable",
    "map": "stable",
    "economy": "stable",
}


def level_for(rule: Any) -> str:
    """一条规则的默认时效等级。

    规则自己写了 ``freshness_level`` 就用它；没写就按类别推断。
    """
    explicit = getattr(rule, "freshness_level", None)
    if explicit in LEVELS:
        return str(explicit)
    data = rule if isinstance(rule, dict) else {}
    explicit = (data.get("freshness_level") or "")
    if explicit in LEVELS:
        return explicit
    category = (getattr(rule, "category", None) or data.get("category") or "")
    return CATEGORY_LEVELS.get(str(category), "versioned")


# --------------------------------------------------------------------------
# 版本更新记录
# --------------------------------------------------------------------------

#: 版本更新记录的文件名（相对 ``app/seed/``）。
PATCH_NOTES_FILE = "{game}_patch_notes.json"


def patch_notes_path(game: str) -> str:
    from . import library_source

    return os.path.join(library_source.library_dir(),
                        PATCH_NOTES_FILE.format(game=game))


def load_patch_notes(game: str) -> List[Dict[str, Any]]:
    """读「版本更新记录」。读不到就返回空列表（**不是**「没有改动」）。

    返回空列表意味着「我们没记录」—— 判定时会把这件事说出来。
    """
    path = patch_notes_path(game)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    notes = data.get("notes") if isinstance(data, dict) else data
    return [n for n in (notes or []) if isinstance(n, dict)]


def _note_subjects(note: Dict[str, Any]) -> List[str]:
    out = []
    for key in ("subjects", "heroes", "categories", "tags"):
        for v in (note.get(key) or []):
            out.append(str(v))
    return out


def _rule_tokens(rule: Any) -> List[str]:
    """一条规则身上所有可以用来和版本记录对照的「词」。

    英雄名、分路、类别、标签都算 —— 版本记录里写「打野改了」，
    那所有打野类的规则都要被翻出来核对。
    """
    data = rule if isinstance(rule, dict) else {}
    get = (lambda k: getattr(rule, k, None) if not isinstance(rule, dict)
           else data.get(k))
    tokens: List[str] = []
    for key in ("subject", "category"):
        v = get(key)
        if v:
            tokens.append(str(v))
    tags = get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    tokens.extend(str(t) for t in tags)
    hero = get("hero") or get("my_hero")
    if hero:
        tokens.append(str(hero))
    # 触发条件里出现的事实也可能是分类线索（position=jungle 之类）
    trigger = get("trigger")
    if isinstance(trigger, dict):
        for cond in (trigger.get("all") or []):
            if isinstance(cond, dict):
                fact = cond.get("fact")
                val = cond.get("value")
                if fact:
                    tokens.append(str(fact))
                if isinstance(val, str):
                    tokens.append(val)
                elif isinstance(val, (list, tuple)):
                    tokens.extend(str(v) for v in val)
    return [t for t in tokens if t]


def _overlaps(rule: Any, note: Dict[str, Any]) -> bool:
    """这条规则和这条版本记录说的是不是同一件事。

    记录里**没写适用对象**时，一律算相关 —— 宁可多提醒一句，
    也不能让玩家拿着过时打法去改。
    """
    subjects = _note_subjects(note)
    if not subjects:
        return True
    tokens = set(_rule_tokens(rule))
    if not tokens:
        return True
    return bool(tokens & set(subjects))


def _version_after(candidate: str, base: str) -> bool:
    """``candidate`` 的版本是不是比 ``base`` 新。"""
    if not candidate or not base:
        return False
    if candidate == base:
        return False
    return str(candidate) > str(base)


# --------------------------------------------------------------------------
# 判定
# --------------------------------------------------------------------------

def judge(rule: Any, current_version: str = "",
          notes: Optional[Iterable[Dict[str, Any]]] = None) -> FreshnessVerdict:
    """判一条规则的时效。

    ``rule`` 可以是 :class:`app.core.dsl.Rule`，也可以是它的字典形式。
    ``notes`` 是版本更新记录；不给就从种子目录读。
    """
    data = rule if isinstance(rule, dict) else {}
    get = (lambda k: getattr(rule, k, None) if not isinstance(rule, dict)
           else data.get(k))
    rule_version = str(get("version") or "")
    status = str(get("status") or "active")
    level = level_for(rule)

    if status == "deprecated":
        return FreshnessVerdict(
            level="deprecated", expire_status="deprecated", penalty=90,
            reason="这条模式已经退役。", rule_version=rule_version,
            current_version=current_version)

    if notes is None:
        notes = load_patch_notes(str(get("game") or "hok"))

    # 只看**比这条规则新**的版本记录
    relevant = [n for n in notes
                if _version_after(str(n.get("version") or ""), rule_version)
                and _overlaps(rule, n)]

    if not relevant:
        if current_version and _version_after(current_version, rule_version):
            # 有更新的版本，但**没有记录**说明改了什么 → 判不出来，明说
            return FreshnessVerdict(
                level=level, expire_status="unknown", penalty=UNKNOWN_PENALTY,
                reason=(f"这条模式写于 {rule_version}，当前已是 {current_version}，"
                        f"但没有记录到相关改动 —— 无法确认是否仍然适用。"),
                rule_version=rule_version, current_version=current_version)
        return FreshnessVerdict(
            level=level, expire_status="active", penalty=0,
            rule_version=rule_version, current_version=current_version)

    # 有相关改动：取最严重的那条
    worst_status = "active_with_warning"
    worst_penalty = 0
    reasons: List[str] = []
    evidence: List[Dict[str, Any]] = []
    for note in relevant:
        change_level = str(note.get("level") or "medium").lower()
        st, penalty = CHANGE_LEVELS.get(change_level,
                                        CHANGE_LEVELS["medium"])
        if penalty > worst_penalty:
            worst_penalty = penalty
            worst_status = st
        text = str(note.get("text") or "").strip()
        reasons.append(f"{note.get('version')}：{text}" if text
                       else f"{note.get('version')} 有相关调整")
        evidence.append({"version": note.get("version"),
                         "level": change_level,
                         "text": note.get("text") or ""})

    return FreshnessVerdict(
        level=level, expire_status=worst_status, penalty=worst_penalty,
        reason="；".join(reasons), rule_version=rule_version,
        current_version=current_version, evidence=tuple(evidence))


# --------------------------------------------------------------------------
# 报告里那句话
# --------------------------------------------------------------------------

#: 判定状态 -> 报告里的一句话模板。
#:
#: ``{v}`` 是写规则的版本，``{c}`` 是当前版本。
NOTES: Dict[str, str] = {
    "active_with_warning": (
        "这条按 {v} 的版本写的，之后有过相关调整，请核对后再照着改。"
    ),
    "potentially_outdated": (
        "这条按 {v} 的版本写的，之后有过较大改动，可能已经不适用，请自行核对。"
    ),
    "deprecated": "这条已经过时，不再作为建议依据。",
    "unknown": (
        "这条按 {v} 的版本写的，当前已是 {c}，但没有记录到相关改动 —— "
        "无法确认是否仍然适用，请自行核对。"
    ),
}


def note_for(verdict: FreshnessVerdict) -> str:
    """报告里提醒玩家的一句话。判定没问题就返回空串。"""
    if verdict is None or verdict.expire_status == "active":
        return ""
    template = NOTES.get(verdict.expire_status)
    if not template:
        return ""
    return template.format(v=verdict.rule_version or "旧版本",
                           c=verdict.current_version or "当前版本")


def summary(verdicts: Sequence[FreshnessVerdict]) -> Dict[str, Any]:
    """一批判定汇总成一个表（给诊断工具与报告头部用）。"""
    counts: Dict[str, int] = {}
    for v in verdicts or []:
        counts[v.expire_status] = counts.get(v.expire_status, 0) + 1
    worst = max((v.penalty for v in verdicts or []), default=0)
    return {
        "total": len(verdicts or []),
        "counts": counts,
        "worst_penalty": worst,
        "needs_review": sum(1 for v in (verdicts or []) if not v.ok),
    }
