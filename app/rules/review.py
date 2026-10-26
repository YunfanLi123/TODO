"""规则库人工审核。

规则库是整个系统的资产，而它的质量目前只由 API 调用者把关。这个模块把
「看规则 → 改规则 → 留痕迹」变成一件能做的事，并且补上一个只读列表永远
看不到的关键视角：

    **这条规则在真实对局里被触发过吗？**

一条从没触发过的规则，要么阈值写错了（比如把 ``>=50`` 写成 ``>500``），
要么描述的情形在实际对局里根本不会出现。两种都意味着它在悄悄占位 ——
不报错、不生效，只是让规则库看起来比实际更有覆盖。

审核能做的事：

  - 看完整结构：触发条件、期望动作、溯源、抽取器原始产出
  - 改阈值 / 改方向 / 改严重度，而不是只能「要或不要」
  - 停用（标记 deprecated）与恢复
  - 查覆盖率：每条规则被检验过几次、偏离过几次

**修改一定要留痕。** 人工改过的东西和机器抽出来的东西可信度不是一个级别，
而「抽取器把 ``不低于`` 抽成 ``<``」这种方向性错误，靠读当前规则是发现不了的 ——
只有把原件和改后版本摆在一起才看得见。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.dsl import Condition, Rule
from ..db import RuleRecord, RuleRevision, utcnow

#: 允许人工修改的字段。只开放这几个 —— 改 id、改游戏、改版本会破坏
#: 规则库的主键语义和版本隔离，那不是「审核」该做的事。
EDITABLE_FIELDS = {"title", "category", "severity", "status",
                   "trigger", "expect", "rationale", "deviation_hint"}


class ReviewError(ValueError):
    pass


# --------------------------------------------------------------------------
# 修改历史
# --------------------------------------------------------------------------


def _record_revision(
    session: Session,
    game: str,
    version: str,
    rule_id: str,
    field_name: str,
    before: Any,
    after: Any,
    note: str = "",
    edited_by: str = "",
) -> None:
    session.add(
        RuleRevision(
            game=game,
            version=version,
            rule_id=rule_id,
            field=field_name,
            before=json.dumps(before, ensure_ascii=False) if before is not None else "",
            after=json.dumps(after, ensure_ascii=False) if after is not None else "",
            note=note,
            edited_by=edited_by,
        )
    )


def revision_history(
    session: Session, game: str, version: str, rule_id: str
) -> List[Dict[str, Any]]:
    """某条规则的修改历史，按时间正序。"""
    stmt = (
        select(RuleRevision)
        .where(
            RuleRevision.game == game,
            RuleRevision.version == version,
            RuleRevision.rule_id == rule_id,
        )
        .order_by(RuleRevision.created_at, RuleRevision.id)
    )
    return [
        {
            "field": r.field,
            "before": json.loads(r.before) if r.before else None,
            "after": json.loads(r.after) if r.after else None,
            "note": r.note,
            "edited_by": r.edited_by,
            "created_at": r.created_at.isoformat() if r.created_at else "",
        }
        for r in session.execute(stmt).scalars()
    ]


# --------------------------------------------------------------------------
# 修改
# --------------------------------------------------------------------------


@dataclass
class UpdateResult:
    rule_id: str
    changed_fields: List[str] = field(default_factory=list)
    revision_count: int = 0
    message: str = ""


def update_rule(
    session: Session,
    game: str,
    version: str,
    rule_id: str,
    changes: Dict[str, Any],
    reviewer: str = "",
    note: str = "",
) -> UpdateResult:
    """修改一条规则并留痕。

    ``changes`` 只需给出要改的字段。改 ``trigger`` / ``expect`` 时给的是
    完整结构（不是补丁）—— 阈值调整往往要连带改别处，给补丁反而容易
    留下前后矛盾的条件。
    """
    record = session.execute(
        select(RuleRecord).where(
            RuleRecord.game == game,
            RuleRecord.version == version,
            RuleRecord.rule_id == rule_id,
        )
    ).scalar_one_or_none()

    if record is None:
        raise ReviewError(f"规则 {rule_id} 不在 {game} {version} 中")

    unknown = set(changes) - EDITABLE_FIELDS
    if unknown:
        raise ReviewError(
            f"不可修改的字段：{sorted(unknown)}；可修改：{sorted(EDITABLE_FIELDS)}"
        )

    current = Rule.from_dict(json.loads(record.payload))
    before = current.to_dict()

    # 先在副本上改并校验 —— 校验不过就不碰数据库
    updated = Rule.from_dict(json.loads(record.payload))
    for key, value in changes.items():
        if key == "trigger":
            updated.trigger = _build_trigger(value)
        elif key == "expect":
            updated.expect_fact = value["fact"]
            updated.expect_op = value["op"]
            updated.expect_value = value.get("value")
        else:
            setattr(updated, key, value)

    # 人工改完必须重新走一遍完整准入，而不是只跑语法校验。
    # 只跑 validate 会漏掉「期望动作必须挂在动作事实上」这类约束 ——
    # 之前 import_rules 拦得住，人工改路径却放行，等于开了个后门。
    #
    # 注意 RuleValidationError 不继承 ReviewError，必须显式转译 ——
    # 漏掉这一句的话，非法修改会变成 500 内部错误，而不是 422 加一句
    #「这个事实没注册」，审核者看到的只会是「服务器炸了」。
    from ..adapters.base import get_adapter
    from ..core.dsl import DEFAULT_SUBJECTIVE_MARKERS, check_admission

    adapter = get_adapter(game)
    try:
        updated.validate(adapter.known_facts())
        reasons = check_admission(
            updated,
            adapter.known_facts(),
            DEFAULT_SUBJECTIVE_MARKERS,
            adapter.known_action_facts(),
        )
    except Exception as exc:                      # noqa: BLE001
        raise ReviewError(f"修改后未通过可判定性校验：{exc}") from exc

    # 主观词检查对人工改的规则意义不同：人写「尽量」有时是有意的口语化。
    # 所以这里只拦「结构性」问题（未注册事实、expect 非动作、缺触发条件、
    # 缺溯源），把纯措辞类的判断留给审核者自己。
    structural = [
        r for r in reasons
        if "主观判断" not in r
    ]
    if structural:
        raise ReviewError("修改后未通过准入校验：" + "；".join(structural))

    after = updated.to_dict()
    changed = [
        k for k in EDITABLE_FIELDS
        if before.get(k) != after.get(k)
    ]

    if not changed:
        return UpdateResult(rule_id, [], 0, "内容无变化，未写入")

    for key in changed:
        _record_revision(
            session, game, version, rule_id, key,
            before.get(key), after.get(key), note, reviewer,
        )

    record.payload = json.dumps(after, ensure_ascii=False)
    record.title = after["title"]
    record.category = after["category"]
    record.severity = after["severity"]
    record.status = after["status"]
    # edited 表示「人工调整过规则内容」。仅切换启用/停用是生命周期操作，
    # 不算内容修改 —— 混在一起会让「人工改过」这个统计失真，
    # 审核者会以为有几条规则被调过，其实只是被停用了。
    content_changed = bool(set(changed) - {"status"})
    if content_changed:
        record.edited = 1
    record.reviewed_by = reviewer
    record.reviewed_at = utcnow()
    record.review_note = note
    if content_changed and not record.original_payload:
        record.original_payload = json.dumps(before, ensure_ascii=False)
    record.updated_at = utcnow()
    session.flush()

    return UpdateResult(
        rule_id=rule_id,
        changed_fields=changed,
        revision_count=len(changed),
        message=f"已修改 {len(changed)} 个字段",
    )


def _build_trigger(spec: Dict[str, Any]) -> Any:
    from ..core.dsl import Trigger

    def mk(items):
        out = []
        for c in items or []:
            out.append(Condition(c["fact"], c["op"], c.get("value")))
        return out

    return Trigger(all=mk(spec.get("all")), any=mk(spec.get("any")),
                   none=mk(spec.get("none")))


def set_status(
    session: Session,
    game: str,
    version: str,
    rule_id: str,
    status: str,
    reviewer: str = "",
    note: str = "",
) -> UpdateResult:
    """启用 / 停用一条规则。

    停用用 ``deprecated`` 而不是删除：历史对局复盘仍需当时的规则状态。
    """
    if status not in {"active", "deprecated"}:
        raise ReviewError(f"status 只能是 active 或 deprecated，收到 {status!r}")
    return update_rule(
        session, game, version, rule_id,
        {"status": status}, reviewer=reviewer, note=note,
    )


# --------------------------------------------------------------------------
# 覆盖率：这条规则到底有没有被用过
# --------------------------------------------------------------------------


@dataclass
class RuleCoverage:
    """单条规则的覆盖情况。"""

    rule_id: str
    checked: int = 0          # 历史对局里被检验过的回合数
    deviated: int = 0
    matches: int = 0          # 参与统计的对局数

    @property
    def rate(self) -> Optional[float]:
        if self.checked == 0:
            return None
        return self.deviated / self.checked

    @property
    def verdict(self) -> str:
        """给审核者的一句话判断。

        ``never`` 是这里最该被看见的状态：一条从未被触发过的规则，
        要么阈值写错了，要么描述的情形在真实对局里压根不出现 ——
        两种都意味着它在占位。
        """
        if self.checked == 0:
            return "never"
        if self.checked < 2:
            return "thin"
        if self.deviated == 0:
            return "always_ok"
        return "active"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "checked": self.checked,
            "deviated": self.deviated,
            "matches": self.matches,
            "rate": round(self.rate, 4) if self.rate is not None else None,
            "verdict": self.verdict,
        }


def compute_coverage(
    session: Session,
    game: str,
    version: str,
    player_id: str = "local",
) -> Dict[str, RuleCoverage]:
    """用当前规则库重跑历史对局，算出每条规则的覆盖情况。

    为什么必须**现算**而不是读分析快照：快照里只存了「偏离」，
    规则被触发但玩家做对了的情况没有留痕。只看偏离会漏掉
    「这条规则其实一直在生效且玩家一直做对」这类重要信息。
    """
    from ..adapters.base import get_adapter
    from ..collection.match import list_matches
    from ..core.matcher import match_rules
    from .store import get_rules

    adapter = get_adapter(game)
    rules = [r for r in get_rules(session, game, version, include_deprecated=True)]
    coverage = {r.id: RuleCoverage(rule_id=r.id) for r in rules}
    if not rules:
        return coverage

    records = list_matches(session, player_id=player_id, game=game)
    matched = 0
    for record in reversed(records):
        if record.game_version != version:
            continue
        try:
            ctx = adapter.build_context(record.get_payload())
        except Exception:                     # noqa: BLE001
            # 单局数据有问题不该让整份覆盖率统计失败
            continue
        ctx.precompute()

        verdicts = match_rules([r for r in rules if r.status == "active"], ctx)
        if not verdicts:
            continue
        matched += 1
        for v in verdicts:
            cov = coverage.get(v.rule_id)
            if cov is None:
                continue
            cov.checked += 1
            if v.status == "deviated":
                cov.deviated += 1

    for cov in coverage.values():
        cov.matches = matched

    return coverage


def coverage_summary(
    coverage: Dict[str, RuleCoverage]
) -> Dict[str, Any]:
    """覆盖率总览 + 最该被审视的规则清单。"""
    items = [c.to_dict() for c in coverage.values()]
    never = [c for c in items if c["verdict"] == "never"]
    thin = [c for c in items if c["verdict"] == "thin"]
    return {
        "total": len(items),
        "matched_matches": max([c["matches"] for c in items], default=0),
        "never_triggered": never,
        "thin_evidence": thin,
        "never_count": len(never),
        "thin_count": len(thin),
        "items": items,
    }


# --------------------------------------------------------------------------
# 审核视图
# --------------------------------------------------------------------------


def build_review_view(
    session: Session,
    game: str,
    version: Optional[str] = None,
    player_id: str = "local",
    with_coverage: bool = True,
) -> Dict[str, Any]:
    """组装审核工作台所需的全部数据。"""
    from ..adapters.base import get_adapter
    from .store import get_rules, latest_version

    adapter = get_adapter(game)
    version = version or latest_version(session, game) or ""

    records = list(
        session.execute(
            select(RuleRecord).where(
                RuleRecord.game == game, RuleRecord.version == version
            ).order_by(RuleRecord.category, RuleRecord.rule_id)
        ).scalars()
    )

    coverage = compute_coverage(session, game, version, player_id) \
        if with_coverage and version else {}

    rules: List[Dict[str, Any]] = []
    for rec in records:
        current = json.loads(rec.payload)
        original = json.loads(rec.original_payload) if rec.original_payload else None
        cov = coverage.get(rec.rule_id)
        rules.append(
            {
                "rule_id": rec.rule_id,
                "title": rec.title,
                "category": rec.category,
                "severity": rec.severity,
                "status": rec.status,
                "payload": current,
                "original_payload": original,
                "edited": bool(rec.edited),
                "edited_fields": (
                    sorted(
                        k for k in EDITABLE_FIELDS
                        if original and original.get(k) != current.get(k)
                    )
                    if original else []
                ),
                "reviewed_by": rec.reviewed_by,
                "reviewed_at": rec.reviewed_at.isoformat() if rec.reviewed_at else "",
                "review_note": rec.review_note,
                "coverage": cov.to_dict() if cov else None,
                "history": revision_history(session, game, version, rec.rule_id),
            }
        )

    summary = coverage_summary(coverage) if coverage else {
        "total": len(rules), "never_triggered": [], "thin_evidence": [],
        "never_count": 0, "thin_count": 0, "items": [], "matched_matches": 0,
    }
    summary["active"] = sum(1 for r in rules if r["status"] == "active")
    summary["deprecated"] = sum(1 for r in rules if r["status"] == "deprecated")
    summary["edited"] = sum(1 for r in rules if r["edited"])

    categories = sorted({r["category"] for r in rules})

    return {
        "game": game,
        "game_display": adapter.display_name,
        "version": version,
        "categories": categories,
        "fact_labels": dict(adapter.fact_labels),
        "action_labels": dict(adapter.action_labels),
        "action_facts": sorted(adapter.known_action_facts()),
        "rules": rules,
        "summary": summary,
    }
