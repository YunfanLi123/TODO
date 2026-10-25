"""采集层之一：对局数据录入。

三条录入路径（对应立项设计里的「采集层」）：
  manual  玩家手工录入 / 修正回合数据
  api     从第三方数据源按结构化格式导入
  video   视频上传（扩展阶段）—— 回合门控 + ROI 精细识别的产物在此落库

设计上把「录入」和「分析」严格分开：录入只负责把原始数据存成
:class:`MatchContext` 友好的 JSON，分析是独立的一步。这样同一局可以用
不同版本的规则库反复分析，而原始数据永不丢失。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..adapters.base import GameAdapter, get_adapter
from ..core.context import ActionRecord, MatchContext, RoundRecord
from ..db import Analysis, Match

#: 低于此置信度的识别结果，报告里必须显式标注为「待人工确认」。
MANUAL_REVIEW_THRESHOLD = 0.75


class MatchValidationError(ValueError):
    pass


@dataclass
class ValidationIssue:
    level: str          # error / warn
    scope: str          # match / round / action
    message: str

    def to_dict(self) -> Dict[str, str]:
        return {"level": self.level, "scope": self.scope, "message": self.message}


def validate_raw(adapter: GameAdapter, raw: Dict[str, Any]) -> List[ValidationIssue]:
    """录入前的数据质量检查。

    原则：**宁可报错也不要静默通过**。一条错误数据进了库，
    后面所有基于它的结论都不可信，而报告的全部价值就建立在可信上。
    """
    issues: List[ValidationIssue] = []

    if not raw.get("rounds"):
        issues.append(
            ValidationIssue("error", "match", "对局缺少回合数据，无法分析")
        )
        return issues

    if not raw.get("game_version"):
        issues.append(
            ValidationIssue("warn", "match", "未标注游戏版本，将无法匹配对应规则库")
        )

    seen_labels: List[str] = []
    for rnd in raw.get("rounds", []):
        label = f"{rnd.get('stage', '?')}-{rnd.get('phase', 1)}"
        seen_labels.append(label)

        if "stage" not in rnd:
            issues.append(
                ValidationIssue("error", "round", f"{label}: 缺少 stage 字段")
            )
            continue

        if not rnd.get("facts") and not rnd.get("actions"):
            issues.append(
                ValidationIssue(
                    "warn",
                    "round",
                    f"{label}: 回合既无事实也无动作，对该回合不会有任何判定",
                )
            )

        for act in rnd.get("actions", []):
            kind = act.get("kind", "")
            if kind not in adapter.fact_names:
                issues.append(
                    ValidationIssue(
                        "error", "action", f"{label}: 动作类型 {kind!r} 未在 {adapter.game_id} 注册"
                    )
                )
            conf = act.get("confidence", 1.0)
            if isinstance(conf, (int, float)) and conf < MANUAL_REVIEW_THRESHOLD:
                issues.append(
                    ValidationIssue(
                        "warn",
                        "action",
                        f"{label}: 动作 {act.get('value')} 置信度 {conf:.2f} 偏低，报告中将标记待人工确认",
                    )
                )

        for fact in (rnd.get("facts") or {}):
            if fact not in adapter.fact_names:
                issues.append(
                    ValidationIssue(
                        "error",
                        "round",
                        f"{label}: 事实 {fact!r} 未在 {adapter.game_id} 注册",
                    )
                )

    dupes = {l for l in seen_labels if seen_labels.count(l) > 1}
    if dupes:
        issues.append(
            ValidationIssue("error", "match", f"回合标签重复：{sorted(dupes)}")
        )

    return issues


def normalize_match(
    adapter: GameAdapter, raw: Dict[str, Any]
) -> Tuple[MatchContext, List[ValidationIssue]]:
    """校验并规范化对局数据。"""
    issues = validate_raw(adapter, raw)
    fatal = [i for i in issues if i.level == "error"]
    if fatal:
        raise MatchValidationError(
            "；".join(f"[{i.scope}] {i.message}" for i in fatal)
        )

    payload = dict(raw)
    payload.setdefault("game", adapter.game_id)
    ctx = adapter.build_context(payload)

    # 适配器 normalize 之后再跑一次结构问题检查（注册事实/未注册动作）
    for rnd in ctx.rounds:
        for problem in adapter.validate_round(rnd):
            issues.append(ValidationIssue("warn", "round", problem))

    return ctx, issues


def save_match(
    session,
    adapter: GameAdapter,
    raw: Dict[str, Any],
    player_id: str = "local",
    source_kind: str = "manual",
) -> Tuple[Match, List[ValidationIssue]]:
    """校验后落库。返回 (Match, 校验问题列表)。"""
    ctx, issues = normalize_match(adapter, raw)

    match_id = ctx.match_id or f"{adapter.game_id}-{ctx.game_version}"
    record = session.execute(
        _match_query(match_id, player_id)
    ).scalar_one_or_none()

    if record is None:
        record = Match(
            id=match_id,
            game=adapter.game_id,
            game_version=ctx.game_version,
            player_id=player_id,
        )
        session.add(record)

    record.source_kind = source_kind
    record.played_at = _parse_played_at(raw.get("played_at"))
    record.set_outcome(ctx.outcome)
    # 存规范化后的完整上下文（含全部回合与事实），保证可重放
    stored = ctx.to_dict()
    # **BP 记录也要带走**：它属于这一局的数据，但不在局内上下文里
    # （BP 在比赛之前，判的是「这一手该禁谁、该选谁」）。
    # 少了这一步，录进来的一局到了 BP 复盘页就变成「还没录 BP」。
    if raw.get("bp"):
        stored["bp"] = raw["bp"]
    record.set_payload(stored)
    session.flush()

    return record, issues


def _match_query(match_id: str, player_id: str):
    from sqlalchemy import select

    return select(Match).where(
        Match.id == match_id, Match.player_id == player_id
    )


#: 对局时间可接受的输入格式。
#:
#: 这里曾经漏了 ``%Y-%m-%d %H:%M``（不带秒），导致系统自己产出的对局数据
#: 全部解析失败、静默回退到「现在」—— 成长曲线的时间轴于是全部塌到同一天，
#: 而「执行度随时间怎么变」恰恰是这个模块存在的理由。静默回退比报错糟得多，
#: 所以这些格式必须覆盖所有真实来源（手工录入 / API / 示例数据）。
PLAYED_AT_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d",
)


def _parse_played_at(value: Any) -> datetime:
    """解析对局时间。

    解析不了就回退到当前时间，但**这会让成长曲线的时间轴塌陷**，
    所以调用方应该优先保证上游给出标准格式。
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        for fmt in PLAYED_AT_FORMATS:
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
    return datetime.utcnow()


def load_match(session, match_id: str, player_id: str = "local") -> Optional[Match]:
    from sqlalchemy import select

    return session.execute(
        select(Match).where(
            Match.id == match_id, Match.player_id == player_id
        )
    ).scalar_one_or_none()


def list_matches(
    session,
    player_id: str = "local",
    game: Optional[str] = None,
    limit: int = 100,
) -> List[Match]:
    """列出对局，**预加载 analyses**。

    预加载不是优化洁癖：成长曲线、版本回测、覆盖率统计都要遍历整份对局
    列表，而每条都会访问 ``match.analyses``。不预加载的话那是典型的 N+1 ——
    60 局就是 63 条 SQL，对局量上千时会先卡在数据库往返上，而不是卡在算法上。

    ``selectinload`` 用一条额外查询换掉 N 条；列表已经一次性查出来了，
    没有任何理由让 ORM 再逐个回表。
    """
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    stmt = select(Match).options(selectinload(Match.analyses))
    stmt = stmt.where(Match.player_id == player_id)
    if game:
        stmt = stmt.where(Match.game == game)
    stmt = stmt.order_by(Match.played_at.desc()).limit(limit)
    return list(session.execute(stmt).scalars())


def count_matches(
    session, player_id: str = "local", game: Optional[str] = None
) -> int:
    """只数数量，不加载对象与关联。

    覆盖率统计需要遍历全部对局，但复盘页面之类的场景往往只要一个总数 ——
    那时把 payload 和 analyses 一起拉出来纯属浪费。
    """
    from sqlalchemy import func, select

    stmt = select(func.count(Match.id)).where(Match.player_id == player_id)
    if game:
        stmt = stmt.where(Match.game == game)
    return int(session.execute(stmt).scalar() or 0)


def save_bp_record(
    session, match_id: str, bp: Dict[str, Any], player_id: str = "local"
) -> bool:
    """把 BP（选人禁用）记录存进对局的 payload。返回是否真的存了。

    BP 记录与局内回合**分开存**：BP 在比赛之前，判定口径、时间坐标、
    动作类别都跟局内不一样，混进 rounds 会污染局内判定。

    找不到对局就返回 False（不报错）：调用方要回答的是「存上没有」，
    而不是「数据库是不是正常的」。
    """
    record = load_match(session, match_id, player_id=player_id)
    if record is None:
        return False
    payload = record.get_payload()
    payload["bp"] = bp
    record.set_payload(payload)
    session.flush()
    return True


def load_bp_record(
    session, match_id: str, player_id: str = "local"
) -> Optional[Dict[str, Any]]:
    """取一局的 BP 记录。没有就返回 None。"""
    record = load_match(session, match_id, player_id=player_id)
    if record is None:
        return None
    bp = (record.get_payload() or {}).get("bp")
    return bp if isinstance(bp, dict) else None


def delete_match(session, match_id: str, player_id: str = "local") -> bool:
    """删掉一局对局。返回是否真的删了。

    关联的 ``analyses`` 由 ORM 的 ``delete-orphan`` 级联删掉 ——
    留着孤儿分析记录的话，删掉一局之后统计口径还会算上它，
    页面看着像没删干净。

    找不到就返回 False（不是报错）：调用方要回答的是「删掉了没有」，
    而不是「数据库是不是正常的」。
    """
    from sqlalchemy import delete

    record = load_match(session, match_id, player_id=player_id)
    if record is None:
        return False
    session.execute(delete(Analysis).where(Analysis.match_id == match_id))
    session.delete(record)
    return True


def delete_matches_by_player(
    session, player_id: str
) -> List[Tuple[str, str]]:
    """删掉某个玩家的**全部**对局，返回 ``[(match_id, video_id), ...]``。

    即焚模式用：关掉网页就把这个设备的对局全清掉。

    视频文件不在数据库里 —— 它躺在磁盘上。所以这里把 video_id 一并带出来，
    由调用方去删文件。漏了这一步，磁盘上会剩一堆没人认领的视频。

    关联的 ``analyses`` 一并删掉（与 :func:`delete_match` 同一个道理：
    留着孤儿分析，统计口径会算上已经不存在的那一局）。
    """
    from sqlalchemy import delete, select

    records = list(
        session.execute(select(Match).where(Match.player_id == player_id)).scalars()
    )
    pairs: List[Tuple[str, str]] = []
    for record in records:
        video_id = (record.get_payload() or {}).get("video_id") or ""
        pairs.append((record.id, str(video_id)))

    for match_id, _video_id in pairs:
        session.execute(delete(Analysis).where(Analysis.match_id == match_id))
    if records:
        session.execute(delete(Match).where(Match.player_id == player_id))
    return pairs


def list_video_ids(session) -> set:
    """当前还有对局引用的视频 id 集合。

    清扫孤儿视频时用它：被引用的不能删 ——
    用户可能开着页面很久没动鼠标，那局还在，视频就该还在。
    """
    from sqlalchemy import select

    out = set()
    for record in session.execute(select(Match)).scalars():
        video_id = (record.get_payload() or {}).get("video_id")
        if video_id:
            out.add(str(video_id))
    return out
