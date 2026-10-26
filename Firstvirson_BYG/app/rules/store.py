"""规则库存储：按游戏 + 版本独立存储，版本切换时处理失效条目。

这是立项设计中「版本适配」机制的落地：
- 规则以 ``(game, version, rule_id)`` 为主键语义，同名规则在不同版本并存
- 版本切换时，旧版本规则被标记为 deprecated 而不是删除 ——
  历史对局仍要用当时生效的规则来复盘，不能用新规则篡改历史结论
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.dsl import (
    DEFAULT_SUBJECTIVE_MARKERS,
    Rule,
    check_admission,
)
from ..db import GuideSource, RuleRecord
from . import library_source

#: 内置兜底目录（可配置来源见 :mod:`app.rules.library_source`）。
SEED_DIR = library_source.BUILTIN_DIR


class RuleLibraryError(RuntimeError):
    pass


@dataclass
class ImportResult:
    """一次规则导入的结果，用于向用户交代「收进来几条、拒了几条、为什么」。"""

    accepted: List[str]
    rejected: List[Tuple[str, List[str]]]
    version: str
    game: str
    #: 本次导入时被标记失效的规则 id（来自旧一版攻略、已被新攻略取代的）。
    retired: List[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.accepted) + len(self.rejected)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game": self.game,
            "version": self.version,
            "total": self.total,
            "accepted_count": len(self.accepted),
            "rejected_count": len(self.rejected),
            "accepted": self.accepted,
            "retired": self.retired,
            "rejected": [
                {"rule_id": rid, "reasons": reasons} for rid, reasons in self.rejected
            ],
        }


# --------------------------------------------------------------------------
# 写入
# --------------------------------------------------------------------------


def upsert_rule(session: Session, rule: Rule) -> None:
    """写入或更新一条规则。"""
    existing = session.execute(
        select(RuleRecord).where(
            RuleRecord.game == rule.game,
            RuleRecord.version == rule.version,
            RuleRecord.rule_id == rule.id,
        )
    ).scalar_one_or_none()

    payload = json.dumps(rule.to_dict(), ensure_ascii=False)

    if existing:
        existing.title = rule.title
        existing.category = rule.category
        existing.severity = rule.severity
        existing.status = rule.status
        existing.payload = payload
        existing.updated_at = datetime.utcnow()
    else:
        session.add(
            RuleRecord(
                game=rule.game,
                version=rule.version,
                rule_id=rule.id,
                title=rule.title,
                category=rule.category,
                severity=rule.severity,
                status=rule.status,
                payload=payload,
            )
        )


def import_rules(
    session: Session,
    rules: Sequence[Rule],
    known_facts: set,
    strict: bool = False,
    action_facts: Optional[set] = None,
) -> ImportResult:
    """导入一批规则，自动执行可判定性准入筛选。

    ``strict=True`` 时，任何一条被拒都抛异常 —— 用于种子规则库的
    CI 校验，确保官方维护的规则不会悄悄混进不可判定的条目。
    """
    accepted: List[str] = []
    rejected: List[Tuple[str, List[str]]] = []
    versions, games = set(), set()

    for rule in rules:
        reasons = check_admission(
            rule, known_facts, DEFAULT_SUBJECTIVE_MARKERS, action_facts
        )
        if reasons:
            rejected.append((rule.id, reasons))
            continue
        upsert_rule(session, rule)
        accepted.append(rule.id)
        versions.add(rule.version)
        games.add(rule.game)

    if strict and rejected:
        detail = "; ".join(f"{rid}: {'/'.join(rs)}" for rid, rs in rejected)
        raise RuleLibraryError(f"以下规则未通过准入筛选：{detail}")

    if strict and len(versions) > 1:
        raise RuleLibraryError(f"单次导入不允许跨版本：{sorted(versions)}")
    if strict and len(games) > 1:
        raise RuleLibraryError(f"单次导入不允许跨游戏：{sorted(games)}")

    session.flush()

    return ImportResult(
        accepted=accepted,
        rejected=rejected,
        version=next(iter(versions), ""),
        game=next(iter(games), ""),
    )


def load_seed_library(session: Session, game: str) -> ImportResult:
    """从 ``app/seed/`` 载入该游戏的操作模式库。

    文件命名约定：``<game>_modes*.json``（旧的 ``<game>_rules*.json``
    也认，规则库改叫操作模式库之前的存量文件不必改名）。
    导入时强制 strict —— 官方维护的模式不允许注水。

    一个文件都没有**不算错误**：云顶 / 金铲铲目前只保留适配器模块，
    还没填资料，返回空结果让调用方照实说「该游戏暂无操作模式库」。
    早先这里抛 ``RuleLibraryError``，页面上就变成一句看不懂的
    「未找到 hok 的种子规则文件」。

    载入的同时会把「退役清单」里的规则标记为失效 ——
    那些规则来自旧一版攻略，已经被新攻略取代。
    它们**不删除**（历史对局复盘仍需当时的规则状态），只是不再生效。
    """
    from ..adapters.base import get_adapter   # 延迟导入，避免循环依赖

    facts = get_adapter(game).known_facts()
    action_facts = get_adapter(game).known_action_facts()
    merged: List[Rule] = []
    for path in sorted(_seed_files(game, skip_bp=True)):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        merged.extend(Rule.from_dict(d) for d in data)

    if not merged:
        return ImportResult(accepted=[], rejected=[], version="", game=game)

    result = import_rules(session, merged, facts, strict=True, action_facts=action_facts)
    result.retired = retire_rules(session, game, result.version, _retired_ids(game))
    return result


#: BP 阶段的规则库版本号。
#:
#: BP 在**比赛之前**，判的是「这一步该禁谁、该选谁」——
#: 事实、动作、时间坐标都跟局内不一样，所以单独一个版本号并存。
#: 局内报告取 ``S45``，BP 复盘取 ``BP-S45``，互不干扰。
#:
#: 版本号用 ``BP-`` 开头还有第二个原因：它按字符串排序时排在 ``S45`` **前面**
#: （``B`` < ``S``），所以就算哪天忘了过滤，它也不会顶掉局内库的「最新版本」。
BP_VERSION = "BP-S45"

#: BP 版本号的前缀。:func:`latest_version` 会跳过带这个前缀的版本 ——
#: 它是局内库的一个**旁支**，不是局内库的新版。
BP_VERSION_PREFIX = "BP-"


def load_bp_seed_library(session: Session, game: str) -> ImportResult:
    """载入 BP 阶段的操作模式库（单独版本号）。

    为什么单独载入：``import_rules`` 在 strict 模式下**不许一次导入跨版本**
    的规则（版本混在一批里，出问题时说不清是哪个版本带来的）。
    BP 库与局内库本来就是两个版本，所以分两次导。
    """
    from ..adapters.base import get_adapter

    adapter = get_adapter(game)
    facts = adapter.known_facts()
    action_facts = adapter.known_action_facts()

    merged: List[Rule] = []
    for path in sorted(_seed_files(game, only_bp=True)):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        merged.extend(Rule.from_dict(d) for d in data)

    if not merged:
        return ImportResult(accepted=[], rejected=[], version="", game=game)

    result = import_rules(session, merged, facts, strict=True, action_facts=action_facts)
    # 退役清单同样管 BP 库：清单里写着的 id 逐个标成失效，
    # 于是「种子里删掉了」这件事在用户的库里也真的消失了。
    # 清单是显式维护的（``app/seed/<game>_retired.json``），
    # 所以「为什么这批规则没了」查得到。
    result.retired = retire_rules(session, game, result.version, _retired_ids(game))
    return result


def _retired_ids(game: str) -> List[str]:
    """退役清单：来自旧一版攻略、已被新攻略取代的规则 id。

    清单落盘成 ``<game>_retired.json``，由维护者显式维护 ——
    「这批规则为什么退役」要能查，不能靠代码里的隐式逻辑。
    """
    path = os.path.join(library_source.library_dir(), f"{game}_retired.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return [str(x) for x in data]


def retire_rules(
    session: Session,
    game: str,
    version: str,
    rule_ids: Iterable[str],
) -> List[str]:
    """把指定规则标记为失效（``deprecated``）。

    只标记、不删除 —— 历史对局复盘仍需当时的规则状态。
    """
    wanted = set(rule_ids)
    if not wanted:
        return []
    stmt = select(RuleRecord).where(
        RuleRecord.game == game,
        RuleRecord.version == version,
        RuleRecord.status == "active",
    )
    marked: List[str] = []
    for record in session.execute(stmt).scalars():
        if record.rule_id not in wanted:
            continue
        record.status = "deprecated"
        record.updated_at = datetime.utcnow()
        payload = json.loads(record.payload)
        payload["status"] = "deprecated"
        record.payload = json.dumps(payload, ensure_ascii=False)
        marked.append(record.rule_id)
    session.flush()
    return marked


def _seed_files(game: str, *, skip_bp: bool = False,
                only_bp: bool = False) -> List[str]:
    """列出该游戏的所有种子文件。

    匹配规则是 **后缀** 而不是前缀：``hok_modes.json`` / ``hok_hero_modes.json``
    / ``hok_rules.json`` 都收。

    早先这里匹配的是前缀 ``hok_modes``，于是英雄层那个文件（叫
    ``hok_hero_modes``）**静默地一次都没被加载过** —— 库看起来是完整的，
    界面也正常，只是英雄专属的模式一条都不生效。文件没被读到，
    而不是报错，这是最坏的一种失败方式。

    BP 阶段的库（``hok_bp_modes.json``）**单独载入** —— 它是另一个版本号，
    而 strict 导入不许一次跨版本。所以这里能只取局内的、只取 BP 的，
    或者两份都要。

    **来源可配置**（见 :mod:`app.rules.library_source`）：用户自己的
    git 仓库优先，内置那份只当兜底。
    """
    return library_source.library_files(game, skip_bp=skip_bp, only_bp=only_bp)


#: 种子指纹的落盘位置（相对数据目录）。
SEED_FINGERPRINT_FILE = "seed_fingerprint.json"


def seed_fingerprint(game: str) -> str:
    """这个游戏的种子文件当前是「哪一版」。

    用**内容哈希**而不是修改时间：打包成 exe 之后，种子文件的 mtime
    是打包那一刻的时间（每次重打包都变），而内容可能一模一样 ——
    按 mtime 判会每次重打包都重灌一遍库（慢，且没必要）。
    内容没变就是没变。
    """
    import hashlib

    h = hashlib.sha256()
    for path in sorted(_seed_files(game)):
        try:
            with open(path, "rb") as fh:
                h.update(os.path.basename(path).encode("utf-8"))
                h.update(fh.read())
        except OSError:
            continue
    return h.hexdigest()


def read_seed_fingerprints() -> Dict[str, str]:
    """上次灌库时，各游戏的种子是哪一版。"""
    import json

    from ..collection import settings

    path = os.path.join(settings.data_dir(), SEED_FINGERPRINT_FILE)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def write_seed_fingerprints(data: Dict[str, str]) -> None:
    import json

    from ..collection import settings

    directory = settings.data_dir()
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, SEED_FINGERPRINT_FILE)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)


# --------------------------------------------------------------------------
# 读取
# --------------------------------------------------------------------------


def get_rules(
    session: Session,
    game: str,
    version: Optional[str] = None,
    include_deprecated: bool = False,
) -> List[Rule]:
    """读取规则库。版本缺省时取该游戏最新的一个版本。"""
    version = version or latest_version(session, game)
    if version is None:
        return []

    stmt = select(RuleRecord).where(
        RuleRecord.game == game, RuleRecord.version == version
    )
    if not include_deprecated:
        stmt = stmt.where(RuleRecord.status == "active")
    stmt = stmt.order_by(RuleRecord.category, RuleRecord.rule_id)

    return [
        Rule.from_dict(json.loads(r.payload))
        for r in session.execute(stmt).scalars()
    ]


def latest_version(session: Session, game: str) -> Optional[str]:
    """该游戏最新的规则库版本。

    版本号形如 ``13.1`` / ``S14`` / ``2026.10``，按字符串降序取最大。
    对自走棋这类小版本号足够可靠；对 MOBA 这类可能需要更复杂的排序规则，
    届时替换此函数即可，调用方无需改动。

    **BP 版本（``BP-`` 开头）不算在内**：BP 是局内库的旁支
    （BP 在比赛之前，事实与动作都不同），它顶掉局内库的「最新版本」的话，
    所有没写明版本号的调用方都会拿到 BP 那 16 条规则 —— 界面上就变成了
    「王者只有 16 条模式」。
    """
    stmt = select(RuleRecord.version).where(RuleRecord.game == game).distinct()
    versions = [
        v for v in session.execute(stmt).scalars()
        if not str(v).startswith(BP_VERSION_PREFIX)
    ]
    return max(versions) if versions else None


def list_versions(session: Session, game: str) -> List[str]:
    stmt = (
        select(RuleRecord.version)
        .where(RuleRecord.game == game)
        .distinct()
    )
    return sorted(session.execute(stmt).scalars(), reverse=True)


def count_rules(session: Session, game: str, version: str) -> Dict[str, int]:
    stmt = select(RuleRecord).where(
        RuleRecord.game == game, RuleRecord.version == version
    )
    rows = session.execute(stmt).scalars().all()
    active = [r for r in rows if r.status == "active"]
    categories = sorted({r.category for r in active})
    return {
        "total": len(rows),
        "active": len(active),
        "deprecated": len(rows) - len(active),
        "categories": len(categories),
    }


def get_rule(session: Session, rule_id: str, version: str, game: str) -> Optional[Rule]:
    record = session.execute(
        select(RuleRecord).where(
            RuleRecord.game == game,
            RuleRecord.version == version,
            RuleRecord.rule_id == rule_id,
        )
    ).scalar_one_or_none()
    return Rule.from_dict(json.loads(record.payload)) if record else None


# --------------------------------------------------------------------------
# 版本切换
# --------------------------------------------------------------------------


def deactivate_missing(
    session: Session,
    game: str,
    version: str,
    present_rule_ids: Iterable[str],
) -> List[str]:
    """版本切换时，把新版本里不存在的规则标记为失效。

    只标记、不删除 —— 历史对局复盘仍需当时的规则状态。
    """
    present = set(present_rule_ids)
    stmt = select(RuleRecord).where(
        RuleRecord.game == game,
        RuleRecord.version == version,
        RuleRecord.status == "active",
    )
    marked: List[str] = []
    for record in session.execute(stmt).scalars():
        if record.rule_id not in present:
            record.status = "deprecated"
            record.updated_at = datetime.utcnow()
            payload = json.loads(record.payload)
            payload["status"] = "deprecated"
            record.payload = json.dumps(payload, ensure_ascii=False)
            marked.append(record.rule_id)
    session.flush()
    return marked


def register_guide_source(
    session: Session,
    game: str,
    title: str,
    url: str = "",
    kind: str = "community",
    note: str = "",
) -> GuideSource:
    src = GuideSource(game=game, title=title, url=url, kind=kind, note=note)
    session.add(src)
    session.flush()
    return src


def list_guide_sources(session: Session, game: Optional[str] = None) -> List[GuideSource]:
    stmt = select(GuideSource).order_by(GuideSource.collected_at.desc())
    if game:
        stmt = stmt.where(GuideSource.game == game)
    return list(session.execute(stmt).scalars())
