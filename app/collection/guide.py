"""采集层之二：攻略来源导入。

立项设计的合规立场（必须保留）：
    **只存结构化规则，不存攻略原文全文。**
    系统定位为「分析方」而非「内容生产方」—— 责任边界清晰。
    因此本模块拿到攻略文本后，只提取规则并丢弃原文；
    数据库里留下的是结构化规则 + 必要的出处元信息与短引用。

流程：
    攻略文本 → 抽取候选（extractor）→ 人工确认 → 准入筛选 → 入库
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..adapters.base import get_adapter
from ..core.dsl import Rule
from ..db import GuideSource
from ..rules import extractor as ex
from ..rules.extractor import RuleCandidate


@dataclass
class GuideImportResult:
    """一次攻略导入的完整交代。"""

    game: str
    version: str
    guide_title: str
    #: 抽取出的候选（含未通过准入的）
    candidates: List[RuleCandidate] = field(default_factory=list)
    #: 直接被主观词表拦下的句子
    rejected_subjective: List[str] = field(default_factory=list)
    #: 已入库的规则 id
    imported: List[str] = field(default_factory=list)
    #: 入库失败的候选及原因
    failed: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game": self.game,
            "version": self.version,
            "guide_title": self.guide_title,
            "candidate_count": len(self.candidates),
            "rejected_subjective": self.rejected_subjective,
            "imported": self.imported,
            "failed": self.failed,
            "candidates": [c.to_dict() for c in self.candidates],
        }


def ingest_guide(
    session,
    text: str,
    game: str,
    version: str,
    guide_title: str,
    guide_url: str = "",
    guide_kind: str = "community",
    accept: Optional[Sequence[str]] = None,
    min_confidence: float = 0.4,
) -> GuideImportResult:
    """导入一份攻略。

    ``accept`` 是人工确认后放行的候选标题集合。传入 ``None`` 表示只抽取
    不入库（预览模式）；传入空列表表示人工确认后**一条都不要**。
    这个设计强制了「候选必须经过人的判断」这道闸门。
    """
    adapter = get_adapter(game)
    facts = adapter.known_facts()

    candidates = ex.extract_candidates(
        text=text,
        game=game,
        version=version,
        known_facts=facts,
        guide_title=guide_title,
        guide_url=guide_url,
        min_confidence=min_confidence,
    )

    result = GuideImportResult(
        game=game,
        version=version,
        guide_title=guide_title,
        candidates=candidates,
    )

    # 记录攻略来源元信息（不含原文全文）
    source = GuideSource(
        game=game, title=guide_title, url=guide_url, kind=guide_kind,
        note=f"抽取 {len(candidates)} 条候选规则",
    )
    session.add(source)
    session.flush()

    result.rejected_subjective = [
        s for s in ex.split_sentences(text) if ex._detect_subjective(s)
    ]

    if accept is None:
        return result   # 预览模式，不入库

    # 人工确认环节
    from ..rules.store import import_rules   # 延迟导入，避免循环依赖

    accepted_titles = set(accept)
    action_facts = adapter.known_action_facts()
    to_import: List[Rule] = []
    prefix = game.upper()[:3]

    for cand in candidates:
        if cand.title not in accepted_titles:
            continue

        rule_id = _make_rule_id(prefix, cand)
        rule = cand.to_rule(rule_id)
        sub = import_rules(
            session, [rule], facts, strict=False, action_facts=action_facts
        )
        if sub.accepted:
            result.imported.append(rule_id)
        else:
            result.failed.append(
                {"rule_id": rule_id, "title": cand.title, "reasons": sub.rejected}
            )

    return result


def _make_rule_id(prefix: str, cand: RuleCandidate) -> str:
    """生成稳定规则 id：``PREFIX-CATEGORY-NNN``。

    编号按「类别 + 标题摘要」稳定化 —— 同一句攻略重复导入会得到同一个 id，
    不会产生重复条目。
    """
    import hashlib

    cat = _slug(cand.category or "misc").upper()[:8]
    digest = hashlib.sha1(cand.title.encode("utf-8")).hexdigest()[:4]
    return f"{prefix}-{cat}-{digest}"


def _slug(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum():
            keep.append(ch)
        elif ch in " -_/":
            keep.append("-")
    return "".join(keep).strip("-") or "misc"
