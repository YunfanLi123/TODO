"""人工写规则 + 来源登记。

到这一步，规则库有三条来路：

  1. 种子规则     官方维护，随版本走
  2. 攻略抽取     从文本抽出候选，人工核对后入库
  3. **手写**     抽取器搞不定、或你就是知道自己想要什么

第三条是补上前两条的窟窿。抽取器是启发式的，理解不了「这条是常识不用写」
也理解不了「这条要跨三个条件一起看」；而手写一条规则只需要在表单里填
几个数字。规则库要的是质量，不是全自动。

来源登记同理：系统**不替你决定该看谁的攻略**。默认拒绝一切域名这件事
在采集器里已经写死了，这里做的是让「允许这个站点」变成一个可以点一下
就能记录的决定，而不是每次都要去配置文件里翻。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.dsl import DEFAULT_SUBJECTIVE_MARKERS, Condition, Rule, Source, Trigger, check_admission


class RuleWriteError(ValueError):
    pass


def _rule_id(game: str, category: str, title: str) -> str:
    """规则 id：前缀-类别-标题摘要。

    标题进摘要而不是自增序号 —— 同一句话反复提交会得到同一个 id，
    天然去重，也不会出现「规则库里有三条几乎一样的规则」而没人知道哪条
    是当初写的那条。
    """
    import hashlib

    prefix = game.upper()[:3]
    cat = "".join(c for c in category.upper() if c.isalnum())[:8] or "MISC"
    digest = hashlib.sha1(title.strip().encode("utf-8")).hexdigest()[:4]
    return f"{prefix}-{cat}-{digest}"


def write_rule(
    session: Session,
    game: str,
    version: str,
    payload: Dict[str, Any],
    overwrite: bool = False,
) -> Dict[str, Any]:
    """写入一条人工规则。

    走的是和抽取候选**完全相同**的准入筛选 —— 手写不代表可以跳过校验。
    手写反而更容易出错：事实名拼错、操作符选反，这些都会被拦住。
    """
    from ..adapters.base import get_adapter

    adapter = get_adapter(game)

    title = (payload.get("title") or "").strip()
    if not title:
        raise RuleWriteError("标题不能为空")

    trigger_spec = payload.get("trigger") or {}
    expect_spec = payload.get("expect") or {}

    def mk(items):
        out = []
        for c in items or []:
            fact = (c.get("fact") or "").strip()
            op = (c.get("op") or "").strip()
            if not fact or not op:
                raise RuleWriteError("触发条件缺 fact 或 op")
            v = c.get("value")
            if isinstance(v, str):
                v = v.strip()
                n = None
                try:
                    n = float(v)
                except ValueError:
                    n = None
                if n is not None and v.replace(".", "", 1).replace("-", "", 1).isdigit():
                    v = int(n) if n == int(n) else n
                elif v.startswith("[") and v.endswith("]"):
                    v = [x.strip().strip("'\"") for x in v[1:-1].split(",") if x.strip()]
            out.append(Condition(fact, op, v))
        return out

    conds = mk(trigger_spec.get("all"))
    if not conds:
        raise RuleWriteError("至少要有一条触发条件 —— 否则这条规则对每局都生效")

    expect_fact = (expect_spec.get("fact") or "").strip()
    if not expect_fact:
        raise RuleWriteError("必须指定期望动作（fact）")
    ev = expect_spec.get("value")
    if isinstance(ev, str):
        ev = [x.strip() for x in ev.split("/") if x.strip()]
        ev = ev or None

    rule = Rule(
        id=_rule_id(game, payload.get("category") or "未分类", title),
        game=game,
        version=(version or "").strip(),
        title=title,
        category=(payload.get("category") or "未分类").strip(),
        trigger=Trigger(all=conds, any=mk(trigger_spec.get("any")),
                        none=mk(trigger_spec.get("none"))),
        expect_fact=expect_fact,
        expect_op=(expect_spec.get("op") or "in").strip(),
        expect_value=ev,
        severity=(payload.get("severity") or "medium").strip(),
        rationale=(payload.get("rationale") or "").strip(),
        deviation_hint=(payload.get("deviation_hint") or "").strip(),
        tags=list(payload.get("tags") or []),
        source=Source(
            guide=(payload.get("source_guide") or "人工编写").strip(),
            locator=(payload.get("source_locator") or "").strip(),
            url=(payload.get("source_url") or "").strip(),
            quoted=title,
            collected_at=__import__("datetime").date.today().isoformat(),
        ),
    )

    reasons = check_admission(
        rule, adapter.known_facts(), DEFAULT_SUBJECTIVE_MARKERS,
        adapter.known_action_facts(),
    )
    # 主观词不拦手写：人写「尽量」有时是有意的口语化表达，
    # 结构性问题（未注册事实、期望动作非动作、缺触发条件）才拦。
    structural = [r for r in reasons if "主观判断" not in r]
    if structural:
        raise RuleWriteError("；".join(structural))

    # 只构造并校验，不落库 —— 落库由 write_rules_bulk 统一做，
    # 这样批量写入时「构造成功但后面失败」的半截数据不会留在库里。
    return {"rule": rule, "warnings": [r for r in reasons if "主观判断" in r]}


def write_rules_bulk(
    session: Session,
    game: str,
    version: str,
    rules: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """批量写入（攻略页的「写入规则库」走这里）。"""
    from ..rules.store import upsert_rule

    imported: List[str] = []
    failed: List[Dict[str, Any]] = []

    for payload in rules or []:
        try:
            out = write_rule(session, game, version, payload)
            upsert_rule(session, out["rule"])
            imported.append(out["rule"].id)
        except RuleWriteError as exc:
            failed.append({
                "title": payload.get("title"),
                "reasons": [str(exc)],
            })
        except Exception as exc:                       # noqa: BLE001
            failed.append({"title": payload.get("title"), "reasons": [str(exc)]})

    session.flush()
    return {
        "imported": imported,
        "failed": failed,
        "message": f"写入 {len(imported)} 条"
                   + (f"，拒绝 {len(failed)} 条" if failed else ""),
    }


# --------------------------------------------------------------------------
# 来源登记
# --------------------------------------------------------------------------

SOURCE_KINDS = {"official", "community", "user"}


def add_source(
    session: Session,
    game: str,
    title: str,
    url: str,
    kind: str = "user",
    note: str = "",
) -> Dict[str, Any]:
    """登记一个攻略来源。

    登记**不等于授权抓取** —— 抓取仍要走采集器的白名单校验，
    两者刻意分开：知道一个站点存在，和允许程序去访问它，是两件事。
    """
    from ..rules.store import register_guide_source

    title = (title or "").strip()
    if not title:
        raise RuleWriteError("来源名称不能为空")
    if kind not in SOURCE_KINDS:
        kind = "user"
    src = register_guide_source(
        session, game=game, title=title, url=(url or "").strip(),
        kind=kind, note=note,
    )
    return {"id": src.id, "title": src.title, "url": src.url, "kind": src.kind}


def list_sources(session: Session, game: Optional[str] = None) -> List[Dict[str, Any]]:
    from ..rules.store import list_guide_sources

    out = []
    for x in list_guide_sources(session, game):
        out.append({
            "id": x.id,
            "game": x.game,
            "title": x.title,
            "url": x.url,
            "kind": x.kind,
            "note": x.note,
            "collected_at": x.collected_at.isoformat() if x.collected_at else "",
        })
    return out


def delete_source(session: Session, game: str, title: str) -> int:
    from ..db import GuideSource

    rows = session.execute(
        select(GuideSource).where(
            GuideSource.game == game, GuideSource.title == title
        )
    ).scalars().all()
    for r in rows:
        session.delete(r)
    session.flush()
    return len(rows)
