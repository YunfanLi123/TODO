"""模型裁决的解析与**硬校验**。

模型看着画面说「这一条不符合」，这句话不能直接进报告。
本模块是裁决与报告之间的闸门，四道闸：

1. **模式必须存在** —— 模型引用了清单里没有的模式 id，一律丢掉。
2. **前提必须被画面证实** —— 攻略的取舍由前提决定
   （「塔的护盾没了」「暴君进倒计时」「对方减员」）。前提在画面里
   看不出来时，那一刻玩家做的**可能就是对的** ——
   原文说「该推塔」，可当时塔还有护盾、敌方打野就在我方半区，
   他退塔下才是对的。拿这种前提去纠正人，就是在冤枉。
3. **玩家做了这个节点认可的事就不纠正** —— 攻略在一个时段往往给好几套
   打法，玩家做了其中**另一件**时，他这会儿是在按攻略做事，
   只是跟规则库写的那一条不一样。
4. **引用的帧必须真实存在** —— 报告里「原视频操作的文字描述」说的
   是模型在**哪一帧**看到的事，片段播放器要放的就是那一帧。
   帧号对不上，文字和画面对不上号，整条结论就跟着塌了。

还有一条兜底：**这一窗什么动作都没读出来**时，一律判「看不出来」——
「看不出来」不等于「没做」。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

#: 裁决的四种取值。
VERDICT_OK = "ok"                # 这一条做对了
VERDICT_DEVIATION = "deviation"  # 前提成立，但做的不是这一条
VERDICT_NEVER = "never"          # 前提成立，可这一条该做的事整段没做
VERDICT_UNCLEAR = "unclear"      # 看不出来 —— **不是**「没做」

#: 模型回答里「前提是否看得见」的取值。
PREMISE_VISIBLE = "visible"
PREMISE_NOT_VISIBLE = "not_visible"
PREMISE_UNKNOWN = "unknown"

#: 裁决条目的两种类型。
KIND_MODE = "mode"    # 逐条偏离
KIND_NODE = "node"    # 整段没做这个时段该做的任何一件事


def _norm(value: Any, allowed: Sequence[str], default: str) -> str:
    text = str(value or "").strip().lower()
    for raw, canon in allowed:
        if text == raw:
            return canon
    return default


def _frame_numbers(raw: Any, frame_count: int) -> List[int]:
    """只留**真实存在**的帧号（1 起）。"""
    out: List[int] = []
    for item in raw or []:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= idx <= frame_count and idx not in out:
            out.append(idx)
    return out


def _action_pairs(actions: Sequence[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """这一窗读到的动作 -> ``(种类, 取值)`` 列表。"""
    out: List[Tuple[str, str]] = []
    for act in actions or []:
        kind = str(act.get("kind") or "")
        value = str(act.get("value") or "")
        if kind and value:
            out.append((kind, value))
    return out


def _subject_applies(subject: Optional[Dict[str, Any]],
                     pairs: Sequence[Tuple[str, str]]) -> bool:
    """这条规则的**适用时机**命中了没有。

    没有适用时机（``subject`` 为空）时一律算命中 ——
    那条规则管的是整个时间窗。
    """
    if not subject:
        return True
    fact = str(subject.get("fact") or "")
    values = [str(v) for v in (subject.get("values") or [])]
    if not fact:
        return True
    for kind, value in pairs:
        if kind != fact:
            continue
        if not values or value in values:
            return True
    return False


def _did_endorsed(node_actions: Dict[str, List[str]],
                   pairs: Sequence[Tuple[str, str]]) -> bool:
    """玩家做的这些事里，有**这个节点认可**的吗。

    ``node_actions`` 是这个时段攻略给出的全部打法（按动作事实分组）。
    玩家做了其中任何一件 —— 说明他这会儿在按攻略做事，
    **不纠正他**。他做得没错，只是跟规则库写的那一条不一样。
    """
    for kind, value in pairs:
        if value in (node_actions.get(kind) or []):
            return True
    return False


def _expect_met(entry: Dict[str, Any],
                pairs: Sequence[Tuple[str, str]]) -> bool:
    """这一条**要求**的事，玩家做了没有。

    这是最后一道闸，也是最硬的一道：模式要求「带兵线推塔」，
    玩家推了塔，那就不管他别的地方做得好不好 —— 这一条他做到了，
    不能再拿它去纠正他。

    没有这道闸的话，模型说一句「不符合」就成立 —— 而模型看的是
    几帧画面，它自己也会看错。
    """
    fact = str(entry.get("expect_fact") or "")
    values = [str(v) for v in (entry.get("expect_values") or [])]
    if not fact or not values:
        return False
    for kind, value in pairs:
        if kind == fact and value in values:
            return True
    return False


def parse_review(
    answer: Any,
    checklist: Sequence[Dict[str, Any]],
    frame_count: int,
    actions: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """把模型回答里的裁决解析出来，并**逐条校验**。

    返回 ``(条目, 说明)``。说明是给内部看的诊断（为什么这条被丢掉），
    不进报告 —— 报告只讲结论。

    条目的形状::

        {
          "kind": "mode" | "node",
          "mode_id": "HOK-MODE-JGL-T0400-02",
          "lane": "jungle",
          "verdict": "deviation" | "never" | "ok" | "unclear",
          "did": "这一段在刷红 buff 野区",
          "why": "第 3 帧龙坑图标在闪，暴君快刷了",
          "frames": [3, 5],
          "confidence": 0.8,
        }
    """
    notes: List[str] = []
    items: List[Dict[str, Any]] = []
    if not isinstance(answer, dict):
        return items, ["模型回答不是一个对象"]

    by_id: Dict[str, Dict[str, Any]] = {}
    for entry in checklist:
        rid = str(entry.get("id") or "")
        if rid:
            by_id[rid] = entry

    pairs = _action_pairs(actions)
    #: 这一窗一个动作都没读出来 —— 什么都判不了。
    blind = not pairs

    raw_items = answer.get("review")
    if isinstance(raw_items, list):
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            mode_id = str(raw.get("mode_id") or "").strip()
            entry = by_id.get(mode_id)
            if entry is None:
                notes.append(f"模型引用了清单里没有的模式 {mode_id!r}（已丢弃）")
                continue

            frames = _frame_numbers(raw.get("frames"), frame_count)
            matched = _norm(raw.get("matched"), (("yes", "yes"), ("no", "no"),
                                                  ("unclear", "unclear"),
                                                  ("true", "yes"), ("false", "no")),
                            "unclear")
            premise = _norm(raw.get("premise"), (("visible", PREMISE_VISIBLE),
                                                  ("not_visible", PREMISE_NOT_VISIBLE),
                                                  ("false", PREMISE_NOT_VISIBLE),
                                                  ("true", PREMISE_VISIBLE),
                                                  ("unknown", PREMISE_UNKNOWN)),
                            PREMISE_UNKNOWN)

            if matched == "yes":
                verdict = VERDICT_OK
            elif matched == "unclear":
                verdict = VERDICT_UNCLEAR
            elif blind:
                # 这一窗什么动作都没读出来 → 判不了，**不是**「没做」。
                verdict = VERDICT_UNCLEAR
            elif _expect_met(entry, pairs):
                # 这一条要求的事玩家做到了 → 这一条他没错。
                verdict = VERDICT_OK
            elif premise != PREMISE_VISIBLE:
                # 前提没被画面证实 → 这一刻玩家做的可能就是对的，不纠正。
                verdict = VERDICT_UNCLEAR
            elif _did_endorsed(entry.get("node_actions") or {}, pairs):
                # 玩家做了**这个节点**认可的另一件事 →
                # 他是在按攻略做事，只是跟这一条不一样。不纠正他。
                verdict = VERDICT_UNCLEAR
            elif not _subject_applies(entry.get("subject"), pairs):
                # 适用时机不命中：玩家这一帧做的不是这条规则管的事。
                verdict = VERDICT_UNCLEAR
            else:
                verdict = VERDICT_DEVIATION

            if verdict == VERDICT_DEVIATION and not frames:
                # 没有可核对的帧 → 这条结论没法在原片上验证，不给。
                notes.append(f"{mode_id} 没有可用的帧号（已丢弃）")
                continue

            try:
                conf = float(raw.get("confidence", 0.6))
            except (TypeError, ValueError):
                conf = 0.6

            # **哪一支成立**。
            #
            # 攻略一个节点往往给好几支（「敌方红开就己方蓝开，敌方蓝开就
            # 己方红开」）。玩家读报告时最想知道的不是这张菜单，而是
            # 「**这一局是哪一支**」—— 模型指认了哪支，报告就给出那一支的
            # 具体做法，而不是把整张菜单摆出来让人自己配对。
            branch_no = 0
            try:
                branch_no = int(raw.get("branch") or 0)
            except (TypeError, ValueError):
                branch_no = 0
            branches = entry.get("branches") or []
            branch = None
            if 1 <= branch_no <= len(branches):
                branch = branches[branch_no - 1]

            items.append({
                "kind": KIND_MODE,
                "mode_id": mode_id,
                "lane": entry.get("lane") or "",
                "lane_label": entry.get("lane_label") or "",
                "title": entry.get("title") or "",
                "premise": entry.get("premise") or "",
                "action": entry.get("action") or "",
                "outcome": entry.get("outcome") or "",
                "advice": entry.get("advice") or "",
                # 这一局**适用的是哪一支**（原文那一支，不是整张菜单）。
                "branch_no": branch_no,
                "branch_premise": (branch or {}).get("premise") or "",
                "branch_action": (branch or {}).get("action") or "",
                "branch_outcome": (branch or {}).get("outcome") or "",
                "verdict": verdict,
                "did": str(raw.get("did") or "").strip(),
                # 「看到了什么」= 画面上的客观状态（模型原话）。
                "evidence": str(raw.get("evidence") or "").strip(),
                # 「凭什么」= 结论与画面的连接。
                "why": str(raw.get("why") or "").strip(),
                "frames": frames,
                "confidence": max(0.0, min(1.0, conf)),
            })

    # -- 整段没做：这个时段该做的任何一件事都没做 -------------------------
    raw_node = answer.get("node_gap")
    if isinstance(raw_node, dict):
        lane = str(raw_node.get("lane") or "").strip()
        options = [e for e in checklist
                    if not lane or e.get("lane") == lane]
        if len(options) < 2:
            # 只有一条路可选时不叫「菜单」——
            # 「你没做那一件事」和「你没做任何一件事」不是一回事。
            notes.append("这个时段只有一条路，不做「整段没做」的判断")
        elif blind:
            notes.append("这一窗动作都没读出来，不做「整段没做」的判断")
        else:
            node_actions: Dict[str, List[str]] = {}
            for entry in options:
                for fact, values in (entry.get("node_actions") or {}).items():
                    bucket = node_actions.setdefault(fact, [])
                    for value in values:
                        if value not in bucket:
                            bucket.append(value)
            if _did_endorsed(node_actions, pairs):
                notes.append("玩家做过这个时段认可的事，不做「整段没做」的判断")
            else:
                frames = _frame_numbers(raw_node.get("frames"), frame_count)
                if not frames:
                    notes.append("「整段没做」没有可用的帧号（已丢弃）")
                else:
                    try:
                        conf = float(raw_node.get("confidence", 0.6))
                    except (TypeError, ValueError):
                        conf = 0.6
                    items.append({
                        "kind": KIND_NODE,
                        "mode_id": "",
                        "lane": lane,
                        "lane_label": options[0].get("lane_label") or "",
                        "title": "",
                        "premise": "",
                        "action": "",
                        "outcome": "",
                        "advice": "",
                        "verdict": VERDICT_NEVER,
                        "did": str(raw_node.get("did") or "").strip(),
                        "why": str(raw_node.get("why") or "").strip(),
                        "frames": frames,
                        "confidence": max(0.0, min(1.0, conf)),
                        # 这一时段攻略给的几条路 —— 建议是「挑一条做」，
                        # 不是「该做某一条」。
                        "options": [
                            {"id": e.get("id") or "",
                             "title": e.get("title") or "",
                             "action": e.get("action") or "",
                             "outcome": e.get("outcome") or ""}
                            for e in options
                        ],
                    })

    return items, notes
