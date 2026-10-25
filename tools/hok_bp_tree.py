"""BP（选人禁用）阶段决策树攻略的解析器。

与 :mod:`tools.hok_decision_tree` 的关系
--------------------------------------
BP 树用的是**同一套树形语法**（``├─ 前提：`` / ``│  动作：`` / ``│  后果：``），
分支与内容的切法可以复用；不一样的是两处：

1. **节点不是时间节点**，而是「决策环节」：
   ``禁用阶段节点（阶段一）``、``选人阶段节点（一抢）``、``收尾节点  阵容完整性检查``。
   分路决策树的 ``NODE_RE`` 只认 ``0:00 节点`` 这类时间标题，认不出这些。
2. **整份文件按 ``━━━ xxx ━━━`` 分节**，只有「决策树」那一节是树，
   其余是散文（全局前提、分位置选人清单、体系与连锁反应、来源）。

BP 在**比赛之前**，跟局内分钟无关 —— 所以这一层的时间坐标是「决策环节」，
不是分钟。规则库按环节分层，判定时也要按环节分层。

为什么单独一个模块
----------------
分路层与英雄层都是「局内某一分钟该做什么」，BP 是「赛前怎么定胜负条件」，
两者的前提事实、动作类别、判定的时机都不一样。硬塞进现有层里，
最后会得到一批永远不触发的规则 —— 那是项目最反对的东西。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from tools.hok_decision_tree import BRANCH_RE, CONTENT_RE, Branch, Node

#: 分节线：``━━━ 决策树 ━━━``。
SECTION_RE = re.compile(r"^━━━\s*(?P<title>.+?)\s*━━━\s*$")

#: BP 决策环节：原文节点标题 -> (环节编码, 中文名)。
#:
#: 显式列出来，而不是从标题里猜 —— 猜出来的编码在攻略改版时会漂，
#: 而规则 id 是用编码拼的，编码一漂，历史对局就再也对不上规则了。
PHASES: List[tuple] = [
    ("0:00 节点  进入 BP", "bp_start", "进入 BP"),
    ("禁用阶段节点（阶段一）", "ban_1", "禁用（阶段一）"),
    ("选人阶段节点（一抢）", "pick_1", "一抢"),
    ("选人阶段节点（二抢）", "pick_2", "二抢"),
    ("选人阶段节点（三抢）", "pick_3", "三抢"),
    ("选人阶段节点（阶段二一抢）", "pick_late_1", "阶段二一抢"),
    ("选人阶段节点（阶段二末选 · 五楼守门员）", "pick_last", "阶段二末选"),
    ("收尾节点  阵容完整性检查", "final_check", "阵容完整性检查"),
]

#: 环节编码 -> 中文名。
PHASE_LABELS: Dict[str, str] = {code: label for _, code, label in PHASES}

#: 环节顺序 —— 报告与库里都按这个顺序讲 BP。
PHASE_ORDER: List[str] = [code for _, code, _ in PHASES]

#: 分位置选人清单里的字段。
PICK_LIST_FIELDS = ("首选", "次选", "禁选", "位置对手", "依据")


@dataclass
class BpNode:
    """一个决策环节（原文里一个节点标题）。"""

    header: str
    phase: str
    phase_label: str
    branches: List[Branch] = field(default_factory=list)

    @property
    def title(self) -> str:
        return f"{self.phase_label}｜{self.header}"


@dataclass
class PickList:
    """一条分路的选人清单（原文「分位置选人清单」里的一节）。"""

    lane_label: str
    fields: Dict[str, str] = field(default_factory=dict)

    @property
    def prefer(self) -> str:
        return self.fields.get("首选", "")

    @property
    def backup(self) -> str:
        return self.fields.get("次选", "")

    @property
    def ban(self) -> str:
        return self.fields.get("禁选", "")

    @property
    def matchup(self) -> str:
        return self.fields.get("位置对手", "")

    @property
    def basis(self) -> str:
        return self.fields.get("依据", "")


@dataclass
class BpGuide:
    """一份 BP 决策树文件。"""

    path: str
    text: str
    nodes: List[BpNode] = field(default_factory=list)
    #: 非树形的散文段落：章节标题 -> 正文。
    prose: Dict[str, str] = field(default_factory=dict)
    #: 分位置选人清单：分路名 -> 清单。
    pick_lists: List[PickList] = field(default_factory=list)

    @property
    def branches(self) -> List[Branch]:
        """所有分支（带节点信息）。"""
        out: List[Branch] = []
        for node in self.nodes:
            out.extend(node.branches)
        return out

    def node_of(self, branch: Branch) -> BpNode:
        for node in self.nodes:
            if branch in node.branches:
                return node
        raise KeyError(branch.key)


def _split_sections(text: str) -> List[tuple]:
    """按 ``━━━ 标题 ━━━`` 把文件切成 ``[(标题, 正文), ...]``。"""
    sections: List[tuple] = []
    title = ""
    buf: List[str] = []
    for raw in text.split("\n"):
        m = SECTION_RE.match(raw.strip())
        if m:
            if title or buf:
                sections.append((title, "\n".join(buf)))
            title = m.group("title").strip()
            buf = []
            continue
        buf.append(raw)
    if title or buf:
        sections.append((title, "\n".join(buf)))
    return sections


def _phase_of(header: str) -> tuple:
    """原文节点标题 -> (环节编码, 中文名)。认不出就报错。"""
    text = header.strip()
    for pattern, code, label in PHASES:
        if text == pattern:
            return code, label
    # 允许原文在标题后补说明（「收尾节点  阵容完整性检查」这类双空格写法
    # 已经在 PHASES 里逐字列了；这里兜底的是前缀完全相同的写法）。
    for pattern, code, label in PHASES:
        if text.startswith(pattern.split("  ")[0]) and pattern.split("  ")[0] in text:
            return code, label
    raise ValueError(f"BP 攻略里出现了不认识的节点标题：{header!r}")


def _parse_tree(body: str, path: str) -> List[BpNode]:
    """解析「决策树」那一节。"""
    nodes: List[BpNode] = []
    node: Optional[BpNode] = None
    stack: List[Optional[Branch]] = []

    def current_branch() -> Optional[Branch]:
        for b in reversed(stack):
            if b is not None:
                return b
        return None

    def branch_at(depth: int) -> Optional[Branch]:
        if 1 <= depth <= len(stack) and stack[depth - 1] is not None:
            return stack[depth - 1]
        return current_branch()

    for raw in body.split("\n"):
        line = raw.rstrip()
        if not line.strip():
            continue
        stripped = line.strip()

        # 节点标题要先认出来：``CONTENT_RE`` 能匹配**任何**行
        # （缩进组是可选的），所以放第一位判断，
        # 否则节点标题会被当成「上一条动作的续行」吃掉。
        if stripped[0] not in "├└│" and not raw[:1].isspace():
            code, label = _phase_of(stripped)
            node = BpNode(header=stripped, phase=code, phase_label=label)
            nodes.append(node)
            stack = []
            continue

        bm = BRANCH_RE.match(line)
        if bm:
            depth = len(bm.group("ind")) // 3 + 1
            text_after = bm.group("text").strip()
            premise = ""
            if text_after.startswith("前提："):
                premise = text_after[len("前提："):].strip()
            parent = None
            if depth >= 2:
                parent = stack[depth - 2] if depth - 2 < len(stack) else None
            if node is None:
                raise ValueError("BP 树里出现了没有节点的分支")
            branch = Branch(depth=depth, premise=premise, node=node,  # type: ignore[arg-type]
                            parent=parent)
            node.branches.append(branch)
            branch.index = len(node.branches)
            while len(stack) < depth:
                stack.append(None)
            stack[depth - 1] = branch
            for k in range(depth, len(stack)):
                stack[k] = None
            continue

        cm = CONTENT_RE.match(line)
        if cm and stripped:
            rest = cm.group("rest").strip()
            depth = len(cm.group("ind")) // 3
            if not rest:
                continue
            if not rest.startswith(("前提：", "动作：", "后果：")):
                # 排版折行：上一条动作的续行
                target = current_branch()
                if target is not None and target.actions:
                    target.actions[-1] = target.actions[-1] + rest
                continue
            target = branch_at(depth)
            if target is None:
                continue
            if rest.startswith("动作："):
                target.actions.append(rest[len("动作："):].strip())
            elif rest.startswith("后果："):
                target.outcome = rest[len("后果："):].strip()

    return nodes


def _parse_pick_lists(body: str) -> List[PickList]:
    """解析「分位置选人清单」那一节。

    格式::

        对抗路
          首选  夏侯惇（全能坦边…）
          次选  牛魔（游走位英雄开团）、
                狂铁（战边，输出型）
    """
    out: List[PickList] = []
    current: Optional[PickList] = None
    last_field = ""
    for raw in body.split("\n"):
        if not raw.strip():
            continue
        if not raw.startswith(" "):
            current = PickList(lane_label=raw.strip())
            out.append(current)
            last_field = ""
            continue
        if current is None:
            continue
        stripped = raw.strip()
        head = stripped.split("  ")[0].strip() if "  " in stripped else ""
        if head in PICK_LIST_FIELDS:
            rest = stripped[len(head):].strip()
            current.fields[head] = rest
            last_field = head
        elif last_field:
            # 续行：接在上一个字段后面
            current.fields[last_field] = (current.fields.get(last_field, "") + stripped)
    return out


def parse_bp_guide(text: str, path: str = "") -> BpGuide:
    """解析一份 BP 决策树文件。"""
    guide = BpGuide(path=path, text=text)
    for title, body in _split_sections(text):
        if title == "决策树":
            guide.nodes = _parse_tree(body, path)
        elif title == "分位置选人清单（S45）":
            guide.pick_lists = _parse_pick_lists(body)
        else:
            guide.prose[title] = body.strip()
    return guide


def load_bp_guide(path: str) -> BpGuide:
    with open(path, "r", encoding="utf-8") as fh:
        return parse_bp_guide(fh.read(), path)


__all__ = [
    "BP_GUIDE_NAME",
    "BpGuide",
    "BpNode",
    "PHASE_LABELS",
    "PHASE_ORDER",
    "PickList",
    "load_bp_guide",
    "parse_bp_guide",
]
