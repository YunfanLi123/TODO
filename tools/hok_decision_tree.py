"""S45 分路决策树攻略的解析器。

为什么要有这个模块
------------------
操作模式库现在**由攻略文件生成**，而不是手写 JSON：

    决策树攻略（仓库内唯一真源）
        ↓  本模块解析
    节点 × 分支（前提 / 动作 / 后果，逐字来自原文）
        ↓  映射表（tools/hok_spec/*.py）决定「这条分支要求什么动作」
    app/seed/hok_modes.json

这样做换来三件事：

1. **引用不可能抄错** —— ``quoted`` 是从原文里切出来的，不是手打的。
2. **改攻略不用改库** —— 重新跑一次生成器就行。
3. **库的结构可查** —— 每条规则都能说清自己来自哪个节点的哪条分支。

文件格式
--------
分节用 ``═`` 分隔线；每节一个分路；节内按时间节点分块::

    0:00 节点
    ├─ 前提：敌方打野为红开
    │  动作：己方蓝开反向开局，蓝 BUFF → 蜥蜴 → 狼 → 红 BUFF → 小鸟
    │  后果：避开前期野区碰撞，状态【野区均势】
    └─ 前提：己方打野前期强度弱于对方
       动作：开局方向不变，但进场前先让中单与游走把视野做出来
            直接换到对方另一半野区

缩进是 3 字符一组的树形符号（``│  `` 或三空格），``├─``/``└─`` 后面是分支，
再往右缩进的是分支的内容。内容分三类：**前提** / **动作** / **后果**。
一条「动作」被排版折成两行时，第二行没有标签 —— 它是上一条动作的续行。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

#: 攻略里的分路名 -> 系统里的分路编码。
LANE_CODES: Dict[str, str] = {
    "打野": "jungle",
    "中路": "mid",
    "对抗路": "side",
    "发育路": "carry",
    "游走位": "roam",
}

#: 节点标题。``12:00 之后`` 没有「节点」两个字，所以后缀可选。
NODE_RE = re.compile(
    r"^(?P<time>\d{1,2}:\d{2}(?:\s*-\s*\d{1,2}:\d{2})?)(?:\s*节点|\s*之后)?$"
)

#: 分支行：若干个 3 字符缩进组 + 分支标记。
BRANCH_RE = re.compile(
    r"^(?P<ind>(?:(?:│\s{2})|(?:\s{3}))*)(?P<mark>[├└])─\s(?P<text>.*)$"
)

#: 内容行：若干个 3 字符缩进组 + 正文。
CONTENT_RE = re.compile(r"^(?P<ind>(?:(?:│\s{2})|(?:\s{3}))*)(?P<rest>.*)$")

#: 内容的三个标签。
LABELS = ("前提：", "动作：", "后果：")


def _strip_groups(text: str) -> str:
    """去掉一行的树形缩进，返回正文。"""
    m = CONTENT_RE.match(text)
    if not m:
        return text.strip()
    return m.group("rest").strip()


#: 逐字比对时要忽略的字符：markdown 强调标记 + 树形绘图符号。
#:
#: 树形符号（``│ ├ └ ─``）不是空白，所以只去空白是不够的 ——
#: 攻略里两行「动作」之间夹着 ``│  ``，比对时会多出这个字符，
#: 于是明明是原文的引用却被判成「找不到」。
COMPARE_STRIP = re.compile(r"[\s*│├└─]+")


def normalize_for_compare(text: str) -> str:
    """把文本压成「只剩字符本身」，用于逐字比对。

    引用必须能在原文里**逐字**找到，否则用户没法复核这条模式。
    比对时忽略排版字符（空白、markdown 强调号、树形绘图符），
    但**不忽略任何实义字符** —— 少一个标点都可能改变结论。
    """
    return COMPARE_STRIP.sub("", text)


def _group_count(text: str) -> int:
    """一行的缩进组数（每组 3 字符）。"""
    m = CONTENT_RE.match(text)
    if not m:
        return 0
    return len(m.group("ind")) // 3


@dataclass
class Branch:
    """决策树的一条分支。

    ``premise`` 是本层的前提；父层前提在 :attr:`parent` 上。
    ``actions`` 是这一层的动作（一条分支可以有多条动作）。
    """

    depth: int
    premise: str
    node: "Node"
    parent: Optional["Branch"] = None
    actions: List[str] = field(default_factory=list)
    outcome: str = ""
    #: 在所属节点里的阅读顺序序号（从 1 开始）。
    index: int = 0

    # -- 派生 ---------------------------------------------------------------

    @property
    def path(self) -> List[Branch]:
        """从根到自己的前提链。"""
        chain: List[Branch] = []
        node: Optional[Branch] = self
        while node is not None:
            chain.append(node)
            node = node.parent
        return list(reversed(chain))

    @property
    def key(self) -> str:
        """分支的稳定标识：前提链用 `` / `` 连起来。

        映射表用它来定位分支。用序号会被后面的插入打乱，
        用前提原文则既稳定又可读 —— 映射表里能直接看出这条规则在说什么。
        """
        return " / ".join(b.premise for b in self.path if b.premise)

    @property
    def premise_text(self) -> str:
        """前提链的完整文字（根 → 叶子）。"""
        return "；".join(b.premise for b in self.path if b.premise)

    @property
    def action_text(self) -> str:
        """全部动作拼成一段话。"""
        return "；".join(self.actions)

    def quote(self) -> str:
        """引用原文：动作逐字，带回「动作：」标签。

        拼出来的文字在原文里是**连续**的（中间只有换行和树形符号），
        所以「逐字引用」这件事由解析器保证，而不是靠人抄。
        """
        return "\n".join(f"动作：{a}" for a in self.actions)

    def outcome_text(self) -> str:
        """后果（若中途层没有后果，取最外层有后果的那一层）。"""
        for b in reversed(self.path):
            if b.outcome:
                return b.outcome
        return ""


@dataclass
class Node:
    """一个时间节点（决策树的一层）。"""

    lane_code: str
    lane_label: str
    time_label: str
    header: str
    branches: List[Branch] = field(default_factory=list)

    def find(self, key: str) -> Optional[Branch]:
        """按前提链找分支。"""
        for b in self.branches:
            if b.key == key:
                return b
        return None


@dataclass
class DecisionTree:
    """整份决策树。"""

    lanes: Dict[str, List[Node]] = field(default_factory=dict)
    text: str = ""

    def node(self, lane_code: str, time_label: str) -> Optional[Node]:
        for n in self.lanes.get(lane_code, []):
            if n.time_label == time_label:
                return n
        return None

    def all_branches(self):
        for lane_nodes in self.lanes.values():
            for n in lane_nodes:
                for b in n.branches:
                    yield n, b


def parse(text: str) -> DecisionTree:
    """解析决策树文本。"""
    tree = DecisionTree(text=text)
    lines = text.split("\n")

    lane_code = ""
    lane_label = ""
    node: Optional[Node] = None
    #: 每个深度上最近的一条分支。内容行按深度归属。
    stack: List[Optional[Branch]] = []

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        if set(line.strip()) <= {"═"}:
            continue

        if line.strip() in LANE_CODES:
            lane_label = line.strip()
            lane_code = LANE_CODES[lane_label]
            tree.lanes.setdefault(lane_code, [])
            node = None
            stack = []
            continue

        m = NODE_RE.match(line.strip())
        if m:
            node = Node(
                lane_code=lane_code,
                lane_label=lane_label,
                time_label=re.sub(r"\s*-\s*", "-", m.group("time")),
                header=line.strip(),
            )
            tree.lanes.setdefault(lane_code, []).append(node)
            stack = []
            continue

        bm = BRANCH_RE.match(line)
        if bm and node is not None:
            depth = len(bm.group("ind")) // 3 + 1
            text_after = bm.group("text").strip()
            premise = ""
            if text_after.startswith(LABELS[0]):
                premise = text_after[len(LABELS[0]):].strip()
            parent = None
            if depth >= 2:
                parent = stack[depth - 2] if depth - 2 < len(stack) else None
            branch = Branch(
                depth=depth,
                premise=premise,
                node=node,
                parent=parent,
            )
            node.branches.append(branch)
            branch.index = len(node.branches)
            # 补齐栈
            while len(stack) < depth:
                stack.append(None)
            stack[depth - 1] = branch
            # 这一层以下的分支已经结束 —— 必须清掉，否则后面某条深度 1 的
            # 内容行会被算到上一条深度 2 的分支头上（症状：动作被记错分支）。
            for i in range(depth, len(stack)):
                stack[i] = None
            continue

        cm = CONTENT_RE.match(line)
        if cm and node is not None:
            rest = cm.group("rest").strip()
            depth = len(cm.group("ind")) // 3
            if not rest:
                continue
            # 没有标签的行是上一条动作的续行（排版折行）。
            if not rest.startswith(LABELS):
                target = _current_branch(stack)
                if target is not None and target.actions:
                    target.actions[-1] = target.actions[-1] + rest
                continue
            target = _branch_for_depth(stack, depth)
            if target is None:
                continue
            if rest.startswith(LABELS[1]):
                target.actions.append(rest[len(LABELS[1]):].strip())
            elif rest.startswith(LABELS[2]):
                target.outcome = rest[len(LABELS[2]):].strip()
            # 前提只在分支行上，这里忽略

    return tree


def _current_branch(stack: List[Optional[Branch]]) -> Optional[Branch]:
    """内容行归属的分支：栈里最深的那个。"""
    for b in reversed(stack):
        if b is not None:
            return b
    return None


def _branch_for_depth(stack: List[Optional[Branch]], depth: int) -> Optional[Branch]:
    """按缩进深度取分支；该深度没有分支时退回最深的那条。

    折行的续行缩进比所属分支深（多出排版用的空格），
    所以深度对不上时不能直接放弃 —— 退回最深分支才对。
    """
    if 1 <= depth <= len(stack) and stack[depth - 1] is not None:
        return stack[depth - 1]
    return _current_branch(stack)


def load_guide(path: str) -> DecisionTree:
    with open(path, "r", encoding="utf-8") as fh:
        return parse(fh.read())
