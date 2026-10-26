"""S45 英雄级决策树攻略的解析器。

与 :mod:`tools.hok_decision_tree` 的关系
--------------------------------------
分路决策树（``hok_s45_lane_decision_tree.txt``）按**分路**分节，
英雄级决策树（本模块）按**英雄**分节，节点的树形语法完全一样。
所以这里不重写树形解析，只做两件事：

1. 把「英雄块」从文件里切出来（两种分隔线写法都认）；
2. 把切出来的英雄块交给 :func:`tools.hok_decision_tree.parse` 的同一套
   节点/分支解析逻辑。

复用同一套树形解析的意义在于**引用逐字**这条约定不变：
``Branch.quote()`` 拼出来的文字在原文里是连续的，
「抄错引用」这件事由解析器保证，不靠人抄。

文件格式
--------
英雄块有两种写法，两份文件都出现过::

    =====================          =====================
    夏侯惇                       赵云
      定位：坦克型边路…             =====================
                                    定位与关键数据：S45 打野梯队 T0
                                    机制要点：……

区别只在于「英雄名后面是不是还有一条分隔线」。
英雄块的判定不靠名字，而靠**块里有没有时间节点** ——
文件末尾的「附注」小节没有节点，就不算英雄块。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from tools.hok_decision_tree import (
    BRANCH_RE,
    CONTENT_RE,
    NODE_RE,
    Branch,
    Node,
    normalize_for_compare,
)

#: 分隔线：整行等号（英雄级文件用 ``=``，分路文件用 ``═``）。
SEP_RE = re.compile(r"^[=═]{10,}$")

#: 英雄块的元信息行（缩进两格）。
META_RE = re.compile(r"^\s{2,}(?P<key>[^：:]{1,12})[：:]\s*(?P<value>.*)$")


@dataclass
class HeroNode:
    """一个英雄的一个时间节点。"""

    hero: str
    lane: str
    time_label: str
    header: str
    branches: List[Branch] = field(default_factory=list)

    def find(self, key: str) -> Optional[Branch]:
        for b in self.branches:
            if b.key == key:
                return b
        return None


@dataclass
class HeroBlock:
    """一个英雄的完整决策树。"""

    hero: str
    lane: str
    meta: Dict[str, str] = field(default_factory=dict)
    nodes: List[HeroNode] = field(default_factory=list)

    @property
    def meta_text(self) -> str:
        return "；".join(f"{k}：{v}" for k, v in self.meta.items())


@dataclass
class HeroGuide:
    """一份英雄级决策树文件。"""

    lane: str
    path: str
    heroes: List[HeroBlock] = field(default_factory=list)
    text: str = ""

    def hero_names(self) -> List[str]:
        return [h.hero for h in self.heroes]


def _split_blocks(text: str) -> List[tuple]:
    """把文件切成「(块名, 块正文)」。

    分隔线一律**跳过**，不参与分块判断 —— 打野那份的写法是
    「分隔线 / 英雄名 / 分隔线」，对抗路那份是「分隔线 / 英雄名」，
    如果按「分隔线后面那行就是块名」来切，打野那份会把英雄的
    「定位与关键数据」那一行当成新块名。

    所以块名的判据是：**顶格、不带树形符号、不是时间节点**的一行。
    顶格且不是树形符号的行在正文里只有两种：文件开头的大标题，
    和英雄名。开头的大标题在第一条分隔线之前，所以还要加一条
    「第一条分隔线之后才开始认块名」。
    """
    lines = text.split("\n")
    blocks: List[tuple] = []
    name: Optional[str] = None
    buf: List[str] = []
    seen_sep = False

    for raw in lines:
        stripped = raw.strip()
        if SEP_RE.match(stripped):
            seen_sep = True
            continue
        if not stripped:
            if name is not None:
                buf.append(raw)
            continue
        is_tree = stripped[0] in "├└│"
        is_node = bool(NODE_RE.match(stripped))
        is_top = (raw[:1] not in (" ", "\t")) and not is_tree and not is_node
        if seen_sep and is_top:
            if name is not None:
                blocks.append((name, "\n".join(buf)))
            name = stripped
            buf = []
            continue
        if name is not None:
            buf.append(raw)
    if name is not None:
        blocks.append((name, "\n".join(buf)))
    return blocks


def _parse_meta(lines: List[str]) -> Dict[str, str]:
    """取节点之前的元信息行（定位 / 梯队 / 机制要点 / 官方数据 / S45 改动）。"""
    meta: Dict[str, str] = {}
    for raw in lines:
        if not raw.strip():
            continue
        if NODE_RE.match(raw.strip()):
            break
        m = META_RE.match(raw)
        if m:
            key = m.group("key").strip()
            value = m.group("value").strip()
            if key and value and key not in meta:
                meta[key] = value
    return meta


def _parse_nodes(hero: str, lane: str, lines: List[str]) -> List[HeroNode]:
    """解析节点与分支（树形语法与分路决策树一致）。"""
    nodes: List[HeroNode] = []
    node: Optional[HeroNode] = None
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

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if SEP_RE.match(stripped):
            continue

        m = NODE_RE.match(stripped)
        if m:
            node = HeroNode(
                hero=hero,
                lane=lane,
                time_label=re.sub(r"\s*-\s*", "-", m.group("time")),
                header=stripped,
            )
            nodes.append(node)
            stack = []
            continue

        bm = BRANCH_RE.match(line)
        if bm and node is not None:
            depth = len(bm.group("ind")) // 3 + 1
            text_after = bm.group("text").strip()
            premise = ""
            if text_after.startswith("前提："):
                premise = text_after[len("前提："):].strip()
            parent = None
            if depth >= 2:
                parent = stack[depth - 2] if depth - 2 < len(stack) else None
            branch = Branch(
                depth=depth,
                premise=premise,
                node=node,  # type: ignore[arg-type]
                parent=parent,
            )
            node.branches.append(branch)
            branch.index = len(node.branches)
            while len(stack) < depth:
                stack.append(None)
            stack[depth - 1] = branch
            for k in range(depth, len(stack)):
                stack[k] = None
            continue

        cm = CONTENT_RE.match(line)
        if cm and node is not None:
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


def parse_hero_guide(text: str, lane: str, path: str = "") -> HeroGuide:
    """解析一份英雄级决策树文件。"""
    guide = HeroGuide(lane=lane, path=path, text=text)
    for name, body in _split_blocks(text):
        lines = body.split("\n")
        nodes = _parse_nodes(name, lane, lines)
        if not nodes:
            # 没有时间节点 —— 文件末尾的「附注」小节之类，不是英雄块。
            continue
        meta = _parse_meta(lines)
        guide.heroes.append(HeroBlock(hero=name, lane=lane, meta=meta, nodes=nodes))
    return guide


def load_hero_guide(path: str, lane: str) -> HeroGuide:
    with open(path, "r", encoding="utf-8") as fh:
        return parse_hero_guide(fh.read(), lane, path)


__all__ = [
    "HeroBlock",
    "HeroGuide",
    "HeroNode",
    "load_hero_guide",
    "normalize_for_compare",
    "parse_hero_guide",
]
