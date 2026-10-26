"""**独立于解析器**的原文扫描器 —— 用来给解析器对账。

为什么要有第二个扫描器
--------------------
``tests/test_hero_mode_library.py`` 里钉死了「882 节点 / 2910 分支 /
147 英雄块」。但那三个数字是 :mod:`tools.hok_hero_tree` **解析器输出**：

    解析器漏读 5 个节点  ->  解析器输出变成 877 节点  ->  测试期望值也跟着变
    ->  测试照样绿

数字和被测对象来自同一份实现，等于没测。本模块用**另一套代码**直接扫
原始文本，把同一个事实数一遍，两边必须对得上。解析器漏读就露馅。

两套实现的差别只在「怎么切块」：

- 分路层解析器把「分隔线 → 英雄名 → 正文」当一个英雄块；
- 本模块按分隔线切段，再用「**含时间节点的段才是英雄块**」这条
  更弱的规则还原 —— 打野那份是「分隔线 → 英雄名 → 分隔线 → 正文」
  （名字单独成段），照样能还原。

跑哪个文件
----------
默认扫**用户上传的原件**（``tools/import_hok_hero_guides.py`` 里的
``SOURCES``），因为这是「当初到底入库了哪份原文」的唯一凭据；
原件不在了可以扫仓库副本，两边由
``tests/test_hero_mode_library.py::test_repo_copies_match_the_original_uploaded_assets``
证明是同一份。
"""

from __future__ import annotations

import os
import re
import sys
from typing import Dict, List, Optional, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.hok_decision_tree import NODE_RE  # noqa: E402  节点标题语法（定义，非解析逻辑）

#: 分支行：去掉树形前缀后以「前提：」开头。
PREMISE_RE = re.compile(r"^前提[:：]")

#: 树形前缀字符（``│├└─``）。
TREE_CHARS = ("│", "├", "└", "─")


def strip_tree(line: str) -> str:
    """去掉行首树形符号与空白，返回正文。"""
    return re.sub(r"^[│├└─\s]*", "", line.strip())


def is_separator(line: str) -> bool:
    """整行等号就是分隔线。"""
    s = line.strip()
    return bool(s) and set(s) == {"="}


def candidate_name(section: List[str]) -> Optional[str]:
    """段里第一行「顶格、非树形、非节点标题」的非空文字；没有则 ``None``。

    顶格且带树形符号的行是分支（``├─ 前提：…``），顶格且匹配时间轴的
    是节点标题，两者都当不成块名。
    """
    for raw in section:
        cand = raw.strip()
        if not cand:
            continue
        if raw.startswith(" ") or raw.startswith(TREE_CHARS):
            continue
        if NODE_RE.match(cand):
            continue
        return cand
    return None


def split_sections(lines: List[str]) -> List[List[str]]:
    """按整行等号分隔线切成段。"""
    sections: List[List[str]] = []
    cur: List[str] = []
    for raw in lines:
        if is_separator(raw):
            sections.append(cur)
            cur = []
            continue
        cur.append(raw)
    sections.append(cur)
    return sections


def raw_scan(path: str) -> Tuple[List[str], int, int]:
    """直接扫原文，返回 ``(英雄名列表, 节点数, 分支数)``。"""
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    sections = split_sections(lines)
    names = [candidate_name(s) for s in sections]
    node_counts = [
        sum(1 for raw in s if raw.strip() and NODE_RE.match(raw.strip()))
        for s in sections
    ]

    heroes: List[str] = []
    nodes = 0
    branches = 0
    for i, body in enumerate(sections):
        if not node_counts[i]:
            continue  # 没有时间节点 —— 不是英雄块
        name = names[i]
        if name is None and i > 0 and names[i - 1] and not node_counts[i - 1]:
            # 打野那份：名字单独成段，正文在下一段
            name = names[i - 1]
        if name is None:
            continue
        heroes.append(name)
        nodes += node_counts[i]
        branches += sum(1 for raw in body if PREMISE_RE.match(strip_tree(raw)))
    return heroes, nodes, branches


def parser_scan(path: str, lane: str) -> Tuple[List[str], int, int]:
    """同一份文件交给解析器，返回同样的三元组（用于对账）。"""
    from tools.hok_hero_tree import load_hero_guide

    guide = load_hero_guide(path, lane)
    return (
        [hb.hero for hb in guide.heroes],
        sum(len(hb.nodes) for hb in guide.heroes),
        sum(len(n.branches) for hb in guide.heroes for n in hb.nodes),
    )


def scan_originals() -> Dict[str, Dict[str, object]]:
    """扫**用户上传的原件**并与解析器对账。

    返回 ``{分路编码: {...}}``；原件缺失的分路会被跳过（附件会被清理）。
    """
    from tools.import_hok_hero_guides import SOURCES

    out: Dict[str, Dict[str, object]] = {}
    for src, lane, label in SOURCES:
        if not os.path.exists(src):
            continue
        r_heroes, r_nodes, r_branches = raw_scan(src)
        p_heroes, p_nodes, p_branches = parser_scan(src, lane)
        out[lane] = {
            "label": label,
            "path": src,
            "raw": (r_heroes, r_nodes, r_branches),
            "parser": (p_heroes, p_nodes, p_branches),
            "names_match": set(r_heroes) == set(p_heroes),
        }
    return out


def main() -> int:
    from tools.import_hok_hero_guides import SOURCES

    print("=" * 66)
    print("原文扫描 vs 解析器 对账")
    print("=" * 66)

    bad = 0
    tot = [0, 0, 0]
    for src, lane, label in SOURCES:
        if not os.path.exists(src):
            print(f"- {label}：原件已不在附件目录，跳过")
            continue
        r_heroes, r_nodes, r_branches = raw_scan(src)
        p_heroes, p_nodes, p_branches = parser_scan(src, lane)
        ok = (
            len(r_heroes) == len(p_heroes)
            and r_nodes == p_nodes
            and r_branches == p_branches
            and set(r_heroes) == set(p_heroes)
        )
        if not ok:
            bad += 1
        print(f"{'✓' if ok else '✗'} {label:4s} 原文扫描 英雄{len(r_heroes):3d}/节点{r_nodes:4d}/"
              f"分支{r_branches:4d}   解析器 英雄{len(p_heroes):3d}/节点{p_nodes:4d}/分支{p_branches:4d}")
        if set(r_heroes) != set(p_heroes):
            only_raw = sorted(set(r_heroes) - set(p_heroes))
            only_parser = sorted(set(p_heroes) - set(r_heroes))
            if only_raw:
                print(f"    原文有但解析器没有：{only_raw}")
            if only_parser:
                print(f"    解析器多出来：{only_parser}")
        tot[0] += r_nodes
        tot[1] += r_branches
        tot[2] += len(r_heroes)

    print("-" * 66)
    print(f"合计（原文扫描）：英雄块 {tot[2]} / 节点 {tot[0]} / 分支 {tot[1]}")
    if bad:
        print(f"✗ 对账失败 {bad} 处")
        return 1
    print("✓ 原文扫描与解析器完全对账一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
