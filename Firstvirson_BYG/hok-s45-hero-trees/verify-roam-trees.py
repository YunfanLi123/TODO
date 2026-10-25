# -*- coding: utf-8 -*-
"""合并《王者荣耀》S45 游走位英雄决策树分块，并对 05-roam-lane-heroes.txt 做硬性验收。

用法：
    python verify-roam-trees.py merge            # 合并 _r1.._r8 并校验
    python verify-roam-trees.py check            # 只校验已存在的最终文件

约定：
    - 分支行形如 "├─ 前提：..." / "└─ 前提：..."，子行以 "│  动作：" 或 "   动作：" 起头
    - 嵌套子分支缩进 = 父分支缩进 + 3
    - 输出统一为 UTF-8 无 BOM + LF
"""
import os
import re
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "05-roam-lane-heroes.txt")
PARTS = [os.path.join(BASE, "_r%d.txt" % i) for i in range(1, 9)]

EXPECTED_NODES = ["0:00 节点", "2:00 节点", "4:00 节点", "6:00 节点", "10:00 节点", "12:00 之后"]
EXPECTED_HEROES = [
    "盾山", "鲁班大师", "元辅", "少司缘", "苏烈", "牛魔", "太乙真人", "张飞", "大乔", "墨子",
    "张良", "东皇太一", "蔡文姬", "瑶", "庄周", "廉颇", "钟馗", "刘禅", "朵莉亚", "孙膑",
    "金蝉", "鬼谷子", "空空儿", "赵怀真", "刘邦", "杨玉环", "桑启", "大禹", "明世隐", "甄姬", "弈星",
]

BRANCH_RE = re.compile(r"^(\s*)([├└])─ (.*)$")
CHILD_RE = re.compile(r"^((?:   │  )|(?:│\s*)|(?:\s+))(动作|后果)：")
NODE_RE = re.compile(r"^(\d{1,2}:\d{2} 节点|12:00 之后)$")
STATE_RE = re.compile(r"状态【[^】\n]+】\s*$")
TAG_RE = re.compile(r"【([^】\n]+)】")

BANNED = [
    "最强", "必选", "版本答案", "最重要", "划算", "很赚", "碾压", "无脑", "闭眼选", "完胜",
    "T0梯队", "版本最强", "神级", "无解", "天花板", "吊打", "封神", "版本之选", "闭眼",
]
ECON = ["经济差", "等级差", "评分", "领先", "落后", "压倒性", "局势"]


def merge():
    """合并仍然存在的分块；分块已清理时保留现有最终文件，仅做规范化与校验。"""
    parts = [p for p in PARTS if os.path.exists(p)]
    if not parts:
        return None
    chunks = []
    for p in parts:
        with open(p, "r", encoding="utf-8") as fh:
            chunks.append(fh.read().rstrip("\n"))
    text = "\n".join(chunks) + "\n"
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return text


def child_kind(line):
    m = CHILD_RE.match(line)
    return m.group(2) if m else None


def is_terminator(line):
    """节点头、空行、分隔线都表示分支作用域结束。"""
    if NODE_RE.match(line.strip()):
        return True
    s = line.strip()
    return s == "" or s.startswith("=")


def scope_end(lines, idx, indent):
    """分支作用域结束位置：下一个缩进 <= indent 的分支头，或节点头/空行/分隔线。"""
    for k in range(idx + 1, len(lines)):
        line = lines[k]
        if is_terminator(line):
            return k
        bm = BRANCH_RE.match(line)
        if bm and len(bm.group(1)) <= indent:
            return k
    return len(lines)


def own_children(lines, idx, indent):
    """直属子行：不含嵌套子分支及其子树。"""
    out = []
    k = idx + 1
    while k < len(lines):
        line = lines[k]
        if is_terminator(line):
            break
        bm = BRANCH_RE.match(line)
        if bm and len(bm.group(1)) <= indent:
            break
        if bm:  # 嵌套子分支，跳过整棵子树
            k = scope_end(lines, k, len(bm.group(1)))
            continue
        out.append((k, line))
        k += 1
    return out


def nested_branches(lines, idx, indent):
    out = []
    k = idx + 1
    while k < len(lines):
        line = lines[k]
        if is_terminator(line):
            break
        bm = BRANCH_RE.match(line)
        if bm and len(bm.group(1)) <= indent:
            break
        if bm:
            out.append((k, bm))
            k = scope_end(lines, k, len(bm.group(1)))
            continue
        k += 1
    return out


def hero_blocks(lines):
    """返回 [(英雄名, 定位行行号, 块起止行号)]，英雄名必须是独占一行。"""
    blocks = []
    hero_set = set(EXPECTED_HEROES)
    for i, line in enumerate(lines):
        if line in hero_set:
            start = i
            end = len(lines)
            for j in range(i + 1, len(lines)):
                if lines[j] in hero_set:
                    end = j
                    break
            blocks.append((line, i, end))
    return blocks


def validate(text):
    problems = []
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()

    blocks = hero_blocks(lines)
    names = [b[0] for b in blocks]
    missing = [h for h in EXPECTED_HEROES if h not in names]
    extra = [h for h in names if h not in EXPECTED_HEROES]
    if missing:
        problems.append("缺少英雄: %s" % ",".join(missing))
    if extra:
        problems.append("多出英雄: %s" % ",".join(extra))
    dup = [h for h in set(names) if names.count(h) > 1]
    if dup:
        problems.append("英雄重复出现: %s" % ",".join(dup))

    nodes_total = 0
    branches_total = 0
    for name, start, end in blocks:
        seg = lines[start:end]
        loc_idx = None
        for k in range(1, len(seg)):
            if seg[k].startswith("  ") and "定位" in seg[k]:
                loc_idx = k
                break
        loc = seg[loc_idx] if loc_idx is not None else ""
        if loc_idx is None:
            problems.append("%s 缺定位与关键数据行" % name)
        else:
            if "梯队" not in loc:
                problems.append("%s 定位行缺梯队标签" % name)
            if "关键数据" not in loc:
                problems.append("%s 定位行缺关键数据" % name)
        mech = [l for l in seg if l.startswith("  机制要点")]
        if not mech:
            problems.append("%s 缺机制要点行" % name)

        node_idx = [k for k, l in enumerate(seg) if NODE_RE.match(l.strip())]
        seq = [seg[k].strip() for k in node_idx]
        if seq != EXPECTED_NODES:
            problems.append("%s 节点序列异常: %s" % (name, seq))
        nodes_total += len(node_idx)

        for ni, k in enumerate(node_idx):
            node_name = seg[k].strip()
            stop = node_idx[ni + 1] if ni + 1 < len(node_idx) else len(seg)
            body = seg[k + 1:stop]
            tops = [(bi, BRANCH_RE.match(l)) for bi, l in enumerate(body) if BRANCH_RE.match(l) and len(BRANCH_RE.match(l).group(1)) == 0]
            if len(tops) < 2:
                problems.append("%s / %s 直属分支数 %d < 2" % (name, node_name, len(tops)))
            branches_total += len(tops)
            seen_tags = {}
            for bi, bm in tops:
                check_branch(body, bi, bm, name, node_name, problems, seen_tags)

    # 状态标签首尾相接：上一节点产出的标签必须出现在下一节点文本中
    for name, start, end in blocks:
        seg = lines[start:end]
        node_idx = [k for k, l in enumerate(seg) if NODE_RE.match(l.strip())]
        for ni in range(len(node_idx) - 1):
            s, e = node_idx[ni], node_idx[ni + 1]
            tags = []
            for l in seg[s:e]:
                if child_kind(l) == "后果":
                    tags += TAG_RE.findall(l)
            nxt = "\n".join(seg[e:e + 80])
            if not any(t in nxt for t in tags):
                problems.append("%s / %s 产出的状态标签未在下一节点被引用: %s" % (name, seg[s].strip(), ",".join(tags)))

    for idx, line in enumerate(lines, 1):
        if line.startswith("*") or line.startswith("#") or "|" in line or "```" in line:
            problems.append("第 %d 行含 markdown 符号: %s" % (idx, line[:40]))
        for w in BANNED:
            if w in line:
                problems.append("第 %d 行含禁用评价词 [%s]: %s" % (idx, w, line[:40]))
        for w in ECON:
            if w in line:
                problems.append("第 %d 行含结果类判断词 [%s]: %s" % (idx, w, line[:40]))

    return problems, names, nodes_total, branches_total, lines


def check_branch(body, bi, bm, name, node_name, problems, seen_tags):
    """校验单个分支：前提 / 动作 / 后果（带状态标签）。"""
    indent = len(bm.group(1))
    kind = bm.group(2)
    head = bm.group(3)
    if not head.startswith("前提："):
        problems.append("%s / %s 分支头缺前提: %s" % (name, node_name, head[:30]))
    kids = own_children(body, bi, indent)
    acts = [(k, l) for k, l in kids if child_kind(l) == "动作"]
    cons = [(k, l) for k, l in kids if child_kind(l) == "后果"]
    nested = nested_branches(body, bi, indent)
    if nested:
        for nk, nbm in nested:
            check_branch(body, nk, nbm, name, node_name, problems, seen_tags)
        if acts or cons:
            problems.append("%s / %s 带子分支的父分支不应自带动作后果: %s" % (name, node_name, head[:30]))
    else:
        if not acts:
            problems.append("%s / %s 分支缺动作: %s" % (name, node_name, head[:30]))
        if len(cons) != 1:
            problems.append("%s / %s 分支后果数 %d: %s" % (name, node_name, len(cons), head[:30]))
    # 子行前缀
    want = " " * indent + ("│  " if kind == "├" else "   ")
    for k, l in kids:
        if not l.startswith(want):
            problems.append("%s / %s 子行缩进不符 [%s]: %s" % (name, node_name, want.strip(), l[:30]))
    # 后果状态标签
    for k, l in cons:
        if not STATE_RE.search(l):
            problems.append("%s / %s 后果缺状态标签: %s" % (name, node_name, l[:40]))
        else:
            tag = TAG_RE.findall(l)[-1]
            if tag in seen_tags:
                problems.append("%s / %s 同一节点内状态标签重复: %s" % (name, node_name, tag))
            seen_tags[tag] = True


def normalize(lines):
    """按父分支类型统一子行前缀：├─ 用 "│  "，└─ 用 "   "；嵌套分支缩进 = 父缩进 + 3。"""
    out = []
    stack = []  # [(indent, kind)]
    fixed = 0
    prev_is_hero = False
    for raw in lines:
        line = raw
        if line.startswith("=") and prev_is_hero:
            # 英雄名正下方的分隔线去掉，让定位行紧跟英雄名（与 02/03 写法一致）
            fixed += 1
            prev_is_hero = False
            continue
        prev_is_hero = line in EXPECTED_HEROES
        if "  定位与关键数据：" in line:
            new = re.sub(r"梯队 (T\d)", r"梯队标签：\1", line)
            if new != line:
                fixed += 1
            line = new
        bm = BRANCH_RE.match(line)
        if bm:
            indent, kind = len(bm.group(1)), bm.group(2)
            while stack and stack[-1][0] >= indent:
                stack.pop()
            want_indent = (stack[-1][0] + 3) if stack else 0
            if indent != want_indent:
                fixed += 1
            stack.append((want_indent, kind))
            line = " " * want_indent + kind + "─ " + bm.group(3)
            out.append(line)
            continue
        cm = CHILD_RE.match(line)
        if cm and stack:
            indent, kind = stack[-1]
            want = " " * indent + ("│  " if kind == "├" else "   ")
            if not line.startswith(want):
                fixed += 1
            line = want + line[len(cm.group(1)):]
            out.append(line)
            continue
        if NODE_RE.match(line.strip()):
            stack = []
        out.append(line)
    return out, fixed


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "merge":
        text = merge()
        if text is None:
            mode = "check"
    if mode == "check" or text is None:
        with open(OUT, "r", encoding="utf-8") as fh:
            text = fh.read()
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    lines, fixed = normalize(lines)
    text = "\n".join(lines) + "\n"
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    problems, names, nodes_total, branches_total, lines = validate(text)
    size = os.path.getsize(OUT)
    with open(OUT, "rb") as fh:
        raw = fh.read()
    report = os.path.join(BASE, "_roam_report.txt")
    with open(report, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("输出文件: %s\n" % OUT)
        fh.write("字节数: %d (BOM=%s)\n" % (size, raw[:3] == b"\xef\xbb\xbf"))
        fh.write("行数: %d\n" % len(lines))
        fh.write("英雄数: %d / %d\n" % (len(names), len(EXPECTED_HEROES)))
        fh.write("节点数: %d (期望 %d)\n" % (nodes_total, len(EXPECTED_HEROES) * 6))
        fh.write("直属分支数: %d\n" % branches_total)
        fh.write("英雄顺序: %s\n" % ",".join(names))
        if problems:
            fh.write("问题 %d 条:\n" % len(problems))
            for p in problems[:200]:
                fh.write("  - %s\n" % p)
        else:
            fh.write("校验通过\n")
    print("out=%s" % OUT)
    print("bytes=%d bom=%s lines=%d" % (size, raw[:3] == b"\xef\xbb\xbf", len(lines)))
    print("heroes=%d/%d nodes=%d/%d branches=%d" % (len(names), len(EXPECTED_HEROES), nodes_total, len(EXPECTED_HEROES) * 6, branches_total))
    print("problems=%d report=%s" % (len(problems), report))


if __name__ == "__main__":
    main()
