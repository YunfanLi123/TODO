# -*- coding: utf-8 -*-
"""合并《王者荣耀》S45 中路英雄决策树分块，统一树形缩进并做硬性验收。

用法：
    python normalize_tree.py <输出文件> <分块文件1> <分块文件2> ...

约定：
    - 分支行形如 "├─ 前提：..." / "└─ 前提：..."，子行以 "│  动作：" 或 "   动作：" 起头
    - 嵌套子分支的子行前缀 = 父分支缩进 + 3 空格 + "│  "（├─）或 + 3 空格 + "   "（└─）
    - 输出统一为 UTF-8 无 BOM + LF
"""
import os
import re
import sys

BRANCH_RE = re.compile(r"^(\s*)([├└])─ (.*)$")
CHILD_RE = re.compile(r"^((?:   │  )|(?:│[ ]*)|(?:[ ]+))(动作|后果)：")
NODE_RE = re.compile(r"^(\d{1,2}:\d{2} 节点|12:00 之后)$")
LOC_RE = re.compile(r"^  定位：")
STATE_RE = re.compile(r"状态【[^】\n]+】\s*$")

EXPECTED_NODES = ["0:00 节点", "2:00 节点", "4:00 节点", "6:00 节点", "10:00 节点", "12:00 之后"]
EXPECTED_HEROES = [
    "海月", "王维", "大乔", "甄姬", "小乔", "沈梦溪", "女娲", "嫦娥", "不知火舞", "西施",
    "貂蝉", "弈星", "上官婉儿", "杨玉环", "诸葛亮", "武则天", "扁鹊", "妲己", "安琪拉",
    "王昭君", "米莱狄", "嬴政", "干将莫邪", "周瑜", "姜子牙", "金蝉", "张良", "海诺", "司空震",
]


def merge(parts):
    chunks = []
    for p in parts:
        with open(p, "r", encoding="utf-8") as fh:
            text = fh.read()
        chunks.append(text.rstrip("\n"))
    return "\n".join(chunks) + "\n"


def clean(line):
    """去掉半角竖线（markdown 表格符号），修正状态标签右括号。返回 (行, 是否改动)。"""
    changed = False
    if "|" in line:
        line = line.replace("|", "")
        changed = True
    if re.search(r"状态【[^】\n]*\]\s*$", line):
        line = re.sub(r"(状态【[^】\n]*\])\s*$", lambda m: m.group(1)[:-1] + "】", line)
        changed = True
    return line, changed


def normalize(lines):
    """修正分支子行与嵌套分支的前缀，返回新行列表与修正计数。"""
    out = []
    stack = []  # [(indent, kind)]，kind 为 '├' 或 '└'
    fixed = 0
    for raw in lines:
        line = raw
        m = BRANCH_RE.match(line)
        if m:
            indent, kind, rest = len(m.group(1)), m.group(2), m.group(3)
            while stack and stack[-1][0] >= indent:
                stack.pop()
            parent = stack[-1] if stack else None
            want_indent = (parent[0] + 3) if parent else 0
            if indent != want_indent:
                fixed += 1
            stack.append((want_indent, kind))
            line = " " * want_indent + kind + "─ " + rest
            line, ch = clean(line)
            fixed += 1 if ch else 0
            out.append(line)
            continue

        cm = CHILD_RE.match(line)
        if cm and stack:
            indent, kind = stack[-1]
            want = " " * indent + ("│  " if kind == "├" else "   ")
            if line[: len(want)] != want:
                fixed += 1
            line = want + line[len(cm.group(1)):]
            line, ch = clean(line)
            fixed += 1 if ch else 0
            out.append(line)
            continue

        line, ch = clean(line)
        fixed += 1 if ch else 0
        out.append(line)
    return out, fixed


def hero_blocks(lines):
    """返回 [(英雄名, 定位行, 起始行, 结束行)]。"""
    blocks = []
    for i, line in enumerate(lines):
        if LOC_RE.match(line) and i > 0:
            start = i - 1
            end = len(lines)
            for j in range(i + 1, len(lines)):
                if LOC_RE.match(lines[j]):
                    end = j - 1
                    break
            blocks.append((lines[start].strip(), line, start, end))
    return blocks


def child_kind(line):
    """子行类型：'动作' / '后果' / None。"""
    m = CHILD_RE.match(line)
    return m.group(2) if m else None


def direct_children(lines, idx, indent):
    """返回某个分支头之后的直属子行（不含嵌套分支自身的行）。"""
    children = []
    for k in range(idx + 1, len(lines)):
        line = lines[k]
        if NODE_RE.match(line.strip()) or LOC_RE.match(line):
            break
        bm = BRANCH_RE.match(line)
        if bm and len(bm.group(1)) <= indent:
            break
        children.append(line)
    return children


def validate(lines):
    problems = []
    blocks = hero_blocks(lines)
    heroes = [b[0] for b in blocks]
    nodes_total = 0
    branches_total = 0

    for name, loc, start, end in blocks:
        if "梯队：" not in loc or "关键数据" not in loc:
            problems.append("%s 定位行缺梯队标签或关键数据" % name)
        seg = lines[start:end + 1]
        nodes = [l.strip() for l in seg if NODE_RE.match(l.strip())]
        if nodes != EXPECTED_NODES:
            problems.append("%s 节点序列异常: %s" % (name, nodes))
        for ni, line in enumerate(seg):
            if not NODE_RE.match(line.strip()):
                continue
            nodes_total += 1
            node_name = line.strip()
            # 该节点的直属分支
            node_children = []
            for k in range(seg.index(line) + 1, len(seg)):
                if NODE_RE.match(seg[k].strip()):
                    break
                bm = BRANCH_RE.match(seg[k])
                if bm and len(bm.group(1)) == 0:
                    node_children.append((k, seg[k]))
            if len(node_children) < 2:
                problems.append("%s / %s 直属分支数 %d < 2" % (name, node_name, len(node_children)))
            branches_total += len(node_children)
            for k, head in node_children:
                if "前提：" not in head:
                    problems.append("%s / %s 分支头缺前提: %s" % (name, node_name, head))
                children = direct_children(seg, k, 0)
                acts = [c for c in children if child_kind(c) == "动作"]
                cons = [c for c in children if child_kind(c) == "后果"]
                nested = [c for c in children if BRANCH_RE.match(c)]
                if (not acts or not cons) and not nested:
                    problems.append("%s / %s 分支缺动作或后果: %s" % (name, node_name, head))
                for c in cons:
                    if not STATE_RE.search(c):
                        problems.append("%s / %s 后果缺状态标签: %s" % (name, node_name, c))

        # 嵌套子分支同样要有动作与后果
        for k, line in enumerate(seg):
            bm = BRANCH_RE.match(line)
            if not bm or len(bm.group(1)) == 0:
                continue
            children = direct_children(seg, k, len(bm.group(1)))
            acts = [c for c in children if child_kind(c) == "动作"]
            cons = [c for c in children if child_kind(c) == "后果"]
            if not acts or not cons:
                problems.append("%s 嵌套分支缺动作或后果: %s" % (name, line))

    if heroes != EXPECTED_HEROES:
        missing = [h for h in EXPECTED_HEROES if h not in heroes]
        extra = [h for h in heroes if h not in EXPECTED_HEROES]
        problems.append("英雄名单不符 缺少:%s 多出:%s" % (missing, extra))

    for idx, line in enumerate(lines, 1):
        if line.startswith("*") or line.startswith("#") or "|" in line or "```" in line:
            problems.append("第 %d 行含 markdown 符号: %s" % (idx, line[:40]))
        if line.rstrip().endswith("]"):
            problems.append("第 %d 行以半角右括号结尾" % idx)
    return problems, heroes, nodes_total, branches_total


def main():
    out_path = sys.argv[1]
    parts = sys.argv[2:]
    if not parts:
        print("用法：python normalize_tree.py <输出文件> <分块文件1> [分块文件2] ...（至少需要一个输入文件）")
        return
    text = merge(parts)
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    lines, fixed = normalize(lines)
    problems, heroes, nodes_total, branches_total = validate(lines)
    payload = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(payload)
    size = os.path.getsize(out_path)
    print("输出: %s" % out_path)
    print("字节数: %d" % size)
    print("英雄数: %d" % len(heroes))
    print("节点数: %d" % nodes_total)
    print("分支数: %d" % branches_total)
    print("格式修正处数: %d" % fixed)
    if problems:
        print("问题 %d 条:" % len(problems))
        for p in problems[:60]:
            print("  - %s" % p)
    else:
        print("校验通过")


if __name__ == "__main__":
    main()
