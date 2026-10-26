# -*- coding: utf-8 -*-
"""统一 01-clash-lane-heroes.txt 的树形缩进前缀，规则与同目录 normalize_tree.py 一致。
只做前缀归一，不改文字内容。
"""
import io
import re
import sys

BRANCH_RE = re.compile(r"^(\s*)([\u251c\u2514])\u2500 (.*)$")
CHILD_RE = re.compile(r"^((?:   \u2502  )|(?:\u2502[ ]*)|(?:[ ]+))(\u52a8\u4f5c|\u540e\u679c)\uff1a")
PIPE_RE = re.compile(r"\|")


def normalize(lines):
    out = []
    stack = []
    fixed = 0
    for raw in lines:
        line = raw.rstrip("\n")
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
            new_line = " " * want_indent + kind + "\u2500 " + rest
            if new_line != line:
                fixed += 1
            out.append(new_line)
            continue

        cm = CHILD_RE.match(line)
        if cm and stack:
            indent, kind = stack[-1]
            want = " " * indent + ("\u2502  " if kind == "\u251c" else "   ")
            new_line = want + line[len(cm.group(1)):]
            if new_line != line:
                fixed += 1
            out.append(new_line)
            continue

        out.append(line)
    return out, fixed


def main():
    path = sys.argv[1]
    with io.open(path, "r", encoding="utf-8", newline="") as fh:
        text = fh.read()
    lines = text.split("\n")
    trailing_nl = lines and lines[-1] == ""
    if trailing_nl:
        lines = lines[:-1]
    out, fixed = normalize(lines)
    result = "\n".join(out) + "\n"
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(result)
    print("LINES_IN=%d LINES_OUT=%d FIXED=%d" % (len(lines), len(out), fixed))


if __name__ == "__main__":
    main()
