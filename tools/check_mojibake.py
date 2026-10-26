"""全仓扫描被损坏的 UTF-8 字符（U+FFFD 替换符）。

编码损坏留下的痕迹混在代码、模板、文档里，肉眼很难发现 ——
页面上会显示成「这??录屏」这种。跑一遍就知道还有没有漏网的。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".venv", "dist", "build", ".tmp", ".git", "__pycache__",
             "node_modules", "recordings_probe", "pipcache", "pytest-of-lyf"}
SKIP_EXT = {".exe", ".dll", ".pyd", ".png", ".jpg", ".jpeg", ".gif", ".mp4",
            ".db", ".npz", ".pyc", ".ico", ".woff", ".woff2", ".ttf"}

hits = []
for p in ROOT.rglob("*"):
    if not p.is_file():
        continue
    if any(part in SKIP_DIRS for part in p.parts):
        continue
    if p.suffix.lower() in SKIP_EXT:
        continue
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        continue
    for i, line in enumerate(text.splitlines(), 1):
        if "\ufffd" in line:
            hits.append((p.relative_to(ROOT), i, line.strip()[:100]))

if not hits:
    print("没有损坏字符 ✓")
else:
    print(f"发现 {len(hits)} 处：")
    for rel, i, line in hits:
        print(f"  {rel}:{i}  {line}")
raise SystemExit(1 if hits else 0)
