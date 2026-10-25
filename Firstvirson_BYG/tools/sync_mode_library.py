"""把内置的操作模式库同步到可编辑的那份（``mode-library/``）。

为什么要它
----------
``app/seed/`` 是**内置兜底**（打包进 exe，用户动不了），
``mode-library/`` 是**可编辑副本**（一个独立的 git 仓库，用户改它、推自己的 GitHub）。

改了内置之后，两边就不一致了 —— 而应用默认读的是可编辑那份。
这个脚本把内置覆盖过去（**只覆盖文件，不动 git 历史**）。

用法::

    python tools/sync_mode_library.py          # 看看有没有差异
    python tools/sync_mode_library.py --apply  # 真同步
"""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILTIN = ROOT / "app" / "seed"
EDITABLE = ROOT / "mode-library"

#: 只同步「操作模式」那几个文件。英雄与标签数据是素材，不是知识库。
FILES = (
    "hok_modes.json",
    "hok_hero_modes.json",
    "hok_bp_modes.json",
    "hok_retired.json",
    "hok_bp_skipped.json",
    "hok_hero_skipped.json",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真同步（默认只报差异）")
    args = ap.parse_args()

    if not EDITABLE.is_dir():
        print(f"没有 {EDITABLE} —— 先建这个目录（它是一个独立的 git 仓库）")
        return 1

    changed = []
    for name in FILES:
        src = BUILTIN / name
        dst = EDITABLE / name
        if not src.is_file():
            continue
        if dst.is_file() and filecmp.cmp(src, dst, shallow=False):
            continue
        changed.append(name)

    if not changed:
        print("两边一致，不用同步。")
        return 0

    print("这些文件和内置不一样：")
    for name in changed:
        print(f"  - {name}")
    if not args.apply:
        print("\n加 --apply 才真同步。")
        return 0

    for name in changed:
        shutil.copyfile(BUILTIN / name, EDITABLE / name)
        print(f"已同步 {name}")
    print("\n（git 历史没动，自己 commit。）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
