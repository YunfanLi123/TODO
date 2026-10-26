"""BP（选人禁用）阶段决策树攻略的**原文位置**与清单。

原文落在 ``app/seed/sources/hok_s45_bp/``，由
``tools/import_hok_bp_guide.py`` 从用户提供的附件字节级复制而来，
并记下 sha256。这里只负责**定位**与**自证**：

- :func:`bp_guide_path` 给出原文在仓库里的路径；
- :func:`verify_sources` 核对 sha256 —— 原文被动过就报错。

为什么不直接引用附件路径
------------------------
规则库是**可复现**的产物：任何人 clone 仓库都能重跑生成器得到
一模一样的种子 JSON。原文在附件目录里就做不到这件事
—— 附件会过期、会换路径，而生成器的输出会跟着漂移。

「谁提供的、什么时候收的」记在 manifest 里，出处不会丢。
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(ROOT, "app", "seed", "sources", "hok_s45_bp")
MANIFEST_PATH = os.path.join(SOURCE_DIR, "manifest.json")

#: BP 攻略的中文名。
BP_GUIDE_NAME = "王者荣耀 S45「月照长安」BP 阶段决策树"


def load_manifest() -> Dict:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bp_guide_path() -> str:
    """原文在仓库里的绝对路径。"""
    manifest = load_manifest()
    return os.path.join(SOURCE_DIR, manifest["files"][0]["file"])


def bp_guide_text() -> str:
    with open(bp_guide_path(), "r", encoding="utf-8") as fh:
        return fh.read()


def verify_sources() -> List[str]:
    """核对原文的 sha256。返回问题列表，空列表表示全部一致。"""
    problems: List[str] = []
    manifest = load_manifest()
    for entry in manifest["files"]:
        path = os.path.join(SOURCE_DIR, entry["file"])
        if not os.path.exists(path):
            problems.append(f"原文缺失：{entry['file']}")
            continue
        actual = sha256_of(path)
        if actual != entry["sha256"]:
            problems.append(
                f"原文被改动：{entry['file']}（清单 {entry['sha256'][:12]}… / "
                f"实际 {actual[:12]}…）"
            )
    return problems


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    import sys

    issues = verify_sources()
    if issues:
        print("✗ 原文自检失败：")
        for p in issues:
            print("  -", p)
        raise SystemExit(1)
    manifest = load_manifest()
    print(f"✓ 原文 sha256 一致（{manifest['files'][0]['file']}）")
