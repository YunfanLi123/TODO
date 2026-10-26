"""英雄级决策树攻略的**原文位置**与清单。

原文落在 ``app/seed/sources/hok_s45_hero_guides/``，由
``tools/import_hok_hero_guides.py`` 从用户提供的附件字节级复制而来，
并记下 sha256。这里只负责**定位**与**自证**：

- :func:`hero_guide_paths` 给出 5 份原文在仓库里的路径（顺序固定）；
- :func:`verify_sources` 核对 sha256 —— 原文被动过就报错。

为什么不直接引用附件路径
------------------------
规则库是**可复现**的产物：任何人 clone 仓库都能重跑生成器得到
一模一样的 ``app/seed/hok_hero_modes.json``。原文在附件目录里就做不到这件事
—— 附件会过期、会换路径，而生成器的输出会跟着漂移。

「谁提供的、什么时候收的」记在 manifest 里，出处不会丢。
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(ROOT, "app", "seed", "sources", "hok_s45_hero_guides")
MANIFEST_PATH = os.path.join(SOURCE_DIR, "manifest.json")

#: 分路编码 -> 中文名。
LANE_LABELS: Dict[str, str] = {
    "jungle": "打野",
    "mid": "中路",
    "side": "对抗路",
    "carry": "发育路",
    "roam": "游走",
}


def load_manifest() -> Dict:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hero_guide_paths() -> List[Tuple[str, str, str]]:
    """返回 ``[(分路编码, 中文名, 原文绝对路径), ...]``，顺序固定。"""
    manifest = load_manifest()
    out = []
    for entry in manifest["files"]:
        out.append(
            (
                entry["lane"],
                entry["lane_label"],
                os.path.join(SOURCE_DIR, entry["file"]),
            )
        )
    order = ["jungle", "mid", "side", "carry", "roam"]
    out.sort(key=lambda item: order.index(item[0]))
    return out


def verify_sources() -> List[str]:
    """核对每份原文的 sha256。返回问题列表，空列表表示全部一致。"""
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


def all_hero_names() -> List[str]:
    """攻略里出现的全部英雄名（去重，按分路顺序排）。

    分路层那 85 条规则要加一条 ``my_hero in (这些英雄)`` 的护栏：
    玩家选了自己的英雄，就只走英雄层的规则 ——
    否则同一件事会被分路通则和英雄专属各指控一遍。
    """
    from tools.hok_hero_tree import load_hero_guide

    seen: Dict[str, None] = {}
    for lane, _label, path in hero_guide_paths():
        guide = load_hero_guide(path, lane)
        for hb in guide.heroes:
            seen.setdefault(hb.hero, None)
    return list(seen)


def heroes_by_lane() -> Dict[str, List[str]]:
    """分路编码 -> 该分路攻略里出现的英雄名。"""
    from tools.hok_hero_tree import load_hero_guide

    out: Dict[str, List[str]] = {}
    for lane, _label, path in hero_guide_paths():
        guide = load_hero_guide(path, lane)
        names: List[str] = []
        for hb in guide.heroes:
            if hb.hero not in names:
                names.append(hb.hero)
        out[lane] = names
    return out


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    issues = verify_sources()
    if issues:
        print("✗ 原文自检失败：")
        for p in issues:
            print("  -", p)
        raise SystemExit(1)
    manifest = load_manifest()
    print(f"✓ {len(manifest['files'])} 份原文 sha256 全部一致")
    for lane, label, path in hero_guide_paths():
        print(f"  {label}  {os.path.basename(path)}")
