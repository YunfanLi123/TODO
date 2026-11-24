"""操作模式库（v2）的读取。

模式库是**种子文件**驱动的：``app/seed/<game>_patterns.json``。
它由 :mod:`app.rules.dsl_migrate` 从 v1 规则库迁来，带上：

* ``source_sha256`` —— 迁移时 v1 规则库的 sha256。
  读的时候**核对一遍**：对不上说明 v1 库改过而模式库没重迁，
  这时宁可退回 v1 引擎，也不要拿一份和规则库对不上的模式去判玩家。

为什么不用数据库
----------------
v1 的规则库是存在数据库里的（可以由用户在界面上导入自己的攻略）。
v2 的模式库目前是**从 v1 迁移出来的派生产物** —— 它不是另一份数据，
而是同一份数据换一种表达。把派生产物也塞进数据库，会出现两份真相：
用户改了规则库，模式库不会跟着变，两边结论不一致且没人说得清谁对。

所以：v1 库是**真源**，v2 模式库是**编译产物**。
真源变了，编译产物必须重编 —— 这也是为什么这里要核对 sha256。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.dsl2 import Pattern

#: 种子目录（与 :mod:`app.rules.library_source` 一致）。
SEED_DIR = Path(__file__).resolve().parents[1] / "seed"


def seed_patterns_path(game: str) -> Path:
    return SEED_DIR / f"{game}_patterns.json"


def seed_pattern_paths(game: str) -> List[Path]:
    """该游戏的**全部**模式库文件。

    分路层与英雄层是**两份**文件，不是同一份文件的两段：

    * ``<game>_patterns.json``      —— 分路层（84 条，通则）；
    * ``<game>_hero_patterns.json``  —— 英雄层（1596 条，英雄专属）。

    两层互斥（玩家填了英雄就走英雄层），所以一份文件放不下；
    但它们是**同一个 DSL** 的产物，用同一套加载与封条规则。
    """
    return [
        SEED_DIR / f"{game}_patterns.json",
        SEED_DIR / f"{game}_hero_patterns.json",
    ]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


class PatternLibraryError(RuntimeError):
    """模式库缺失或与真源对不上。"""


def load_pattern_document(game: str) -> Optional[Dict[str, Any]]:
    """读模式库原始文档。文件不存在返回 ``None``（不是错误 —— 允许没有）。

    只读**分路层**那份。整库用 :func:`load_pattern_documents`。
    """
    path = seed_patterns_path(game)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_source(doc: Dict[str, Any]) -> None:
    """核对模式库与它的真源（v1 规则库）是不是同一份。

    对不上就**拒绝**：宁可退回 v1 引擎，
    也不要拿一份和规则库对不上的模式去判玩家。
    """
    src_name = doc.get("generated_from") or ""
    want = doc.get("source_sha256") or ""
    if not (src_name and want):
        return
    src = SEED_DIR / src_name
    if not src.exists():
        raise PatternLibraryError(
            f"模式库的真源 {src_name} 不在种子目录里，无法核对。"
        )
    got = sha256_of(src)
    if got != want:
        raise PatternLibraryError(
            f"模式库与真源对不上：{src_name} 已改动"
            f"（记录 {want[:12]}…，实际 {got[:12]}…）。"
            f"请重跑 `python -m app.rules.dsl_migrate`。"
        )


def load_pattern_documents(
    game: str,
    *,
    verify_source: bool = True,
) -> List[Dict[str, Any]]:
    """读该游戏的**全部**模式库文档（分路层 + 英雄层）。"""
    out: List[Dict[str, Any]] = []
    for path in seed_pattern_paths(game):
        if not path.exists():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        if verify_source:
            _verify_source(doc)
        out.append(doc)
    return out


def load_patterns(
    game: str,
    version: Optional[str] = None,
    *,
    verify_source: bool = True,
) -> List[Pattern]:
    """读该游戏的操作模式库（分路层 + 英雄层）。

    ``verify_source=True`` 时核对 ``source_sha256``；对不上就**拒绝**，
    并把原因写在异常里 —— 调用方据此退回 v1 引擎。
    """
    out: List[Pattern] = []
    for doc in load_pattern_documents(game, verify_source=verify_source):
        for raw in doc.get("patterns") or []:
            pat = Pattern.from_dict(raw)
            if version and pat.version != version:
                continue
            out.append(pat)
    return out


def patterns_digest(game: str) -> str:
    """模式库的内容指纹（用于报告里说明「用的是哪一版模式库」）。"""
    blob = json.dumps(
        [
            doc.get("patterns") or []
            for doc in load_pattern_documents(game, verify_source=False)
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    if not blob.strip("[]"):
        return ""
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
