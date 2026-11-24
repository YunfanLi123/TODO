"""种子产物的**封条**：schema 版本 + 规范化字节串 + 内容哈希 + 溯源 + 失败记账。

借鉴自哪里
----------
`github.com/olivia-happy/moba-build-agent`（局内出装助手）的
``backend/app/portrait_index.py`` 与 ``backend/scripts/build_portrait_index.py``。

**那个仓库没有 LICENSE 文件**（GitHub API 的 ``license`` 字段为 ``null``，
``git/trees?recursive=1`` 的完整文件树里也没有 ``LICENSE``）→ 默认「保留所有权利」。
所以这里**只借鉴它的约定，不复制它的代码**。

它立的规矩，我们借了六条
----------------------
1. **可复现**：同样的输入 → 同样的产物 + 同样的哈希；
2. **canonical_payload**：一份**明确的规范化字节串**，原文嵌在产物里，
   哈希只覆盖它 —— 避免两个端各写一个序列化器、哈希对不上；
3. **双层哈希**：每条 entry 一个（单条），整份产物一个（总）；
4. **溯源**：来源、生成时间、生成方式都写在产物里；
5. **schema_version 门**：版本不认识就报错，不静默按老结构读；
6. **加载即校验**：读回来重算哈希，对不上就报错。

再加上它 ``evidence.py`` 里的两条：
7. **失败要记账**：哪条失败、为什么，写在产物里，不能悄悄少掉；
8. **quality_flags**：缺什么要**明说**，不装作完整。

我们不借它的**描述子**（16×16 亮度网格）—— 它索引的是英雄头像（单张、单尺度），
而我们匹配的是**技能图标**，靠的是彩色相关性 + 多尺度 + 多平移。
换成 16×16 亮度会把候选之间的余量吃掉，**认错英雄比认不出更糟**。
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

#: 当前封条格式的版本。
#:
#: 改了产物结构就 +1 —— 老版本读到新文件会**报错**，而不是按老结构
#: 默默读出一堆错的来。
SCHEMA_VERSION = 1

#: 这些键是**元信息**，不进哈希覆盖范围。
#:
#: 哈希只覆盖「数据本身」：说明、条目、失败记账。
#: 元信息（什么时候生成的、谁生成的）允许随每次构建变化，
#: 而数据部分必须**逐字节可复现**。
META_KEYS = ("schema_version", "kind", "built_at", "source",
             "quality_flags", "canonical_payload", "content_hash")


class SeedIndexError(RuntimeError):
    """种子产物不合法（版本不认识 / 哈希对不上 / 结构坏了）。"""


def canonical_payload(document: Dict[str, Any]) -> str:
    """产物里**真正进哈希**的那部分的规范化字节串。

    规范化 = 键排序 + 紧凑分隔符 + 不转义中文。

    为什么要把这个字符串**单独存进产物**
    ----------------------------------
    哈希覆盖的是这一串，不是「产物文件本身」。这样一来：
    谁来读都先取 ``canonical_payload`` 再算哈希，
    不会因为「读的那边和写的那边序列化方式不一样」而对不上。
    """
    data = {k: v for k, v in document.items() if k not in META_KEYS}
    return json.dumps(data, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def content_hash(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def seal(document: Dict[str, Any]) -> Dict[str, Any]:
    """给产物**盖章**：填上 ``canonical_payload`` 与 ``content_hash``。

    用法::

        doc = {"schema_version": SCHEMA_VERSION, "kind": "...",
               "built_at": now_utc(), "source": {...},
               "heroes": {...}}
        seal(doc)
    """
    doc = dict(document)
    doc["schema_version"] = int(doc.get("schema_version", SCHEMA_VERSION))
    doc.setdefault("built_at", now_utc())
    doc.setdefault("source", {})
    doc.setdefault("quality_flags", [])
    payload = canonical_payload(doc)
    doc["canonical_payload"] = payload
    doc["content_hash"] = content_hash(payload)
    return doc


def verify(document: Dict[str, Any],
           expect_schema: Optional[int] = SCHEMA_VERSION) -> List[str]:
    """校验产物，返回**问题清单**（空 = 没问题）。

    为什么返回清单而不是抛异常
    --------------------------
    运行时不该因为一张种子表的哈希对不上就**整个起不来** ——
    那是「宁可少一句也不能崩」的取舍。真正需要硬失败的场合
    （构建脚本自检、CI）用 ``strict=True``。
    """
    problems: List[str] = []
    version = document.get("schema_version")
    if expect_schema is not None and version != expect_schema:
        problems.append(f"schema_version 是 {version}，期望 {expect_schema}")
        return problems

    stored_payload = document.get("canonical_payload")
    stored_hash = document.get("content_hash")
    if not stored_payload or not stored_hash:
        problems.append("产物没有 canonical_payload / content_hash（没封过章）")
        return problems

    actual_payload = canonical_payload(document)
    if actual_payload != stored_payload:
        problems.append("canonical_payload 与产物当前内容对不上（被改过？）")
    actual_hash = content_hash(actual_payload)
    if actual_hash != stored_hash:
        problems.append("content_hash 与内容对不上（被改过？）")
    return problems


def save(path: str, document: Dict[str, Any]) -> str:
    """写产物（原子写：先写临时文件再换名）。

    写一半断电不该留下半个 JSON —— 那下次启动直接读不出来。
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(document, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp, path)
    return str(document.get("content_hash") or "")


def load(path: str, expect_schema: Optional[int] = SCHEMA_VERSION,
         strict: bool = False) -> Dict[str, Any]:
    """读产物并校验。

    ``strict=True`` 时校验不过直接抛 :class:`SeedIndexError`；
    默认不抛 —— 运行时宁可带着问题继续跑，也要把问题**记下来**
    （见 :func:`load_with_problems`）。
    """
    document, problems = load_with_problems(path, expect_schema)
    if problems and strict:
        raise SeedIndexError("；".join(problems))
    return document


def load_with_problems(path: str,
                       expect_schema: Optional[int] = SCHEMA_VERSION
                       ) -> "tuple[Dict[str, Any], List[str]]":
    """读产物 + 返回问题清单（读不出来也算一个问题）。"""
    problems: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            document = json.load(fh)
    except FileNotFoundError:
        return {}, ["文件不存在：%s" % path]
    except (OSError, ValueError) as exc:
        return {}, ["读不出来：%s" % exc]
    if not isinstance(document, dict):
        return {}, ["不是一个 JSON 对象"]
    problems.extend(verify(document, expect_schema))
    return document, problems
