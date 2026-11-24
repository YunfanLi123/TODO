"""**模式库的现场构建** —— 首次运行时把库建出来。

为什么需要它
------------
模式库是**随包发布**的内容，不是用户录的数据：

* 真源在 ``app/seed/sources/*.txt``（仓库里），随包发布的是它的**构建产物**
  ``app/seed/hok_modes.json`` + ``app/seed/hok_hero_modes.json``；
* 检索需要**向量**（4 077 个块）。把建好向量的数据库塞进 exe 要多 17 MB ——
  不值当（exe 现在 13.7 MB），而且每次改一条攻略都要重打 exe。

所以改成：**第一次用到检索时，从包内的 JSON 现场建库**
（文档 + 分块 + hash 嵌入），建在**用户的数据目录**里
（``<数据目录>/mode_library.db``）。之后就直接读。

**绝不复用用户的 ``app.db``** —— 那是用户的对局数据与配置，
模式库是我们发版带过去的内容，两者混在一起就分不清谁是谁了
（模式库现在是 **Qdrant**，住在 ``<数据目录>/qdrant``）。

一致性保证
----------
现场构建用的 JSON 与构建流水线解析出来的内容**逐条一致**（已核对：
84 + 1 596 条，字段无差异）—— 所以「检索到的打法」与「判定用的清单」
来自同一份内容，不会出现两套打法。

失败怎么办
----------
建不出来就**返回 False**，调用方按「没有检索能力」降级
（判定仍走原清单，绝不因此报错）。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from . import qdrant_client as qc

#: 包内种子 JSON（层 → 文件名）。
SEED_FILES = {
    "lane": "hok_modes.json",
    "hero": "hok_hero_modes.json",
}

#: 一次构建最多读多少条（防呆：不该读到十万条）。
_MAX_DOCS = 20000

#: 本进程里试过几次（只试一次，别每次检索都去摸库）。
_ENSURED: Optional[bool] = None


def desired_backend() -> str:
    """当前**应该**用哪个嵌入后端（``minimax`` / ``hash``）。

    口径与 :func:`app.rag.embeddings.get_provider` 一致：
    显式入参 > 环境变量 > 应用配置 > ``hash``。
    """
    from .embeddings import get_provider  # noqa: F401  （只为触发 import 检查）

    backend = (os.environ.get("MATCH_EMBEDDING_BACKEND") or "").strip()
    if not backend:
        try:
            from .embeddings import _configured_credentials

            backend = str(_configured_credentials().get("backend") or "").strip()
        except Exception:                              # noqa: BLE001
            backend = ""
    return backend or "hash"


def _seed_paths() -> List[str]:
    """找到可用的模式库 JSON（``*_modes.json``）。

    优先用**应用真正在读的那一份**（:func:`app.rules.library_source.library_files`）
    —— 用户手改了 ``mode-library/`` 时，检索到的内容要跟判定用的清单一致，
    否则两套打法会打架。

    只收 ``*_modes.json``：``*_patterns.json`` / ``*_skipped.json`` /
    ``*_retired.json`` 不是模式记录，混进来会污染库。
    """
    candidates: List[str] = []
    try:
        from ..rules.library_source import library_files

        candidates.extend(str(p) for p in library_files("hok", skip_bp=True))
    except Exception:                                  # noqa: BLE001
        pass
    # 兜底：包内种子目录。
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    seed_dir = os.path.join(base, "seed")
    try:
        candidates.extend(os.path.join(seed_dir, name)
                          for name in sorted(os.listdir(seed_dir))
                          if name.endswith("_modes.json"))
    except OSError:
        pass

    out: List[str] = []
    for path in candidates:
        if path in out:
            continue
        base = os.path.basename(path)
        if "_bp_modes.json" in base:
            # BP 层本轮不 RAG 化（见计划「非目标」）。
            continue
        if os.path.isfile(path) and path.endswith("_modes.json"):
            out.append(path)
    return out


def _read_rules() -> List[Dict[str, Any]]:
    """读出全部规则记录（原始形状，**逐字**）。

    **按 id 去重**（保留**第一个**）——
    查找顺序里可能同时出现「用户那份」与「包内那份」，
    内容一样但 id 相同；不去重会撞数据库的唯一约束。
    """
    rules: List[Dict[str, Any]] = []
    seen: set = set()
    for path in _seed_paths():
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        if not isinstance(data, list):
            continue
        for row in data:
            if not isinstance(row, dict):
                continue
            rid = str(row.get("id") or "")
            if not rid or rid in seen:
                continue
            seen.add(rid)
            rules.append(row)
            if len(rules) >= _MAX_DOCS:
                return rules
    return rules


def has_content() -> bool:
    """库里已经有内容了吗。

    **文档和块都要有** —— 只看文档会踩一个坑：
    建库时 ``upsert_documents`` 成功、``upsert_chunks`` 失败，
    库里就剩「有文档、无块」的**半成品**。那时检索一条都召不回，
    而这里判成「已就绪」，:func:`ensure` 就不再重建 ——
    失败被当成「库里没有」，永远好不了。
    """
    try:
        from .store import ModeStore

        stats = ModeStore().stats()
        return bool(stats.get("documents")) and bool(stats.get("chunks"))
    except Exception:                                  # noqa: BLE001
        return False


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _retired_ids(game: str = "hok") -> List[str]:
    """已作废的模式 id（``<game>_retired.json``）。

    **必须过滤** —— 否则检索会把退休打法捞回来，
    拿旧打法纠正玩家。
    """
    try:
        from ..rules import library_source

        path = os.path.join(library_source.library_dir(),
                            "%s_retired.json" % game)
        if not os.path.isfile(path):
            return []
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:                                  # noqa: BLE001
        return []
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict):
        for key in ("retired", "ids", "rules"):
            if isinstance(data.get(key), list):
                return [str(x) for x in data[key]]
    return []


def build(*, embed: str = "") -> Dict[str, Any]:
    """从包内 JSON 现场建库。返回统计。

    **不覆盖**已有的库（除非调用方先删掉它）—— 用户的库里可能有
    我们不知道的编辑。

    ``embed`` 留空 = 用**当前配置的后端**（见 :func:`desired_backend`）。
    早先这里写死 ``"hash"``，等于永远建 hash 索引 —— 配了 minimax 也没用。
    """
    from .embeddings import get_provider, provider_name

    if not embed:
        embed = desired_backend()
    from .store import ModeStore, make_chunks, normalize_document

    rules = _read_rules()
    if not rules:
        return {"ok": False, "reason": "找不到模式库 JSON"}

    store = ModeStore()
    docs: List[Dict[str, Any]] = []
    for rule in rules:
        rid = str(rule.get("id") or "")
        layer = "hero" if rid.startswith("HOK-HERO-") else "lane"
        docs.append(normalize_document(
            rule,
            game=str(rule.get("game") or "hok"),
            version=str(rule.get("version") or "S45"),
            layer=layer,
        ))
    retired = _retired_ids()

    # **退役清单与库对不上时要报出来**。
    #
    # 清单里的 id 在库里不存在 = 那条规则已经随旧种子一起消失了
    # （旧 id 体系整体换掉时留下的）。它既不在库里、也不在退役记录里 ——
    # 凭空消失，而外面看不出任何异常。
    #
    # 这不是阻断（历史复盘靠的是 ``Analysis`` 快照，不是重算），
    # 但必须**记账**：报出来，让人知道这批规则去哪了。
    doc_ids = {str(d.get("id") or "") for d in docs}
    vanished = sorted(r for r in retired if r not in doc_ids)

    chunks: List[Dict[str, Any]] = []
    for doc in docs:
        chunks.extend(make_chunks(doc))

    # **先算向量，再动库。**
    #
    # 早先的顺序是「先写文本块、再算向量」—— 嵌入接口一失败（额度用尽就是
    # 最常见的那种），库里就剩下**有块没向量**的半成品：块在、向量不在，
    # 于是稠密路直接死掉，检索悄悄退化成**纯 BM25**。
    # 实测踩过：交付目录里的 ``mode_library.db`` 正是这个状态
    # （4 077 个块、0 个向量），检索质量差一截而外面看不出异常。
    #
    # 现在改成**算不成就什么都不写**：失败时旧索引原封不动，
    # 下次检索还会按 ``_index_is_stale`` 的判断重试。
    provider = get_provider(embed)
    pname = provider_name(provider)
    # 记下这批向量是**哪种嵌入**算的 —— 检索时要核对。
    # 拿 hash 的查询向量去比 minimax 的库向量，相似度全是噪声。
    vectors = provider.embed([c["text"] for c in chunks])
    mapping = {c["chunk_id"]: vectors[i] for i, c in enumerate(chunks)}

    # 到这儿才算成了 —— 现在才动库。
    #
    # **集合要先建好**：维度必须与这批向量对得上。
    # 对不上时 ``ensure_collections`` 会**删掉重建**（旧向量已无意义）。
    dim = int(vectors.shape[1])
    qc.ensure_collections(store.client, dim)
    # 先清检索块（它会**重建集合** → 新的 SQLite 连接），
    # **再**关 fsync —— 顺序反了的话，新连接又用回默认的 fsync。
    store.clear_chunks(dim)
    # 建库期间把本地存储的 fsync 关掉（每个点一次 commit 会慢好几倍），
    # **写完立刻打开** —— 索引是我们自己写的，不是用户的数据。
    bulk = qc.bulk_write_mode(store.client, True)
    try:
        store.upsert_documents(docs, retired=retired)
        store.upsert_chunks(chunks, embeddings=mapping,
                            embedding_model=pname,
                            embedding_dim=dim)
    finally:
        if bulk:
            qc.bulk_write_mode(store.client, False)
    # **索引的元信息** —— 「这批向量是哪种嵌入算的」「建了多少条」，
    # 存在索引旁边的一个小集合里。检索前要核对（见 ``_index_is_stale``）。
    meta_ok = store.set_index_meta({
        "embedding_model": pname,
        "embedding_backend": embed,
        "embedding_dim": dim,
        "doc_count": len(docs),
        "chunk_count": len(chunks),
        "retired_count": len(retired),
        "index_version": 2,
        "built_at": _now(),
    })
    if not meta_ok:
        # **元信息写不进去要说出来** —— 索引就说不清自己是哪一档的向量，
        # 下次会拿错档的查询向量去比库向量（相似度全是噪声，外面看不出来）。
        from .store import last_meta_error

        stats_meta_error = last_meta_error
    else:
        stats_meta_error = ""
    stats = store.stats()
    stats["ok"] = True
    stats["db"] = qc.qdrant_path()
    if stats_meta_error:
        stats["meta_error"] = stats_meta_error
    stats["embedding_model"] = pname
    stats["embedding_backend"] = embed
    if vanished:
        stats["vanished"] = vanished
        stats["vanished_count"] = len(vanished)
    try:
        from .embeddings import last_fallback_reason

        if last_fallback_reason:
            stats["fallback_reason"] = last_fallback_reason
    except Exception:                                  # noqa: BLE001
        pass
    return stats


def ensure(*, embed: str = "", force: bool = False,
           rebuild: bool = False) -> Dict[str, Any]:
    """确保模式库可用。

    已经有内容 → 直接返回（**不重建**，省时间也不覆盖用户的编辑）。
    没有 → 现场建；建不出来 → 返回 ``ok=False``，调用方降级。

    ``rebuild=True`` —— **强制重建**（用户点了「载入规则库」时用：
    他可能刚改了 JSON，这时必须重建，索引也要跟着重建）。

    **索引过期自动重建**：库里存着 hash 档向量、而当前配置要 minimax 档时，
    必须重建 —— 拿 hash 的查询向量去比 minimax 的库向量，相似度全是噪声，
    而且外面看不出异常（检索照样返回结果，只是全都不相关）。

    「有没有内容」每次都查（一次计数，很便宜）；**建**只试一次 ——
    建库要算 4 077 个向量，不能每次检索都来一遍。
    """
    global _ENSURED
    try:
        backend = embed or desired_backend()
        if not rebuild and has_content():
            stale = _index_is_stale(backend)
            if not stale:
                _ENSURED = True
                return {"ok": True, "rebuilt": False}
            # 索引与当前后端对不上 —— 重建（这是必须，不能省）。
            rebuild = True
        if _ENSURED is False and not force and not rebuild:
            # 已经试过并且失败了 —— 不再反复试。
            return {"ok": False, "reason": "模式库建不起来（已试过一次）"}
        stats = build(embed=backend)
        stats["rebuilt"] = bool(stats.get("ok"))
        _ENSURED = bool(stats.get("ok"))
        return stats
    except Exception as exc:                          # noqa: BLE001
        _ENSURED = False
        return {"ok": False, "reason": str(exc)}


def _index_is_stale(backend: str) -> bool:
    """库里已有的向量是不是**当前后端算的**。

    判断口径：索引的元信息里记着 ``embedding_model`` 与 ``embedding_dim``，
    与当前 provider 对不上就是过期。

    注意一个坑：provider 名字带维度（``hash-char-ngram-256``），
    而 hash 档的维度可由 ``MATCH_EMBEDDING_DIM`` 改 —— 维度变了同样要重建。

    **查的是索引旁边那一小条元信息**，不是 ``stats()`` ——
    后者要把 4 077 个块读一遍，而这个问题每次检索都要问。
    """
    try:
        from .embeddings import get_provider, provider_name
        from .store import ModeStore

        want = provider_name(get_provider(backend))
        store = ModeStore()
        meta = store.index_meta()
        model = str(meta.get("embedding_model") or "")
        if not model:
            # 有块但没向量 —— 也算过期（检索会退化到纯 BM25）。
            try:
                if store.stats().get("chunks"):
                    return True
            except Exception:                          # noqa: BLE001
                pass
            return False
        if model != want:
            return True
        # 维度也要对 —— 名字一样但维度变了（hash 档可配维度）同样是过期。
        dim = int(meta.get("embedding_dim") or 0)
        current = qc.dense_dim(store.client)
        if dim and current and dim != current:
            return True
        return False
    except Exception:                                  # noqa: BLE001
        # 读不到就**不重建** —— 建库要花钱（远程嵌入按量计费），
        # 判不出来时宁可让用户手动点「载入规则库」。
        return False


if __name__ == "__main__":                             # pragma: no cover
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                  # noqa: BLE001
        pass
    print(json.dumps(ensure(), ensure_ascii=False))
