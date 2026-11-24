"""**Qdrant 连接与集合布局**。

为什么换成 Qdrant
----------------
模式库原来装在 SQLite 里（``mode_library.db``），向量以 ``LargeBinary``
存 float32 字节，检索时把 4 077 个块**全部读进内存**再 numpy 矩阵乘。
规模小的时候够用，但它有三个硬伤：

* **检索不是数据库的事**：过滤（时间窗/英雄/分路）在 Python 里做，
  向量也在 Python 里比 —— 数据库只当了个文件柜。
* **索引过期看不出来**：向量换档（hash ↔ minimax）时，
  维度对不上会算出**噪声相似度**，而外面看上去一切正常。
* **加一个维度就要改表**：多路召回、稀疏向量、混合检索都得自己拼。

Qdrant 把这些收进引擎里：HNSW/精确检索、**payload 过滤**、
**BM25 稀疏检索**（full-text）、混合打分，都是现成的。

部署形态
--------
三种，**同一套代码**：

===============  ======================================================
本地嵌入式      默认。``QdrantClient(path=<数据目录>/qdrant)`` ——
                 进程内跑，数据落磁盘，**不需要 Docker、不占端口**。
独立服务        设 ``QDRANT_URL=http://127.0.0.1:6333``。
Qdrant Cloud    设 ``QDRANT_URL`` + ``QDRANT_API_KEY``。
===============  ======================================================

本地模式是**精确检索**（不是近似），4 077 条规模下毫秒级 ——
召回不会因为换存储而掉点。
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional, Tuple

#: 文档集合（一条攻略 = 一个点，**不带向量**）。
DOCS_COLLECTION = "mode_documents"
#: 检索块集合（一个「模式 × 分支」= 一个点，带稠密向量）。
CHUNKS_COLLECTION = "mode_chunks"

#: 稠密向量的命名空间。
DENSE_VECTOR = "dense"

#: 文档集合用的**占位向量维度**（文档不参与向量检索，见
#: :func:`ensure_collections` 的说明）。
DOCS_DIM = 1

#: **预分词文本**所在的 payload 字段 —— BM25 稀疏检索就是查它。
#:
#: 为什么字段名是 ``search_text`` 而不是 ``text``：存进去的**不是原文**，
#: 而是我们自己切好的词（空格分隔）。见 :mod:`app.rag.sparse`。
TEXT_FIELD = "search_text"


def qdrant_path() -> str:
    """本地嵌入式模式的数据目录。

    跟着 ``MATCH_DATA_DIR`` 走 —— exe 与开发环境指同一个地方，
    打包后也不会跑到临时目录里（临时目录退出就没了）。

    ``MATCH_QDRANT_PATH`` 可以单独指一个目录（测试与多库场景用）。
    """
    override = (os.environ.get("MATCH_QDRANT_PATH") or "").strip()
    if override:
        return os.path.abspath(override)
    base = os.environ.get("MATCH_DATA_DIR")
    if not base:
        from ..collection.settings import data_dir

        base = data_dir()
    return os.path.join(base, "qdrant")


def server_url() -> str:
    """服务端 / 云的地址。空串 = 用本地嵌入式模式。"""
    return (os.environ.get("QDRANT_URL") or "").strip()


def api_key() -> str:
    return (os.environ.get("QDRANT_API_KEY") or "").strip()


_client = None
_client_lock = threading.Lock()


def get_client() -> Any:
    """拿客户端（进程内单例）。

    本地模式**会锁住数据目录** —— 所以整个进程只能有一个实例。
    这里用模块级单例保证「同一个进程里只有一份句柄」。
    """
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        url = server_url()
        if url:
            from qdrant_client import QdrantClient

            _client = QdrantClient(url=url, api_key=api_key() or None,
                                   timeout=30)
        else:
            from qdrant_client import QdrantClient

            path = qdrant_path()
            os.makedirs(path, exist_ok=True)
            _client = QdrantClient(path=path)
        return _client


def close_client() -> None:
    """关掉客户端（本地模式会释放目录锁）。

    **顺带把 ``_client`` 属性删掉** —— ``QdrantClient.__del__`` 会在解释器
    退出时再调一次 ``close()``，而那时 ``sys.meta_path`` 已经是 None，
    本地模式会抛一个 ``ImportError`` 刷在控制台上（程序本身没事，
    但用户看到一串报错会以为坏了）。
    """
    global _client
    with _client_lock:
        if _client is not None:
            try:
                _client.close()
            except Exception:                              # noqa: BLE001
                pass
            try:
                del _client._client
            except Exception:                              # noqa: BLE001
                pass
            _client = None


def _atexit_close() -> None:
    try:
        close_client()
    except Exception:                                      # noqa: BLE001
        pass


import atexit                                              # noqa: E402

atexit.register(_atexit_close)


def collection_exists(client: Any, name: str) -> bool:
    try:
        return bool(client.collection_exists(name))
    except Exception:                                      # noqa: BLE001
        return False


def collection_state(client: Any, name: str) -> Tuple[bool, str]:
    """集合在不在？**连不上**的时候把原因带回来。

    为什么不能只看在不在
    ------------------
    :func:`collection_exists` 只回 True/False ——
    **连不上**与**不存在**长得一模一样。于是检索失败会被写成
    「库是空的」，而真正该做的是修连接。
    那句话会直接喂给模型（``harness/tools.py``），
    模型就照着「库是空的」去建议用户补攻略了 —— 方向完全错。
    """
    try:
        return bool(client.collection_exists(name)), ""
    except Exception as exc:                              # noqa: BLE001
        return False, "%s: %s" % (type(exc).__name__, exc)


def delete_collection(client: Any, name: str) -> None:
    """删集合（重建前用）。不存在也**不算失败**。"""
    try:
        if collection_exists(client, name):
            client.delete_collection(name)
    except Exception:                                      # noqa: BLE001
        pass


def ensure_collections(client: Any, dim: int) -> Dict[str, bool]:
    """确保两个集合存在。返回每个集合是不是**这次新建**的。

    ``dim`` = 稠密向量维度。**维度对不上必须重建** ——
    拿 256 维的查询向量去点乘 1536 维的库向量，结果是噪声相似度，
    而外面看不出任何异常（这条在 SQLite 时代就踩过，见
    ``app/rag/retriever.py`` 里的维度守卫）。
    """
    from qdrant_client import models

    created: Dict[str, bool] = {}

    # -- 文档集合：1 维占位向量 ---------------------------------------
    #
    # **Qdrant 的每个点都得带向量**（本地模式不支持纯 payload 点），
    # 而文档**不参与向量检索** —— 给它一个 1 维零向量即可。
    # 单独用 1 维还有个好处：它**不会**跟检索块集合的真实维度（1536 / 256）
    # 互相牵连 —— 建文档时还不知道向量维度。
    if collection_exists(client, DOCS_COLLECTION):
        if _dense_dim(client.get_collection(DOCS_COLLECTION)) == DOCS_DIM:
            created[DOCS_COLLECTION] = False
        else:
            delete_collection(client, DOCS_COLLECTION)
    if not collection_exists(client, DOCS_COLLECTION):
        client.create_collection(
            collection_name=DOCS_COLLECTION,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=DOCS_DIM, distance=models.Distance.COSINE),
            },
        )
        created[DOCS_COLLECTION] = True

    # -- 检索块集合：真实维度 -------------------------------------------
    if collection_exists(client, CHUNKS_COLLECTION):
        current = _dense_dim(client.get_collection(CHUNKS_COLLECTION))
        # ``current == 0`` = 这个集合**没有稠密向量**（老版本建的纯 payload
        # 集合，或者别的工具建错了）→ 一样要重建。
        if current and int(dim) and int(current) != int(dim):
            # **维度变了** —— 旧向量已经没法用了，只能重建。
            delete_collection(client, CHUNKS_COLLECTION)
        elif current:
            created[CHUNKS_COLLECTION] = False
        else:
            delete_collection(client, CHUNKS_COLLECTION)
    if not collection_exists(client, CHUNKS_COLLECTION):
        client.create_collection(
            collection_name=CHUNKS_COLLECTION,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=int(dim), distance=models.Distance.COSINE),
            },
        )
        _create_text_index(client)
        created[CHUNKS_COLLECTION] = True
    return created


def _create_text_index(client: Any) -> None:
    """在**预分词文本**字段上建 BM25 文本索引。

    这是 Qdrant 的稀疏检索入口：查询时给一段文字，它用 BM25
    （含 IDF）算分。全程在引擎里做，Python 不用维护词表。

    **分词器用默认的 word 切分** —— 因为存进去的文本**已经是我们切好的词**
    （空格分隔），再切一次正好还原。中文没有空格，默认分词器会把整句
    当一个词，所以必须由我们自己先切（见 :mod:`app.rag.sparse`）。
    """
    from qdrant_client import models

    client.create_payload_index(
        collection_name=CHUNKS_COLLECTION,
        field_name=TEXT_FIELD,
        field_schema=models.TextIndexParams(
            type=models.TextIndexType.TEXT,
            tokenizer=models.TokenizerType.WORD,
            lowercase=True,
            scoring=models.TextScoringParams(type=models.TextScoringType.BM25),
        ),
    )


def _dense_dim(info: Any) -> int:
    """集合里稠密向量的维度。取不到就返回 0（= 不判过期）。"""
    try:
        params = info.config.params.vectors
    except Exception:                                      # noqa: BLE001
        return 0
    if params is None:
        return 0
    # 未命名向量（``VectorParams``）与命名向量（dict）两种形状都要认。
    size = getattr(params, "size", None)
    if size:
        return int(size)
    try:
        for value in params.values():
            if getattr(value, "size", None):
                return int(value.size)
    except Exception:                                      # noqa: BLE001
        pass
    return 0


def dense_dim(client: Any) -> int:
    """当前集合的稠密向量维度（没有集合 = 0）。"""
    if not collection_exists(client, CHUNKS_COLLECTION):
        return 0
    try:
        return _dense_dim(client.get_collection(CHUNKS_COLLECTION))
    except Exception:                                      # noqa: BLE001
        return 0


def describe(client: Any) -> Dict[str, Any]:
    """给界面看的库状态。读不出来就说读不出来，不假装。"""
    out: Dict[str, Any] = {
        "mode": "server" if server_url() else "local",
        "path": server_url() or qdrant_path(),
        "documents": 0,
        "chunks": 0,
        "vector_dim": 0,
    }
    try:
        if collection_exists(client, DOCS_COLLECTION):
            out["documents"] = int(
                client.count(DOCS_COLLECTION, exact=True).count)
    except Exception:                                      # noqa: BLE001
        pass
    try:
        if collection_exists(client, CHUNKS_COLLECTION):
            out["chunks"] = int(
                client.count(CHUNKS_COLLECTION, exact=True).count)
            out["vector_dim"] = dense_dim(client)
    except Exception:                                      # noqa: BLE001
        pass
    return out


def point_id(text: str) -> str:
    """点 id：由字符串**确定性地**映射成 UUID。

    同一个 chunk_id 永远得到同一个 id → 重建时是**覆盖**而不是堆积，
    也不会撞「同名两条」的坑。
    """
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, "match-analyzer:" + text))


def bulk_write_mode(client: Any, enabled: bool) -> bool:
    """建库时把本地存储的 **fsync 关掉**（写完再打开）。

    为什么
    ------
    本地模式每个点**一次 commit**（fsync）—— 4 077 个点实测 70 秒，
    而其中绝大部分时间花在「等磁盘确认」上。

    建索引时这 70 秒是**白等**：索引写完就是完整的，中途断电大不了重建一次
    （而重建只需要重新调一遍嵌入接口 —— 有花钱的风险，所以更该保成品的完整性）。

    所以：**建库期间关 fsync，建完立刻打开**。
    索引是**我们自己写的**（不是用户的数据），这个取舍成立。

    只在**本地嵌入式模式**下有效（服务端用 Qdrant 自己的存储策略）。
    返回是否真的改到了。
    """
    if server_url():
        return False
    try:
        local = getattr(client, "_client", None)
        collections = getattr(local, "collections", None) or {}
        touched = False
        for collection in collections.values():
            storage = getattr(collection, "storage", None)
            conn = getattr(storage, "storage", None)
            if conn is None:
                continue
            conn.execute("PRAGMA synchronous=%s" % ("OFF" if enabled else "FULL"))
            conn.commit()
            touched = True
        return touched
    except Exception:                                      # noqa: BLE001
        return False


def all_collections() -> List[str]:
    return [DOCS_COLLECTION, CHUNKS_COLLECTION]
