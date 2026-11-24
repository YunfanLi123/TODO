"""RAG 层：把操作模式库变成**可检索**的库。

四块：

* :mod:`app.rag.qdrant_client` —— Qdrant 连接与集合布局（本地嵌入式 / 服务端同接口）；
* :mod:`app.rag.store` —— 文档与检索块的读写（**Qdrant**）；
* :mod:`app.rag.embeddings` —— 嵌入向量（三档降级：本地 hash → 远程 API）；
* :mod:`app.rag.retriever` —— 检索（过滤 → 多路召回 → 打分融合 → 预算裁剪）。

为什么要做成 RAG
----------------
原来清单完全由代码按时间窗硬过滤（:func:`app.rules.checklist.window_checklist`），
模型**没有检索能力** —— 它只能对代码挑好的那几条发表意见。
RAG 化之后，模型可以自己写「我这一段在干嘛」，检索层把**相关的攻略原文**
取回来给它判。
"""

from .retriever import RetrievalResult, Retriever, search_modes
from .store import ChunkHit, ModeStore, make_chunks, normalize_document

__all__ = [
    "Retriever",
    "RetrievalResult",
    "search_modes",
    "ChunkHit",
    "ModeStore",
    "make_chunks",
    "normalize_document",
]
