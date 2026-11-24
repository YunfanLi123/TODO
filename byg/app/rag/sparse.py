"""**稀疏检索用的分词**（中文 char bigram）。

为什么自己切
------------
Qdrant 的 BM25 走的是**文本索引**：建索引时按分词器切词，查询时同样切。
而它的默认分词器把一段中文当成**一个词** ——

    "龙坑遭遇战该先占视野"  ->  ["龙坑遭遇战该先占视野"]

中文没有空格，整句一个词等于「一个字都匹配不上」。
（本地模式的分词规则是 ``[^\\W_]+``，CJK 全算词字符。）

所以：**我们自己切好，再把词用空格拼起来存进去**。
Qdrant 的分词器再按空格拆回来，得到的正是我们的词。

    原文   "龙坑遭遇战该先占视野"
    切词   龙 坑 龙坑 遭遇 遇战 战该 该先 先占 占视 视野 …
    存储   "龙 坑 龙坑 遭遇 遇战 战该 该先 先占 占视 视野"

这一套与 :func:`app.rag.embeddings._tokens` **同一口径** ——
稠密路与稀疏路看到的词一致，混合检索时不会各说各话。
"""

from __future__ import annotations

from typing import List

#: 单个词的最大长度（字符）。超过就截掉 ——
#: 防止把整篇文章当成一个词。
MAX_TOKEN_LEN = 24


def tokens(text: str) -> List[str]:
    """中文按**字符 bigram** 切（不需要分词器，零依赖）。

    单字 + 相邻二字组合：既能抓住「龙坑」「龙坑图标」这种词，
    又不会因为切错词而整条检索失效。

    与 :func:`app.rag.embeddings._tokens` 同一口径 ——
    故意**重新实现**而不是直接 import：那个函数是 embeddings 的私有实现，
    它可能为了哈希嵌入调整过；这里要的是**稳定的词**。
    """
    chars = [c for c in (text or "") if not c.isspace()]
    out: List[str] = list(chars)
    for i in range(len(chars) - 1):
        out.append(chars[i] + chars[i + 1])
    return [t[:MAX_TOKEN_LEN] for t in out]


def pretokenized(text: str) -> str:
    """把文本变成**预分词串**（空格分隔），存进 payload 的文本字段。

    存这个而不是原文，是为了让 Qdrant 的默认分词器把词**原样还原**。
    """
    return " ".join(tokens(text))


def query_text(text: str) -> str:
    """查询用词串。与 :func:`pretokenized` 同一口径 ——
    切法不一样的话，BM25 一个词都匹配不上。"""
    return pretokenized(text)
