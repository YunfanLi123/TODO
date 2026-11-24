"""嵌入向量：三档降级。

为什么是三档
------------
本机只有 numpy（没有 sentence-transformers / torch / faiss），
而远程嵌入 API 又要**额度**（2026-10-31 实测 Token Plan 额度已用尽）。
所以：

1. ``hash``（默认兜底）：字符 n-gram 哈希 → 固定维度向量。
   **离线、零成本、完全可复现**。精度一般，但足够做「同一批词」的粗排，
   而且**索引与单测都能在没有网络、没有额度的机器上跑**。
2. ``minimax``：MiniMax ``embo-01``（OpenAI 兼容 ``/v1/embeddings``，
   ``type=query|db``）。生产质量，需要配置 ``embedding`` 槽位。
3. ``sentence-transformers``：本地模型。**本轮不做**（要新增依赖）。

借鉴：同作者 xiagu 项目的 ``app/knowledge/embeddings.py``
（``HashEmbeddings`` / ``DashScopeEmbeddings`` / ``SentenceTransformerEmbeddings``
+ ``get_embedding_provider()`` 三档切换）。
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

#: 上一次 ``get_provider()`` **发生降级的原因**（空串 = 没降级）。
#:
#: 为什么要有这个：降级本身是合理的兜底，但「配了 minimax 却悄悄跑在 hash 上」
#: 是**最坏的一种**——用户以为自己在用生产质量，实际检索质量差一截，
#: 而且从外面看不出来。这里留痕，调用方（建索引、检索）可以把它冒泡到报告里。
last_fallback_reason: str = ""


def _tokens(text: str) -> List[str]:
    """中文按**字符 bigram** 切（不需要分词器，零依赖）。

    单字 + 相邻二字组合：既能抓住「龙坑」「龙坑图标」这种词，
    又不会因为切错词而整条检索失效。
    """
    chars = [c for c in (text or "") if not c.isspace()]
    out = list(chars)
    for i in range(len(chars) - 1):
        out.append(chars[i] + chars[i + 1])
    return out


class HashEmbeddings:
    """字符 n-gram 哈希嵌入（本地兜底）。

    同样的文本永远得到同样的向量 —— 所以**索引可复现、单测可断言**。
    """

    name = "hash-char-ngram"

    def __init__(self, dim: int = 256):
        self.dim = int(dim)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            counts: Dict[str, int] = {}
            for tok in _tokens(text):
                counts[tok] = counts.get(tok, 0) + 1
            for tok, n in counts.items():
                digest = hashlib.md5(tok.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                # 次线性词频：出现 10 次不该有 10 倍权重。
                mat[i, idx] += sign * (1.0 + np.log(n))
        # L2 归一化 → 余弦相似度就是点积。
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class MiniMaxEmbeddings:
    """MiniMax ``embo-01``。

    **文档与查询要用不同的 type**（``db`` / ``query``）——
    两者的编码策略不同，混用会明显掉点。

    请求/响应格式（2026-10-31 实测确认）
    -----------------------------------
    MiniMax 的嵌入端点是**自家原生格式**，不是 OpenAI 兼容格式：

    * 请求字段是 ``texts``（不是 ``input``），
    * 响应字段是 ``vectors``（不是 ``data[].embedding``）。

    早先这里按 OpenAI 格式写（``input`` + ``data``），结果服务端回
    ``{"vectors": null, "base_resp": {"status_code": 2013,
    "status_msg": "invalid params, binding: expr_path=texts,
    cause=missing required parameter"}}`` —— 也就是说**这一档从来没真正跑通过**：
    配了 Key 也只是把错误换成另一种错误。

    维度实测 1536。
    """

    name = "minimax-embo-01"

    def __init__(self, api_key: str, endpoint: str = "", model: str = "embo-01",
                 timeout: float = 60.0):
        self.api_key = api_key
        self.endpoint = endpoint or "https://api.minimaxi.com/v1/embeddings"
        self.model = model
        self.timeout = timeout
        self._dim = 0

    def _call(self, texts: Sequence[str], embed_type: str) -> np.ndarray:
        payload = {
            "model": self.model,
            "texts": list(texts),
            "type": embed_type,
        }
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "authorization": "Bearer %s" % self.api_key,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        # 服务端出错时（额度、参数）会带 base_resp，且 vectors 为 null。
        # 那时要**把服务端原因说出来** —— 只说「没返回向量」等于什么都没说。
        base = data.get("base_resp") or {}
        status = base.get("status_msg") or ""
        code = base.get("status_code")
        if code and int(code) != 0:
            raise RuntimeError("嵌入接口报错 %s: %s" % (code, status))

        vectors = data.get("vectors") or []
        if not vectors:
            raise RuntimeError(
                "嵌入接口没返回向量%s" % ("（服务端：%s）" % status if status else ""))
        mat = np.asarray(vectors, dtype="float32")
        if mat.ndim != 2 or mat.shape[0] != len(texts):
            raise RuntimeError(
                "嵌入接口返回的条数不对：要 %d 条，拿到 %s"
                % (len(texts), list(mat.shape)))
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._dim = int(mat.shape[1])
        return mat / norms

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """批量嵌入**文档**。

        **必须分批**：模式库有 4 077 个块，一次性发过去会被服务端拒
        （超长请求 / 超额 token）。分批后逐批取回再拼起来。
        """
        return self._batched(texts, "db")

    def embed_query(self, text: str) -> np.ndarray:
        return self._call([text], "query")[0]

    def _batched(self, texts: Sequence[str], embed_type: str) -> np.ndarray:
        """分批调用并拼接。任一批失败就整体失败（不返回半截向量）。"""
        items = list(texts)
        if not items:
            return np.zeros((0, self._dim or 1), dtype="float32")
        batch = max(1, int(os.environ.get("MATCH_EMBEDDING_BATCH") or 32))
        chunks: List[np.ndarray] = []
        for start in range(0, len(items), batch):
            part = items[start:start + batch]
            chunks.append(self._call(part, embed_type))
        return np.vstack(chunks)


def _configured_credentials() -> Dict[str, str]:
    """从应用配置里取嵌入用的凭据。

    配置形状（``config.yaml`` 顶层）::

        embedding:
          backend: minimax
          endpoint: https://api.minimaxi.com/v1/embeddings
          model: embo-01
          api_key: 可选 —— 不写就继承 model.vlm.api_key

    **Key 默认继承 vlm 槽位**：同一供应方同一把 Key，不该让用户贴两遍。
    嵌入没有独立槽位（``model_presets.SLOTS`` 只有 vlm / text），
    所以继承只在 ``model.vlm.provider`` 与嵌入后端一致时才成立。

    读的是 :func:`app.collection.settings.app_config`，
    它认 ``MATCH_DATA_DIR``（exe 与开发环境指向同一个数据目录）。
    """
    try:
        from ..collection.settings import app_config
    except Exception:                                  # noqa: BLE001
        return {}
    try:
        cfg = app_config() or {}
    except Exception:                                  # noqa: BLE001
        return {}

    emb = cfg.get("embedding")
    if not isinstance(emb, dict):
        emb = {}
    model = cfg.get("model") or {}
    vlm = model.get("vlm") or {}

    backend = str(emb.get("backend") or "").strip()
    api_key = str(emb.get("api_key") or "").strip()
    if not api_key:
        # 继承 vlm 的 Key —— 但只在供应方对得上时（拿 minimax 的 Key 去请求
        # 百炼的地址 = 401，而那个 401 看不出是「借错了」）。
        vlm_provider = str(vlm.get("provider") or "").strip().lower()
        same = (backend in ("minimax", "embo-01") and vlm_provider == "minimax")
        if same:
            api_key = str(vlm.get("api_key") or "").strip()

    return {
        "api_key": api_key,
        "endpoint": str(emb.get("endpoint") or "").strip(),
        "model": str(emb.get("model") or "").strip(),
        "backend": backend,
    }


def get_provider(backend: str = "") -> Any:
    """按配置选一个 provider。

    顺序：**显式入参 > 环境变量 > 应用配置 > hash 兜底**。

    降级只在**两种明确许可**下发生：
      1. 没指定后端（默认就是 hash 档）；
      2. 指定了 minimax 但拿不到凭据 —— 这时**记一笔**（``last_fallback_reason``），
         免得「配了却还在降级」这件事没人发现。

    **不静默换档**：一旦配置明确要 minimax，没有 Key 就该让上层知道，
    而不是偷偷退回 hash 让检索质量悄悄变差。
    """
    global last_fallback_reason
    last_fallback_reason = ""

    configured = _configured_credentials()
    backend = (backend or os.environ.get("MATCH_EMBEDDING_BACKEND")
               or configured.get("backend") or "").strip()
    if not backend:
        backend = "hash"
    if backend == "hash":
        dim = int(os.environ.get("MATCH_EMBEDDING_DIM") or 256)
        return HashEmbeddings(dim)
    if backend in ("minimax", "embo-01"):
        api_key = (os.environ.get("MATCH_EMBEDDING_API_KEY")
                   or configured.get("api_key") or "").strip()
        endpoint = (os.environ.get("MATCH_EMBEDDING_ENDPOINT")
                    or configured.get("endpoint") or "").strip()
        model = (os.environ.get("MATCH_EMBEDDING_MODEL")
                 or configured.get("model") or "embo-01").strip()
        if not api_key:
            # 没凭据就退回 hash —— 宁可粗一点，也不能让整条链路瘫掉。
            # 但**必须留下痕迹**，否则「以为配好了」和「真的配好了」长得一样。
            last_fallback_reason = (
                "嵌入后端指定为 %s，但没有可用凭据（"
                "MATCH_EMBEDDING_API_KEY 或 config.yaml 的 model.vlm.api_key）"
                "，已退回 hash 降级档" % backend)
            return HashEmbeddings(
                int(os.environ.get("MATCH_EMBEDDING_DIM") or 256))
        return MiniMaxEmbeddings(api_key, endpoint, model)
    return HashEmbeddings(int(os.environ.get("MATCH_EMBEDDING_DIM") or 256))


def provider_name(provider: Any) -> str:
    name = getattr(provider, "name", "") or provider.__class__.__name__
    dim = int(getattr(provider, "dim", 0) or 0)
    return "%s-%d" % (name, dim) if dim else str(name)
