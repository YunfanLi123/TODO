"""MiniMax / DeepSeek / Qwen 的多模态通道（薄客户端）。

这一层只做一件事：**把「内容块」送出去，把「回答」拿回来**。
它不认识王者、不认识模式库、不认识时间窗 —— 那些是
:mod:`app.collection.vlm_recognizer` 的事。

两种协议都支持
--------------
- **OpenAI 兼容**（DeepSeek / 百炼 / 大多数中转）：
  ``POST {endpoint}``，内容块 ``{"type": "image_url", "image_url": {"url": "data:..."}}``，
  鉴权头 ``Authorization: Bearer``。
- **Anthropic 兼容**（MiniMax）：
  ``POST {endpoint}``，内容块 ``{"type": "image", "source": {"type": "base64", ...}}``，
  鉴权头 ``x-api-key``。

**端点由调用方给完整 URL，客户端不拼路径** ——
DeepSeek 是 ``/chat/completions``（不带 ``/v1``），百炼是
``/compatible-mode/v1/chat/completions``，MiniMax 是 ``/anthropic/v1/messages``。
我们自��拼过一次路径就 404 过（少写 ``/v1``），不能再靠猜。

**「关思考」的字段名按供应方不同**：
MiniMax / DeepSeek 用 ``thinking: {type: "disabled"}``，
百炼用 ``enable_thinking: false``。写死一个，Qwen 那条会不认 ——
模型就会开始长篇大论，我们的 JSON 解析直接失败。

凭据
----
只从**应用自己的配置**来（见 :mod:`app.collection.settings`）：
环境变量 → 数据目录的 ``config.yaml``。**不要**读运行时自己的配置 ——
那里面是代理端点与占位密钥。
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: 一次请求的最长等待（秒）。图片理解比纯文本慢得多。
DEFAULT_TIMEOUT = 300.0

#: 重试次数（不含首次）。
MAX_RETRY = 2

#: 请求体上限（百炼/DeepSeek 官方限制在几 MB 量级，留足余量）。
MAX_BODY_BYTES = 48 * 1024 * 1024

#: 「看画面」那个槽位的名字。
SLOT_VLM = "vlm"


class VlmError(RuntimeError):
    """调用失败。**所有失败都要能说清是哪一类**。

    分类不是为了好看：调用方要按类别决定「重试 / 降级 / 记缺失」，
    一句「调用失败」没法处理。
    """

    def __init__(self, message: str, kind: str = "unknown",
                 status: int = 0, detail: str = ""):
        super().__init__(message)
        self.kind = kind          # auth / rate_limit / timeout / server / bad_request / body
        self.status = status
        self.detail = detail


@dataclass
class Credentials:
    """一次调用要用的凭据与端点。"""
    api_key: str
    endpoint: str
    protocol: str = "openai"
    thinking_field: str = "thinking"
    model: str = ""
    source: str = ""            # 凭据来自哪儿（排查用）


@dataclass
class VlmResult:
    """一次调用的结果。"""
    text: str = ""
    model: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    def json(self) -> Any:
        """把回答当 JSON 解析。解析不了就抛 :class:`VlmError`（kind=bad_json）。

        模型的回答里常带 markdown 代码块围栏，先剥掉再解析；
        剥完还不行就是**格式不对**，不是「内容不对」——两者的处置不同。
        """
        raw = (self.text or "").strip()
        if raw.startswith("```"):
            parts = raw.split("```", 2)
            raw = (parts[1] if len(parts) > 1 else raw).strip()
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        if not raw:
            raise VlmError("模型没有返回内容", kind="empty")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise VlmError(
                f"模型回答不是合法 JSON：{exc}", kind="bad_json",
                detail=(self.text or "")[:500])


# ---------------------------------------------------------------- 凭据


def find_credentials(slot: str = SLOT_VLM,
                     data_dir: Optional[str] = None) -> Optional[Credentials]:
    """某个槽位的凭据。没有配就返回 None（**不猜、不编**）。"""
    from . import settings

    if data_dir:
        old = os.environ.get("MATCH_DATA_DIR")
        os.environ["MATCH_DATA_DIR"] = data_dir
        try:
            return settings.credentials_for(slot)
        finally:
            if old is None:
                os.environ.pop("MATCH_DATA_DIR", None)
            else:
                os.environ["MATCH_DATA_DIR"] = old
    return settings.credentials_for(slot)


# ---------------------------------------------------------------- 内容块


def image_block(data: bytes, media_type: str = "image/jpeg") -> Dict[str, Any]:
    """一张图片的内容块（base64 内联）。

    两种协议都要用：``image`` 是 Anthropic 形状，``image_url`` 是 OpenAI 形状。
    客户端按协议挑一个用（见 :func:`build_request`）。
    """
    encoded = base64.b64encode(data).decode("ascii")
    return {
        "anthropic": {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": encoded},
        },
        "openai": {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{encoded}"},
        },
    }


def text_block(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text}


# ---------------------------------------------------------------- 请求


def _thinking_body(thinking_field: str) -> Dict[str, Any]:
    """「关思考」的那个字段 —— **每家字段名不一样**。

    - MiniMax / DeepSeek：``{"thinking": {"type": "disabled"}}``
    - 百炼：``{"enable_thinking": false}``
    """
    if thinking_field == "enable_thinking":
        return {"enable_thinking": False}
    return {"thinking": {"type": "disabled"}}


def build_request(
    model: str,
    blocks: Sequence[Dict[str, Any]],
    creds: Credentials,
    system: Optional[str] = None,
    max_tokens: int = 4096,
    thinking: bool = False,
    temperature: Optional[float] = None,
) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    """造出 (url, headers, body)。拆出来是为了能单测。

    ``url`` 就是配置里的**完整端点**，这里一个字都不加 ——
    加路径是踩过坑的地方。

    ``temperature`` 默认 ``None`` 表示**不传**，走供应方自己的默认值。
    读画面那几个调用显式传 0 —— 它们的默认温度不为 0，同一帧问两次
    能给出两个不同的英雄（实测连问四次得到韩信/哪吒/孙策/曜），
    结论跟着采样抖动，报告就无法复现。
    只有需要稳定的调用才传，其余不传：有的模型（推理类）不接受这个字段，
    一律塞进去会让那些配置直接 400。
    """
    url = creds.endpoint
    if creds.protocol == "anthropic":
        headers = {"Content-Type": "application/json", "x-api-key": creds.api_key}
        content: List[Dict[str, Any]] = []
        for b in blocks:
            if b.get("type") == "text":
                content.append({"type": "text", "text": b["text"]})
            else:
                content.append(b["anthropic"])
        body: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if system:
            body["system"] = system
        if temperature is not None:
            body["temperature"] = temperature
    else:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {creds.api_key}",
        }
        content = []
        for b in blocks:
            if b.get("type") == "text":
                content.append({"type": "text", "text": b["text"]})
            else:
                content.append(b["openai"])
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if temperature is not None:
            body["temperature"] = temperature
    if not thinking:
        body.update(_thinking_body(creds.thinking_field))
    return url, headers, body


def _classify(status: int) -> str:
    if status in (401, 403):
        return "auth"
    if status == 429:
        return "rate_limit"
    if status in (400, 404, 413, 422):
        return "bad_request"
    if status >= 500:
        return "server"
    return "unknown"


def call_model(
    model: str,
    blocks: Sequence[Dict[str, Any]],
    creds: Optional[Credentials] = None,
    slot: str = SLOT_VLM,
    data_dir: Optional[str] = None,
    system: Optional[str] = None,
    max_tokens: int = 4096,
    thinking: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    retry: int = MAX_RETRY,
    temperature: Optional[float] = None,
) -> VlmResult:
    """发一次请求。失败抛 :class:`VlmError`，并带上可判别的 ``kind``。"""
    if creds is None:
        creds = find_credentials(slot, data_dir)
    if creds is None:
        raise VlmError(f"没有配置「{slot}」这个槽位的模型", kind="auth")

    url, headers, body = build_request(
        model, blocks, creds, system=system, max_tokens=max_tokens,
        thinking=thinking, temperature=temperature)
    raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
    if len(raw) > MAX_BODY_BYTES:
        raise VlmError(
            f"请求体 {len(raw) // 1024 // 1024} MB 超过上限", kind="body")

    last: Optional[VlmError] = None
    for attempt in range(retry + 1):
        req = urllib.request.Request(url, data=raw, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return VlmResult(
                text=_extract_text(data), model=model,
                usage=data.get("usage") or {}, raw=data)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:800]
            except Exception:                          # noqa: BLE001
                pass
            kind = _classify(exc.code)
            last = VlmError(f"HTTP {exc.code}", kind=kind, status=exc.code, detail=detail)
            # 4xx（除限流）是**请求本身**的问题，重试没有意义。
            if kind in ("bad_request", "auth", "body"):
                raise last
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            last = VlmError(f"网络错误：{reason}", kind="timeout", detail=str(reason))
        except TimeoutError as exc:
            last = VlmError(f"超时（{timeout:.0f} 秒）", kind="timeout", detail=str(exc))
        except json.JSONDecodeError as exc:
            last = VlmError(f"响应不是 JSON：{exc}", kind="server", detail=str(exc))

        if attempt < retry:
            # 指数退避：限流与 5xx 都值得再等等。
            time.sleep(min(30.0, 2.0 ** (attempt + 1)))
    raise last or VlmError("调用失败", kind="unknown")


def _extract_text(data: Dict[str, Any]) -> str:
    """从 Anthropic 形状 / OpenAI 形状的响应里取文本。"""
    content = data.get("content")
    if isinstance(content, list):
        parts = [b.get("text") or "" for b in content if isinstance(b, dict)
                 and b.get("type") == "text"]
        if parts:
            return "\n".join(parts).strip()
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") or {}
        if isinstance(msg, dict) and msg.get("content"):
            return str(msg["content"]).strip()
    return ""
