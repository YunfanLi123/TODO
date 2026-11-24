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
我们自己拼过一次路径就 404 过（少写 ``/v1``），不能再靠猜。

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

#: **视频**直传（URL / base64）的上限：MiniMax 文档写 50 MB。
MAX_VIDEO_INLINE_BYTES = 50 * 1024 * 1024

#: 但真正能内联的**原始视频**上限是 45 MB ——
#: base64 会把体积撑大三分之一，45 MB → 60 MB，加字段仍要压在
#: 请求体上限（64 MB）之内。写成 50 是拿文档数字当能用的数字，
#: 实际会 413。
MAX_VIDEO_INLINE_RAW_BYTES = 45 * 1024 * 1024

#: Files API 的视频上限：512 MB（``purpose=video_understanding``）。
MAX_VIDEO_FILES_BYTES = 512 * 1024 * 1024

#: 带视频时的**请求体**上限：64 MB（MiniMax 文档）。
MAX_VIDEO_BODY_BYTES = 64 * 1024 * 1024

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
    #: Files API 的**根地址**（``.../v1/files/upload`` 由客户端拼）。
    #:
    #: 为什么单独一个字段：模型端点是「调用方给完整 URL」（见模块
    #: 开头），而 Files 是**另一个**地址族 —— 从模型端点反推它
    #: 就是在猜路径（踩过一次 404 的坑）。**没有就等于这个供应方
    #: 不支持上传**，调用方要改走直传，不能瞎拼。
    files_base: str = ""


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
    """某个槽位的凭据。没有配就返回 None（**不猜、不编**）。

    **换目录不改进程环境变量** —— 识别链每个扫描任务一个线程，
    临时改 ``os.environ["MATCH_DATA_DIR"]`` 的话，两个任务一交错就串味：
    一个任务把变量还原了，另一个任务就读到了别的目录，
    表现出来是「模型接口说密钥不对」。
    所以用**线程局部**覆盖（见 :func:`app.collection.settings.use_data_dir`）。
    """
    from . import settings

    if data_dir:
        with settings.use_data_dir(data_dir):
            return settings.credentials_for(slot)
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


# ---------------------------------------------------------------- 视频块


def video_block(data: bytes, media_type: str = "video/mp4",
                fps: float = 1.0, detail: str = "default") -> Dict[str, Any]:
    """一段**视频**的内容块（base64 内联）。

    为什么要有它
    ------------
    抽帧给模型看，等于把一局**切碎**再问：模型每一段都只看见几帧静止
    的画面，看不出「这一下是不是打到了」「是不是交了位移」。
    直接给**视频**，模型能自己看连续的动作 —— 这才是「看懂对局」。

    形状（Anthropic 兼容）::

        {"type": "video",
         "source": {"type": "base64", "media_type": "video/mp4", "data": "..."},
         "fps": 1.0, "detail": "default"}

    ``fps`` 是**每秒取几帧**给模型看（0.2~5）。它不是视频本身的帧率，
    是模型看片的采样密度 —— 取太小会漏掉动作，取太大只是烧钱。
    """
    encoded = base64.b64encode(data).decode("ascii")
    return {
        "anthropic": {
            "type": "video",
            "source": {"type": "base64", "media_type": media_type,
                       "data": encoded},
            "fps": float(fps),
            "detail": detail,
        },
        "openai": {
            "type": "video_url",
            "video_url": {"url": f"data:{media_type};base64,{encoded}",
                          "detail": detail},
        },
    }


def video_ref_block(file_id: str, fps: float = 1.0,
                    detail: str = "default") -> Dict[str, Any]:
    """引用**已上传**的视频（``mm_file://{file_id}``）。

    大于 45 MB 的视频只能走这条路：直传有 50 MB 上限、请求体有 64 MB
    上限，而 Files API 能收 512 MB。传进来的是 ``file_id`` 还是已经
    带前缀的引用都行。
    """
    ref = str(file_id or "").strip()
    if not ref:
        raise VlmError("空的 file_id", kind="bad_request")
    if not ref.startswith("mm_file://"):
        ref = "mm_file://%s" % ref
    return {
        "anthropic": {
            "type": "video",
            "source": {"type": "url", "url": ref},
            "fps": float(fps),
            "detail": detail,
        },
        "openai": {
            "type": "video_url",
            "video_url": {"url": ref, "detail": detail},
        },
    }


def upload_file(path: str, api_key: str, base_url: str,
                purpose: str = "video_understanding",
                timeout: float = 900.0) -> str:
    """把文件传到 Files API，返回 ``file_id``。

    **上传的视频会留在对方的服务器上**（``video_understanding`` 用途
    最多留 7 天）—— 这是「把整局录像交出去」的代价，调用方要清楚。

    ``base_url`` 是**根地址**（如 ``https://api.minimaxi.com``），
    这里拼 ``/v1/files/upload``。空 = 这个供应方不支持上传，
    调用方要改走直传。
    """
    if not base_url:
        raise VlmError("这个供应方没有配 Files 上传地址 —— 只能直传小视频",
                       kind="bad_request")
    if not os.path.isfile(path):
        raise VlmError("要上传的文件不存在：%s" % path, kind="bad_request")
    try:
        import httpx
    except ImportError:                                # pragma: no cover
        raise VlmError("缺少 httpx，没法上传文件", kind="unknown")

    url = base_url.rstrip("/") + "/v1/files/upload"
    with open(path, "rb") as fh:
        files = {"file": (os.path.basename(path), fh, "video/mp4")}
        try:
            resp = httpx.post(
                url, headers={"Authorization": f"Bearer {api_key}"},
                data={"purpose": purpose}, files=files, timeout=timeout)
        except Exception as exc:                       # noqa: BLE001
            raise VlmError("上传文件失败：%s" % exc, kind="timeout",
                           detail=str(exc))
    if resp.status_code >= 400:
        raise VlmError("上传文件失败：HTTP %d" % resp.status_code,
                       kind=_classify(resp.status_code),
                       status=resp.status_code,
                       detail=resp.text[:500])
    try:
        payload = resp.json()
    except ValueError:
        raise VlmError("上传文件后返回的不是 JSON", kind="server",
                       detail=resp.text[:300])
    file_id = ""
    if isinstance(payload, dict):
        file_id = str(payload.get("file_id") or "")
        if not file_id and isinstance(payload.get("file"), dict):
            file_id = str(payload["file"].get("file_id") or "")
        base = payload.get("base_resp")
        if not file_id and isinstance(base, dict):
            file_id = str(base.get("file_id") or "")
    if not file_id:
        raise VlmError("上传成功但没拿到 file_id", kind="server",
                       detail=json.dumps(payload, ensure_ascii=False)[:300])
    return file_id



# ---------------------------------------------------------------- 请求


def _has_video(blocks: Sequence[Dict[str, Any]]) -> bool:
    """这些内容块里有没有**视频**。"""
    for b in blocks or ():
        for shape in ("anthropic", "openai"):
            part = b.get(shape)
            if isinstance(part, dict) and part.get("type") in ("video",
                                                                "video_url"):
                return True
    return False


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
) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    """造出 (url, headers, body)。拆出来是为了能单测。

    ``url`` 就是配置里的**完整端点**，这里一个字都不加 ——
    加路径是踩过坑的地方。
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
) -> VlmResult:
    """发一次请求。失败抛 :class:`VlmError`，并带上可判别的 ``kind``。"""
    if creds is None:
        creds = find_credentials(slot, data_dir)
    if creds is None:
        raise VlmError(f"没有配置「{slot}」这个槽位的模型", kind="auth")

    url, headers, body = build_request(
        model, blocks, creds, system=system, max_tokens=max_tokens,
        thinking=thinking)
    raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
    limit = MAX_BODY_BYTES
    if _has_video(blocks):
        # 带视频时，请求体上限是 **64 MB**（MiniMax 文档），
        # 比图片那条宽 —— 图片那条留的余量是给别的供应方的。
        limit = MAX_VIDEO_BODY_BYTES
    if len(raw) > limit:
        raise VlmError(
            f"请求体 {len(raw) // 1024 // 1024} MB 超过上限"
            f"（{limit // 1024 // 1024} MB）", kind="body")

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


#: 各类失败对应的人话。给报告和日志看 —— 错在哪要能一眼看出来。
_ERROR_TEXT = {
    "auth": "模型接口拒绝了这个密钥（HTTP 401/403），配置里的 API Key 无效或没权限",
    "rate_limit": "模型接口限流或套餐额度已用尽（HTTP 429）",
    "timeout": "模型接口没在超时时间内返回（网络或服务端排队）",
    "server": "模型服务端返回了错误（5xx 或响应不是 JSON）",
    "bad_request": "模型接口认为请求不合法（HTTP 400/404/413/422）",
    "body": "请求体超过模型接口的上限",
    "empty": "模型没有返回内容",
    "unknown": "调用失败，原因未知",
}


def explain_error(exc: VlmError) -> str:
    """把一次调用失败翻译成**能进报告**的一句话。

    为什么要有它
    ------------
    失败如果不带原因，报告就会变成一张**空表**，而空表最容易被
    误读成「打得没问题」。2026-10-31 实测：模型接口配额用尽
    （HTTP 429），所有窗都读不出来，报告却显示「0 条结论」，
    看上去像分析成功但玩家没毛病 —— 实际上是**根本没分析**。

    ``detail`` 里通常带着服务端给的原因（比如「Token Plan 用量上限」），
    那句才是用户真正需要看到的东西，所以优先带出来。
    """
    base = _ERROR_TEXT.get(getattr(exc, "kind", "unknown"), _ERROR_TEXT["unknown"])
    detail = (getattr(exc, "detail", "") or "").strip()
    # **「不收视频」有它自己的说法**（2026-11-13 实测）：整局看片要发
    # ``video`` / ``video_url`` 内容块，而 OpenAI 兼容的接口只认
    # ``text / image_url / file`` —— 服务端回的是
    # ``422 unknown variant `video_url` ``，翻成「请求参数不对」用户
    # 会去改提示词，白折腾。这里直接说清该换成谁。
    if getattr(exc, "kind", "") == "bad_request" and (
            "video_url" in detail or "unknown variant" in detail):
        return ("整局没看成：这个模型不收视频输入（整局看片要发 video 内容块）"
                "—— 把「观察者模型」换成 MiniMax M3")
    if detail:
        try:
            payload = json.loads(detail)
            msg = ((payload.get("error") or {}).get("message")
                   or payload.get("message") or "")
        except (ValueError, AttributeError):
            msg = ""
        if not msg:
            msg = detail[:160]
        return "%s：%s" % (base, msg)
    return base


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
