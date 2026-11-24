"""大模型供应方预设。

为什么要有这一层
----------------
用户只该填一个 **API Key**。基址、协议、「关思考」那个字段叫什么、
默认模型名 —— 这些都是**供应方的事**，让用户填就会填错，
而且填错了报错信息往往看不出是哪错了（少写一段路径 → 404；
字段名不对 → 参数被忽略、模型开始长篇大论）。

所以每个供应方把**完整端点 URL** 备好，客户端**不猜路径**。
（我们已经因为自己拼路径踩过一次：少写 ``/v1`` → 404。）

参数来源（2026-10-24 逐条核实）
------------------------------
- DeepSeek：``api-docs.deepseek.com/zh-cn/`` —— 官方 base **不带** ``/v1``；
  默认模型 ``deepseek-flash``（= DeepSeek-V4.1-Flash-1017）；
  思考开关是 ``thinking: {type: ...}``。
- 阿里云百炼：``help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope``
  —— 北京地域 ``.../compatible-mode/v1/chat/completions``；
  **key 按地域绑定**（跨地域 401 ``invalid_api_key``）；
  思考开关是 ``enable_thinking``（**字段名与别家不一样**）。
- MiniMax：``api.minimaxi.com/anthropic/v1/messages``（已实测能读王者画面）。

模型名**不做成下拉清单**：官方文档明确要求「复制控制台展示的准确模型 ID」，
而清单天天在变（DeepSeek 刚把 ``deepseek-v4-flash`` 下线成 ``deepseek-flash``；
百炼已经出现 ``qwen3.8-max`` 这种新命名）。所以是「自由输入 + 默认值」。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

#: 供应方预设。``endpoint`` 是**完整请求地址**，客户端不再拼路径。
PROVIDERS: Dict[str, Dict[str, Any]] = {
    "qwen": {
        "label": "Qwen（阿里云百炼）",
        "protocol": "openai",
        "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        # 业务空间专属域名（官方建议迁移，稳定性与性能更好）。
        "endpoint_note": ("业务空间专属域名（北京）："
                          "https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com"
                          "/compatible-mode/v1/chat/completions"),
        "key_url": "https://bailian.console.aliyun.com/model/settings/api-key",
        "key_prefix": "sk-",
        "default_model": "qwen-vl-max",
        "default_text_model": "qwen-plus",
        "thinking_field": "enable_thinking",
        "note": "key 与地域绑定；百炼也直供 DeepSeek / MiniMax（仅北京地域，需先开通）。",
    },
    "deepseek": {
        "label": "DeepSeek",
        "protocol": "openai",
        "endpoint": "https://api.deepseek.com/chat/completions",
        "endpoint_note": "",
        "key_url": "https://platform.deepseek.com/api_keys",
        "key_prefix": "sk-",
        "default_model": "deepseek-flash",
        "default_text_model": "deepseek-flash",
        "thinking_field": "thinking",
        "note": "官方 base 不带 /v1；模型名以控制台为准。",
    },
    "minimax": {
        "label": "MiniMax",
        "protocol": "anthropic",
        "endpoint": "https://api.minimaxi.com/anthropic/v1/messages",
        "endpoint_note": "",
        "key_url": "https://platform.minimaxi.com",
        "key_prefix": "",
        "default_model": "MiniMax-M3",
        "default_text_model": "MiniMax-M3",
        "thinking_field": "thinking",
        # Files API 的根地址 —— 传整局视频时用（>45 MB 只能走这条路）。
        "files_base": "https://api.minimaxi.com",
        "note": "已实测能读王者画面。",
    },
    "custom": {
        "label": "自定义",
        "protocol": "openai",
        "endpoint": "",
        "endpoint_note": "",
        "key_url": "",
        "key_prefix": "",
        "default_model": "",
        "default_text_model": "",
        "thinking_field": "thinking",
        "note": "填完整的请求地址（只填域名会 404）。",
    },
}

#: 型号名里出现这些片段 → **大概率能看画面**。
#:
#: 用途有两个：部署时只给真能看图的型号注入密钥；
#: 「测试连接」时给型号名一个先验。
#:
#: 判断不了的一律当成「不能」—— 猜错了的代价是「报告是空的」，
#: 而空报告会被当成结论（"系统说没问题"）。
#:
#: 2026-10-29 实测更正：``deepseek-flash`` 原来被归在纯文本名单里，
#: 实际它**能看图** —— ``GET /models`` 报 ``input_modalities=['text','image']``，
#: 且探针图（深蓝底橙色圆）它答「橙色，圆形」，答对了。
#: 不改的后果很具体：部署脚本会因为「读不了画面」**跳过整个 VLM 槽位**，
#: 用户配的 DeepSeek key 根本不会上云。
VISION_CAPABLE_TAGS = (
    "qwen-vl", "qwen3-vl", "qvq", "minimax-m3", "minimax-m2",
    "gpt-4o", "gpt-4.1", "glm-4v", "doubao-vision", "hunyuan-vision",
    "step-2v", "internvl", "kimi-vl", "deepseek-vl",
    "deepseek-flash", "deepseek-chat", "deepseek-reasoner",
)

#: 型号名里出现这些片段 → **确定是纯文本**（不收图片）。
#:
#: 这些是「当场否掉」的名单：它们不收图片，给它们发探针图，
#: 有些接口直接 400，有些接口默默忽略图片然后**猜一个答案** ——
#: 后者最坏：纯文本模型冒充能看画面，报告整份是空的。
#:
#: 2026-10-29 实测更正：``deepseek-flash`` / ``deepseek-chat`` /
#: ``deepseek-reasoner`` 原来都在这份名单里，实测**三个都能看图**
#: （``/models`` 报 image 输入，探针图答对颜色形状），已移出。
#: 同批实测里 ``deepseek-v4-pro`` 的 ``input_modalities`` 只有 ``['text']``，
#: 发图过去它**默默忽略图片**、正文直接空 —— 这才是真正的纯文本型号，
#: 它由下面的 ``deepseek-v4`` 片段覆盖。
TEXT_ONLY_TAGS = (
    "deepseek-coder",
    "deepseek-v3", "deepseek-v4",
    "qwen-plus", "qwen-turbo", "qwen-max",
    "gpt-3.5", "gpt-4o-mini", "text-embedding",
)


def vision_hints(model: str) -> Optional[bool]:
    """按型号名判断能不能看画面。**判断不了返回 None**（别硬猜）。

    - ``True``  ：型号名就是视觉型号；
    - ``False`` ：型号名确定是纯文本；
    - ``None``  ：看不出来 —— 交给探针图实测。

    纯文本名单**先查**：``gpt-4o-mini`` 里含有 ``gpt-4o``，
    先查视觉名单会把它误判成能看图。
    """
    name = (model or "").strip().lower()
    if not name:
        return None
    if any(tag in name for tag in TEXT_ONLY_TAGS):
        return False
    if any(tag in name for tag in VISION_CAPABLE_TAGS):
        return True
    return None


#: 两个槽位：看画面 / 对账措辞。
SLOTS = ("vlm", "text")

#: 能看画面的型号怎么换（401 / 400 那两句提示里用）。
#:
#: 必须带上 ``deepseek-flash``：这一份代码里实测过它能看图
#: （``/models`` 报 image 输入，探针图答对），而 DeepSeek 自己的
#: 型号名又最容易被写错（``deepseek-v4`` 系列实测只有 text 输入）。
VISION_MODEL_HINTS = "qwen-vl-max / MiniMax-M3 / deepseek-flash"

#: UUID 形状（8-4-4-4-12）。**任何一家的 API Key 都不是这个形状** ——
#: 它更像 DeepSeek 返回的 ``request_id``，或者别处的会话 id。
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE)


def provider_of_endpoint(endpoint: str) -> str:
    """这个请求地址是哪一家的（认不出返回空串）。

    用途：用户选了「自定义」、自己填了 DeepSeek 的地址时，
    页面仍要能说出「你这是 DeepSeek 的地址，Key 得是他们家的」——
    按供应方 id 判断做不到（id 就是 ``custom``），只能按域名反查。
    """
    try:
        from urllib.parse import urlparse

        host = (urlparse(str(endpoint or "").strip()).netloc or "").lower()
    except Exception:                                   # noqa: BLE001
        return ""
    if not host:
        return ""
    for pid, spec in PROVIDERS.items():
        preset = str(spec.get("endpoint") or "").strip()
        if not preset:
            continue
        try:
            from urllib.parse import urlparse as _up

            if (_up(preset).netloc or "").lower() == host:
                return pid
        except Exception:                               # noqa: BLE001
            continue
    return ""


def video_support(provider: str = "", endpoint: str = "") -> Optional[bool]:
    """这个供应方收不收**视频块**（整局录像直接给模型看）。

    ``True`` 收 / ``False`` 确定不收 / ``None`` 认不出来（那就照老路试一次）。

    为什么必须问一声（2026-11-13 实测）
    ----------------------------------
    这一套识别链是「**把整局录像打包后直接给模型看**」，
    发的内容块是 ``video``（Anthropic 形状）或 ``video_url``（OpenAI 形状）——
    而 **OpenAI 兼容的接口根本没有这两种块**。实测拿 DeepSeek 的
    ``deepseek-flash`` 发 ``video_url``，返回：

        422 unknown variant `video_url`, expected one of
        `text`, `image_url`, `file`

    也就是说：**只有 MiniMax M3 收整局视频**（它的 Files 上传也是为这个准备的）。
    不问这一声的后果很具体：整局照打包装、照发请求、报 422 之后
    一条事件都没有 → 报告是空的，而报错被别的一条留痕盖住，
    用户看到的是「这一局没有分析出任何内容：视频重编码过……」——
    完全指不到该改哪里。

    **认不出来就返回 None**：用户完全可能把 MiniMax 的兼容地址填进
    「自定义」，那时不该拦他 —— 试一次比误判强。
    """
    pid = str(provider or "").strip().lower()
    if pid and pid != "custom":
        return pid == "minimax"
    found = provider_of_endpoint(endpoint)
    if found:
        return found == "minimax"
    return None


def key_shape_hint(key: str, endpoint: str = "") -> str:
    """这把 Key 的**形状**和这个地址对得上吗？对不上就给一句能照着改的话。

    为什么值得单独判一次（2026-11-13 实测）
    --------------------------------------
    真实的一例：页面上「测试连接」报 ``HTTP 401``，提示写着
    「Key 不对，或不是这家供应方的 Key」—— 用户不知道该改哪一边。
    实际存下来的 Key 是 ``abd8693f-3313-4852-924e-42ffb06fecbe``：
    **一串 UUID**。而 DeepSeek / 百炼的 Key 一律以 ``sk-`` 开头，
    MiniMax 的也不是这个形状 —— 也就是说**它不是任何一家的 Key**，
    多半是把别处的 id（或某次报错里的 ``request_id``）复制进来了。

    ``sk-`` 这种前缀判据是**各家的公开约定**，不是猜：
    判不准的（比如 MiniMax）就返回空串，一句话都不多说。

    返回空串 = 看不出问题（**不代表 Key 一定对**，那要真调一次才知道）。
    """
    text = str(key or "").strip()
    if not text:
        return ""
    provider = provider_of_endpoint(endpoint)
    spec = PROVIDERS.get(provider) or {}
    prefix = str(spec.get("key_prefix") or "")
    label = str(spec.get("label") or "")
    if _UUID_RE.match(text):
        if label:
            tail = ("（他们的 Key 以「%s」开头）" % prefix) if prefix else ""
            return ("这串是 UUID 形状，不像 %s 的 API Key%s—— "
                    "多半是把别处的 id 复制进来了" % (label, tail))
        return ("这串是 UUID 形状，不像任何一家的 API Key —— "
                "多半是把别处的 id 复制进来了")
    if prefix and not text.startswith(prefix):
        return ("%s 的 Key 以「%s」开头，这串不是 —— 可能拿的是别家的 Key"
                % (label or provider, prefix))
    return ""


#: 槽位的说明（页面上就这一行，不写长文案）。
SLOT_LABELS = {
    "vlm": "观察者模型",
    "text": "分析者模型（可选）",
}

#: 槽位说明里那句「没有会怎样」——让人知道不填的后果，但不啰嗦。
SLOT_HINTS = {
    "vlm": "没有它，这一局就没有自动结论。",
    "text": "没有它就用规则库里那套条件匹配，照旧能出报告。",
}


def get(provider: str) -> Dict[str, Any]:
    """取一个供应方的预设。认不出来就回落到「自定义」。"""
    return PROVIDERS.get(provider) or PROVIDERS["custom"]


def resolve(provider: str, model: str = "", slot: str = "vlm") -> Dict[str, Any]:
    """把「供应方 + 模型名」解析成一份完整配置。

    - 供应方给了默认模型就用它；
    - 模型名留空 → 用该槽位的默认（看画面与对账措辞的默认可能不同）；
    - 「自定义」没有默认模型 → 必须用户自己填。
    """
    spec = get(provider)
    name = (model or "").strip()
    if not name:
        key = "default_text_model" if slot == "text" else "default_model"
        name = str(spec.get(key) or "")
    return {
        "provider": provider if provider in PROVIDERS else "custom",
        "label": spec["label"],
        "protocol": spec["protocol"],
        "endpoint": str(spec.get("endpoint") or ""),
        "thinking_field": str(spec.get("thinking_field") or "thinking"),
        "files_base": str(spec.get("files_base") or ""),
        "model": name,
        "key_prefix": str(spec.get("key_prefix") or ""),
        "key_url": str(spec.get("key_url") or ""),
        "note": str(spec.get("note") or ""),
    }


def catalog() -> List[Dict[str, Any]]:
    """给界面用的供应方清单（不含任何密钥）。"""
    out: List[Dict[str, Any]] = []
    for pid, spec in PROVIDERS.items():
        out.append({
            "id": pid,
            "label": spec["label"],
            "protocol": spec["protocol"],
            "endpoint": spec.get("endpoint") or "",
            "endpoint_note": spec.get("endpoint_note") or "",
            "key_url": spec.get("key_url") or "",
            "key_prefix": spec.get("key_prefix") or "",
            "default_model": spec.get("default_model") or "",
            "default_text_model": spec.get("default_text_model") or "",
            "thinking_field": spec.get("thinking_field") or "thinking",
            "note": spec.get("note") or "",
        })
    return out


def mask_key(key: str) -> str:
    """把密钥掩成 ``sk-1***cdef``。读接口只给这个。

    密钥**只写不回** —— 回显等于把密钥发到浏览器里。
    """
    k = (key or "").strip()
    if not k:
        return ""
    if len(k) <= 8:
        return "***"
    return f"{k[:4]}***{k[-4:]}"
