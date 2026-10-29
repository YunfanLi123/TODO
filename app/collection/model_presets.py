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
        "note": "已实测能读王者画面。",
    },
    "ark": {
        "label": "火山方舟（豆包）",
        "protocol": "openai",
        # 方舟控制台常展示的是 /api/v3 基址；这里必须存完整的 Chat
        # 地址，客户端按原样 POST，不能期待它再替我们补路径。
        "endpoint": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        "endpoint_note": "方舟基础地址会自动补为 /chat/completions。",
        "key_url": "https://console.volcengine.com/ark/region:cn-beijing/api-key",
        "key_prefix": "",
        "default_model": "doubao-seed-2-1-pro-260915",
        "default_text_model": "doubao-seed-2-1-pro-260915",
        "thinking_field": "thinking",
        "note": "选择视觉模型；完整请求地址为 /api/v3/chat/completions。",
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


ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


def normalize_endpoint(endpoint: str) -> str:
    """把方舟常见的基础地址补成此客户端需要的完整 Chat 地址。

    其他供应商的地址一律原样保留。客户端使用 OpenAI Chat Completions
    请求体，因此这里不能把方舟基址指向 ``/responses``。
    """
    value = (endpoint or "").strip()
    if value.rstrip("/") == ARK_BASE_URL:
        return ARK_BASE_URL + "/chat/completions"
    return value

#: 两个槽位：看画面 / 对账措辞。
SLOTS = ("vlm", "text")

#: 槽位的说明（页面上就这一行，不写长文案）。
SLOT_LABELS = {
    "vlm": "看画面的模型",
    "text": "对账措辞的模型（可选）",
}

#: 槽位说明里那句「没有会怎样」——让人知道不填的后果，但不啰嗦。
SLOT_HINTS = {
    "vlm": "没有它，这一局就没有自动结论。",
    "text": "没有它就用规则库里那套条件匹配，照旧能出报告。",
}


def provider_ids() -> List[str]:
    return list(PROVIDERS.keys())


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
