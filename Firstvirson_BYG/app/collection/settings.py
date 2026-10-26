"""应用级设置：模型怎么配、用哪套识别器。

设置放在**数据目录**的 ``config.yaml`` ——
打包成 exe 之后数据目录在 exe 旁边，用户自己看得见、改得动、备份得走；
环境变量可以覆盖（部署时更方便）。

配置的形状
----------
``model.vlm`` / ``model.text`` 两个槽位，各存一份：
``provider`` / ``api_key`` / ``model`` / ``endpoint`` / ``protocol`` /
``thinking_field`` / ``can_see_images``。

``can_see_images`` **不是猜的** —— 由「测试连接」带一张小图实测出来的
（DeepSeek 与百炼都有视觉型号，按供应方断定「能不能看图」会错）。

密钥**只写不回**：读接口只给掩码。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from . import model_presets

#: 两个槽位（看画面 / 对账措辞）。
SLOTS = model_presets.SLOTS

#: 识别器：``vlm``（默认，多模态模型）或 ``cv``（旧的几何探测器）。
RECOGNIZERS = ("vlm", "cv")


def data_dir() -> str:
    """数据目录。exe 与开发环境都指同一个地方。

    ``app/collection/xxx.py`` 往上退**两级**才是项目根 —— 少退一级
    会把配置写到 ``app/`` 下面去（那时 exe 打包后是只读目录）。
    """
    return os.environ.get("MATCH_DATA_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "data")


def config_path() -> str:
    return os.path.join(data_dir(), "config.yaml")


def _read_yaml(path: str) -> Optional[Dict[str, Any]]:
    try:
        import yaml
    except Exception:                              # noqa: BLE001
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except Exception:                              # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


def app_config() -> Dict[str, Any]:
    """读应用自己的 ``config.yaml``（没有就返回空表）。"""
    data = _read_yaml(config_path())
    return data if isinstance(data, dict) else {}


def _write_config(data: Dict[str, Any]) -> str:
    import yaml

    path = config_path()
    os.makedirs(data_dir(), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
    return path


# ---------------------------------------------------------------- 槽位


def slot_config(slot: str) -> Dict[str, Any]:
    """某个槽位**原始**的配置（含密钥，给内部用）。"""
    conf = app_config()
    models = conf.get("model") or {}
    raw = models.get(slot) or {}
    return dict(raw) if isinstance(raw, dict) else {}


def all_slot_configs() -> Dict[str, Dict[str, Any]]:
    return {slot: slot_config(slot) for slot in SLOTS}


def display_slots() -> Dict[str, Dict[str, Any]]:
    """给界面看的槽位状态（**密钥已掩码**）。"""
    out: Dict[str, Dict[str, Any]] = {}
    for slot in SLOTS:
        raw = slot_config(slot)
        provider = str(raw.get("provider") or "")
        resolved = model_presets.resolve(provider, str(raw.get("model") or ""), slot)
        out[slot] = {
            "slot": slot,
            "label": model_presets.SLOT_LABELS.get(slot, slot),
            "hint": model_presets.SLOT_HINTS.get(slot, ""),
            "provider": provider,
            "provider_label": resolved["label"],
            "model": str(raw.get("model") or resolved["model"]),
            "endpoint": str(raw.get("endpoint") or resolved["endpoint"]),
            "protocol": str(raw.get("protocol") or resolved["protocol"]),
            "has_key": bool(str(raw.get("api_key") or "")),
            "key_masked": model_presets.mask_key(str(raw.get("api_key") or "")),
            "can_see_images": bool(raw.get("can_see_images")),
            "tested_at": str(raw.get("tested_at") or ""),
            "test_note": str(raw.get("test_note") or ""),
        }
    return out


def save_slot(slot: str, provider: str, api_key: str = "", model: str = "",
              endpoint: str = "", protocol: str = "",
              can_see_images: Optional[bool] = None,
              test_note: str = "", tested_at: str = "") -> Dict[str, Any]:
    """保存一个槽位。

    - ``api_key`` 留空 = **不改动**已存的密钥（界面不回显明文，只能这样改）；
    - ``can_see_images`` 只有在真的测过之后才写（``None`` = 不动）。
    """
    if slot not in SLOTS:
        raise ValueError(f"没有这个槽位：{slot}")
    resolved = model_presets.resolve(provider, model, slot)
    if provider == "custom":
        if not endpoint.strip():
            raise ValueError("自定义供应方要填完整的请求地址")
        resolved["endpoint"] = endpoint.strip()
    if not resolved["model"]:
        raise ValueError("模型名不能为空（从控制台复制完整的模型 ID）")

    conf = app_config()
    models = dict(conf.get("model") or {})
    raw = dict(models.get(slot) or {})
    raw.update({
        "provider": resolved["provider"],
        "model": resolved["model"],
        "endpoint": resolved["endpoint"],
        "protocol": protocol.strip() or resolved["protocol"],
        "thinking_field": resolved["thinking_field"],
    })
    key = (api_key or "").strip()
    if key:
        raw["api_key"] = key
    if can_see_images is not None:
        raw["can_see_images"] = bool(can_see_images)
    if test_note:
        raw["test_note"] = test_note
    if tested_at:
        raw["tested_at"] = tested_at
    models[slot] = raw
    conf["model"] = models
    _write_config(conf)
    return display_slots()[slot]


def env_override(slot: str) -> Dict[str, str]:
    """环境变量覆盖（部署时更方便）。

    认这三个：

    - ``MATCH_<槽位>_API_KEY``   例：``MATCH_VLM_API_KEY``
    - ``MATCH_<槽位>_ENDPOINT``  完整请求地址
    - ``MATCH_<槽位>_MODEL``     模型 ID

    环境变量**盖过配置文件** —— 部署时改一个环境变量就换了模型，
    不用动磁盘上的文件。
    """
    prefix = f"MATCH_{slot.upper()}"
    return {
        "api_key": os.environ.get(f"{prefix}_API_KEY", "").strip(),
        "endpoint": os.environ.get(f"{prefix}_ENDPOINT", "").strip(),
        "model": os.environ.get(f"{prefix}_MODEL", "").strip(),
    }


def credentials_for(slot: str):
    """某个槽位的调用凭据。没有配就返回 None（**不猜、不编**）。"""
    from . import vlm_client

    raw = slot_config(slot)
    env = env_override(slot)
    key = env["api_key"] or str(raw.get("api_key") or "").strip()
    if not key:
        return None
    provider = str(raw.get("provider") or "")
    resolved = model_presets.resolve(provider, str(raw.get("model") or ""), slot)
    endpoint = (env["endpoint"] or str(raw.get("endpoint") or "")
                or resolved["endpoint"]).strip()
    if not endpoint:
        return None
    model = env["model"] or str(raw.get("model") or "") or resolved["model"]
    return vlm_client.Credentials(
        api_key=key,
        endpoint=endpoint,
        protocol=str(raw.get("protocol") or resolved["protocol"]),
        thinking_field=str(raw.get("thinking_field") or resolved["thinking_field"]),
        model=model,
        source=f"config:{slot}:{provider}",
    )


def model_name(slot: str = "vlm") -> str:
    """某个槽位配的模型名（给报告标注来源用）。"""
    env = env_override(slot)
    if env["model"]:
        return env["model"]
    raw = slot_config(slot)
    return str(raw.get("model") or "")


# ---------------------------------------------------------------- 其它设置


def recognizer_kind() -> str:
    """用哪套识别器：``vlm``（默认）或 ``cv``。"""
    env = os.environ.get("MATCH_RECOGNIZER", "").strip().lower()
    if env in RECOGNIZERS:
        return env
    kind = str(app_config().get("recognizer") or "vlm").strip().lower()
    return kind if kind in RECOGNIZERS else "vlm"


def set_recognizer(kind: str) -> str:
    """切识别器。写回 ``config.yaml``。"""
    kind = (kind or "").strip().lower()
    if kind not in RECOGNIZERS:
        raise ValueError("识别器只能是 vlm 或 cv")
    conf = app_config()
    conf["recognizer"] = kind
    return _write_config(conf)


def limits() -> Dict[str, Any]:
    """调用限额（写进配置，页面上可改）。"""
    conf = app_config()
    raw = conf.get("limits") or {}
    if not isinstance(raw, dict):
        raw = {}
    return {
        "timeout": float(raw.get("timeout") or 300),
        "concurrency": max(1, min(8, int(raw.get("concurrency") or 4))),
        "max_calls": max(1, int(raw.get("max_calls") or 20)),
    }


def set_limits(**kw: Any) -> Dict[str, Any]:
    conf = app_config()
    raw = dict(conf.get("limits") or {})
    for key in ("timeout", "concurrency", "max_calls"):
        if kw.get(key) is not None:
            raw[key] = kw[key]
    conf["limits"] = raw
    _write_config(conf)
    return limits()


def status() -> Dict[str, Any]:
    """给页面上看的一眼状态。"""
    slots = display_slots()
    vlm = slots.get("vlm") or {}
    return {
        "recognizer": recognizer_kind(),
        "has_key": bool(vlm.get("has_key")),
        "can_see_images": bool(vlm.get("can_see_images")),
        "model": vlm.get("model") or "",
        "provider": vlm.get("provider_label") or "",
        "slots": slots,
        "limits": limits(),
        "data_dir": data_dir(),
    }
