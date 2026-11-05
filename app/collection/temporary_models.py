"""匿名公开版的临时模型凭据。

Key 仅放在本进程内存，并按浏览器设备 ID 分隔：不落盘、不进数据库、
不写全局 config.yaml。服务重启后自动清空，用户需重新导入。
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from . import model_presets
from .vlm_client import Credentials


class TemporaryModels:
    def __init__(self) -> None:
        self._items: Dict[str, Dict[str, Credentials]] = {}
        self._lock = threading.Lock()

    def get(self, device_id: str, slot: str = "vlm") -> Optional[Credentials]:
        with self._lock:
            return self._items.get(device_id, {}).get(slot)

    def display(self, device_id: str) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            mine = dict(self._items.get(device_id, {}))
        out: Dict[str, Dict[str, Any]] = {}
        for slot in model_presets.SLOTS:
            creds = mine.get(slot)
            out[slot] = {
                "slot": slot,
                "provider": (creds.source.split(":")[-1] if creds else "qwen"),
                "model": creds.model if creds else "",
                "endpoint": creds.endpoint if creds else "",
                "has_key": bool(creds),
                "key_masked": "已导入（仅本次服务运行）" if creds else "",
                "can_see_images": slot != "vlm" or bool(creds),
            }
        return out

    def save(self, device_id: str, slot: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if slot not in model_presets.SLOTS:
            raise ValueError(f"没有这个槽位：{slot}")
        key = str(payload.get("api_key") or "").strip()
        if not key:
            raise ValueError("请填入你自己的 API Key")
        provider = str(payload.get("provider") or "").strip()
        resolved = model_presets.resolve(provider, str(payload.get("model") or ""), slot)
        endpoint = str(payload.get("endpoint") or "").strip()
        if resolved["provider"] == "custom":
            if not endpoint:
                raise ValueError("自定义供应方要填完整的请求地址")
            resolved["endpoint"] = model_presets.normalize_endpoint(endpoint)
        if not resolved["model"] or not resolved["endpoint"]:
            raise ValueError("模型名和请求地址不能为空")
        creds = Credentials(
            api_key=key, endpoint=resolved["endpoint"],
            protocol=str(payload.get("protocol") or resolved["protocol"]),
            thinking_field=resolved["thinking_field"], model=resolved["model"],
            source=f"temporary:{resolved['provider']}",
        )
        with self._lock:
            self._items.setdefault(device_id, {})[slot] = creds
        return self.display(device_id)[slot]


STORE = TemporaryModels()
