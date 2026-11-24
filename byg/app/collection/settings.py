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

import contextvars
import os
import threading
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from . import model_presets
from ..core import device as device_mod

#: 两个槽位（看画面 / 对账措辞）。
SLOTS = model_presets.SLOTS


#: 设备配置的**基准目录**，在请求开始时定下来（见 :func:`set_device`）。
#:
#: 为什么单独记一份：识别链会用 :func:`use_data_dir` 临时换目录，
#: 那个目录**不一定**是基准目录。配置路径要是跟着它走，
#: Key 就可能存到 A 处、读从 B 处 —— 表现是「报告是空的」，
#: 而真正的原因是路径拼错了，跟密钥本身没关系。
_DEVICE_BASE: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "ma_device_base", default=None)


#: **线程局部**的数据目录覆盖（见 :func:`use_data_dir`）。
#:
#: 为什么不用进程环境变量：识别链每个扫描任务一个线程，
#: 临时换目录靠改 ``os.environ`` 的话，两个任务一交错就串味 ——
#: 一个任务把变量还原了，另一个任务就读到了别的目录，
#: 表现出来是「模型接口说密钥不对」，而真正的原因是目录拿错了。
_local = threading.local()


@contextmanager
def use_data_dir(path: Optional[str]):
    """在这个线程里临时换一个数据目录（**不碰进程环境变量**）。

    目录是空的（或没给）就什么都不做 —— 走默认那套。
    """
    if not path:
        yield
        return
    old = getattr(_local, "data_dir", None)
    _local.data_dir = path
    try:
        yield
    finally:
        if old is None:
            try:
                del _local.data_dir
            except AttributeError:
                pass
        else:
            _local.data_dir = old


def data_dir() -> str:
    """数据目录。exe 与开发环境都指同一个地方。

    ``app/collection/xxx.py`` 往上退**两级**才是项目根 —— 少退一级
    会把配置写到 ``app/`` 下面去（那时 exe 打包后是只读目录）。

    **线程局部覆盖优先**（见 :func:`use_data_dir`）——
    凭据与型号必须来自同一份配置，不能一个来自这里、一个来自环境变量。
    """
    override = getattr(_local, "data_dir", None)
    if override:
        return override
    return os.environ.get("MATCH_DATA_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "data")


def set_device(device: Optional[str], base_dir: Optional[str] = None):
    """把当前请求的设备标识设进来。

    返回 token，交给 :func:`reset_device`（中间件用）。
    ``base_dir`` 是这次请求的**基准数据目录** —— 记下来，
    免得识别链临时换目录时把配置路径拼到别处去。
    """
    token = device_mod.set_current_device(device)
    _DEVICE_BASE.set(base_dir or data_dir())
    return token


def reset_device(token: Any = None) -> None:
    """还原上一个设备标识。"""
    device_mod.reset_current_device(token)
    _DEVICE_BASE.set(None)


def current_device() -> Optional[str]:
    """当前请求属于哪个设备（没设 = 本地单人使用）。"""
    return device_mod.current_device()


def device_config_dir(device: str, base_dir: Optional[str] = None) -> str:
    """某个设备的配置目录。"""
    return os.path.join(base_dir or data_dir(), "devices", device)


def config_path() -> str:
    """配置文件路径。

    **按设备隔离**：隔离开着时（多人共用 / 即焚模式），各人的 API Key
    存在各自的目录里 —— 共用一份的话，A 填的 Key B 看得见、用得上。
    本地单人使用（隔离没开）时还是原来那个 ``config.yaml``，行为一个字不变。
    """
    device = current_device()
    if device:
        base = _DEVICE_BASE.get() or data_dir()
        return os.path.join(base, "devices", device, "config.yaml")
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
    # 建**配置所在的目录**（不是数据目录）—— 按设备隔离时配置在
    # ``devices/<设备>/`` 下面，那个目录还不存在。
    os.makedirs(os.path.dirname(path) or data_dir(), exist_ok=True)
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


def effective_slot(slot: str) -> Dict[str, Any]:
    """某个槽位**实际生效**的配置（读路径上的修正都在这里做）。

    一把 Key 只对一个供应方有效。两个槽位拿着**同一把** Key 却配着不同供应方，
    其中一个必然 401 —— 那个必然 401 的就是文字槽位
    （看画面的槽位是主槽位，它说了算）。

    这不是「替用户做决定」：同一把 Key 分给两个供应方，本来就只有一个能通。
    旧版本里文字槽位的供应方默认是百炼，用户没改直接保存，
    于是就攒下了这种「拿 DeepSeek 的 Key 去请求百炼」的组合。
    """
    raw = slot_config(slot)
    if slot != "text":
        return raw
    vlm = slot_config("vlm")
    same_key = (raw.get("api_key") and vlm.get("api_key")
                and raw.get("api_key") == vlm.get("api_key"))
    if not same_key:
        return raw
    if not (raw.get("provider") and vlm.get("provider")):
        return raw
    if raw.get("provider") == vlm.get("provider"):
        return raw
    fixed = dict(raw)
    for field in ("provider", "model", "endpoint", "protocol", "thinking_field"):
        if vlm.get(field):
            fixed[field] = vlm[field]
    return fixed


def display_slots() -> Dict[str, Dict[str, Any]]:
    """给界面看的槽位状态（**密钥已掩码**）。

    **必须把环境变量算进去**：云端的 ``/data`` 是临时盘，``config.yaml``
    存不住，密钥只能走环境变量注入。而调用侧（:func:`credentials_for`）
    本来就认环境变量 —— 如果这里只读配置文件，页面会显示「没有密钥」，
    而实际调用是通的。用户看着页面会以为没配好，反复去填一遍，
    却发现报告还是空的（因为环境变量把填的值盖掉了）。

    同一个道理，``provider`` 也要看环境变量：只有环境变量时，
    配置文件里没有供应方名，得靠模型名反推，否则显示成「未选择」。
    """
    out: Dict[str, Dict[str, Any]] = {}
    for slot in SLOTS:
        raw = effective_slot(slot)
        env = env_override(slot)
        inherited = _borrow_profile(slot)
        provider = (env["provider"] or str(raw.get("provider") or "")
                    or inherited.get("provider", ""))
        model_name = (env["model"] or str(raw.get("model") or "")
                      or inherited.get("model", ""))
        resolved = model_presets.resolve(provider, model_name, slot)
        # 密钥：环境变量优先（部署场景），配置文件次之（本地 exe）。
        # 自己的没有就借另一个槽位的（供应方相同才行）。
        key = slot_key(slot)
        # 能不能看画面：环境变量注入的那份模型说了算；
        # 没有环境变量时，页面「测试连接」实测出来的结果算数。
        #
        # **测过与否要分得开**：没测过是「不知道」（None），
        # 测过读不了才是 False。把没测过说成「读不了」，
        # 页面就会对刚配好的模型报一句假警。
        if env["api_key"]:
            can_see = (_flag(env["can_see_images"])
                       if env["can_see_images"] else None)
        else:
            raw_flag = raw.get("can_see_images")
            can_see = None if raw_flag is None else bool(raw_flag)
        out[slot] = {
            "slot": slot,
            "label": model_presets.SLOT_LABELS.get(slot, slot),
            "hint": model_presets.SLOT_HINTS.get(slot, ""),
            "provider": provider,
            "provider_label": resolved["label"],
            "model": model_name or resolved["model"],
            "endpoint": env["endpoint"] or str(raw.get("endpoint") or "")
            or inherited.get("endpoint", "") or resolved["endpoint"],
            "protocol": env["protocol"] or str(raw.get("protocol") or "")
            or inherited.get("protocol", "") or resolved["protocol"],
            "has_key": bool(key),
            "key_masked": model_presets.mask_key(key),
            "can_see_images": can_see,
            "tested_at": str(raw.get("tested_at") or ""),
            "test_note": str(raw.get("test_note") or ""),
            #: 密钥来自环境变量（部署预置）还是配置文件（页面保存的）。
            #: 页面据此提示「这个 Key 改环境变量才生效」，免得白填。
            "key_source": "env" if env["api_key"] else (
                "config" if raw.get("api_key") else ""),
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

    认这些：

    - ``MATCH_<槽位>_API_KEY``   例：``MATCH_VLM_API_KEY``
    - ``MATCH_<槽位>_ENDPOINT``  完整请求地址
    - ``MATCH_<槽位>_MODEL``     模型 ID
    - ``MATCH_<槽位>_PROVIDER``  供应方 id（qwen / deepseek / minimax / custom）
    - ``MATCH_<槽位>_PROTOCOL``  协议（openai / anthropic）
    - ``MATCH_<槽位>_CAN_SEE_IMAGES``  能不能看画面（1/0）

    环境变量**盖过配置文件** —— 部署时改一个环境变量就换了模型，
    不用动磁盘上的文件。

    但**只给「真能看画面」的模型注入 VLM 槽位**：
    把纯文本模型（deepseek）塞进去，看着像配好了，实际一帧画面都读不出来，
    报告是空的；而用户自己粘一个能看图的 Key 又会被环境变量盖掉 —— 页面成了摆设。
    """
    prefix = f"MATCH_{slot.upper()}"
    return {
        "api_key": os.environ.get(f"{prefix}_API_KEY", "").strip(),
        "endpoint": os.environ.get(f"{prefix}_ENDPOINT", "").strip(),
        "model": os.environ.get(f"{prefix}_MODEL", "").strip(),
        "provider": os.environ.get(f"{prefix}_PROVIDER", "").strip(),
        "protocol": os.environ.get(f"{prefix}_PROTOCOL", "").strip(),
        "can_see_images": os.environ.get(f"{prefix}_CAN_SEE_IMAGES", "").strip(),
    }


def _flag(text: str) -> bool:
    """把环境变量里的开关读成布尔（认 1/true/yes/on）。"""
    return str(text).strip().lower() in ("1", "true", "yes", "on")


def slot_key(slot: str, provider: str = "") -> str:
    """某个槽位该用的密钥：**自己的优先，没有就借另一个槽位的**。

    借的条件只有一个 —— **供应方相同**。
    两个槽位通常用同一把 Key，让人贴两遍纯属折磨；
    但供应方不同就不能借：拿 DeepSeek 的 Key 去请求百炼的地址 = 401，
    而那个 401 看不出是「借错了」。

    ``provider`` 可以覆盖「这个槽位当前算哪个供应方」——
    页面上用户刚把供应方下拉框换了，密钥的归属要按新的那个算。
    """
    raw = effective_slot(slot)
    key = env_override(slot)["api_key"] or str(raw.get("api_key") or "")
    if key:
        return key
    other = "text" if slot == "vlm" else "vlm"
    other_raw = effective_slot(other)
    if not other_raw.get("api_key"):
        return ""
    mine = provider or env_override(slot)["provider"] or str(raw.get("provider") or "")
    theirs = (env_override(other)["provider"]
              or str(other_raw.get("provider") or ""))
    if mine and theirs and mine != theirs:
        return ""
    return str(other_raw.get("api_key") or "")


def _borrow_profile(slot: str) -> Dict[str, str]:
    """文字槽位没配过时，沿用看画面那个槽位的供应方与型号。

    两个槽位通常是一套配置，让用户把供应方、型号再选一遍纯属折磨。
    只有文字槽位沿用 —— 看画面那个槽位是主槽位，它不反过来借。
    """
    if slot != "text":
        return {}
    env = env_override("vlm")
    raw = slot_config("vlm")
    return {
        "provider": env["provider"] or str(raw.get("provider") or ""),
        "model": env["model"] or str(raw.get("model") or ""),
        "endpoint": env["endpoint"] or str(raw.get("endpoint") or ""),
        "protocol": env["protocol"] or str(raw.get("protocol") or ""),
    }


def credentials_for(slot: str):
    """某个槽位的调用凭据。没有配就返回 None（**不猜、不编**）。"""
    from . import vlm_client

    raw = effective_slot(slot)
    env = env_override(slot)
    key = slot_key(slot)
    if not key:
        return None
    inherited = _borrow_profile(slot)
    provider = (env["provider"] or str(raw.get("provider") or "")
                or inherited.get("provider", ""))
    model = env["model"] or str(raw.get("model") or "") or inherited.get("model", "")
    resolved = model_presets.resolve(provider, model, slot)
    endpoint = (env["endpoint"] or str(raw.get("endpoint") or "")
                or inherited.get("endpoint", "") or resolved["endpoint"]).strip()
    if not endpoint:
        return None
    protocol = (env["protocol"] or str(raw.get("protocol") or "")
                or inherited.get("protocol", "") or resolved["protocol"])
    return vlm_client.Credentials(
        api_key=key,
        endpoint=endpoint,
        protocol=protocol,
        thinking_field=str(raw.get("thinking_field") or resolved["thinking_field"]),
        model=model or resolved["model"],
        source=f"config:{slot}:{provider}",
        # Files API 的根地址（传整局视频时用）。空 = 这个供应方
        # 不支持上传，只能直传小视频。
        files_base=str(raw.get("files_base") or resolved.get("files_base") or ""),
    )


def model_name(slot: str = "vlm") -> str:
    """某个槽位配的模型名（给报告标注来源用）。"""
    env = env_override(slot)
    if env["model"]:
        return env["model"]
    raw = effective_slot(slot)
    return str(raw.get("model") or "")


# ---------------------------------------------------------------- 其它设置


def limits() -> Dict[str, Any]:
    """调用限额（写进配置，页面上可改）。"""
    conf = app_config()
    raw = conf.get("limits") or {}
    if not isinstance(raw, dict):
        raw = {}
    return {
        "timeout": float(raw.get("timeout") or 300),
        "concurrency": max(1, min(8, int(raw.get("concurrency") or 4))),
        "max_calls": max(1, int(raw.get("max_calls") or 30)),
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
        "has_key": bool(vlm.get("has_key")),
        "can_see_images": bool(vlm.get("can_see_images")),
        "model": vlm.get("model") or "",
        "provider": vlm.get("provider_label") or "",
        "slots": slots,
        "limits": limits(),
        "data_dir": data_dir(),
    }
