"""设备身份：这个浏览器是谁。

公网部署时不能所有人共用一个 ``player_id="local"`` ——
那意味着 A 录的对局 B 也能看见、也能点删除。

但用户要的是「无需登录」。所以身份不用账号，用**浏览器本地的一个随机标识**：
第一次访问时生成，写进 Cookie，之后一直带着。
它不是账号（没有密码、没有注册、没有邮箱），
但足以把各人的对局分开 —— 数据层本来就是按 ``player_id`` 分区的。

开关：环境变量 ``MULTI_TENANT=1`` 才启用。
**默认关闭** —— 本地单人使用行为完全不变，测试也不受影响。
"""

from __future__ import annotations

import os
import uuid
from typing import Optional

from . import session

#: 存放设备标识的 Cookie 名。
DEVICE_COOKIE = "ma_device"

#: 未开启隔离时的玩家标识 —— 本地单人使用的历史默认值。
DEFAULT_PLAYER_ID = "local"

#: 设备标识的 Cookie 有效期（一年）。
_COOKIE_MAX_AGE = 60 * 60 * 24 * 365

_TRUTHY = {"1", "true", "yes", "on"}


def multi_tenant_enabled() -> bool:
    """是否按设备隔离数据（多人共用一个部署）。"""
    return (os.environ.get("MULTI_TENANT") or "").strip().lower() in _TRUTHY


def isolation_enabled() -> bool:
    """要不要按设备隔离。

    两个来源：
      * ``MULTI_TENANT=1`` —— 多人共用一个部署，各看各的；
      * ``EPHEMERAL_SESSION=1`` —— 即焚模式。

    即焚模式**必须**隔离：清一个人的数据时如果不分设备，
    就会把所有人的对局一起清掉 —— 那不是"清干净"，是"清成空"。
    """
    return multi_tenant_enabled() or session.enabled()


def new_device_id() -> str:
    """生成一个新的设备标识。"""
    return uuid.uuid4().hex[:16]


def device_id(request, default: Optional[str] = None) -> str:
    """当前请求属于哪个玩家。

    优先级：显式 ``?player_id=``（脚本、测试、分享链接）→ Cookie 里的设备标识
    → 默认值（``local``）。

    隔离没开时，永远返回默认值 —— 不看 Cookie，
    本地行为一个字都不变。
    """
    fallback = default or DEFAULT_PLAYER_ID
    if not isolation_enabled():
        return fallback
    explicit = request.query_params.get("player_id")
    if explicit:
        return explicit
    cookie = request.cookies.get(DEVICE_COOKIE)
    return cookie or fallback


def ensure_device_cookie(request, response, device: str) -> None:
    """第一次访问时把设备标识写进 Cookie。

    已经是访客了（Cookie 里有）就什么都不做 ——
    每次响应都重设一遍会让 Cookie 一直续期，
    也让「换浏览器就是换个人」这件事变得不明确。
    """
    if not isolation_enabled():
        return
    if request.cookies.get(DEVICE_COOKIE):
        return
    response.set_cookie(
        DEVICE_COOKIE,
        device,
        max_age=_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
    )
