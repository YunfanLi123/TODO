"""会话生命周期：用完即焚 + 每人隔离。

这个服务支持一种部署形态：**不留存**。
用户的报告、上传的视频，关掉网页就删掉，服务器上不留痕迹。

有一件事必须说清楚：
    视频要解码、报告要算 —— 处理的过程中数据必然在服务器上存在过。
    能做到的是「**用完即删、不可恢复**」，不是「从未存在过」。
    真要连临时都不落，只能把识别整个搬到浏览器本地做，那是另一个架构。

三个触发点，缺一不可：
    1. 页面关闭 → 前端发 beacon（``/api/session/end``）→ 排一次删除；
    2. beacon 丢了（浏览器崩溃、断网、被杀）→ 兜底清扫：
       超过 TTL 没有任何动静的会话，连同它的数据一起删；
    3. 延迟一小段再删，而且**期间有请求就不删** ——
       多标签页只关了一个、或者关了又回来，不该被误删。

第 3 条是这里最容易做错的地方：只要"标签页关闭"就立刻删，
另一个标签页里正在看的报告会被抽走。
"""

from __future__ import annotations

import glob
import os
import shutil
import time
from typing import Dict, List, Optional, Set, Tuple

#: 打开即焚模式的环境变量。
ENV_NAME = "EPHEMERAL_SESSION"

#: 关掉网页后延迟多久真的删。
#: 太短会误伤「关了标签页又切回来」，太长等于没删。
DEFAULT_GRACE = 30.0

#: 超过多久没有任何动静就当这个会话死了（beacon 丢了的兜底）。
DEFAULT_TTL = 3600.0

#: 设备标识 -> 最后一次见到它的时间。
_LAST_SEEN: Dict[str, float] = {}

#: 设备标识 -> 标记为「已离开」的时间。
_CLOSED_AT: Dict[str, float] = {}

_TRUTHY = {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def enabled() -> bool:
    """即焚模式开没开。"""
    return (os.environ.get(ENV_NAME) or "").strip().lower() in _TRUTHY


def grace_seconds() -> float:
    """关掉网页后，等多久才真的删。"""
    return _float_env("EPHEMERAL_GRACE_SECONDS", DEFAULT_GRACE)


def ttl_seconds() -> float:
    """多久没有任何动静就算这个会话死了。"""
    return _float_env("EPHEMERAL_TTL_SECONDS", DEFAULT_TTL)


def touch(device_id: str) -> None:
    """这个设备刚刚来过。

    注意**不要**在这里取消待删标记：多标签页场景下，
    老标签页关了会打标记，而新标签页还在用 ——
    要靠「标记之后还有没有请求」来判断该不该删，不是靠清除标记。
    """
    if device_id:
        _LAST_SEEN[device_id] = time.time()


def mark_closed(device_id: str) -> None:
    """页面关了。记下来，过一会儿真的删。"""
    if device_id:
        _CLOSED_AT[device_id] = time.time()


def due_devices(now: Optional[float] = None) -> List[str]:
    """现在该删的设备标识。

    判据是「标记为关闭之后，**再没有任何请求**」——
    所以另一个标签页还在用的时候，数据不会被抽走。
    """
    now = now if now is not None else time.time()
    due: List[str] = []
    for dev, closed in list(_CLOSED_AT.items()):
        last_seen = _LAST_SEEN.get(dev, closed)
        if now - max(closed, last_seen) >= grace_seconds():
            due.append(dev)
    return due


def expired_devices(now: Optional[float] = None) -> List[str]:
    """太久没动静的设备（beacon 丢了的那一批）。"""
    now = now if now is not None else time.time()
    ttl = ttl_seconds()
    return [
        dev
        for dev, seen in list(_LAST_SEEN.items())
        if now - seen >= ttl and dev not in _CLOSED_AT
    ]


def forget(device_id: str) -> None:
    """删干净，别留下幽灵记录。"""
    _LAST_SEEN.pop(device_id, None)
    _CLOSED_AT.pop(device_id, None)


def reset() -> None:
    """清空会话状态（测试用）。

    进程重启时这些状态自然就没了 —— 所以这不是什么需要小心翼翼的东西。
    """
    _LAST_SEEN.clear()
    _CLOSED_AT.clear()


# --------------------------------------------------------------------------
# 真正动手删
# --------------------------------------------------------------------------


def purge_device(device_id: str, base_dir: str = "") -> int:
    """删掉这个设备的全部对局与分析，以及它上传的视频文件。

    返回删掉的对局数。
    """
    from ..collection import match as match_col
    from ..db import session_scope

    with session_scope() as s:
        pairs = match_col.delete_matches_by_player(s, player_id=device_id)
    for _match_id, video_id in pairs:
        if video_id:
            delete_video_files(video_id, base_dir)
    forget(device_id)
    return len(pairs)


def delete_video_files(video_id: str, base_dir: str = "") -> None:
    """删掉一个视频的文件与抽出来的帧。"""
    from ..collection import video_upload as vu

    store = vu.video_store_dir(base_dir)
    for path in glob.glob(os.path.join(store, video_id + ".*")):
        try:
            os.remove(path)
        except OSError:
            pass
    shutil.rmtree(os.path.join(store, video_id + "_frames"), ignore_errors=True)


def sweep(base_dir: str = "") -> int:
    """兜底清扫：删掉到期会话的数据，以及没人认领的孤儿视频。"""
    now = time.time()
    removed = 0
    for dev in due_devices(now) + expired_devices(now):
        try:
            removed += purge_device(dev, base_dir)
        except Exception:  # noqa: BLE001  清扫失败不该把整个进程带崩
            pass
    removed += _sweep_orphan_videos(base_dir, now)
    return removed


def _sweep_orphan_videos(base_dir: str, now: float) -> int:
    """删掉超过 TTL、又没有任何对局引用的视频文件。

    传了视频但没提交标注的那一批，数据库里没有它们的记录 ——
    它们只以文件的形式躺在磁盘上。没人认领的，就该清掉。

    **有对局引用的不能动**：用户可能开着页面看了一小时没动鼠标，
    那局还在、视频就该还在。
    """
    from ..collection import match as match_col
    from ..collection import video_upload as vu
    from ..db import session_scope

    with session_scope() as s:
        referenced = match_col.list_video_ids(s)

    ttl = ttl_seconds()
    store = vu.video_store_dir(base_dir)
    removed = 0
    for path in glob.glob(os.path.join(store, "*")):
        name = os.path.basename(path)
        video_id = name.split(".", 1)[0]
        if video_id in referenced:
            continue
        try:
            if now - os.path.getmtime(path) < ttl:
                continue
        except OSError:
            continue
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
            removed += 1
        except OSError:
            pass
    return removed


def referenced_video_ids() -> Set[str]:
    """当前还有对局引用的视频 id（给测试与排查用）。"""
    from ..collection import match as match_col
    from ..db import session_scope

    with session_scope() as s:
        return match_col.list_video_ids(s)
