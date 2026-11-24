"""分片上传：大文件在云端的唯一活路。

为什么需要它
------------
云端（CloudBase 云托管）前面的网关对**单次请求体**有硬上限。实测：
20MB 能过，24MB 会被直接掐断连接 —— 客户端收到的是 TLS EOF，
而不是一个像样的 4xx。而一局王者录屏动辄几百 MB。

一次性 POST 必然失败，而且失败得很难看：网关回一个 HTML 错误页，
前端把它当 JSON 解析，于是报 ``Unexpected token '<'`` ——
跟「文件太大」这个真实原因毫无关系。**这是实测出来的，不是猜的。**

所以大文件切成 8MB 的片，一片一片传，服务端按偏移量写进**最终的**
视频文件里 —— 不另存再合并：780MB 复制一遍既慢又占双份空间，
而云端的临时存储是拿实例内存换的（文档原话：过度使用可能导致内存不足），
双份是实打实的风险。

分片是**幂等**的：同一片重传只是把同样的字节写回同一位置。
中途断了，续传就行，不用从头再来。
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .video_upload import (
    MAX_UPLOAD_BYTES,
    UploadedVideo,
    VideoProcessingError,
    video_store_dir,
)


#: 单片大小。
#:
#: 网关的实测硬上限是 20MB（24MB 直接断连）。取 8MB 是为了留足余量：
#: 请求体算上 multipart 边界和网络重传，不能顶着线走。
CHUNK_SIZE = 8 * 1024 * 1024

#: 一个上传会话活多久。超时连同它的分片一起清掉 ——
#: 半截的视频留在磁盘上，既占地方，下次列目录还会误导人。
SESSION_TTL_SECONDS = 3600.0


@dataclass
class ChunkSession:
    """一次分片上传。

    分片**直接写进最终视频文件**，按偏移量落位。所以这个会话
    没有「合并」这一步 —— 传完最后一片，这个文件就已经是完整的视频了。
    """

    upload_id: str
    video_id: str
    filename: str
    size: int
    chunk_size: int
    total_chunks: int
    path: str
    player_id: str
    #: 玩家上传前填的英雄 / 分路（整条链路以它为准）。
    hero: str = ""
    position: str = ""
    created_at: float = field(default_factory=time.time)
    #: 每片实际收到多少字节。键是片号，值是长度。
    lengths: Dict[int, int] = field(default_factory=dict)

    @property
    def received(self) -> int:
        return len(self.lengths)

    @property
    def received_bytes(self) -> int:
        return sum(self.lengths.values())


#: 进程内的会话表。
#:
#: 用内存而不是数据库：这是一次性的中转状态，进程重启后重新上传即可
#: —— 跟扫描任务的缓存是同一个理由。
_SESSIONS: Dict[str, ChunkSession] = {}
_LOCK = threading.Lock()


def _target_path(base_dir: str, video_id: str, filename: str) -> str:
    """分片文件的最终落点。

    与 :func:`app.collection.video_upload._target_path` 同一套规则
    （扩展名白名单 + 同一个 videos 目录），但这里只认位置不搬文件 ——
    合并那一步在传完最后一片时自然就完成了。
    """
    from .video_upload import _target_path as _vt

    return _vt(base_dir, video_id, filename)


def start(
    filename: str, size: int, base_dir: str, player_id: str = "local",
    hero: str = "", position: str = "",
) -> ChunkSession:
    """开一个上传会话，返回分片计划。

    文件名先过扩展名白名单：不合法就当场拒掉，别等传完 780MB
    才告诉用户「格式不支持」。
    """
    name = (filename or "").strip() or "video.mp4"
    if size <= 0:
        raise VideoProcessingError("上传内容为空")
    if size > MAX_UPLOAD_BYTES:
        raise VideoProcessingError(
            f"视频 {size / 1024 ** 3:.2f} GB 超过上限 "
            f"{MAX_UPLOAD_BYTES / 1024 ** 3:.1f} GB"
        )

    video_id = uuid.uuid4().hex[:12]
    path = _target_path(base_dir, video_id, name)   # 扩展名不对时在这里抛

    chunk_size = CHUNK_SIZE
    total = (size + chunk_size - 1) // chunk_size

    session = ChunkSession(
        upload_id=uuid.uuid4().hex[:16],
        video_id=video_id,
        filename=name,
        size=size,
        chunk_size=chunk_size,
        total_chunks=total,
        path=path,
        player_id=player_id or "local",
        hero=str(hero or "").strip()[:20],
        position=str(position or "").strip(),
    )

    # 建一个空文件：之后每片按偏移量写进去。
    #
    # **不预先分配大小** —— 预先分配会在不支持稀疏文件的文件系统上
    # 立刻占掉 780MB。云端的临时存储拿的是实例内存，这种账不能这么花。
    with open(path, "wb"):
        pass

    with _LOCK:
        _SESSIONS[session.upload_id] = session
    return session


def get(upload_id: str, player_id: str = "local") -> Optional[ChunkSession]:
    """取会话。**校验归属** —— 别的设备不该能接手你的上传。"""
    with _LOCK:
        session = _SESSIONS.get(upload_id)
    if session is None:
        return None
    if player_id and session.player_id != player_id:
        return None
    return session


def put_chunk(session: ChunkSession, index: int, data: bytes) -> int:
    """收一片，写到对应偏移量上。返回已收到的片数。

    幂等：同一片重传会把同样的字节写回同一位置，
    所以断点续传不需要服务端记「哪几片已经好了」以外的东西。
    """
    if index < 0 or index >= session.total_chunks:
        raise VideoProcessingError(
            f"片号 {index} 越界（总共 {session.total_chunks} 片）"
        )
    if not data:
        raise VideoProcessingError("这一片是空的")
    if len(data) > session.chunk_size:
        raise VideoProcessingError(
            f"这一片 {len(data) / 1024 ** 2:.1f} MB，超过单片上限 "
            f"{session.chunk_size / 1024 ** 2:.0f} MB"
        )

    offset = index * session.chunk_size
    with open(session.path, "r+b") as fh:
        fh.seek(offset)
        fh.write(data)

    with _LOCK:
        session.lengths[index] = len(data)
    return session.received


def finish(session: ChunkSession) -> UploadedVideo:
    """校验完整性并交出视频对象。

    缺片或者总字节对不上都当场拒掉 —— 半截的视频扫出来只有莫名其妙的结果，
    比直接报错糟糕得多。
    """
    missing: List[int] = [
        i for i in range(session.total_chunks) if i not in session.lengths
    ]
    if missing:
        raise VideoProcessingError(
            f"还差 {len(missing)} 片没传完（共 {session.total_chunks} 片）"
        )
    if session.received_bytes != session.size:
        raise VideoProcessingError(
            f"传上来的字节数对不上：应为 {session.size}，实际 {session.received_bytes}"
        )

    try:
        actual = os.path.getsize(session.path)
    except OSError as exc:
        raise VideoProcessingError(f"视频文件读不出来：{exc}") from exc
    if actual != session.size:
        raise VideoProcessingError(
            f"视频文件大小不对：应为 {session.size}，实际 {actual}"
        )

    with _LOCK:
        _SESSIONS.pop(session.upload_id, None)

    return UploadedVideo(
        video_id=session.video_id,
        path=session.path,
        original_name=session.filename,
        size=session.size,
        stored_at=time.strftime("%Y-%m-%d %H:%M"),
    )


def drop(upload_id: str) -> None:
    """放弃一个会话，并删掉它已经写下的半截文件。"""
    with _LOCK:
        session = _SESSIONS.pop(upload_id, None)
    if session is None:
        return
    try:
        os.remove(session.path)
    except OSError:
        pass


def sweep(max_age: float = SESSION_TTL_SECONDS) -> int:
    """清掉超时未完成的会话（含半截文件）。

    半截文件必须删 —— 留着既占磁盘，下次列目录还会让人以为
    「有个视频在这儿，传到一半断了」。
    """
    now = time.time()
    stale = []
    with _LOCK:
        for uid, session in list(_SESSIONS.items()):
            if now - session.created_at >= max_age:
                stale.append(uid)
    for uid in stale:
        drop(uid)
    return len(stale)


