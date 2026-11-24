"""采集会话：跨请求保存候选，让「确认入库」不必重新抓取正文。

系统不存攻略原文全文，所以采集完正文就丢弃了。但如果确认入库时再抓一次
——既慢，又可能站点内容已经变了，导致「你看到的是 A，入库的是 B」。
所以这里把抽取出的**候选**（结构化、不含原文）暂存在会话里，确认时直接用。

会话是有时限的：候选放着不确认就过期，避免几天后对着早就忘了的内容
稀里糊涂地点确认。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..collection.collector import CollectionResult

#: 会话有效期（秒）。超时的会话会被自动清理。
DEFAULT_TTL_SECONDS = 30 * 60

#: 最多保留的会话数，超出后按最旧的淘汰。
MAX_SESSIONS = 64


@dataclass
class CollectionSession:
    """一次采集的暂存结果。"""

    session_id: str
    result: CollectionResult
    created_at: float = field(default_factory=time.time)

    @property
    def age(self) -> float:
        return time.time() - self.created_at

    def expired(self, ttl: float) -> bool:
        return self.age > ttl


class SessionStore:
    """线程安全的会话存储。

    内存态且带 TTL 是刻意的：这些数据只是「从采集到确认」之间的中转，
    没有任何长期价值，丢掉了只需要重新采一次。放进数据库反而会让人误以为
    系统在长期保存攻略内容 —— 而那恰恰是合规立场不允许的。
    """

    def __init__(self, ttl: float = DEFAULT_TTL_SECONDS) -> None:
        self._sessions: Dict[str, CollectionSession] = {}
        self._lock = threading.Lock()
        self._ttl = ttl

    def put(self, session_id: str, result: CollectionResult) -> None:
        with self._lock:
            self._evict_expired_locked()
            self._sessions[session_id] = CollectionSession(session_id, result)
            self._evict_overflow_locked()

    def get(self, session_id: str) -> Optional[CollectionSession]:
        with self._lock:
            self._evict_expired_locked()
            return self._sessions.get(session_id)

    def drop(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def size(self) -> int:
        with self._lock:
            self._evict_expired_locked()
            return len(self._sessions)

    # -- 淘汰 ---------------------------------------------------------------

    def _evict_expired_locked(self) -> None:
        for sid in [
            s.session_id for s in self._sessions.values() if s.expired(self._ttl)
        ]:
            self._sessions.pop(sid, None)

    def _evict_overflow_locked(self) -> None:
        while len(self._sessions) > MAX_SESSIONS:
            oldest = min(self._sessions.values(), key=lambda s: s.created_at)
            self._sessions.pop(oldest.session_id, None)


#: 进程级单例
_STORE = SessionStore()
