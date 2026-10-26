"""数据持久层：SQLAlchemy 模型。

四张表对应四层架构里需要长期保存的东西：
  Match        采集层产出的对局（含原始数据，原样留存以便重算）
  Analysis     分析内核产出的判定结果快照
  RuleRecord   规则层的结构化规则库（按游戏 + 版本独立存储）
  GuideSource  攻略来源溯源

设计要点：原始对局数据永不丢弃。规则库会迭代、口径会调整，
保留 raw 才能用新规则重跑历史对局 —— 这是版本回测的前提。
"""

from __future__ import annotations

import json
import contextlib
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: 数据目录。
#:
#: 可以被环境变量 ``MATCH_DATA_DIR`` 覆盖 —— 打包成 exe 后这是**必须**的：
#: onefile 模式会把代码解包到临时目录，而临时目录会在进程退出时被清掉。
#: 照默认逻辑走的话，数据库会跟着临时目录一起消失，用户重启一次
#: 录过的对局就没了。exe 启动器会把这个变量指到用户可写的地方。
DATA_DIR = os.environ.get("MATCH_DATA_DIR") or os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "app.db")


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Match(Base):
    __tablename__ = "matches"

    id = Column(String(64), primary_key=True)
    game = Column(String(32), nullable=False, index=True)
    game_version = Column(String(32), nullable=False, index=True)
    player_id = Column(String(64), nullable=False, index=True, default="local")
    played_at = Column(DateTime, nullable=False, default=utcnow, index=True)
    outcome = Column(Text, default="{}")           # JSON：名次/胜负
    payload = Column(Text, default="{}")           # JSON：原始对局数据
    source_kind = Column(String(16), default="manual")  # manual / api / video
    created_at = Column(DateTime, default=utcnow)

    analyses = relationship(
        "Analysis", back_populates="match", cascade="all, delete-orphan"
    )

    def set_outcome(self, value: Dict[str, Any]) -> None:
        self.outcome = json.dumps(value, ensure_ascii=False)

    def set_payload(self, value: Dict[str, Any]) -> None:
        self.payload = json.dumps(value, ensure_ascii=False)

    def get_outcome(self) -> Dict[str, Any]:
        return json.loads(self.outcome or "{}")

    def get_payload(self) -> Dict[str, Any]:
        return json.loads(self.payload or "{}")


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(64), ForeignKey("matches.id"), nullable=False, index=True)
    rules_version = Column(String(32), nullable=False)
    compliance_rate = Column(Float, nullable=True)   # 无规则触发时为 NULL
    triggered = Column(Integer, default=0)
    satisfied = Column(Integer, default=0)
    deviated = Column(Integer, default=0)
    undetermined = Column(Integer, default=0)
    detail = Column(Text, default="{}")              # JSON：完整 gaps/verdicts
    created_at = Column(DateTime, default=utcnow)

    match = relationship("Match", back_populates="analyses")

    def set_detail(self, value: Dict[str, Any]) -> None:
        self.detail = json.dumps(value, ensure_ascii=False)

    def get_detail(self) -> Dict[str, Any]:
        return json.loads(self.detail or "{}")


class RuleRecord(Base):
    """规则库条目。``(game, version, rule_id)`` 唯一 —— 同名规则在不同版本
    可以并存，这是版本适配的基础。"""

    __tablename__ = "rules"
    __table_args__ = (UniqueConstraint("game", "version", "rule_id", name="uq_rule_ver"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    game = Column(String(32), nullable=False, index=True)
    version = Column(String(32), nullable=False, index=True)
    rule_id = Column(String(64), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    category = Column(String(64), nullable=False, index=True)
    severity = Column(String(16), default="medium")
    status = Column(String(16), default="active", index=True)
    payload = Column(Text, nullable=False)           # JSON：Rule.to_dict()
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    # -- 审核痕迹 ----------------------------------------------------------
    #: 谁改的。单机部署时就是本机用户名，取不到就留空。
    reviewed_by = Column(String(64), default="")
    reviewed_at = Column(DateTime, nullable=True)
    #: 审核备注：为什么改、为什么驳回
    review_note = Column(Text, default="")
    #: 抽取器原始产出。人工可能改过阈值甚至改过方向，留着原件才能
    #: 回答「这个抽取器到底哪里不行」—— 那正是改进抽取器的唯一依据。
    original_payload = Column(Text, default="")    # JSON：Rule.to_dict() 原样
    #: 人工是否修改过。区分「机器抽的」和「人改过的」，
    #: 两者在报告里的可信度不是一个级别。
    edited = Column(Integer, default=0)

    @property
    def original(self):
        """抽取器最初抽出的版本；没留痕时返回当前版本。"""
        import json as _json

        if not self.original_payload:
            return _json.loads(self.payload)
        return _json.loads(self.original_payload)

    def set_payload(self, rule_dict: Dict[str, Any], edited: bool = False) -> None:
        self.payload = json.dumps(rule_dict, ensure_ascii=False)
        if edited and not self.original_payload:
            self.original_payload = self.payload


class RuleRevision(Base):
    """规则修改历史。

    单看当前值看不出「为什么变成这样」。阈值从 ``<=30`` 改成 ``>=30`` 看起来
    更合理，但只有连着看几次修改才知道是人一直在跟抽取器的方向错误较劲，
    还是这一版 meta 变了。留历史是让人能回顾自己判断的依据。
    """

    __tablename__ = "rule_revisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    game = Column(String(32), nullable=False, index=True)
    version = Column(String(32), nullable=False, index=True)
    rule_id = Column(String(64), nullable=False, index=True)
    #: 改动的字段名（trigger / expect / severity / status …）
    field = Column(String(32), nullable=False)
    before = Column(Text, default="")
    after = Column(Text, default="")
    note = Column(Text, default="")
    edited_by = Column(String(64), default="")
    created_at = Column(DateTime, default=utcnow)


class GuideSource(Base):
    """攻略来源记录。规则库的溯源目标。

    遵循立项设计：只存结构化规则，不存攻略原文全文，
    这里只保留出处元信息与必要引用片段。"""

    __tablename__ = "guide_sources"

    id = Column(Integer, primary_key=True, autoincrement=True)
    game = Column(String(32), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    url = Column(String(512), default="")
    kind = Column(String(32), default="community")   # official / community / user
    note = Column(Text, default="")
    collected_at = Column(DateTime, default=utcnow)


# --------------------------------------------------------------------------
# 复合索引
#
# 必须定义在全部模型之后 —— 索引语句要引用表对象，而表要等类定义完才存在。
# （踩过：写在 Match 之前，import 直接 NameError。）
# --------------------------------------------------------------------------

#: 覆盖「某个玩家某款游戏的对局，按时间倒序」这一条最常见的查询。
#:
#: 单列索引看着够用（player_id、game、played_at 各有一个），但数据库只能
#: 挑一个用，其余的仍要回表过滤。数据量上千之后这条查询会明显拖慢成长曲线
#: 和版本回测 —— 而它们恰恰是每次打开报告都会走的路径。
Index(
    "ix_match_player_game_time",
    Match.__table__.c.player_id,
    Match.__table__.c.game,
    Match.__table__.c.played_at,
)

#: 回测重建上下文时按版本过滤，走 (game, game_version)。
Index(
    "ix_match_game_version",
    Match.__table__.c.game,
    Match.__table__.c.game_version,
)


_engine = None
_SessionFactory = None


#: 数据库连接串。
#:
#: 默认留空 = 用本地 SQLite（``data/app.db``）。
#:
#: 云端（CloudBase 云托管）必须设成 PostgreSQL：容器实例的本地磁盘是
#: **临时**的，实例回收、版本切换、缩容到 0 之后磁盘会重置，
#: SQLite 文件会跟着消失 —— 录过的对局和灌好的规则库一起没。
#: 所以云上走 CloudBase 内置 PostgreSQL（内网直连，不走公网）。
#:
#: 形如：``postgresql+psycopg://user:pass@host:5432/dbname``
def _database_url() -> str:
    """连接串在**调用时**读，而不是 import 时。

    import 时读会有个具体的坑：测试里换了临时数据库，
    而环境变量是进程启动时定下的 —— 两边对不上时，
    测试会莫名其妙地去连生产库。
    """
    return (os.environ.get("DATABASE_URL") or "").strip()


def get_engine(db_path: Optional[str] = None):
    global _engine
    if _engine is None:
        url = _database_url()
        if url:
            # 外部数据库：pool_pre_ping 让断连后自动重连（云托管重启、
            # PG 休眠唤醒时连接会失效，没有它会直接抛 DisconnectionError）。
            _engine = create_engine(
                url,
                pool_pre_ping=True,
                pool_recycle=1800,
                future=True,
            )
        else:
            path = db_path or DB_PATH
            os.makedirs(os.path.dirname(path), exist_ok=True)
            _engine = create_engine(
                f"sqlite:///{path}",
                connect_args={"check_same_thread": False},
                future=True,
            )
        Base.metadata.create_all(_engine)
    return _engine


def get_session_factory():
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(
            bind=get_engine(), expire_on_commit=False, future=True
        )
    return _SessionFactory


def init_db(db_path: Optional[str] = None) -> None:
    get_engine(db_path)


@contextlib.contextmanager
def session_scope():
    """打开一个数据库会话，正常退出提交、异常回滚。

    用 ``@contextmanager`` 而非裸生成器，是为了让同一份代码既能
    ``with session_scope() as s`` （脚本、测试），也能作为 FastAPI 的
    ``Depends`` 使用 —— 生成器依赖由 FastAPI 托管生命周期，
    而直接 ``with`` 一个生成器会抛 TypeError。
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


#: 旧命名别名，保持与分层命名一致
get_db = session_scope
