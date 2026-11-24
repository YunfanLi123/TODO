"""PostgreSQL 的行级安全（RLS）—— 「彻底隔离」的那道数据库层防线。

为什么要有它
------------
应用层的「按 player_id 过滤」是**代码写的**：哪个接口忘了加过滤条件，
或者哪个接口把 ``player_id`` 当参数传了（``?player_id=别人``），
数据就漏出去了。RLS 是**数据库层**的第二道：

    策略写死「只看 player_id = 当前设备」，
    应用层就算传错、就算没传，数据库也不会把别人的行给他。

三件必须做对的事
-----------------
1. **身份来自事务变量** ``app.player_id`` ——
   由 :func:`app.db.set_player` / 引擎的 begin 钩子写入（``SET LOCAL``）。
2. **必须 ``FORCE ROW LEVEL SECURITY``** ——
   PG 默认让**表 owner 绕过**策略。CloudBase 给的连接用户就是 owner，
   不 FORCE 等于策略根本没生效（而表面上看一切正常）。
3. **全局内容表不加策略** —— 规则库、攻略源是**产品内容**，所有人读同一份，
   它们本来就不含用户数据。
"""
from __future__ import annotations

from typing import List, Tuple

#: 需要保护、且**自己就有 player_id 列**的用户数据表。
#:
#: ``matches``   有 player_id 列，直接比。
#:
#: ``analyses``  **不在这个列表里** —— 它没有 player_id 列（靠外键挂到
#: matches 上），得用下面那条单独的策略。把它塞进来会生成
#: 「按 player_id 过滤 analyses」的策略 → 建策略时直接报错
#: （列不存在）。这是这张表与 matches 的唯一差别，别混了。
RLS_USER_TABLES: Tuple[str, ...] = ("matches",)

#: 靠外键判断归属的表：``(表名, 父表)``
RLS_CHILD_TABLES: Tuple[Tuple[str, str], ...] = (("analyses", "matches"),)

#: 全局内容表 —— **不加策略**，所有人读同一份。
RLS_GLOBAL_TABLES: Tuple[str, ...] = ("rules", "rule_revisions", "guide_sources")


def rls_statements() -> List[str]:
    """建策略的 SQL（一条一条，幂等）。"""
    out: List[str] = []
    for table in RLS_USER_TABLES:
        out.append("DROP POLICY IF EXISTS %s_own_rows ON %s" % (table, table))
        out.append("ALTER TABLE %s ENABLE ROW LEVEL SECURITY" % table)
        out.append("ALTER TABLE %s FORCE ROW LEVEL SECURITY" % table)
        out.append(
            "CREATE POLICY %s_own_rows ON %s"
            " USING (player_id = current_setting('app.player_id', true))"
            " WITH CHECK (player_id = current_setting('app.player_id', true))"
            % (table, table))
    for table, parent in RLS_CHILD_TABLES:
        out.append("DROP POLICY IF EXISTS %s_own_rows ON %s" % (table, table))
        out.append("ALTER TABLE %s ENABLE ROW LEVEL SECURITY" % table)
        out.append("ALTER TABLE %s FORCE ROW LEVEL SECURITY" % table)
        out.append(
            "CREATE POLICY %s_own_rows ON %s"
            " USING (EXISTS (SELECT 1 FROM %s p"
            "   WHERE p.id = %s.match_id"
            "     AND p.player_id = current_setting('app.player_id', true)))"
            " WITH CHECK (EXISTS (SELECT 1 FROM %s p"
            "   WHERE p.id = %s.match_id"
            "     AND p.player_id = current_setting('app.player_id', true)))"
            % (table, table, parent, table, parent, table))
    for table in RLS_GLOBAL_TABLES:
        out.append("ALTER TABLE %s DISABLE ROW LEVEL SECURITY" % table)
    return out


def ensure_rls(engine) -> bool:
    """在 PG 上把策略建起来（幂等；SQLite 上什么都不做）。

    **为什么放在启动时做**：PG 实例只能从 VPC 内访问，本机连不上 ——
    与其让部署脚本单独跑一遍（还得是容器里跑），不如启动时自己建。
    """
    if engine.dialect.name != "postgresql":
        return False
    from sqlalchemy import text

    with engine.begin() as conn:
        for statement in rls_statements():
            conn.execute(text(statement))
    return True
