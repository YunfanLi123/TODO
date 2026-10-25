"""页面路由：只留三个入口。

    /         封面
    /upload   上传视频
    /matches  对局记录（点进去出报告）
    /rules    操作模式库（分路 → 英雄 → 模式）

曾经还有「攻略工坊」和「版本管理」两个独立页面，现在都并进规则库了：
攻略只是**往规则库里灌条目的一个入口**，不是一项独立能力；
版本切换也只是规则库的一个筛选维度。拆成两个页面只会让人多点两次，
还会以为「在版本页改的规则和在攻略页改的不是一回事」。

这个文件刻意短 —— 路由多起来之后，再拆成 collector/authoring/
dashboard 三四个模块，最后没人说得清某个页面在哪儿。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse

from ..core import device
from ..report import organize
from ..report.views import render_page

#: 这个产品现在只做王者荣耀。
#:
#: 曾经这里是三款游戏的切换（tft / jcc / hok），而那两款在产品形态上
#: 和王者根本不同 —— 自走棋没有分路、没有「前中后期」，它的模式库
#: 组织方式和王者对不上。留着切换就是留着一条「切过去看到的是另一套
#: 东西」的路，而两条路的数据口径并不一致。
GAME = "hok"


def register_pages(app: FastAPI, base_dir: str) -> None:
    from ..adapters.registry import REGISTRY
    from ..db import session_scope
    from ..collection import match as match_col
    from ..rules.review import build_review_view
    from ..rules.store import get_rules, latest_version, list_versions

    adapter = REGISTRY[GAME]

    # ================================================================ 落地页
    @app.get("/", response_class=HTMLResponse)
    def landing() -> HTMLResponse:
        """封面页。只有一个动作：进入。

        它不是「欢迎页」那种可有可无的东西 —— 整个产品的立论是
        「不断向被验证的模式优化拟合」，那句话说清楚再让人进来，
        比一上来就甩一个上传框更符合这个承诺。
        """
        return HTMLResponse(render_page("landing.html"))

    # ================================================================ 上传
    @app.get("/upload", response_class=HTMLResponse)
    def upload_page() -> HTMLResponse:
        """工作台。只有一件事：把视频传上去。

        页面上原来还有一排彩色数字卡（对局数 / 模式数 / 技术报告数），
        而侧边栏就有同样的三个入口 —— 同一件事在两处各说一次，
        数字还可能对不上（统计口径不同）。现在只留上传本身，
        让上传区把整块空白占满。
        """
        return HTMLResponse(render_page("index.html"))

    # ================================================================ 模型
    @app.get("/model", response_class=HTMLResponse)
    def model_page() -> HTMLResponse:
        """模型配置页。

        只填一个 **API Key** 就能配好：供应方、完整端点、协议、
        「关思考」那个字段叫什么、默认模型名 —— 全是供应方的事，
        预设里备好了，用户不用碰。

        密钥**只写不回**：这一页读到的永远是掩码。
        """
        return HTMLResponse(render_page("model.html"))

    # ============================================================== 对局记录
    @app.get("/matches", response_class=HTMLResponse)
    def matches_page(
        request: Request,
        lane: str = Query(""),
        q: str = Query(""),
    ) -> HTMLResponse:
        """对局记录：按分路分组，检索只看对局名。

        曾经这里是 ``?game=`` 切三款游戏，标题显示的是内部 id
        （``hok-video-5052bd0c136e``）。两样都是「不明所以」：
        游戏切换在只做一款游戏之后成了多余的分叉；而内部 id 不是人读
        的东西 —— 玩家要找的是「昨天那局打野」，不是一串十六进制。

        所以现在标题由「分路·英雄·胜负」现算，检索也只认这个名字。

        列表按**设备**过滤（不是固定的 "local"）：公网部署时
        开着隔离就各看各的，没开则仍是本机那一份。
        """
        with session_scope() as s:
            records = match_col.list_matches(
                s, player_id=device.device_id(request), game=GAME, limit=200
            )
            rows: List[Dict[str, Any]] = []
            for r in records:
                payload = r.get_payload() or {}
                ident = organize.match_identity(payload, adapter)
                rows.append({
                    "id": r.id,
                    "name": ident["name"],
                    "lane": ident["lane"],
                    "lane_label": ident["lane_label"],
                    "hero": ident["hero"],
                    "result_label": ident["result_label"],
                    "played_at": r.played_at.strftime("%Y-%m-%d %H:%M")
                    if r.played_at else "—",
                    "played_sort": r.played_at.isoformat()
                    if r.played_at else "",
                    "source": "视频" if r.source_kind == "video" else "手工",
                    # 这里没有执行度：它是判定口径，不是对局事实 ——
                    # 同一局换个规则版本数字就变，摆在列表上会让人
                    # 以为这个分数是这局游戏本身打出来的。
                    # 录过 BP 的对局给「BP 复盘」入口，没录的给「录 BP」——
                    # 少一次「点进去发现没东西」的往返。
                    "has_bp": bool(payload.get("bp")),
                })
            total = len(rows)

        # 检索只看**对局名**。曾经这里还能按游戏筛、按版本筛，
        # 三个条件叠在一起时用户会怀疑「为什么搜不到」——
        # 因为他搜的是版本号而不是名字。
        keyword = q.strip().lower()
        if keyword:
            rows = [r for r in rows if keyword in r["name"].lower()]

        groups = organize.group_matches_by_lane(
            rows, adapter.position_order, adapter.position_labels
        )
        lane_labels = {
            code: adapter.position_labels.get(code, code)
            for code in adapter.position_order
        }
        active = lane if lane in lane_labels else ""

        return HTMLResponse(render_page(
            "matches.html",
            rows=rows, total=total, game=GAME,
            groups=groups, lane_labels=lane_labels,
            active_lane=active, q=q.strip(),
        ))

    # ================================================================ 规则库
    @app.get("/rules", response_class=HTMLResponse)
    def rules_page(
        version: str = Query(""),
        lane: str = Query(""),
        hero: str = Query(""),
        bp: str = Query(""),
    ) -> HTMLResponse:
        """操作模式库：分路 → 英雄 → 模式。

        三层是这么来的：

        1. 先选**分路**。攻略讲的就是「谁在什么时候该做什么」，
           混在一条列表里看不出每条分路的打法全貌。
        2. 分路下先给**该分路前中后期的总体模式**，再列**英雄** ——
           英雄层是可选的深入，不是必答项。
        3. 选了英雄，给这个英雄前中后期对应的模式。

        英雄只列**确实有模式**的那些：给一个没有模式的英雄开一节，
        点进去是空的。

        BP 那一块是**另一个库**（版本 ``BP-S45``）：BP 在比赛之前，
        按分路组织不了，所以它自己一节，坐标是「哪个环节」。
        """
        from ..rules.store import BP_VERSION, latest_version, list_versions

        with session_scope() as s:
            available = list_versions(s, GAME)
            # 版本下拉只管**局内库**。BP 是旁支（另一个版本号），
            # 混进下拉的话，选中它会得到一个「一条模式都没有」的页面 ——
            # BP 模式在下面那块自己的入口里。
            in_game_versions = [
                v for v in available if not str(v).startswith("BP-")
            ]
            if not in_game_versions:
                version = adapter.default_version
            elif version in in_game_versions:
                pass
            else:
                version = latest_version(s, GAME) or in_game_versions[0]
            view = build_review_view(s, GAME, version=version, with_coverage=False)

            # BP 库：另一个版本号，自己一节。
            # 覆盖率对 BP 没有意义（BP 规则不在局内回合上被检验），
            # 所以不算它 —— 硬算出来的是「从未触发」，看着像坏了。
            bp_view = build_review_view(
                s, GAME, version=BP_VERSION, with_coverage=False
            )
            bp_modes = organize.build_bp_modes(bp_view["rules"], adapter)

        tree = organize.build_mode_tree(view["rules"], adapter)

        lane_node = organize.find_lane_node(tree, lane) if lane else None
        hero_node = None
        if lane_node and hero:
            for h in lane_node["heroes"]:
                if h["name"] == hero:
                    hero_node = h
                    break

        return HTMLResponse(render_page(
            "rules.html",
            game=GAME,
            version=version,
            versions=in_game_versions or [adapter.default_version],
            tree=tree,
            lane_node=lane_node,
            hero_node=hero_node,
            hero=hero,
            lane=lane,
            # BP 那一节。bp=1 时直接进 BP 库。
            show_bp=bool(bp),
            bp_modes=bp_modes,
            bp_phase_count=len({
                m["phase"] for m in bp_modes
            }),
            bp_phase_labels=dict(getattr(adapter, "bp_phase_labels", {}) or {}),
            fact_labels=dict(adapter.fact_labels),
            lane_labels={
                code: adapter.position_labels.get(code, code)
                for code in adapter.position_order
            },
            matchup_labels=dict(getattr(adapter, "matchup_labels", {}) or {}),
        ))
