"""报告渲染：把 report dict 交给 Jinja2 模板输出 HTML。

单独一层是为了让「数据组装」和「排版」互不干扰 ——
builder 产出结构化报告，views 只负责把它变成一张页面。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import organize

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(["html", "xml"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


def asset_url(name: str) -> str:
    """静态资源地址，带一个版本号（文件修改时间）。

    浏览器会把 CSS 缓存住。改完样式刷新页面看到的还是老样子 ——
    这种「代码明明改了却没生效」最费时间，因为要怀疑自己写的选择器。

    加一个 mtime 当查询参数：文件一改，地址就变，浏览器自然重新取。
    打包成 exe 之后也一样管用 —— 静态文件在包里，mtime 跟着打包走。
    """
    rel = name.lstrip("/")
    try:
        stamp = int(os.path.getmtime(os.path.join(STATIC_DIR, rel)))
    except OSError:
        stamp = 0
    return f"/static/{rel}?v={stamp}"


_env.globals["asset"] = asset_url


def segment_unit(game: str) -> str:
    """这个游戏的操作单位：MOBA 叫「时间片」，自走棋叫「回合」。

    王者荣耀的一局是连续的一整场，没有「回合」这个东西；
    页面上还写「回合」会让玩家去找回合。
    """
    try:
        from ..adapters.base import get_adapter

        return getattr(get_adapter(game), "segment_unit", None) or "回合"
    except Exception:                              # noqa: BLE001
        return "回合"


def render_report(report: Dict[str, Any], as_json: bool = False) -> str:
    """渲染报告页面。

    ``as_json=True`` 时输出原始 JSON，供前端二次开发或 API 直接取用。
    """
    if as_json:
        return json.dumps(report, ensure_ascii=False, indent=2, default=str)

    # 模板里需要序列化一小块数据给前端 JS 画曲线用
    template = _env.get_template("report.html")
    game = (report.get("meta") or {}).get("game") or ""
    return template.render(
        report=report,
        curve_json=json.dumps(report.get("curve", {}), ensure_ascii=False, default=str),
        unit=segment_unit(game),
        generated_at=report.get("meta", {}).get("generated_at")
        or datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


def render_page(template_name: str, context: Dict[str, Any] = None,
                **extra: Any) -> str:
    """渲染一个普通页面（首页、对局、版本、攻略）。

    ``context`` 里的键可能和模板全局变量同名（比如页面自己要一个
    ``game``），所以调用方既可以走 dict 也可以直接传关键字参数 ——
    后者优先级更高，不用为了绕开命名冲突去发明 ``ctx`` 这种前缀。
    """
    merged: Dict[str, Any] = {}
    if context:
        merged.update(context)
    merged.update(extra)
    return _env.get_template(template_name).render(**merged)


def render_bp_report(
    report: Dict[str, Any], match_id: str = "", standalone: bool = False
) -> str:
    """渲染 BP 复盘页。

    与局内报告共用一套主题变量（同一份 ``report.css``）—— 同一件事的
    两个阶段，看起来像同一个产品才对。

    ``match_id`` 用来摆「改 BP 记录」「看局内报告」这两个按钮 ——
    没有它，复盘页就只能看，回不去。

    ``standalone=True`` 给**离线导出的单文件**用：那种文件离开了服务器，
    左栏和指向服务器的按钮全是点不动的，所以一个都不摆。
    """
    template = _env.get_template("bp_report.html")
    return template.render(
        report=report,
        match_id=match_id,
        standalone=standalone,
        generated_at=report.get("meta", {}).get("generated_at")
        or datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


def render_bp_form(record, bp: Dict[str, Any]) -> str:
    """渲染 BP 录入页。

    页面上要能直接选英雄、选环节、勾局面 ——
    BP 复盘的全部结论都来自这份录入，录起来麻烦就会没人录。
    """
    from ..adapters.base import get_adapter
    from ..adapters.honor_of_kings import (
        BP_ACTION_LABELS,
        BP_PHASE_LABELS,
        BP_PHASE_LEGACY_LABELS,
        HEROES_BY_LANE,
    )

    heroes = sorted({h for lane in HEROES_BY_LANE.values() for h in lane})
    template = _env.get_template("bp_annotate.html")
    return template.render(
        match=record,
        # 「这是哪一局」用人能认的名字（视频文件�� · 分路 · 胜负），
        # 不摆内部 id —— 那是机器认的东西。
        match_name=organize.match_display_name(
            record.get_payload() or {}, get_adapter(record.game)),
        bp=bp or {},
        # 已经录过没有 —— 决定了「查看复盘」这个按钮摆不摆。
        # 没录过也摆，点进去就是一个 404：那是点不动的按钮。
        has_bp=bool(bp),
        heroes=heroes,
        phases=BP_PHASE_LABELS,
        # 已撤下的环节（「进入 BP」那一环）只作旧记录显示用 ——
        # 页面要把老记录里那一步原样摆出来，不能让它在保存时变成别的环节。
        legacy_phases=BP_PHASE_LEGACY_LABELS,
        actions=BP_ACTION_LABELS,
        # 页脚与 BP 复盘页同款：两页是同一件事的两面，骨架得一模一样。
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


def render_review(data: Dict[str, Any]) -> str:
    """渲染规则库审核工作台。

    与报告共用一套主题变量（同一份 ``report.css``），让两个页面看起来
    属于同一个产品 —— 它们本来就是同一件事的两面：这边定规则，
    那边用规则出结论。
    """
    template = _env.get_template("review.html")
    return template.render(
        data=data,
        unit=segment_unit(data.get("game") or ""),
        rules_json=json.dumps(data.get("rules", []), ensure_ascii=False, default=str),
        summary_json=json.dumps(data.get("summary", {}), ensure_ascii=False, default=str),
        labels_json=json.dumps(
            {
                "facts": data.get("fact_labels", {}),
                "actions": data.get("action_labels", {}),
            },
            ensure_ascii=False,
            default=str,
        ),
    )
