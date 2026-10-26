"""报告导出：把网页报告变成**一个能留在自己电脑上的文件**。

为什么要有这个：
    报告是网页 —— 数据在服务器上，样式也指着服务器。
    用户真正想要的却是「把这一局的分析留在我自己电脑里」：
    发给队友看、存档、离线翻。
    所以导出物必须**自包含**：样式内联、脚本内联，
    双击就能打开，不依赖服务器还在不在。

    两个入口：局内操作模式分析报告、BP 复盘报告。

有一件事导出物做不到：**视频片段**。
    片段是服务器上的文件，离线打开时播不了。
    导出页顶部会写明这一点 —— 不假装它能播，
    也不把片段悄悄删掉（那会让人以为报告里本来就没视频）。
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Dict, Optional
from urllib.parse import quote

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

#: 样式表 / 脚本的引用。渲染出来的页面里长这样：
#:   <link rel="stylesheet" href="/static/report.css?v=1758000000">
#:   <script src="/static/report.js?v=1758000000"></script>
_LINK_RE = re.compile(r'<link\b[^>]*\bhref="([^"]+)"[^>]*>', re.I)
_SCRIPT_RE = re.compile(r'<script\b[^>]*\bsrc="([^"]+)"[^>]*>\s*</script>', re.I)

#: 文件名里不能有的字符（Windows 最严）。
_ILLEGAL = re.compile(r'[\\/:*?"<>|\r\n\t]+')

_EXPORT_NOTE_STYLE = (
    "margin:0;padding:10px 18px;font-size:13px;line-height:1.7;"
    "background:#111426;color:#b9c3e6;border-bottom:1px solid #2a3152;"
)


def _asset_file(url: str) -> Optional[str]:
    """把静态资源地址解析成 ``app/static/`` 下的文件名。

    ``/static/report.css?v=123`` → ``report.css``。
    只认同目录下的文件 —— 别的域名的东西一律不内联。
    """
    path = url.split("?", 1)[0].split("#", 1)[0]
    if "/static/" not in path:
        return None
    name = path.rsplit("/", 1)[-1]
    if not name or "/" in name or "\\" in name:
        return None
    full = os.path.join(STATIC_DIR, name)
    return full if os.path.isfile(full) else None


def inline_assets(html: str) -> str:
    """把页面引用的 CSS/JS 换成内联内容。

    内联失败（文件不在、路径异常）就**原样保留那个标签** ——
    宁可样式不对，也不能让整页白掉。
    """

    def link_repl(match: "re.Match[str]") -> str:
        tag = match.group(0)
        full = _asset_file(match.group(1))
        if not full:
            return tag
        try:
            with open(full, "r", encoding="utf-8") as fh:
                css = fh.read()
        except OSError:
            return tag
        return "<style>\n%s\n</style>" % css

    def script_repl(match: "re.Match[str]") -> str:
        tag = match.group(0)
        full = _asset_file(match.group(1))
        if not full:
            return tag
        try:
            with open(full, "r", encoding="utf-8") as fh:
                js = fh.read()
        except OSError:
            return tag
        # JS 里出现 </script> 会提前关掉标签，脚本就被截断了
        js = js.replace("</script>", "<\\/script>")
        return "<script>\n%s\n</script>" % js

    html = _LINK_RE.sub(link_repl, html)
    html = _SCRIPT_RE.sub(script_repl, html)
    return html


#: 导出物是**快照**，不是活会话 —— 里面不该留着「页面关了就通知服务器」
#: 的那段脚本（导出的文件在本地 file:// 下打开，那个请求只会失败）。
_LIVE_ONLY_RE = re.compile(
    r"<script>.*?(?:sendBeacon|/api/session/policy).*?</script>", re.S
)


def build_export_page(
    html: str,
    *,
    title: str = "",
    stamp: str = "",
    note: str = "",
) -> str:
    """把渲染好的报告页改成「可离线打开的单文件」。

    顶部加一条说明：什么时候导出的、离线能看什么、不能看什么。
    没有这条说明，收到文件的人会以为是自己刚截的图。
    """
    html = inline_assets(html)
    html = _LIVE_ONLY_RE.sub("", html)

    generated = stamp or datetime.now().strftime("%Y-%m-%d %H:%M")
    banner_text = "本文件于 %s 从网页导出，可离线打开。" % generated
    if title:
        banner_text = "「%s」于 %s 从网页导出，可离线打开。" % (title, generated)
    if note:
        banner_text += " " + note

    banner = (
        '<div style="%s">%s</div>' % (_EXPORT_NOTE_STYLE, banner_text)
    )

    # 插到 <body> 之后，正文前面
    idx = html.lower().find("<body")
    if idx == -1:
        return banner + html
    end = html.find(">", idx)
    if end == -1:
        return banner + html
    return html[: end + 1] + "\n" + banner + html[end + 1:]


def safe_filename(text: str, max_len: int = 80) -> str:
    """把标题变成一个能存下来的文件名。"""
    text = _ILLEGAL.sub("_", (text or "").strip())
    text = text.replace(" ", "_")
    text = re.sub(r"_+", "_", text).strip("._")
    if len(text) > max_len:
        text = text[:max_len]
    return text or "report"


def export_filename(title: str, stamp: str = "", suffix: str = "html") -> str:
    """报告的文件名：``王者荣耀_操作模式分析_2026-10-22_1830.html``。"""
    parts = [p for p in (title, stamp) if p]
    base = safe_filename("_".join(parts))
    return "%s.%s" % (base, suffix)


def attachment_headers(filename: str) -> Dict[str, str]:
    """让浏览器把响应存成文件，而不是显示在页面里。

    中文文件名要靠 ``filename*=UTF-8''``（RFC 5987），
    同时给一个纯 ASCII 的 ``filename=`` 兜底 ——
    老浏览器只认后者，拿到乱码名字也比拿不到文件强。
    """
    ascii_name = re.sub(r"[^\x20-\x7e]", "_", filename).replace('"', "_") or "report"
    return {
        "Content-Disposition": (
            'attachment; filename="%s"; filename*=UTF-8\'\'%s'
            % (ascii_name, quote(filename))
        )
    }


def export_json(report: Dict[str, Any]) -> str:
    """报告的原始数据（存档用）。"""
    import json

    return json.dumps(report, ensure_ascii=False, indent=2, default=str)
