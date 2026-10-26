"""页内弹层：确认与提醒不用浏览器的 confirm() / alert()。

盯三件事：
1. **不许有原生弹窗** —— 系统那个黑底白字的小框跟整页的浅紫底、圆角卡片
   完全不像，而且只能塞一行字，说不清「删的是哪一局、删了会怎样」。
2. **危险动作的焦点** —— 删一局是不可逆的，默认焦点必须在「再想想」上：
   手一按回车不该把局删掉。
3. **正文不走 innerHTML** —— 对局名是用户自己起的，可能带尖括号。
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _tpl(name: str) -> str:
    return (APP_ROOT / "app" / "templates" / name).read_text("utf-8")


def test_no_native_confirm_anywhere_in_the_pages():
    """页面里不许再有 confirm() / alert()。"""
    offenders = []
    for path in sorted((APP_ROOT / "app" / "templates").glob("*.html")):
        text = path.read_text("utf-8")
        # 注释里提到不算：// 行注释、{# #} 模板注释
        text = re.sub(r"\{#.*?#\}", " ", text, flags=re.S)
        for i, line in enumerate(text.splitlines(), 1):
            code = line.split("//", 1)[0]
            for bad in ("confirm(", "alert("):
                if bad in code:
                    offenders.append(f"{path.name}:{i} {line.strip()[:70]}")
    assert not offenders, "还有原生弹窗：\n  " + "\n  ".join(offenders)


def test_the_delete_confirmation_asks_in_page():
    """删除对局要用页内弹层，并且说清「删了找不回来」。"""
    tpl = _tpl("matches.html")
    assert 'include "_dialog.html"' in tpl, "对局记录页没接页内弹层"
    assert "askConfirm" in tpl, "删除确认没走页内弹层"
    assert "askNotice" in tpl, "删除失败提示还是原生 alert"
    assert "找不回来" in tpl, "确认里没说明删了找不回来"
    assert 'asset(\'dialog.css\')' in tpl, "没引弹层样式"
    assert 'asset(\'dialog.js\')' in tpl, "没引弹层脚本"


def test_the_dialog_component_is_shared():
    """弹层是全站共用的一个组件，不是每页各写一份。"""
    tpl = _tpl("_dialog.html")
    assert 'role="dialog"' in tpl and 'aria-modal="true"' in tpl
    assert 'id="dlg"' in tpl
    for page in ("matches.html", "index.html"):
        assert 'include "_dialog.html"' in _tpl(page), f"{page} 没用共享弹层"

    js = (APP_ROOT / "app" / "static" / "dialog.js").read_text("utf-8")
    assert "askConfirm" in js and "askNotice" in js
    # 危险动作默认把焦点放在「再想想」上
    assert "danger" in js
    assert "Escape" in js, "Esc 关闭没做"


def test_the_dialog_never_injects_html():
    """正文默认走 textContent —— 对局名是用户自己起的。"""
    js = (APP_ROOT / "app" / "static" / "dialog.js").read_text("utf-8")
    assert "innerHTML" not in js, "弹层正文用了 innerHTML"
    assert "textContent" in js


def test_the_upload_page_uses_the_shared_dialog():
    """上传确认也走同一套弹层。"""
    tpl = _tpl("index.html")
    assert 'include "_dialog.html"' in tpl
    assert "askConfirm" in tpl
    assert "confirm(" not in tpl
    # 旧的 .ask 组件不该再留着（两套样式以后会改漏）
    assert ".ask-card" not in tpl
    css = (APP_ROOT / "app" / "static" / "mascot.css").read_text("utf-8")
    assert ".ask-card" not in css, "mascot.css 里还留着旧的弹层样式"
