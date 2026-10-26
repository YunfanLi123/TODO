"""封面（落地页）的内容约束。

封面是这个项目的门面：别人第一次打开就看这一屏。所以它上面写的每一句
都要**是写好的**，链接都要**真的能进去**。

这里盯三件事：

1. **第二页的三个框** —— 大标题不变，框里只有那三段话，不夹带别的；
2. **第三页的模式库入口** —— 一个库一张卡，标题白色、描述在、
   最后一行是能点进去的「进入XX库」；
3. **横排** —— 一页放三个半框，箭头能把藏起来的框带出来。
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

APP_ROOT = Path(__file__).resolve().parent.parent
LANDING = APP_ROOT / "app" / "templates" / "landing.html"


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from app import db as dbmod
    from app.main import app

    dbmod._engine = None
    dbmod._SessionFactory = None
    dbmod.get_engine(str(tmp_path / "landing.db"))

    yield TestClient(app)

    dbmod._engine = None
    dbmod._SessionFactory = None


#: 第二页三个框：标题 + 正文（按提示词逐字）。
PAGE2 = [
    ("定义什么是好模式",
     "模式如何被定义，决定玩法在哪些方向上被丰富和优化，也决定时间最终流向哪里。"),
    ("定义什么是好标准",
     "可信的模式不仅关注来源是否真实，更关注依据是否充分、推理是否完整、边界是否清晰。"),
    ("持续丰富玩法与操作模式",
     "操作模式库不会一蹴而就，而是在持续迭代中不断演进。"),
]


def test_page2_keeps_its_big_title_and_only_three_boxes(client):
    """第二页：大标题不变，框里只有那三段话。"""
    html = client.get("/").text
    assert "从路径依赖到拓展更多玩法可能" in html, "第二页的大标题被改了"

    for title, body in PAGE2:
        assert title in html, f"第二页少了「{title}」"
        assert body in html.replace("\n", "").replace("  ", "") or body in html, \
            f"第二页的「{title}」正文不对"

    # 框里不再夹带别的（旧版有「内核不动」这类小注脚）
    for gone in ("内核不动", "口径不动", "标准不动"):
        assert gone not in html, f"第二页的框里还留着「{gone}」"


def test_page3_is_the_mode_library_entrance(client):
    """第三页：一个库一张卡，最后一行是能点进去的入口。"""
    html = client.get("/").text
    assert "我们共建真正有价值的攻略数据库" in html, "第三页的大标题被改了"
    assert "集思广益，一起挑战操作模式的边界" in html, "第三页的副标题不对"

    libs = [
        ("BP", "/rules?bp=1", "进入BP库"),
        ("对抗路", "/rules?lane=side", "进入对抗路库"),
        ("打野", "/rules?lane=jungle", "进入打野库"),
        ("中路", "/rules?lane=mid", "进入中路库"),
        ("发育路", "/rules?lane=carry", "进入发育路库"),
        ("游走", "/rules?lane=roam", "进入游走库"),
    ]
    for name, href, entry in libs:
        assert f'href="{href}"' in html, f"{name} 库的入口不在"
        assert entry in html, f"缺少「{entry}」"
        assert f">{name}</h3>" in html, f"缺 {name} 这一张卡"

    # 描述要真的写了字，不能是空框
    for name, _, _ in libs:
        block = html.split(f">{name}</h3>", 1)[1].split("</article>", 1)[0]
        text = re.sub(r"<[^>]+>", "", block)
        assert len(text.strip()) > 20, f"{name} 那张卡没有描述"


def test_the_library_cards_have_white_titles(client):
    """框的标题是白色的。"""
    css = LANDING.read_text("utf-8")
    m = re.search(r"\.rail \.card h3\s*\{([^}]*)\}", css)
    assert m, "横排的卡片标题没有单独定样式"
    assert "color:#fff" in m.group(1).replace(" ", ""), "框的标题不是白色"


def test_the_rail_shows_three_and_a_half_cards(client):
    """一页放三个半框 —— 半个框露在外面，是「右边还有」的提示。"""
    css = LANDING.read_text("utf-8")
    m = re.search(r"\.rail \.card\s*\{([^}]*)\}", css)
    assert m, "横排卡片没有宽度规则"
    rule = m.group(1).replace(" ", "")
    assert "flex:00calc((100%-3*18px)/3.5)" in rule, \
        "卡宽不是「三个半框」"

    # 箭头：能把藏起来的框带出来，滑到头就藏起来
    assert 'id="railNext"' in LANDING.read_text("utf-8"), "没有往右看的箭头"
    html = client.get("/").text
    assert 'id="railNext"' in html
    assert "rail-next" in css


def test_cards_glow_on_hover(client):
    """静止时边框微微发光；鼠标压上去时渐变光出现、框微微浮起。"""
    css = LANDING.read_text("utf-8")
    assert ".card::after" in css, "卡片没有光晕层"
    assert "mask-composite" in css, "光晕层不是渐变边框的做法"
    assert ".card:hover" in css, "卡片没有悬停态"
    hover = re.search(r"\.card:hover\s*\{([^}]*)\}", css)
    assert hover, "卡片没有悬停规则"
    assert "translateY" in hover.group(1), "悬停时框没有浮起"


def test_the_page_titles_are_gradient(client):
    """第二、三页的大标题是渐变的，不是纯白。

    纯白的大字落在深底上太平了；渐变才有质感，
    两屏也就成了同一个东西的两页。
    """
    css = LANDING.read_text("utf-8")
    m = re.search(r"\.page h2\s*\{([^}]*)\}", css)
    assert m, "第二三页的大标题没有样式"
    rule = m.group(1)
    assert "linear-gradient" in rule, "大标题没有渐变"
    assert "background-clip: text" in rule.replace("-webkit-background-clip", "background-clip"), \
        "渐变没有裁到字里"
    assert "text-fill-color: transparent" in rule.replace("-webkit-", ""), \
        "渐变没有作用在字上（会变成一个渐变方块）"


def test_the_rail_has_both_arrows(client):
    """左右两个箭头都要有 —— 只有右边那个的话，滑到头就回不去了。"""
    html = client.get("/").text
    assert 'id="railNext"' in html, "没有往右的箭头"
    assert 'id="railPrev"' in html, "没有往左的箭头"

    css = LANDING.read_text("utf-8")
    assert ".rail-prev" in css and ".rail-next" in css
    # 两端的渐隐跟着滚动出现：没滑到头时左边是干净的
    assert "at-start" in css and "at-end" in css
    assert "railPrev" in LANDING.read_text("utf-8"), "往左的箭头没有绑事件"


def test_the_cover_has_no_function_entries_at_the_top_right(client):
    """封面右上角不摆功能入口。

    封面是门面，不是导航条：摆两排入口，看着就像功能清单。
    模式库的入口在第三页那排「进入XX库」的卡 —— 想看的人往下滚。
    """
    html = client.get("/").text
    # 只看顶栏那块标记 —— 页面里别处（第三页的卡）本来就该有「进入XX库」。
    topbar = html.split('<header class="topbar"', 1)[1].split("</header>", 1)[0]
    for gone in ("对局记录", "模式库"):
        assert gone not in topbar, f"封面右上角还有「{gone}」"
    for href in ("/matches", "/rules"):
        assert f'href="{href}"' not in topbar, f"封面右上角还有 {href} 的入口"


def test_the_inner_pages_carry_the_atmosphere_layer(client):
    """内层页面的背景是氛围层（缓慢漂移的竖向光褶），不是一块死底色。

    只加背景：版式、字体、内容一个字不动。
    """
    html = client.get("/matches").text
    assert 'id="atmo-canvas"' in html, "内层页面没有氛围背景层"
    assert "atmosphere.js" in html, "氛围层没挂上脚本"
    assert "atmosphere.css" in html, "氛围层没挂上样式"


def test_the_atmosphere_respects_reduced_motion():
    """开了「减少动态效果」就只画一帧，不动。

    满屏动画在前庭敏感者眼里会引发眩晕，而页面上没有任何提示。
    """
    js = (APP_ROOT / "app" / "static" / "atmosphere.js").read_text("utf-8")
    assert "prefers-reduced-motion" in js
    assert "visibilitychange" in js, "标签页藏起来没有停 —— 后台白烧电"
    assert "requestAnimationFrame" in js

