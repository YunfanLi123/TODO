"""上传页与上传接口的接线测试。

盯的是「版本从哪来」这一件事。版本填错不会抛异常 —— 上传成功、扫描成功、
报告页面也打开，只是里面一条规则都没有。所以这条链路出错的形态永远是
**静默的空报告**，只能靠断言钉住。

版本现在由后端取该游戏规则库里最新的那一个，不接受前端传值：
少一个能填错的地方，比在界面上加提示有用。
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.adapters import registry as _registry  # noqa: F401  触发适配器注册

APP_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def client(tmp_path):
    """独立临时库的 TestClient。

    独立库是硬要求：规则库状态直接决定上传接口是 200 还是 409，
    共享主库的话这条断言的成败取决于「上一轮跑过什么」。
    """
    from fastapi.testclient import TestClient

    from app import db as dbmod
    from app.main import app

    dbmod._engine = None
    dbmod._SessionFactory = None
    dbmod.get_engine(str(tmp_path / "upload_page.db"))

    yield TestClient(app)

    dbmod._engine = None
    dbmod._SessionFactory = None


# ---------------------------------------------------------------- 首页表单


def test_upload_page_has_no_game_picker(client):
    """上传页不再有游戏下拉 —— 这个产品现在只做王者荣耀。

    早先这里有三个选项（云顶 / 金铲铲 / 王者），而只有第三个用得上。
    留着它的坏处不是「多一个选项」，是让人以为选错游戏会拿到另一个
    游戏的规则库 —— 而实际上另外两款连规则库都没有。
    """
    html = client.get("/upload").text
    assert 'id="game"' not in html, "上传页不该再有游戏下拉"
    assert "云顶" not in html and "金铲铲" not in html, \
        "上传页不该再出现另外两款游戏"
    # 前端固定按王者荣耀上传
    assert "fd.append('game', 'hok')" in html, \
        "上传时游戏没有固定成王者荣耀"


def test_upload_page_has_no_version_control(client):
    """版本不该让人填。"""
    html = client.get("/upload").text
    assert 'id="ver"' not in html, "上传页仍有版本控件"
    assert "game_version" not in html, "前端不该再发 game_version"
    # 页面里也不该剩下版本输入的痕迹
    assert not re.search(r"<span>版本</span>", html)


def test_workbench_page_is_only_the_upload(client):
    """工作台只剩上传本身。

    页面上原来还有一排彩色数字卡（对局数 / 模式数 / 技术报告数），
    而侧边栏就有同样的三个入口 —— 同一件事在两处各说一次，数字还可能
    对不上（统计口径不同）。空出来的位置让上传模块占满。
    """
    html = client.get("/upload").text
    assert "对局复盘系统" in html, "工作台标题不对"
    assert "上传对局视频" in html
    for gone in ("stat-row", "stat-card", "工作台"):
        assert gone not in html, f"工作台页面上还留着 {gone}"
    # 侧边栏留「上传」「对局」两个入口。
    for href in ("/upload", "/matches"):
        assert f'href="{href}"' in html, f"侧边栏少了 {href}"
    # 模式库**不在侧边栏**：它只有一个入口 —— 封面第三页那排「进入XX库」的卡。
    # 内层再挂一个，等于把它说成一个随手可翻的功能，而它其实是一份
    # 要挑着看的清单。
    assert 'href="/rules"' not in html, "侧边栏还有模式库入口"


def test_app_shell_has_no_brand_square(client):
    """左上角那个方块去掉了，换成产品名。

    那个方块不是 logo、不是开关、不是状态指示器，却占着整个产品
    第一个视线落点 —— 用户会以为它是能点的。
    """
    html = client.get("/upload").text
    assert 'class="brand-mark"' not in html, "左上角那个方块还在"
    assert "brand-word" in html, "左上角没有换成文字"


# ---------------------------------------------------------------- 上传接口


def test_upload_endpoint_accepts_no_game_version_field(client):
    """接口层面就不该再有 game_version 表单字段。"""
    schema = client.get("/openapi.json").json()
    body = schema["paths"]["/api/videos/upload"]["post"]["requestBody"]
    props = schema["components"]["schemas"]["Body_upload_video_api_videos_upload_post"]
    assert "game_version" not in props["properties"], \
        "接口还在接受调用方传的版本"
    assert body  # 顺带确认这个接口还在


def test_upload_is_rejected_when_no_mode_library(client):
    """操作模式库一个版本都没有时，当场拒掉上传。

    版本解析不到时如果放行，用户会白等几分钟扫完视频，最后撞上的还是
    「没有操作模式库」。不如上传这一秒就说清楚。
    """
    r = client.post(
        "/api/videos/upload",
        files={"file": ("x.mp4", b"\x00" * 32, "video/mp4")},
        data={"game": "hok"},
    )
    assert r.status_code == 409, f"期望 409，实际 {r.status_code}：{r.text[:200]}"
    assert "操作模式库" in r.json()["detail"]


def test_root_is_the_landing_page(client):
    """``/`` 是封面，不是上传表单。

    上传表单挪到了 ``/upload``。少这一步的后果是封面上的「探索玩法」点了
    直接跳回上传框 —— 那一屏存在的意义就是先说清楚再让人进来。
    """
    html = client.get("/").text
    assert "不断向被验证的模式优化拟合" in html
    assert "探索玩法" in html
    assert 'href="/upload"' in html
    # 封面不该混进上传表单的东西
    assert 'id="game"' not in html, "上传表单不该出现在封面上"
    assert 'id="go"' not in html
    # 「成为高手」已经改掉了，别再回流
    assert "成为高手" not in html
    # 封面下面还有两页内容，所以底部有滚动提示。
    # 早先这里断言「不许有滚动提示」—— 那是封面还是单屏时的决定，
    # 现在封面要往下滚了，提示必须回来，而且必须是**能点的锚**，
    # 不是一段没用的装饰文字。
    assert "scroll-out" in html, "封面底部没有滚动提示"
    assert 'href="#page-2"' in html, "滚动提示没指向第二页"
    assert 'id="page-2"' in html and 'id="page-3"' in html, "封面下面没接上两页内容"


def test_landing_brand_is_sufe_scai_todo(client):
    """左上角是项目名文字，不是那个占位圆环。

    早先画的是纯 CSS 的环（``brand-mark``），等真 logo 位图资产就位。
    现在直接写 ``SUFE_SCAI_TODO`` —— 环去掉了，名字的位置留着。
    """
    html = client.get("/").text
    assert "SUFE_SCAI_TODO" in html
    assert 'class="brand-mark"' not in html, "占位圆环已经去掉了"
    assert ">SU<" not in html, "占位圆环里的 SU 也该一起去掉"


def test_landing_background_is_a_real_galaxy_asset(client):
    """封面背景必须是**真的能打开的银河资源**，不是一句承诺。

    这条替代了原先盯 canvas 流场的断言 —— 背景已换成第三方开源项目
    galaxy-explorer 的 WebGL 银河，走 iframe 引入。

    风险换了：从「画得像不像」变成「资源在不在」。iframe 指向静态文件，
    一旦打包漏掉 vendor 目录，页面照样渲染得漂漂亮亮，
    只是背景是空的，而且不会有任何报错提示。所以这里把**每个依赖文件**
    都真的 GET 一遍。
    """
    html = client.get("/").text
    assert 'class="galaxy-bg"' in html
    assert 'src="/static/galaxy/index.html"' in html

    r = client.get("/static/galaxy/index.html")
    assert r.status_code == 200, "银河页面取不到，背景会是空白"
    page = r.text
    assert "<canvas" in page, "页面里没有 canvas"
    # 演示 UI 要藏起来，只留满屏银河
    assert "body > *:not(#cv)" in page

    # 银河自己依赖 three.js 与后期处理链，少一个都是黑屏或报错
    for dep in re.findall(r'src="([^"]*vendor/[^"]+)"', page):
        assert client.get("/static/galaxy/" + dep).status_code == 200, \
            f"缺依赖 {dep}，银河渲染不出来"


def test_landing_credits_the_third_party_galaxy(client):
    """抄来的代码必须留出处，否则日后没人知道这段是哪来的。"""
    page = client.get("/static/galaxy/index.html").text
    assert "galaxy-explorer" in page
    assert "MIT" in page
    assert "Justin Zhang Jun" in page


def test_upload_route_is_the_form(client):
    r = client.get("/upload")
    assert r.status_code == 200
    assert 'id="go"' in r.text
    # 拖放区现在就是梦奇站的那块框（id=stage）——
    # 早先叫 "drop"，改名字是因为它不再是一个「放文件的区」，
    # 而是角色的地盘。
    assert 'id="stage"' in r.text
    assert "上传并扫描" in r.text


def test_landing_has_no_placeholder_copy(client):
    """封面不摆占位文案。没信息就不写，别留占位让人以为功能没做完。

    项目名 ``SUFE_SCAI_TODO`` 本身带 TODO，是**指定的品牌写法**，
    所以这里只查它之外的部分 —— 不能因为名字里有个 TODO 就放过
    真正没写完的地方。
    """
    html = client.get("/").text
    body = html.replace("SUFE_SCAI_TODO", "")
    assert "TODO" not in body
    assert "金凡" not in html, "别把参考站截图里的用户名搬进来"


def test_landing_honours_reduced_motion(client):
    """开了「减少动态效果」时，银河必须停下来。

    这不是锦上添花：一个每秒 60 次重绘的满屏动画，在前庭敏感者眼里
    会引发眩晕，而页面上没有任何提示。停掉之后画面仍然成立 ——
    因为布局本身不依赖运动。

    动画现在跑在 iframe 里，所以偏好要在**银河那一页**里读，
    只在 landing 检查是不够的。
    """
    html = client.get("/").text
    assert "prefers-reduced-motion: reduce" in html

    page = client.get("/static/galaxy/index.html").text
    assert "matchMedia('(prefers-reduced-motion: reduce)')" in page, \
        "银河页没有读这个偏好，设置里关不掉动画"


def test_app_shell_wraps_without_a_media_query():
    """外壳必须是 Sidebar 原语（flex-wrap），不能是固定定位。

    之前用的是 ``position: fixed`` + ``margin-left``：窄屏下左栏永远
    压着内容，横向滚动条拉到底也看不到正文 —— 那是媒体查询写死的布局，
    不是布局本身在响应。
    """
    css = (APP_ROOT / "app" / "static" / "app.css").read_text("utf-8")
    # 外壳这一段：.app-body（容器）到 .annotate-top（页面自己的上下文条）
    shell = css[css.index(".app-body {"):css.index(".annotate-top {")]
    # 注释里也会出现这些词，查之前先剥掉 —— 否则「没加媒体查询」这句
    # 注释自己就把断言绊倒了
    shell = re.sub(r"/\*.*?\*/", "", shell, flags=re.S)

    assert "flex-wrap: wrap" in shell, "外壳没有用 flex-wrap"
    assert "position: fixed" not in shell, \
        "外壳还在用固定定位，窄屏不会换行"
    assert "flex-basis: var(--side-width" in shell
    assert "min-inline-size: var(--content-min" in shell

    # 一个 @media 都不该出现在外壳里 —— 容器宽度才是正确的信号
    assert "@media" not in shell, "外壳里出现了媒体查询"


def test_layout_tokens_are_defined_once():
    """间距与 measure 是全局公理，不是每页各写一遍。"""
    css = (APP_ROOT / "app" / "static" / "report.css").read_text("utf-8")
    assert "--measure: 60ch" in css
    for token in ("--space-s:", "--space-m:", "--space-l:", "--side-width:",
                  "--content-min:"):
        assert token in css, f"缺少布局变量 {token}"


def test_match_api_carries_the_game_clock(client):
    """局内时间必须能通过 API 录进来。

    少了这一句，pydantic 会把 ``time_min`` **静默丢掉** —— 接口返回 200、
    库也建好了，而适配器随后把 time_min 标成缺失，所有带时间窗的模式在缺失事实上一律判不成立。表现是「报告一条结论都没有」，且毫无报错。
    """
    client.post("/api/games/hok/rules/seed")
    r = client.post("/api/matches", json={
        "match_id": "clock-1", "game": "hok", "game_version": "S45",
        "player": {"nickname": "示例玩家", "position": "side",
                   "hero": "哪吒", "matchup": "disadvantage"},
        "outcome": {"result": "lose"}, "played_at": "2026-10-18 21:00",
        "rounds": [{
            "stage": 1, "phase": 1, "time_min": 2,
            "facts": {"gold": 1800, "gold_diff": 200, "level": 4,
                      "tower_diff": 0, "kill_diff": 0},
            "actions": [{"kind": "positioning_action", "value": "push_wave",
                         "label": "推线压塔", "confidence": 1.0}],
            "missing_facts": [],
        }],
    })
    assert r.status_code == 200, r.text[:300]
    assert r.json()["stats"]["triggered"] > 0, \
        "局内时间没进去，所有带时间窗的模式都不会触发"

    # 玩家填了英雄（哪吒），所以走的是**英雄层**：哪吒自己的对抗路打法。
    # 分路通则同时让路 —— 否则同一件事会被说两遍。
    ids = {f["mode_id"] for f in r.json()["lanes"]["findings"]}
    assert any(i.startswith("HOK-HERO-SID-NZ-") for i in ids), ids
    assert "HOK-MODE-SID-T0200-02" not in ids, \
        "选了英雄还在判分路通则，同一条建议会被说两遍"
    assert "HOK-MODE-SID-T0200-01" not in ids, \
        "劣势对位时优势分支也生效了，两条会互相指控"


def test_upload_takes_no_scan_sensitivity_fields(client):
    """扫描灵敏度不再由调用方给。

    界面曾经是两个滑块（边界阈值 / 最短回合间隔）。让人自己拖只会得到
    更差的切片：调高漏回合，调低把一次转场切成好几段 —— 而这种错误在
    报告里看不出来，只会让后面所有模式的判定都建立在错误的回合切分上。
    """
    schema = client.get("/openapi.json").json()
    props = schema["components"]["schemas"]["Body_upload_video_api_videos_upload_post"]
    names = set(props["properties"])
    assert names == {"file", "game"}, f"上传接口的字段不对：{sorted(names)}"
    for gone in ("diff_threshold", "min_gap", "game_version"):
        assert gone not in names, f"{gone} 还留在上传接口上"


def test_unknown_game_is_still_rejected(client):
    """版本解析之前先判游戏名，未知游戏仍应是 422。"""
    r = client.post(
        "/api/videos/upload",
        files={"file": ("x.mp4", b"\x00" * 32, "video/mp4")},
        data={"game": "dota"},
    )
    assert r.status_code == 422
