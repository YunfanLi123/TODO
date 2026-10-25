"""上传页吉祥物的约束测试。

盯的是三件**错了也不会报错**的事：

1. **角色不能飞出框。**
   框有 ``overflow: hidden``，角色一旦被推到框外，
   耳朵会被齐刷刷切掉一道 —— 页面照常渲染，只是看着像贴图坏了。
   这里把 SVG 画出来的范围、CSS 里的盒子尺寸与升降行程
   拿来算一遍，算出来必须每一帧都在框内。

2. **脸不能退回「两个黑窟窿」。**
   眼睛必须有眼白、瞳孔和能动的眼皮；
   眼皮还得是**裁剪**出来的，不能是拿头色补的一块 ——
   补的那块跟着眼睛动，头的渐变就对不上位，
   眼睛上方会浮出一块补丁形状的浅色。

3. **借来的动作要留出处。**
   动作系统借自 GitHub 上的开源实现（janobarnard/form-friends，MIT）。
   抄来的东西必须写明出处，否则日后没人知道这段是哪来的。
"""

from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.adapters import registry as _registry  # noqa: F401  触发适配器注册

APP_ROOT = Path(__file__).resolve().parent.parent
CSS_PATH = APP_ROOT / "app" / "static" / "mascot.css"


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from app import db as dbmod
    from app.main import app

    dbmod._engine = None
    dbmod._SessionFactory = None
    dbmod.get_engine(str(tmp_path / "mascot_page.db"))

    yield TestClient(app)

    dbmod._engine = None
    dbmod._SessionFactory = None


@pytest.fixture(scope="module")
def css() -> str:
    return CSS_PATH.read_text("utf-8")


# ---------------------------------------------------------------- 工具


def _block(text: str, selector: str) -> str:
    """取一条规则的内容（按大括号配平）。"""
    m = re.search(re.escape(selector) + r"\s*\{", text)
    assert m, f"样式表里没有 {selector}"
    depth = 1
    i = m.end()
    while depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[m.end():i - 1]


def _keyframes(text: str, name: str) -> str:
    m = re.search(r"@keyframes\s+" + re.escape(name) + r"\s*\{", text)
    assert m, f"样式表里没有 @keyframes {name}"
    depth = 1
    i = m.end()
    while depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[m.end():i - 1]


def _pct(block: str, prop: str) -> float:
    m = re.search(prop + r"\s*:\s*([\d.]+)%", block)
    assert m, f"规则里没有 {prop}: N%"
    return float(m.group(1))


# ------------------------------------------------- 一、角色不许飞出框


def test_mascot_never_leaves_the_stage(css: str, client):
    """角色在整个动画周期里都必须完整落在框内。

    算的是真几何，不是「看起来好像没出去」：

    * 盒子占框宽多少（``.mascot`` 的 inline-size）；
    * 角色在 viewBox 里画了多高（SVG 上的 ``data-ms-bounds``）；
    * 泳姿那一下最多歪多少度（``msSwim`` 里最大的 rotate）——
      歪了以后上下两头的位置也跟着挪；
    * 游上蹦极的行程走到哪（``msRise`` 的 inset-block-start）。

    盒子现在还多一道约束（见
    :func:`test_the_character_is_bounded_by_the_stage_height`），
    所以下面这套算法是**偏保守**的：它按「盒子高度 = 框高的 box%」
    来算，而实际尺寸被 cqmin 压着只会更小。
    """
    html = client.get("/upload").text

    tag = re.search(r"<svg\b[^>]*>", html)
    assert tag, "上传页里没有吉祥物的 SVG"
    svg = tag.group(0)
    vb = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg)
    bounds = re.search(r'data-ms-bounds="(\d+) (\d+)"', svg)
    assert vb and bounds, \
        "SVG 上要写清 viewBox 与 data-ms-bounds（画出来的那部分有多高），否则算不了"

    vb_h = int(vb.group(2))
    top_u, bottom_u = int(bounds.group(1)), int(bounds.group(2))
    assert 0 <= top_u < bottom_u <= vb_h, "data-ms-bounds 写得不对"

    box = _pct(_block(css, ".mascot"), "inline-size")
    assert box > 0 and box < 100

    # 泳姿最多歪多少度 —— 歪了以后上下两头往外挪
    swim = _keyframes(css, "msSwim")
    rots = [abs(float(x)) for x in re.findall(r"rotate\((-?[\d.]+)deg\)", swim)]
    max_rot = max(rots) if rots else 0.0

    char_h = (bottom_u - top_u) / vb_h * box          # 角色占框高多少
    rot_margin = char_h / 2 * math.sin(math.radians(max_rot))

    rise = _keyframes(css, "msRise")
    positions = [float(v) for v in
                 re.findall(r"inset-block-star<本机目录>)%", rise)]
    assert positions, "msRise 里没有 inset-block-start，角色根本不会动"

    # 放大招 / 解析 / 完成这三个状态停在另一个高度，也要算
    cast = _pct(_block(css, ".mascot.is-casting,\n.mascot.is-scanning,\n"
                              ".mascot.is-done"), "inset-block-start")
    positions.append(cast)

    for pos in positions:
        head = (pos - box / 2) + top_u / vb_h * box
        foot = (pos - box / 2) + bottom_u / vb_h * box
        top_gap = head - rot_margin
        bottom_gap = 100 - (foot + rot_margin)
        assert top_gap > 0, (
            f"角色在 inset-block-start={pos}% 时会探出框顶 "
            f"{rot_margin - head:.1f}%，耳朵会被切掉")
        assert bottom_gap > 0, (
            f"角色在 inset-block-start={pos}% 时会探出框底 "
            f"{rot_margin + foot - 100:.1f}%")


def test_the_character_is_bounded_by_the_stage_height(css: str):
    """角色的尺寸要同时受框的宽**和**高约束。

    框铺满整张上传卡之后不再是正方形。只按宽度算的话，
    角色会比框还高 —— 上下各被裁掉一截，耳朵齐刷刷断一道。
    而框高一点、框宽一点，比例就变了，所以这条约束不能省。

    做法：框声明自己是尺寸容器（``container-type: size``），
    角色的上限用 ``cqmin``（框宽高里小的那个）。
    老浏览器不认 cqmin 时退回 20rem —— 那时角色更小，同样不出框。
    """
    stage = _block(css, ".mascot-stage")
    assert "container-type" in stage and "size" in stage, \
        "框没有声明尺寸容器 —— 角色没法按框高收尺寸"

    mascot = _block(css, ".mascot")
    assert "cqmin" in mascot, \
        "角色的上限没有按框高收（只有宽度约束）—— 框一扁角色就会被裁"

    # 兜底那条必须在 cqmin 那条**之前**：不认 cqmin 的浏览器要能退回它
    caps = re.findall(r"max-inline-siz<本机目录>;]+);", mascot)
    assert len(caps) >= 2, "缺少不认 cqmin 时的兜底上限"
    assert "cqmin" in caps[-1], "cqmin 那条要写在最后（它覆盖兜底）"
    assert "rem" in caps[0], "兜底上限要是一个不依赖 cqmin 的固定值"


def test_the_stage_never_lets_the_character_out_through_rotation(css: str):
    """横向漂移也不能把角色甩出去。

    横向漂移这一层早先还带着 ``rotate``，而它转的是**整个框那么大的**元素 ——
    歪 2.5 度，框角上的东西就能挪出去十几像素。
    现在这一层只许平移。
    """
    drift = _block(css, ".mascot-drift")
    assert "animation" in drift, "漂移层没有动画"
    keyframes = _keyframes(css, "msDrift")
    assert "rotate(" not in keyframes, "漂移又长出旋转来了"
    assert "translate" in keyframes, "漂移应该只做平移"


def test_the_cast_state_keeps_the_vortex_inside(css: str):
    """放大招时漩涡要整个露出来，所以角色停的高度不能太高。

    漩涡的外圈半径 100 个用户单位，中心在脑袋上；
    框里能放下的半径就是框高的一半减去角色中心离框沿的距离。
    """
    cast = _block(css, ".mascot.is-casting,\n.mascot.is-scanning,\n"
                       ".mascot.is-done")
    pos = _pct(cast, "inset-block-start")
    box = _pct(_block(css, ".mascot"), "inline-size")
    # 角色的头在 viewBox 的 106 处，漩涡半径 100
    head_u = 106
    head_pct = (pos - box / 2) + head_u / 300 * box
    radius_pct = 100 / 300 * box
    assert head_pct - radius_pct > 0, \
        "放大招时漩涡的外圈会超出框顶"
    assert head_pct + radius_pct < 100, \
        "放大招时漩涡的外圈会超出框底"


# ------------------------------------------------- 二、脸


def test_the_face_has_an_eye_not_a_hole(client):
    """眼睛得有眼白、瞳孔和眼皮。

    早先画的是两个纯黑椭圆 —— 那是两个洞，不是眼睛。
    """
    html = client.get("/upload").text
    assert 'class="ms-sclera"' in html, "眼睛没有眼白"
    assert 'class="ms-pupil"' in html, "眼睛没有瞳孔"
    assert 'class="ms-lid-rest"' in html, "眼皮不在了"
    assert 'class="ms-shine"' in html, "眼睛没有高光"
    assert 'class="ms-eye-closed"' in html, "闭眼的那条线不在了"


def test_the_lid_is_a_clip_not_a_patch(client):
    """眼皮必须是裁剪出来的，不能是拿头色补的一块。

    补的那块（跟眼睛同色、盖在眼睛上）会跟着眼睛一起动，
    而头的渐变是钉在头坐标系里的 —— 一动就对不上位，
    眼睛上方就会浮出一块补丁形状的浅色。
    裁剪没有这个问题：盖住的地方露出的是**画在底下的头本身**。
    """
    html = client.get("/upload").text
    assert 'id="msLidRest"' in html, "眼皮切口不在了"
    assert 'clip-path="url(#msLidRest)"' in html, "眼睛没有按眼皮线裁剪"
    assert 'id="msLidBlink"' in html, "眨眼的那道切口不在了"

    # 拿头渐变去补一块的写法不许回来
    assert not re.search(r'<[^>]*fill="url\(#msHeadG\)"[^>]*class="ms-lid', html), \
        "眼皮又变回「拿头色补一块」了，那块会跟着眼睛动、跟头对不上位"


def test_the_lid_has_two_independent_drivers(css: str):
    """眼皮有两条线：静止的那条由状态管，眨眼那条由时间管。

    合成一条的话，「被瞄准时睁大眼睛」和「每隔几秒眨一下」
    会互相顶掉 —— 状态一变，眨眼就重置，反之亦然。
    """
    assert ".ms-lid-rest" in css, "静止眼皮没了"
    assert ".ms-blink-cut" in css, "眨眼的那道切口没了"
    rest = _block(css, ".ms-lid-rest")
    assert "transition" in rest, "静止眼皮应当能平滑过渡"
    blink = _block(css, ".ms-blink-cut")
    assert "animation" in blink, "眨眼应当自己按周期走"


# ------------------------------------------------- 三、动作与出处


def test_the_borrowed_motion_keeps_its_attribution(css: str):
    """借来的动作必须留出处。"""
    assert "form-friends" in css, "借用的动作没有写出处"
    assert "MIT" in css, "借用的动作没有写协议"
    assert "github.com/janobarnard/form-friends" in css, \
        "出处只写了项目名，没写地址"


def test_the_eyes_follow_the_pointer(client, css: str):
    """眼睛得会看光标。

    做法照搬借来的那套：视线向量写进 CSS 变量（``--ms-lx`` / ``--ms-ly``），
    整张脸平移，瞳孔再叠一层视差。
    写进 DOM 而不是 JS 状态，是因为每动一次鼠标就重渲染一遍页面太贵。
    """
    html = client.get("/upload").text
    assert "--ms-lx" in html and "--ms-ly" in html, \
        "页面没有把视线向量写进 CSS 变量"
    assert "pointermove" in html, "没有监听光标移动"
    assert "--ms-lx" in css and "--ms-ly" in css, \
        "样式表里没有消费视线向量"


def test_the_stage_has_no_text(client):
    """框里不放任何文字。

    框是角色的地盘，不是说明区。框里一有字，
    角色的注意力就被分走一半 —— 而且说明文字该放在标题旁边。
    """
    html = client.get("/upload").text
    start = html.index('<div class="mascot-stage"')
    depth = 0
    inner = None
    for m in re.finditer(r"<div\b[^>]*>|</div>", html[start:]):
        if m.group(0).startswith("</"):
            depth -= 1
            if depth == 0:
                inner = html[start:start + m.end()]
                break
        else:
            depth += 1
    assert inner, "找不到舞台容器"

    text = re.sub(r"<[^>]*>", " ", inner)
    text = re.sub(r"\s+", " ", text).strip()
    assert not text, f"框里还有文字：{text!r}"


def test_all_four_states_are_handled(css: str):
    """四个状态都得有：待机、被瞄准、蓄力、解析中、完成。"""
    for state in ("is-hover", "is-casting", "is-scanning", "is-done"):
        assert f".mascot.{state}" in css, f"状态 {state} 没有对应的样式"
    assert "prefers-reduced-motion" in css, \
        "没处理「减少动态效果」—— 前庭敏感的人看不了满屏动画"


# ------------------------------------- 五、控件不许压到角色身上（两列布局）


def _app_css() -> str:
    return (APP_ROOT / "app" / "static" / "app.css").read_text("utf-8")


def test_the_controls_and_the_character_are_in_separate_columns(css: str, client):
    """文字与进度条不许横穿角色。

    以前角色是**绝对定位铺满整张上传卡**，而按钮与进度条是整行宽度的 ——
    两者一交叉，「上传并扫描」四个字就压在角色脸上（用户截图里就是这样）。

    现在卡片是 grid：控件一列、角色一列，**结构上**就碰不上。
    """
    app = _app_css()
    assert re.search(r"\.upload-card\s*\{[^}]*displa<本机目录>", app), \
        "上传卡不是 grid —— 两列排不出来"
    # 工作台那条曾经写着 display:flex —— 特异性比 .upload-card 高，
    # 会把 grid 顶掉：行高不撑满，角色被压成一小只。
    assert not re.search(r"\.workbench\s+\.upload-card\s*\{[^}]*display:", app), \
        "工作台那条还留着 display —— 特异性更高，会把 grid 顶掉"

    assert re.search(r"\.up-body\s*\{[^}]*grid-template-columns", app), \
        "控件与角色没有分列"
    assert re.search(r"\.mascot-stage\s*\{[^}]*grid-colum<本机目录>", app), \
        "角色没有落到第二列 —— 可能又压回控件上"

    stage = _block(css, ".mascot-stage")
    assert "position: absolute" not in stage, \
        "角色又被放回绝对定位了 —— 它会盖到文字与进度条上"

    # 窄屏竖排的兜底：小窗口下两列会挤到一起
    assert "@container" in app, "窄屏没有兜底 —— 小窗口又会压上"


def test_the_content_column_holds_the_controls(client):
    """文件名、按钮、进度条都必须在控件列里（.up-content）。"""
    html = client.get("/upload").text
    assert 'class="up-body"' in html, "没有两列的容器"
    assert 'class="up-content"' in html, "控件没有收进自己的列"
    # 控件列里确实有那几样东西
    body = html[html.index('class="up-body"'):]
    content = body[body.index('class="up-content"'):]
    content = content[:content.index("</form>")]
    for marker in ("pickedLine", 'id="go"', 'id="prog"', 'id="barIn"'):
        assert marker in content, f"{marker} 不在控件列里"


def test_the_done_face_is_happy_not_angry(css: str, client):
    """完成态不能是「怒眉 + 吼叫」。

    原来完成时把眼睛换成两道**向上弯的弧** —— 那两道弧长在眼睛的位置上，
    读起来是怒眉；嘴又换成一块深色嘴腔，看着像在吼。整张脸就凶了
    （用户原话：「完成后的表情太难看」）。

    现在：圆眼睛照旧睁着，嘴换成玫瑰色的笑口，底下加一条舌头。
    """
    html = client.get("/upload").text
    assert "ms-eye-happy" not in html, "那两道怒眉还在"

    done_mouth = _block(css, ".mascot.is-done .ms-mouth")
    assert "opacity: 0" in done_mouth, "完成态没有把原来的小嘴收掉"
    assert re.search(r"\.mascot\.is-done\s+\.ms-mouth-open\s*\{[^}]*opacit<本机目录>", css), \
        "完成态没有换成笑口"
    # 圆眼睛要留着 —— 藏掉就只剩那两道弯，又变回凶脸
    assert not re.search(r"\.mascot\.is-done\s+\.ms-eye\s*\{[^}]*opacit<本机目录>", css), \
        "完成态把圆眼睛藏了 —— 只剩两道弯就是怒眉"
    # 笑口要有舌头（深色一块会读成「洞」）
    assert "ms-mouth-tongue" in html or "F7B9CB" in html, \
        "笑口里没有舌头 —— 深色一块看着像在吼"


# ------------------------------------------------- 四、眼睛的睁闭

def test_the_default_eyes_are_wide_open(css: str):
    """默认是**正常眼睛**，不是半眯着。

    曾经为了「慵懒」把眼皮压到 13（半眯），结果很难看：
    只露出眼睛下半截，瞳孔上沿被切掉一块，看着像翻白眼。
    慵懒靠**慢**做出来（眨眼慢、动作沉），不靠眯眼。
    """
    rest = _block(css, ".ms-lid-rest")
    m = re.search(r"var\(--ms-lid,\s*([\d.]+)\)", rest)
    assert m, "静止眼皮没有写默认睁眼程度"
    assert float(m.group(1)) == 0, \
        f"默认眼皮是 {m.group(1)}，应该是 0（全睁）—— 眯着眼会被看成翻白眼"


def test_the_eyes_are_shut_while_parsing(css: str):
    """解析的时候眼睛要**彻底闭住**。

    眼皮线落到眼睛底部（30 = 眼睛整个高度）才算闭严，
    差一点就是眯着 —— 解析要五分钟，眯着看起来像没在使劲。
    闭眼时还要显出一条闭眼线，不然脸上就是两个空窟窿。
    """
    scanning = _block(css, ".mascot.is-scanning")
    m = re.search(r"--ms-li<本机目录>)", scanning)
    assert m, "解析态没有写眼皮位置"
    assert float(m.group(1)) >= 30, \
        f"解析时眼皮只压到 {m.group(1)}，没闭严（要 ≥ 30）"
    assert ".mascot.is-scanning .ms-eye { opacity: 0; }" in css, \
        "解析时眼球没收掉"
    assert ".mascot.is-scanning .ms-eye-closed { opacity: 1; }" in css, \
        "解析时没有显出闭眼线"
