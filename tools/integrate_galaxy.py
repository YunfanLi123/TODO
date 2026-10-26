"""把第三方开源项目 galaxy-explorer 搬进本项目，做成封面背景。

来源：https://github.com/zjoooooo/galaxy-explorer
      Milky Way Galaxy — MIT License, (c) 2026 Justin Zhang Jun

为什么是脚本而不是直接放文件：这是一段**第三方代码**，日后升级上游、
或换掉它，都要能重跑这一份才知道我们到底改了什么。手工改 HTML 的话，
下次没人分得清哪些是上游的、哪些是我们的。

原始素材放在 .tmp/galaxy/（上游 index.html + vendor/）。
跑之前先把它放好，然后：

    .venv\\Scripts\\python.exe tools\\integrate_galaxy.py

改动只有三处，每一处都是为了适配本项目，而不是改效果：

  1) 隐藏演示 UI（标题条、语言切换、天体标签、底部索引条…）
     原页面的重点是「可玩的银河」，我们要的是「当背景用的银河」。
     用 `body > *:not(#cv)` 一刀切，比逐个列 id 可靠：它把 JS 动态生成的
     标签也一起藏了，而隐藏不等于移除，JS 初始化不会因为元素不可见而报错。
     注意别改成「按可见性过滤」去写 —— 真正要的就是这条通配规则。
  2) 保留 canvas 满屏与鼠标交互（拖动可转视角，正是参考视频里的效果）
  3) 注明来源与许可，别让第三方代码来历不明

踩过的坑（改这个脚本前先读）：
  - HTML 注释里不能出现连续两个连字符。`<!--` 之后写个 `----` 就会提前闭合，
    后面的 CSS 全部当正文显示在页面顶部。
  - 注入的 CSS 必须包在 <style> 里，否则同样被当成文本渲染。
  - 校验注释时要把开标签 `<!--` 一起去掉再查 `--`，否则 `<!--` 自己就误报。
"""

import io
import os
import re
import shutil
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, ".tmp", "galaxy")
DST = os.path.join(ROOT, "app", "static", "galaxy")

if not os.path.isdir(SRC):
    sys.exit(
        "找不到上游素材：{}\n"
        "先从 https://github.com/zjoooooo/galaxy-explorer 取 index.html "
        "与 vendor/ 放到该目录，再跑本脚本。".format(SRC)
    )

# ---------------------------------------------------------------- vendor
os.makedirs(os.path.join(DST, "vendor"), exist_ok=True)
for name in os.listdir(os.path.join(SRC, "vendor")):
    shutil.copy2(os.path.join(SRC, "vendor", name),
                 os.path.join(DST, "vendor", name))
print("vendor 文件:", len(os.listdir(os.path.join(DST, "vendor"))))

# ---------------------------------------------------------------- 注入
with open(os.path.join(SRC, "index.html"), encoding="utf-8") as f:
    idx = f.read()

HIDE_CSS = """/* 本项目适配。来源：github.com/zjoooooo/galaxy-explorer（MIT）。
   只保留满屏的银河 canvas，演示 UI 全部隐藏。
   用 body > *:not(#cv) 一刀切：动态生成的天体标签也一起藏了，
   而隐藏不等于移除，JS 初始化不会因为元素不可见而报错。 */
body > *:not(#cv) { display: none !important; }
html, body { background: #03040a; overflow: hidden; margin: 0; }
#cv { display: block; position: fixed; inset: 0; width: 100%; height: 100%; }"""

# 注意：注释正文里不能出现连续两个连字符
NOTE = ("<!-- 第三方代码的适配版本。来源：github.com/zjoooooo/galaxy-explorer\n"
        "     (Milky Way Galaxy, MIT License, (c) 2026 Justin Zhang Jun)。\n"
        "     仅保留其 WebGL 银河渲染，演示 UI 已隐藏。 -->\n")

idx = re.sub(r"<title>.*?</title>",
             "<title>galaxy background</title>", idx, count=1, flags=re.S)
idx = NOTE + idx
idx = idx.replace("</head>", "<style>\n" + HIDE_CSS + "\n</style>\n</head>", 1)

note_body = NOTE[len("<!--"):-len("-->")]
assert "--" not in note_body, "注入的注释正文含连续连字符，会提前闭合"
assert "body > *:not(#cv)" in idx, "注入的 CSS 丢了"

out = os.path.join(DST, "index.html")
with open(out, "w", encoding="utf-8", newline="\n") as f:
    f.write(idx)

print("galaxy/index.html:", round(os.path.getsize(out) / 1024, 1), "KB")

# ---------------------------------------------------------------- 校验
# 依赖文件全在不在。少一个 vendor 文件，页面照样加载成功，只是渲染不出来。
missing = [d for d in re.findall(r'src="([^"]*vendor/[^"]+)"', idx)
           if not os.path.isfile(os.path.join(DST, d))]
print("缺失依赖:", missing or "无")
assert not missing, "vendor 依赖不全，银河渲染不出来"
print("完成")
