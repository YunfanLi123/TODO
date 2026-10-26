# 落地页第二页改写提示词

> 这份文件本身是提示词。把它整份交给要执行的人（或模型）即可。
> 目标文件：`\app\templates\landing.html`

---

## 任务

把落地页（路由 `/`）的第二页 `<section class="page" id="page-2">` 里的三张卡，
换成下面三组内容：**编号 + 小标题 + 两段正文**。

这是全站唯一的对外门面，文案与动效都不能有临时感。

## 要改的文件

- `\app\templates\landing.html`
- 只动两处：`#page-2` 这个 section（当前在第 317–360 行），以及同一文件内嵌 `<style>` 里
  与 `.card` 相关的规则。**不要新建文件，不要拆出外部 CSS。**

## 一个字节都不许动的部分

1. `<header class="topbar">` —— 品牌与导航
2. `<section class="hero">` —— 银河背景、eyebrow「不断向被验证的模式优化拟合」、
   大标题「对局之外」、CTA「探索玩法」、底部「向下滚动」提示
3. `<section class="page" id="page-3">` —— 攻略数据库那四张卡（条形图、脚注、链接全在里面）
4. `:root` 里的 CSS 变量（`--ink` / `--ink-dim` / `--ink-faint` / `--blue` / `--violet` /
   `--line` / `--card` / `--font`）
5. 银河 iframe 及其来源 `/static/galaxy/index.html`
6. `.page` / `.cards` / `.cards.four` / `.card` / `.card-no` / `.card h3` / `.card p` /
   `.card-note` / `.card-foot` 现有排版规则（字体族、字号、字重、字距、行高、颜色）
7. 现有媒体查询（1040px 两列、680px 一列、`prefers-reduced-motion`）

## 第二页里唯一保留的东西

`<div class="page-head">` 里的 `<h2>从路径依赖到拓展更多玩法可能</h2>` **原样保留**。

`<p class="lead">` 那一段引导文字**删掉** —— 第二页除大标题外不再有别的文字。

## 三张卡的内容（逐字照抄，不要润色、不要改标点）

### 卡 01

- 编号：`01`
- 小标题：`定义什么是好模式`
- 正文第一段：`模式如何被定义，决定玩法在哪些方向上被丰富和优化，也决定时间最终流向哪里。`
- 正文第二段：`只有在具体对局中遇到的关键决策与复杂边界，才能暴露玩法真正的能力不足。高质量的模式定义，也是玩法能力持续提升的起点。`

### 卡 02

- 编号：`02`
- 小标题：`定义什么是好标准`
- 正文第一段：`可信的模式不仅关注来源是否真实，更关注依据是否充分、推理是否完整、边界是否清晰。`
- 正文第二段：`清晰一致的标准，让操作模式库真正可衡量、可优化。`

### 卡 03

- 编号：`03`
- 小标题：`持续丰富玩法与操作模式库`
- 正文第一段：`操作模式库不会一蹴而就，而是在持续迭代中不断演进。`
- 正文第二段：`通过不断引入新的复杂对局情况与真实反馈，构建对局—复盘—更新的闭环，推动玩法和操作模式库不断丰富。`

## DOM 结构

保持和现有卡完全同构：

```html
<article class="card">
  <div class="card-no">01</div>
  <h3>定义什么是好模式</h3>
  <p>模式如何被定义，决定玩法在哪些方向上被丰富和优化，也决定时间最终流向哪里。</p>
  <p>只有在具体对局中遇到的关键决策与复杂边界，才能暴露玩法真正的能力不足。高质量的模式定义，也是玩法能力持续提升的起点。</p>
</article>
```

- 编号只放 `01` / `02` / `03`，小标题放 `<h3>`。`.card-no::after` 会自动补一条渐变发丝线，
  不要手动加横线，也不要把编号和标题拼成一个字符串。
- **删掉 `.card-note`**（原来那行「内核不动」「口径不动」）。新内容里没有它的位置。
- 三张卡仍由 `.cards` 容器控制（`repeat(3, minmax(0, 1fr))`），列数、间距、`max-width: 1180px` 都不改。
- 两段正文各用一个 `<p>`，不要塞进同一个 `<p>` 里用 `<br>` 分隔。
- **两段正文之间必须留间距**：`.card p { margin: 0; }` 意味着两个 `<p>` 会贴在一起。
  加一条**只作用于第二页**的规则，别用 `.card p + p` —— 第三页第一张卡是
  `<p>` + `<p class="card-note">`，通用选择器会把那边撑开：

  ```css
  #page-2 .card p + p { margin-top: 12px; }
  ```

## 字体质感

「质感」指的是现有这套排版参数，**不是新造一种风格**：

- 中文字形走 `var(--font)`（PingFang SC → HarmonyOS Sans SC → 思源黑体 → Noto Sans SC →
  等线 → 微软雅黑 → 系统 UI）
- `font-weight` **封顶 700**：800 在雅黑上是合成加粗，笔画会发胀
- 编号 `.card-no`：11.5px / 700 / `letter-spacing: .18em` / `#7fa0ff`
- 小标题 `h3`：17px / 600 / `letter-spacing: .02em` / `#fff`
- 正文 `p`：13.5px / `line-height: 1.95` / `rgba(255, 255, 255, .48)`
- 「对局—复盘—更新」里的是全角破折号 U+2014，保持原样，不要换成连字符
- 不新增字体族、不引 web font、不用 `text-transform`（中文没有大小写）
- 正文里的中文标点（，。：、）保持全角

## 动效

现在的 `.card` 没有任何 hover 效果。要加的是一组：**静止微光 → 悬停浮起 + 渐变描边**。

### 静止态

每张卡常驻一圈很淡的光晕，只是「有光」，不是「亮」：

```css
.card {
  transition: transform .3s cubic-bezier(.22, .61, .36, 1),
              box-shadow .3s ease, border-color .3s ease;
  box-shadow: 0 18px 44px rgba(0, 0, 0, .3),
              0 0 20px rgba(127, 160, 255, .08);
}
```

（原 `.card` 只有 `box-shadow: 0 18px 44px rgba(0,0,0,.3);` 一层，追加第二层即可。）

### 悬停态

- 整张卡上浮 `translateY(-4px)`。`.cta:hover` 用的是 -2px，卡片面积大，给到 -4px 才有「被托起来」的感觉
- 边框变成**渐变描边**：`--blue` → `--violet` 绕边框一圈
- 外发光加强到 `0 0 34px rgba(124, 92, 255, .30)`

### 渐变描边的实现

**不要用 `border-image`** —— 圆角处会崩。用伪元素 + mask 挖空中间：

```css
.card::after {
  content: ""; position: absolute; inset: 0;
  border-radius: inherit; padding: 1px;
  pointer-events: none; opacity: 0;
  background: linear-gradient(135deg, var(--blue), var(--violet), var(--blue));
  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite: xor;
          mask-composite: exclude;
  transition: opacity .3s ease;
}
.card:hover::after { opacity: 1; }
```

`.card` 已经是 `position: relative`，不用再加。`pointer-events: none` 保证文字仍可选中。

### 降级

文件里已有 `@media (prefers-reduced-motion: reduce)`，把卡片一起关掉：

```css
@media (prefers-reduced-motion: reduce) {
  .card { transition: none; }
  .card:hover { transform: none; }
}
```

### 不要做流光

不要用 `conic-gradient` 旋转边框。要的是「渐变光出现」，静止渐变就够；
旋转光会和封面的银河动效打架。

## 交付前自查

- [ ] `#page-2` 里除 `h2` 外没有别的文字（`lead` 已删、`.card-note` 已删）
- [ ] `hero` 段和 `#page-3` 的 diff 为空
- [ ] 三张卡在 1040px 以下两列、680px 以下一列（媒体查询已存在，别动）
- [ ] 没有新增字体族，`font-weight` 没有超过 700
- [ ] `prefers-reduced-motion` 下卡片不动
- [ ] 跑 `.\.venv\Scripts\python.exe run.py`，开 `http://127.0.0.1:8000/` 滚到第二页，
      依次划过三张卡，确认：静止时微光隐约可见 → 悬停时渐变描边出现、整卡上浮约 4px →
      移开后复原，且三张卡高度一致（grid 默认 stretch）
