# tools/

验证脚本与标定工具。每个脚本都能单独跑，互不依赖。

## verify_*.py —— 端到端验证

用 FastAPI 的 `TestClient` 起一个真实应用（独立数据库，不碰你的数据目录），
跑完一条完整链路并逐项打印结论。改动之后先跑这几个。

| 脚本 | 验的是什么 |
|---|---|
| `verify_moba_rules.py` | 王者荣耀操作模式库：分路 × 阶段的对位、护栏、覆盖 |
| `verify_video_flow.py` | 上传 → 扫描 → 建对局 → 报告 全链路（含视频片段定位） |
| `verify_video_contract.py` | 上传接口的输入输出契约 |
| `verify_http_big.py` | 超过 500MB 的大文件走 HTTP 上传不被打回 |
| `verify_big_upload.py` | 大文件落盘与清理 |
| `verify_scan_speed.py` | 扫描耗时与采样量 |
| `verify_rule_review.py` | 操作模式库审核页的读写 |
| `verify_guide_collect.py` | 攻略抽取与候选确认 |
| `verify_version_flow.py` | 版本切换流程 |
| `verify_real_video_flow.py` | **真实录屏**走完整 HTTP 链路：上传 → 扫描 → 报告 |

```powershell
.\.venv\Scripts\python.exe tools\verify_moba_rules.py
```

## bench_scan_strategies.py —— 扫描策略基准

三种策略放在同一段真实录屏上跑同一件事（采样 + 缩略图 + 读时钟），
比「样本/秒」。**别再凭直觉选策略** —— 早先「seek 更快」这个假设
就是靠直觉定的，真实录屏上一测就翻了。

```powershell
.\.venv\Scripts\python.exe tools\bench_scan_strategies.py "<录屏>" --seconds 120
```

本机实测（1.7 GB / 89.7 fps / 前 120 秒，采样间隔 0.3 秒）：

```
seek       2.66 样本/秒      顺序 read  6.50      顺序 grab  8.67  ← 密集采样时用这个
```

顺序解码比 seek 快 **3.26 倍**。原因：真实录屏关键帧稀疏，
每次 seek 都要从上一个关键帧解起 —— 「跳过去的那几帧根本不解码」
这个假设不成立。顺序解码时跳过的帧用 `grab()`：照样解码，
但不把像素取出来。

### 采样间隔变了，结论就反过来

同一段视频（240 秒窗口），采样间隔 2 秒时：

```
seek       56.5 秒   ← 最快
顺序 grab  110.8 秒
顺序 read  153.3 秒
```

**seek 的单次代价固定**（从上一个关键帧解到目标帧），**顺序解码的代价
正比于总帧数**。间隔越密，顺序解码摊到的帧越少，所以密集采样时顺序解码
赢；间隔越疏，seek 赢。

于是有了分工（见 `app/collection/scan_jobs.py`）：

| 场景 | 要什么 | 采样间隔 | 策略 |
|---|---|---|---|
| MOBA（按阶段抽帧） | 只要时钟 | 2 秒 | **seek** |
| 自走棋（回合边界） | 灰度差 + 时钟 | 0.3 秒 | 顺序 grab |

真实录屏的端到端效果：MOBA 那一局扫描从 **619 秒降到 329 秒**（含上传 11 秒），
产物是 12 张阶段帧 + 10 张 BP 帧。

## 模型配置：只填一个 API Key

采集层现在**不再等人填**了 —— 分路、胜负、BP、每段操作都由多模态模型
从画面里读，读不出来的记「读不出来」（**不回头让人填**）。

所以第一步是**配模型**（页面上是侧边栏的「模型」→ `/model`）：

| 供应方 | 端点 | 关思考的字段 | 默认模型 |
|---|---|---|---|
| Qwen（阿里云百炼） | `https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions` | `enable_thinking` | `qwen-vl-max` |
| DeepSeek | `https://api.deepseek.com/chat/completions` | `thinking` | `deepseek-flash` |
| MiniMax | `https://api.minimaxi.com/anthropic/v1/messages` | `thinking` | `MiniMax-M3` |

端点**由预设带出来**，客户端一个字都不加 —— 自己拼过一次路径，
少写一段就 404（DeepSeek 官方 base 不带 `/v1`）。

两个要注意的：

- **百炼的 key 按地域绑定**，跨地域会 401 `invalid_api_key`；
- 百炼报 400 `current user api does not support http call`
  说明那个模型不支持 OpenAI 兼容的 HTTP 调用，换一个。

「测试连接」会带一张 64px 的纯红小图问颜色 —— **答对才算能看画面**
（DeepSeek 与百炼都有视觉型号，按供应方断定会错）。这一下**不写盘**。

`verify_video_flow.py` 用一个**本地 mock 模型服务**把整条链路钉住，
不花钱也不联网。

## 打包给 exe 的坑：--add-data 对「文件」源要写目标目录

PyInstaller 的 `--add-data SRC;DEST` 对**文件**源，是把文件**放进 DEST 这个
目录**里。把 DEST 写成完整文件名（`app/collection/clock_digits.json`）的话，
它会把那个名字建成**目录**，文件落到：

```
_internal/app/collection/clock_digits.json/clock_digits.json   ← 实际
_internal/app/collection/clock_digits.json                      ← 代码要找的（是个目录）
```

代码按 `__file__` 旁边找文件 → `open()` 抛 `IsADirectoryError`
（`OSError` 的子类）→ 被 `load_digits()` 吞掉 → 返回空列表 →
**「读不出时钟」**。于是每一局都退回「阶段要手填」，而界面上看不出任何异常。

**这类失败只能靠检查拦**。`scripts/build_exe.py` 里的冒烟测试因此必须验
「是个文件」且「大小与源文件一致」—— 只查「路径存在」的话，目录也算存在，
会给个假 PASS（实测就假过一回：打印「已打包（0 KB）」，exe 里其实读不出时钟）。

`app/main.py` 启动时也会自检一次，模板读不到就在控制台说一声。

## scan_video_product.py —— 本地扫一遍，看抽帧落点

绕开上传与入库，只跑采集层。调「按阶段抽帧」的落点时用它 ——
走 HTTP 上传意味着每次都要传 1.7 GB 再等十分钟扫描。

```powershell
.\.venv\Scripts\python.exe tools\scan_video_product.py "<录屏>"
```

输出含：时钟读数与拟合、对局起止、每个阶段的抽帧落点、
以及**边界检查**（有没有两帧挤在同一分钟）。

## contact_sheet.py —— 视频时刻联系表

把若干时刻拼成一张图，用来看清「这一段画面到底是什么」。
设计 BP 帧序列的取样落点时靠它 —— 0~200 秒里哪些是征召、哪些是选人、
哪些是加载，看图才知道，不靠猜。

```powershell
.\.venv\Scripts\python.exe tools\contact_sheet.py "<录屏>" 3 12 25 40 60 90 130 170 195
```

本机这段录屏的实际分布（决定了 BP 取 10 帧的落点）：

```
0-20s   征召/匹配（双方头像、禁用）
25s     地图加载
40-130s 选人（英雄选择网格）
170s    加载 3D 模型
195s    最终对阵（双方 5 人 + VS）
198.5s  对局开始（时钟第一次读得到）
```

## list_rules.py —— 把规则库摊开看

47 条规则的触发条件、期望动作、严重度、护栏 —— 手工读 JSON 太慢，
而且容易看漏 `none` 护栏。

```powershell
.\.venv\Scripts\python.exe tools\list_rules.py hok S45
```

## check_none_guards.py —— 护栏与期望有没有撞车

`none` 护栏用到了哪些事实、同一格里哪些模式期望同一个动作事实。

期望判定是「这一回合做过符合要求的动作就算符合」（看全部同类动作），
触发条件看的是镜像出来的**最后一个**动作 —— 两边口径不同，
所以库里**不能**在同一个格子里既用动作事实写 none 护栏、
又有另一条模式期望同一个动作事实。这个脚本就是查这件事的。

```powershell
.\.venv\Scripts\python.exe tools\check_none_guards.py hok S45
```

## make_hok_demo_match.py —— 造一局有结论的对局

给报告页 / 标注页做样式验证用。真实录屏那一局要等扫描 + 标注十几分钟，
等不起也不方便反复调。数据是**手工构造**的：分路、英雄、对位、
每分钟的动作都照着 S45 规则库里打野那几条填，因此能同时造出
「做对了」和「做反了」两种结局。

```powershell
.\.venv\Scripts\python.exe tools\make_hok_demo_match.py
```

## build_clock_templates.py —— 游戏内时钟标定

**这是标定值，不是自动适配。** 槽位坐标随分辨率和屏幕比例变化，
换机型必须重跑。

```powershell
.\.venv\Scripts\python.exe tools\build_clock_templates.py "<录屏路径>"
.\.venv\Scripts\python.exe tools\build_clock_templates.py "<录屏路径>" --reuse
```

`--reuse` 复用 `.tmp/clock_patches.npz` 的样本缓存。调聚类/采样参数时
用得上 —— 顺序解一遍 2670×1200@90fps 要十分钟，不该每次都重来。

### 做法

不聚类、不手标。**锚点 + 起点扫描**：

1. 抽帧，按四个数字位的窗口切出字形并归一化
2. 锚点给出一个**已确认的读数**（`ANCHOR_T` / `ANCHOR_READING`，
   默认 `t=220s → 00:21`），由此反推对局起点
3. 围绕这个起点做 ±12 秒扫描，取「类内平方距离和」最小的那个 ——
   起点错一秒，所有标签整体错位，同一个数字被拆散，代价立刻变大。
   扫描结果与锚点反推值一致，等于交叉验证了两遍
4. 按标签分组求均值，得到每位各自的模板

### 为什么必须按位建模板

同一个数字在四个位置上的落点不一样。实测「0」在分钟十位与秒个位之间
相关只有 0.86，混在一起求均值会把字形抹平，模板直接糊成一团。
所以每个位置一套模板。

代价：某一位未必覆盖全部十个数字（一局 12 分钟，分钟十位只出现过
0 和 1）。缺哪个数字就不建哪个 —— 读时钟时该位出现没建过的数字，
绝对分达不到下限，直接报「读不出」，**不硬凑**。

### 本机标定记录

设备：2670×1200（超宽屏，比例 2.225）

```
列段实测（阈值 140，逐列亮像素 >= 2，允许 1 列空隙）
  M1  1823-1835   中心 1829
  M2  1839-1852   中心 1845.5
  冒号 1855-1859
  S1  1861-1875   中心 1868
  S2  1880-1887   中心 1883.5（「1」偏窄）

由此定四个 20px 宽的取样窗
  M1 1819-1839   M2 1836-1856   S1 1859-1879   S2 1876-1896
  y 10-35
```

换设备时按同样步骤重做：截几帧 → 放大画网格 → 读列段 → 改 `SLOTS`。

## verify_clock_on_video.py / scan_clock_on_video.py —— 验证读数

```powershell
.\.venv\Scripts\python.exe tools\verify_clock_on_video.py "<录屏>"
.\.venv\Scripts\python.exe tools\scan_clock_on_video.py "<录屏>" 2
```

- `verify` 只 seek 五个**事先看图确认过**的时刻（`00:21 / 02:01 /
  03:41 / 05:21 / 08:21`），秒级出结果，判模板对不对
- `scan` 顺序解完整个视频，看读出率、拟合斜率与残差、阶段落点，
  以及**读不出来的是哪几段**

这两个时刻表是与模板无关的独立答案，所以能反过来判模板；
反过来不能 —— 用模板自己生成的读数去验模板是循环论证。

## integrate_galaxy.py —— 封面银河

把第三方开源项目 [galaxy-explorer](https://github.com/zjoooooo/galaxy-explorer)
（MIT，© 2026 Justin Zhang Jun）搬进 `app/static/galaxy/`，
改造成只保留满屏银河、隐藏演示 UI。

上游素材放在 `.tmp/galaxy/`（`index.html` + `vendor/`）。

```powershell
.\.venv\Scripts\python.exe tools\integrate_galaxy.py
```

改第三方代码必须留出处，也必须知道**我们到底改了什么** —— 所以
这步是脚本而不是手工改 HTML。
