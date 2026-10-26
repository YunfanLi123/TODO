# 提示词：把 S45 BP 阶段操作模式入库

> 用途：交给任意执行者（人或工具）的一段自包含任务提示词。
> 前提：执行者能读写 ``（非 git 仓库），并能跑该项目的 Python 虚拟环境。

---

## 提示词正文（从这里开始复制使用）

### 任务

把「王者荣耀 S45 BP 阶段操作模式」入库为本项目操作模式库的 **BP 层**。

王者这一套（`hok`）的模式库目前分三层：**分路 → 英雄 → 时期**。分路层已有 85 条
（`app/seed/hok_modes.json`），英雄层与 BP 层是空的。你要做的是把 BP 层补上。

### 输入素材（已就位，只读）

**BP 阶段操作模式存放在这个目录：**

```
\hok-s45-bp-modes\
```

| 文件 | 内容 | 大小 |
| --- | --- | --- |
| `00-bp-decision-tree.txt` | S45 BP 阶段决策树：全局前提 + 决策树（禁用阶段 / 一抢 / 二抢 / 三抢 / 阶段二一抢 / 末选 / 阵容收尾）+ 分位置选人清单 + 体系与反制关系 + 来源 | 21079 B |

素材规模：BP 决策树 8 个节点、约 60 条分支，另有 5 个位置各 5 条选人/禁选要点。

### 代码侧已经铺好的脚手架（先读这些，别重复造）

动手前先读懂下面几处，**它们已经为英雄层与 BP 层预留好了接口**：

1. `app/adapters/honor_of_kings.py`
   - 事实词表 `MOBA_FACT_KINDS` 里**已经有 `my_hero`**（自己玩的英雄，对局级，标注页一次选完）。
   - `_load_heroes_by_lane()` 会去读 `app/seed/hok_heroes.json`，读不到就退回
     `HEROES_BY_LANE_FALLBACK`（每路 3 个英雄的兜底清单）。
   - 注释里写明：真正的下拉清单在 `app/seed/hok_heroes.json`，**由
     `tools/build_hok_hero_modes.py` 从攻略原文推出，覆盖攻略里出现的全部英雄**。
     适配器只读不写。

2. `app/rules/store.py::_seed_files()`
   - 种子文件按**后缀**匹配，`hok_modes.json` / `hok_hero_modes.json` 都收。
   - 注释里记着一个已修的坑：早先按前缀匹配 `hok_modes`，于是英雄层那个文件
     **静默地一次都没被加载过**——库看起来完整、界面正常，只是英雄专属模式一条都不生效。
     这次加新种子文件要确认它真的被读到了。

3. `app/collection/video_upload.py`
   - `BP_SAMPLES = 10`：BP 段（0–200 秒，含征召、加载、选人、最终对阵）抽 10 帧。
   - 注释原文：「它的价值是让用户看清『我方和敌方各选了谁』——**英雄层模式要靠这个才能触发**，
     而对位优劣本来就只能由人判断。」

4. `tools/hok_decision_tree.py` + `tools/hok_spec/*.py` + `tools/build_hok_modes.py`
   - 分路层的四层链路，BP 层照着扩一层。

### 先做一个判断：BP 层在当前架构里是「结论型」还是「指控型」

**这一步决定后面所有设计，先想清楚再动手。**

现有 85 条分路层模式是**指控型**：玩家在打的时候做了某个动作，规则判他该不该做。
它们的触发事实全是对局内的（`time_min` / `position` / `my_hero` / `gold_diff` /
`kill_diff` / `tower_diff` / `has_ult_ready` / `matchup` / `pressure`）。

BP 发生在对局开始之前，视频里能看到的是**选人界面与最终对阵**，看不到玩家当时
「点了哪个按钮」。所以 BP 层天然是**结论型**：

> 「你和对方各选了谁 → 按 S45 决策树，这套阵容缺什么 / 敌方走什么体系 / 下一手该 ban 谁」

**默认按结论型做**（推荐）。理由：现有事实词表与动作词表全部是对局内的，
BP 用它们既触发不了、也判不了。若要做成指控型（实时判玩家 ban/pick 对不对），
需要新增一整套「选人状态事实 + BP 动作词表」，且视频采集与判定逻辑都要改，
那是另一个量级的工程 —— 遇到要动这里时，先跟人确认，别自己扩。

### 四层具体要做什么

#### 1. 原文层

把 `00-bp-decision-tree.txt` 落到
`app/seed/sources/hok_s45_bp_decision_tree.txt`。

格式归一到现有 `app/seed/sources/hok_s45_lane_decision_tree.txt` 的约定：
- 分节线用 `═` 独占一行（素材现在用的是 `=` 与 `━━━`，入库前要转）
- 节点标题行形如 `0:00 节点` / `禁用阶段节点` / `一抢节点` 等
  —— **BP 的节点不是时间，是决策环节**，解析器要能按自定义节点名解析
- 每个节点内按「前提 / 动作 / 后果」分叉，后果末尾带【状态标签】
- 素材头部「全局前提」「分位置选人清单」「体系与连锁反应」「来源」是**说明性内容，
  不是树**，解析时必须跳过（可另存为附注文件）
- UTF-8 无 BOM、LF 行尾
- **动作文字一个字都不要改写**——原文必须逐字保留，这是逐字引用的前提

#### 2. 解析器层

新增 `tools/hok_bp_decision_tree.py`。**不要改现有 `tools/hok_decision_tree.py`**
——分路层 85 条靠它跑。

- 复用它的 `Branch` / `quote()` / `normalize_for_compare()` / `outcome_text()`。
- 差异点：BP 的节点标题不是 `0:00 节点` 这种时间格式，要单独一套识别规则。
- 解析结果要多带两个维度：`phase`（BP 环节：禁用 / 一抢 / 二抢 / 三抢 / 阶段二一抢 /
  末选 / 收尾）和 `lane`（分位置选人清单那一节的五个位置）。
- 分位置选人清单那一节结构不同于决策树，**单独用一个解析函数**，不要硬塞进同一棵树。

#### 3. 映射表层

新增 `tools/hok_bp_spec/`（一个模块一环节，或一个模块一阶段，见下）。

字段沿用 `tools/hok_spec/__init__.py` 的那套，但 **BP 层要加两个东西**：

- `comp`（可选）：这条模式判的是哪一类阵容问题。取值建议限定在一组固定标签上，
  例如 `no_engage`（缺开团点）/ `no_frontline`（缺前排）/ `no_counter`
  （缺反手）/ `single_damage_type`（伤害类型单一）/ `countered`（被针对）/
  `comp_mismatch`（与计划体系不符）。
- `advice`（结论型模式专用）：给玩家的话。**结论型模式没有 `expect` 动作事实可写**，
  它判的是「这套阵容成立吗」，所以用 `advice` 承载结论，`expect` 留空或写
  一个新增的 BP 结论事实。

事实词表要新增（加在 `app/adapters/honor_of_kings.py` 的 `MOBA_FACT_KINDS` 里）：

- `enemy_lineup`：敌方已选英雄集合
- `ally_lineup`：己方已选英雄集合
- `lane_filled`：已确定位置（对抗路/打野/中路/发育路/游走）
- `comp_gap`：我方阵容缺件（无开团 / 无前排 / 无反手 / 伤害类型单一）
- `enemy_comp`：敌方体系标签（冲脸 / 消耗 / 拉扯 / 单带 / 后期）

动作词表方面：结论型模式不判动作，**不要硬塞进 `objective_action` /
`positioning_action` 那套对局内词表**。如果确实要判「BP 阶段玩家点了什么」，
那是指控型，需要另立 `bp_action` 词表（`ban_hero` / `pick_hero` / `hold_counter`），
并同时改视频采集与判定逻辑 —— 先跟人确认再做。

决策程序（结论型，与分路层的合并规则不同）：
1. 一条分支 → 一条结论。分支的「前提」是阵容事实，「后果」是原文给的结论，
   `advice` 直接取原文后果的语义。
2. 同一 BP 环节里，多条分支若会同时命中同一份阵容（例如「缺前排」和
   「缺开团」可能同时成立），**用 `extra` / `guards` 互斥开**；互斥不开就合并成一条，
   `comp` 写成组合标签。
3. 判不了的分支（对方的 BP 意图、绝活哥身份、预选）→ 放进 `SKIPPED` 并写明理由。
4. 每条分支都要有归宿（`RULES` ∪ `SKIPPED` == 原文全部分支），漏写生成器要报错。

#### 4. 生成器层

新增 `tools/build_hok_bp_modes.py`，仿 `tools/build_hok_modes.py`：

- 合成 `app/seed/hok_bp_modes.json`（**生成产物，不许手改**）
- 跑全套校验：逐字引用、分支全覆盖、同环节互斥、事实词表合法、id 唯一
- **不要动 `app/seed/hok_modes.json`（分路层 85 条是历史资产）**

#### 5. 接线与文档

- `tests/test_mode_library.py` 增加 BP 层断言（模式数、引用逐字、互斥、分支全覆盖、
  id 唯一、种子文件确实被 `store.py::_seed_files()` 读到）。
- `app/rules/store.py` 确认新种子文件被后缀匹配收进去，并**加一条测试**证明它被加载了
  —— 英雄层那个文件静默没被加载过一次，这个坑不能再踩。
- 标注页 / 模式库页能按 BP 环节筛选。
- 更新 README §8 的四层表格，加 BP 层一行，并写明：
  **BP 层原文位于 `\hok-s45-bp-modes\00-bp-decision-tree.txt`，
  入库后的唯一真源为 `app/seed/sources/hok_s45_bp_decision_tree.txt`。**

### 验收

```
.\.venv\Scripts\python.exe -m pytest tests\ -q
python tools\verify_version_flow.py
python tools\verify_video_contract.py
python tools\verify_guide_collect.py
python tools\verify_rule_review.py
python tools\verify_hok_bp_modes.py      # 新增
```

基线（改动前）：pytest **414 通过 + 4 跳过**，四个 verify 脚本共 **113 项断言**。
改动后**不得回归**，并补上 BP 层自己的断言。

### 硬约束（踩过的坑）

- 不要手改任何生成产物（`hok_modes.json` / `hok_hero_modes.json` / `hok_bp_modes.json`）。
- 不要为了省事把 BP 层降级成手写 JSON。
- 引用逐字，改一个字就报错。
- 新增事实/动作词表值，必须同时在 `app/adapters/honor_of_kings.py` 补中文标签。
- BP 规则是**结论型**，不要硬套成对局内的动作指控。
- 旧规则只退役不删除：退役清单落 `app/seed/hok_retired.json`，
  `load_seed_library` 自动标 `deprecated`，历史对局仍按当时规则复盘。
- 打包 exe 前先停 `MatchAnalyzer.exe`，否则 `dist\MatchAnalyzer\MatchAnalyzer.exe` 被占用报
  `PermissionError [WinError 5]`。
- 打包后 `dist\` 被清空重建，**exe 的数据目录也会没**，需要重灌/重传视频。

---

## 提示词正文结束

## 附：位置速查

| 东西 | 位置 |
| --- | --- |
| **BP 阶段操作模式（本次入库的素材）** | `\hok-s45-bp-modes\` |
| 分路层英雄操作模式（上一轮，147 英雄） | `\hok-s45-hero-trees\` |
| 分路层原文（唯一真源，先例） | `app\seed\sources\hok_s45_lane_decision_tree.txt` |
| 分路层解析器 / 映射表 / 生成器 | `tools\hok_decision_tree.py` / `tools\hok_spec\` / `tools\build_hok_modes.py` |
| 分路层产物（生成物，勿手改） | `app\seed\hok_modes.json`（85 条） |
| 英雄层产物（待生成，代码已预留） | `app\seed\hok_hero_modes.json` + `app\seed\hok_heroes.json` |
| 事实与动作词表 | `app\adapters\honor_of_kings.py` |
| 种子文件加载 | `app\rules\store.py::_seed_files()`（按后缀匹配） |
| BP 段视频帧采集 | `app\collection\video_upload.py`（`BP_SAMPLES = 10`，0–200 秒） |
| 约定记录 | `README.md` §8 |
| 回归测试 | `tests\test_mode_library.py` |
| 上一轮的同类提示词 | `docs\hok-hero-modes-ingest-prompt.md` |
