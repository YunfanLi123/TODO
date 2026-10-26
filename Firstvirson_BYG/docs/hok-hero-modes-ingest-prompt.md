# 提示词：把 S45 五分路英雄操作模式入库

> 用途：交给任意执行者（人或工具）的一段自包含任务提示词。
> 前提：执行者能读写 ``（非 git 仓库），并能跑该项目的 Python 虚拟环境。

---

## 提示词正文（从这里开始复制使用）

### 任务

把「王者荣耀 S45 五分路英雄操作模式」入库为本项目操作模式库的**英雄层**，产出 `app/seed/hok_hero_modes.json`。

模式库现在分三层：**分路 → 英雄 → 时期**。分路层已有 85 条，**英雄层是空的**（`app/seed/hok_hero_modes.json` 已被删除）。你要做的是把英雄层补上。

### 输入素材（已就位，只读）

**所有分路英雄的操作模式存放在这个目录：**

```
\hok-s45-hero-trees\
```

| 文件 | 内容 | 大小 |
| --- | --- | --- |
| `01-clash-lane-heroes.txt` | 对抗路 33 名英雄 | 196130 B |
| `02-farm-lane-heroes.txt` | 发育路 22 名英雄 | 113812 B |
| `03-mid-lane-heroes.txt` | 中路 29 名英雄 | 108159 B |
| `04-jungle-lane-heroes.txt` | 打野位 32 名英雄 | 135827 B |
| `05-roam-lane-heroes.txt` | 游走位 31 名英雄 | 145020 B |
| `00-README.txt` | 总索引：全局前提、数据口径、已知局限 | 18480 B |

同目录另有校验脚本 `verify-all-lanes.py` / `verify-all-lanes.txt`（可重跑，验证英雄覆盖、节点数与顺序、三分支完整、状态标签链、无 markdown 符号、无主观评价词、无不可观测前提）。

素材规模：147 名英雄 / 882 个时间节点 / 2910 条分支。

### 必须遵守的既有约定（README §8 已记录）

**攻略原文是唯一真源**，四层链路，每层职责不能混：

```
攻略原文  →  解析器  →  映射表  →  生成器  →  app/seed/*.json
```

- `app/seed/hok_modes.json` 是**生成产物，不要手改**。`tests/test_mode_library.py` 会重跑生成器逐条比对，手改立刻红。
- **引用必须逐字来自原文**。校验器拿每条模式的 `source.quoted` 回到原文里逐字比对，改一个字就报错。
- **同格（分路 × 阶段）两条模式必须触发互斥**，否则玩家被两条同时指控。
- `expect.values` **必须来自王者动作词表**（`app/adapters/honor_of_kings.py`），否则标注页勾不出来。
- 攻略里**每条分支都要有归宿**（入 `RULES` 或 `SKIPPED`），漏写生成器会报错。

### 四层具体要做什么

#### 1. 原文层

把五份英雄决策树**合并成一个**英雄层原文，落到
`app/seed/sources/hok_s45_hero_decision_tree.txt`
（分五份文件 + 索引也可以，但要在 README §8 的表里写清入口）。

格式要归一到现有 `app/seed/sources/hok_s45_lane_decision_tree.txt` 的约定：

- 分节线用 `═` 独占一行。**素材现在用的是 `=`，必须转。**
- 节标题 = 分路名，独占一行，取值必须是 `打野` / `中路` / `对抗路` / `发育路` / `游走位`。
  注意素材里写的是「打野位」，解析器的 `LANE_CODES` 认的是「打野」，要改。
- 段内按英雄分块：**英雄名单独占一行**（如 `赵云`），其后是「定位与关键数据」「机制要点」两行，
  再是 `0:00 节点` … `12:00 之后` 六个节点。
- 素材文件头的「全局前提 / 时间轴 / 装备改动 / 视野机制」是**说明性内容，不是树**，合并时归到
  文件头或单独一节，解析时必须跳过。
- UTF-8 无 BOM、LF 行尾。
- **一个字的动作文字都不要改写**——原文必须逐字保留，这是逐字引用的前提。

#### 2. 解析器层

现有 `tools/hok_decision_tree.py` 只有「分路」一个维度。英雄层需要多一个「英雄」维度：

- 新增 `tools/hok_hero_decision_tree.py`。**不要改现有解析器**——分路层 85 条靠它。
- 复用它的 `Branch` / `quote()` / `normalize_for_compare()` / `outcome_text()`，只在 `Node` 上加 `hero` 字段。
- 英雄块标题行需要新的识别规则，注意不能和 `LANE_CODES` 撞车（`赵云` 这类两字名也可能出现在别的行里）。
- `Branch.key` 是「前提链用 ` / ` 连起来」。素材里有些前提行以 `【状态标签】+ ` 开头，
  这会让 key 带上前缀。**key 建议剥掉 `【...】` 前缀**，否则同一分支在不同状态下 key 不稳定。
- 逐字引用继续由解析器拼，不许手抄。

#### 3. 映射表层

新增 `tools/hok_hero_spec/{jungle,mid,side,carry,roam}.py`，每个模块导出 `LANE` / `RULES` / `SKIPPED`。

字段沿用 `tools/hok_spec/__init__.py` 里的那套：`node` / `key` / `merge` / `title` / `expect` /
`subject` / `scope` / `extra` / `guards` / `why` / `hint` / `tags` / `severity`。

决策程序（同 `tools/hok_spec/__init__.py` 第 1–7 条）：

1. 读每条分支的**前提**和**动作**，分开看。
2. 前提里能写成标量事实的（敌方减员 → `kill_diff`、己方落后 → `gold_diff`、对位优劣 → `matchup`、
   塔差 → `tower_diff`、有大 → `has_ult_ready`）单独成条，条件写进 `extra`。
3. 前提判不了的（对方开野色、暴君倒计时、控制交没交、队友位置、对方有几个刺客……）
   和同一节点里其它分支**按「可接受动作集合」合并**：集合有交集就合并，`expect` 取并集，
   `why` 里说明合并了哪几条。
4. 动作里只有一部分能判定时（惩击时机、装备、喊话、指挥），判不了的部分写进 `hint` / `why`；
   整条都判不了 → 放进 `SKIPPED` 并写明理由。
5. 同一个节点里两条规则若期望**同一动作事实**且取值不重叠，必须用 `extra` / `guards` 互斥开；
   互斥不开就合并。
6. `subject` 要比 `expect` **宽**：它圈的是「这件事发生在哪一类操作里」，比 expect 宽才有判断力。
7. 严重度：`high` = 关键取舍（开不开团、控不控龙、什么时候转线）；`medium` = 节奏与效率。

英雄层比分路层多一层：每条模式要带 `hero` 字段，id 要能区分英雄（现有格式
`HOK-MODE-JGL-T0000-01`，英雄层要另起一套不撞车的 id 段）。
`SKIPPED` 必须写明理由，生成器会检查每条分支都有归宿。

#### 4. 生成器层

新增 `tools/build_hok_hero_modes.py`，仿 `tools/build_hok_modes.py`：

- 合成 `app/seed/hok_hero_modes.json`（**生成产物，同样不许手改**）。
- 跑全套校验：逐字引用、分支全覆盖（`RULES` ∪ `SKIPPED` == 攻略全部分支）、
  同格互斥、`expect.values` 在动作词表内、id 唯一、hero 字段非空。
- **不要动 `app/seed/hok_modes.json`**（分路层 85 条是历史资产，保持原样）。

#### 5. 接线与文档

- `tests/test_mode_library.py` 增加英雄层断言（模式数、引用逐字、互斥、分支全覆盖、id 唯一）。
- `load_seed_library`（`app/seed/__init__.py`）同时加载 `hok_hero_modes.json`。
- 标注页 / 模式库页要能按英雄筛选（README §8 说模式分「分路 → 英雄 → 时期」三层）。
- 更新 README §8 的四层表格，加英雄层一行，并写明：
  **英雄层原文位于 `\hok-s45-hero-trees\`（五份分路文件），
  入库后的唯一真源为 `app/seed/sources/hok_s45_hero_decision_tree.txt`。**
- 英雄层规则若与分路层冲突，**不要删旧的**。旧规则只退役不删除：
  退役清单落 `app/seed/hok_retired.json`，`load_seed_library` 自动标 `deprecated`，
  历史对局仍按当时规则复盘。

### 验收

```
.\.venv\Scripts\python.exe -m pytest tests\ -q
python tools\verify_version_flow.py
python tools\verify_video_contract.py
python tools\verify_guide_collect.py
python tools\verify_rule_review.py
python tools\verify_hok_hero_modes.py      # 新增
```

基线（改动前）：pytest **414 通过 + 4 跳过**，四个 verify 脚本共 **113 项断言**。
改动后**不得回归**，并补上英雄层自己的断言。

### 硬约束（踩过的坑）

- 不要手改任何生成产物（`hok_modes.json` / `hok_hero_modes.json`）。
- 不要为了省事把英雄层降级成手写 JSON——那会毁掉「原文可复核」这个属性。
- 引用逐字，改一个字就报错。
- 动作词表里没有的取值，一律不许写进 `expect.values`；确实需要新词，先在
  `app/adapters/honor_of_kings.py` 里加，并补中文标签。
- 前提要落到可观测事实。**不要把「我方处于劣势」「经济差」这类结果类判断直接当前提**——
  素材里已经全部改写成地图事实（塔是否已破、谁阵亡、技能是否已交、敌方在哪个位置），映射表
  同样只能引用这些前提。
- 打包 exe 前先停 `MatchAnalyzer.exe`，否则 `dist\MatchAnalyzer\MatchAnalyzer.exe` 被占用报
  `PermissionError [WinError 5]`。
- 打包后 `dist\` 被清空重建，**exe 的数据目录也会没**，需要重灌/重传视频。

---

## 提示词正文结束

## 附：素材与现有库的位置速查

| 东西 | 位置 |
| --- | --- |
| **五分路英雄操作模式（本次入库的素材）** | `\hok-s45-hero-trees\` |
| 现有分路层原文（唯一真源，先例） | `app\seed\sources\hok_s45_lane_decision_tree.txt` |
| 分路层解析器 | `tools\hok_decision_tree.py` |
| 分路层映射表 | `tools\hok_spec\{jungle,mid,side,carry,roam}.py` |
| 分路层生成器 | `tools\build_hok_modes.py` |
| 分路层产物（生成物，勿手改） | `app\seed\hok_modes.json`（85 条） |
| 退役清单 | `app\seed\hok_retired.json` |
| 动作词表 | `app\adapters\honor_of_kings.py` |
| 约定记录 | `README.md` §8 |
| 回归测试 | `tests\test_mode_library.py` |
