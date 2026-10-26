# 操作模式库

这里放**操作模式**（也就是攻略里的「该做什么」）。它是**知识**，不是设置：
有版本、有人 review、要能 diff、要能回滚 —— 所以它独立成一个 git 仓库，
而不是锁在应用代码里。

## 应用怎么找到它

按顺序找，找到就停（见 `app/rules/library_source.py`）：

1. 环境变量 `MODE_LIBRARY_PATH`（指向一个目录）；
2. `<数据目录>/mode-library` —— exe 旁边的数据目录，把仓库 clone 到这儿就生效；
3. `<项目>/mode-library` —— 就是**这个目录**（开发机上默认用这份）；
4. `app/seed/` —— **内置兜底**。仓库拉不到时不能让产品开天窗。

界面上「操作模式库」那一页会写明当前用的是哪一份、是内置还是你的仓库。

## 想改一条模式

1. 在这里改 JSON（见下「一条模式长什么样」）；
2. 提交：`git commit -am "…"`；
3. 推到你自己的 GitHub：`git remote add origin <你的仓库地址>` → `git push`；
4. 在应用里重新载入一次操作模式库（模式库页有「载入」按钮），报告立刻按新的走。

## 换一台机器

```bash
git clone <你的仓库地址> <数据目录>/mode-library
```

## 一条模式长什么样

```json
{
  "id": "HOK-MODE-JGL-T0000-01",
  "game": "hok",
  "version": "S45",
  "title": "打野 0:00：反向开局刷野，1:50 前到 4 级",
  "category": "打野·前期",
  "trigger": { "all": [{"fact": "time_min", "op": "between", "value": [0, 0]}] },
  "expect": { "fact": "objective_action", "op": "in", "value": ["farm_jungle", "invade_jungle"] },
  "premise": "敌方打野为红开（已知或从选人预选推测）；……",
  "action": "己方蓝开反向开局，蓝 BUFF → 蜥蜴 → 狼 → 红 BUFF → 小鸟，1:15 至 1:50 到 4 级；……",
  "outcome": "避开前期野区碰撞，4 级时可直接到对抗路形成夹击，野区均势",
  "deviation_hint": "敌方红开就己方蓝开（蓝 BUFF→蜥蜴→狼→红 BUFF→小鸟），……",
  "rationale": "原文前提：……",
  "severity": "high",
  "status": "active",
  "expect_scope": "at_least_once",
  "subject": { "fact": "objective_action", "values": ["farm_jungle", "invade_jungle", "clear_vision_sprite"] },
  "tags": ["打野", "前期", "开局"],
  "source": {
    "guide": "王者荣耀 S45「月照长安」分路决策树",
    "locator": "打野 / 0:00 节点 / 敌方打野为红开（已知或从预选推测）",
    "quoted": "动作：己方蓝开反向开局，蓝 BUFF → 蜥蜴 → 狼 → 红 BUFF → 小鸟，1:15 至 1:50 到 4 级\n……"
  }
}
```

报告里的四件套就是从这些字段拼出来的：

| 报告上写的 | 来自哪个字段 |
| --- | --- |
| 看到了什么 | **模型**读的（`probe_moments`），不是模式里的 |
| 该做什么 | `action` / `deviation_hint` |
| 为什么有用 | `outcome` |
| 证据片段 | 模型给的时间戳切出来的 |

**`outcome` 是收益，不许应用自己另写** —— 收益必须来自攻略原文，
否则就成了应用在替攻略编话。

## 改的时候注意

- **`source.quoted` 不能空**：模式必须能指回原文（准入检查会拦）。
- **`id` 不能重复**：id 是规则与历史对局的关联键。
- **`expect.value` 必须是适配器认的动作取值**：见
  `app/adapters/honor_of_kings.py` 的 `MOBA_ACTION_VALUES`。
  写了不认的值，导入时会被拒（strict 导入）。
- **`subject`** 是「哪些动作算这一类」的白名单：不在白名单里的动作**不算违反**，
  模式不会因为「你没做某个没关系的动作」而报偏差。

## 和内置那份的关系

`app/seed/` 是**内置兜底**，这份是可编辑副本。
改了内置之后要同步过来：

```bash
python tools/sync_mode_library.py
```

（同步只覆盖文件，不会动你的 git 历史。）
