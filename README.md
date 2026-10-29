# 基于攻略规范的个人对局执行度分析系统（Firstvirson_BYG）

把《王者荣耀》等游戏的文字攻略 / 操作模式转成可判定的规则，对个人对局录屏逐时段分析
「哪一段没照着做、应该怎么做」，生成可回看原视频片段的复盘报告。

- **分层架构**：采集层（OpenCV 抽帧 / 时钟 / 小地图）→ 规则层（DSL / 准入校验 / 模式库）
  → 内核（上下文 / 匹配器 / 归因）→ 报告层（中文复盘 + 导出）→ FastAPI 接入。
- **多模态识别**：抽帧后交给视觉大模型读分路、英雄、胜负与各时段操作；
  支持 OpenAI 兼容接口（如火山方舟、阿里云百炼）。
- **可判定性优先**：识别置信度分级，画面读不出来的前提不硬判，避免冤枉玩家。

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

打开 http://127.0.0.1:8000 ，在「模型配置」页填入视觉模型的 API Key 后即可上传录屏分析。

## 目录

- `app/` 应用主体（adapters / collection / core / report / rules / templates / static）
- `tools/` 模式库构建与校验脚本
- `tests/` 测试
- `mode-library/`、`app/seed/` 王者 S45 操作模式库

---

组名：Todo
成员：李云帆，李佳轩，全家宝
