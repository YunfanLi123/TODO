# 用视觉语言模型（VLM）分析王者荣耀对局录屏：外部技术调研

调研日期：2026-10-24
调研范围：GitHub、HuggingFace、arXiv
取证方式：
- GitHub：直接调 `api.github.com`（本机可达），star 数与 `license.spdx_id` 均以该 API 当日返回为准。
  ⚠️ **注意**：本机连到的 GitHub 是个**过滤后的快照**，部分真实存在的仓库会返回 404（例如 `myBoris/wzry_ai`、`jstzwj/HonorOfKingsObjectDetection`、`jayleicn/moment_dense_trainer` 均取不到）。这类条目在下文**按来源标注**（来自本项目上一轮直读源码 / OpenDataLab 元数据 / 第三方镜像站），**不把 404 当作「仓库不存在」**。
- arXiv：`arxiv.org` 页面可直连，论文标题/作者/摘要逐条核对。
- HuggingFace：**本机 HTTP 超时、HF API 不可达**。HF 相关条目靠检索引擎抓到的模型卡/数据卡原文取证；凡没读到卡原文的，许可证一栏写「未验证」，不补全。

> 本轮未逐条核验各仓库 issue 区的复现记录，下文「已被验证的证据」只写**可点开复核**的东西：star 数、会议/期刊、论文链接、论文自报的评测分数。凡论文自报的分数都标注「论文自报」，不当成第三方复现。

---

## 0. 先和已有调研对齐

本文件是 `docs/screen-action-recognition-survey.md`（同日，CV 路线）与 `docs/recognizer-design.md`（识别层设计）的**互补件**，不是替代件。三条既有结论直接决定本轮选型：

| 既有结论 | 出处 | 对本轮选型的约束 |
| --- | --- | --- |
| 「没有现成的『王者录屏 → 玩家操作』开源模型或数据集」；能找到的只有「读服务端状态的 RL 环境」和「在王者截图上打靶的 YOLO」 | `docs/screen-action-recognition-survey.md` §0 | MOBA 垂直方案确实空缺 → 必须用**通用 VLM + 我们的模式库**补，而不是等现成方案 |
| 唯一公开的王者目标检测数据集 `jstzwj/HonorOfKingsObjectDetection` **README 明写「仅供学习研究使用，禁止商业用途」** | 同上 §表 | 训练/评测素材这条路上，王者域**没有可商用数据**；只能力求不依赖任何王者域素材 |
| 我们实测三段录屏**全部没有音轨**（mp4 只有 `vide` handler） | 同上 §0.4 | VLM 路线**只能走视觉分支**；一切依赖击杀播报/技能语音的方案（含 Proact-VL 的音频分支）对我们无效 |
| 分路（打野/中单/…）是**对局级**事实，「视频里看不出自己在打什么位置」 | `app/adapters/base.py` §`position_values` | **VLM 不能替代用户选分路**。模式库大量规则带 `position` 条件，位置错了规则会整批不触发或整批误判 |

---

## 1. 结论先行：三条最值得走的路线

1. **路线 A — 本地 Qwen3-VL 做「窗口级动作事件抽取」**（首选）：权重现成、Apache-2.0 可商用、**不需要训练**，输出直接填进我们已经存在的 `DetectedAction` / `DetectedFact` 契约，规则对账层（`app/core/matcher.py`）一行不用改。
2. **路线 B — 规则库定向问答（模式级对账）**：把模式库 JSON 当作提问清单逐条问模型，一次调用同时判几十条模式，输出形状与现有 `Verdict` 的 `satisfied/deviated/undetermined` 三态一一对应，**不确定性天然落进已有的 `missing_facts → undetermined` 语义**。
3. **路线 C — 时间定位优先的两段式**：先用带时间戳定位能力的 VLM 整段粗定位可疑片段，再把片段交给 A/B 精读。Qwen3-VL 自带「文本-时间戳对齐」能力，不必训练。

一句话理由：这三条的共同点是**都不需要我们训模型**，并且都把 VLM 的输出**收敛到我们已经定义好的数据契约里**（`app/collection/video.py` 的 `VideoRecognizer`），因此新方案的产出仍然可复查、可定位到秒、可以喂给现有规则引擎；而微调路线（Proact-VL / Video-Thinker / RefTrace 那一类）虽然效果更好，但算力成本（8×H100 量级）把我们排除在外。

---

## 2. 候选清单

### 2.1 通用视频 VLM 权重（可直接用，无需训练）

| 名称 | 类别 | 链接 | 许可证与商用 | 已被验证的证据 | 能不能接进我们的栈 |
| --- | --- | --- | --- | --- | --- |
| **Qwen3-VL-8B-Instruct** | 通用视频 VLM | [HF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) / [GitHub](https://github.com/QwenLM/Qwen3-VL) | **Apache-2.0，可商用**：GitHub 仓库 `license.spdx_id` = Apache-2.0（API 确认）；模型卡许可由第三方模型索引页记录为 apache-2.0（HF 直连不可达，**未读到卡原文，落地前请再核一次**） | GitHub ★20,114（2026-10-24 API 返回），最后推送 2026-01-30；模型卡称原生 256K（可扩 1M）、支持视频输入、「Text–Timestamp Alignment」提供**秒级事件定位**、可处理小时级视频 | **能**。Windows 本地用 transformers 跑 4bit（模型卡给了 Transformers 调用示例）；**llama.cpp 的视频输入支持本轮未验证**，若要走 GGUF 路线需先实测图像输入、再确认视频输入 |
| **VideoChat-Flash** | 长视频 VLM | [GitHub](https://github.com/OpenGVLab/VideoChat-Flash) | 仓库 **MIT**（API 确认）；HF 权重许可未单独核验 | GitHub ★531，ICLR 2026；分层压缩做长上下文 | 能（备选底座，用于超长视频） |
| **LongVU** | 长视频 VLM | [GitHub](https://github.com/Vision-CAIR/LongVU) | 仓库 **Apache-2.0**（API 确认） | GitHub ★435，ICML 2025 | 能（备选） |
| **TimeChat** | 时间敏感 / 时间戳定位 VLM | [GitHub](https://github.com/RenShuhuai-Andy/TimeChat) | 仓库 **BSD-3-Clause**（API 确认） | GitHub ★429，CVPR 2024 | 能；专精「定位到具体时间点」，适合路线 C 的定位段 |
| **VideoLLaMA3** | 通用视频 VLM | [GitHub](https://github.com/DAMO-NLP-SG/VideoLLaMA3) | 仓库 **Apache-2.0**（API 确认） | GitHub ★1,191 | 能（备选） |
| **Video-R1** | RL 视频推理 | [GitHub](https://github.com/tulerfeng/Video-R1) | 仓库 license 字段**为空=未声明**；商用前必须自行确认 | GitHub ★895，最后推送 2025-12-14 | **谨慎**。权重是 RL 后产物，许可与「我们的用途」是否兼容未验证 |
| **TinyLLaVA-Video-R1** | 小模型视频推理 | [GitHub](https://github.com/ZhangXJ199/TinyLLaVA-Video-R1) | 仓库 **Apache-2.0**（API 确认） | GitHub ★116 | 能（显存受限时的小底座） |
| **Video-Thinker-7B** | 「定位→描述→推理」CoT | [GitHub](https://github.com/DeepExperience/Video-Thinker) / [HF 权重](https://huggingface.co/ShijianW01/Video-Thinker-7B) / [论文](https://arxiv.org/abs/2510.23473) | 仓库 license 字段**为空=未声明**；**不建议商用依赖** | GitHub ★8；论文自报 Video-Holmes 43.22%、VRBench 80.69%、时序定位 mIoU 48.22%（基座 27.47%）、Recall@0.3 79.29%；用 `<time>/<caption>/<think>` 结构化 CoT | **能，但只借格式不借权重**：它的输出格式（先定位时间窗→再描述画面→再推理）正是我们要的 prompt 模板；权重本身是 Qwen2.5-VL-7B 经 SFT+GRPO 训出来的问答模型，不适合当通用识别器 |

### 2.2 游戏域模型与基准

| 名称 | 类别 | 链接 | 许可证与商用 | 已被验证的证据 | 能不能接进我们的栈 |
| --- | --- | --- | --- | --- | --- |
| **VideoGameBunny-V1** | 游戏画面专用 VLM（**图像域，非视频**） | [HF](https://huggingface.co/VideoGameBunny/VideoGameBunny-V1) / [论文](https://arxiv.org/abs/2407.15295) / [项目页](https://videogamebunny.github.io/) | 数据集 [asgaardlab/VideoGameBunny-Dataset](https://huggingface.co/datasets/asgaardlab/VideoGameBunny-Dataset) 卡头写明 `license: mit`；**模型卡许可证未验证** | 论文自报 185,259 张图 / 413 款游戏 / 389,565 指令对 / 136,974 张图的 16 元素 JSON；7B 级模型在游戏域超过参数量 4 倍的 LLaVA-1.6-34b | **部分能**。它读的是游戏**截图**——正好是「抽帧」那一步的产物。可作为单帧判定的辅助小模型（判 HUD 元素），但**不能替代视频理解** |
| **VideoGameQA-Bench** | 游戏视频 QA 基准 | [HF 数据集](https://huggingface.co/datasets/taesiri/VideoGameQA-Bench) / [论文](https://arxiv.org/abs/2505.15952) / [项目页](https://asgaardlab.github.io/videogameqa-bench/) | **未验证**（HF 直连不可达，卡原文未读到） | NeurIPS 2025 D&B；覆盖图像与视频的视觉单元测试 / 回归测试 / needle-in-a-haystack / glitch 检测 / bug 报告生成 | **不能直接用**（我们要的是「操作序列」不是 QA 对）。可作**评测集**：用它的任务形式给我们自己的抽取器出考题 |
| **Proact-VL**（Microsoft） | 主动式游戏视频 LLM + 游戏视频基准 | [GitHub](https://github.com/microsoft/AnthropomorphicIntelligence/tree/main/Proact-VL) / [OpenReview](https://openreview.net/forum?id=k9PKgV0L4C) / [权重 oaaoaa/proactvl_base_qwen3vl](https://huggingface.co/oaaoaa/proactvl_base_qwen3vl) / [LiveGamingBenchmark](https://huggingface.co/datasets/oaaoaa/LiveGamingBenchmark) | 仓库 **MIT**（API 确认）；HF 权重/数据许可**未验证** | ICML 2026；底座含 **Qwen3-VL-8B-Instruct**；评测指标含 **F1 / Time-Diff / PAUC**（后两个正是「答案对不对 + 时间戳偏不偏 + 时序定位准不准」）；训练数据覆盖 LOL / Minecraft / CSGO / 黑神话等 10 款游戏；论文自报训练成本 8×H100 约 24 小时 | **不能直接用**（需训练）。**但值得抄两样东西**：①它的**指标定义**（Time-Diff、PAUC）可以直接变成我们抽取器的自评指标；②它的**提示词范式**（solo 解说 / 协同解说 / 玩家引导三分支）里的「玩家引导」分支最接近我们要做的事。注意它是**音视频**模型，而我们没有音轨 |
| **GameplayQA** | 决策密集型 POV 游戏视频基准 | [GitHub](https://github.com/HATS-ICT/GameplayQA) / [主页](https://hats-ict.github.io/gameplayqa) | 仓库 **MIT**（API 确认） | ACL 2026；★1 | 能（**评测参考**）。「决策密集 + POV 同步多视频」的问题形态跟我们「一局录像 + 规则库」最像 |
| **GliDe** | 开放式游戏 glitch 检测（agentic 推理 + 时间定位） | [GitHub](https://github.com/SandyyyZheng/GliDe) | 仓库 **MIT**（API 确认） | ACM MM 2026；★3，最后推送 2026-10-18 | 能（**架构参考**）。它的「先用 agent 决定看哪一段，再逐段核验」是两段式的现成范式 |
| **RefTrace** | 视频「规范裁决」（带规则库的判定） | [GitHub](https://github.com/sreevadde/reftrace) | 仓库 license 字段**为空=未声明** | ★1；论文自述用 Game-State Temporal Hypergraph + 6 类动作语法 + 两阶段 SFT→GRPO 做 NFL 判罚裁决 | **概念上最像我们**（拿规则去判视频里的行为对错），但需要训练。**当设计蓝本读，不当依赖** |

### 2.3 游戏 / 电竞数据集（带时间戳标注优先）

| 名称 | 类别 | 链接 | 许可证与商用 | 已被验证的证据 | 能不能接进我们的栈 |
| --- | --- | --- | --- | --- | --- |
| **Game-MUG** | LoL 电竞多模态态势理解 + 解说生成 | [论文](https://arxiv.org/abs/2404.19175) | **未验证**（论文页可读，未找到官方仓库；许可与数据可得性均未核验） | 论文自报：2020–2022 年 YouTube/Twitch 的 216 场 LoL 直播；文本 + 解说音频 + **时间序列事件日志**（击杀 / 非史诗野怪 / 防御塔 / 小龙 / 镀层 / 水晶枢纽，共 15,221 个事件）；70,711 转录片段、370 万观众聊天实例 | **不能当依赖**（许可与下载渠道未验证）。**当设计蓝本**：它的「事件日志按时间戳对齐」正是我们想要的 `probe_moments` 形状 |
| **MMOU**（NVIDIA） | 长视频多模态推理基准（含 Video Games 领域 + 证据时间戳） | [HF 数据集](https://huggingface.co/datasets/lililiy/MMOU) / [视频集](https://huggingface.co/datasets/sonalkum/MMOU-Videos) / [论文](https://arxiv.org/abs/2603.14145) | **未验证** | 数据卡自报：15,000 条 QA / 9,038 个网页视频（平均 711.6 秒，7–7,255 秒）、10 大领域含 **Video Games**、每题带 `start_time`/`end_time` **证据时间戳**；**当前发布不含 ground-truth 答案**；论文自报最强开源模型 Qwen3-VL-32B 仅 44.0（人类 84.3） | **不能当评测集**（当前没有答案）。但「带证据时间戳的长视频问答」这个任务设计可以直接抄 |
| **QVHighlights** | 自然语言查询式时刻检索 | [论文](https://arxiv.org/abs/2107.09609) / [官方仓库](https://github.com/jayleicn/moment_dense_trainer) / [OpenDataLab 条目](https://opendatalab.com/OpenDataLab/QVHighlights) | OpenDataLab 标注 **CC BY-NC-SA 4.0 → 禁止商用 + 相同方式共享**（官方仓库 LICENSE 本轮未读到，**按最严口径处理**） | 论文自报 10,000+ YouTube 视频、带时间段标注 | **不能商用**。仅在需要「时刻检索」能力时参考标注格式 |
| **MMAI**（Ubisoft） | 游戏视频多轮指令数据集 | 候选地址 `https://huggingface.co/datasets/VideoGameBench/MMAI` | **未验证** | **本轮未取到任何可核验证据**（HF 直连超时、GitHub 无 `VideoGameBench` 组织索引） | **未验证 → 不列入依赖**。若后续要用，必须先读到数据卡原文的许可条款 |
| **LiveGamingBenchmark**（Proact-VL 配套） | 多款游戏视频基准 | [HF](https://huggingface.co/datasets/oaaoaa/LiveGamingBenchmark) | **未验证** | 仓库 README 列出的目录结构含 LOL / Minecraft / CSGO / 黑神话 / 艾尔登法环等 10 款游戏 | **不能当依赖**（许可未验证）。可作「我们的抽取器该覆盖哪些游戏类型」的清单参考 |

### 2.4 教练 / 复盘类系统

| 名称 | 类别 | 链接 | 许可证与商用 | 已被验证的证据 | 能不能接进我们的栈 |
| --- | --- | --- | --- | --- | --- |
| **forging** | AI 游戏教练（上传视频 → 带时间戳的建议） | [GitHub](https://github.com/lucaslencinas/forging) | README 声明 **MIT**（GitHub API `license` 字段为空，未在仓库根目录核到 LICENSE 文件） | ★2，最后推送 2026-02-09；README 自报：上传 MP4（≤700MB/30 分钟）→ **Observer / Validator 双智能体** → **带可点击时间戳的 coaching tips** → Validator 复核并按置信度**剔除幻觉建议**；支持 AoE2 与 CS2，架构与游戏无关；后端 FastAPI + 前端 Next.js | **架构上高度可抄**（Observer/Validator、置信度门控、带时间戳建议），但它**强依赖 Gemini 云 API**，不满足「本地跑」。**我们抄结构、用本地权重替换模型** |
| **League-of-Legends-AI-Coach** | 实时 LoL 打野教练 | [GitHub](https://github.com/sean-qin-usa/League-of-Legends-AI-Coach) | license 字段**为空=未声明** | ★1；README 自报用 **Riot 官方 API** 取数据（非视频），300+ 局训练、对高分段决策 80%+ 准确率 | **不能接**。依赖官方 API 对局数据，王者没有等价物；只作「教练该输出什么」的对照 |
| **Honor-of-Kings_RolePlay（峡谷小狐仙）** | 王者多模态角色扮演 | [GitHub](https://github.com/YongXie66/Honor-of-Kings_RolePlay) | 仓库 **MIT**（API 确认） | ★52；基于 InternLM2 + RAG + ASR + TTS + 数字人，覆盖 100+ 英雄知识 | **不能接**。唯一直接以王者为对象的多模态项目，但**不看对局、只答英雄设定与台词** |
| **MOBA-AI-Gamer** | YOLOv5 + OCR + RL 打 LoL | [GitHub](https://github.com/adrian27513/MOBA-AI-Gamer) | license 字段**为空=未声明** | ★5，最后推送 2022-12-26；README 自报 mAP@0.5 0.83，目标是「自动打 LoL」 | **反面参照**（见 §4） |
| **esports-coaching-tool** | CNN 逐帧判游戏状态 | [GitHub](https://github.com/ESPort-Coach/esports-coaching-tool) | license 字段**为空=未声明** | ★0；Apex Legends 帧分类（Gameplay/Looting/Map/Interface），输出状态时长分布 | **反面参照**（见 §4） |

---

## 3. 三条推荐路线的详细设计

### 3.0 共同的落点：现有契约已经预留好了

新方案**不需要改动内核**。现有 `app/collection/video.py` 已经定义好：

- `VideoRecognizer`（抽象基类，`app/collection/video.py:270`）两个方法：`detect_round_boundaries()`（廉价全片门控）与 `recognize_rounds()`（昂贵精细识别）。
- `RecognizedRound`（`app/collection/video.py:163`）：带 `facts: List[DetectedFact]`、`actions: List[DetectedAction]`、`probe_moments`、`probe_absent`、`video_time`、`gate_confidence`。
- 置信度三态阈值（`app/collection/video.py:49-63`）：事实 `0.30/0.75`、动作 `0.40/0.75`、门控 `0.50`。
- `to_raw_match()` 把识别结果降级成内核可消费的对局数据。

**所以新工作的产物就是：一个新的 `VideoRecognizer` 实现。** 下面三条路线共用这个外壳，差别只在「一次调用问什么、问多少帧」。

```
                    ┌─ 现有 CV 门控（保留）────────────────────────┐
视频 ──► detect_round_boundaries() ──► GateFrame[]（廉价全片）
                    │                    │
                    │                    ▼
                    │            关键帧 / 时间窗
                    │                    │
                    │        ┌───────────┴───────────┐
                    │        │  路线 A：动作事件抽取   │
                    │        │  路线 B：模式级对账     │
                    │        │  路线 C：先粗定位再精读  │
                    │        └───────────┬───────────┘
                    │                    ▼
                    │        RecognizedRound(facts/actions/probe_moments)
                    │                    │
                    └────────────────────┼──────────────────────┐
                                         ▼                      ▼
                            HonorOfKingsAdapter.normalize()   （不改）
                                         ▼
                            app/core/matcher.py 规则对账      （不改）
                                         ▼
                                 app/report/builder.py         （不改）
```

---

### 路线 A（首选）：本地 Qwen3-VL 做窗口级动作事件抽取

**输入**
- 门控帧序列：现有 `detect_round_boundaries()` 的 `GateFrame[]`（带 `timestamp` 秒）。
- 每窗关键帧：每窗 4~8 帧，720p；**必须裁剪 HUD 区域并放大**（王者血条/技能图标在小地图与 HUD 上，字号极小，整帧丢给模型等于没给）。

**输出**（受控 JSON，不许自由文本）

```json
{
  "events": [
    {"t": 512.3, "t_end": 519.0,
     "kind": "objective_action",
     "value": "take_dragon",
     "evidence_frame": "f_0087.jpg",
     "evidence_text": "左下角出现小龙血条与击杀提示"}
  ]
}
```

- `kind` ∈ `objective_action` / `positioning_action` / `teamfight_action`（**直接复用 `app/adapters/honor_of_kings.py:270` 的 `MOBA_ACTION_VALUES` 词表**）。
- `value` ∈ 该词表的封闭取值集合（`take_dragon`、`farm_jungle`、`last_hit`、`contest_dragon`、`engage`、`push_wave` …）。
- 词表外的取值一律丢弃并计入 `probe_absent`/告警——这是防幻觉的第一道闸。

**对接我们哪个模块**
- 新增 `app/collection/vlm_recognizer.py`，实现 `VideoRecognizer.recognize_rounds()`。
- 内部把 JSON 事件转成 `RecognizedRound(actions=[DetectedAction(kind, value, label, confidence, ts, source="probe")], probe_moments=[{"t":.., "why":..}])`。
- `confidence` **不要让模型自报**：用「同一窗抽样 2 次，词表命中是否一致」+「是否给出可核对证据帧」两个客观量合成，映射到既有三态阈值（`<0.40` 不采信 → 该事实进 `missing_facts` → 判定为 `undetermined`）。这一步是这条路线的成败关键：模型自报置信度没有校准意义，而我们的报告体系要求「不可信就明说不可信」。
- `to_raw_match()` → `HonorOfKingsAdapter.normalize()` → `app/core/matcher.py` **全部不改**。

**代价**
- 权重：Qwen3-VL-8B-Instruct（Apache-2.0）。Windows 本地建议 4bit 量化，显存 ~6–8GB；纯 CPU 不可行。
- 吞吐：一局 15–25 分钟、约 60–120 个窗、每窗 6 帧，**单局推理时间必须在实测后重新定预算**（本轮无实测数据，不给数字）。
- 提示词工程是主要工作量：词表、证据要求、否定情形（「画面里没看到开龙」→ `probe_absent`）都要写进 system prompt。

**风险**
1. **王者 HUD 文字太小** → 必须 ROI 裁剪 + 放大；这是我们相对通用 VLM 的最大增益点，也是最容易翻车的地方。
2. **模型幻觉词表外动作** → 双重保险：JSON schema 强约束 + 词表白名单校验。
3. **动作 ≠ 意图**：「补刀」需要看小兵血条，「开团」需要看谁先放控制技能。单帧可能判不准 → 每窗多帧投票是必须的。
4. **经济差/等级这类量 VLM 读不准** → 这些事实继续由现有 CV/用户录入提供，**不要交给 VLM**。

---

### 路线 B：规则库定向问答（模式级对账）

**输入**
- 本局**过滤后**的模式子集：用模式库里的 `trigger` 条件做预筛（`position` / `phase` / `time_min` 字段都在 JSON 里，可直接过滤），把 1696 条压到 20~60 条。
  > 条数口径：本轮实测 `app/seed/` 下三个模式文件合计 **1696 条**（`hok_hero_modes.json` 1596 + `hok_modes.json` 84 + `hok_bp_modes.json` 16）。与立项口径的「1680 条」有出入（应为统计范围不同：是否含 BP 层 16 条），**以代码里的实际文件为准**。
- 每条模式对应的：标题、`premise`（前提）、`action`（原文动作）、`expect`（期望动作取值）——**注意 `expect` 本身就是词表取值**，可以直接翻译成「玩家有没有做 X」。
- 对应时间窗的关键帧。

**输出**（每条模式一条）

```json
{"mode_id": "HOK-MODE-JGL-T0000-01",
 "verdict": "deviated",
 "evidence_t": 95.0,
 "evidence_frame": "f_0031.jpg",
 "why": "0:00–2:00 打野在野区，但未清野怪；画面里只有移动没有攻击野怪"}
```

`verdict` 三态直接对齐 `app/core/matcher.py` 的 `satisfied` / `deviated` / `undetermined`。

**对接我们哪个模块**
- 规则来源：`app/rules/store.py`（读库）+ `app/seed/hok_modes.json` / `hok_hero_modes.json` / `hok_bp_modes.json`。
- 规则 → prompt 的翻译器是新增的小模块；**这一步要用我们自己的字段名**（`title`/`premise`/`action`/`outcome`/`expect`），不要让模型自由发挥。
- 输出 → `RoundRecord.actions` + `probe_moments`；模型答不上来的模式 → 对应事实进 `missing_facts` → 现有 `evaluate_round()` 自动给出 `undetermined`。
- 报告侧 `app/report/builder.py` 不改。

**代价**
- 一次调用覆盖多条模式，**token 效率最高**（相对路线 A 逐动作抽取，同样的帧能问更多问题）。
- 但 prompt 变长（几十条模式的原文），可能超出小模型的指令跟随能力 → 需要先做「按 category 分批」。

**风险**
1. **模式库的 `trigger` 大量依赖我们读不出的事实**（经济差、等级、核心大招是否就绪）→ 这些模式**不能问模型**，只能让 CV/用户录入提供 → 需要一个「哪些模式可交给 VLM」的判定器（按 `expect.fact` 是否属于动作事实来筛，参照 `app/adapters/base.py` 的 `action_fact_names`）。
2. **模型会顺着 prompt 里的「期望动作」编答案**（提示里写了「应该刷野」，模型倾向于说「在刷野」）→ 提问时**不给出期望动作**，只给模式标题与前提，期望动作留到后置校验。
3. **「模式没做到」和「模式不适用」是两件事**——现有 `Subject` 机制（`app/core/dsl.py:230`）已经在区分，VLM 侧也要显式区分，否则会重演「冤枉人」的老问题。

---

### 路线 C：时间定位优先的两段式（先定位，后精读）

**输入**
- 整局视频（或大段）低帧率过一遍（Qwen3-VL 支持视频输入 + 文本-时间戳对齐；或用 TimeChat 这类专精时间戳的模型）。
- 或者：先用**低成本事件检测**（画面突变、开团音效替代品如血条消失）挑出候选窗。

**输出**：可疑片段列表 `[{start, end, caption}]`——即 Proact-VL / Video-Thinker 的 `<time>…</time><caption>…</caption>` 结构。

**对接我们哪个模块**
- 粗定位结果直接落进 `RecognizedRound.probe_moments`（带 `t` 与 `why`）与 `GateFrame.is_boundary`，即**升级现有门控**：现在的门控找「时间片边界」，VLM 门控找「有事发生的时刻」。
- 拿到候选窗后，把窗内帧喂给路线 A 或 B 精读。
- 报告侧「看这一段」定位到具体秒 → 与现有 `RoundRecord.video_time` 语义一致。

**代价**
- 整局视频一次性推理的 token 成本最高（帧数 × token），但比逐窗询问的总成本低（一次看全片，定位若干窗）。
- 需要 `Qwen3-VL` 级别模型的视频输入支持；8B 在 Windows 上跑整局视频要实测显存与耗时。

**风险**
1. 王者画面「安静期」很长（回城、补兵），模型可能把整片都标成「有事件」→ 必须收紧成「与规则库里某条模式相关的事件」，否则筛选失效。
2. **定位精度要单独评测**：建议直接采用 Proact-VL 的 Time-Diff / PAUC 定义作为我们自己的指标（见 §2.2），否则「定位到秒」只能靠感觉。

---

### 路线 D（列出但建议不做）：微调路线

Proact-VL（8×H100 ≈ 24 小时，论文自报）、Video-Thinker（SFT + GRPO）、RefTrace（两阶段 SFT→GRPO）都证明微调能显著提升。**但我们明确不接 PyTorch 训练，硬件也不匹配**，因此本轮不推荐。若将来要做，最省力的形态是：用 Proact-VL 的 `game_commentary_sft` 或 VideoGameQA-Bench 的标注，**只训一个「窗口 → 动作词表」的分类头**（而不是训整个 MLLM），这样单卡可训、且输出天然落在我们的封闭词表里。注意这条路的训练数据链包含 Ego4D Goal-Step（Proact-VL README 明确要求下载），**Ego4D 系列的许可条款本轮未核验，不要默认可商用**。

---

## 4. 反面清单（看起来相关，但别用）

| 别用 | 为什么 |
| --- | --- |
| **传统 CV 路线**（YOLO / 模板匹配 / HSV 掩膜 / OCR / ONNX 小分类器）：`myBoris/wzry_ai`（327★，本项目上一轮已直读源码）、[`MOBA-AI-Gamer`](https://github.com/adrian27513/MOBA-AI-Gamer)、[`esports-coaching-tool`](https://github.com/ESPort-Coach/esports-coaching-tool) | **我们自己的实验已经证伪**：只能稳定读出时钟、技能冷却、小地图位置这类窄信号；「补刀」「开团」「经济差」读不出来，产出的报告既不可信也没价值。这些项目要么只到「状态分类」这一层，要么需要我们没有的标注 |
| **NVIDIA NitroGen**（[HF](https://huggingface.co/nvidia/NitroGen) / [数据集](https://huggingface.co/datasets/nvidia/NitroGen) / [站点](https://nitrogen.minedojo.org/) / [论文](https://arxiv.org/abs/2601.02427)） | ①**禁止商用**：模型卡写明「This model is for research and development only」，许可为 NVIDIA One-Way Noncommercial License；②**模型卡自述 MOBA 不行**——「works best on games designed for gamepad controls… less effective on games that rely heavily on mouse and keyboard (e.g., RTS, MOBA)」；③它输出的是**手柄动作**（21 维连续摇杆 + 17 位按键），不是战术判断，对复盘没有语义价值 |
| **Ubisoft La Forge 系列资产**（如 [ubisoft-laforge-chord](https://github.com/ubisoft/ubisoft-laforge-chord) 及其 [HF 许可](https://huggingface.co/Ubisoft/ubisoft-laforge-chord/blob/main/LICENSE)） | 许可为 **Ubisoft Machine Learning License (Research-Only - Copyleft)**：「Permitted Purpose: Use for academic or research purposes only. **Commercial use is strictly prohibited.**」，且衍生品必须以同一协议分发。注意 CHORD 本身是 **PBR 材质估计**模型，与游戏视频理解无关（检索时极易误认） |
| **QVHighlights**（[论文](https://arxiv.org/abs/2107.09609)） | OpenDataLab 标注 **CC BY-NC-SA 4.0** → **禁止商用**，且要求相同方式共享 |
| **`jstzwj/HonorOfKingsObjectDetection`**（本项目上一轮已记录） | README 明写「仅供学习研究使用，**禁止商业用途**」，且标签未标完。这是王者域**唯一**公开检测数据集——我们不能依赖它 |
| **MMAI（Ubisoft）/ Game-MUG / MMOU / LiveGamingBenchmark** | **许可证或数据可得性本轮未验证**。未验证 = 不可依赖。可以读它们的**设计**（事件日志格式、证据时间戳字段），不能把它们的**数据**写进我们的路线 |
| **「让 LLM 看完整局视频，出一篇自然语言复盘」** | 产物不可复查：无法定位到具体时间点、无法逐条回溯证据、无法回答「凭什么说我没做到」。而我们系统的核心卖点恰恰是**每条结论都能顺着证据复查回原始数据**（`app/core/matcher.py` 的证据链设计）。这正是我们上一版失败的地方，换个模型不会自动修好 |
| **LoL AI Coach（Riot API 路线）**（[仓库](https://github.com/sean-qin-usa/League-of-Legends-AI-Coach)） | 数据来自 **Riot 官方 API**，王者没有等价数据源；我们要处理的恰恰是「只有录屏」的处境 |
| **gameplay-vision-llm 个人项目**（[仓库](https://github.com/chasemetoyer/gameplay-vision-llm)） | README 自述在 GlitchBench 上 **0% 准确率**（"Current model says 'no glitch' for confirmed GlitchBench samples (0% accuracy)"）——它自己也承认识别能力没跑通，不能当「已被实践验证」的依据 |
| **峡谷小狐仙类王者角色扮演项目**（[仓库](https://github.com/YongXie66/Honor-of-Kings_RolePlay)） | 直接以王者为对象，容易让人误以为「已经有王者多模态方案了」。实际它**不看对局**，只做英雄知识问答与角色扮演，与录屏复盘无关 |

---

## 5. 许可证与商用限制汇总

### 5.1 明确**不能商用**（踩红线 = 项目停摆）

| 资产 | 许可 | 出处 |
| --- | --- | --- |
| NVIDIA NitroGen 模型 + 数据集 | NVIDIA One-Way Noncommercial License；模型卡自述「for research and development only」 | [模型卡](https://huggingface.co/nvidia/NitroGen) / [许可原文](https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf) |
| Ubisoft La Forge 资产（CHORD 等） | Ubisoft Machine Learning License (Research-Only - Copyleft)：「Commercial use is strictly prohibited」；衍生分发须同协议 | [LICENSE 原文](https://huggingface.co/Ubisoft/ubisoft-laforge-chord/blob/main/LICENSE) |
| QVHighlights | CC BY-NC-SA 4.0（非商用 + 相同方式共享），以 OpenDataLab 标注为准 | [OpenDataLab](https://opendatalab.com/OpenDataLab/QVHighlights) |
| `jstzwj/HonorOfKingsObjectDetection`（王者检测数据集） | README 明写「仅供学习研究使用，禁止商业用途」 | 见 `docs/screen-action-recognition-survey.md` ｜ 仓库 https://github.com/jstzwj/HonorOfKingsObjectDetection |

### 5.2 可商用（Apache-2.0 / MIT / BSD-3），可作为依赖

| 资产 | 许可 | 出处 |
| --- | --- | --- |
| Qwen3-VL-8B-Instruct | Apache-2.0（GitHub 仓库 `license.spdx_id` 确认；HF 模型卡原文本轮未直读，落地前复核一次） | [HF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) / [GitHub](https://github.com/QwenLM/Qwen3-VL) |
| VideoChat-Flash | MIT（仓库） | [GitHub](https://github.com/OpenGVLab/VideoChat-Flash) |
| LongVU | Apache-2.0 | [GitHub](https://github.com/Vision-CAIR/LongVU) |
| TimeChat | BSD-3-Clause | [GitHub](https://github.com/RenShuhuai-Andy/TimeChat) |
| VideoLLaMA3 | Apache-2.0 | [GitHub](https://github.com/DAMO-NLP-SG/VideoLLaMA3) |
| TinyLLaVA-Video-R1 | Apache-2.0 | [GitHub](https://github.com/ZhangXJ199/TinyLLaVA-Video-R1) |
| Proact-VL 代码仓库 | MIT | [GitHub](https://github.com/microsoft/AnthropomorphicIntelligence) |
| GameplayQA / GliDe 代码仓库 | MIT | [GameplayQA](https://github.com/HATS-ICT/GameplayQA) / [GliDe](https://github.com/SandyyyZheng/GliDe) |
| VideoGameBunny 数据集 | MIT | [HF 数据集](https://huggingface.co/datasets/asgaardlab/VideoGameBunny-Dataset) |
| Honor-of-Kings_RolePlay | MIT | [GitHub](https://github.com/YongXie66/Honor-of-Kings_RolePlay) |

### 5.3 **未验证**（未读到许可原文，一律按「不可商用」处理，直到人工确认）

Video-R1、Video-Thinker、RefTrace、LoL AI Coach、MOBA-AI-Gamer、esports-coaching-tool、forging（API license 字段为空，仅 README 自述 MIT）、VideoGameBunny **模型卡**、VideoGameQA-Bench、MMAI、Game-MUG、MMOU、LiveGamingBenchmark、VideoChat-Flash 的 **HF 权重**、Ego4D 系列。

---

## 6. 参考链接全表

**通用视频 VLM**
- Qwen3-VL-8B-Instruct：https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct ｜ https://github.com/QwenLM/Qwen3-VL
- VideoChat-Flash：https://github.com/OpenGVLab/VideoChat-Flash
- LongVU：https://github.com/Vision-CAIR/LongVU
- TimeChat：https://github.com/RenShuhuai-Andy/TimeChat
- VideoLLaMA3：https://github.com/DAMO-NLP-SG/VideoLLaMA3
- Video-R1：https://github.com/tulerfeng/Video-R1
- TinyLLaVA-Video-R1：https://github.com/ZhangXJ199/TinyLLaVA-Video-R1
- Video-Thinker：https://github.com/DeepExperience/Video-Thinker ｜ https://huggingface.co/ShijianW01/Video-Thinker-7B ｜ https://arxiv.org/abs/2510.23473
- InternVideo：https://github.com/OpenGVLab/InternVideo

**游戏域模型与基准**
- VideoGameBunny-V1：https://huggingface.co/VideoGameBunny/VideoGameBunny-V1 ｜ https://arxiv.org/abs/2407.15295 ｜ https://videogamebunny.github.io/ ｜ 数据集 https://huggingface.co/datasets/asgaardlab/VideoGameBunny-Dataset
- VideoGameQA-Bench：https://huggingface.co/datasets/taesiri/VideoGameQA-Bench ｜ https://arxiv.org/abs/2505.15952 ｜ https://asgaardlab.github.io/videogameqa-bench/
- Proact-VL：https://github.com/microsoft/AnthropomorphicIntelligence/tree/main/Proact-VL ｜ https://openreview.net/forum?id=k9PKgV0L4C ｜ 权重 https://huggingface.co/oaaoaa/proactvl_base_qwen3vl ｜ 基准 https://huggingface.co/datasets/oaaoaa/LiveGamingBenchmark ｜ SFT 数据 https://huggingface.co/datasets/oaaoaa/game_commentary_sft
- GameplayQA：https://github.com/HATS-ICT/GameplayQA ｜ https://hats-ict.github.io/gameplayqa
- GliDe：https://github.com/SandyyyZheng/GliDe
- RefTrace：https://github.com/sreevadde/reftrace
- Video-MME：https://github.com/MME-Benchmarks/Video-MME
- LongVideoBench：https://github.com/longvideobench/LongVideoBench

**数据集**
- Game-MUG：https://arxiv.org/abs/2404.19175
- MMOU：https://huggingface.co/datasets/lililiy/MMOU ｜ https://huggingface.co/datasets/sonalkum/MMOU-Videos ｜ https://arxiv.org/abs/2603.14145
- QVHighlights：https://arxiv.org/abs/2107.09609 ｜ https://github.com/jayleicn/moment_dense_trainer ｜ https://opendatalab.com/OpenDataLab/QVHighlights
- MMAI（未验证）：https://huggingface.co/datasets/VideoGameBench/MMAI

**教练 / 复盘 / 反面参照**
- forging：https://github.com/lucaslencinas/forging
- League-of-Legends-AI-Coach：https://github.com/sean-qin-usa/League-of-Legends-AI-Coach
- Honor-of-Kings_RolePlay：https://github.com/YongXie66/Honor-of-Kings_RolePlay
- MOBA-AI-Gamer：https://github.com/adrian27513/MOBA-AI-Gamer
- esports-coaching-tool：https://github.com/ESPort-Coach/esports-coaching-tool
- gameplay-vision-llm：https://github.com/chasemetoyer/gameplay-vision-llm

**不能商用 / 许可原文**
- NitroGen：https://huggingface.co/nvidia/NitroGen ｜ https://huggingface.co/datasets/nvidia/NitroGen ｜ https://nitrogen.minedojo.org/ ｜ https://arxiv.org/abs/2601.02427 ｜ https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf
- Ubisoft La Forge：https://github.com/ubisoft/ubisoft-laforge-chord ｜ https://huggingface.co/Ubisoft/ubisoft-laforge-chord/blob/main/LICENSE ｜ https://huggingface.co/spaces/ksangk/chord-demo/blob/main/LICENSE

**本项目内部资料**
- `docs/screen-action-recognition-survey.md`（CV 路线调研，同日）
- `docs/recognizer-design.md`（识别层设计）
- `app/collection/video.py`（识别结果契约）、`app/core/dsl.py`（规则 DSL）、`app/core/matcher.py`（规则对账）、`app/adapters/honor_of_kings.py`（动作词表）

---

## 7. 没找到 / 未验证（明确留白，不凑数）

1. **没有找到**「王者荣耀对局录屏 → 玩家操作」的现成开源模型或数据集。与上一轮 CV 调研的结论一致：王者域公开资产要么禁止商用、要么需要联系作者、要么只覆盖静态截图。
2. **没有找到**可直接商用的、带时间戳标注的**MOBA 对局录屏**数据集（Game-MUG 最接近，但它基于直播 + 事件日志，不是录屏，且许可与下载渠道未验证）。
3. **没有找到**在 Windows 本地可直接跑「视频输入 + 秒级定位」的、开箱即用的**游戏域**模型权重。游戏域权重（VideoGameBunny）都是图像域；视频域权重（Proact-VL / Video-Thinker）都需要训练。
4. **未验证**：MMAI、MMOU、Game-MUG、VideoGameQA-Bench、LiveGamingBenchmark 的许可条款与数据可得性（HF 本机不可达）。
5. **未实测**：任何一局的 VLM 推理耗时与显存占用。本文档不提供吞吐数字，避免给出没有依据的估算。
6. **未验证**：各仓库 issue 区的第三方复现记录（本轮未逐条读）。
7. **取证口径提醒**：本机 GitHub API 是过滤快照，部分真实仓库返回 404（见文首说明）；凡本文给出的 star 数，均来自本轮 API 返回或本项目上一轮调研的直读记录，两者已在表中标注。
