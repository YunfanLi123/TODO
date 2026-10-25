# 从录屏帧集合识别玩家操作：非大模型方法调研

调研日期：2026-10-24。检索来源：GitHub Search API（未认证）＋ 直接读取仓库源码。

问题（用户原话）：**「如果把对局视频切成很多帧的话，除了接入大模型，
还有什么办法可以识别出这些帧作为一个集合的玩家操作是什么」**

---

## 0. 先说结论

1. **没有现成的「王者录屏 → 玩家操作」开源模型或数据集。** 搜到的全是两类：
   *「读**状态**来打王者」的 RL 环境，和「在王者截图上打**靶**」的 YOLO。
   后者要么素材要另行索取，要么**禁止商业用途**。
2. **最贴合的现成参照是 `myBoris/wzry_ai`（327★）**，我把它的读屏源码读完了：
   它**一个 LLM 都不用**，读屏全靠「固定相对坐标裁剪 + HSV 颜色掩膜 +
   ONNX 小模型 + OCR」，跟我们正在收敛的路子几乎一样。
3. **三条可落地的路线**，成本递增：
   * 路线 1 **轨迹聚合**：零标注、零模型、纯统计，直接复用已有的 `hero_pos` + 地图区域表；
   * 路线 2 **ONNX 小分类器**：逐帧判「在干什么」，窗内投票 + 时序平滑（wzry_ai 的范式）；
   * 路线 3 **YOLO 检测 + 事件时间线**：需要框级标注，成本最高。
4. **一条必须说的负面结论**：我实测了三段录屏（用例1、用例2、`测试用例1-赵云.mp4`），
   **全部没有音轨**（mp4 `trak` 只有 `vide` handler，没有 `soun`）。
   ⇒ 「击杀播报 / 技能语音」这条本来最便宜的事件信号，**对我们的录屏不可用**。

---

## 1. 腾讯官方 hok_env：**不读画面**，读的是服务端状态

| | |
| --- | --- |
| 仓库 | `tencent-ailab/hok_env`（916★，Python） |
| 论文 | *Honor of Kings Arena: an Environment for Generalization in Competitive RL*，NeurIPS 2022 Datasets & Benchmarks，[arXiv:2209.08483](https://arxiv.org/abs/2209.08483) |
| 文档 | <https://doc.aiarena.tencent.com/paper/hok3v3/latest/hok3v3_env/honor-of-kings/> |

README 原文：「Hok_env SDK is used to interact with the **gamecore** of Honor of Kings」。

⇒ 它是**服务端 SDK**（读游戏内核），不是视觉识别。**对录屏完全无用**。
但它的价值在别处：**它定义了「一局王者里有哪些状态量」** ——
英雄坐标、血量、buff、击杀数、装备、等级 —— 可以直接当我们的
「要读哪些事实」的清单参照，避免自己拍脑袋列。

同类的多智能体实现（`JiapengLuan/MARL_HOK`、`AngelaLi03/MARL-Research-Hok`、
`Unakar/AI_Game_KingGlory`）也都跑在**简化环境/开悟平台**上，不是真实画面。

---

## 2. 最贴近的参照：wzry_ai 怎么读王者画面（源码级）

仓库：`myBoris/wzry_ai`（327★）。我把 `getReword.py` / `onnxRunner.py` / `wzry_env.py`
读完了。它每一帧做的事就是这四件：

| 手段 | 实现 | 对应我们项目 |
| --- | --- | --- |
| **相对坐标裁剪 + HSV 颜色掩膜 + 最右像素 → 数值** | `calculate_attack_reword`：裁 `(0.57, 0.019)` 起、宽 `0.116`×高 `0.024` 的条，inRange 出红色（HSV±10），取**最右非零列**算面积，映射成 0~11 | 与我们的 `read_bar_ratio`（血条）同法 |
| **ONNX 小模型做目标检测/分类** | `OnnxRunner('models/death.onnx', classes=['death'])`，onnxruntime，`CUDAExecutionProvider`→`CPUExecutionProvider` 回退 | **这正是「动作事实」最该抄的范式** |
| **OCR** | `ppocronnx.TextSystem().detect_and_ocr(image)`，判「胜利/失败」 | 顶栏击杀数那条卡住的路 |
| **多路并行** | `ThreadPoolExecutor` 同时跑 OCR / 分类 / 颜色判据 | 采集层提速 |

它给我们的三条直接启发：

1. **「死亡」这种状态，用一个二分类小模型就够**，不必上检测器。
   我们要把「刷野 / 开龙 / 带线 / 转线 / 回城」做成**多分类小模型**，
   类别数从 1 涨到 6，模型仍然只有几 MB，ONNX，能直接塞进 exe。
2. **它的动作空间设计**（`wzry_env.py`）值得抄结构：
   `move(0/1) × angle(0-359) × info_action(0-8) × attack_action(0-10)
   × action_type(0-2) × arg1/2/3` —— 把「玩家操作」**离散化**成可枚举的组合。
   我们的动作事实体系（26 个动作类别，`tools/hok_hero_actions.py`）也是同一思路。
3. **它的 `showposition.py`** 是个「坐标查看器」，点一下屏幕就打印
   `(x,y)/(w,h) = (x%,y%)` —— 说明这套 UI 坐标**就是靠人点出来标定的**，
   和我们刚做的「键位档位」是同一类工作。

---

## 3. 路线 1：轨迹聚合（**零标注、零模型，最该先做**）

这是「帧集合」这个提法最直接的答案：**单帧语义 × 时序聚合 = 窗级操作**。

我们已经有「英雄在小地图上的位置」（`hero_pos`，实测能读出 (0.031, 0.935)），
也已经有了**用户提供的三张真实地图**做区域表。把一窗内的位置序列算成运动特征，
再按几何形态归类：

| 操作 | 轨迹特征（可算，无标注） |
| --- | --- |
| **刷野** | 停留点集中在野怪点位、位移幅度小、方向反复、总路程长 |
| **开龙** | 在龙坑区域**长时间停留** + 主宰/暴君图标状态变化（图标认的是**颜色**，绝对可靠） |
| **带线** | 沿**单一方向**持续平移（方向方差小、总位移大）、常在边路 |
| **转线** | 跨区大位移（路径跨越小地图对角、方向突变）、中途换边路 |
| **回城** | 出现在泉水区域 + 画面活动度骤降 |

方法论上这就是**无监督运动聚类 / 轨迹行为分析**：
把每个窗的轨迹抽象成特征向量（总位移、路径长度、方向直方图熵、
停留点分布熵、覆盖区域集合），再聚类或直接用规则阈值判定。

**为什么先做它**：① 复用已有 `minimap.py` 与地图区域表；
② 不需要任何标注数据；③ 结论**可解释**（能说清「因为这一窗在龙坑停了 18 秒」）；
④ 符合项目「读错比读不到危险」的原则 —— 阈值不满足就记缺失。

---

## 4. 路线 2：ONNX 小分类器逐帧 + 窗内聚合

**训练一个「这一帧在干什么」的多分类 CNN**（刷野 / 打架 / 开龙 / 带线 / 转线 / 回城 / 空闲），
逐帧推理后**窗内投票 + 时序平滑**。

| 环节 | 可用的开源件 | 星 |
| --- | --- | --- |
| 训练/推理框架 | `open-mmlab/mmaction2`（视频理解工具箱） | 5182★ |
| 视频模型库 | `facebookresearch/pytorchvideo`（SlowFast/TSM/X3D/R(2+1)D） | 3571★ |
| 自监督预训练（**无标注起步**） | `MCG-NJU/VideoMAE`（NeurIPS'22 Spotlight）、`OpenGVLab/VideoMAEv2`（CVPR'23） | 1807★ / 829★ |
| 时序分割 / 平滑 | `sj-li/MS-TCN2`（TPAMI'20，**支持一段内多个动作并存**）、`yabufarha/ms-tcn` | 194★ / 263★ |
| 时序动作定位 | `happyharrycn/actionformer_release`（ECCV'22） | 578★ |

**MS-TCN2 尤其值得看**：它的 "action set" 变体就是为
「一段视频里同时发生了多个动作」设计的 —— 而王者一窗里
「刷野 + 回城 + 转线」本来就是并存的，这与我们 `aggregate_actions`
的去重合并口径完全对得上。

**弱监督路线**（只有窗口级标签、没有逐帧标注时）：`Pilhyeon/Awesome-Weakly-Supervised-Temporal-Action-Localization`（157★）是论文清单；
`jacobswan1/...Sparse-Temporal-Pooling-Network` 是 STPN 复现。
我们的数据形态天然就是「一窗 = 一个 bag」，
**MIL（多示例学习）**：窗是 bag、帧是 instance、窗级标签是 bag 标签 —— 正好对上。

---

## 5. 路线 3：YOLO 检测 → 事件时间线

| 仓库 | 内容 | 星 | 注意 |
| --- | --- | --- | --- |
| `jstzwj/HonorOfKingsObjectDetection` | **王者荣耀目标检测数据集**（810 个文件，yolo_data + yolo_labels） | 1★ | **README 明写「仅供学习研究使用，禁止商业用途」**，且标签「尚未标注完整」 |
| `ironmanhcx/-YOLO-` | YOLOv5 + 王者目标检测，**带预训练权重** `runs/train/exp/weights/best.pt`（COCO 迁移学习） | 8★ | **数据集需联系作者** |
| `moluyangyu/HOK` | YOLOv5s 训练**王者荣耀英雄头像**检测 | 5★ | 兰州理工实训作业，有 train/val 划分 |
| `Microsoft/OmniParser` | 通用 GUI 截图解析（图标检测 + 结构化元素） | 25.5k★ | 图标检测部分是 YOLO（非大模型），但**语义描述部分依赖 VLM** |
| `ibaiGorordo/ONNX-SAM2-Segment-Anything` | SAM2 的 ONNX 导出（可分割野区/河道/龙坑） | 288★ | 分割非识别 |
| `facebookresearch/dinov2` | 自监督特征（13.5k★）；`lavaman131/dinov2.cpp` 是 **C++/ggml 推理** | — | 「这段像不像刷野」的**最近邻**原型法 |

⚠️ **授权红线**：唯一公开的王者目标检测数据集**禁止商业用途**。
本项目是商业产品 ⇒ **不能直接用那份数据**，只能拿它当**标注规范的参考**，
素材自己标。

---

## 6. OCR 这条独立支线（顶栏击杀数）

我们卡住的「击杀数模板验收 2/17」，本质是**11 像素高的数字**。
模板匹配被抗锯齿和字距打败；**OCR 是同一问题的另一条独立解法**：

| 仓库 | 说明 | 星 |
| --- | --- | --- |
| `RapidAI/RapidOCR` | ONNX 版 OCR 工具包，后端可换 ONNXRuntime/OpenVINO/MNN/PaddlePaddle/TensorRT/PyTorch | 8297★ |
| `SvenVincent/cnocr` | 中文/英文/数字 OCR 工具包，自带 20+ 模型 | 63★ |
| `RapidAI/RapidOcrOnnx` | rapidocr 的 C++ 版 | 376★ |

wzry_ai 用的 `ppocronnx` 就是 PaddleOCR 的 ONNX 导出路线，
且它证明 **OCR 在王者画面上可用**（中文「胜利/失败」识别）。
**OCR 也能打包进 exe**（ONNX，无 torch 依赖）—— 这对我们是关键。

但要诚实：11 像素高的数字对 OCR 也不友好，**需要实测**才能下结论。
顶栏击杀数还有两个有利条件：**蓝/红两色天然分开左右**，且**数字位数少**（0~30）。

---

## 7. 一条被否掉的免费路线：音频

`lulaide/WZRY-audio`（18★）提供了王者荣耀**语音数据集** ——
理论上「第一滴血」「双杀」「推塔」这些**游戏自己播的语音**，
就是免费的事件标注信号，对「击杀 / 开龙 / 推塔」类事件极有价值。

**但我用 mp4 box 解析实测了三段录屏**（`393ed2eb8a8e.mp4` / `case2-aoyin.mp4` /
`测试用例1-赵云.mp4`）：`trak` 里**只有 `vide` handler，没有 `soun`**。
⇒ **这条路线对我们的录屏不可用**。除非以后要求用户录屏时带上声音。

---

## 8. 给本项目的落地建议（按性价比排序）

| 优先 | 做什么 | 依赖 | 产出 |
| --- | --- | --- | --- |
| **P0** | **轨迹聚合**（路线 1）：一窗内 `hero_pos` 序列 → 运动特征 → 刷野/开龙/带线/转线/回城 | 已有 `minimap.py` + 用户给的三张地图 + 区域表 | 动作事实层，规则能判了 |
| **P0** | **区域表补齐**：把用户给的三张图标成区域多边形（`tools/calibrate_map.py` 已就绪） | 用户给图 | `hero_zone` 有意义 |
| P1 | **OCR 试读顶栏击杀数**（RapidOCR ONNX） | pip 装 rapidocr-onnxruntime | 解决 2/17 那个卡点 |
| P1 | **ONNX 多分类小模型**（路线 2）：逐帧判「在干什么」，窗内投票 | 需要标注几百到几千帧 | 覆盖轨迹法读不出的操作 |
| P2 | **YOLO 事件检测**（路线 3）：英雄/野怪/龙 | 自己标素材（授权受限） | 事件级时间线 |
| P2 | **VideoMAE/DINOv2 自监督特征 + 原型法** | GPU 环境 | 无标注冷启动 |

**共同约束**（沿用项目既有原则）：
- 读不出就记**缺失**，不给猜的值；
- 每个新判据都要有**验收标准**（多少个已知时刻读对多少个），
  未验收的模板/模型**不启用**；
- 模型一律 **ONNX**，以便塞进 exe（wzry_ai 已经证明了这条路可行）。

---

## 附：检索到的仓库速查

| 仓库 | 星 | 用途 |
| --- | --- | --- |
| `tencent-ailab/hok_env` | 916 | 腾讯官方王者 RL 环境（读服务端状态，非视觉） |
| `myBoris/wzry_ai` | 327 | **最贴近的读屏参照**（ONNX + 颜色掩膜 + OCR） |
| `microsoft/OmniParser` | 25.5k | GUI 截图结构化解析 |
| `facebookresearch/dinov2` | 13.5k | 自监督特征 |
| `RapidAI/RapidOCR` | 8297 | ONNX OCR |
| `open-mmlab/mmaction2` | 5182 | 视频理解工具箱 |
| `facebookresearch/pytorchvideo` | 3571 | 视频模型库 |
| `MCG-NJU/VideoMAE` | 1807 | 自监督视频预训练 |
| `Pilhyeon/Awesome-Weakly-Supervised-TAL` | 157 | 弱监督时序定位论文清单 |
| `happyharrycn/actionformer_release` | 578 | 时序动作定位（ECCV'22） |
| `sj-li/MS-TCN2` | 194 | 时序分割，支持动作集合 |
| `yabufarha/ms-tcn` | 263 | MS-TCN 实现 |
| `ibaiGorordo/ONNX-SAM2` | 288 | SAM2 ONNX 导出 |
| `ironmanhcx/-YOLO-` | 8 | 王者 YOLOv5 + 权重（数据需索取） |
| `moluyangyu/HOK` | 5 | 王者英雄头像 YOLOv5 |
| `jstzwj/HonorOfKingsObjectDetection` | 1 | 王者检测数据集（**禁止商用**） |
| `lulaide/WZRY-audio` | 18 | 王者语音数据集（**我们的录屏无音轨**） |
