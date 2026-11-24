"""**认英雄**：切一段录像给模型，让它自己看。

为什么不是「抽帧 → 裁技能按钮 → 像素匹配」
--------------------------------------------
2026-11-03 实测（``hok-video-372808da845d``）：那条旧路每一步都在读噪声 ——
* 局内技能按钮的裁图里**叠着冷却倒计时数字**（合成后能看到「06」「01」「1」），
  美术本身被压掉一大半；
* 像素前两名只差 **0.0119**（弈星 0.4855 / 敖隐 0.4735）—— 纯噪声；
* 换 5 种合成策略（饱和度前 5 / 全中位数 / 亮度归一 / 中心残差 / 全图残差）
  **没有一种**能让真值排到第一。

现在这条路
------------
**把视频本身给模型** —— 像 Claude Code 重新看一眼素材那样：

1. 切一段**画面上印着名字**的界面（选人/BP 段）；
2. 把「名字所在的那块」**拉近**（裁剪 + 放大）——
   **视频会被降采样**，整帧里的小字模型看不清；拉近之后它就看得清了；
3. 问它「玩家自己那一行印着的英雄名是什么」，**读不到就说不知道**。

2026-11-03 实测（同一局，10 次模型调用）
--------------------------------------
* **凭美术认英雄：10 次没有一次对**。局内技能图标 → 答「裴擒虎/不知火舞/杨玉环」；
  加载页立绘 → 10 张卡认对 1 张；跨片段对照 → 编出「百里玄策」还编了细节。
* **读画面上的字：读得准**。把名单区拉近后，5 个昵称全读对了
  （含「澜和守约别禁」）。
* 所以：**名字只能靠读字**，靠长相认是认不准的。

三条纪律
--------
1. **看不出来就说看不出来**（``hero`` 写空）——
   「认不准 > 认错」：错一个英雄，英雄层会去套别人的打法指责玩家。
2. **先描述、后命名** —— 让模型先把看到的画面写出来，
   名字才有依据可查（只给名字时，它会把「看着像什么」当成「是什么」）。
3. **报出来源** —— ``hero_source="text"`` 表示这个名字是**从画面上的字
   读到的**；没读到就留空。裁决层只认这一种来源。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import game_knowledge, videopack

#: **名单区**在画面里的位置（相对比例，x0, y0, x1, y1）。
#:
#: 为什么是左边那一竖条：选人界面上，本队 5 个玩家从上到下排在左边，
#: **金色高亮的那一行就是录屏的人自己**，每一行印着头像和昵称。
#: 用相对比例是为了不挑分辨率（录屏有 1280x574，也有别的）。
ROSTER_BOX: Tuple[float, float, float, float] = (0.02, 0.05, 0.30, 0.82)

#: 拉近几倍。
#:
#: 为什么必须拉近：**视频会被降采样**，整帧里的小字（昵称、行内英雄名）
#: 模型看不清；裁出来放大 3 倍之后，同一段视频里的昵称它全读对了。
ZOOM = 3

#: 切多长一段。
#:
#: 选人界面是**静止**的，节奏慢，一秒一帧足够。
DEFAULT_WINDOW_SEC = 150.0

#: 剪掉多少秒再切 —— 录像开头可能录了一段别的（应用启动、匹配）。
DEFAULT_LEAD_IN_SEC = 5.0

#: **常识在前、纪律在后** —— 最后的输出要求离输出最近。
#:
#: 常识来自 :mod:`app.recognition2.game_knowledge`：
#: 冷却数字不是美术、技能类型是白字、「强化」是形态键、
#: 选人界面上金色那行是自己、**预选不等于最后选的那个**……
#: 不讲这些，模型会把界面元素当成画面内容（2026-11-03 实测：
#: 它把冷却数字「06」当成了「红金色环形徽记」这个美术特征）。
SYSTEM_PROMPT = game_knowledge.knowledge_for("hok") + (
    "\n【这一问要你做什么】\n"
    "你是《王者荣耀》录像的分析助手。你只根据**画面里真实看到的东西**回答。\n"
    "看不清就明说看不清 —— 不许猜、不许编、也不许把一个「有点像的英雄」"
    "当成答案。\n"
)

PROMPT = (
    "这是一段《王者荣耀》**选人(BP)界面**的录像片段（画面被拉近了，"
    "能看到本队 5 个玩家那一列名单）。\n\n"
    "请只回答一个 JSON：\n"
    '{"quote": "你在这段画面里**照抄**读到的文字（读不到就写 null）", '
    '"hero": "玩家自己操控的英雄名", "look": "你在这段画面里看到什么", '
    '"why": "你的依据"}\n\n'
    "要求：\n"
    "1. **金色高亮的那一行是录屏的人自己**，别的行是队友 —— "
    "只认金色那一行的英雄。\n"
    "2. **只读画面上印着的字**：画面上印着英雄的名字就把那个名字读出来；"
    "**没有印着就写 null**，并说明「那一行没印英雄名」。\n"
    "   ⚠️ 2026-11-13 逐帧核对过：这一屏的名单**只印昵称、分路与「预选」状态**，"
    "**通常不印英雄名**；右边候选网格里的名字是「可以选谁」，"
    "**不代表谁选了它**。读不到就照实写 null，这不是失败。\n"
    "3. ⚠️ **「预选」不等于最终选的那个**：选人界面上写着「预选」的名字，"
    "是**还没锁定**的那一个 —— 倒计时最后一秒会被抢掉。"
    "**你只能读到「预选」时，把 hero 写成 null**，"
    "在 why 里说明「只能读到预选的名字」。\n"
    "4. **绝对不要根据头像去猜英雄名字** —— 2026-11-03 实测，"
    "凭长相认 10 次没有一次对；**读到字才是准的**。\n"
    "5. quote 里写你**真读到的字**（照抄画面上的字），"
    "hero 必须能从 quote 里对上。\n"
    "6. 先写 look，再写 hero。"
)


@dataclass
class HeroRead:
    """认英雄的结果。

    ``hero`` 为空 = **没认出来**（不要拿它去套英雄层规则）。
    """

    hero: str = ""
    #: 名字的来源。``"text"`` = 画面上印着的字；空 = 没认出来。
    hero_source: str = ""
    why: str = ""
    look: str = ""
    #: 照抄下来的画面文字（用来核对 hero 是不是真读到的）。
    quote: str = ""
    #: 看的哪一段（原视频秒）。
    window: Tuple[float, float] = (0.0, 0.0)
    notes: List[str] = field(default_factory=list)
    calls: int = 0
    tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hero": self.hero,
            "hero_source": self.hero_source,
            "why": self.why,
            "look": self.look,
            "quote": self.quote,
            "window": [round(float(self.window[0]), 1),
                       round(float(self.window[1]), 1)],
            "notes": list(self.notes),
        }


def _pick_window(start_t: Optional[float], duration_sec: float,
                 window_sec: float, lead_in_sec: float) -> Tuple[float, float]:
    """挑一段**选人界面**来看。

    选人界面在**对局开始之前**：录像的前半段是 BP，
    紧跟着才是加载与战场。所以：

    1. 知道对局起点 → 取它**之前**那一段（BP 就在那儿）；
    2. 不知道 → 取录像开头那一段；
    3. 砍掉开头几秒（应用启动/匹配画面）。

    BP 段不会太长（两三分钟），但为保险起见有个上限 ——
    宁可切最后三分钟（锁定都发生在末尾），也别切十分钟。
    """
    window_sec = max(5.0, float(window_sec))
    duration = float(duration_sec or 0.0)
    lead = max(0.0, float(lead_in_sec))

    if start_t is not None and float(start_t) > 0:
        hi = float(start_t)
    elif duration > 0:
        hi = min(duration, window_sec + lead)
    else:
        hi = window_sec + lead
    hi = max(5.0, hi)
    lo = max(0.0, hi - window_sec)
    if lo > 0:
        lo = max(0.0, lo - lead)
    return lo, hi


def _zoom_roster(frame: Any) -> Any:
    """把**名单区**裁出来放大（画面被降采样过，小字要拉近才看得清）。"""
    import cv2

    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return frame
    x0, y0, x1, y1 = ROSTER_BOX
    crop = frame[int(y0 * h):max(int(y0 * h) + 1, int(y1 * h)),
                 int(x0 * w):max(int(x0 * w) + 1, int(x1 * w))]
    if crop.size == 0:
        return frame
    big = cv2.resize(crop, (crop.shape[1] * ZOOM, crop.shape[0] * ZOOM),
                     interpolation=cv2.INTER_CUBIC)
    # 视频编码器要偶数边长。
    return big[:big.shape[0] - (big.shape[0] % 2), :big.shape[1] - (big.shape[1] % 2)]


#: 抽帧那条路一次给几张。
#:
#: 选人界面是**静止**的，多给几张是为了「有些帧刚好被气泡/动画挡住」时
#: 还能在别的一张上读到名字。6 张足够，再多只是烧钱。
FRAME_COUNT = 6

#: 抽帧那条路的提问：和 PROMPT 只差开头一句（告诉它这是几张静图）。
FRAMES_PROMPT = (
    "下面是同一段《王者荣耀》**选人(BP)界面**的 **%d 张画面**"
    "（按时间先后，画面被拉近了，能看到本队 5 个玩家那一列名单）。\n\n"
    % FRAME_COUNT
) + PROMPT.split("\n\n", 1)[1]


def _takes_video(creds: Any, data_dir: str,
                 slot: str = "vlm") -> Optional[bool]:
    """这个槽位的模型收不收视频（``None`` = 认不出，按老路试）。"""
    try:
        from app.collection import model_presets, settings

        with settings.use_data_dir(data_dir or None):
            cfg = settings.effective_slot(slot)
        return model_presets.video_support(
            str(cfg.get("provider") or ""),
            str(cfg.get("endpoint") or getattr(creds, "endpoint", "") or ""))
    except Exception:                                  # noqa: BLE001
        return None


def _roster_frames(video_path: str, lo: float, hi: float,
                   prep: Callable[[Any], Any],
                   count: int = FRAME_COUNT) -> List[bytes]:
    """在这一段里均匀抽几张（每张都按 ``prep`` 裁到名单区再放大）。"""
    from .frames_watch import read_frames

    span = max(1.0, float(hi) - float(lo))
    times = [float(lo) + span * (i + 0.5) / float(count) for i in range(count)]
    return [f.jpeg for f in read_frames(video_path, times, prep) if f.jpeg]


def read_hero(video_path: str, video_dir: str, *,
              start_t: Optional[float] = None,
              duration_sec: float = 0.0,
              window_sec: float = DEFAULT_WINDOW_SEC,
              lead_in_sec: float = DEFAULT_LEAD_IN_SEC,
              transform: Optional[Callable[[Any], Any]] = None,
              slot: str = "vlm", model: str = "", data_dir: str = "",
              clip_fps: float = videopack.DEFAULT_CLIP_FPS,
              model_fps: float = 1.0, detail: str = "high",
              max_tokens: int = 1200, timeout: float = 600.0) -> HeroRead:
    """切一段选人界面给模型，问它「玩家自己那一行印着的英雄名是什么」。

    返回 :class:`HeroRead`。**任何一步失败都只是留痕 + 空结果** ——
    这一层没答上来不等于「玩家没有这个英雄」，上层还有别的读数。
    """
    from app.collection import vlm_client

    out = HeroRead()
    try:
        creds = vlm_client.find_credentials(slot, data_dir or None)
        if not model:
            from app.collection import settings
            model = settings.model_name(slot)
    except Exception as exc:                           # noqa: BLE001
        out.notes.append("取不到「%s」槽位的模型配置：%s" % (slot, exc))
        return out
    if not creds or not model:
        out.notes.append("没有配置「%s」这个槽位的模型 —— 没认英雄。" % slot)
        return out

    lo, hi = _pick_window(start_t, duration_sec, window_sec, lead_in_sec)
    out.window = (lo, hi)

    def _prep(frame: Any) -> Any:
        out_frame = frame
        if transform is not None:
            try:
                out_frame = transform(out_frame)
            except Exception:                          # noqa: BLE001
                out_frame = frame
        return _zoom_roster(out_frame)

    # **收不了视频的模型改发抽帧**（2026-11-13）：这一段本来就是**静止**的
    # 选人界面，抽几张和一段视频看到的是同一件事 —— 而 DeepSeek 这类
    # OpenAI 兼容接口压根不认 video 块（422 unknown variant）。
    if _takes_video(creds, data_dir) is False:
        frames = _roster_frames(video_path, lo, hi, _prep, count=FRAME_COUNT)
        if not frames:
            out.notes.append("这一槽位的模型不收视频，抽帧也没抽到画面"
                             "（%.0f~%.0f 秒）—— 没认英雄。" % (lo, hi))
            return out
        out.notes.append("这一槽位的模型不收视频 —— 改用 %d 张抽帧认英雄。"
                         % len(frames))
        blocks = [vlm_client.text_block(FRAMES_PROMPT)]
        for data in frames:
            blocks.append(vlm_client.image_block(data))
    else:
        if not video_dir:
            out.notes.append("没给打包目录 —— 没认英雄。")
            return out
        os.makedirs(video_dir, exist_ok=True)
        clip_path = os.path.join(video_dir, "hero_clip.mp4")

        try:
            info = videopack.cut_clip(video_path, lo, hi, clip_path,
                                      fps=clip_fps, transform=_prep)
        except Exception as exc:                       # noqa: BLE001
            out.notes.append("切不出选人界面那一段：%s" % exc)
            return out
        if info.size <= 0:
            out.notes.append("切不出选人界面那一段（%.0f~%.0f 秒）—— 没认英雄。"
                             % (lo, hi))
            return out

        try:
            with open(clip_path, "rb") as fh:
                data = fh.read()
        except OSError as exc:
            out.notes.append("读切好的片段失败：%s" % exc)
            return out

        blocks = [vlm_client.text_block(PROMPT),
                  vlm_client.video_block(data, fps=model_fps, detail=detail)]
    try:
        result = vlm_client.call_model(
            model, blocks, creds=creds, system=SYSTEM_PROMPT,
            max_tokens=max_tokens, timeout=timeout)
    except vlm_client.VlmError as exc:
        out.notes.append("没认成英雄：%s" % vlm_client.explain_error(exc))
        return out
    out.calls = 1
    try:
        usage = result.usage or {}
        out.tokens = int(usage.get("output_tokens") or 0)
    except Exception:                                  # noqa: BLE001
        out.tokens = 0

    try:
        raw = result.json()
    except vlm_client.VlmError as exc:
        out.notes.append("模型回答不是合法 JSON（%s）—— 没认成英雄。" % exc)
        return out
    if not isinstance(raw, dict):
        out.notes.append("模型回答不是 JSON 对象 —— 没认成英雄。")
        return out

    out.why = str(raw.get("why") or "").strip()
    out.look = str(raw.get("look") or "").strip()
    out.quote = str(raw.get("quote") or "").strip()
    hero = str(raw.get("hero") or "").strip()

    # **只采信从画面上读到的名字** —— 模型要是编的，
    # quote 里就找不到这个名字（quote 是它照抄的）。
    if hero and out.quote and hero not in out.quote:
        out.notes.append("模型说「%s」，但它照抄的画面文字里没有这个名字 —— "
                         "当成没认出来。" % hero)
        hero = ""
    if hero:
        out.hero = hero
        out.hero_source = "text"
    else:
        out.notes.append("模型在选人界面那段没读出英雄名（%s）"
                         % (out.why or "没说原因"))
    return out
