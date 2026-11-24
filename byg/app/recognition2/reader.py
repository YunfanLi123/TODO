"""**逐段读行为** —— 新识别子系统的第二层，也是 harness 的核心。

这一层只做一件事：**让模型看一段画面，说清这段时间里玩家在干什么。**

harness 的五个部件在这里都有落点
--------------------------------
=================  ==========================================================
部件               落点
=================  ==========================================================
工具注册中心       复用 :mod:`app.harness.tools`（``zoom_frame`` / 打转检测）
Agentic Loop       模型回答里带 ``need_more`` 就去取料，取回来再问一轮
预算与止损        每段轮数上限、整局调用/token 上限、重复请求指纹去重
上下文装配        抽帧 → 编号 → set-of-mark（模型能说「第 3 张图」）
降级留痕          每段记 ``why_not_read``（看不清 / 没调用 / 超预算 / 额度尽）
=================  ==========================================================

两条纪律
--------
1. **看不清就写 unclear**，不许编。``unclear`` 的段照样进时间线，
   但 ``confidence=0``，判定时算「无法判定」，**永远不算「没做」**。
2. **只描述，不评判**。这一层**不查模式库**（不接 ``search_modes``）——
   给了攻略清单，模型会去凑答案而不是看画面。
   「该不该这么做」是判定层的事，且必须晚于这一层。

读时刻：只有一个来源，模型逐帧读顶栏
------------------------------------
局内时刻**只有一个来源**：多模态模型**逐帧**读到的顶栏时刻。

- 每张图分别报时刻，编号跟图号对上（不再「每段报一个」）；
- 锚点帧额外附一张**定点放大**的计时器（35×12 → 280×96，约 5 个 token）；
- 所有读数走抗离群拟合，与拟合线差 >3 秒的判离群、不参与定轴；
- 都没读到 → ``game_time = -1``（**不猜**，也绝不用相邻段的值硬填）。

**OpenCV 模板匹配读钟整套撤掉了** —— 它在这段录屏上 5 次读错 4 次
（见 :mod:`app.recognition2.clockaxis` 的开头表）。
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .beat import (Beat, Timeline, UNCLEAR, PHASE_UNKNOWN,
                   PHASE_IN_GAME, PHASE_PRE_GAME, PHASE_POST_GAME)
from .sampler import Segment

#: 放大的局部区域（相对坐标 ``[x0, y0, x1, y1]``）。
#:
#: 这些坐标以前存放在 ``app/collection/hud_regions.json`` 里 ——
#: 那条链（局内技能按钮 → 裁图 → 像素比对）2026-11-04 整条退役，
#: 标定文件跟着删了，坐标**搬进下面这张表**（原值照搬，
#: 都是对 1280×574 的原录屏量的）。
#: 注意：**这些框只是裁给模型看的**，不再拿去做像素比对。
#: ``minimap`` 用 :file:`app/collection/map_geometry.json` 里更准的那份。
_ZOOM_BOXES: Dict[str, Tuple[float, float, float, float]] = {
    "clock": (0.6815, 0.0095, 0.709, 0.031),      # 顶栏计时器
    "skill_ult": (0.813, 0.521, 0.876, 0.652),    # 大招按钮
    "skill_all": (0.60, 0.78, 0.90, 0.99),        # 技能按钮群（含回城/恢复）
    "scoreboard": (0.60, 0.0, 0.80, 0.08),        # 顶栏比分
}

#: 局部区域的中文名（印在图上，模型才知道看的是哪块）。
_ZOOM_LABELS: Dict[str, str] = {
    "minimap": "小地图",
    "clock": "计时器",
    "skill_ult": "大招按钮",
    "skill_all": "技能按钮群",
    "scoreboard": "顶栏比分",
    "whole": "整帧",
}


# ---------------------------------------------------------------- 词表


def action_vocabulary() -> Dict[str, Dict[str, str]]:
    """动作词表：**从适配器取，不另编一套**。

    两套词表对不上，判定层就认不出来了。
    """
    try:
        from ..adapters.honor_of_kings import MOBA_ACTION_VALUES
        return {k: dict(v) for k, v in MOBA_ACTION_VALUES.items() if v}
    except Exception:                                  # noqa: BLE001
        return {}


def flat_vocabulary() -> Dict[str, str]:
    """扁平的 ``取值 -> 中文名``。"""
    try:
        from ..adapters.honor_of_kings import MOBA_ACTION_LABELS
        return dict(MOBA_ACTION_LABELS)
    except Exception:                                  # noqa: BLE001
        return {}


def position_vocabulary() -> Dict[str, str]:
    """位置词表：**从适配器取，不另编一套**。

    规则的前提里就有 ``position``（``when: position == "jungle"``）——
    判定层要拿它和规则的取值**逐字比对**，所以这里必须是**规范值**，
    不能是「红buff野区及下方野怪营地」这种自由描述。

    自由描述另有去处（``detail["where"]``），信息不丢。
    """
    try:
        from ..adapters.honor_of_kings import POSITION_LABELS
        return {k: v for k, v in POSITION_LABELS.items() if k != "unknown"}
    except Exception:                                  # noqa: BLE001
        return {}


# ---------------------------------------------------------------- 配置


@dataclass
class ReaderConfig:
    """读行为这一层的参数。"""

    #: 槽位名（配置里 ``model.vlm`` 那个）。
    slot: str = "vlm"
    #: 模型名。空 = 从配置里取。
    model: str = ""
    #: 数据目录。空 = 走默认（``MATCH_DATA_DIR``）。
    data_dir: Optional[str] = None
    #: 每段最多问几轮（第一轮 + 点单补料）。
    max_rounds_per_segment: int = 3
    #: 每轮最多处理几个点单请求。
    max_tool_calls_per_round: int = 2
    #: 整局最多调多少次模型。
    max_calls_per_game: int = 400
    #: 整局最多花多少 token（0 = 不限）。
    max_tokens_per_game: int = 0
    #: 单次请求的 token 上限。
    #:
    #: **为什么是 4096 而不是 1200**
    #: ----------------------
    #: 实测（2026-10-31，`赵云.mp4` 第 32 段）：模型把 **1200 个 output token
    #: 全花在思考上**（``usage.output_tokens_details.thinking_tokens = 1200``），
    #: ``stop_reason = max_tokens``，**一个字答案都没吐** —— 解析出来是空的，
    #: 报「模型没有返回内容」。它纠结的是「这英雄是镜还是澜」这种问题。
    #:
    #: 传了 ``thinking: {type: disabled}`` 也拦不住（MiniMax 偶尔不认），
    #: 所以预算必须留够 —— 思考 + 答案都装得下。
    max_tokens: int = 4096
    #: 单次请求超时（秒）。
    timeout: float = 300.0
    #: 抽帧目录。空 = 不落盘（那 ``Beat.frames`` 就没有可回看的文件）。
    frames_dir: str = ""
    #: JPEG 质量。
    jpeg_quality: int = 80
    #: 只读前 N 段（调试用）。0 = 全读。
    limit_segments: int = 0
    #: 续跑：读这份时间线里**已经读成功**的段，跳过它们。
    #: 一局 33 万 token，从头再跑一遍太贵 —— 失败的那几段单独重读即可。
    resume_path: str = ""
    #: 单次调用返回空/坏 JSON 时的重试次数。
    empty_retries: int = 2
    #: 要不要给模型附带一张「锚点帧顶栏计时器放大图」。
    #:
    #: 为什么要有
    #: ----------
    #: 顶栏计时器在 1280×574 的画面上只有 **35×12 像素** —— 五个字符挤在
    #: 那儿，整帧里几乎看不清。放大 3 倍后是 105×36，模型能稳稳读出来。
    #:
    #: **几乎不花钱**：按 Anthropic 的图片计费公式（宽×高/750），
    #: 这张小图约 5 个 token，而一张整帧约 980 个。
    #: 用它换来一个**锚定到具体某帧**的可靠读数，很划算。
    read_clock_zoom: bool = True
    #: 时刻读数与拟合线相差超过这个秒数，就算读错了（不采信）。
    clock_tolerance: float = 3.0
    #: 进度回调：``(已读段数, 总段数)``。
    #:
    #: 为什么要它
    #: ------------
    #: 读一局要好几分钟。后台任务不报进度的话，前端进度条会**一直停在
    #: 原地** —— 那比「慢」更让人以为程序死了。
    #: 而且进度必须是**真的**：按已读段数算，不是按时间瞎猜。
    on_progress: Optional[Callable[[int, int], None]] = None


@dataclass
class SegmentTrace:
    """一段的留痕。**失败也要留痕** —— 不留痕就没人知道哪段没读出来。"""

    index: int = 0
    start: float = 0.0
    end: float = 0.0
    rounds: int = 0
    calls: int = 0
    tokens: int = 0
    tool_calls: List[str] = field(default_factory=list)
    ok: bool = False
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "rounds": self.rounds,
            "calls": self.calls,
            "tokens": self.tokens,
            "tool_calls": list(self.tool_calls),
            "ok": self.ok,
            "error": self.error,
        }


# ---------------------------------------------------------------- 帧


class FrameCursor:
    """按时间取帧的游标。

    **必须按时间戳 seek，不能用 ``grab()`` 逐帧推进**
    ------------------------------------------------
    实测（2026-10-31，赵云.mp4）：``cap.grab()`` 推进 21601 次取到的帧，
    画面上的局内时间是 **03:40**（220 秒）；而帧号 21600 的真实 PTS 是
    360.985 秒、对应画面是 **03:33**（212 秒）—— **差 6.7 秒，是另一张画面**。
    这段录屏上 ``grab()`` 会跳过帧，「grab 了 N 次」≠「显示顺序的第 N 帧」。

    用它取帧的话，**证据帧会指错原视频时刻**（文件名叫 361.0 秒，
    实际是 367.7 秒那一张），人工回看时对不上。

    所以这里一律 ``set(POS_MSEC)`` seek，并用读回来的真实 PTS 当时间戳。
    """

    def __init__(self, path: str, transform=None):
        import cv2

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError("打不开视频：%s" % path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        #: 上一帧的**真实 PTS**（秒）。文件名要用它，不能用请求的时间。
        self.last_time = 0.0
        #: **画面归一**（转正 + 裁黑边），可选。
        #:
        #: 上传的录屏可能横竖屏颠倒、上下还有黑边 —— 不归一的话，
        #: 模型看到的是转过来的画面，读出来的东西全是错的。
        #: 直接读原片的测试视频不需要它（``None``）。
        self.transform = transform

    def get(self, t: float):
        """取第 ``t`` 秒那一帧。取不到返回 None。"""
        import cv2

        if t < 0:
            return None
        self.cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.last_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if self.transform is not None:
            try:
                frame = self.transform(frame)
            except Exception:                              # noqa: BLE001
                #: 归一失败就**照原样给** —— 半张转正的图也比没有强，
                #: 而且不能因为归一失败就丢掉这一帧。
                pass
        return frame

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:                              # noqa: BLE001
            pass


# ---------------------------------------------------------------- 工具上下文


class ZoomContext:
    """给 :mod:`app.harness.tools` 用的上下文。

    只实现 ``zoom_image`` —— **不实现 ``search_modes``**：
    读行为这一层不许查打法（给了清单，模型会去凑答案而不是看画面）。

    ``index`` 是**当前帧**在段内帧列表里的下标：模型点单时说的是
    「第 2 帧」，取料必须作用在那**一帧**上，不能一律取第一帧。
    """

    def __init__(self, frames: List[Any], labels: List[str], index: int = 0):
        self._frames = frames
        self._labels = labels
        self.index = max(0, min(index, len(frames) - 1)) if frames else 0

    def _current(self):
        if not self._frames:
            return None
        return self._frames[min(self.index, len(self._frames) - 1)]

    def zoom_image(self, zoom: str, label: str = "") -> Optional[bytes]:
        """把某一帧的某个局部裁出来、放大。

        ``zoom`` 认这些：``whole`` / ``minimap`` / ``clock`` /
        ``skill_ult`` / ``skill_all`` / ``scoreboard``。
        认不出来的**如实返回 None** —— 绝不裁一块错的地方给模型。
        """
        import cv2

        frame = self._current()
        if frame is None:
            return None
        crop = None
        if zoom in (None, "", "whole"):
            crop = frame
        elif zoom == "minimap":
            try:
                from ..collection.minimap import MapGeometry, minimap_crop
                crop = minimap_crop(frame, MapGeometry.load())
            except Exception:                          # noqa: BLE001
                crop = None
        else:
            box = _ZOOM_BOXES.get(zoom)
            if box is None:
                return None
            h, w = frame.shape[:2]
            x0, y0, x1, y1 = box
            px0, py0 = int(x0 * w), int(y0 * h)
            px1, py1 = int(x1 * w), int(y1 * h)
            if px1 - px0 < 4 or py1 - py0 < 4:
                return None
            crop = frame[py0:py1, px0:px1]
        if crop is None or crop.size == 0:
            return None
        # 放大：模型要看清小地图上的图标、按钮上的字。
        big = cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3),
                         interpolation=cv2.INTER_CUBIC)
        ok, buf = cv2.imencode(".jpg", big, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return None
        return buf.tobytes()


# ---------------------------------------------------------------- 提示词

SYSTEM_PROMPT = (
    "你是一个王者荣耀对局复盘助手。你会拿到**同一段时间里的几张连续画面**"
    "（按时间先后编号）。\n"
    "你的活只有一件事：**说清这段时间里玩家在干什么**。\n"
    "硬规矩：\n"
    "1. 只写你在画面里看到的。看不清就写 \"unclear\"，不要猜、不要美化、不要靠常识补全。\n"
    "2. **叙述（narrative）是正文，必须写**；动作码（primary_action / all_actions）"
    "只是**顺手归的一个类**，用来做后续的检索与统计。清单里没有合适的就写 "
    "\"unclear\" —— **不要为了凑一个码而把事情说小**。"
    "「在装备商店挑装备」「打开战绩面板看数据」「阵亡等复活」"
    "这些清单里没有，但你看得到，就要写进叙述里。\n"
    "3. narrative 用**一句话**写清「玩家在做什么、在哪、对谁」，"
    "一段里干了多件事就**按先后顺序**写（\"先在野区打怪，接着转去中路\"）。"
    "不要写空话（\"正在进行游戏\"、\"玩家在操作\"这种不算）。\n"
    "4. 画面不足时可以**点单要料**：在 need_more 里写 "
    "{\"frame\": 编号, \"zoom\": \"minimap|clock|skill_all|skill_ult|scoreboard|whole\", "
    "\"about\": \"想确认什么\"}。一轮最多点 2 个。\n"
    "5. 录屏没有声音，不要提语音播报；不要编造英雄名字和击杀数。\n"
    "6. 顶栏正中的计时器就是局内时间（MM:SS）。"
    "**逐张图报**：在 clocks 里写 [{\"id\": 图号, \"clock\": \"MM:SS\"}]，"
    "读不到的那张写 null。编号必须和图号对上，不要串位，"
    "更不要拿别的数字（等级、冷却、比分）凑。"
    "随附里如果有「计时器放大图」，那一格的时刻用 anchor_clock 报。\n"
    "7. 这一段可能根本不是对局（选人、加载、结算）。"
    "那就写 action=\"unclear\"，并在 narrative 里说明这是哪个界面。\n"
    "8. **位置要写成位置码**：detail.position 只能取位置清单里的码"
    "（jungle/mid/side/carry/roam），读不准就写 \"unknown\"。"
    "具体在地图哪儿，写进 detail.where（自由描述）。\n"
    "9. **只输出那一个 JSON 对象**，不要写分析过程、不要写前后说明。"
    "拿不准的地方就写进 detail.note 里，一句话带过。\n"
)


def build_prompt(vocab: Dict[str, Dict[str, str]],
                 seg: Segment,
                 frame_labels: List[str],
                 tool_notes: Optional[List[str]] = None,
                 clock_zoom_label: Optional[str] = None) -> str:
    """造一段的用户提问。

    **这里不能告诉模型「局内大约几点」**
    ------------------------------------
    时间轴是由模型自己读钟建起来的（见 :mod:`app.recognition2.clockaxis`），
    在它读之前我们并不知道这一段是几点几分 —— 告诉它一个数，
    它会顺着那个数编，而不是去读画面。
    """
    lines: List[str] = []
    lines.append("这是第 %d 段：视频第 %.0f~%.0f 秒。"
                 % (seg.index, seg.start, seg.end))
    lines.append("顶栏的局内时刻**由你自己从图里读**，读不到就写 null，不要猜。")
    lines.append("")
    lines.append("随附画面（按时间顺序）：")
    for i, label in enumerate(frame_labels, start=1):
        lines.append("  图%d = %s" % (i, label))
    if clock_zoom_label:
        lines.append("  图%d = %s" % (len(frame_labels) + 1, clock_zoom_label))
    lines.append("")

    if vocab:
        lines.append("动作清单（primary_action / all_actions **尽量**从这里选，"
                     "是给后面检索用的分类；没有合适的就写 unclear，"
                     "**但叙述照样要写清楚**）：")
        for group, items in vocab.items():
            lines.append("  %s: %s" % (
                group, "、".join("%s=%s" % (k, v) for k, v in items.items())))
        lines.append("")

    positions = position_vocabulary()
    if positions:
        lines.append("位置清单（detail.position 只能从这里选）：")
        for code, label in positions.items():
            lines.append("  %s = %s" % (code, label))
        lines.append("  拿不准就写 unknown，具体地点写进 where。")
        lines.append("")

    if tool_notes:
        lines.append("你要的料已经取到（见随附的补充图）：")
        for note in tool_notes:
            lines.append("  - %s" % note)
        lines.append("")

    lines.append("只回答一个 JSON 对象：")
    lines.append('{"clocks": [{"id": 1, "clock": "MM:SS 或 null"}], '
                 '"anchor_clock": "计时器放大图上的 MM:SS 或 null", '
                 '"primary_action": "取值 或 unclear", '
                 '"action_label": "中文短名", '
                 '"narrative": "一句话说清玩家在做什么", '
                 '"all_actions": ["取值", ...], '
                 '"confidence": 0.0~1.0, '
                 '"detail": {"position": "位置码 或 unknown", '
                 '"where": "在哪（自由描述）", '
                 '"target": "对谁/对什么", "note": "补充"}, '
                 '"need_more": [{"frame": 2, "zoom": "minimap", "about": "想确认什么"}]}')
    return "\n".join(lines)


# ---------------------------------------------------------------- 解析

_CLOCK_RE = re.compile(r"(\d{1,2})\s*[:：]\s*(\d{1,2})")


def parse_clock(value: Any) -> Optional[float]:
    """把模型给的时刻解析成局内秒数。解析不了返回 None（不猜）。"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in ("null", "none", "unknown", "n/a", "-"):
        return None
    m = _CLOCK_RE.search(text)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    try:
        return float(text)
    except ValueError:
        return None


def parse_frame_clocks(raw: Any) -> Dict[int, float]:
    """把模型报的逐帧时刻解析成 ``{帧号: 局内秒数}``。

    **为什么要逐帧**
    --------------
    早期版本让模型「每段报一个时刻」，但它读的是段内**哪一帧**没人知道 ——
    这个时刻天生带 ±段长/2 的抖动（实测 18 秒的段 → 残差 9.5~11.3 秒），
    拿来对账永远差个十来秒。
    逐帧报以后，每个时刻都**钉在一张具体的图上**，抖动消失。

    认得出来的写法都收：``[{"id": 1, "clock": "03:12"}]``、
    ``{"1": "03:12"}``、``[{"id": 1, "clock": 192}]``。
    读不到的（null）不进字典 —— **不猜**。
    """
    out: Dict[int, float] = {}
    items: List[Any]
    if isinstance(raw, dict):
        items = []
        for key, value in raw.items():
            try:
                fid = int(key)
            except (TypeError, ValueError):
                continue
            items.append({"id": fid, "clock": value})
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        return out

    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            fid = int(item.get("id") or item.get("frame") or 0)
        except (TypeError, ValueError):
            continue
        if fid <= 0:
            continue
        value = item.get("clock")
        if value is None:
            value = item.get("time")
        seconds = parse_clock(value)
        if seconds is None:
            continue
        out[fid] = float(seconds)
    return out


#: 证据帧文件名里的时间：``seg0034_f1_00569.6.jpg`` → 569.6
_FRAME_TIME_RE = re.compile(r"_(\d+\.\d+)\.jpg$")


def _describe_empty(result: Any, exc: Any) -> str:
    """说清「模型为什么没给答案」。

    **笼统的「模型没有返回内容」是没用的** —— 看了也不知道该改哪边。
    实测最常见的一种是：模型把 token 全花在**思考**上，
    ``stop_reason = max_tokens``，答案一个字没吐（连一个 text 块都没有）。
    这种要明说「是 token 预算不够」，而不是「模型没返回」。
    """
    raw = getattr(result, "raw", None) or {}
    usage = raw.get("usage") or {}
    details = usage.get("output_tokens_details") or {}
    thinking_tokens = details.get("thinking_tokens")
    stop = raw.get("stop_reason")
    blocks = raw.get("content")
    kinds = []
    if isinstance(blocks, list):
        kinds = [b.get("type") for b in blocks if isinstance(b, dict)]

    if thinking_tokens or (stop == "max_tokens" and "text" not in kinds):
        out = usage.get("output_tokens")
        return ("模型把 %s 个 token 全花在思考上了（stop_reason=%s），"
                "没吐出答案 —— 是 token 预算不够，不是模型不会"
                % (thinking_tokens or out or "?", stop or "?"))
    if kinds and "text" not in kinds:
        return ("模型只返回了 %s 块、没有文本块（stop_reason=%s）"
                % ("、".join(str(k) for k in kinds), stop or "?"))
    return "模型没有返回内容"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _token_count(usage: Dict[str, Any]) -> int:
    """从各家返回的 usage 里取 token 数。

    **各家字段名不一样**：Anthropic 形状是 ``input_tokens`` /
    ``output_tokens``，OpenAI 形状是 ``prompt_tokens`` / ``completion_tokens``，
    也有只给 ``total_tokens`` 的。认不出就返回 0 —— 宁可少报，
    不要拿一个编出来的数字去算预算。
    """
    total = usage.get("total_tokens")
    if total:
        try:
            return int(total)
        except (TypeError, ValueError):
            pass
    parts = 0
    for key in ("input_tokens", "output_tokens", "prompt_tokens",
                "completion_tokens"):
        value = usage.get(key)
        if value is None:
            continue
        try:
            parts += int(value)
        except (TypeError, ValueError):
            continue
    return parts


def parse_answer(raw: Any, vocab: Dict[str, Dict[str, str]],
                 labels: Dict[str, str]) -> Dict[str, Any]:
    """把模型回答整理成规范结构。

    **动作码必须认得出**：认不出来的码当成 ``unclear``，
    绝不把模型自己发明的词塞进判定层。
    """
    if not isinstance(raw, dict):
        return {"ok": False, "why": "回答不是 JSON 对象"}

    allowed = set(labels.keys())

    def _norm(code: Any) -> str:
        c = str(code or "").strip()
        if not c or c.lower() in ("unclear", "unknown", "null", "none"):
            return UNCLEAR
        if c in allowed:
            return c
        # 模型可能给了中文名 —— 能对上就对上，对不上就当看不清。
        for k, v in labels.items():
            if c == v:
                return k
        return UNCLEAR

    primary = _norm(raw.get("primary_action") or raw.get("action"))
    all_actions: List[str] = []
    for item in (raw.get("all_actions") or []):
        c = _norm(item)
        if c != UNCLEAR and c not in all_actions:
            all_actions.append(c)
    if primary != UNCLEAR and primary not in all_actions:
        all_actions.insert(0, primary)

    label = str(raw.get("action_label") or "").strip()
    if not label:
        label = labels.get(primary, "") if primary != UNCLEAR else ""

    narrative = str(raw.get("narrative") or "").strip()
    conf = _as_float(raw.get("confidence"), 0.0)
    conf = max(0.0, min(1.0, conf))
    if primary == UNCLEAR:
        # 动作看不清，置信度就必须是 0 —— 不能又「看不清」又「很确定」。
        conf = 0.0

    detail = raw.get("detail")
    detail = dict(detail) if isinstance(detail, dict) else {}

    #: **位置要归一成规范值** —— 规则的前提里就是 ``position == "jungle"``
    #: 这种写法，判定层要逐字比对，自由描述对不上。
    #: 模型如果直接给了「打野」「野区」也能归一；认不出来就 ``unknown``
    #: （**不猜** —— 猜错位置会让每一条带位置的规则都判错，而那种错
    #: 在报告里看不出来，只是安静地一条都不触发）。
    try:
        from ..adapters.honor_of_kings import normalize_position
        position = normalize_position(detail.get("position"))
    except Exception:                                  # noqa: BLE001
        position = "unknown"
    #: 自由描述另存 —— 位置码只有 5 个，细节全在这句里，不能丢。
    where = str(detail.get("where") or "").strip()
    if not where:
        raw_pos = str(detail.get("position") or "").strip()
        #: 模型把描述写在 position 里、没给位置码时，别把描述丢掉。
        where = "" if raw_pos == position else raw_pos
    detail["position"] = position
    if where:
        detail["where"] = where

    return {
        "ok": True,
        # 旧格式（每段一个时刻）仍然认，但只当**旁证**用。
        "clock": parse_clock(raw.get("clock")),
        "clocks": parse_frame_clocks(raw.get("clocks")),
        "anchor_clock": parse_clock(raw.get("anchor_clock")),
        "primary_action": primary,
        "action_label": label,
        "narrative": narrative,
        "all_actions": all_actions,
        "confidence": conf,
        "detail": detail,
        "need_more": raw.get("need_more") or [],
    }


# ---------------------------------------------------------------- 读


class SegmentReader:
    """逐段让模型读「玩家做了什么」，产出 :class:`Timeline`。"""

    def __init__(self, video_path: str, config: Optional[ReaderConfig] = None,
                 transform=None):
        self.video_path = video_path
        self.config = config or ReaderConfig()
        self.traces: List[SegmentTrace] = []
        self.calls = 0
        self.tokens = 0
        self._cursor: Optional[FrameCursor] = None
        #: **画面归一**（转正 + 裁黑边），可选 —— 见 :class:`FrameCursor`。
        self.transform = transform

    # -- 配置 ------------------------------------------------------------

    def _resolve(self):
        """取凭据与模型名（**每次重新读** —— 配置随时会改）。"""
        from app.collection import settings, vlm_client

        data_dir = self.config.data_dir
        if data_dir:
            old = os.environ.get("MATCH_DATA_DIR")
            os.environ["MATCH_DATA_DIR"] = data_dir
            try:
                creds = settings.credentials_for(self.config.slot)
            finally:
                if old is None:
                    os.environ.pop("MATCH_DATA_DIR", None)
                else:
                    os.environ["MATCH_DATA_DIR"] = old
        else:
            creds = settings.credentials_for(self.config.slot)
        model = self.config.model or settings.model_name(self.config.slot)
        return creds, model

    # -- 帧 --------------------------------------------------------------

    def _get_frame(self, t: float):
        if self._cursor is None:
            self._cursor = FrameCursor(self.video_path, transform=self.transform)
        return self._cursor.get(t)

    def _encode(self, frame) -> Optional[bytes]:
        import cv2

        if frame is None:
            return None
        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality])
        return buf.tobytes() if ok else None

    @staticmethod
    def _clock_zoom(frame) -> Optional[bytes]:
        """把顶栏计时器裁出来、放大 8 倍。

        计时器在 1280×574 的画面上只有 **35×12 像素** —— 五个字符挤在
        那一小块里，整帧里几乎看不清（实测模型读整帧时把 02:50 读成 11:21）。
        放大到 280×96 之后字很清楚。

        裁不出来就返回 None（**绝不裁一块错的地方给模型**）。
        """
        import cv2

        if frame is None:
            return None
        box = _ZOOM_BOXES.get("clock")
        if box is None:
            return None
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = box
        px0, py0 = int(x0 * w), int(y0 * h)
        px1, py1 = int(x1 * w), int(y1 * h)
        if px1 - px0 < 4 or py1 - py0 < 4:
            return None
        crop = frame[py0:py1, px0:px1]
        if crop is None or crop.size == 0:
            return None
        big = cv2.resize(crop, (crop.shape[1] * 8, crop.shape[0] * 8),
                         interpolation=cv2.INTER_CUBIC)
        ok, buf = cv2.imencode(".jpg", big, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            return None
        return buf.tobytes()

    def _save_frame(self, seg_index: int, frame_no: int, t: float,
                    data: bytes) -> str:
        """把证据帧落盘 —— ``Beat.frames`` 里的文件名必须真能回看到原画面。"""
        base = self.config.frames_dir or os.path.join(
            os.path.dirname(os.path.abspath(self.video_path)), "frames")
        os.makedirs(base, exist_ok=True)
        name = "seg%04d_f%d_%07.1f.jpg" % (seg_index, frame_no, t)
        with open(os.path.join(base, name), "wb") as fh:
            fh.write(data)
        return name

    # -- 主流程 ----------------------------------------------------------

    def read(self, segments: List[Segment]) -> Timeline:
        """逐段读，产出时间线。"""
        from app.collection import vlm_client
        from app.harness import tools as htools

        cfg = self.config
        vocab = action_vocabulary()
        labels = flat_vocabulary()
        creds, model = self._resolve()

        timeline = Timeline(game="hok")
        timeline.meta.update({
            "video": os.path.basename(self.video_path),
            "model": model,
            "segments": len(segments),
            "frames_dir": cfg.frames_dir,
        })

        if creds is None:
            reason = "没有配置「%s」这个槽位的模型" % cfg.slot
            timeline.notes.append("整局没有读任何一段：%s" % reason)
            for seg in segments:
                timeline.add(self._unclear_beat(seg, reason))
            return timeline

        # 打转检测**每段独立** —— 共享的话，A 段和 B 段都点小地图第 2 帧
        # 会被误判成「原地打转」（模型并没有打转，是两段问的同一处）。
        work = segments[:cfg.limit_segments] if cfg.limit_segments else segments

        # -- 续跑：跳过上一轮已经读成功的段 --
        done: Dict[int, Beat] = {}
        by_time: List[Tuple[float, Beat]] = []
        if cfg.resume_path and os.path.exists(cfg.resume_path):
            try:
                with open(cfg.resume_path, encoding="utf-8") as fh:
                    prev = json.load(fh)
                for raw in prev.get("beats") or []:
                    # **判据是「有没有说得出话」，不是「动作码归类到了没有」。**
                    # 选人/结算那几段的 action 是 unclear（动作码不在词表里），
                    # 但模型**说清楚了**（「画面显示 MVP 与个人战绩结算界面」）——
                    # 那是读出来了，续跑时必须沿用，
                    # 否则会被重读；额度不够时重读只会变成「额度用尽」，
                    # 把本来有结论的段弄丢。
                    if not (raw.get("narrative") or "").strip():
                        continue
                    beat = Beat.from_dict(raw)
                    idx = (raw.get("detail") or {}).get("segment_index")
                    if idx is not None:
                        done[int(idx)] = beat
                    else:
                        # 上一轮是旧格式（没有 segment_index）——
                        # 按视频时刻兜底匹配。用**最近邻**而不是取整做键：
                        # 取整会把相邻两段撞进同一个键，害得其中一段被重读。
                        by_time.append((float(raw.get("video_time") or 0), beat))
                by_time.sort(key=lambda x: x[0])
            except (OSError, ValueError) as exc:
                timeline.notes.append("续跑失败（读不了 %s）：%s"
                                      % (cfg.resume_path, exc))
        if done or by_time:
            timeline.notes.append("续跑：%d 段沿用上一轮结果，只重读剩下的。"
                                  % (len(done) + len(by_time)))

        for seg in work:
            beat = done.get(seg.index)
            if beat is None and by_time:
                # 最近邻匹配（容差 0.6 秒 —— 段与段之间不重叠）。
                nearest = min(by_time, key=lambda x: abs(x[0] - seg.midpoint))
                if abs(nearest[0] - seg.midpoint) <= 0.6:
                    beat = nearest[1]
            if beat is not None:
                timeline.add(beat)
                if cfg.on_progress is not None:
                    cfg.on_progress(len(timeline.beats), len(work))
                continue
            beat, trace = self._read_one(seg, vocab, labels, creds, model,
                                         htools.LoopGuard(window=3), htools,
                                         vlm_client)
            timeline.add(beat)
            self.traces.append(trace)
            if cfg.on_progress is not None:
                cfg.on_progress(len(timeline.beats), len(work))

        skipped = len(segments) - len(work)
        if skipped:
            timeline.notes.append("只读了前 %d 段，剩下 %d 段没读（调试参数）。"
                                  % (len(work), skipped))
            # 没读的段**照样进时间线**（标 unclear）——
            # 时间线必须覆盖整局视频，漏掉的段比读错更让人误解。
            for seg in segments[len(work):]:
                timeline.add(self._unclear_beat(
                    seg, "未读取（调试参数 limit_segments=%d）" % cfg.limit_segments))

        self._note_failures(timeline)

        timeline.meta["calls"] = self.calls
        timeline.meta["tokens"] = self.tokens
        self._build_clock_axis(timeline)
        self._annotate(timeline)
        if self._cursor is not None:
            self._cursor.close()
            self._cursor = None
        return timeline

    @staticmethod
    def _annotate(timeline: Timeline) -> None:
        """补标注 —— 续跑复用的段也要补，所以放在统一收尾处。

        **动作码归类不到、但模型说清楚了**：这是「读出来了但归类不了」，
        记下来让人看得见，而不是悄悄算成「没读出来」。
        """
        for b in timeline.beats:
            if b.action == UNCLEAR and b.action_label:
                b.detail.setdefault("action_label_unmapped", b.action_label)

    # -- 一段 ------------------------------------------------------------

    def _read_one(self, seg: Segment, vocab, labels, creds, model,
                  guard, htools, vlm_client):
        """读一段。返回 (Beat, SegmentTrace)。"""
        from app.harness import tools as htools

        cfg = self.config
        trace = SegmentTrace(index=seg.index, start=seg.start, end=seg.end)

        # -- 取代表帧 --
        frames = []
        labels_for_prompt = []
        saved_names = []
        #: 每一张**成功取到**的帧的真实视频时刻（秒），顺序跟 ``frames`` 一致。
        #: 模型的逐帧时刻读数就是按这个下标对应到画面的。
        frame_pts: List[float] = []
        for i, t in enumerate(seg.frames, start=1):
            frame = self._get_frame(t)
            if frame is None:
                continue
            data = self._encode(frame)
            if data is None:
                continue
            frames.append((frame, data))
            # 标签与文件名都用**这一帧的真实 PTS** ——
            # 请求的时间与真实时间可能差几秒（见 FrameCursor 的说明），
            # 证据帧要能精确回到原视频的那一秒。
            actual = t
            if self._cursor is not None:
                actual = self._cursor.last_time or t
            frame_pts.append(float(actual))
            labels_for_prompt.append("视频 %.1f 秒" % actual)
            if cfg.frames_dir:
                saved_names.append(self._save_frame(seg.index, i, actual, data))

        if not frames:
            trace.error = "抽不出画面"
            return self._unclear_beat(seg, "抽不出画面"), trace

        # -- 定点读钟：把锚点帧的计时器放大，附给模型 --
        #
        # 计时器在 1280×574 的画面上只有 **35×12 像素**，整帧里那五个字符
        # 几乎看不清；放大 8 倍（280×96）就清楚了。
        # 这张小图按 Anthropic 的图片计费公式（宽×高/750）约 **5 个 token**，
        # 几乎不花钱，却换来一个**钉在具体某一帧上**的可靠时刻读数。
        anchor_index = 0
        clock_zoom: Optional[bytes] = None
        clock_zoom_label = ""
        if frame_pts:
            from . import clockaxis
            anchor_index = clockaxis.pick_anchor(frame_pts)
        if cfg.read_clock_zoom and frames:
            clock_zoom = self._clock_zoom(frames[anchor_index][0])
            if clock_zoom is not None:
                clock_zoom_label = ("第 %d 帧（视频 %.1f 秒）的顶栏计时器放大"
                                    % (anchor_index + 1,
                                       frame_pts[anchor_index]))

        tool_notes: List[str] = []
        tool_images: List[bytes] = []
        answer: Optional[Dict[str, Any]] = None
        last_error = ""

        for round_no in range(1, max(1, cfg.max_rounds_per_segment) + 1):
            trace.rounds = round_no
            if self._over_budget():
                last_error = ("超出预算（本局已用 %d 次调用、%d tokens）"
                              % (self.calls, self.tokens))
                break

            # -- 组装请求 --
            content = [vlm_client.text_block(build_prompt(
                vocab, seg, labels_for_prompt, tool_notes, clock_zoom_label))]
            for _f, data in frames:
                content.append(vlm_client.image_block(data))
            if clock_zoom is not None:
                # 固定排在整帧之后、补料图之前 —— 提示词里它是「图N+1」。
                content.append(vlm_client.image_block(clock_zoom))
            for img in tool_images:
                content.append(vlm_client.image_block(img))

            try:
                result = vlm_client.call_model(
                    model, content, creds=creds, system=SYSTEM_PROMPT,
                    max_tokens=cfg.max_tokens, timeout=cfg.timeout)
            except vlm_client.VlmError as exc:
                last_error = vlm_client.explain_error(exc)
                break
            self.calls += 1
            trace.calls += 1
            used = _token_count(result.usage or {})
            self.tokens += used
            trace.tokens += used

            raw = None
            for attempt in range(cfg.empty_retries + 1):
                try:
                    raw = result.json()
                    break
                except vlm_client.VlmError as exc:
                    # **模型返回空 / 不是合法 JSON 是可救的**。
                    # 实测最常见的一种：模型把 token 全花在思考上
                    # （stop_reason=max_tokens），答案一个字没吐。
                    # 这种情况下**用更大的 token 预算重发**就能拿到答案 ——
                    # 用同样的预算重发只会再失败一次。
                    why = _describe_empty(result, exc)
                    if (exc.kind in ("empty", "bad_json")
                            and attempt < cfg.empty_retries):
                        trace.tool_calls.append(
                            "%s；加预算重发第 %d 次" % (why, attempt + 1))
                        budget = min(cfg.max_tokens * (2 ** (attempt + 1)), 16384)
                        try:
                            result = vlm_client.call_model(
                                model, content, creds=creds, system=SYSTEM_PROMPT,
                                max_tokens=budget, timeout=cfg.timeout)
                        except vlm_client.VlmError as exc2:
                            last_error = vlm_client.explain_error(exc2)
                            break
                        self.calls += 1
                        trace.calls += 1
                        used = _token_count(result.usage or {})
                        self.tokens += used
                        trace.tokens += used
                        continue
                    last_error = why
                    break
            if raw is None:
                break

            parsed = parse_answer(raw, vocab, labels)
            if not parsed.get("ok"):
                last_error = str(parsed.get("why") or "回答解析不了")
                break
            answer = parsed

            # -- 要点单就点单 --
            wants = self._parse_need_more(parsed.get("need_more"))
            if not wants or round_no >= cfg.max_rounds_per_segment:
                break

            got_any = False
            for call in wants[:cfg.max_tool_calls_per_round]:
                trace.tool_calls.append("%s#%d/%s" % (call.name, call.frame,
                                                      call.zoom))
                # 打转检测逐次记录 —— 同一个（工具, 帧, 区域）连着要就停。
                if guard.push(call):
                    tool_notes.append(
                        "你要的料重复要过了，这里不再给了，请按已有画面回答。")
                    break
                # 缩放要作用在**模型点名的那一帧**上。
                ctx = ZoomContext([f for f, _ in frames], labels_for_prompt,
                                  index=max(0, call.frame - 1))
                res = htools.dispatch(call, ctx)
                if res.ok and res.image:
                    tool_images.append(res.image)
                    tool_notes.append(
                        "补充图：%s（%s）" % (res.note or call.zoom, call.about or ""))
                    got_any = True
                else:
                    tool_notes.append(
                        "第 %d 帧的「%s」取不到：%s"
                        % (call.frame, _ZOOM_LABELS.get(call.zoom, call.zoom),
                           res.error or "未知原因"))
            if not got_any:
                break

        # -- 出结果 --
        if answer is None:
            trace.error = last_error
            beat = self._unclear_beat(seg, last_error or "没读出来")
            if saved_names:
                beat.frames = saved_names
            return beat, trace

        trace.ok = True
        beat = self._to_beat(seg, answer, saved_names, frame_pts)
        return beat, trace

    # -- 工具请求 --------------------------------------------------------

    def _note_failures(self, timeline: Timeline) -> None:
        """把每段失败的原因汇总成留痕。

        为什么必须留
        ------------
        每段失败的原因只写在 ``beat.why_not_read`` 里 —— **报告层看不到它**。
        而「一个时刻都没读到」→ 时间轴建不起来 → 一段都没有 → 报告是
        一张**空表**。

        **空表最容易被读成「打得没问题」**，实际是「根本没分析」。
        所以这里把原因冒泡成留痕，一路走到报告顶部的说明位
        （``degradation.warnings`` → ``version_note``）。

        两种失败要分清楚
        ----------------
        - ``trace.calls == 0``：**模型压根没调通**（额度用尽、接口 429、
          密钥不对）。这时一句「模型没调通」比「这一段没读出来」有用得多 ——
          玩家看到的是「要充值/换密钥」，不是「这一段没分析」。
        - ``trace.calls > 0``：模型调通了但这一段没读出来
          （没抽到画面、回答解析不了）。这是**个别段**的问题。
        """
        buckets: Dict[Tuple[str, str], int] = {}
        for trace in self.traces:
            if trace.ok:
                continue
            reason = (trace.error or "没读出来").strip()
            kind = "模型没调通" if trace.calls == 0 else "读不出来"
            key = (kind, reason)
            buckets[key] = buckets.get(key, 0) + 1
        if not buckets:
            return
        ordered = sorted(buckets.items(), key=lambda kv: (-kv[1], kv[0][1]))
        for (kind, reason), count in ordered[:3]:
            if kind == "模型没调通":
                timeline.notes.append(
                    "模型没调通，%d 段读不出来（%s）" % (count, reason))
            else:
                timeline.notes.append(
                    "%d 段读不出来（%s）" % (count, reason))

    @staticmethod
    def _parse_need_more(raw: Any) -> List[Any]:
        """把 ``need_more`` 解析成工具调用。

        复用 :func:`app.harness.tools.parse_need_more` ——
        **动作解析只有一份**，将来端点支持原生 tool_call 时只改那一层。
        """
        from app.harness import tools as htools

        out = htools.parse_need_more(raw)
        if out:
            return out
        # 模型有时把 need_more 写成字符串或对象 —— 认得出来的就认。
        if isinstance(raw, dict):
            raw = [raw]
        out = []
        for item in raw or []:
            if not isinstance(item, dict):
                continue
            zoom = str(item.get("zoom") or "whole")
            if zoom not in _ZOOM_BOXES and zoom not in ("whole", "minimap"):
                continue
            try:
                frame = int(item.get("frame") or 1)
            except (TypeError, ValueError):
                frame = 1
            out.append(htools.ToolCall(name="zoom_frame", frame=frame,
                                       zoom=zoom,
                                       about=str(item.get("about") or "")))
        return out

    # -- 出 Beat ---------------------------------------------------------

    def _to_beat(self, seg: Segment, answer: Dict[str, Any],
                 saved_names: List[str],
                 frame_pts: Optional[List[float]] = None) -> Beat:
        detail = dict(answer.get("detail") or {})
        all_actions = list(answer.get("all_actions") or [])
        if all_actions:
            detail["all_actions"] = all_actions

        # -- 时刻：逐帧读数原样留着，等时间轴建好再统一落到锚点上 --
        #
        # 这里**不**直接把某个读数写进 ``game_time``：
        # 单帧读数要经过抗离群拟合、跟相邻帧比过，才知道它是不是读错了。
        clocks = dict(answer.get("clocks") or {})
        if clocks:
            detail["frame_clocks"] = [
                {"id": int(k), "clock": round(float(v), 1)}
                for k, v in sorted(clocks.items())
            ]
        anchor_clock = answer.get("anchor_clock")
        if anchor_clock is not None:
            detail["anchor_clock"] = round(float(anchor_clock), 1)
        if answer.get("clock") is not None:
            # 旧格式（每段一个时刻）—— 只作旁证，不参与定轴。
            detail["clock_raw_model"] = round(float(answer["clock"]), 1)
        if frame_pts:
            detail["frame_times"] = [round(float(t), 2) for t in frame_pts]
        detail["video_span"] = [round(float(seg.start), 2),
                                round(float(seg.end), 2)]

        detail.setdefault("phase", PHASE_UNKNOWN)
        detail["segment_index"] = int(seg.index)

        return Beat(
            game_time=-1.0,
            video_time=0.0,
            action=answer.get("primary_action") or UNCLEAR,
            action_label=answer.get("action_label") or "",
            narrative=answer.get("narrative") or "",
            frames=list(saved_names),
            confidence=float(answer.get("confidence") or 0.0),
            detail=detail,
        )

    # -- 时间轴 ----------------------------------------------------------

    def _build_clock_axis(self, timeline: Timeline) -> None:
        """用模型逐帧读出来的时刻建一条时间轴，再落到每一段上。

        **真源只有一个：多模态模型读到的画面上的时刻。**
        OpenCV 模板匹配那条路已经撤掉 —— 它在这段录屏上 5 次读错 4 次
        （02:38 读成 02:16、02:50 读成 11:21、12:13 读成 02:43…）。

        三步
        ----
        1. 把每一段每一帧的读数收齐（``detail.frame_clocks``）；
        2. 抗离群拟合出一条线（:class:`app.recognition2.clockaxis.ClockAxis`）；
        3. 每段取**锚点帧**（时间居中的那一帧），把时刻和区间落到 Beat 上。
        """
        from . import clockaxis

        cfg = self.config
        readings: List[clockaxis.FrameClock] = []
        for b in timeline.beats:
            detail = b.detail or {}
            pts = list(detail.get("frame_times") or [])
            # ``frame_clocks`` 存的是**列表** ``[{"id": 1, "clock": 192.0}, ...]``，
            # 这里转成 ``{帧号: 秒数}`` 好按帧号取。
            clocks: Dict[int, float] = {}
            for item in (detail.get("frame_clocks") or []):
                try:
                    clocks[int(item.get("id") or 0)] = float(item.get("clock"))
                except (TypeError, ValueError):
                    continue
            anchor_clock = detail.get("anchor_clock")
            anchor_index = clockaxis.pick_anchor(pts) if pts else 0
            for i, t in enumerate(pts, start=1):
                value = clocks.get(i)
                source = "model"
                if value is None and (i - 1) == anchor_index and anchor_clock is not None:
                    value = float(anchor_clock)
                    source = "model_zoom"
                if value is None:
                    continue
                readings.append(clockaxis.FrameClock(
                    frame_id=i, video_t=float(t), game_s=float(value), source=source))

        axis = clockaxis.ClockAxis.build(readings, tolerance=cfg.clock_tolerance)
        timeline.meta["clock_axis"] = axis.to_dict()
        timeline.meta["clock_fit"] = axis.fit.to_dict() if axis.fit else {}

        if not axis.ok:
            timeline.notes.append(
                "局内时刻没定下来：%s（画面上的时刻是模型读出来的，读不到就不填）"
                % (axis.fit.reason if axis.fit else "没有读数"))
        else:
            fit = axis.fit
            if axis.outliers:
                timeline.notes.append(
                    "%d 个时刻读数与拟合线差得远，没采信"
                    "（多半是把等级/冷却当成时间了）：%s"
                    % (len(axis.outliers),
                       "；".join("图%s %s" % (o.get("id"), o.get("reason"))
                                for o in axis.outliers[:5])))
            timeline.notes.append(
                "对局起点在视频第 %.1f 秒，局内时间按真实速度走"
                "（斜率 %.4f，残差 ±%.2f 秒，%d 个读数参与定轴）。"
                % (fit.start_t or 0, fit.slope or 1.0, fit.residual or 0.0,
                   fit.kept or 0))

        self._apply_axis(timeline, axis)

    def _apply_axis(self, timeline: Timeline, axis: Any) -> None:
        """把时间轴落到每一段：锚点 + 区间 + 阶段。"""
        from . import clockaxis

        cfg = self.config
        for b in timeline.beats:
            detail = b.detail or {}
            pts = [float(t) for t in (detail.get("frame_times") or [])]
            if not pts:
                # 续跑来的旧格式段：没有逐帧读数，**保住它原来的时刻**，
                # 别因为新格式没有字段就把结论抹掉。
                if b.game_time >= 0:
                    detail.setdefault("clock_source", "resumed")
                detail["phase"] = self._phase_of(axis.fit if axis else None,
                                                 b.video_time)
                b.detail = detail
                continue
            raw_clocks = detail.get("frame_clocks") or []
            clocks: List[Optional[float]] = []
            for i in range(len(pts)):
                value = None
                for item in raw_clocks:
                    if int(item.get("id") or 0) == i + 1:
                        value = item.get("clock")
                        break
                clocks.append(float(value) if value is not None else None)

            anchor_index = clockaxis.pick_anchor(pts) if pts else 0
            anchor_clock = detail.get("anchor_clock")
            if anchor_clock is not None and 0 <= anchor_index < len(clocks):
                # 定点放大的读数比整帧读得准，优先用它。
                if clocks[anchor_index] is None:
                    clocks[anchor_index] = float(anchor_clock)

            bounds = detail.get("video_span") or None
            resolved = clockaxis.resolve_segment(
                pts, clocks, axis, tolerance=cfg.clock_tolerance,
                anchor_index=anchor_index,
                video_bounds=(float(bounds[0]), float(bounds[1]))
                if bounds else None)
            b.game_time = resolved["game_time"]
            b.game_time_start = resolved["game_time_start"]
            b.game_time_end = resolved["game_time_end"]
            b.anchor_frame = resolved["anchor_index"]
            if pts:
                b.video_time = resolved["video_time"]
                b.video_time_start = resolved["video_time_start"]
                b.video_time_end = resolved["video_time_end"]
            detail["clock_source"] = resolved["clock_source"]
            if resolved.get("clock_residual") is not None:
                detail["clock_residual"] = resolved["clock_residual"]
            detail["phase"] = self._phase_of(axis.fit if axis else None,
                                             b.video_time)
            b.detail = detail

    @staticmethod
    def _phase_of(fit, video_t: float) -> str:
        """这一段在对局的哪个阶段。"""
        from . import beat as beatmod

        if fit is None or not getattr(fit, "ok", False):
            return beatmod.PHASE_UNKNOWN
        start = fit.start_t
        end = getattr(fit, "end_t", None)
        if start is None:
            return beatmod.PHASE_UNKNOWN
        if video_t < start - 2.0:
            return beatmod.PHASE_PRE_GAME
        if end is not None and video_t > end + 5.0:
            return beatmod.PHASE_POST_GAME
        return beatmod.PHASE_IN_GAME

    @staticmethod
    def _unclear_beat(seg: Segment, why: str) -> Beat:
        """读不出来时的 Beat。

        **照样进时间线**，但 ``confidence=0``、``action=unclear`` ——
        判定时算「无法判定」，绝不算「没做」。
        """
        return Beat(
            game_time=-1.0,
            video_time=seg.midpoint,
            action=UNCLEAR,
            action_label="",
            narrative="",
            frames=[],
            confidence=0.0,
            detail={"phase": PHASE_UNKNOWN,
                    "video_span": [round(float(seg.start), 2),
                                   round(float(seg.end), 2)]},
            why_not_read=why,
        )

    # -- 预算 ------------------------------------------------------------

    def _over_budget(self) -> bool:
        cfg = self.config
        if self.calls >= cfg.max_calls_per_game:
            return True
        if cfg.max_tokens_per_game and self.tokens >= cfg.max_tokens_per_game:
            return True
        return False
