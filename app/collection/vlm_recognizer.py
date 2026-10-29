"""用多模态模型读一窗的画面，产出判定内核要的动作与事实。

这个模块是「识别层」的新实现：它替代原来的
:mod:`app.collection.probes`（几何/颜色阈值那一套）。

为什么换掉
----------
探测器判的是**低层信号**：轨迹在不在某区域、折返率多少、位移多大。
而「刷野」「转线」「带线」是**语义** —— 低层信号在它们上面没有阈值，
硬判就是编。模型能看画面，语义就是它的原生能力。

三条规矩
--------
1. **只给封闭选项**：动作取值来自适配器的词表（``MOBA_ACTION_VALUES``），
   位置、局面要素都是枚举。模型不许发明新词。
2. **只答「第几帧」**：时间戳由我们自己换算（帧的秒数是抽帧时记下的）。
   模型估的时间不准，我们记的准。
3. **答不了就说不知道**：``unknown`` 里的动作类转成「缺失」，
   内核判「无法判定」—— 不是「该做而没做」。

产出形状
--------
与 :meth:`app.collection.recognizer.WindowRecognizer.analyze` **完全一致**
（``facts`` / ``actions`` / ``missing`` / ``notes`` / ``probe_actions`` /
``probe_moments`` / ``absent``），所以上层（``to_raw_match``、判定内核、报告）
一行都不用改。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import vlm_client

#: 提示词版本。改了提示词就 +1 —— 缓存键里带着它，
#: 换版之后旧缓存自动失效（不然「哪版提示词得出的结论」就说不清了）。
#:
#: **v2**（2026-10-25）：提示词里多了三样东西 ——
#: 攻略的**分支**（模型要指认第几支成立）、「看到了什么」（画面客观状态）
#: 与「凭什么」（结论）的分工。结论的判法变了，版本号必须跟着变 ——
#: 否则老结论与新结论在报告里分不出来。
# 提示词增加了跨帧控制识别；必须换缓存键，不能复用旧版单纯动作识别结果。
PROMPT_VERSION = "v7"

#: 位置（模型能看出来的位置）。与区域表的说法一致。
POSITION_VALUES = ("泉水", "野区", "河道", "对抗路", "发育路", "中路", "unknown")

#: 局面要素：能读出来的就报，读不出写 unknown。
SITUATION_KEYS = ("position", "has_ult_ready", "in_combat")


@dataclass
class FrameInput:
    """一帧。``t`` 是**原视频里的秒数**（抽帧时记下的，不是模型说的）。"""
    t: float
    data: bytes
    #: 这一帧在磁盘上的文件名（jpg）。
    #:
    #: 为什么要有它：报告里「原视频操作的文字描述」说的是模型在**哪一帧**
    #: 看到的事，片段播放器要放的是**那一帧**的画面。
    #: 没有文件名的话，播放器只能指向回合的另一张切片 ——
    #: 文字说「在龙坑」，画面却来自另一个时刻，玩家点开一看对不上，
    #: 整条结论就跟着塌了。
    name: str = ""


@dataclass
class WindowInput:
    """要读的一窗。"""
    lo: int
    hi: int
    frames: List[FrameInput] = field(default_factory=list)
    panel: Optional[bytes] = None
    #: 这一窗适用的**操作模式清单**（给提示词当「该做什么」的依据）。
    #:
    #: 每一项带攻略原文的前提 / 该做 / 后果（逐字），由
    #: :func:`app.rules.checklist.window_checklist` 从模式库取。
    #:
    #: 清单是**裁判的尺子**，不是让模型照抄的答案 ——
    #: 模型要拿画面去对：前提看不看得见成立、玩家做的是不是这一条要求的事。
    modes: List[Dict[str, Any]] = field(default_factory=list)
    #: 玩家打哪条线（已知就只给这条线的清单）。不知道就是空。
    lane: str = ""
    segment_key: str = ""
    #: 局内 0:00 对应的视频秒（用来把视频秒换算成局内分钟）
    game_start_t: Optional[float] = None

    def window(self):
        from .window_analysis import Window
        return Window(lo=self.lo, hi=self.hi)


# ---------------------------------------------------------------- 词表


def action_vocabulary() -> Dict[str, Dict[str, str]]:
    """动作词表：从适配器取，**不另编一套**。

    两套词表对不上，判定层就认不出来了 —— 报告会出现「模型说做了 A，
    规则查的是 B」。
    """
    try:
        from ..adapters.honor_of_kings import MOBA_ACTION_VALUES
        return {k: dict(v) for k, v in MOBA_ACTION_VALUES.items() if v}
    except Exception:                              # noqa: BLE001
        return {}


def _mode_brief(mode: Dict[str, Any]) -> str:
    """模式压成一行（给提示词当清单）。保留给调试用。

    正式提示词里用的是 :func:`app.rules.checklist.checklist_prompt_block` ——
    它带的是攻略原文的前提 / 该做 / 后果，模型要拿画面去对的就是这三样。
    """
    expect = mode.get("expect") or {}
    values = expect.get("value") or []
    if isinstance(values, str):
        values = [values]
    return "- " + str(mode.get("title") or "") + \
        (f"（期望动作：{'、'.join(str(v) for v in values)}）" if values else "")


# ---------------------------------------------------------------- 提示词

SYSTEM_PROMPT = (
    "你是一个王者荣耀对局复盘助手。你**只看画面**，只描述你看到的东西。\n"
    "硬规矩：\n"
    "1. 不许编造英雄名字、不许编造击杀数、不许提语音播报（录屏没有声音）。\n"
    "2. 看不清的字段一律写 unknown，不要猜、不要美化。\n"
    "3. 动作取值只能从给你的清单里选，不许发明新词。\n"
    "4. 每条动作必须给出**至少两个帧号**（1 起）；只有技能特效只闪过一帧时，"
    "才允许给一个帧号。没有跨帧或清晰单帧依据就不要写这一条。\n"
    "5. 随题给你的那份攻略清单是**尺子**，不是让你抄的答案。\n"
    "   你的活是拿画面去对这份攻略：这一条的前提在画面里**看不看得见成立**，\n"
    "   玩家做的**是不是**这一条要求的事。\n"
    "   前提在画面里看不出来时，写 unknown —— 那时候玩家做的可能就是对的，\n"
    "   不许算他错。\n"
    "6. 只在**真有画面依据**的时候才下结论，并且必须给出帧号。"
)


def build_prompt(inp: WindowInput) -> str:
    """用户提示：帧的编号与时间 + 封闭选项 + 攻略清单 + 输出格式。"""
    lines: List[str] = []
    for i, f in enumerate(inp.frames, start=1):
        minute = ""
        if inp.game_start_t is not None:
            m = (f.t - inp.game_start_t) / 60.0
            minute = f"（局内第 {int(m)} 分 {int((m % 1) * 60):02d} 秒）"
        lines.append(f"第 {i} 帧 = 视频第 {f.t:.1f} 秒 {minute}")

    vocab = action_vocabulary()
    vocab_lines: List[str] = []
    for kind, values in vocab.items():
        vocab_lines.append(f"{kind}：")
        for code, label in values.items():
            vocab_lines.append(f"  - {code}（{label}）")

    modes_block = ""
    if inp.modes:
        from ..rules.checklist import checklist_prompt_block
        lanes = sorted({str(e.get("lane") or "") for e in inp.modes
                        if e.get("lane")})
        lane_hint = ""
        if inp.lane:
            lane_hint = (f"玩家打的是**{inp.lane}**这条线，清单里只有这一条线的模式。")
        elif len(lanes) > 1:
            # 分路没读到时清单是全线的 —— 这时候要**先让模型自己判分路**，
            # 否则它可能对着别的线的打法说话。
            lane_hint = (
                "清单里有各条线的模式。**先判断玩家打的是哪条线**"
                "（人常年待在野区=打野、中路=中路、对抗路、发育路、"
                "河道与泉水之间游走=游走），"
                "然后只对照**那一条线**的模式，别的线一条都不要判。"
            )
        modes_block = (
            "\n这一段在操作模式库里的「该做什么」清单。"
            "这些是**攻略原文**，你的任务是拿画面去对它们，不是照抄：\n"
            + (lane_hint + "\n" if lane_hint else "")
            + checklist_prompt_block(inp.modes) + "\n"
        )

    return (
        f"这是同一局王者荣耀里局内第 {inp.lo} 分 00 秒到第 {inp.hi} 分 00 秒这一段，"
        f"共 {len(inp.frames)} 帧，按时间先后编号：\n"
        + "\n".join(lines)
        + ("\n\n最后一张图是这一段的 HUD 放大面板：左边是小地图、"
           "中间是右下角技能按钮、右边是顶部计时器。\n" if inp.panel else "")
        + modes_block
        + "\n可选动作清单（只能用这些值）：\n" + "\n".join(vocab_lines)
        + "\n\n只看画面回答下面这些：\n"
        + "1. 玩家（你）在这一段**做了哪些事**？从动作清单里选，"
        "给出帧号与一句依据（依据里写你看到了什么，不要写坐标）。\n"
        + "2. 玩家大部分时间在哪个位置？取值：" + "、".join(POSITION_VALUES) + "。\n"
        + "3. 有没有明显的团战（多个英雄在一起、技能特效）？true / false / unknown。\n"
        + "4. 这一段有没有能明确看见的控制事件（击飞、眩晕、冰冻、压制等）？"
        "先按时间顺序核对：谁先放控制 → 谁被控 → 控制后谁接招。"
        "只在至少两帧能看出同一过程时写 cc_state=seen，并在 cc_evidence 写清这三步；"
        "看不清目标、只有普通伤害特效、或无法判断先后时写 unknown，绝不猜。\n"
        + "5. 大招按钮是亮着（可用）还是灰着（冷却）？ready / cooling / unknown。\n"
        + "6. 计时器你读到了吗？读到的数字写在 clock_readings 里（按帧号），"
        "读不出就写空数组。\n"
        + "7. 哪一类动作你**看不出来**？写进 unknown（例如你分不清是在刷野还是在赶路）。\n"
        + "8. **只列出有充分画面证据的偏离**，不要逐条复述整份清单：\n"
        + "   - premise：这一条的前提，在画面里**看得见成立**吗？"
        "visible / not_visible / unknown（前提根本读不出来就写 unknown）。\n"
        + "   - branch：**第几支**成立（上面列出来的分支编号；"
        "看得见成立就写那支的编号，看不清写 0）。\n"
        + "   - matched：玩家这一段做的，是这一支要求的事吗？yes / no / unclear。\n"
        + "   - frames：你据以判断的帧号。\n"
        + "   - premise_evidence：前提为什么成立（只写支持这一支前提的客观画面，"
        "如「一塔护盾标识已消失」；不能证明前提就写 unknown，premise 也写 unknown）。\n"
        + "   - action_evidence：实际操作为什么不匹配（只写玩家实际画面，"
        "如「连续两帧在红 buff 野区攻击野怪」，不得写攻略建议或推测）。\n"
        + "   - why：**凭什么说这一条不成立**（一句话，要**对照着说**："
        "「攻略这一支要的是 X，你做的是 Y」——"
        "只说 X 的话，玩家不知道自己错在哪一步）。\n"
        + "   - did：玩家这一段实际在做什么（一句话）。\n"
        + "   看得清且确实不符合的才写；做对、前提不成立或看不出都不要写。\n"
        + "9. node_gap：**这一整段**里，玩家做的这些事，是不是一条都不在清单里？"
        + "是的话填 {\"lane\": \"这一段的分路\", \"frames\": [...], "
        "\"did\": \"这段在做什么\", \"why\": \"为什么说这整段没走攻略\"}，"
        + "不是的话填空对象 {}\n"
        + "\n只回答一个 JSON 对象，不要加别的文字：\n"
        + json.dumps({
            "actions": [{
                "kind": "objective_action",
                "value": "farm_jungle",
                "frames": [1, 3],
                "why": "第 1、3 帧小地图在红 buff 野区来回移动",
                "confidence": 0.8,
            }],
            "position": "野区",
            "in_combat": "unknown",
            "cc_state": "unknown",
            "cc_frames": [],
            "cc_evidence": "",
            "ult_button": "ready",
            "clock_readings": [{"frame": 1, "text": "02:08"}],
            "unknown": [],
            "review": [{
                "mode_id": "HOK-MODE-JGL-T0400-02",
                "premise": "visible",
                "branch": 1,
                "matched": "no",
                "frames": [3, 5],
                "premise_evidence": "小地图上龙坑图标在闪，暴君已进入刷新前倒计时",
                "action_evidence": "玩家连续两帧在红 buff 野区攻击野怪",
                "did": "在刷红 buff 野区",
                "why": "这会儿龙快刷了，该去占龙坑、惩击留到最后",
                "confidence": 0.8,
            }],
            "node_gap": {},
        }, ensure_ascii=False, indent=2)
    )


# ---------------------------------------------------------------- 解析


@dataclass
class ParsedReading:
    """模型回答解析后的结果（还没映射到内核的形状）。"""
    actions: List[Dict[str, Any]] = field(default_factory=list)
    position: str = "unknown"
    in_combat: str = "unknown"
    cc_state: str = "unknown"
    cc_frames: List[int] = field(default_factory=list)
    cc_evidence: str = ""
    ult_button: str = "unknown"
    clock_readings: List[Dict[str, Any]] = field(default_factory=list)
    unknown: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    #: 模型对攻略清单的**逐条裁决**。原样留着，校验交给
    #: :func:`app.collection.vlm_review.parse_review`（闸门在那里，不在这里）。
    review: List[Dict[str, Any]] = field(default_factory=list)
    #: 「这一整段没做这个时段该做的任何一件事」的判断。
    node_gap: Dict[str, Any] = field(default_factory=dict)


def parse_answer(answer: Any, vocab: Dict[str, Dict[str, str]]) -> ParsedReading:
    """把模型回答解析成结构。**不合规的条目丢掉并记一条说明**。

    丢掉而不是「凑合用」：模型答了一个词表外的取值，
    那是判定层认不出来的动作，塞进去等于污染对局数据。
    """
    out = ParsedReading()
    if not isinstance(answer, dict):
        out.notes.append("模型回答不是一个对象")
        return out

    acts = answer.get("actions")
    if isinstance(acts, list):
        for item in acts:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "").strip()
            value = str(item.get("value") or "").strip()
            if kind not in vocab:
                out.notes.append(f"动作种类不在词表里：{kind!r}（已丢弃）")
                continue
            if value not in vocab[kind]:
                out.notes.append(f"动作取值不在词表里：{kind}:{value}（已丢弃）")
                continue
            frames = []
            for f in (item.get("frames") or []):
                try:
                    frames.append(int(f))
                except (TypeError, ValueError):
                    continue
            if not frames:
                out.notes.append(f"{kind}:{value} 没有给帧号（已丢弃）")
                continue
            why = str(item.get("why") or "").strip()
            if not why:
                # 动作会直接进入规则引擎；没有客观画面依据的标签等同于猜测。
                out.notes.append(f"{kind}:{value} 没有画面依据（已丢弃）")
                continue
            try:
                conf = float(item.get("confidence", 0.6))
            except (TypeError, ValueError):
                conf = 0.6
            out.actions.append({
                "kind": kind,
                "value": value,
                "frames": frames,
                "why": why,
                "confidence": max(0.0, min(1.0, conf)),
            })

    pos = str(answer.get("position") or "").strip()
    out.position = pos if pos in POSITION_VALUES else "unknown"
    combat = str(answer.get("in_combat") or "").strip().lower()
    out.in_combat = combat if combat in ("true", "false", "unknown") else "unknown"
    cc_state = str(answer.get("cc_state") or "").strip().lower()
    # 不同模型会把同一个“画面可见的控制”写成 visible / true / yes；
    # 统一归一，但仍要走下面的“两帧 + 说明”硬闸。
    if cc_state in ("seen", "visible", "true", "yes", "有", "可见"):
        out.cc_state = "seen"
    else:
        out.cc_state = "unknown"
    if out.cc_state == "seen":
        for frame in answer.get("cc_frames") or []:
            try:
                out.cc_frames.append(int(frame))
            except (TypeError, ValueError):
                continue
        out.cc_frames = sorted(set(out.cc_frames))
        out.cc_evidence = str(answer.get("cc_evidence") or "").strip()
        # 控制判断必须有连续画面和客观描述；否则当作没读到。
        if len(out.cc_frames) < 2 or not out.cc_evidence:
            out.cc_state = "unknown"
            out.cc_frames = []
            out.cc_evidence = ""
    ult = str(answer.get("ult_button") or "").strip().lower()
    out.ult_button = ult if ult in ("ready", "cooling", "unknown") else "unknown"

    readings = answer.get("clock_readings")
    if isinstance(readings, list):
        for r in readings:
            if not isinstance(r, dict):
                continue
            try:
                frame = int(r.get("frame"))
            except (TypeError, ValueError):
                continue
            out.clock_readings.append({"frame": frame, "text": str(r.get("text") or "")})

    unknown = answer.get("unknown")
    if isinstance(unknown, list):
        for k in unknown:
            k = str(k or "").strip()
            if k in vocab:
                out.unknown.append(k)

    # 裁决部分：**原样收下**，不在这层做判断。
    #
    # 判断要连着「这一窗有哪些帧」「这一窗读到了哪些动作」「清单里有哪几条」
    # 一起看 —— 那是 :mod:`app.collection.vlm_review` 的活。在这儿判的话，
    # 就没有「前提没被证实就不纠正」「帧号必须存在」这些闸门了。
    review = answer.get("review")
    if isinstance(review, list):
        for item in review:
            if isinstance(item, dict):
                out.review.append(item)
    node_gap = answer.get("node_gap")
    if isinstance(node_gap, dict) and node_gap:
        out.node_gap = node_gap
    return out


# ---------------------------------------------------------------- 识别器


class VlmRecognizer:
    """窗级识别器（多模态模型）。

    ``analyze`` 的返回值与 :class:`app.collection.recognizer.WindowRecognizer`
    同构 —— 上层不区分这两者。
    """

    recognizer_id = "vlm"
    recognizer_version = PROMPT_VERSION

    def __init__(
        self,
        slot: str = "vlm",
        creds: Optional[vlm_client.Credentials] = None,
        data_dir: Optional[str] = None,
        model: str = "",
        max_tokens: int = 3000,
        timeout: float = vlm_client.DEFAULT_TIMEOUT,
        cache_dir: Optional[str] = None,
    ):
        self.slot = slot
        self.creds = creds
        self.data_dir = data_dir
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.cache_dir = cache_dir

    # -- 配置 ---------------------------------------------------------------

    def _resolve(self) -> Tuple[Optional[vlm_client.Credentials], str]:
        """取凭据与模型名。

        凭据优先用外面传进来的（测试用），否则从配置里取
        （配置随时可能改，所以**每次都重新读**）。
        """
        if self.creds is not None:
            return self.creds, self.model or self.creds.model
        from . import settings

        if self.data_dir:
            old = os.environ.get("MATCH_DATA_DIR")
            os.environ["MATCH_DATA_DIR"] = self.data_dir
            try:
                creds = settings.credentials_for(self.slot)
            finally:
                if old is None:
                    os.environ.pop("MATCH_DATA_DIR", None)
                else:
                    os.environ["MATCH_DATA_DIR"] = old
        else:
            creds = settings.credentials_for(self.slot)
        model = self.model or settings.model_name(self.slot)
        return creds, model

    # -- 缓存 ---------------------------------------------------------------

    def _cache_key(self, inp: WindowInput, model: str = "",
                   provider: str = "") -> str:
        """生成与**实际调用模型**绑定的缓存键。

        ``self.model`` 在生产路径通常为空，真正使用的模型来自设置文件。
        缓存若只记前者，用户换模型后会继续拿到旧模型的识别结果；这会让
        报告无法复现，也会把升级模型的效果完全藏掉。
        """
        h = hashlib.sha256()
        h.update(PROMPT_VERSION.encode("utf-8"))
        h.update((model or self.model).encode("utf-8"))
        h.update(provider.encode("utf-8"))
        h.update(build_prompt(inp).encode("utf-8"))
        for f in inp.frames:
            h.update(f"{f.t:.2f}".encode("utf-8"))
            h.update(hashlib.sha256(f.data).digest()[:8])
        if inp.panel:
            h.update(hashlib.sha256(inp.panel).digest()[:8])
        return h.hexdigest()[:32]

    def _cache_path(self, key: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"{key}.json")

    def _cache_read(self, key: str) -> Optional[Any]:
        path = self._cache_path(key)
        if not path or not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def _cache_write(self, key: str, value: Any) -> None:
        path = self._cache_path(key)
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(value, fh, ensure_ascii=False)
        except OSError:
            pass

    # -- 主流程 -------------------------------------------------------------

    def analyze(self, inp: WindowInput) -> Dict[str, Any]:
        """读这一窗，返回与旧识别器同构的结果。"""
        empty = {
            "facts": {}, "actions": {}, "missing": [], "notes": [],
            "probe_actions": [], "probe_moments": [], "absent": [],
        }
        if not inp.frames:
            empty["notes"].append("这一窗没有画面")
            empty["missing"] = list(action_vocabulary().keys())
            return empty

        # 先解析一次配置，再将真正的 provider/model 纳入缓存身份。
        # ``_ask`` 会再次读取配置以保留既有的动态配置行为；二者不一致时
        # 最坏情况是本次不命中缓存，不会把另一模型的旧答案当成本模型答案。
        creds, model = self._resolve()
        provider = (creds.endpoint if creds is not None else "")
        key = self._cache_key(inp, model=model, provider=provider)
        answer = self._cache_read(key)
        if answer is None:
            answer = self._ask(inp)
            if answer is None:
                # 调不通：整窗记「读不出来」，**不许**编。
                empty["notes"].append("模型没调通，这一窗读不出来")
                empty["missing"] = list(action_vocabulary().keys())
                return empty
            self._cache_write(key, answer)

        return self._to_reading(inp, answer)

    def _ask(self, inp: WindowInput) -> Optional[Any]:
        creds, model = self._resolve()
        if creds is None:
            return None
        blocks: List[Dict[str, Any]] = [vlm_client.text_block(build_prompt(inp))]
        for f in inp.frames:
            blocks.append(vlm_client.image_block(f.data))
        if inp.panel:
            blocks.append(vlm_client.image_block(inp.panel))
        try:
            result = vlm_client.call_model(
                model, blocks, creds=creds, system=SYSTEM_PROMPT,
                max_tokens=self.max_tokens, timeout=self.timeout,
                temperature=0.0)
        except vlm_client.VlmError:
            return None
        try:
            return result.json()
        except vlm_client.VlmError:
            return None

    # -- 映射 ---------------------------------------------------------------

    def _to_reading(self, inp: WindowInput, answer: Any) -> Dict[str, Any]:
        vocab = action_vocabulary()
        parsed = parse_answer(answer, vocab)
        notes = list(parsed.notes)
        facts: Dict[str, Dict[str, Any]] = {}
        actions: Dict[str, float] = {}
        probe_actions: List[Dict[str, Any]] = []
        moments: List[Dict[str, Any]] = []
        missing: List[str] = []

        by_index = {i: f for i, f in enumerate(inp.frames, start=1)}

        for act in parsed.actions:
            frames = [i for i in act["frames"] if i in by_index]
            if not frames:
                notes.append(f"{act['kind']}:{act['value']} 的帧号不存在（已丢弃）")
                continue
            conf = self._confidence(act, len(inp.frames), frames)
            label = vocab.get(act["kind"], {}).get(act["value"], act["value"])
            probe_actions.append({
                "kind": act["kind"],
                "value": act["value"],
                "label": label,
                "confidence": conf,
                "source": "vlm",
            })
            actions[act["kind"]] = max(actions.get(act["kind"], 0.0), conf)
            # 事实（动作镜像进 facts，判定层就认得）
            facts.setdefault(act["kind"], {"value": act["value"],
                                           "confidence": conf})
            first = by_index[min(frames)]
            moments.append({
                "t": round(first.t, 1),
                "nx": 0.0, "ny": 0.0,
                # 这一帧的文件名：报告的片段播放器要放**这一帧**，
                # 而不是回合的另一张切片 —— 否则文字与画面对不上号。
                "image": first.name,
                "why": f"{int(first.t // 60)}:{int(first.t % 60):02d} {act['why']}"
                       if act["why"] else f"{int(first.t // 60)}:{int(first.t % 60):02d}",
                "kind": act["kind"],
                "value": act["value"],
            })

        # 局面类事实
        if parsed.position != "unknown":
            facts["position"] = {"value": parsed.position, "confidence": 0.7}
        if parsed.ult_button != "unknown":
            facts["has_ult_ready"] = {
                "value": parsed.ult_button == "ready", "confidence": 0.7}
        if parsed.in_combat != "unknown":
            facts["in_combat"] = {"value": parsed.in_combat == "true",
                                  "confidence": 0.6}
        if parsed.cc_state == "seen":
            cc_frames = [i for i in parsed.cc_frames if i in by_index]
            if len(cc_frames) >= 2:
                # ``cc`` 不是“猜测谁被控”，而是确认同一控制过程跨帧可见。
                # 匹配器只在这里有证据时才不把控制前提降级为“判不了”。
                facts["cc"] = {"value": "seen", "confidence": 0.7}
                first = by_index[cc_frames[0]]
                moments.append({
                    "t": round(first.t, 1), "nx": 0.0, "ny": 0.0,
                    "image": first.name,
                    "why": f"{int(first.t // 60)}:{int(first.t % 60):02d} "
                           f"{parsed.cc_evidence}",
                    "kind": "cc", "value": "seen",
                })

        # 模型读不出哪一类 → 那一类记缺失（内核判「无法判定」）
        for kind in parsed.unknown:
            if kind not in missing:
                missing.append(kind)
                facts.setdefault(kind, {"value": None, "confidence": 0.0})
        # 没被任何动作覆盖、也没被说读不出来的类 → 同样记缺失
        for kind in vocab:
            if kind in facts or kind in missing:
                continue
            missing.append(kind)
            facts[kind] = {"value": None, "confidence": 0.0}

        # 计时器交叉核对：模型读到的时间与我们 OCR 的对不上 → 说明起点偏了
        notes.extend(self._clock_notes(inp, parsed))

        # -- 裁决 -------------------------------------------------------------
        # 模型的逐条判断要过四道闸才算数（见 :mod:`app.collection.vlm_review`）：
        # 模式得在清单里、前提得被画面证实、玩家做过节点认可的事就不纠正、
        # 引用的帧必须真实存在。
        review: List[Dict[str, Any]] = []
        if inp.modes:
            from . import vlm_review

            items, review_notes = vlm_review.parse_review(
                {"review": parsed.review, "node_gap": parsed.node_gap},
                inp.modes, len(inp.frames), probe_actions,
            )
            notes.extend(review_notes)
            for item in items:
                # 片段要指向**模型看到那件事的那一帧** ——
                # 报告里的文字描述说的就是这一帧看到的事。
                # 「看到了什么」那一栏放的是**客观状态**（模型原话），
                # 结论与画面的连接另放「建议理由」那一栏 ——
                # 两栏说同一句话等于白占一行。
                item["moments"] = [
                    {
                        "t": round(by_index[idx].t, 1),
                        "image": by_index[idx].name,
                        "why": item.get("evidence") or item.get("why") or "",
                    }
                    for idx in item.get("frames") or []
                    if idx in by_index
                ]
                if item["kind"] == vlm_review.KIND_NODE:
                    item["mode_id"] = ""
                review.append(item)

        return {
            "facts": facts,
            "actions": actions,
            "missing": missing,
            "notes": notes,
            "probe_actions": probe_actions,
            "probe_moments": moments,
            "absent": [],
            # 模型对攻略清单的裁决（已过硬校验）。
            "review": review,
        }

    def _confidence(self, act: Dict[str, Any], total_frames: int,
                    frames: List[int]) -> float:
        """置信度：模型自评 × 证据覆盖。

        帧越少、给的帧越少 → 越不可信。**不让模型自己说了算**。
        """
        conf = float(act.get("confidence", 0.6))
        # 动作的持续时长不能按“占满整个窗口的比例”扣分：一个 24 帧窗口
        # 里，控制链、闪现进场、补刀都可能只出现 2~3 帧，恰好是用户要
        # 抓的关键动作。已有两帧证据就保留模型置信度；只有一帧才降权。
        if len(frames) == 1:
            conf *= 0.72
        if total_frames < 4:
            conf = min(conf, 0.5)
        return max(0.0, min(1.0, round(conf, 3)))

    def _clock_notes(self, inp: WindowInput,
                     parsed: ParsedReading) -> List[str]:
        """拿模型读到的计时器和我们 OCR 的对一遍。

        对不上说明**对局起点偏了** —— 这是个能直接告诉用户的信号
        （比我们自己猜准，因为模型读的是画面上的真数字）。
        """
        if inp.game_start_t is None or not parsed.clock_readings:
            return []
        notes: List[str] = []
        for item in parsed.clock_readings:
            idx = item.get("frame")
            text = str(item.get("text") or "").strip()
            m = re.search(r"(\d{1,2}):(\d{2})", text)
            if not m or idx not in range(1, len(inp.frames) + 1):
                continue
            frame = inp.frames[idx - 1]
            game_sec = frame.t - inp.game_start_t
            want = int(game_sec // 60) * 60 + int(game_sec % 60)
            got = int(m.group(1)) * 60 + int(m.group(2))
            diff = got - want
            if abs(diff) > 3:
                notes.append(
                    f"第 {idx} 帧的计时器读成 {text}，我们算的是 "
                    f"{want // 60}:{want % 60:02d}（差 {diff:+d} 秒，"
                    f"对局起点可能偏了）")
        return notes
