"""关键时刻的「逐秒细看」：在阵亡/击杀/拆塔/丢塔前后 ±30 秒，一秒一帧，
让模型把过程讲清楚，并**用画面提示那一帧校正事件时刻**。

为什么单独一层
--------------
第一遍看整局是**每批一分钟、最多 32 张**：能看出「这里打起来了」，
但看不出「谁先手、技能怎么交、兵线到哪」，事件时刻也只有 ±几十秒。
所以第一遍只负责**定位关键时刻**，第二遍：
* 一窗一事件（不把几个事件塞进同一段，避免模型按整窗讲故事）；
* 前后各 30 秒、一秒一帧；
* 把事件那一帧（画面弹出「击败 / 阵亡 / 已摧毁」的时刻）标成锚点；
* 让模型回报**真实秒数**，代码用它校正事件时间；
* 对写回来的文字做一致性校验，不合格就保留原描述（不拿幻觉覆盖）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: 窗口半径（秒）。
SPAN = 30.0

#: 一秒一帧。
FPS = 1.0

#: 一次调用最多多少帧（30 + 30 + 1）。
MAX_FRAMES = 61

#: 只细看这四类事件（丢塔本版不输出，不在其列）。
DETAIL_TYPES = ("death", "kill", "tower_taken", "dragon")

KIND_CN = {"death": "阵亡", "kill": "击杀",
           "tower_taken": "拆塔", "dragon": "拿龙"}

#: 每类事件在文字里必须出现的「结果词」——校验用（宁可保留粗描述，不留幻觉）。
RESULT_WORDS = {
    "kill": ("击杀", "击败", "收掉", "收下", "拿下", "击杀提示"),
    "death": ("阵亡", "倒地", "倒下", "被击杀", "被控", "黑屏"),
    "tower_taken": ("摧毁", "拆掉", "推掉", "告破", "塔被", "拆塔", "摧毁防御塔"),
    "dragon": ("龙", "暴君", "主宰", "拿龙", "击杀龙", "控龙"),
}

SYSTEM_PROMPT = """你在看《王者荣耀》对局里**一次关键时刻**（阵亡 / 击杀 / 拆塔 / 丢塔）
前后的逐秒画面。画面按秒排列，每张上面标着它在原视频里的秒数。

【先找提示，再写】
- 先在画面里找这次事件的**提示**：击杀是「击败/第一滴血/双杀」，
  阵亡是「您已阵亡」或黑屏，拆塔是「已摧毁」，丢塔是我方塔被摧毁。
- 提示**不一定**在提示里说的那一帧：首次定位可能差几十秒。
  请**把整窗从头到尾逐帧看一遍**，不要只盯那一帧。
- **整窗都找不到提示，才写 found: false，并且不要写过程**。
- 找到以后，以**提示出现的那一帧**为准，写它前后约 15 秒内的动作；
  窗口里其它时刻的事（打野、清兵、推塔）不要写进这一段。

【严禁编造】
- 没有看到的事一律不写：没看到打蓝 Buff / 打龙 / 反野，就不要提。
- 不要根据 MOBA 常识把「抓人 → 打野 → 击杀」补成一段完整故事。
- 只写画面上真实出现的动作与提示。

【称呼】
- 玩家一律用英雄名（会告诉你），不要写「他」「玩家」；
- 敌方一律写「敌方英雄」，我方写「我方」；不要写红方 / 蓝方；
- 队友写「队友」。

【地点】
- 写「归属 + 路」：己方 / 敌方 + 上路 / 中路 / 下路；不知道归属只写「上路」；
- 河道、野区不写归属；不要写「野区防御塔前」这类组合。

【拆塔：必须有证据】
- 只有**同时**看到「玩家英雄在塔前」且「塔爆（血条归零）」才写 tower_taken；
  写清 side（必须是敌方）、lane、hp_after（0）、player_in_front（true）；
- 只看到「已摧毁」提示、没看到玩家在塔前，就写 found: false，不要编。

【拿龙：必须有证据】
- 只有**同时**看到「玩家在龙前」且「龙死」才写 dragon；
  写清 player_in_front（true）、dragon_died（true）；
- 看不到就写 found: false，不要编。

【时间与位置】
- 叙述里不要写任何时间数字；
- 叙述里不要写路名（上路/中路/下路）—— 位置由系统按小地图标在标题上，
  你只写画面里的动作；
- **禁止写具体技能**（大招、一技能、位移、闪现、控制技能、技能名）——
  你看不清技能，只写「输出 / 攻击 / 换血 / 走位」。

只输出一个 JSON 对象：
{"details": [{"i": 事件序号, "found": true 或 false,
              "t": 提示出现那一帧的原视频秒数（找不到写 null）,
              "text": "过程（found=false 时留空）",
              "side": "己方|敌方（仅拆塔）",
              "lane": "上路|中路|下路（仅拆塔）",
              "hp_after": 0（仅拆塔，塔血归零）,
              "player_in_front": true/false（拆塔/拿龙：玩家在塔/龙面前）,
              "dragon_died": true/false（仅拿龙：龙已死）}]}
"""


@dataclass
class DetailWindow:
    """一个细看窗口：一个事件，原视频 [lo, hi] 秒。"""

    lo: float
    hi: float
    event_index: int

    @property
    def indexes(self) -> List[int]:
        """兼容旧接口：这一窗只负责一个事件。"""
        return [self.event_index]


def build_windows(events: Sequence[Any], span: float = SPAN,
                  max_frames: int = MAX_FRAMES) -> List[DetailWindow]:
    """**一窗一事件**：每个关键时刻单独开一窗，互不合并。"""
    windows: List[DetailWindow] = []
    for index, event in enumerate(events):
        if str(getattr(event, "event_type", "") or "") not in DETAIL_TYPES:
            continue
        start = float(getattr(event, "video_start", 0.0) or 0.0)
        # 以前后各 30 秒为窗，正好 61 帧（事件持续多久不影响窗口）。
        lo = max(0.0, start - span)
        hi = start + span
        hi = min(hi, lo + (max_frames - 1) / max(FPS, 0.1))
        windows.append(DetailWindow(lo=lo, hi=hi, event_index=index))
    return windows


def _times(window: DetailWindow) -> List[float]:
    """一秒一帧，含两端。"""
    count = int(round((window.hi - window.lo) * FPS)) + 1
    return [window.lo + i / FPS
            for i in range(max(1, min(count, MAX_FRAMES)))]


def build_prompt(window: DetailWindow, events: Sequence[Any],
                 hero: str = "") -> str:
    """造这一窗的提问：写清事件类型、锚点帧、只写这一次事件。"""
    hero = (hero or "").strip()
    event = events[window.event_index]
    kind = KIND_CN.get(str(getattr(event, "event_type", "")), "事件")
    at = float(getattr(event, "video_start", 0.0) or 0.0)
    anchor = at - window.lo
    lines: List[str] = []
    lines.append("这一窗是原视频第 %.0f ~ %.0f 秒的画面，按秒排列。"
                 % (window.lo, window.hi))
    if hero:
        lines.append("玩家自己用的是「%s」。" % hero)
    lines.append("事件 #%d：%s。" % (window.event_index, kind))
    lines.append("首次定位：事件大约在原视频第 %.0f 秒（窗口第 %.0f 秒）附近，"
                 "**可能有 ±30 秒误差**。" % (at, anchor))
    lines.append("请在**整个窗口逐帧**找「击败 / 您已阵亡 / 已摧毁」这类提示，"
                 "以真正出现提示的那一帧为准；找到后再写它前后约 15 秒的过程。")
    lines.append("")
    lines.append('只输出 JSON：{"details": [{"i": %d, "t": 那一帧的原视频秒数, '
                 '"text": "过程"}]}' % window.event_index)
    return "\n".join(lines)


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "1", "yes", "是", "有"):
            return True
        if s in ("false", "0", "no", "否", "无", ""):
            return False
        return None
    return bool(value) if value is not None else None


def parse_details(raw: Any) -> Dict[int, Dict[str, Any]]:
    """解析成 {事件序号: {"text": ..., "t": ...}}。"""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = {}
    items = raw.get("details") if isinstance(raw, dict) else None
    out: Dict[int, Dict[str, Any]] = {}
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("i"))
        except (TypeError, ValueError):
            continue
        text = str(item.get("text") or item.get("detail") or "").strip()
        found = item.get("found")
        if isinstance(found, str):
            found = found.strip().lower() not in ("false", "0", "no", "")
        elif found is not None:
            found = bool(found)
        if not text and found is not False:
            continue
        try:
            at = float(item.get("t")) if item.get("t") is not None else None
        except (TypeError, ValueError):
            at = None
        out[index] = {
            "text": text, "t": at, "found": found,
            "side": str(item.get("side") or "").strip(),
            "lane": str(item.get("lane") or "").strip(),
            "hp_after": item.get("hp_after"),
            "player_in_front": _as_bool(item.get("player_in_front")),
            "dragon_died": _as_bool(item.get("dragon_died")),
        }
    return out


def _looks_valid(kind: str, text: str) -> Tuple[bool, str]:
    """粗校验：文字得对得上这类事件，且不是一段空话。"""
    if len(text) < 12:
        return False, "太短"
    words = RESULT_WORDS.get(kind) or ()
    if words and not any(w in text for w in words):
        return False, "没写结果（%s）" % "/".join(words[:3])
    return True, ""


def detail_events(events: List[Any], video_path: str, *, model: str,
                  creds: Any, transform: Any = None, hero: str = "",
                  span: float = SPAN,
                  on_note: Optional[Callable[[str], None]] = None,
                  read_frames: Optional[Callable[..., Any]] = None,
                  call_model: Optional[Callable[..., Any]] = None) -> Dict[str, Any]:
    """逐个关键时刻细看；校正事件时刻，并校验回写的文字。"""
    from app.collection import vlm_client

    if read_frames is None:
        from .frames_watch import read_frames as _read
        read_frames = _read
    if call_model is None:
        call_model = vlm_client.call_model

    windows = build_windows(events, span=span)
    notes: List[str] = []
    calls = 0
    tokens = 0
    fixed = 0
    rejected = 0

    for window in windows:
        event = events[window.event_index]
        kind = str(getattr(event, "event_type", "") or "")
        frames = read_frames(video_path, _times(window), transform)
        if not frames:
            continue
        blocks: List[Dict[str, Any]] = [vlm_client.text_block(
            build_prompt(window, events, hero))]
        for frame in frames:
            blocks.append(vlm_client.text_block("第 %.0f 秒" % frame.t))
            blocks.append(vlm_client.image_block(frame.jpeg))
        try:
            result = call_model(model, blocks, creds=creds,
                                system=SYSTEM_PROMPT, max_tokens=2048,
                                timeout=300.0)
        except Exception as exc:                       # noqa: BLE001
            notes.append("细看第 %.0f~%.0f 秒那一段失败：%s"
                         % (window.lo, window.hi, exc))
            continue
        calls += 1
        usage = getattr(result, "usage", {}) or {}
        tokens += int(usage.get("total_tokens") or 0)
        try:
            raw = result.json()
        except Exception:                              # noqa: BLE001
            notes.append("细看第 %.0f~%.0f 秒：回答不是 JSON。" % (window.lo, window.hi))
            continue

        parsed = parse_details(raw).get(window.event_index)
        if not parsed:
            notes.append("细看第 %.0f~%.0f 秒：没有这一条事件的描述。"
                         % (window.lo, window.hi))
            continue
        if parsed.get("found") is False:
            rejected += 1
            notes.append("第 %d 条（%s）在窗口里**没找到提示帧**，标为未验证。"
                         % (window.event_index, KIND_CN.get(kind, kind)))
            if kind in ("tower_taken", "tower_lost"):
                event.detail_verified = False
            continue
        if parsed.get("t") is None:
            rejected += 1
            notes.append("第 %d 条（%s）没给出提示帧秒数，标为未验证。"
                         % (window.event_index, KIND_CN.get(kind, kind)))
            if kind in ("tower_taken", "tower_lost"):
                event.detail_verified = False
            continue

        # 拆塔：必须有「玩家在塔前 + 塔爆 + 敌方塔」的证据，否则不算数。
        if kind == "tower_taken":
            side = str(parsed.get("side") or "").strip()
            lane = str(parsed.get("lane") or "").strip()
            hp = str(parsed.get("hp_after")).strip() if parsed.get("hp_after") is not None else ""
            in_front = parsed.get("player_in_front")
            if (side != "敌方" or lane not in ("上路", "中路", "下路")
                    or hp not in ("0", "0.0", "0%") or in_front is not True):
                rejected += 1
                notes.append("第 %d 条（拆塔）没看到「玩家在塔前 + 塔爆」，标为未验证。"
                             % window.event_index)
                try:
                    event.detail_verified = False
                except Exception:                      # noqa: BLE001
                    pass
                continue
        # 拿龙：确定性校验（小地图绿圈 + 龙图标消失），不信模型自报。
        if kind == "dragon":
            from . import objective_verify

            in_front, died = objective_verify.verify_dragon(
                video_path, float(event.video_start))
            if not in_front or not died:
                rejected += 1
                notes.append(
                    "第 %d 条（拿龙）确定性校验不过（在龙前=%s、龙死=%s），标为未验证。"
                    % (window.event_index, in_front, died))
                try:
                    event.detail_verified = False
                except Exception:                      # noqa: BLE001
                    pass
                continue

        text = str(parsed.get("text") or "").strip()
        ok, why = _looks_valid(kind, text)
        if not ok:
            rejected += 1
            notes.append("第 %d 条（%s）的细看没通过校验（%s），标为未验证。"
                         % (window.event_index, KIND_CN.get(kind, kind), why))
            if kind in ("tower_taken", "tower_lost"):
                event.detail_verified = False
            continue
        event.narrative = text

        at = parsed.get("t")
        if at is not None and window.lo <= float(at) <= window.hi:
            # 用画面提示那一帧校正事件时刻（偏差太大就不动，防止被带跑）。
            if abs(float(at) - float(event.video_start)) <= span:
                event.video_start = float(at)
                event.video_end = max(float(at), float(event.video_end))
                fixed += 1

    if notes:
        notes.append("细看：校正了 %d 个事件时刻，拒绝 %d 条不合格描述。"
                     % (fixed, rejected))
    if on_note is not None:
        for note in notes:
            on_note(note)
    return {"calls": calls, "tokens": tokens, "fixed": fixed,
            "rejected": rejected, "notes": notes}
