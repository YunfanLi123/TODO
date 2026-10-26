"""自动识别层：把一窗的若干帧，变成这一窗的**事实与动作**。

为什么是「窗」而不是「帧」
------------------------
用户定的口径：「要在一个时间区间对帧做综合分析」。
一条规则要的是「这一段有没有做 X」，不是「这一帧有没有做 X」——
所以识别单元是窗，窗内的帧一起看，出一份结论。

为什么不是端到端模型
--------------------
GitHub 调研（详见 ``docs/recognizer-design.md``）：能跑王者的公开项目，
用的都是**固定区域裁剪 → 局部判读**（wzry_ai 的 YOLO ONNX 跑在裁好的小块上），
没有「端到端看懂一局」的现成方案。

我们走同一条路，但判据要**能指回画面**：「英雄在小地图上的位置」+
「镜头里是龙坑还是兵线」两条证据都能复查，判错时看得出是哪儿错。
端到端模型给的答案没法复查 —— 而这份产物的价值全在可复查。

读不出来就**不读**
------------------
每个读取器都有「没标定 / 没认出」两种情况，都产出**缺失**而不是猜。
缺失的事实进内核 = 判成「无法判定」，报告会写明缺哪一项；
猜一个值 = 替玩家编一条他没下过的结论。

自动识别的产物
--------------
.. code-block:: text

    窗（时间区间）
      ├─ 事实：局势类的（局内时间 / 金币差 / 击杀差 / 大招就绪）
      └─ 动作：操作类的（刷野 / 转线 / 开龙 / 带线 / 惩击…）

    事实与动作都在 ``app/collection/video.py`` 的数据结构里，
    判定层与报告层完全不用改。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: 标定文件（随包打包，exe 里也在）。
_REGIONS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "hud_regions.json")

#: 判定「这个区域算不算认出来了」的置信度下限。
#:
#: 低于它就当**没认出来** —— 报告里会写「缺这一项」，
#: 而不是拿一个八成把握的值去判规则。
MIN_CONFIDENCE = 0.6

#: 大招按钮中心的饱和度门槛：高于它算**就绪**。
#:
#: 实测就绪时 84.6~87.6、冷却时 16.3~39.5，中间没有模糊地带，
#: 所以取中间值即可（60）。
ULT_SAT_READY = 60.0


@dataclass
class HudRegions:
    """HUD 各区域的**相对坐标**（占画布宽高的比例）。

    换机型、换录屏软件、换清晰度时重跑 ``tools/calibrate_hud.py`` 标定，
    **识别代码不动** —— 这是这套东西能跨设备用的前提。

    键位档位
    --------
    王者荣耀的**键位能自己拖**，不同玩家的技能按钮位置能差几十像素。
    所以除了公共区域，还存了若干**档位**：每档记一份右下角按钮群的
    签名（位置清单）和它自己的大招框。扫描时先稳定检出按钮群，
    匹配上哪一档就用哪一档 —— 匹配不上就记缺失，不猜。
    """

    regions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    anchors: Dict[str, Tuple[float, float, float, float]] = field(
        default_factory=dict)
    #: 键位档位：``{档位名: {"大招": {...}, "按钮": [...], "覆盖": {...}}}``
    profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    #: 当前选中的档位名；None = 没匹配上，用公共区域
    profile_name: Optional[str] = None

    @classmethod
    def load(cls, path: Optional[str] = None) -> "HudRegions":
        p = path or _REGIONS_JSON
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return cls()
        regions = {}
        for name, spec in (data.get("regions") or {}).items():
            box = spec.get("box") or []
            if len(box) != 4:
                continue
            regions[name] = {
                "box": tuple(float(v) for v in box),
                "purpose": spec.get("用途", ""),
                "calibrated": bool(spec.get("已标定")),
                # 「档位敏感」= 这个区域的位置**跟着键位走**。
                # 一个档位都没匹配上时，这个坐标是**不可信的** ——
                # 宁可记缺失，也不要拿一个偏了几十像素的框去读。
                "layout_sensitive": bool(spec.get("档位敏感")),
            }
        anchors = {}
        for name, box in (data.get("anchors") or {}).items():
            if len(box) == 4:
                anchors[name] = tuple(float(v) for v in box)
        profiles: Dict[str, Dict[str, Any]] = {}
        for name, spec in (data.get("档位") or {}).items():
            if not isinstance(spec, dict):
                continue
            profiles[name] = spec
        return cls(regions=regions, anchors=anchors, profiles=profiles)

    # -- 键位档位 ---------------------------------------------------------

    def select_profile(self, buttons: Sequence[Any]) -> Optional[str]:
        """按检出的按钮群挑一个档位。挑不中返回 None（调用方记缺失）。

        判据是**大招按钮的位置**：每个档位存了自己大招的归一化坐标，
        检出的按钮里离它最近的那个在容差内就算命中。
        """
        from . import skillbar

        if not self.profiles or not buttons:
            return None
        best: Optional[str] = None
        best_d = 1e9
        for name, spec in self.profiles.items():
            ult = spec.get("大招") or {}
            hit = skillbar.match_button(buttons, ult)
            if hit is None:
                continue
            d = abs(hit.x - float(ult.get("x", 0.0))) \
                + abs(hit.y - float(ult.get("y", 0.0)))
            if d < best_d:
                best_d = d
                best = name
        self.profile_name = best
        return best

    def _override(self, name: str) -> Optional[Dict[str, Any]]:
        """当前档位对某个区域的覆盖。"""
        if not self.profile_name:
            return None
        spec = self.profiles.get(self.profile_name) or {}
        return (spec.get("覆盖") or {}).get(name)

    def box(self, name: str) -> Optional[Tuple[float, float, float, float]]:
        """某个区域的相对坐标框 ``(x0, y0, x1, y1)``。没标定返回 None。"""
        ov = self._override(name)
        if ov and len(ov.get("box") or []) == 4:
            return tuple(float(v) for v in ov["box"])
        spec = self.regions.get(name)
        if not spec:
            return None
        return spec["box"]

    def calibrated(self, name: str) -> bool:
        ov = self._override(name)
        if ov is not None:
            return bool(ov.get("已标定"))
        spec = self.regions.get(name)
        if not spec:
            return False
        if not spec.get("calibrated"):
            return False
        # 档位敏感的区域：一个档位都没匹配上 → 这个坐标不可信 → 不认。
        if spec.get("layout_sensitive") and self.profiles \
                and not self.profile_name:
            return False
        return True

    def layout_note(self) -> str:
        """没匹配到键位档位时的一句人话（写进报告的说明里）。"""
        if not self.profiles:
            return ""
        if self.profile_name:
            return f"键位档位：{self.profile_name}"
        return ("键位档位没认出来（右下角按钮群跟已知的都对不上）—— "
                "依赖技能键位位置的项目记为缺失")


    def crop(self, frame, name: str):
        """按标定框裁出一块画面。没标定返回 None。"""
        box = self.box(name)
        if box is None or frame is None:
            return None
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = box
        px0, py0 = int(x0 * w), int(y0 * h)
        px1, py1 = int(x1 * w), int(y1 * h)
        if px1 - px0 < 2 or py1 - py0 < 2:
            return None
        if px0 < 0 or py0 < 0 or px1 > w or py1 > h:
            return None
        return frame[py0:py1, px0:px1]


@dataclass
class WindowReading:
    """一窗的识别结果。"""

    #: 识别出来的事实：``{事实名: (取值, 置信度)}``
    facts: Dict[str, Tuple[Any, float]] = field(default_factory=dict)
    #: 识别出来的动作：``{动作名: 置信度}``
    actions: Dict[str, float] = field(default_factory=dict)
    #: 哪些项**没**读出来（连同原因）—— 报告里要写明缺什么
    missing: List[str] = field(default_factory=list)
    #: 人话说明
    notes: List[str] = field(default_factory=list)
    #: **探测器**认出来的动作（``kind`` → 动作取值）。
    #:
    #: 与 :attr:`actions` 不同：``actions`` 是「这一类动作做过没有」，
    #: 这里记的是「做过哪几件」，报告里「你实际在做什么」靠它。
    probe_actions: List[Dict[str, Any]] = field(default_factory=list)
    #: 证据时刻：``[{"t": 视频秒, "nx": .., "ny": .., "why": "看到了什么"}]``
    probe_moments: List[Dict[str, Any]] = field(default_factory=list)
    #: 明确**没做**的动作类（探测器读到信号、但没找到）。
    #:
    #: 只有强探测器能给这一项 —— 读不出「没有」就不要说「没有」。
    absent: List[str] = field(default_factory=list)

    def fact(self, name: str) -> Optional[Any]:
        """取一个事实的值；没读到返回 None（不猜）。"""
        item = self.facts.get(name)
        return None if item is None else item[0]

    def confidence(self, name: str) -> float:
        item = self.facts.get(name)
        return 0.0 if item is None else float(item[1])

    def has_action(self, name: str) -> bool:
        return name in self.actions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": {k: {"value": v[0], "confidence": round(v[1], 3)}
                      for k, v in self.facts.items()},
            "actions": {k: round(v, 3) for k, v in self.actions.items()},
            "missing": list(self.missing),
            "notes": list(self.notes),
            "probe_actions": list(self.probe_actions),
            "probe_moments": list(self.probe_moments),
            "absent": list(self.absent),
        }


class WindowRecognizer:
    """窗口级识别器。

    现在能读什么、读不出什么，写在 :meth:`analyze` 里 ——
    **读不出就写进 missing，绝不猜**。
    """

    def __init__(self, regions: Optional[HudRegions] = None):
        self.regions = regions or HudRegions.load()

    # -- 主入口 ---------------------------------------------------------
    def analyze(self, frames: Sequence[Any], window: Any = None,
                track: Optional[Sequence[Any]] = None,
                t0: Optional[float] = None,
                t1: Optional[float] = None) -> WindowReading:
        """对**一窗的若干帧**做综合分析，给出这一窗的事实与动作。

        ``frames`` 是窗内的画面（按时间先后）。空窗或没画面 → 全部缺失。

        ``track`` 是**整局**的英雄轨迹（:mod:`app.collection.skilltrack`
        的点），``t0``/``t1`` 是这一窗在原视频里的秒数区间 ——
        探测器靠它把轨迹切出这一段。
        """
        out = WindowReading()
        if not frames:
            out.missing = ["画面"]
            out.notes.append("这一窗没有画面")
            return out

        note = self.regions.layout_note()
        if note:
            out.notes.append(note)

        self._read_scoreboard(frames, out)
        self._read_ult_ready(frames, out)
        self._read_position(frames, out)
        self._run_probes(frames, out, track=track, t0=t0, t1=t1)
        return out

    # -- 操作模式项的证据探测器 -------------------------------------------
    def _run_probes(self, frames: Sequence[Any], out: WindowReading,
                    track: Optional[Sequence[Any]] = None,
                    t0: Optional[float] = None,
                    t1: Optional[float] = None) -> None:
        """拿着操作模式项，去这一段里找证据。

        三态怎么变成内核认得的东西
        ----------------------
        - **found** → 记一个动作（``kind`` + 取值）+ 一条事实；
        - **absent** → 记一条「采信但为空」的事实：信号可读、但没找到
          ⇒ 内核判「该做而没做」；
        - **unknown** → 记一条「未采信」的事实 ⇒ 内核判「无法判定」。

        同一个动作种类下有多个探测器时，**有 found 就不记 missing**：
        找到了就说明这一类信号读得出来；没找到的那几个只是「没探测到」，
        不是「读不出来」。这时规则里靠 ``subject`` 兜底 ——
        动作不匹配前提的规则判「不适用」，不会变成假偏差。
        """
        from . import minimap as mm_mod
        from . import probes as probe_mod
        from . import skilltrack as st

        specs = probe_mod.load_specs()
        if not specs:
            out.notes.append("没有探测器规格文件")
            return

        geo = mm_mod.MapGeometry.load()
        mini = mm_mod.minimap_crop(frames[0], geo)
        marks = mm_mod.find_objective_marks(mini) if mini is not None else {}
        flipped = mm_mod.map_is_flipped(mini) if mini is not None else None

        pts: List[Any] = []
        total: Optional[int] = None
        if track and t0 is not None and t1 is not None:
            pts, total = st.slice_track_rows(track, t0, t1)
        motion = st.motion_of(
            pts, zone_of=lambda nx, ny: geo.zone_of(nx, ny, marks, flipped),
            flipped=flipped, total=total)
        lanes = [mm_mod.lane_of(p.nx, p.ny, marks) for p in pts]
        ctx = probe_mod.ProbeContext(
            motion=motion, points=list(pts), lanes=lanes, marks=marks,
            flipped=flipped,
            zone_of=lambda nx, ny: geo.zone_of(nx, ny, marks, flipped))
        results = probe_mod.run_probes(ctx, specs)

        # 按动作种类归并：found 优先，其次 absent，最后 unknown
        by_kind: Dict[str, Dict[str, List[Any]]] = {}
        for key, res in results.items():
            spec = specs.get(key)
            if spec is None:
                continue
            slot = by_kind.setdefault(spec.kind, {"found": [], "absent": [],
                                                  "unknown": []})
            slot[res.status].append((spec, res))
        for kind, slot in by_kind.items():
            if slot["found"]:
                for spec, res in slot["found"]:
                    out.probe_actions.append({
                        "kind": kind, "value": spec.value,
                        "label": spec.label, "confidence": res.confidence,
                        "source": "probe"})
                    out.actions[kind] = max(out.actions.get(kind, 0.0),
                                            res.confidence)
                    out.facts[kind] = (spec.value, res.confidence)
                    out.probe_moments.extend(
                        dict(m, kind=kind, value=spec.value) for m in res.moments)
            elif slot["absent"]:
                spec, res = slot["absent"][0]
                # 记**取值**（kind:value），报告里才说得出
                # 「画面里没看到开小龙」而不是笼统一句「没找到操作」。
                out.absent.append(spec.key)
                out.facts[kind] = (None, 1.0)
                out.notes.append(f"{kind}：{res.reason}")
            else:
                spec, res = slot["unknown"][0]
                if kind not in out.missing:
                    out.missing.append(kind)
                out.facts[kind] = (None, 0.0)
                out.notes.append(f"{kind}：{res.reason}")

    # -- 英雄在小地图上的位置 ---------------------------------------------
    def _read_position(self, frames: Sequence[Any], out: WindowReading) -> None:
        """读英雄在小地图上的位置。

        判据：小地图上有一处**绿色标记**。在**玩家自己视角**的录屏里它
        跟着玩家走（模板录屏实证：开局在泉水位 (0.04,0.95)，之后沿地图移动）。

        但同一批录屏里也有另一种画面（回放/观战）—— 那里的绿圈是**固定图标**，
        像素位置 5 分钟不变（2670x1200 原片帧 584/842/868/896 实测都是 134,240）。
        两种画面单帧长得一样，所以靠一条**保守**规则分开：

        **整段都没动过 → 当固定图标 → 不报位置**（记缺失）。
        宁可少一条事实，也不给一个错的坐标（错的坐标会把「在龙坑」判成「在野区」，
        报告上完全看不出来）。

        产出两个事实：

        - ``hero_pos``：``(横比例, 纵比例)``，小地图内的位置（0~1）
        - ``hero_zone``：落在哪个区域（**区域表为空时没有这个事实** ——
          区域名必须对着真实地图标，猜一个就是编）
        """
        from . import minimap as minimap_mod

        try:
            track = minimap_mod.track_hero(frames)
        except Exception:                              # noqa: BLE001
            track = None
        if track is None:
            out.missing.append("英雄位置")
            out.notes.append("这一段里没能稳定认出小地图上的绿色标记")
            return
        if not track.moved:
            out.missing.append("英雄位置")
            out.notes.append("小地图上的绿色标记整段没动过，"
                             "分不清是玩家位置还是固定图标，不报位置")
            return
        out.facts["hero_pos"] = ((round(track.nx, 3), round(track.ny, 3)),
                                 round(min(1.0, 0.4 + track.agreement * 0.6), 3))
        if track.zone:
            out.facts["hero_zone"] = (track.zone, round(track.agreement, 3))

    # -- 顶栏：金币差 / 击杀差 -------------------------------------------
    def _read_scoreboard(self, frames: Sequence[Any], out: WindowReading) -> None:
        """读顶栏的金币与击杀数字。

        两件事分开说，因为它们的状态不一样：

        - **金币差**：王者顶栏**默认不显示经济**，除非玩家在设置里开了。
          所以大多数录屏里这一项根本不存在 → 记缺失（不是记 0）。
        - **击杀差**：顶栏左右各一串数字，靠**颜色**分开
          （左队偏蓝、右队偏红，中间灰白的「vs」自然被排除）。
          要认得准必须有数字模板；模板是标定值，没标定时记缺失。
        """
        from . import scoreboard as sb

        out.missing.append("金币差")
        out.notes.append("王者顶栏默认不显示经济，这一项读不到")

        if not (self.regions.calibrated("scoreboard_kill_left")
                and self.regions.calibrated("scoreboard_kill_right")):
            out.missing.append("击杀差")
            out.notes.append("顶栏击杀数还没标定")
            return
        templates = sb.load_templates()
        if not (templates.get("left") or templates.get("right")):
            out.missing.append("击杀差")
            out.notes.append("顶栏数字模板还没标定（跑 tools/build_scoreboard_templates.py）")
            return

        left = right = None
        for frame in frames:
            crop_l = self.regions.crop(frame, "scoreboard_kill_left")
            crop_r = self.regions.crop(frame, "scoreboard_kill_right")
            if crop_l is not None:
                left = sb.read_number(crop_l, "left", templates)
            if crop_r is not None:
                right = sb.read_number(crop_r, "right", templates)
            if left is not None and right is not None:
                break
        if left is None or right is None:
            out.missing.append("击杀差")
            return
        out.facts["kill_diff"] = (int(left) - int(right), 1.0)
    # -- 大招就绪 -------------------------------------------------------
    def _read_ult_ready(self, frames: Sequence[Any], out: WindowReading) -> None:
        """大招按钮上的**冷却遮罩**。

        判据用**色彩饱和度**
        ------------------
        王者里技能就绪时按钮上是**彩色的技能图标**，冷却时会被一层
        灰暗的遮罩盖住（还带一圈扇形扫描），于是图标**掉色**。

        实测（1280x574 高清模板录屏，按钮中心 50% 区域）：

        ==================  ==========
        状态                  饱和度均值
        ==================  ==========
        就绪                   84.6 ~ 87.6
        冷却中                 16.3 ~ 39.5
        ==================  ==========

        两簇之间**没有模糊地带**，所以阈值取中间值即可。

        取窗内**多数**：偶尔一帧被特效盖住不算数。

        大招框是**档位敏感**的
        ----------------------
        王者键位能自己拖，不同玩家的技能簇能差几十像素。框一旦对不上，
        这里读到的就是**别的按钮**，「就绪/冷却」变成随机数 ——
        而报告上只会显示「没触发」，看不出错在哪。所以框对不上就**不读**。
        """
        if not self.regions.calibrated("skill_ult"):
            out.missing.append("大招就绪")
            if self.regions.profiles and not self.regions.profile_name:
                out.notes.append("键位档位没认出来，大招按钮的位置定不了")
            else:
                out.notes.append("大招按钮还没标定")
            return
        import numpy as np

        # 逐帧判定 + **去抖**：要连续两帧同状态才承认切换。
        #
        # 实测不抖的话会在 404/408/414 秒这样快速翻转 ——
        # 那是技能特效和按键高亮造成的瞬时假信号，不是真的放好了。
        states: List[bool] = []
        for frame in frames:
            crop = self.regions.crop(frame, "skill_ult")
            if crop is None:
                continue
            ch, cw = crop.shape[:2]
            if ch < 4 or cw < 4:
                continue
            core = crop[int(ch * 0.25):int(ch * 0.75),
                        int(cw * 0.25):int(cw * 0.75)]
            if core.size == 0:
                continue
            b = core[:, :, 0].astype(np.float32)
            g = core[:, :, 1].astype(np.float32)
            r = core[:, :, 2].astype(np.float32)
            sat = np.maximum(np.maximum(b, g), r) - np.minimum(np.minimum(b, g), r)
            states.append(float(sat.mean()) > ULT_SAT_READY)
        if not states:
            out.missing.append("大招就绪")
            return
        # 去抖后的状态序列
        debounced: List[bool] = []
        for i, st in enumerate(states):
            if len(debounced) and i + 1 < len(states) \
                    and states[i + 1] != st:
                continue          # 只有一帧、下一帧就变了 → 当抖动丢掉
            debounced.append(st)
        if not debounced:
            debounced = states
        ready = sum(1 for s in debounced if s)
        out.facts["has_ult_ready"] = (ready * 2 >= len(debounced), 0.8)


def cv2_color_gray(img):
    """小工具：把 BGR 裁剪转成灰度（避免顶层 import cv2）。"""
    import cv2

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
