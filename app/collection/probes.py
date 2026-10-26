"""操作模式项的**证据探测器**。

一句话
------
**拿着一条操作模式，去它自己的时间窗里，在帧上找这件事发生过的证据。**

为什么不是「先给画面分类，再拿规则去对账」
--------------------------------------
分类的做法要先把画面分成「刷野 / 开龙 / 打架 / …」，再拿规则去比。
两个问题：一是分类器要维护一整套标签和训练数据；二是分类错了，
错误会被**扩散**到所有规则上 —— 分类器把一段认成「刷野」，
于是所有该开龙的规则都判成「该做而没做」。

探测器不做分类，它**只回答一条模式自己问的那个问题**：
「在 2:00~4:00 这一段，画面上有没有出现开龙的迹象？」
找不到就是找不到 —— 找不到只影响这一条模式，不影响别的。

三态
----
| 状态 | 含义 | 内核怎么判 |
| --- | --- | --- |
| ``found`` | 看到了证据 | 做到了 |
| ``absent`` | 信号**可读**，但明确没有 | 该做而没做 |
| ``unknown`` | 读不出来 | 无法判定 |

**只有强探测器能给 ``absent``** —— 读不出「没有」就不要说「没有」。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .probe_specs import ProbeSpec, load_specs, spec_key
from .skilltrack import Motion, TrackPoint

#: 三态取值
FOUND = "found"
ABSENT = "absent"
UNKNOWN = "unknown"

#: 位置信号可读性的下限：覆盖率低于它，连「在哪」都不确定，
#: 更谈不上「在不在龙坑」—— 这时候一律 unknown。
MIN_COVERED = 0.55


@dataclass
class ProbeResult:
    """一个探测器的结论。"""

    kind: str
    value: str
    status: str = UNKNOWN
    confidence: float = 0.0
    #: 证据：``[{"t": 视频秒, "nx": .., "ny": .., "why": "看到了什么"}]``
    moments: List[Dict[str, Any]] = field(default_factory=list)
    #: 读不出来/判没做的理由
    reason: str = ""

    @property
    def found(self) -> bool:
        return self.status == FOUND

    @property
    def absent(self) -> bool:
        return self.status == ABSENT

    @property
    def unknown(self) -> bool:
        return self.status == UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "value": self.value, "status": self.status,
                "confidence": round(self.confidence, 3),
                "moments": self.moments, "reason": self.reason}


@dataclass
class ProbeContext:
    """探测器能看到的东西。"""

    #: 窗口内的运动特征
    motion: Motion
    #: 窗口内的轨迹点
    points: List[TrackPoint] = field(default_factory=list)
    #: 每个点落在哪条路（对抗路/发育路/中路）—— 没有大小龙图标就没有
    lanes: List[str] = field(default_factory=list)
    #: 龙坑图标位置（主宰/暴君）
    marks: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    #: 龙坑图标随时间的**有无**：``{坑名: [(视频秒, 在不在)]}``
    pit_state: Dict[str, List[Tuple[float, bool]]] = field(default_factory=dict)
    #: 本局是不是红方视角（None = 分不出来）
    flipped: Optional[bool] = None
    #: 把点归到区域的函数（``(nx, ny) -> 区域名``）。
    #:
    #: 探测器用它算区域占比，**取证据时刻时也用它** ——
    #: 两处必须用同一个口径，否则「占比 80% 在野区」却给不出证据点。
    zone_of: Optional[Any] = None

    def zone_ratio(self, name: str) -> float:
        return float(self.motion.zone_ratio.get(name, 0.0))

    def zone_sec(self, name: str) -> float:
        return float(self.motion.zone_sec.get(name, 0.0))

    def lane_ratio(self, names: Sequence[str]) -> float:
        if not self.lanes:
            return 0.0
        hit = sum(1 for l in self.lanes if l in names)
        return hit / float(len(self.lanes))

    def readable(self) -> bool:
        """位置信号够不够读。"""
        return self.motion.ok and self.motion.covered >= MIN_COVERED


def _moments(points: Sequence[TrackPoint], limit: int = 3,
             why: str = "") -> List[Dict[str, Any]]:
    """从一段点里挑几个当**证据**。

    挑的是「最分散的几个」—— 报告要能看出这段时间他在这片区域里
    来回跑，而不是只给一个点。
    """
    if not points:
        return []
    if len(points) <= limit:
        chosen = list(points)
    else:
        chosen = [points[0], points[len(points) // 2], points[-1]]
    return [{"t": round(p.t, 1), "nx": round(p.nx, 4), "ny": round(p.ny, 4),
             "why": why} for p in chosen]


def _unknown(spec: ProbeSpec, reason: str) -> ProbeResult:
    return ProbeResult(kind=spec.kind, value=spec.value, status=UNKNOWN,
                        reason=reason)


def _zone_probe(ctx: ProbeContext, spec: ProbeSpec) -> ProbeResult:
    """在某个区域待够时间就算（开龙 / 回城 / 拆高地 / 回防）。"""
    zone = spec.params.get("区域", "")
    need = float(spec.params.get("停留秒数下限", 10))
    if not zone:
        return _unknown(spec, "规格里没写区域")
    if not ctx.readable():
        return _unknown(spec, "这段画面里认不出位置，读不了")
    dwell = ctx.zone_sec(zone)
    if dwell >= need:
        pts = [p for p in ctx.points
               if ctx.motion.zone_sec.get(zone, 0) > 0]  # 占位，下面按区域过滤
        # 精确取在该区域里的点
        pts = _points_in_zone(ctx, zone)
        return ProbeResult(
            kind=spec.kind, value=spec.value, status=FOUND,
            confidence=min(1.0, dwell / max(1.0, need * 1.5)),
            moments=_moments(pts, why=f"在{zone}停留 {dwell:.0f} 秒"),
            reason=f"在{zone}停留 {dwell:.0f} 秒（门槛 {need:.0f} 秒）")
    if spec.allow_absent:
        return ProbeResult(kind=spec.kind, value=spec.value, status=ABSENT,
                           confidence=0.6,
                           reason=f"没在{zone}待够 {need:.0f} 秒")
    return _unknown(spec, f"没在{zone}待够 {need:.0f} 秒，但这类信号不够干净，"
                          f"不敢说「没做」")


def _points_in_zone(ctx: ProbeContext, zone: str) -> List[TrackPoint]:
    """轨迹点里落在某个区域的那些点。

    用**上下文里的**区域函数（跟算占比时同一个口径）——
    占比说「80% 在野区」却给不出证据点，报告里就没法回看。
    """
    if ctx.zone_of is not None:
        return [p for p in ctx.points if ctx.zone_of(p.nx, p.ny) == zone]
    from . import minimap as mm_mod

    geo = mm_mod.MapGeometry.load()
    return [p for p in ctx.points
            if geo.zone_of(p.nx, p.ny, ctx.marks, ctx.flipped) == zone]


def _farm_probe(ctx: ProbeContext, spec: ProbeSpec) -> ProbeResult:
    """在野区**来回折返**（打小野怪）才算刷野。

    只看「在野区」不够 —— 开龙前路过野区也算在野区。
    区分靠的是**折返率**：刷野是小范围反复来回，方向一直在变；
    路过是单向穿过去。
    """
    zone = spec.params.get("野区", "己方野区")
    need_ratio = float(spec.params.get("野区占比下限", 0.30))
    need_revisit = float(spec.params.get("折返率下限", 0.20))
    max_straight = float(spec.params.get("直线性上限", 0.60))
    if not ctx.readable():
        return _unknown(spec, "这段画面里认不出位置，读不了")
    ratio = ctx.zone_ratio(zone)
    if ratio >= need_ratio and ctx.motion.revisit >= need_revisit \
            and ctx.motion.straightness <= max_straight:
        pts = _points_in_zone(ctx, zone)
        return ProbeResult(
            kind=spec.kind, value=spec.value, status=FOUND,
            confidence=min(1.0, 0.4 + 0.3 * ratio + 0.3 * ctx.motion.revisit),
            moments=_moments(pts, why=f"在{zone}来回走（折返率 "
                                      f"{ctx.motion.revisit:.2f}）"),
            reason=f"{ratio:.0%} 的时间在{zone}，且在里面反复折返")
    if spec.allow_absent and ratio < need_ratio * 0.5:
        return ProbeResult(kind=spec.kind, value=spec.value, status=ABSENT,
                           confidence=0.6,
                           reason=f"只有 {ratio:.0%} 的时间在{zone}，不像在打野")
    return _unknown(spec, f"{zone}占比 {ratio:.0%}、折返率 "
                          f"{ctx.motion.revisit:.2f}，说不准是不是在打野")


def _recall_probe(ctx: ProbeContext, spec: ProbeSpec) -> ProbeResult:
    """回城 = 回到泉水。"""
    zone = spec.params.get("区域", "泉水")
    need = float(spec.params.get("停留秒数下限", 6))
    if not ctx.readable():
        return _unknown(spec, "这段画面里认不出位置，读不了")
    dwell = ctx.zone_sec(zone)
    if dwell >= need:
        return ProbeResult(
            kind=spec.kind, value=spec.value, status=FOUND,
            confidence=min(1.0, 0.5 + dwell / (need * 3)),
            moments=_moments(_points_in_zone(ctx, zone),
                             why=f"在{zone}停留 {dwell:.0f} 秒"),
            reason=f"在{zone}停留 {dwell:.0f} 秒")
    if spec.allow_absent:
        return ProbeResult(kind=spec.kind, value=spec.value, status=ABSENT,
                           confidence=0.6, reason=f"没回过{zone}")
    return _unknown(spec, f"没回过{zone}，但说不准")


def _where(ctx: ProbeContext, pt) -> str:
    """一个轨迹点**在地图上的什么地方** —— 说人话。

    以前这里写的是归一化坐标（``(0.05,0.96)``），
    那是给算法看的，不是给人看的：页面上「从 (0.05,0.96) 到 (0.14,0.18)」
    这句话谁也读不出是什么意思。换成区域名（「从泉水走到对抗路」）
    才知道它在讲哪。

    认不出区域时退回坐标 —— 宁可给个难看的坐标，也不给个瞎编的地名。
    """
    if pt is None:
        return "未知位置"
    if ctx.zone_of is not None:
        try:
            zone = ctx.zone_of(pt.nx, pt.ny)
        except Exception:                              # noqa: BLE001
            zone = None
        if zone:
            return str(zone)
    return f"({pt.nx:.2f},{pt.ny:.2f})"


def _rotate_probe(ctx: ProbeContext, spec: ProbeSpec) -> ProbeResult:
    """转线支援 = **净位移大而路径短**（直穿过去，不是绕着走）。"""
    need_net = float(spec.params.get("净位移下限", 0.22))
    max_ratio = float(spec.params.get("路径净位移比上限", 0.65))
    if not ctx.readable():
        return _unknown(spec, "这段画面里认不出位置，读不了")
    m = ctx.motion
    if m.net_len >= need_net and m.path_len > 0 \
            and (m.net_len / m.path_len) >= max_ratio:
        return ProbeResult(
            kind=spec.kind, value=spec.value, status=FOUND,
            confidence=min(1.0, 0.4 + m.net_len),
            moments=_moments([m.start, m.end] if m.start and m.end else [],
                             why=f"从{_where(ctx, m.start)}走到{_where(ctx, m.end)}"),
            reason=f"净位移 {m.net_len:.2f}、总路程 {m.path_len:.2f} —— 直穿过去")
    if spec.allow_absent and m.net_len < need_net * 0.5:
        return ProbeResult(kind=spec.kind, value=spec.value, status=ABSENT,
                           confidence=0.6,
                           reason=f"净位移只有 {m.net_len:.2f}，没有跨区移动")
    return _unknown(spec, f"净位移 {m.net_len:.2f}、总路程 {m.path_len:.2f}，"
                          f"跨没跨区说不准")


def _push_probe(ctx: ProbeContext, spec: ProbeSpec) -> ProbeResult:
    """带线 = 沿一条边路**单向持续推进**。

    两个特征一起看：直线性强（不绕）+ 在边路上（不是在中路或野区）。
    """
    need_straight = float(spec.params.get("直线性下限", 0.65))
    need_net = float(spec.params.get("净位移下限", 0.10))
    need_lane = float(spec.params.get("边路占比下限", 0.45))
    need_span = float(spec.params.get("时长下限秒", 0))
    if not ctx.readable():
        return _unknown(spec, "这段画面里认不出位置，读不了")
    m = ctx.motion
    lane_ratio = ctx.lane_ratio(("对抗路", "发育路"))
    if not ctx.lanes:
        return _unknown(spec, "认不出大小龙图标，分不出是哪条路")
    ok = (m.straightness >= need_straight and m.net_len >= need_net
          and lane_ratio >= need_lane and m.span_sec >= need_span)
    if ok:
        return ProbeResult(
            kind=spec.kind, value=spec.value, status=FOUND,
            confidence=min(1.0, 0.4 + 0.4 * m.straightness),
            moments=_moments([m.start, m.end] if m.start and m.end else [],
                             why="沿边路一路推进"),
            reason=f"直线性强（{m.straightness:.2f}）、{lane_ratio:.0%} 的时间在边路")
    if spec.allow_absent and m.straightness < need_straight * 0.6:
        return ProbeResult(kind=spec.kind, value=spec.value, status=ABSENT,
                           confidence=0.6,
                           reason=f"直线性只有 {m.straightness:.2f}，不是一路推过去")
    return _unknown(spec, f"直线性 {m.straightness:.2f}、边路占比 {lane_ratio:.0%}，"
                          f"说不准是不是在带线")


def _hold_probe(ctx: ProbeContext, spec: ProbeSpec) -> ProbeResult:
    """卡位蹲伏 = 位置几乎不变，而且不在泉水（那叫回城）。"""
    max_net = float(spec.params.get("净位移上限", 0.05))
    max_path = float(spec.params.get("总路程上限", 0.10))
    if not ctx.readable():
        return _unknown(spec, "这段画面里认不出位置，读不了")
    m = ctx.motion
    if m.net_len <= max_net and m.path_len <= max_path:
        if ctx.zone_sec("泉水") > 0:
            return _unknown(spec, "一直在泉水附近，那更像回城而不是卡位")
        return ProbeResult(
            kind=spec.kind, value=spec.value, status=FOUND,
            confidence=0.55,
            moments=_moments([m.start] if m.start else [], why="整段基本没动"),
            reason=f"整段净位移只有 {m.net_len:.3f}")
    return _unknown(spec, f"净位移 {m.net_len:.2f}，不算卡位")


def _dragon_probe(ctx: ProbeContext, spec: ProbeSpec) -> ProbeResult:
    """开龙 / 拿下龙。

    - ``contest_dragon``（开小龙团）/ ``fight_baron``（开大龙团）：
      在对应坑里待够时间就算**参与了龙的争夺**；
    - ``take_dragon``（拿下小龙）：还要**龙图标消失** ——
      只在坑里待着可能是被打跑了，那不叫「拿下」。
    """
    pit = spec.params.get("龙坑", "")
    need = float(spec.params.get("停留秒数下限", 8))
    if not ctx.readable():
        return _unknown(spec, "这段画面里认不出位置，读不了")
    dwell = ctx.zone_sec(f"{pit}坑") if pit else 0.0
    if dwell < need:
        if spec.allow_absent:
            return ProbeResult(kind=spec.kind, value=spec.value, status=ABSENT,
                               confidence=0.6,
                               reason=f"没在{pit}坑待够 {need:.0f} 秒")
        return _unknown(spec, f"没在{pit}坑待够 {need:.0f} 秒")
    if spec.params.get("需要图标消失"):
        states = ctx.pit_state.get(f"{pit}坑") or []
        seen = [b for _t, b in states]
        if len(seen) < 2 or not (seen[0] and not seen[-1]):
            return _unknown(spec,
                            f"在{pit}坑待了 {dwell:.0f} 秒，但看不出龙被拿下"
                            f"（图标没变化）")
        return ProbeResult(
            kind=spec.kind, value=spec.value, status=FOUND,
            confidence=0.8,
            moments=_moments(_points_in_zone(ctx, f"{pit}坑"),
                             why=f"在{pit}坑停留且龙图标消失"),
            reason=f"在{pit}坑待了 {dwell:.0f} 秒，龙图标随后消失")
    return ProbeResult(
        kind=spec.kind, value=spec.value, status=FOUND,
        confidence=min(1.0, 0.4 + dwell / (need * 2)),
        moments=_moments(_points_in_zone(ctx, f"{pit}坑"),
                         why=f"在{pit}坑停留 {dwell:.0f} 秒"),
        reason=f"在{pit}坑待了 {dwell:.0f} 秒")


#: 探测器实现表：``规格键 -> 函数``
PROBES = {
    "objective_action:farm_jungle": _farm_probe,
    "objective_action:invade_jungle": _farm_probe,
    "objective_action:take_dragon": _dragon_probe,
    "objective_action:contest_dragon": _dragon_probe,
    "objective_action:fight_baron": _dragon_probe,
    "objective_action:recall": _recall_probe,
    "objective_action:siege": _zone_probe,
    "objective_action:defend_inner": _zone_probe,
    "positioning_action:rotate_support": _rotate_probe,
    "positioning_action:push_wave": _push_probe,
    "objective_action:split_push": _push_probe,
    "positioning_action:hold_position": _hold_probe,
}


def run_probe(spec: ProbeSpec, ctx: ProbeContext) -> ProbeResult:
    """跑一个探测器。规格里没实现的 → unknown（不猜）。"""
    fn = PROBES.get(spec.key)
    if fn is None or not spec.enabled:
        return _unknown(spec, "这一类还没有探测器")
    try:
        return fn(ctx, spec)
    except Exception as exc:                      # noqa: BLE001
        return _unknown(spec, f"探测器出错：{type(exc).__name__}")


def run_probes(ctx: ProbeContext,
               specs: Optional[Dict[str, ProbeSpec]] = None,
               keys: Optional[Sequence[str]] = None) -> Dict[str, ProbeResult]:
    """跑一批探测器，返回 ``{规格键: 结论}``。

    ``keys`` 给定时只跑这些；不给就跑**规格文件里全部**。
    """
    table = specs if specs is not None else load_specs()
    out: Dict[str, ProbeResult] = {}
    for key in (keys if keys is not None else table.keys()):
        spec = table.get(key)
        if spec is None:
            continue
        out[key] = run_probe(spec, ctx)
    return out


def probe_for_action(specs: Dict[str, ProbeSpec],
                     kind: str, value: str) -> Optional[ProbeSpec]:
    """按「动作种类 + 动作取值」找规格。"""
    return specs.get(spec_key(kind, value))
