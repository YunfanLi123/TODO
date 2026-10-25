"""操作模式项的证据探测器。

盯三件事：
1. **三态**：看到了 / 明确没有 / 读不出来；
2. **只有强探测器能给「明确没有」** —— 读不出「没有」就不要说「没有」；
3. **读不出来就是读不出来** —— 点不够、认不出位置时一律 unknown，
   宁可不给，也不给一个可能错的结论。
"""
from __future__ import annotations

import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from app.collection import probes as pm              # noqa: E402
from app.collection import skilltrack as st          # noqa: E402
from app.collection.probe_specs import ProbeSpec     # noqa: E402


def _points(seq, step=2.0):
    return [st.TrackPoint(t=step * i, nx=nx, ny=ny)
            for i, (nx, ny) in enumerate(seq)]


def _ctx(points, zone_of=None, lanes=None, covered=None):
    motion = st.motion_of(points, zone_of=zone_of)
    if covered is not None:
        motion.covered = covered
        motion.ok = covered >= pm.MIN_COVERED
    return pm.ProbeContext(
        motion=motion,
        points=list(points),
        lanes=list(lanes or ["中路附近"] * len(points)),
        zone_of=zone_of,
    )

def _spec(key, strong=True, allow_absent=None, params=None):
    kind, _, value = key.partition(":")
    return ProbeSpec(
        key=key, kind=kind, value=value, desc=value, strong=strong,
        allow_absent=(strong if allow_absent is None else allow_absent),
        params=params or {},
    )


# ---------------------------------------------------------------- 刷野
def _jungle_zone(nx, ny):
    return "己方野区" if ny > 0.5 else "中路附近"


def test_farm_jungle_found_when_oscillating_in_the_jungle():
    """在野区里来回折返 → 刷野。"""
    seq = [(0.20, 0.70), (0.26, 0.72), (0.20, 0.74), (0.26, 0.70),
           (0.20, 0.73), (0.26, 0.71), (0.20, 0.72), (0.26, 0.74),
           (0.20, 0.70), (0.26, 0.72)]
    ctx = _ctx(_points(seq), zone_of=_jungle_zone)
    spec = _spec("objective_action:farm_jungle", params={"野区": "己方野区"})
    res = pm.run_probe(spec, ctx)
    assert res.found, res.reason
    assert res.moments, "找到了证据却没有给证据时刻"


def test_farm_jungle_absent_when_never_in_the_jungle():
    """一直在中路、一次没进野区 → 明确没刷野。"""
    seq = [(0.40 + 0.01 * i, 0.45) for i in range(12)]
    ctx = _ctx(_points(seq), zone_of=_jungle_zone)
    spec = _spec("objective_action:farm_jungle", params={"野区": "己方野区"})
    res = pm.run_probe(spec, ctx)
    assert res.absent, res.reason


def test_weak_probe_never_says_absent():
    """弱探测器（卡位蹲伏）读不出「没有」→ 只能 unknown。"""
    seq = [(0.40 + 0.01 * i, 0.45) for i in range(12)]
    ctx = _ctx(_points(seq), zone_of=_jungle_zone)
    spec = _spec("positioning_action:hold_position", strong=False,
                 allow_absent=False, params={"净位移上限": 0.05})
    res = pm.run_probe(spec, ctx)
    assert res.unknown, "弱探测器不该给「明确没做」"


def test_unknown_when_position_is_not_readable():
    """覆盖率不够 → 位置都不确定，更不能说「没做」。"""
    seq = [(0.20, 0.70)] * 12
    ctx = _ctx(_points(seq), zone_of=_jungle_zone, covered=0.2)
    spec = _spec("objective_action:farm_jungle", params={"野区": "己方野区"})
    res = pm.run_probe(spec, ctx)
    assert res.unknown
    assert "认不出位置" in res.reason


# ---------------------------------------------------------------- 转线
def test_rotate_support_found_on_a_long_straight_move():
    """净位移大而路径短 → 转线支援。"""
    seq = [(0.10 + 0.03 * i, 0.10 + 0.01 * i) for i in range(12)]
    ctx = _ctx(_points(seq))
    spec = _spec("positioning_action:rotate_support",
                 params={"净位移下限": 0.22, "路径净位移比上限": 0.65})
    res = pm.run_probe(spec, ctx)
    assert res.found, res.reason


def test_rotate_support_absent_when_nothing_moved():
    """位置读得出来、整段没动 → 明确**没有**转线。

    这是「absent」的典型用法：信号可读，所以能说「没做」——
    跟「读不出来」（unknown）是两件事。
    """
    seq = [(0.10, 0.10)] * 12
    ctx = _ctx(_points(seq))
    spec = _spec("positioning_action:rotate_support",
                 params={"净位移下限": 0.22, "路径净位移比上限": 0.65})
    res = pm.run_probe(spec, ctx)
    assert res.absent, res.reason


# ---------------------------------------------------------------- 泉水
def test_recall_found_when_at_the_fountain():
    """回到泉水待着 → 回城补给。"""
    seq = [(0.05, 0.92)] * 10

    def zone(nx, ny):
        return "泉水"

    ctx = _ctx(_points(seq), zone_of=zone)
    spec = _spec("objective_action:recall",
                 params={"区域": "泉水", "停留秒数下限": 6})
    res = pm.run_probe(spec, ctx)
    assert res.found, res.reason


# ---------------------------------------------------------------- 规格
def test_shipped_specs_are_consistent():
    """规格文件里的每一条都要有实现，且「允许判没做」的必须是强探测器。"""
    from app.collection.probe_specs import load_specs

    specs = load_specs()
    assert specs, "探测器规格文件是空的"
    for key, spec in specs.items():
        if spec.allow_absent:
            assert spec.strong, f"{key} 允许判「没做」但不是强探测器"
        if spec.key in pm.PROBES:
            assert spec.enabled, f"{key} 标了启用却没有实现"


def test_run_probes_returns_one_result_per_spec():
    from app.collection.probe_specs import load_specs

    specs = load_specs()
    ctx = _ctx(_points([(0.2, 0.7)] * 10), zone_of=_jungle_zone)
    out = pm.run_probes(ctx, specs)
    assert set(out) == set(specs)
    for res in out.values():
        assert res.status in (pm.FOUND, pm.ABSENT, pm.UNKNOWN)
