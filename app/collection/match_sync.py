"""局内时间同步：把「视频第几秒」对齐到「对局第几秒」。

对局起点是**测出来的**，不是推出来的
------------------------------------
用户交代得很清楚：

  「一定要进入游戏后，也就是 BP 结束后，加载完成后，在对局中才有时间」
  「不同对局视频进入对局的时间会不同」

于是局内时间有两个来源，缺一不可：

  1. **起点**（本模块）：计时器只在对局中才有 —— 它随对局 HUD 一起出现，
     出现的那一刻就是局内 0:00。每段视频进入对局的时刻不同，所以起点
     必须**每段视频单独检测**。写死一个数、或者按视频总长按比例推，
     都会错几十秒，后面每一条带时间窗的模式都会判到错误的阶段。
  2. **读数**（:mod:`app.collection.clock`）：清晰录屏能直接读出 MM:SS。

两条互为验证：读数拟合出的起点与检测出的起点应当相差无几；
差得多，说明其中一个不靠谱 —— 那就不该拿它去判带时间窗的模式。

怎么测起点
----------
对局里的 HUD 一直在（小地图在左上、摇杆在左下、技能在右下），一直到结算。
而 BP 界面、选人界面、加载界面**偶尔**也会撞上「小地图在左上、摇杆在左下」
这个图案，但只撞一小段。

所以取**最长连续成立段**的起点，而不是「第一次成立的时刻」：

  - 第一次成立 → 会被 BP 界面的巧合带偏；
  - 最长连续段 → 对局那一段一路到底，BP 的巧合段短得多。

实测（聊天软件转发的王者荣耀录屏，画面被压到 217 像素宽）：
BP 与加载占了前 120 秒，对局 HUD 从第 120 秒起一路稳定到录屏结束。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: 连续成立多久才算「真的在对局里」。
#:
#: 对局 HUD 会一直保持到结算，所以几秒的巧合不算数。
#: 取 5 秒：BP 界面里的小地图形状最多持续一两个采样点，
#: 而真对局里会连续几百个。
MIN_RUN_SEC = 5.0

#: 平滑窗口（秒）：窗口内多数成立就算成立。
#:
#: 对局里偶尔有技能特效扫过左下角，那一两帧会突然变亮
#: （实测对局期间约 10% 的采样点会这样，BP 期间只有 1.5%）。
#: 不平滑的话，连续段会被这些瞬间打断成碎段，起点就测不准了。
SMOOTH_SEC = 2.0

#: 允许把间隔不超过这个秒数的两段**合并**成一段。
#:
#: 对局本来就是**连续的一整场**，中间不该有洞。就算某一小段判据被特效
#: 带偏了（实测最长的一次约 9 秒），也不该因此把对局切碎。
#: 真正该分开的两种界面之间隔着几十秒的 BP 与加载，容差远小于那个间隔，
#: 不会被误合。
GAP_TOLERANCE_SEC = 15.0

#: 两条来源（读数拟合的起点 / 检测的起点）允许的差值（秒）。
#:
#: 超过这个差就说明其中一个不靠谱，报告里要说明白，别装作很确定。
CROSS_CHECK_TOLERANCE = 12.0


@dataclass
class MatchStart:
    """对局起点（局内 0:00 对应的视频时刻）。"""

    #: 对局第 0 秒对应的视频时刻（秒）
    t: Optional[float] = None
    #: 判定依据：``hud``（HUD 就位）/ ``clock``（时钟读数拟合）/ ``manual``（手填）
    source: str = ""
    #: 给人看的一句话说明
    reason: str = ""
    #: 是否可用
    ok: bool = False
    #: 这一段连续成立了多久（秒）—— 越长越可信
    run_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": round(self.t, 1) if self.t is not None else None,
            "source": self.source,
            "reason": self.reason,
            "ok": self.ok,
            "run_sec": round(self.run_sec, 1),
        }


def _smooth(pts: List[Tuple[float, bool]], smooth_sec: float) -> List[Tuple[float, bool]]:
    """窗口内多数成立就算成立。

    对局里偶尔有技能特效扫过左下角，那一两帧会突然变亮；不平滑的话，
    连续段会被这些瞬间打断成碎段，起点就测不准了。
    """
    if smooth_sec <= 0 or len(pts) < 3:
        return pts
    n = len(pts)
    out: List[Tuple[float, bool]] = []
    for i, (t, _ok) in enumerate(pts):
        lo = i
        while lo > 0 and t - pts[lo][0] <= smooth_sec:
            lo -= 1
        hi = i
        while hi < n - 1 and pts[hi][0] - t <= smooth_sec:
            hi += 1
        seg = pts[lo:hi + 1]
        wins = sum(1 for _, v in seg if v)
        out.append((t, wins * 2 >= len(seg)))
    return out


def detect_match_start(samples: Sequence[Tuple[float, bool]],
                       min_run_sec: float = MIN_RUN_SEC,
                       smooth_sec: float = SMOOTH_SEC,
                       gap_tolerance_sec: float = GAP_TOLERANCE_SEC) -> MatchStart:
    """从「HUD 是否就位」的逐点序列里找出对局起点。

    ``samples`` 是 ``[(视频秒, HUD 是否就位), ...]``，按时间先后排好即可。

    三步：
      1. **平滑**（窗口多数表决）—— 挡住特效带来的瞬间误判；
      2. 找出所有「成立段」，把**间隔很小的段合并**
         —— 对局是连续的一整场，中间不该有洞；
      3. 取**最长**那一段的起点 —— BP 界面偶尔会撞上对局的 HUD 图案，
         但只撞一小段；对局那一段会一路持续到录屏结束。
    """
    pts = sorted((float(t), bool(ok)) for t, ok in samples)
    if not pts:
        return MatchStart(reason="没有采样，无法判断对局从哪一秒开始")

    pts = _smooth(pts, smooth_sec)

    # 找出所有「成立段」
    runs: List[Tuple[float, float]] = []
    cur_start: Optional[float] = None
    cur_end = 0.0
    for t, ok in pts:
        if ok:
            if cur_start is None:
                cur_start = t
            cur_end = t
        elif cur_start is not None:
            runs.append((cur_start, cur_end))
            cur_start = None
    if cur_start is not None:
        runs.append((cur_start, cur_end))
    if not runs:
        return MatchStart(
            reason="整段录屏里都没测到对局 HUD —— 录的可能是 BP、结算或别的内容"
        )

    # 合并间隔很小的段：对局是连续的一整场，中间不该有洞
    merged: List[Tuple[float, float]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= gap_tolerance_sec:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    start, end = max(merged, key=lambda r: r[1] - r[0])
    run = end - start

    if run < min_run_sec:
        return MatchStart(
            run_sec=run,
            reason=(
                f"最像对局的一段只有 {run:.0f} 秒（不足 {min_run_sec:.0f} 秒），"
                f"不敢认成对局开始"
            ),
        )

    return MatchStart(
        t=start,
        source="hud",
        ok=True,
        run_sec=run,
        reason=(
            f"对局 HUD 从视频第 {start:.0f} 秒开始就位，"
            f"一直保持到第 {end:.0f} 秒（连续 {run:.0f} 秒）"
        ),
    )


def game_seconds_at(start_t: Optional[float], video_t: float) -> Optional[float]:
    """视频时刻对应的局内秒数。起点没测出来就返回 None（不猜）。"""
    if start_t is None:
        return None
    return max(0.0, float(video_t) - float(start_t))


def cross_check(clock_start_t: Optional[float],
                match_start_t: Optional[float]) -> Optional[Dict[str, Any]]:
    """比较两条来源给出的对局起点。

    读数拟合出来的起点（:mod:`app.collection.clock`）与检测出来的起点
    （本模块）本该一致。差得多说明其中一条不靠谱 —— 报告里要写清楚，
    不能装作很确定。
    """
    if clock_start_t is None or match_start_t is None:
        return None
    diff = float(clock_start_t) - float(match_start_t)
    return {
        "clock_start_t": round(float(clock_start_t), 1),
        "match_start_t": round(float(match_start_t), 1),
        "diff_sec": round(diff, 1),
        "consistent": abs(diff) <= CROSS_CHECK_TOLERANCE,
        "note": (
            "时钟读数与 HUD 检测给出的对局起点一致"
            if abs(diff) <= CROSS_CHECK_TOLERANCE
            else (
                f"时钟读数给出的起点与 HUD 检测相差 {abs(diff):.0f} 秒 —— "
                f"以时钟为准（那是游戏自己的钟）；"
                f"HUD 检测在泉水那几秒会失效，起点判晚"
            )
        ),
    }
