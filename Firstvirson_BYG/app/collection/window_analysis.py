"""分析单元：把「时间片」换成「时间窗」。

为什么分析单元是时间窗，不是时间片
------------------------------------
用户说得很直白：

  「逐时间片标注不适合王者」
  「要在一个时间区间对帧做综合分析」

王者的一局是**连续的一整场**，没有「回合」这种天然边界。切出来的时间片
只是**证据**（哪几帧有事发生），不是**分析单元**。拿它当分析单元会出两个问题：

1. **问得多**：一局切 32 片，逐片标注就是 32 张表；
2. **判得碎**：同一条规则（「6 ~ 9 分钟该转线」）会被判 8 次 —— 碎片里
   任何一片没做到就记一条偏差，报告上变成「这局转线转错 8 次」，
   而玩家可能只做错过一次。

窗口从哪儿来
------------
**从规则库倒推**，不拍脑袋：每条规则的触发条件都写着
``time_min between [lo, hi]``，把这些窗合并去重，就是这张局天然的分析分段。

实测王者 S45 规则库（1681 条）只用了 **8 种窗**：

    0~0 / 0~1 / 1~1 / 2~3 / 4~4 / 6~9 / 10~10 / 12~60

合并后是 **6 个分析窗**：0~1、2~3、4、6~9、10、12 之后。
一局就分析 6 次，每次把窗内的帧**一起**交给识别器。

两条数据流
----------
- **窗** = 分析单元：事实、动作、判定、报告的结论都按窗给。
- **片** = 证据：窗里装着哪几片、每片的截图与视频秒数，
  报告里「看这一段」定位到窗内最贴切的那一片。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

#: 窗口的上界哨兵：表示「之后」（规则里写 60）。
END_OF_GAME = 60


@dataclass(frozen=True)
class Window:
    """一个分析窗：局内分钟区间 ``[lo, hi]``。"""

    lo: int
    hi: int

    @property
    def key(self) -> str:
        return f"{self.lo}-{self.hi}"

    def contains(self, minute: Optional[int]) -> bool:
        if minute is None:
            return False
        return self.lo <= minute <= self.hi

    def label(self) -> str:
        """窗口的中文标签（口径与标题一致）。

        - 单点 → 那个时点（「4:00」）
        - 一直到 60 分钟 → 「12:00 之后」
        - 其余 → 范围（「2:00 ~ 4:00」）
        """
        if self.lo == self.hi:
            return f"{self.lo}:00"
        if self.hi >= END_OF_GAME:
            return f"{self.lo}:00 之后"
        return f"{self.lo}:00 ~ {self.hi + 1}:00"

    def to_dict(self) -> Dict[str, Any]:
        return {"lo": self.lo, "hi": self.hi, "key": self.key,
                "label": self.label()}


def merge_windows(windows: Iterable[Tuple[int, int]],
                  adjacent: bool = True) -> List[Window]:
    """把重叠（或相邻）的窗合并成不重叠的区间（按起点排序）。

    ``adjacent=False`` 时**只合并重叠的**，相邻的（中间隔着一分钟）
    保持分开 —— 规则用不到的那几分钟是有意义的空档
    （王者 S45 的第 5 分钟和第 11 分钟一条规则都没有），
    把空档并进去会让分析单元凭空变大。
    """
    items = sorted({(int(lo), int(hi)) for lo, hi in windows if lo is not None})
    out: List[Tuple[int, int]] = []
    for lo, hi in items:
        if out and lo <= out[-1][1] + (1 if adjacent else 0):
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return [Window(lo, hi) for lo, hi in out]


def windows_from_conditions(conditions: Iterable[Dict[str, Any]],
                            merge: bool = True) -> List[Window]:
    """从规则触发条件里抽出时间窗。

    认 ``time_min`` 上的 ``between``（``[lo, hi]``）与 ``==``（``[v, v]``）。
    认不出来的窗一律不猜 —— 漏一个窗，那窗里的规则就永远不触发，
    而报告上只是「没结论」。

    ``merge=True`` 会把相邻或重叠的窗并成一段（分析段用）；
    ``merge=False`` 保留规则原本的窗（判定单元用）。
    """
    raw: List[Tuple[int, int]] = []
    for c in conditions or []:
        if not isinstance(c, dict):
            continue
        if c.get("fact") != "time_min":
            continue
        op = c.get("op")
        val = c.get("value")
        if op == "between" and isinstance(val, (list, tuple)) and len(val) == 2:
            raw.append((int(val[0]), int(val[1])))
        elif op == "==" and isinstance(val, (int, float)):
            raw.append((int(val), int(val)))
    if not merge:
        return [Window(lo, hi) for lo, hi in sorted(set(raw))]
    return merge_windows(raw)


def windows_from_rules(rules: Iterable[Any],
                       merge: bool = True) -> List[Window]:
    """从规则对象（或规则字典）里抽出分析窗。"""
    raw: List[Tuple[int, int]] = []
    for r in rules or []:
        conds = []
        if isinstance(r, dict):
            trig = r.get("trigger") or {}
            conds = (list(trig.get("all") or [])
                     + list(trig.get("any") or [])
                     + list(trig.get("none") or []))
        else:
            conds = (list(getattr(r, "all", []) or [])
                     + list(getattr(r, "any", []) or [])
                     + list(getattr(r, "none", []) or []))
        for w in windows_from_conditions(conds, merge=False):
            raw.append((w.lo, w.hi))
    if not merge:
        return [Window(lo, hi) for lo, hi in sorted(set(raw))]
    return merge_windows(raw)


def window_for_minute(windows: Sequence[Window],
                       minute: Optional[int]) -> Optional[Window]:
    """某个局内分钟落在哪个分析窗里。找不到返回 None（不硬塞）。"""
    for w in windows:
        if w.contains(minute):
            return w
    return None


def rule_windows(rules: Iterable[Any]) -> List[Window]:
    """规则用到的**全部**时间窗（去重，不合并）。

    这是**判定单元**：一条规则的窗就是一条规则的时间范围，
    窗内做一次综合分析，那条规则就有一次判定。

    王者 S45 实测 8 个：0~0 / 0~1 / 1~1 / 2~3 / 4~4 / 6~9 / 10~10 / 12~60。
    """
    return windows_from_rules(rules, merge=False)


def segments(windows: Sequence[Window]) -> List[Window]:
    """把判定单元合并成**分析段**（相邻或重叠的并成一段）。

    这是**分析单元**：识别器 / 玩家对着**一段**的帧做一次综合分析，
    段里的所有判定单元共用这份结论。

    王者 S45 实测 8 个判定单元合成 6 个分析段：
    0~1、2~3、4、6~9、10、12 之后。
    """
    return merge_windows(((w.lo, w.hi) for w in windows), adjacent=False)


def segment_for(windows: Sequence[Window],
                segments_: Sequence[Window]) -> Optional[Window]:
    """某个判定单元属于哪个分析段。"""
    for s in segments_:
        if s.contains(windows.lo) and s.contains(windows.hi):
            return s
    for s in segments_:
        if s.contains(windows.lo):
            return s
    return None


def assign_windows(minutes: Sequence[Optional[int]],
                    windows: Sequence[Window]) -> List[Optional[Window]]:
    """把一批局内分钟分别归到分析窗。"""
    return [window_for_minute(windows, m) for m in minutes]


def aggregate_facts(facts_list: Sequence[Dict[str, Any]]
                    ) -> Dict[str, Any]:
    """把窗内各片的事实合并成窗的事实。

    规则：**一致才采信，不一致就不给**。

    经济差、击杀差这类事实在窗内本来就会变（打了人就会变），
    取「最后一片」是编数字，取「窗内一致」才是真的。矛盾的事实直接**不给** ——
    给了就是拿一个假的局面去判规则，报告上却看不出来。
    """
    out: Dict[str, Any] = {}
    if not facts_list:
        return out
    keys = set()
    for f in facts_list:
        keys |= set(f.keys())
    for k in sorted(keys):
        values = [f[k] for f in facts_list if k in f and f[k] is not None]
        if not values:
            continue
        first = values[0]
        if all(v == first for v in values):
            out[k] = first
    return out


def aggregate_actions(actions_list: Sequence[Sequence[Any]]
                      ) -> List[Any]:
    """把窗内各片的动作并起来（按 ``(fact, value)`` 去重）。

    动作的口径是「这一段里做过就算做过」——
    攻略说的是「中期这一段必须转线」，不是「每一帧都得转」。
    """
    seen = set()
    out: List[Any] = []
    for acts in actions_list:
        for a in acts or []:
            key = (getattr(a, "fact", None) or
                   (a.get("fact") if isinstance(a, dict) else None),
                   str(getattr(a, "value", None) or
                       (a.get("value") if isinstance(a, dict) else None)))
            if key in seen:
                continue
            seen.add(key)
            out.append(a)
    return out


@dataclass
class WindowBundle:
    """一个分析窗连同它装着的时间片。"""

    window: Window
    #: 窗内的时间片（按局内分钟排好）
    slices: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def minutes(self) -> List[int]:
        return [int(s.get("game_minute") or 0) for s in self.slices]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lo": self.window.lo,
            "hi": self.window.hi,
            "key": self.window.key,
            "label": self.window.label(),
            "slices": [
                {
                    "round_no": s.get("round_no"),
                    "timestamp": s.get("timestamp"),
                    "game_minute": s.get("game_minute"),
                    "image": s.get("image", ""),
                    "image_url": s.get("image_url", ""),
                }
                for s in self.slices
            ],
        }


def bundle_slices(slices: Sequence[Dict[str, Any]],
                  windows: Sequence[Window]) -> List[WindowBundle]:
    """把时间片按分析窗打包。

    一片可以同时落在**好几个**窗里 —— 规则窗本来就可能重叠
    （王者 S45 有 0~0、0~1、1~1 三个窗都覆盖第 1 分钟）。
    这不是重复劳动：这几个窗判的是不同批规则，只是证据相同。

    落在所有窗之外的片（规则管不到的那几分钟）不打包 ——
    那些时间没有可判的模式，问了也是白问。
    """
    out: List[WindowBundle] = [WindowBundle(window=w) for w in windows]
    for s in slices or []:
        minute = s.get("game_minute")
        if minute is None:
            continue
        minute_i = int(minute)
        for b in out:
            if b.window.contains(minute_i):
                b.slices.append(s)
    return [b for b in out if b.slices]
