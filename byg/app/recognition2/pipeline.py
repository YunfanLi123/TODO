"""**整条链路**：视频 → 模型看视频 → 模型裁决 → 全局综述。

这是「上传一个视频」这条路径上**唯一**跑的识别流程。

链路
----
```
视频 ─▶ ① 打包（压到能上传的大小）  app/recognition2/videopack.py
     ─▶ ② 模型看整局录像            app/recognition2/watcher.py
     ─▶ ③ 定对局级事实（英雄/分路）  ← 模型自己讲的
     ─▶ ④ 选候选（模型自己检索）    app/recognition2/selector.py
     ─▶ ⑤ 逐条裁决（模型看录像片段）app/recognition2/adjudicator.py
     ─▶ ⑥ 全局综述（模型看整局的总结）app/recognition2/observer.py
```

**和旧链路的根本区别**
--------------------
旧链路是「切段 → 抽帧 → 逐段问模型」：模型看到的是一堆**静止的、
互不相干**的画面，而且每段都要单独调用一次。

现在模型**直接看录像** —— 连续画面，能看出「这一下是不是打到了」
「是不是交了位移」「是不是在被打」，而且它能把整局串起来讲。
这是「让模型真正看懂对局」的前提：**抽帧是把一局切碎再问**。

第 ④ 步也是同一件事的延续：**要对照哪几条攻略，由看过这一局的模型
决定**（它写检索请求，RAG 去取攻略原文），而不是由代码按字段
（作用域/英雄/时间窗）把整库筛一遍 —— 筛出来的尺子与这一局无关。

**旧链路不在这里** —— 旧系统的「几何探测 + 对照清单裁决」那套
（``app/collection/probes.py`` / ``vlm_recognizer.py`` / ``vlm_review.py``）
不参与这条路径：它的产物是「动作枚举」，而那套枚举只覆盖得到
模式库词表里的动作，18/70 段归类不到（详见
:mod:`app.recognition2.reader` 的提示词说明）。

产物的形状
----------
``run_pipeline`` 返回一个**纯字典**（能直接进 JSON）：

```python
{
  "timeline": {...},     # Beat 列表（统一真源）
  "verdicts": {...},     # 模型逐条裁决
  "overview": {...},     # 全局综述 + 攻略之外的问题
  "meta": {...},         # 调用数、tokens、覆盖率、用的哪个模型
  "notes": [...],        # 留痕（失败原因、没读出来的段）
}
```

转成报告层认识的形状用 :func:`app.recognition2.bridge.to_legacy_rounds`。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from . import hero_read, hp_bar, selector, videopack, watcher
from .adjudicator import AdjudicatorConfig, adjudicate
from .beat import Beat, Timeline
from .bridge import to_legacy_rounds
from .judge import MatchFacts
from .lane_evidence import POSITION_LABELS, judge_lane
from .observer import ObserverConfig, guide_digest, observe


@dataclass
class PipelineConfig:
    """整条链路的配置。"""

    #: 数据目录（模型配置与凭据在这）。
    data_dir: str = ""
    #: 证据帧存哪。报告的「看这一段」按这个目录里的文件名去取。
    frames_dir: str = ""
    #: 打包好的视频存哪（整局录像、裁决片段）。
    video_dir: str = ""
    #: 本局英雄与分路 —— **筛子**要用（分路不对的规则不算他的卡点）。
    hero: str = ""
    position: str = ""
    game: str = "hok"
    #: 画面归一：``{"rotation": 角度, "band": (x0, y0, x1, y1) 或 None}``。
    #: 上传的录屏可能横竖屏颠倒、上下有黑边 —— 不归一模型看到的是错的画面。
    view: Optional[Dict[str, Any]] = None
    #: 对局级事实（英雄 / 分路）的**兜底来源**。
    #:
    #: 它收到「对局在视频里的第几秒开始」，返回 ``{"hero": ..., "position": ...}``。
    #:
    #: 为什么是「兜底」而不是主来源
    #: --------------------------
    #: 新链路里**模型自己看录像**就会说「我玩的是谁、走哪路」——
    #: 它看的是连续画面，比认图标可靠。主来源是模型；模型说不出来时
    #: 才用这个兜底。**两个都没有**就不筛英雄层的规则
    #: （认不准比认错强：认错会拿别的英雄的打法去指责玩家）。
    resolve_facts: Optional[Callable[[Optional[float]], Dict[str, Any]]] = None
    #: 要不要跑「让模型看选人界面那一段」来认英雄（2026-11-03 新增）。
    hero_read: bool = True
    #: 看选人界面那一段时切多长（秒）。选人界面是静止的，一两分钟足够。
    hero_read_window: float = 150.0
    #: 一次裁决调用最多问几条规则。
    max_rules_per_call: int = 6
    #: 一次裁决调用最多给多少张画面（只有退回抽帧时才用得到）。
    max_frames_per_call: int = 10
    #: 打包视频的帧率（模型看片的采样率与它无关）。
    pack_fps: float = 2.0
    #: 模型看整局录像时的采样帧率（0.2~5）。
    watch_fps: float = 0.5
    #: 模型看整局录像时的细节档（low / default / high）。
    watch_detail: str = "default"
    #: 打包出来的体积上限。超过就走 Files 上传。
    max_upload_bytes: int = 480 * 1024 * 1024
    #: 允许把整局录像传到对方的文件服务（大视频只能走这条）。
    allow_upload: bool = True
    #: 裁决时的证据：``video``（看片段）或 ``frames``（抽帧）。
    evidence: str = "video"
    #: 攻略摘要给模型看多少条（用来找「攻略没写但画面里有的」）。
    guide_limit: int = 60
    #: 一次检索带回几条打法（**模型自己发起**的检索）。
    retrieval_top_k: int = 5
    #: 一局最多用几个检索请求。
    max_retrieval_requests: int = 8
    #: 候选最多留几条（再多的按检索顺序截断，**并写进留痕**）。
    max_candidates: int = 18
    #: 要不要跑全局综述（要花一次调用）。
    observe: bool = True
    #: 要不要**逐次死亡复盘**（每死一次花一次调用：判这波赚还是亏）。
    #:
    #: 用户的要求（2026-11-13）：「分析对局内有益还是有害」。
    #: 关掉它，正文只剩「怎么死的」，读者看不出这次到底亏不亏。
    coach_deaths: bool = True
    #: 要不要**读顶栏的 K/D/A**（把「死没死」变成事实）。
    #:
    #: 判「赚还是亏」要有事实：死前死后我方有没有拿到人头。
    #: 「读数字」是模型的事，「算增量、推事件」是代码的事
    #: （见 :mod:`app.recognition2.kda_read`）。
    kda_read: bool = True
    #: 要不要**整局复盘**（错在哪 + 教官分析，报告正文前两部分）。
    #:
    #: 用户 2026-11-13 的要求：「第一部分总结指出玩家错在哪了，第二部分
    #: 让模型当作教官给出分析，**不要再考虑命中规则的事情**」。
    coach_match: bool = True
    #: 要不要**把规则前提交给模型判**（2026-11-14 用户要求：
    #: 对局势的理解和对规则的理解都归模型）。
    #:
    #: 打开后，代码不再用「金差/压迫/对位」这些粗字段替模型判「适不适用」，
    #: 只把对不上的前提项当线索交出去，由模型看着画面裁决。
    model_premise: bool = True
    #: 要不要**让模型判对线优劣**（2026-11-14 用户要求：不要对位表了）。
    #:
    #: 判的是「这一局打出来的对线」—— 被压塔下、血量常低、被单杀就是劣势；
    #: 判不出来就 unknown，带这个前提的规则按「不适用」处理。
    matchup_judge: bool = True
    #: 要不要读血条（像素法读全程曲线 + 模型复核）。
    hp_read: bool = True
    #: 血条要不要让模型复核（只在残血时刻裁剪放大问一次）。
    hp_confirm: bool = True
    #: 一次最多让模型复核几段残血（一次调用问完，不按段计费）。
    hp_confirm_shots: int = 4
    #: 最多给几段残血检索对照打法（每段一次检索）。
    hp_guide_shots: int = 3
    #: 进度回调：``(阶段说明, 0~1 的进度)``。
    #:
    #: 读一局要好几分钟；不报进度的话前台进度条会一直停在原地 ——
    #: 那比「慢」更让人以为程序死了。进度按**已完成的阶段**算，是真的。
    on_progress: Optional[Callable[[str, float], None]] = None
    notes: List[str] = field(default_factory=list)

def make_transform(view: Optional[Dict[str, Any]]):
    """按 ``view`` 造一个画面归一函数（转正 + 裁黑边）。

    复用旧系统的**标定**代码（``orient_frame`` / ``crop_to_band``）——
    那是几何换算，不是旧的识别逻辑。转正与裁黑边的口径必须两边一致，
    不然同一个视频 v1 读到的是正着的画面、v2 读到的是歪的。

    新链路里这个函数用在**重编码**上 —— 转正要**烙进视频里**，
    整段录像才会是正着的。
    """
    if not view:
        return None
    rotation = int(view.get("rotation") or 0)
    band = view.get("band")
    if not rotation and not band:
        return None

    def _transform(frame):
        from app.collection.video_upload import crop_to_band, orient_frame
        import cv2

        if rotation:
            frame = orient_frame(cv2, frame, rotation)
        if band:
            frame = crop_to_band(frame, band)
        return frame

    return _transform


def _load_patterns(game: str):
    from app.rules.patterns import load_patterns

    return load_patterns(game, verify_source=False)


#: 筛子把规则筛掉的理由 -> **归类名**（报告里按这个分类计数）。
#:
#: 为什么要分类：只说「筛掉了 1 400 条」没用 —— 读者要的是
#: 「哪一步没对上」（分路？英雄？时间窗？），那才是能拿去改的。
_SCREEN_BUCKETS = (
    ("没有带局内时刻的段", "这一局没有局内时刻"),
    ("已退役", "已退役"),
    ("这条是给", "别的分路的规则"),
    ("只针对", "别的英雄的规则"),
    ("没认出本局玩的是谁", "没认出英雄"),
    ("排除名单", "护栏排除（这条有英雄专属版）"),
    ("规则窗口是第", "时间窗没交上"),
    ("前提不成立", "前提不成立"),
)


def _screen_buckets(screened: Iterable[Any]) -> Dict[str, int]:
    """筛子把规则一条条筛掉了 —— **按是哪一步**归类计数。

    返回 ``{分类: 条数}``。分类顺序按「先大后小」排（调用方自己排），
    这里只负责数。

    认不出来的理由归到「其它」—— 宁可有一类说不清，
    也不要假装每条都归好了（那样这份诊断就不可信了）。
    """
    buckets: Dict[str, int] = {}
    for j in screened or []:
        reason = str(getattr(j, "reason", "") or "")
        key = "其它"
        for needle, name in _SCREEN_BUCKETS:
            if needle in reason:
                key = name
                break
        buckets[key] = buckets.get(key, 0) + 1
    return buckets


#: 筛掉理由里，**被否掉的是哪个值** —— 拿它才写得清「库里要的是哪条路」。
_SCREEN_RE_LANE = re.compile(r"这条是给「(.+?)」的")
_SCREEN_RE_HERO = re.compile(r"这条只针对 (.+?)，本局玩的")


def _rule_digests(candidates: Sequence[Any], limit: int = 8) -> List[str]:
    """候选规则 → 给整局复盘参照的**打法要点**（短、不带 id）。

    报告第二段「该怎么做」要让模型**参照规则描述 + 触发前提的对局描述**
    自己润色，所以这里只给「情形 → 该做」这两句，不给整张卡。
    """
    out: List[str] = []
    for cand in list(candidates or [])[:limit]:
        try:
            from ..rules import cards

            card = cards.rule_card(cand.pattern)
            when = card.get("situation") or ""
            do = card.get("should") or ""
            if not do and card.get("branches"):
                do = card["branches"][0].get("do") or ""
                when = when or card["branches"][0].get("when") or ""
            text = ("%s → %s" % (when, do)).strip(" →") if (when or do) else ""
        except Exception:                              # noqa: BLE001
            text = ""
        if text:
            out.append(text)
    return out


def _situation_brief(meta: Dict[str, Any]) -> str:
    """把前面几步的结论拼成一段**局势摘要**，给裁决层当上下文。

    用户定的顺序（2026-11-14）：逐段叙述 → 对线优劣 / 死亡复盘 →
    **再综合这些裁决规则** → 三段式报告。所以裁决拿到的不能只有画面，
    还要有「这一局对线打成什么样、死过几次各是赚是亏」。
    """
    lines: List[str] = []
    matchup = (meta or {}).get("matchup") or {}
    label = str(matchup.get("label") or "")
    if label and label != "看不出来":
        why = str(matchup.get("why") or "").strip()
        lines.append("· 对线：%s%s" % (label, ("（%s）" % why) if why else ""))
    deaths = ((meta or {}).get("coach") or {}).get("deaths") or []
    if deaths:
        parts = []
        for item in deaths:
            trade = str(item.get("trade") or "unknown")
            name = {"beneficial": "不亏", "harmful": "亏了",
                    "even": "打平", "unknown": "看不出来"}.get(trade, trade)
            parts.append("%s %s" % (item.get("clock") or "?", name))
        lines.append("· 死亡：%s" % "；".join(parts))
    return "\n".join(lines)


def _screen_details(screened: Iterable[Any]) -> Dict[str, Dict[str, int]]:
    """筛子否掉每条规则时，**被否掉的是哪个值** —— 按值归类计数。

    为什么要这一层：「别的分路的规则 6 条」只说了数目，
    读者还是不知道库里要的是哪条路。把被否掉的值摆出来，
    才谈上「该改分路还是该改英雄」。
    """
    out: Dict[str, Dict[str, int]] = {}
    for j in screened or []:
        reason = str(getattr(j, "reason", "") or "")
        key = "其它"
        for needle, name in _SCREEN_BUCKETS:
            if needle in reason:
                key = name
                break
        value = ""
        m = _SCREEN_RE_LANE.search(reason)
        if m:
            value = POSITION_LABELS.get(m.group(1), m.group(1))
        else:
            m = _SCREEN_RE_HERO.search(reason)
            if m:
                value = m.group(1)
        if not value:
            continue
        slot = out.setdefault(key, {})
        slot[value] = slot.get(value, 0) + 1
    return out


def _detail_suffix(detail: Optional[Dict[str, int]]) -> str:
    """「（库里要的是「游走」）」—— 有话说才加，没话不加。""" 
    if not detail:
        return ""
    top = sorted(detail.items(), key=lambda kv: -kv[1])[:3]
    return "（" + "、".join("%s %d 条" % (v, n) for v, n in top) + "）"


def _bp_labels() -> Tuple[Any, Any]:
    """BP 环节与操作类别的中文标签（拿不到就返回空表）。"""
    try:
        from app.adapters.honor_of_kings import BP_ACTION_LABELS, BP_PHASE_LABELS

        return BP_PHASE_LABELS, BP_ACTION_LABELS
    except Exception:                                   # noqa: BLE001
        return {}, {}


def _bp_decisions(bp: Any) -> List[Dict[str, Any]]:
    """BP 记录里的决策列表（形状对不上就当没有）。"""
    if not isinstance(bp, dict):
        return []
    decisions = bp.get("decisions")
    if not isinstance(decisions, list):
        return []
    return [d for d in decisions if isinstance(d, dict)]


def _is_ban(decision: Dict[str, Any]) -> bool:
    """这一手是**禁用**吗。

    环节标签最靠得住（「禁用（阶段一）」）；操作类别当二手线索
    （「禁掉坦边」这类）。
    """
    phase = str(decision.get("phase") or "")
    action = str(decision.get("action") or "")
    if "禁用" in phase:
        return True
    return "禁" in action


def _is_pick(decision: Dict[str, Any]) -> bool:
    """这一手是**选人**吗。"""
    phase = str(decision.get("phase") or "")
    action = str(decision.get("action") or "")
    if "禁用" in phase:
        return False
    phase_labels, action_labels = _bp_labels()
    pick_phases = {v for k, v in phase_labels.items() if str(k).startswith("pick")}
    pick_actions = {v for k, v in action_labels.items() if str(k).startswith("pick")}
    if phase and phase in pick_phases:
        return True
    if action and action in pick_actions:
        return True
    return False


def _banned_heroes(bp: Any) -> Set[str]:
    """这一局**被禁用**的英雄（双方都算）。

    用来否掉「物理上不可能」的读数 —— 2026-11-03 实测：
    看完整局的模型说了一个**我方禁用**的英雄。
    """
    out: Set[str] = set()
    for d in _bp_decisions(bp):
        if not _is_ban(d):
            continue
        hero = str(d.get("hero") or "").strip()
        if hero:
            out.add(hero)
    return out


def _my_pick_pool(bp: Any) -> Set[str]:
    """我方**选**过的英雄。

    空集 = BP 记录里没有选人记录（那就拿它当没有约束，别误杀读数）。
    """
    out: Set[str] = set()
    for d in _bp_decisions(bp):
        if str(d.get("side") or "my").strip() != "my":
            continue
        if not _is_pick(d):
            continue
        hero = str(d.get("hero") or "").strip()
        if hero:
            out.add(hero)
    return out


def _bp_last_pick(bp: Any) -> str:
    """我方**最后**选的那个英雄。

    「最后」= 记录里我方选人环节的**最后一个**；
    有「阶段二末选」就取它（那一手就是本人的）。
    """
    picks: List[Dict[str, Any]] = []
    for d in _bp_decisions(bp):
        if str(d.get("side") or "my").strip() != "my":
            continue
        if not _is_pick(d):
            continue
        hero = str(d.get("hero") or "").strip()
        if hero:
            picks.append(d)
    if not picks:
        return ""
    for d in reversed(picks):
        if "末选" in str(d.get("phase") or ""):
            return str(d.get("hero") or "").strip()
    return str(picks[-1].get("hero") or "").strip()


def _bp_is_model_guess(bp: Any) -> bool:
    """这份 BP 记录是不是**模型看 BP 帧认出来的**（不是玩家自己录的）。

    为什么这很重要
    --------------
    ``prefill.read_bp`` 让模型看 BP 阶段的帧，直接吐「谁禁了谁、谁选了谁」。
    可**画面上没印着英雄名** —— 选人界面只印昵称、分路与「预选」状态，
    模型只能凭头像认。本项目实测：凭美术认英雄，10 次没有一次对
    （README §32）。所以那份记录是**猜测**，不是「画面上印着的字」。

    老记录（2026-11-13 之前落的）没有这个标记，按**玩家自己录的**对待 ——
    那时确实是玩家在 BP 记录页上手填的。
    """
    if not isinstance(bp, dict) or not bp:
        return False
    if str(bp.get("source") or "").strip() == "model":
        return True
    for d in _bp_decisions(bp):
        if str(d.get("source") or "").strip() == "model":
            return True
    return False


def _hero_official_lanes(hero: str) -> Set[str]:
    """这个英雄的**官方分路**（可能不止一条；查不到就是空集）。

    具体口径在适配器里（:func:`app.adapters.honor_of_kings.official_lanes_of`）——
    报告层也要用同一份，两处各写一遍迟早漂。
    """
    name = (hero or "").strip()
    if not name:
        return set()
    try:
        from app.adapters.honor_of_kings import official_lanes_of

        return set(official_lanes_of(name))
    except Exception:                                   # noqa: BLE001
        return set()


def _position_from_hero(hero: str) -> str:
    """按英雄查分路（**推断**，不是读数）。

    只有「唯一对应一条分路」的英雄查得到；查不到就空 ——
    猜错位置会让整局每一条带位置的规则都判错，而那种错误在报告里看不出来。

    为什么在这里再算一遍：图标那条路的分路是**按它认出的那个英雄**推断的。
    英雄既然以模型的为准，这份推断就得跟着换成模型那个英雄。
    """
    name = (hero or "").strip()
    if not name:
        return ""
    try:
        from app.adapters.honor_of_kings import infer_position_from_hero

        return infer_position_from_hero(name)
    except Exception:                                   # noqa: BLE001
        return ""

def _mmss(seconds: float) -> str:
    """秒 -> ``MM:SS``。"""
    total = max(0, int(round(float(seconds or 0.0))))
    return "%d:%02d" % (total // 60, total % 60)


def _guide_at_death(game: str, lane: str, hero: str,
                    death: Dict[str, Any]) -> Dict[str, Any]:
    """查这次死亡那一刻的攻略（失败局复盘要「下次怎么做」）。

    **直接检索攻略语料** —— 不走规则库的判定层与模式筛选
    （那一套判不了死因）。

    **只取模式库的原话**：检索不到就返回空，报告层写「没查到」；
    检索失败（接口不通、索引打不开）同样返回空 ——
    **静默失败必须堵住**，但这里的失败只是「这一段没有建议」，
    不能拖垮整局。
    """
    # **没有局内时刻就不检索** —— 那时的时间窗只能塌到 0~2 分钟，
    # 捞回来的那条攻略跟这次死亡根本无关，却会被当成它的「下次怎么做」
    # 摆在报告里（第 15 分钟的死，配一句「开局怎么打」）。
    # 残血那一层早就有同一道闸（见 :func:`_guide_at_hp`），这里补上。
    if death.get("minute") is None:
        return {}
    query = " ".join(x for x in (str(death.get("cause") or "").strip(),
                                 str(death.get("where") or "").strip()) if x)
    if not query:
        return {}
    minute = float(death.get("minute") or 0.0)
    lo = max(0, int(minute) - 1)
    hi = max(lo + 1, int(minute) + 2)
    try:
        from ..rag.context import ModeSearchService

        service = ModeSearchService(game=game)
        result = service.search(query, lane=lane, hero=hero,
                                 lo=lo, hi=hi, top_k=3)
        entries = service.entries(result)
    except Exception:                                   # noqa: BLE001
        return {}
    if not entries:
        return {}
    best = entries[0] or {}
    return {
        "id": str(best.get("id") or ""),
        "title": str(best.get("title") or ""),
        "premise": str(best.get("premise") or ""),
        "action": str(best.get("action") or ""),
        "outcome": str(best.get("outcome") or ""),
        "advice": str(best.get("advice") or ""),
    }


def _guide_at_hp(game: str, lane: str, hero: str,
                 episode: Dict[str, Any]) -> Dict[str, Any]:
    """查这段残血该对照的打法（**模式库原文**）。

    和死亡复盘同一套口径：**只取模式库的原话** —— 检索不到就空，
    报告层写「没查到」；检索失败同样返回空，但**不能拖垮整局**。

    **没有局内时刻就不检索** —— 那时的时间窗只能塌到 0~2 分钟，
    捞回来的那条攻略跟这段残血根本无关，却会被当成它的建议展示。
    """
    if episode.get("minute") is None:
        return {}
    query = "残血 血量少 被追 打不赢 保守 撤退 回城"
    minute = float(episode.get("minute") or 0.0)
    lo = max(0, int(minute) - 1)
    hi = max(lo + 1, int(minute) + 2)
    try:
        from ..rag.context import ModeSearchService

        service = ModeSearchService(game=game)
        result = service.search(query, lane=lane, hero=hero,
                                 lo=lo, hi=hi, top_k=3)
        entries = service.entries(result)
    except Exception:                                   # noqa: BLE001
        return {}
    if not entries:
        return {}
    best = entries[0] or {}
    return {
        "id": str(best.get("id") or ""),
        "title": str(best.get("title") or ""),
        "premise": str(best.get("premise") or ""),
        "action": str(best.get("action") or ""),
        "outcome": str(best.get("outcome") or ""),
        "advice": str(best.get("advice") or ""),
    }


def match_start_t(timeline: Timeline) -> Optional[float]:
    """对局在**原视频**里从第几秒开始（**兜底**）。

    模型看录像时会直接说对局从第几秒开始；那是主来源。
    这个函数是**旧办法**（按局内时刻反推），只在模型说不出来时用。
    """
    fit = (timeline.meta or {}).get("clock_fit") or {}
    start = fit.get("start_t")
    if start is not None:
        try:
            return float(start)
        except (TypeError, ValueError):
            pass
    diffs: List[float] = []
    for beat in timeline.beats:
        if beat.game_time is None or beat.game_time < 0:
            continue
        diffs.append(float(beat.video_time) - float(beat.game_time))
    if not diffs:
        return None
    diffs.sort()
    mid = len(diffs) // 2
    if len(diffs) % 2:
        return diffs[mid]
    return (diffs[mid - 1] + diffs[mid]) / 2.0

def timeline_from_watch(watch: "watcher.WatchResult",
                        game: str = "hok",
                        video_path: str = "",
                        frames_dir: str = "",
                        transform=None) -> Timeline:
    """把「模型看整局录像」的结果落成时间线。

    **局内时刻怎么来的**
    ------------------
    模型看的是录像，报的是**原视频第几秒**。局内时刻 = 原视频秒 −
    对局起点。对局起点是模型说的（或者由它读到的顶栏时刻反推）。

    对局起点读不出来 → 局内时刻就是 ``-1``（**不猜**）→ 带时间窗的
    规则一条都判不了，这件事会写进留痕。

    **代表帧**
    ----------
    报告上每一段时间要有一张能回看的图 —— 那是**给人看的缩略图**，
    不是给模型判读的（模型看的是录像）。所以在这里从原视频抽一张，
    挂到对应的那一段上（``Beat.frames`` 与 ``detail["frame_times"]``
    必须一一对应）。
    """
    timeline = Timeline(game=game)
    start = watch.match_start
    mids: List[float] = []
    for index, event in enumerate(watch.events):
        mids.append((float(event.video_start) + float(event.video_end)) / 2.0)
        game_time = -1.0
        if start is not None and event.video_start >= start - 1.0:
            game_time = max(0.0, float(event.video_start) - float(start))
        game_end = game_time
        if start is not None and event.video_end >= start - 1.0:
            game_end = max(0.0, float(event.video_end) - float(start))
        beat = Beat(
            game_time=game_time,
            game_time_start=game_time,
            game_time_end=game_end,
            video_time=float(event.video_start),
            video_time_start=float(event.video_start),
            video_time_end=float(event.video_end),
            anchor_frame=0,
            action=event.action or "unclear",
            action_label=event.action_label or "",
            narrative=event.narrative or "",
            confidence=float(event.confidence or 0.0),
            detail={
                "position": event.position or "unknown",
                "where": event.where or "",
                "phase": event.phase or "unknown",
                "video_span": [round(float(event.video_start), 2),
                               round(float(event.video_end), 2)],
                "event_index": index,
            },
            why_not_read=("" if (event.narrative or "").strip()
                          else "模型没讲这一段"),
        )
        timeline.add(beat)

    # -- 代表帧（报告里的缩略图）--
    if frames_dir and video_path and timeline.beats:
        try:
            thumbs = videopack.save_thumbnails(
                video_path, mids, frames_dir, prefix="ev", index=0,
                transform=transform)
        except Exception:                              # noqa: BLE001
            thumbs = []
        # **按位置配对，但存不下来的那个位置要空着** ——
        # 缩略图函数返回的列表与时刻一一对应（存不下来就是 None），
        # 跳过会让后面每一段都拿到**前一段的图**
        # （报告「看这一段」跳到错误的秒数，证据与结论对不上）。
        for beat, thumb in zip(timeline.beats, thumbs):
            if not thumb:
                continue
            name, actual = thumb
            beat.frames.append(name)
            detail = beat.detail or {}
            detail["frame_times"] = [round(float(actual), 2)]
            beat.detail = detail
    timeline.meta.update({
        "video": os.path.basename(watch.pack.get("path") or ""),
        "segments": len(timeline.beats),
        "match_start": (round(float(watch.match_start), 2)
                        if watch.match_start is not None else None),
        "watch_summary": watch.summary or "",
        "hero": watch.hero or "",
        "position": watch.position or "",
    })
    timeline.notes.extend(watch.notes or [])
    return timeline

def _model_takes_video(cfg: "PipelineConfig") -> Optional[bool]:
    """「观察者模型」收不收**视频块**（``None`` = 认不出来）。

    认不出来的情形很实际：用户选「自定义」、填了个我们没见过的地址。
    那时不该拦他 —— 按老路发一次，报错再说。
    """
    try:
        from app.collection import model_presets, settings

        with settings.use_data_dir(cfg.data_dir or None):
            slot = settings.effective_slot("vlm")
        takes = model_presets.video_support(str(slot.get("provider") or ""),
                                              str(slot.get("endpoint") or ""))
        # 本机编不出 H.264 时，视频块会被模型拒收 —— 改走抽帧。
        if takes is True:
            from . import videopack
            if not videopack.h264_available():
                return False
        return takes
    except Exception:                                   # noqa: BLE001
        return None


def run_pipeline(video_path: str,
                 config: Optional[PipelineConfig] = None) -> Dict[str, Any]:
    """跑完整条链路，返回可落库的产物。

    **任何一步失败都要留痕** —— 报告里宁可写「这一局没分析」，
    也不能给一张空表让人以为「打得没问题」（见
    :func:`app.collection.vlm_client.explain_error`）。
    """
    cfg = config or PipelineConfig()
    notes: List[str] = list(cfg.notes)
    meta: Dict[str, Any] = {
        "video_path": video_path,
        "watch_calls": 0,
        "watch_tokens": 0,
        "judge_calls": 0,
        "judge_tokens": 0,
        "observe_calls": 0,
        "observe_tokens": 0,
    }

    # -- ① 打包 --
    #
    # **模型不收视频时不打包**（2026-11-13）：打包是为了把录像塞进视频块，
    # 而收不了视频的模型（DeepSeek 这类 OpenAI 兼容接口）改走**抽帧** ——
    # 那条路直接从原片取帧，打包这一步纯属白等（实测 749 MB 的片子要几分钟）。
    transform = make_transform(cfg.view)
    takes_video = _model_takes_video(cfg)
    meta["watch_mode"] = "video" if takes_video is not False else "frames"
    if takes_video is False:
        if cfg.on_progress is not None:
            cfg.on_progress("正在准备抽帧", 0.05)
        info = videopack.probe(video_path)
        pack = videopack.PackResult(
            path="", mode="skip", codec=info.codec,
            duration=float(info.duration or 0.0),
            source_duration=float(info.duration or 0.0),
            size=int(info.size or 0),
            notes=["模型不收视频 —— 没打包（抽帧那条路直接从原片取帧）。"])
    else:
        if cfg.on_progress is not None:
            cfg.on_progress("正在把录像压成能上传的形态", 0.05)
        pack = videopack.pack_video(video_path, cfg.video_dir,
                                    max_bytes=cfg.max_upload_bytes,
                                    fps=cfg.pack_fps, transform=transform,
                                    stem="match")
    meta["pack"] = pack.to_dict()
    notes.extend(pack.notes or [])

    # -- ② 模型看整局录像 --
    if cfg.on_progress is not None:
        cfg.on_progress("正在让模型看整局录像", 0.20)
    watch = watcher.watch_video(
        video_path,
        watcher.WatchConfig(
            data_dir=cfg.data_dir,
            video_dir=cfg.video_dir,
            frames_dir=cfg.frames_dir,
            fps=cfg.watch_fps,
            detail=cfg.watch_detail,
            max_upload_bytes=cfg.max_upload_bytes,
            allow_upload=cfg.allow_upload,
            pack_fps=cfg.pack_fps,
            hero=cfg.hero,
            position=cfg.position,
        ),
        transform=transform, pack=pack)
    meta["watch_calls"] = watch.calls
    meta["watch_tokens"] = watch.tokens
    notes.extend(watch.notes or [])
    meta["uploaded"] = bool(watch.uploaded)

    timeline = timeline_from_watch(watch, cfg.game,
                                  video_path=video_path,
                                  frames_dir=cfg.frames_dir,
                                  transform=transform)
    meta["segments"] = len(timeline.beats)
    meta["coverage"] = timeline.coverage()

    # -- ③ 对局级事实（英雄 / 分路） --------------------------------
    #
    # **谁说了算** —— 三个读数，按「证据有多硬」排队（2026-11-03）：
    #
    # 1. **看选人界面那一段**（模型读画面上**印着的字**）；
    # 2. **BP 记录里我方最后选的那个**（也是画面上的字，但那是
    #    「这一手选了什么」的明确记录）；
    # 3. **看完整局那个模型**。
    #
    # （原来还有两条：**专门认技能按钮那一问**与**局内技能按钮的像素比对** ——
    # 2026-11-03 随旧像素链一起退役了。它们给的名字全是「猜的」：
    # 冷却倒计时数字叠在图标上，美术被压掉一大半；前两名只差 0.0119，
    # 纯噪声；而且分不清「玩家自己」和「队友」。）
    #
    # 为什么加一条「**禁用英雄一律作废**」
    # --------------------------------------
    # 2026-11-03 实测（那局）：看完整局的模型说「马超」——
    # 而 **马超是这一局我方禁用的英雄**（BP 记录：「禁用（阶段一）马超」）。
    # 被禁的英雄不可能被选上，那句话**物理上不可能**。
    # 以前没人校验这一层，于是报告拿「马超」的打法去指责玩家，
    # 而玩家一眼就看出那不是自己。
    #
    # 为什么还有一条「**不在我方选人名单里的也作废**」
    # ----------------------------------------------
    # 同一次实测：旧像素链认成「敖隐」—— 而敖隐是**队友**（紅绳）的英雄，
    # 不是玩家自己的。名单能挡住这种「认成别人」的错。
    model_hero = (watch.hero or "").strip()
    position = (watch.position or "").strip()
    if not position or position == "unknown":
        position = cfg.position
    start_t = watch.match_start
    meta["match_start_t"] = round(start_t, 1) if start_t is not None else None

    facts: Dict[str, Any] = {}
    if cfg.resolve_facts is not None:
        try:
            facts = cfg.resolve_facts(start_t) or {}
        except Exception:                              # noqa: BLE001
            facts = {}
    if facts:
        meta["facts"] = dict(facts)

    # -- 胜负：两处读数对账 -------------------------------------------
    #
    # 「看整局」那次看完了整局（**含结算画面**）；「读最后几帧」那次
    # 只抽了录屏末尾几帧 —— 而结算画面出现后，录屏往往还录了十几秒
    # （玩家点掉结算、回到主界面），那几帧正好落在结算画面**之前**。
    #
    # 于是会出现「叙述里已经写着『水晶爆裂并出现失败画面』，
    # 结果栏却空着」（2026-11-07 实测：敖隐那局输给了敌方水晶）。
    #
    # 口径：**以看整局那次为准**（它看的是整局），两处对不上要说清楚。
    watch_result = str(getattr(watch, "result", "") or "unknown").strip().lower()
    prefill_result = str((facts or {}).get("result") or "unknown").strip().lower()
    result = ""
    if watch_result in ("win", "lose"):
        result = watch_result
        if prefill_result in ("win", "lose") and prefill_result != watch_result:
            notes.append(
                "这一局的胜负两处读数对不上（看整局说%s、读末帧说%s）"
                "—— 以「看整局」为准。" % (watch_result, prefill_result))
        elif prefill_result not in ("win", "lose"):
            notes.append(
                "这一局的胜负按「看整局」那次读的（%s）—— "
                "读最后几帧那次没落在结算画面上。" % result)
    elif prefill_result in ("win", "lose"):
        result = prefill_result
    meta["result"] = result
    meta["result_source"] = (
        "watch" if result and result == watch_result
        else ("prefill" if result else ""))

    # -- 认英雄：让模型看选人界面那一段（读画面上印着的字）--------------
    clip = hero_read.HeroRead()
    # 玩家自己填了英雄 —— 不用再花一次调用去猜（也避免猜错盖过真值）。
    if cfg.hero_read and not cfg.hero:
        try:
            clip = hero_read.read_hero(
                video_path, cfg.video_dir, start_t=start_t,
                duration_sec=float(pack.source_duration or 0.0),
                window_sec=cfg.hero_read_window,
                data_dir=cfg.data_dir, transform=transform)
        except Exception as exc:                       # noqa: BLE001
            notes.append("切选人界面那一段失败：%s" % exc)
        meta["hero_clip"] = clip.to_dict()
        notes.extend(clip.notes or [])

    # -- 几个读数 + 两道校验 ---------------------------------------------
    #
    # **读数的强弱**（2026-11-13 重排过一次，原因见下）
    # ---------------------------------------------
    # 1. ``clip`` —— 看选人界面那一段，读**画面上印着的字**（最硬）；
    # 2. ``watch`` —— 看完整局的模型（它看得见技能区、战绩面板、你的英雄）；
    # 3. ``bp`` —— 从 BP 记录推的，**最弱**。
    #
    # 为什么 ``bp`` 从第二降到第三：那份记录不是「画面上印着的字」，
    # 而是 **``prefill.read_bp`` 让模型看 BP 帧认出来的头像**。
    # 2026-11-13 核对了那一局的选人帧（``hok-video-8178dc031675``）：
    # 选人界面上只印**昵称、分路与「预选」状态**，**不印英雄名** ——
    # 模型只能靠美术认，而本项目实测「凭美术认英雄 10 次没有一次对」。
    #
    # 2026-11-13 那一局就是这么翻的：clip 没读出名字、watch 没答英雄，
    # 只剩 BP 那个猜测，于是它成了「本局英雄」—— 报的是公孙离（发育路），
    # 而画面上玩家一直在中路，真值是**敖隐**（局内聊天与战绩面板上都印着）。
    # 结果英雄层按公孙离（发育路）筛、分路是中路 → 一条都对不上 →
    # 候选 0 条 → 整张报告是空的，玩家一眼就看出英雄不是他。
    #
    # 所以现在：**模型读出来的 BP 记录不再自动算「本局英雄」**，
    # 它只在**有旁证**（别的读数说的是同一个名字）时才定案，
    # 没有旁证就按「没认出来」处理（认不准比认错强）。
    # 玩家自己录的 BP 记录（没有 ``source`` 标记）仍按老口径 —— 那是他自己填的。
    bp = facts.get("bp") or {}
    bp_guess = _bp_is_model_guess(bp)
    banned = _banned_heroes(bp)
    my_pool = _my_pick_pool(bp)
    bp_hero = _bp_last_pick(bp)
    #: 这一局看的是**抽帧**还是整局视频 —— 抽帧看到的英雄读数要旁证。
    frames_mode = (meta.get("watch_mode") == "frames")

    readings: List[Tuple[str, str]] = []
    if cfg.hero:
        # **玩家自己填的排第一**（2026-11-14）：他就在玩这一局，比模型猜得准。
        readings.append((str(cfg.hero).strip(), "user"))
    readings += [
        (clip.hero, "clip"),          # 看选人界面那一段（读画面上的字）
        (model_hero, "watch"),        # 看完整局那个模型
        (bp_hero, "bp"),              # BP 记录里我方最后选的那个（模型认的头像）
    ]
    voided: List[Dict[str, str]] = []
    #: **只是提醒、不作废**的读数（BP 记录是模型认的时，它没资格否掉别人）。
    soft: List[Dict[str, str]] = []
    legal: List[Tuple[str, str]] = []
    for name, source in readings:
        if not name:
            continue
        if source == "user":
            # 玩家自己填的不做「禁用/名单」校验：那份 BP 记录本身可能是模型猜的。
            legal.append((name, source))
            continue
        if name in banned:
            item = {"hero": name, "source": source, "why": "这一局被禁用了"}
            if bp_guess:
                # 禁用名单也是模型看 BP 帧认的 —— 拿它否掉别处读出来的名字，
                # 等于让一个猜测去否决另一个读数。先只提醒。
                soft.append(dict(item, why="BP 记录里说这一局禁了它"
                                            "（那是模型看头像读的，不据此作废）"))
                legal.append((name, source))
                continue
            voided.append(item)
            continue
        if my_pool and name not in my_pool:
            item = {"hero": name, "source": source,
                    "why": "不在我方选的人里面"}
            if bp_guess:
                soft.append(dict(item, why="BP 记录里没有它"
                                            "（那是模型看头像读的，不据此作废）"))
                legal.append((name, source))
                continue
            voided.append(item)
            continue
        legal.append((name, source))

    hero = ""
    hero_source = ""
    if legal:
        name, source = legal[0]
        if source == "bp" and bp_guess:
            # **只剩 BP 那个猜测** —— 没有旁证，不作数。
            voided.append({"hero": name, "source": "bp",
                           "why": "BP 记录是模型看头像认的，没有旁证"})
        elif (source == "watch" and frames_mode
              and not any(n == name for n, s in legal if s != "watch")):
            # **抽帧那条路上的英雄读数要旁证**（2026-11-13）。
            #
            # 抽帧只给几十张静图，模型看到战绩面板时**很容易把别的行
            # 当成玩家自己**（实测那一批：它读了第一行「吕布」，而红底
            # 那一行是「敖隐」）。认错一个英雄，英雄层整套打法都换成
            # 别人的，报告上还看不出来 —— 按「认不准比认错强」处理。
            voided.append({"hero": name, "source": "watch",
                           "why": "抽帧看到的英雄没有旁证（模型可能读错了"
                                  "战绩面板里的那一行）"})
        else:
            hero, hero_source = name, source
    if not hero and cfg.hero:
        hero, hero_source = str(cfg.hero).strip(), "config"
    if voided:
        meta["hero_voided"] = voided
        for v in voided:
            notes.append("认英雄这一步有个读数被否掉（%s，%s）—— %s。"
                         % (v.get("hero") or "", v.get("source") or "",
                            v.get("why") or ""))
    if soft:
        meta["hero_soft"] = soft
        for v in soft:
            notes.append("认英雄这一步有个读数没被采信（%s，%s）—— %s。"
                         % (v.get("hero") or "", v.get("source") or "",
                            v.get("why") or ""))
    # **这份 BP 记录是哪来的** —— 报告与排查都要看得到。
    # ``model`` = 模型看 BP 帧认出来的（可能认错）；空 = 玩家自己录的。
    meta["bp_source"] = "model" if bp_guess else ("record" if bp else "")

    # **几个读数对不上** —— 谁说了算、别的读数是什么，都记下来。
    distinct: List[Tuple[str, str]] = []
    for name, source in legal:
        if name not in [n for n, _ in distinct]:
            distinct.append((name, source))
    # **被否掉的读数也是读数** —— 它跟定案的那个不一样，
    # 读者有权知道「模型还说过谁」，所以它也算进「几个读数」。
    said: List[Tuple[str, str]] = list(distinct)
    for v in voided:
        vname = v.get("hero") or ""
        if vname and vname not in [n for n, _ in said]:
            said.append((vname, v.get("source") or ""))
    if hero_source and len(said) > 1:
        meta["hero_conflict"] = {
            "model": model_hero,
            "clip": clip.hero, "bp": bp_hero,
            "decision": hero_source,
        }
        notes.append("认英雄这一步几个读数对不上（%s）—— 以「%s」为准。"
                     % ("、".join("%s说%s" % (s, n) for n, s in said),
                        hero_source))
    elif hero_source and model_hero and model_hero != hero:
        # **模型看整局说的那个名字不对**（实测：说了一个被禁用的英雄）。
        # 这不是「口径变了」，是**读数本身错了** —— 但读者有权知道。
        meta["hero_conflict"] = {
            "model": model_hero,
            "clip": clip.hero, "bp": bp_hero,
            "decision": hero_source,
        }
    meta["hero_source"] = hero_source
    if hero_source in ("clip", "bp") and model_hero and model_hero != hero:
        # **模型看整局说的那个名字不对**（实测：说了一个被禁用的英雄）。
        # 这不是「口径变了」，是**读数本身错了** —— 但读者有权知道。
        meta["hero_conflict"] = {
            "model": model_hero,
            "clip": clip.hero, "bp": bp_hero,
            "decision": hero_source,
        }
    # -- 分路：画面逐段统计 / 顶层读数 / 英雄查表，三处对账 ----------
    #
    # 为什么要有这一层（2026-11-07 实测）：那一局玩家走的是**上路**，
    # 报告却写成「中路」—— 模型顶层那个 ``position`` 答了 ``mid``，
    # 而它自己的叙述里全是「上路河道」「上路一塔」。**顶层那一个词
    # 孤证独大，错了没人拦**，检索也就按错的路去筛攻略（那条路的一条
    # 攻略都进不来，覆盖表整列「—」）。
    #
    # 现在：模型**逐段**标 ``lane``，按众数定分路（几十条证据），
    # 再跟顶层读数、英雄查表对账；谁被否了、为什么被否，要说清楚。
    hero_position = _position_from_hero(hero)
    lane = judge_lane(watch.events,
                      watch_position=watch.position,
                      hero_position=hero_position,
                      hero_name=hero)
    declared_position = str(cfg.position or "").strip()
    if declared_position:
        # **玩家填的分路优先**（2026-11-14）：他知道自己走哪路。
        # 画面众数只用来对账 —— 对不上就说清，但不改他的答案。
        if lane.position and lane.position != declared_position:
            notes.append(
                "画面多数时间在「%s」，但你填的是「%s」—— 按你填的算。"
                % (POSITION_LABELS.get(lane.position, lane.position),
                   POSITION_LABELS.get(declared_position, declared_position)))
        position = declared_position
        meta["position_source"] = "你填的"
        meta["position_provenance"] = "declared"
    elif lane.position:
        position = lane.position
        meta["position_source"] = lane.source
        meta["position_provenance"] = lane.provenance
    else:
        if not position or position == "unknown":
            position = hero_position
        if position and position != "unknown":
            # 只有「你填的」或「按英雄推的」撑着 —— 说清是哪一种。
            if declared_position:
                meta["position_source"] = "你填的"
                meta["position_provenance"] = "declared"
            elif hero_position:
                meta["position_source"] = "按英雄推断（%s）" % hero
                meta["position_provenance"] = "inferred"
    meta["hero"] = hero
    meta["position"] = position
    meta["lane_detail"] = lane.lane_detail
    meta["lane_evidence"] = lane.to_dict()
    # **英雄层的打法按它的「官方分路」写，而这一局可能打的不是那条路。**
    #
    # 2026-11-13 实测（那一局）：公孙离的 12 条打法全写在发育路，
    # 而画面上玩家在中路 —— 英雄层按分路筛掉、分路通则又被
    # 「已认到英雄」的互斥护栏挡掉，于是**候选 0 条**，报告整页空着。
    # 真值敖隐也一样：敖隐的官方分路是发育路，这局打中路。
    #
    # 这种局面不是「没认准英雄」，而是**英雄层的攻略不覆盖这局打法**。
    # 那时按「英雄未知」去筛，让分路通则能进来（并写清楚为什么）——
    # 总比拿一张空表让玩家以为「没得判」强。
    judge_hero = hero
    hero_lanes = _hero_official_lanes(hero)
    if hero and position and hero_lanes and position not in hero_lanes:
        judge_hero = ""
        meta["hero_off_lane"] = True
        notes.append(
            "英雄：「%s」的攻略都是按「%s」写的，而这一局你走的是「%s」—— "
            "英雄层没有一条对得上，这次按分路通则判。"
            % (hero,
               "、".join(POSITION_LABELS.get(x, x) for x in sorted(hero_lanes)),
               POSITION_LABELS.get(position, position)))
    #: **判规则时当成「谁」** —— 与 ``meta["hero"]``（报告上显示的那个）分开：
    #: 英雄层的攻略不覆盖这局打法时，这里会是空串。
    meta["judge_hero"] = judge_hero
    # -- 死亡清单：失败局的倒推复盘要用 --------------------------------
    #
    # 清单**始终**记（读不到就是空数组）；「下次怎么做」那几句只在
    # **失败局**去检索 —— 赢了的局没有败因要推，也省下一次嵌入调用。
    #
    # 攻略是**模式库的原话**，不是我们编的：检索不到就空，
    # 报告层会写「这一段没查到可对照的攻略」。
    #
    # 死亡点**只用来定位**（第几次、怎么死的、落在哪个时间窗）；
    # 「下次怎么做」由**判定层按模式库判出来的那条**给出 ——
    # 这里**不另起一套检索**，也不自己拼建议（见 death_review.py）。
    #
    # 「下次怎么做」**自己检索**攻略语料 —— 这一层**绕过规则库**：
    # 规则库那套判不了死因（种子模式里写着「『有没有被切』读不到」），
    # 也不该由它来挑这一局该看哪一条。
    deaths: List[Dict[str, Any]] = []
    # **按时间决定留哪几条** —— 模型可能倒着报（「从最后一次死亡往前推」）。
    # 照原序截前 8 条，丢掉的其实是**最早**的那几次。
    #
    # **落库的顺序保持模型原样** —— 顺序对报告没有影响
    # （复盘那层自己按时间排），没必要在这里动它。
    all_deaths = list(watch.deaths or [])
    _by_time = sorted(all_deaths,
                      key=lambda d: float(getattr(d, "video_start", 0.0) or 0.0))
    _keep_ids = {id(d) for d in _by_time[:8]}
    kept = [d for d in all_deaths if id(d) in _keep_ids]
    if len(all_deaths) > len(kept):
        notes.append("死亡：模型报了 %d 次死亡，这次只记了时间上最后 %d 次 —— "
                     "更早的那 %d 次没进复盘。"
                     % (len(all_deaths), len(kept), len(all_deaths) - len(kept)))
    for d in kept:
        item = d.to_dict()
        at = float(item.get("video_start") or 0.0)
        if start_t is not None:
            item["clock"] = _mmss(max(0.0, at - float(start_t)))
            item["minute"] = round(max(0.0, at - float(start_t)) / 60.0, 2)
        if str(meta.get("result") or "") == "lose":
            item["guide"] = _guide_at_death(cfg.game, position, hero, item)
        deaths.append(item)
    meta["deaths"] = deaths
    # **模型一共报了几次** —— 落库的清单是截断过的（上面最多 8 条），
    # 报告层要说出真实总数，「这局共 N 次」才不是拿截断后的数在骗人。
    meta["deaths_total"] = len(all_deaths)

    # -- 血条：像素法读全程曲线，模型在残血时刻复核 ------------------
    #
    # **为什么不让模型直接读血量**（2026-11-08 实测）：
    # 模型看整局录像时每次都答「满血」—— 连亲眼确认只有两成血的那一帧
    # 也答满。原因是血条是画面中央一条 95×10 px 的细条，视频降采样之后
    # 彻底糊了，它其实是**按满血猜**的。拉近裁剪放大之后它才读得出来。
    #
    # 所以这里**像素法为主、模型复核**：像素量出全程曲线（零模型成本），
    # 模型只在最险的几段裁剪放大问一次，与像素读数**对账**；
    # 对不上就留痕，不装作没这回事。
    if cfg.hp_read:
        # 扫**打包片**（模型看的就是它）；打包没成功才退回原片。
        hp_path = pack.path or video_path
        hp_transform = None if pack.path else transform
        meta["hp"] = hp_bar.analyze(
            hp_path, transform=hp_transform, data_dir=cfg.data_dir,
            confirm=cfg.hp_confirm, match_start=start_t,
            time_scale=pack.time_scale, source_start=pack.source_start,
            max_shots=cfg.hp_confirm_shots)
        hp_notes = [str(n) for n in (meta["hp"].get("notes") or [])]
        notes.extend(hp_notes)
        # **残血时段对照打法** —— 模式库原文，一条一次检索。
        episodes = meta["hp"].get("episodes") or []
        picked = sorted(episodes, key=lambda e: float(e.get("min_pct") or 0.0)
                        )[:max(0, int(cfg.hp_guide_shots))]
        for episode in picked:
            episode["guide"] = _guide_at_hp(cfg.game, position, hero, episode)
            # **标出「查过了」** —— 没查过的段不能说「没查到攻略」，
            # 那是在替检索结果撒谎（2026-11-12 复核发现）。
            episode["guide_searched"] = episode.get("minute") is not None
        # **留痕要落在用户看得见的说明位**（前缀「血量：」）。
        if not meta["hp"].get("ok"):
            # **说清楚是哪一步没成** —— 血条模块自己知道原因
            # （视频打不开 / 帧率读不到 / 帧数太少 / 画面归一失败），
            # 写死一句「画面里找不到血条」会把人引去调错的东西。
            reason = hp_notes[0] if hp_notes else "这一局没读到血量"
            notes.append("血量：%s —— 血线这次没有结论。" % reason)
        elif int(meta["hp"].get("disagreed") or 0):
            notes.append("血量：%d 段残血两个读数不在同一档"
                         "（像素说残血、模型说不残血，或反过来）"
                         "—— 已按像素读数记，模型的说法留在明细里。"
                         % int(meta["hp"]["disagreed"]))
    else:
        meta["hp"] = {"ok": False, "skipped": True, "episodes": [], "notes": []}
    if meta.get("position_provenance") == "inferred" and hero:
        # 旧口径是「按英雄推断（赵云）」—— **带上英雄名**，读者才知道按谁推的。
        meta["position_source"] = "按英雄推断（%s）" % hero
    if lane.conflict:
        # 前缀「分路：」是**给报告层抓的** —— 页面模板冻结，
        # 说明只能写进 version_note，报告层靠这个前缀挑出这一句。
        notes.append("分路：" + lane.conflict)
    if declared_position and position and declared_position != position:
        notes.append("分路：你填的是「%s」，画面上看是「%s」—— 以画面为准。"
                     % (POSITION_LABELS.get(declared_position, declared_position),
                        POSITION_LABELS.get(position, position)))
    if not hero:
        # 这句话会一路走到**报告顶部的说明位**（``version_note``）——
        # 报告页只渲染那一个说明位，所以它必须**用户看得懂**，
        # 不能写成「英雄层没参与判定」这种内部话。
        notes.append("没认准这一局的英雄 —— 英雄层的打法这次一条都没判，"
                     "只判了分路层的。")
    if not position or position == "unknown":
        notes.append("没定出这一局的分路 —— 带分路条件的规则不参与判定。")
    if start_t is None:
        notes.append("没定出对局从哪一秒开始 —— 带时间窗的规则一条都判不了。")

    # -- ④ 对线优劣：**让模型看这一局的对线期**（2026-11-14 用户要求）----
    #
    # 原话：「不要对位表了，让模型去看」。
    # 对线劣势不是英雄决定的：同一个对位被反野三次、血量一直半条，就是劣势；
    # 表只能回答「理论上谁强」，回答不了「这一局打成什么样」。
    if cfg.matchup_judge:
        if cfg.on_progress is not None:
            cfg.on_progress("正在判断对线优劣", 0.60)
        try:
            from . import matchup as _matchup

            judged = _matchup.judge(
                beats=list(timeline.beats or []),
                deaths=list(meta.get("deaths") or []),
                hp=meta.get("hp") or {}, hero=hero or "",
                lane=position or "", model="", data_dir=cfg.data_dir)
            meta["matchup"] = judged.to_dict()
            meta["matchup"]["calls"] = judged.calls
            meta["matchup"]["tokens"] = judged.tokens
            notes.extend(judged.notes or [])
            if judged.matchup != "unknown":
                notes.append(
                    "对线优劣：模型判为「%s」（%s）。"
                    % (_matchup.LABELS.get(judged.matchup, judged.matchup),
                       judged.why or "没写依据"))
        except Exception as exc:                       # noqa: BLE001
            meta["matchup"] = {"matchup": "unknown", "notes": [str(exc)]}
            notes.append("对线优劣没判成（%s）—— 带这个前提的规则这次不判。" % exc)

    # -- ⑤ 死亡复盘：**这波是赚还是亏** ----------------------------------
    #
    # 用户的要求（2026-11-13）：「针对局内死亡，模型可以参考模式库的基础上，
    # 并自己搜索攻略，给出死亡分析，分析对局内有益还是有害」。
    #
    # 一层只做一件事：旧的死亡清单（``meta["deaths"]``）是**看整局时记下的
    # 死因**；这一步逐条复盘 —— 换了谁、丢了什么、值不值、下次怎么做。
    # 攻略由模型自己写检索请求、我们替它查（见 :mod:`app.recognition2.coach`）。
    #
    # **先读顶栏 K/D/A**：判「赚还是亏」要有事实 —— 死前死后我方有没有拿到人头。
    # 「读数字」交给模型（这是它擅长的），「算增量、推死亡时刻」交给代码
    # （见 :mod:`app.recognition2.kda_read`）—— 数数不该有概率。
    if cfg.kda_read:
        if cfg.on_progress is not None:
            cfg.on_progress("正在读顶栏的击杀/死亡/助攻", 0.92)
        try:
            from . import kda_read

            times = kda_read.sample_times(
                start_t or 0.0, float(pack.source_duration or 0.0))
            kda = kda_read.read_series(
                video_path, times, model="", data_dir=cfg.data_dir,
                transform=transform)
            meta["kda"] = kda.to_dict()
            notes.extend(kda.notes or [])
            if meta.get("deaths"):
                check = kda_read.cross_check(meta["deaths"], kda.events)
                meta["kda_check"] = check
                if not check.get("ok"):
                    notes.append(
                        "死亡次数对不上：模型报了 %d 次，顶栏 K/D/A 推出来 %d 次"
                        "（对上 %d 次）—— 报告里的死亡复盘按模型那份写，"
                        "对不上的地方请对着录像核。"
                        % (check["model_count"], check["hud_count"],
                           check["matched"]))
        except Exception as exc:                       # noqa: BLE001
            meta["kda"] = {"ok": False, "notes": [str(exc)]}
            notes.append("读顶栏 K/D/A 失败（%s）—— 这次死亡复盘没有用它。" % exc)

    if cfg.coach_deaths and meta.get("deaths"):
        if cfg.on_progress is not None:
            cfg.on_progress("正在复盘每一次死亡", 0.64)
        try:
            from . import coach

            insights, coach_notes = coach.review_deaths(
                meta["deaths"], rounds=list(timeline.beats or []),
                video_path=video_path, facts=facts, lane=position or "",
                hero=hero, model="", data_dir=cfg.data_dir,
                transform=transform,
                kda_events=list((meta.get("kda") or {}).get("events") or []))
            meta["coach"] = {
                "deaths": [d.to_dict() for d in insights],
                "calls": sum(int(d.calls or 0) for d in insights),
                "tokens": sum(int(d.tokens or 0) for d in insights),
            }
            notes.extend(coach_notes)
            trade = {}
            for item in meta["coach"]["deaths"]:
                trade[item.get("trade") or "unknown"] = \
                    trade.get(item.get("trade") or "unknown", 0) + 1
            if trade:
                notes.append(
                    "死亡复盘：%d 次死亡里有 %s —— 每一次的「赚还是亏」写在正文的前几条。"
                    % (len(insights),
                       "、".join("%s %d 次" % (k, v) for k, v in
                                sorted(trade.items()))))
        except Exception as exc:                       # noqa: BLE001
            meta["coach"] = {"deaths": [], "error": str(exc)}
            notes.append("死亡复盘没做成（%s）—— 正文只写了死因。" % exc)

    # -- ⑥ 选候选（模型自己写检索请求 → RAG） --------------------------
    #
    # 这一步以前是代码按**作用域/英雄/时间窗**把 1 680 条整条筛一遍
    # （:func:`app.recognition2.judge.screen_candidates`）—— 筛出来的
    # 与这一局发生了什么无关，模型被塞了一堆用不着的尺子。
    #
    # 现在由**看过这一局的模型**决定查什么：它写检索请求，
    # RAG 库去取攻略，代码只保留数据级不变量（见 :mod:`app.recognition2.selector`）。
    patterns = _load_patterns(cfg.game)
    meta["rules_total"] = len(patterns)
    # **⑥ 完全依赖 ②④⑤**（2026-11-14 用户要求）：
    # 把死亡/问题操作直接当检索请求 —— 前提是要按真实情况触发的。
    try:
        _situation_reqs = selector.requests_from_situation(
            (meta.get("coach") or {}).get("deaths") or [],
            list(getattr(overview, "operations", []) or []))
    except Exception:                                  # noqa: BLE001
        _situation_reqs = []
    selection = selector.select_candidates(
        timeline,
        situation_requests=_situation_reqs,
        requests=watch.requests,
        # **用 ``judge_hero``，不是 ``hero``** —— 英雄层的攻略不覆盖这局打法时
        # 它是空串，让分路通则能进筛（见上面 ``judge_hero`` 那一段）。
        facts=MatchFacts(my_hero=judge_hero, position=position),
        patterns=patterns,
        config=selector.SelectorConfig(
            game=cfg.game,
            lane=position,
            hero=judge_hero,
            top_k=cfg.retrieval_top_k,
            max_requests=cfg.max_retrieval_requests,
            max_candidates=cfg.max_candidates,
            # **前提归模型判**（2026-11-14 用户要求：局势与规则的理解都归模型）。
            model_premise=cfg.model_premise,
        ),
    )
    candidates = selection.candidates
    screened = selection.screened
    notes.extend(selection.notes or [])
    meta["candidates"] = len(candidates)
    meta["screened_out"] = len(screened)
    meta["candidate_ids"] = [c.rule_id for c in candidates]
    # **这次走的是哪条路**（模型自己找 / 退回顾则筛）—— 要能查。
    meta["selection_mode"] = selection.mode
    meta["selection_queries"] = [r.query for r in selection.requests]
    meta["selection_hits"] = selection.hits
    # **检索能力有没有降级**（向量那路没走通 → 只按字面匹配）。
    meta["selection_degraded"] = str(
        (selection.degradation or {}).get("dense_error") or "")
    if not candidates:
        notes.append("这一局没有要对照的攻略条目 —— 没有裁决。")
        # **说清楚是哪一步筛掉的** —— 只说「没对上」等于没说。
        #
        # 筛掉的理由是**整句**（带分路名、带规则 id），上千条摆出来没人看；
        # 但「别的分路 1382 条 / 别的英雄 1596 条」这种**数**一眼就能看出
        # 是哪一步没对上 —— 这才是能拿去改的诊断。
        buckets = _screen_buckets(screened)
        meta["screened_buckets"] = buckets
        if buckets:
            details = _screen_details(screened)
            notes.append("筛掉明细：" + "；".join(
                "%s %d 条%s" % (k, buckets[k],
                                _detail_suffix(details.get(k)))
                for k in sorted(buckets, key=lambda k: -buckets[k])))

    # -- ⑦ 逐条裁决（模型看录像片段） --
    if cfg.on_progress is not None:
        cfg.on_progress("正在对照攻略判断", 0.74)
    # **局势摘要**：对线优劣 + 每次死亡的赚亏 —— 裁决要「综合以上所有信息」。
    _situation = _situation_brief(meta)
    judgement = adjudicate(timeline, candidates, AdjudicatorConfig(
        model="", data_dir=cfg.data_dir, frames_dir=cfg.frames_dir,
        situation=_situation,
        max_rules_per_call=cfg.max_rules_per_call,
        max_frames_per_call=cfg.max_frames_per_call,
        # **不收视频的模型改用抽帧判读** —— 否则每一条候选都 422。
        evidence=("frames" if meta.get("watch_mode") == "frames"
                  else cfg.evidence),
        video_path=video_path, video_dir=cfg.video_dir),
        transform=transform)
    meta["judge_calls"] = judgement.calls
    meta["judge_tokens"] = judgement.tokens
    notes.extend(judgement.notes)

    # -- ⑧ 全局综述 --
    overview = None
    if cfg.observe:
        if cfg.on_progress is not None:
            cfg.on_progress("正在复盘整局", 0.90)
        guide = guide_digest(patterns, lane=position, hero=hero,
                              limit=cfg.guide_limit)
        meta["guide_items"] = len(guide)
        overview = observe(timeline, guide, ObserverConfig(
            model="", data_dir=cfg.data_dir, frames_dir=cfg.frames_dir))
        meta["observe_calls"] = overview.calls
        meta["observe_tokens"] = overview.tokens
        notes.extend(overview.notes)
        # 模型看整局时自己写的总结 —— 综述层没写成时拿它顶上，
        # 免得报告里「这一局怎么样」这一栏是空的。
        if not (overview.summary or "").strip() and (watch.summary or "").strip():
            overview.summary = watch.summary

    # -- ⑨ 整局复盘：**错在哪 + 教官分析**（报告正文第一、二部分）-------
    #
    # 用户 2026-11-13 的要求：正文改成三段式 ——
    #   ① 总结：指出玩家错在哪；② 模型当教官给分析；③ 逐条明细。
    # 而且明确说「**不要再考虑命中规则的事情**」：这一步拿的是这一局的事实
    # （时间线描述 + 每次死亡 + 有问题的操作），**不看规则命中**。
    if cfg.coach_match:
        if cfg.on_progress is not None:
            cfg.on_progress("正在写整局复盘", 0.97)
        try:
            from . import coach

            review = coach.review_match(
                rounds=list(timeline.beats or []),
                deaths=list((meta.get("coach") or {}).get("deaths") or []),
                operations=list(getattr(overview, "operations", []) or []),
                situation=_situation_brief(meta),
                rules=_rule_digests(candidates),
                hero=hero or "", lane=position or "",
                result=str(meta.get("result") or ""),
                minutes=float(pack.source_duration or pack.duration or 0.0) / 60.0,
                model="", data_dir=cfg.data_dir)
            meta.setdefault("coach", {})
            meta["coach"]["match"] = review.to_dict()
            meta["coach"]["calls"] = int(meta["coach"].get("calls") or 0) + \
                int(review.calls or 0)
            meta["coach"]["tokens"] = int(meta["coach"].get("tokens") or 0) + \
                int(review.tokens or 0)
            notes.extend(review.notes or [])
            if review.wrong_points:
                notes.append(
                    "整局复盘：指出 %d 处问题、给出 %d 条教练分析（正文前两部分）。"
                    % (len(review.wrong_points), len(review.analysis)))
        except Exception as exc:                       # noqa: BLE001
            meta.setdefault("coach", {})["match"] = {"wrong_points": [],
                                                     "analysis": [],
                                                     "error": str(exc)}
            notes.append("整局复盘没做成（%s）—— 正文只有逐条明细。" % exc)

    # -- 转成报告层认识的形状 --
    #
    # 报告层读的是 ``raw["rounds"]``：每一段带 ``review``（模型逐条裁决）、
    # ``narrative``（模型原话）、``time_min``（局内分钟）。
    # 转换放在这里做，调用方拿到的就是能直接落库的形状。
    rounds = to_legacy_rounds(timeline, judgement.verdicts,
                              {str(getattr(p, "id", "") or ""): p
                               for p in patterns})

    if cfg.on_progress is not None:
        cfg.on_progress("完成", 1.0)
    return {
        "timeline": timeline.to_dict(),
        "verdicts": judgement.to_dict(),
        "overview": overview.to_dict() if overview else None,
        "rounds": rounds,
        "meta": meta,
        "notes": notes,
    }
