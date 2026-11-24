"""**判定层**：把时间线（Beat）对齐到规则库，算出「该做而没做」。

这一层是识别子系统 v2 的第三层
------------------------------
:mod:`app.recognition2.reader` 负责「**玩家做了什么**」，
这一层负责「**该做而没做**」—— 两件事的**口径必须分开**，
否则报告会把「我们没看清」说成「玩家没做」。

三条铁律
--------
1. **看不清 ≠ 没做**。一段 ``unclear`` 的画面是「我们没看见」，
   不是「玩家没做」；判「没做」必须有**证据**（这段画面看清了，
   而且做的是别的事）。
2. **不是给这个玩家的 ≠ 卡点**。规则的 ``applies_to`` / ``when_none``
   把它排除掉时，它是**不适用**，不计入分母 —— 说它是「卡点」是在
   冤枉一个本来就不该被这条规则管的人。
3. **前提不成立 ≠ 没做**。``when.position == "jungle"`` 是
   「玩家在野区时这条才适用」；他这会儿在别处，那这条通则压根不是给他的。

候选从哪来
----------
「这一局要检验哪几条」现在**由模型决定**
（:mod:`app.recognition2.selector`：模型自己写检索请求，RAG 去取攻略原文）。
这一层负责的是**另一件事**：一条规则**这一局要不要检验** ——
退役的、不是给这个玩家的、命中排除名单的、时间窗没交上的、前提不成立的，
都不该拿去问模型。

所以这里的「筛子」**不再决定查什么**，只当**护栏**；
整库筛一遍的老口径留作兜底（RAG 走不通时用），见
:func:`app.recognition2.selector.select_candidates`。

为什么要**纯代码**判
--------------------
规则的前提极简 —— 翻遍全库，只有 ``position`` / ``my_hero`` / ``time_min``
三个事实是常态（另有 ``matchup`` / ``has_ult_ready`` 各 10 条上下），
期望则是**动作码**，和 :class:`~app.recognition2.beat.Beat` 的
``action`` **同一套词表**。所以能直接比对，不需要模型。

**但这份比对不参与判定** —— 判定归模型（:mod:`app.recognition2.adjudicator`）。
它只用来**对账**：模型说「做到」而代码比对说「没做到」，说明两边看的
不是同一件事，报告要把它写出来，而不是只挑一边说。

读不到的事实（``matchup`` 等）判 **unknown**，不判「没做」——
这也是为什么这里宁可少判一条，也不多冤枉一条。

时间怎么对
----------
规则的时间窗是**局内分钟**（``node.from_min`` / ``to_min``），
Beat 的 ``game_time_start/end`` 是**局内秒**。判定用**区间求交**：
规则窗口 ∩ 对局区间 → 这一局里真正该检验的那段时间。
取交不取点，是因为「这一段覆盖了 03:00~03:20」和「这一段发生在 03:12」
是两回事（见 :mod:`app.recognition2.clockaxis`）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .beat import UNCLEAR, Beat, Timeline

#: 做到了。
STATUS_SATISFIED = "satisfied"
#: 该做而没做。
STATUS_DEVIATED = "deviated"
#: 判不了（画面看不清 / 事实读不到 / 那段时间根本没看到）。
STATUS_UNDETERMINED = "undetermined"
#: **不适用**（不是给这个玩家的 / 前提不成立 / 这一局没到这个时间点）。
STATUS_NOT_APPLICABLE = "not_applicable"

#: 计入分母的状态 —— 只有「判得了」的才算执行度。
_DECIDED = (STATUS_SATISFIED, STATUS_DEVIATED)

#: 分路层规则的**排除名单**靠这个事实命中（``when_none``）。
_HERO_FACT = "my_hero"

#: 这些事实 v2 **读不到**（模型读画面读不出来，也不该去猜）。
_UNREADABLE_FACTS = {
    "matchup", "has_ult_ready", "bp_phase", "my_missing_frontline",
    "my_picks_before", "opp_picks_before", "my_output_count",
    "my_other_has_tank_or_support", "my_missing_engage",
    "my_missing_damage", "opp_missing_frontline", "my_has_flex",
    "map_position", "in_combat", "is_decision_point", "pressure",
    "phase", "lane",
}


# ---------------------------------------------------------------- 事实


@dataclass
class MatchFacts:
    """**对局级**事实 —— 外部给的，判定层自己不猜。

    ``position`` 是本局玩家的分路（``jungle`` / ``mid`` / ``side`` /
    ``carry`` / ``roam``）。留空 = **不知道**，那就不拿它筛规则，
    更不能拿它当「不适用」的借口。
    """

    my_hero: str = ""
    position: str = ""

    def known_hero(self) -> Optional[str]:
        h = str(self.my_hero or "").strip()
        return h or None

    def known_position(self) -> Optional[str]:
        p = str(self.position or "").strip()
        if not p or p == "unknown":
            return None
        return p


# ---------------------------------------------------------------- 判定结果


@dataclass
class RuleJudgement:
    """一条规则的判定结果。"""

    rule_id: str
    rule_title: str = ""
    category: str = ""
    lane: str = ""
    severity: str = "medium"
    status: str = STATUS_UNDETERMINED
    #: **人话**：为什么是这个状态。解释与判定共用一套口径。
    reason: str = ""
    #: 规则的时间窗（局内分钟）。
    window: Tuple[float, float] = (0.0, 0.0)
    #: 实际参与检验的局内秒区间。
    span: Tuple[float, float] = (0.0, 0.0)
    #: 做到了的段（摘要：时刻 + 动作 + 叙述）。
    matched: List[Dict[str, Any]] = field(default_factory=list)
    #: 这一段里实际观察到的动作（去重、排序）。
    observed: List[str] = field(default_factory=list)
    #: 读不到的事实 —— 判不了的原因。
    unknown_facts: List[str] = field(default_factory=list)
    #: 规则的**原文前提**（「什么情况下才该这么做」）。
    premise: str = ""
    #: 建议（规则自带的纠正文案）。
    advice: str = ""

    @property
    def decided(self) -> bool:
        return self.status in _DECIDED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_title": self.rule_title,
            "category": self.category,
            "lane": self.lane,
            "severity": self.severity,
            "status": self.status,
            "reason": self.reason,
            "window": list(self.window),
            "span": list(self.span),
            "matched": list(self.matched),
            "observed": list(self.observed),
            "unknown_facts": list(self.unknown_facts),
            "premise": self.premise,
            "advice": self.advice,
        }


@dataclass
class JudgementStats:
    """一局的判定统计。

    **分母只算判得了的** —— ``undetermined`` 和 ``not_applicable``
    都不进分母。否则「模型没看清」会把执行度算低，
    「这条不适用」会把执行度算成 unfairly 的低分。
    """

    considered: int = 0
    satisfied: int = 0
    deviated: int = 0
    undetermined: int = 0
    not_applicable: int = 0
    #: 因**规则本身是条件性的**而判不了的条数。
    #:
    #: 这批是「代码判不了、得让模型看画面判前提」的清单 ——
    #: 攻略的适用条件（「对方位移刚交」「暴君倒计时最后一段」）
    #: 画面读不出来。数字摆在这儿，才知道下一步的工作量在哪。
    conditional: int = 0

    @property
    def decided(self) -> int:
        return self.satisfied + self.deviated

    @property
    def compliance_rate(self) -> Optional[float]:
        """执行度。一条都判不了时返回 ``None`` —— 不编一个数字。"""
        if self.decided == 0:
            return None
        return self.satisfied / self.decided

    @property
    def coverage(self) -> float:
        """**被检验到的比例** —— 必须和执行度**一起看**。

        只看执行度会骗人：判得了的 5 条全做到 → 执行度 1.0，
        但还有 4 条**判不了**（条件性规则），读者会以为「全做对了」。
        覆盖率把这件事说清楚：5 / 9 = 0.56。
        """
        if not self.considered:
            return 0.0
        return self.decided / self.considered

    def to_dict(self) -> Dict[str, Any]:
        return {
            "considered": self.considered,
            "satisfied": self.satisfied,
            "deviated": self.deviated,
            "undetermined": self.undetermined,
            "not_applicable": self.not_applicable,
            "decided": self.decided,
            "conditional": self.conditional,
            "compliance_rate": (
                round(self.compliance_rate, 4)
                if self.compliance_rate is not None else None
            ),
            "coverage": round(self.coverage, 4),
        }


@dataclass
class Judgement:
    """一局的判定结果。"""

    judgements: List[RuleJudgement] = field(default_factory=list)
    stats: JudgementStats = field(default_factory=JudgementStats)
    #: 整局级别的留痕（为什么判不了、哪些没检验到…）。
    notes: List[str] = field(default_factory=list)
    #: 对局区间（局内秒）。
    game_span: Tuple[float, float] = (0.0, 0.0)

    def by_status(self, status: str) -> List[RuleJudgement]:
        return [j for j in self.judgements if j.status == status]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "judgements": [j.to_dict() for j in self.judgements],
            "stats": self.stats.to_dict(),
            "notes": list(self.notes),
            "game_span": list(self.game_span),
        }


# ---------------------------------------------------------------- 工具


def _beat_actions(beat: Beat) -> Set[str]:
    """这一段做过的动作（``action`` + ``all_actions``）。

    ``all_actions`` 也要算 —— 玩家一段里干了多件事（先刷野后推线），
    只看 ``action`` 会漏掉「期望动作在这段里做过」。
    """
    out: Set[str] = set()
    action = str(beat.action or "").strip()
    if action and action != UNCLEAR:
        out.add(action)
    detail = beat.detail or {}
    for item in (detail.get("all_actions") or []):
        code = str(item or "").strip()
        if code and code != UNCLEAR:
            out.add(code)
    return out


def _position_codes() -> Set[str]:
    """规范位置码（从适配器取，不另编一套）。"""
    try:
        from ..adapters.honor_of_kings import POSITION_VALUES
        return {str(v) for v in POSITION_VALUES if v != "unknown"}
    except Exception:                                  # noqa: BLE001
        return {"jungle", "mid", "side", "carry", "roam"}


def _action_labels() -> Dict[str, str]:
    """动作码 -> 中文名（从适配器取，不另编一套）。

    理由要**给人看** —— 直接甩 ``farm_jungle`` 出来，
    报告页上就是一行没人认得的码。
    """
    try:
        from ..adapters.honor_of_kings import MOBA_ACTION_LABELS
        return dict(MOBA_ACTION_LABELS)
    except Exception:                                  # noqa: BLE001
        return {}


def _action_text(codes: Iterable[str]) -> str:
    """动作码列表 -> 中文（``刷野补经济`` 这种）。"""
    labels = _action_labels()
    out: List[str] = []
    for code in codes:
        text = labels.get(code) or code
        if text not in out:
            out.append(text)
    return "、".join(out)


def _beat_position(beat: Beat) -> Optional[str]:
    """这一段的**规范**位置码；读不到返回 ``None``（不是 ``unknown``）。

    **只认规范码**。自由描述（「红buff野区及下方野怪营地」这种）
    必须当「读不到」—— 拿它去比 ``position == "jungle"``
    只会得到「不成立」，那会把「读不到」变成「前提不成立」，
    进而把一条本来该检验的规则判成「不适用」——
    **结论看起来很干净，其实全是错的**。
    """
    detail = beat.detail or {}
    pos = str(detail.get("position") or "").strip()
    if not pos or pos not in _position_codes():
        return None
    return pos


def dominant_position(beats: Sequence[Beat]) -> Optional[str]:
    """一段区间里玩家**主要在哪个位置**。

    取**众数**：一段 20 秒里前 12 秒在野区、后 8 秒在中路，
    这段算「在野区」。取众数而不是取第一段或最后一段，
    因为一段的区间边界是**切出来的**，不是玩家自己选的。

    对局级分路读不到时，这是**兜底**（认不准就别拿它筛规则）。
    """
    counter: Dict[str, int] = {}
    for b in beats:
        pos = _beat_position(b)
        if pos:
            counter[pos] = counter.get(pos, 0) + 1
    if not counter:
        return None
    best = max(counter.items(), key=lambda kv: (kv[1], kv[0]))
    return best[0]


def _overlap(a_lo: float, a_hi: float, b_lo: float, b_hi: float
             ) -> Optional[Tuple[float, float]]:
    """两个区间求交；不相交返回 ``None``。"""
    lo = max(a_lo, b_lo)
    hi = min(a_hi, b_hi)
    if hi < lo:
        return None
    return (lo, hi)


def _pattern_window(pattern) -> Tuple[float, float]:
    """规则的时间窗（局内**秒**）。"""
    node = getattr(pattern, "node", None)
    lo = float(getattr(node, "from_min", 0.0) or 0.0) * 60.0
    hi = float(getattr(node, "to_min", 0.0) or 0.0) * 60.0
    if hi < lo:
        lo, hi = hi, lo
    return (lo, hi)


def _expected_codes(pattern) -> Set[str]:
    """期望的动作码（``expect`` + ``accept``）。"""
    out: Set[str] = set()
    for chk in (getattr(pattern, "expect", None) or []):
        value = getattr(chk, "value", None)
        if isinstance(value, (list, tuple, set)):
            out.update(str(v) for v in value)
        elif value is not None:
            out.add(str(value))
    for code in (getattr(pattern, "accept", None) or []):
        out.add(str(code))
    return out


# ---------------------------------------------------------------- 事实取值


def _fact_value(fact: str, facts: MatchFacts,
                beats: Sequence[Beat]) -> Tuple[Any, bool]:
    """取事实的取值。返回 ``(取值, 读得到吗)``。

    ``读得到吗`` 是 ``False`` 时，取值没意义 —— 判定层要判 **unknown**，
    不能拿 ``None`` 去跑 ``matches_value``（那会返回「不成立」，
    把「读不到」变成「没做」）。
    """
    if fact == _HERO_FACT:
        hero = facts.known_hero()
        return (hero, hero is not None)
    if fact == "position":
        # **对局级分路优先**。
        #
        # 为什么不能用逐段读数当主口径：分路是「你打哪条线」，是**对局级**事实。
        # 规则前提里的 ``position`` 就是**作用域**（1680 条规则每一条都有
        # ``when: position == <自己的分路>``，与 ``applies_to.lane`` 同一件事），
        # 而作用域在筛子的第 2 步已经按**对局级分路**判过了。
        #
        # 第 5 步再拿逐段读数否决一遍，等于用两套口径判同一件事 ——
        # 第 2 步说「你是发育路，这条是给发育路的」，
        # 第 5 步说「你这一段不在发育路上，这条不适用」，**自相矛盾**。
        #
        # 真实踩过的坑（2026-11-02 熬隐那局）：模型逐段读出的是
        # 「野区」「河道」「中路」——「野区/河道」不是分路，归一不成就是
        # 「读不到」；某几段读到「中路」就成了这一窗的众数。
        # 于是这一局 13 条该检验的发育路规则**一条不剩**被筛掉，
        # 报告全空 —— 而玩家玩的就是发育路。
        #
        # 逐段读数只在对局级分路**读不到**时兜底（那时它至少是「在场上」的证据）。
        pos = facts.known_position()
        if pos:
            return (pos, True)
        pos = dominant_position(beats)
        return (pos, pos is not None)
    if fact in ("time_min", "time_sec"):
        #: 时间前提由**规则自己的窗口**保证成立，判定时永远为真。
        return (0, True)
    if fact in _UNREADABLE_FACTS:
        return (None, False)
    return (None, False)


def _check_matches(chk, actual: Any) -> bool:
    """按条件**自己的**算子判。

    直接复用 :meth:`app.core.dsl2.Check.matches_value` —— 别自己拿
    取值集合硬碰：``!=`` / ``not_in`` 会被判反，而护栏组里恰好有这两种。
    """
    try:
        return bool(chk.matches_value(actual))
    except Exception:                                  # noqa: BLE001
        return False


# ---------------------------------------------------------------- 判定


@dataclass
class _Screen:
    """筛子的结果。

    ``beats`` 为 ``None`` = **不用检验**（退役 / 不是给这个玩家的 /
    命中排除名单 / 时间窗没交上 / 前提不成立）。
    """

    judgement: RuleJudgement
    beats: Optional[List[Beat]] = None
    span: Optional[Tuple[float, float]] = None
    #: 前提里有**读不到**的事实 —— 分不清「他没做」还是「这条不适用他」。
    premise_unknown: bool = False


def _screen(pattern, facts: MatchFacts, in_game: Sequence[Beat],
            game_span: Tuple[float, float],
            model_premise: bool = False) -> _Screen:
    """**筛子**：只回答「这条规则这一局**要不要检验**」。

    理由都写在 ``判定.reason`` 里，调用方照原样带走。

    为什么要拆出来
    --------------
    「要不要检验」和「检验结果」是**两件事**：

    - 前者是**机械的** —— 分路对不对、英雄对不对、时间窗有没有交上，
      代码筛得比模型准，也更便宜；
    - 后者是**理解的** —— 「玩家做的这件事，算不算攻略说的那件事」，
      以及「攻略的前提（敌方位移刚交、暴君倒计时到哪了）画面上能不能确认」，
      只有看画面才答得上来。

    所以模型裁决层（:mod:`app.recognition2.adjudicator`）只拿这里筛出来的
    候选，判官是模型，筛子还是这段代码。
    """
    j = RuleJudgement(
        rule_id=str(getattr(pattern, "id", "") or ""),
        rule_title=str(getattr(pattern, "title", "") or ""),
        category=str(getattr(pattern, "category", "") or ""),
        lane=str(getattr(pattern, "lane", "") or ""),
        severity=str(getattr(pattern, "severity", "medium") or "medium"),
        premise=str(getattr(pattern, "premise", "") or ""),
        advice=str(getattr(pattern, "deviation_hint", "") or ""),
    )

    # -- 1. 退役 / 非生效 --
    status = str(getattr(pattern, "status", "active") or "active")
    if status != "active":
        j.status = STATUS_NOT_APPLICABLE
        j.reason = "这条规则已退役（status=%s），不参与判定。" % status
        return _Screen(j)

    # -- 2. 适用范围：不是给这个玩家的，就不是卡点 --
    applies = getattr(pattern, "applies_to", None)
    lane = str(getattr(applies, "lane", "") or "")
    heroes = tuple(getattr(applies, "heroes", ()) or ())
    my_position = facts.known_position()
    if lane and my_position and lane != my_position:
        j.status = STATUS_NOT_APPLICABLE
        j.reason = ("这条是给「%s」的，本局玩家是「%s」—— 不是给他写的，"
                    "不算他没做。" % (lane, my_position))
        return _Screen(j)
    my_hero = facts.known_hero()
    if heroes and not my_hero:
        # **没认出本局玩的是谁 → 英雄专属的规则一条都不参与判定。**
        #
        # 以前这里是 `heroes and my_hero and ...` —— 英雄为空时短路，
        # 于是「认不准」变成了「全都算」：模型被拿着扁鹊、貂蝉、王昭君
        # 的打法去问「你这一段有没有照做」，答出来的是「你玩的是别人」。
        # 真实跑一遍看到的就是这种报告：「玩家并未操控王昭君……所以不能
        # 视为满足」——理由自己说没玩这个英雄，结论却写着「没按攻略做」。
        #
        # 认不准比认错强：认错会拿别的英雄的打法去指责玩家。
        # 宁可少判一条，也不冤枉一个本来就不该被这条规则管的人。
        j.status = STATUS_NOT_APPLICABLE
        j.reason = ("没认出本局玩的是谁 —— 这条是「%s」专属的，先不判"
                    "（认不准比认错强：认错会拿别的英雄的打法去指责玩家）。"
                    % "、".join(heroes))
        return _Screen(j)
    if heroes and my_hero and my_hero not in heroes:
        j.status = STATUS_NOT_APPLICABLE
        j.reason = ("这条只针对 %s，本局玩的是 %s —— 不适用。"
                    % ("、".join(heroes), my_hero))
        return _Screen(j)

    # -- 3. 排除名单（when_none）：命中就不适用 --
    when_none = list(getattr(pattern, "when_none", None) or [])
    excluded = False
    for chk in when_none:
        actual, known = _fact_value(str(chk.fact), facts, in_game)
        if not known:
            j.unknown_facts.append(str(chk.fact))
            continue
        if _check_matches(chk, actual):
            excluded = True
            j.reason = "本局玩家在排除名单里（%s %s）—— 这条不适用他。" % (
                chk.fact, actual)
            break
    if excluded:
        j.status = STATUS_NOT_APPLICABLE
        return _Screen(j)

    # -- 4. 时间窗求交 --
    win = _pattern_window(pattern)
    j.window = (round(win[0] / 60.0, 2), round(win[1] / 60.0, 2))
    span = _overlap(win[0], win[1], game_span[0], game_span[1])
    if span is None:
        j.status = STATUS_NOT_APPLICABLE
        j.reason = ("规则窗口是第 %g~%g 分钟，这一局的对局区间是第 %g~%g 分钟，"
                    "没交上 —— 这条根本没到检验的时候。"
                    % (j.window[0], j.window[1],
                       round(game_span[0] / 60.0, 1),
                       round(game_span[1] / 60.0, 1)))
        return _Screen(j)
    j.span = (round(span[0], 2), round(span[1], 2))

    beats = [b for b in in_game
             if b.game_time_end >= span[0] and b.game_time_start <= span[1]]
    beats.sort(key=lambda b: b.game_time)

    # -- 5. 前提 --
    premise_ok = True
    premise_unknown = False
    #: 前提不成立时，是**哪个事实**对不上 —— 理由要写得出名字，
    #: 不然读者只看到「前提不成立」，不知道是分路对不上还是别的。
    failed_facts: List[str] = []
    for group in (getattr(pattern, "when", None) or [],
                  getattr(pattern, "when_any", None) or []):
        for chk in group:
            actual, known = _fact_value(str(chk.fact), facts, beats)
            if not known:
                premise_unknown = True
                if str(chk.fact) not in j.unknown_facts:
                    j.unknown_facts.append(str(chk.fact))
                continue
            if not _check_matches(chk, actual):
                premise_ok = False
                if str(chk.fact) not in failed_facts:
                    failed_facts.append(str(chk.fact))
    if not premise_ok and not model_premise:
        j.status = STATUS_NOT_APPLICABLE
        j.reason = ("前提不成立（%s）—— 这条本来就不是给他这种情况写的，"
                    "不算他没做。" % "、".join(failed_facts))
        return _Screen(j)
    if not premise_ok and model_premise:
        # **前提判定交给模型**（2026-11-14 用户要求：局势与规则的理解都归模型）。
        #
        # 代码只能说「按我手上这几个字段，条件对不上」—— 而字段是粗的：
        # ``gold_diff`` 读不到、``matchup`` 是模型刚判的、``pressure`` 是推的。
        # 拿这些去**替模型下结论**，等于让一个读不全事实的筛子说了算。
        # 所以这里只把「哪几项对不上」记下来，当线索交给模型，
        # 由它判这条打法适不适用这一段（不适用就写 ``not_applicable``）。
        if "、".join(failed_facts) and "前提项对不上" not in j.reason:
            j.reason = ("前提项对不上（%s）—— 交给模型判断是否适用。"
                        % "、".join(failed_facts))
        # 对不上的那几项也要进 ``unknown_facts``：裁决层就是靠这个字段
        # 把「哪几项前提对不上」当线索写进提示词的。
        for _fact in failed_facts:
            if _fact not in j.unknown_facts:
                j.unknown_facts.append(_fact)
        premise_unknown = True

    return _Screen(j, beats, (span[0], span[1]), premise_unknown)


def _judge_one(pattern, facts: MatchFacts,
               in_game: Sequence[Beat], game_span: Tuple[float, float],
               screen: Optional[_Screen] = None
               ) -> RuleJudgement:
    """判一条规则（**纯代码**口径）。

    「要不要检验」交给 :func:`_screen`；这里只管**检验结果**。

    ``screen`` 可以传进来 —— 筛与判共用一次筛选，省一遍时间；
    不传就自己筛一次。
    """
    if screen is None:
        screen = _screen(pattern, facts, in_game, game_span)
    j = screen.judgement
    if screen.beats is None:
        return j
    beats = screen.beats
    span = screen.span or (0.0, 0.0)
    premise_unknown = screen.premise_unknown

    # -- 6. 期望 --
    wanted = _expected_codes(pattern)
    matched: List[Beat] = []
    observed: Set[str] = set()
    clear_count = 0
    for b in beats:
        actions = _beat_actions(b)
        observed.update(actions)
        if not actions:
            clear_count += 1
            continue
        if actions & wanted:
            matched.append(b)
    j.observed = sorted(observed)
    j.matched = [{
        "game_time": round(float(b.game_time), 1),
        "action": b.action,
        "action_label": b.action_label,
        "narrative": (b.narrative or "")[:160],
        "frames": list(b.frames),
    } for b in matched]

    min_count = 1
    for chk in (getattr(pattern, "expect", None) or []):
        if chk.min_count is not None:
            min_count = max(min_count, int(chk.min_count))
    if len(matched) >= min_count:
        j.status = STATUS_SATISFIED
        j.reason = "做到了（第 %g~%g 分钟这一段里做了 %s）。" % (
            j.window[0], j.window[1],
            _action_text(sorted({m["action_label"] or m["action"]
                                 for m in j.matched})))
        return j

    # -- 7. 没做到：先分清「看不清」与「不是他的」 --
    if not beats or clear_count == len(beats):
        j.status = STATUS_UNDETERMINED
        j.reason = ("第 %g~%g 分钟这一段**没看清**（%d 段画面都没读出动作）—— "
                    "判不了，不算他没做。"
                    % (j.window[0], j.window[1], len(beats)))
        return j
    if premise_unknown:
        j.status = STATUS_UNDETERMINED
        j.reason = ("这一段没做到期望动作，但前提里的「%s」读不到，"
                    "分不清是「他没做」还是「这条不适用他」—— 判不了。"
                    % "、".join(j.unknown_facts))
        return j

    # -- 8. 条件性规则：判不了「该做而没做」 --
    #
    # 攻略规则的**适用条件**是局面条件（「对方位移或闪现刚交」
    # 「首条暴君倒计时进入最后一段」「敌方打野最后出现位置在另一半地图」），
    # 这些**画面读不出来**（v2 读不到敌方打野在哪、暴君倒计时到哪了）。
    #
    # 所以：**前提读不到 → 判不了，绝不判「该做而没做」**。
    # 判成「偏离」是在冤枉一个「条件根本没成立」的人 ——
    # 说谎的诊断比没有诊断更坏。
    if j.premise.strip():
        j.status = STATUS_UNDETERMINED
        j.reason = ("这条是**条件性规则** —— 原文前提是「%s」。"
                    "这些前提（敌方动向 / 地图状态 / 倒计时）画面读不出来，"
                    "所以判不了是「该做而没做」还是「条件没成立」。"
                    % j.premise[:80])
        return j

    j.status = STATUS_DEVIATED
    j.reason = ("第 %g~%g 分钟这一段做的是 %s，规则期望的是 %s —— "
                "该做而没做。"
                % (j.window[0], j.window[1],
                   _action_text(j.observed) or "（没读到动作）",
                   _action_text(sorted(wanted)) or "（没写）"))
    return j


def judge(timeline: Timeline, patterns: Iterable[Any],
          facts: Optional[MatchFacts] = None) -> Judgement:
    """把一局的时间线对齐到规则库。

    ``patterns`` 是规则清单（用 :func:`app.rules.patterns.load_patterns`
    拿）。判定层**不自己读规则库** —— 那样它就跟存储方式绑死了，
    也就没法用构造出来的规则测。

    ``facts`` 是对局级事实（英雄、分路）。留空 = 不知道，
    那就**不拿它筛规则**（更不能拿它当「不适用」的借口）。
    """
    facts = facts or MatchFacts()
    out = Judgement()

    in_game = [b for b in timeline.beats
               if b.game_time_start >= 0 and b.game_time_end >= 0]
    if not in_game:
        out.notes.append("这一局没有带局内时刻的段 —— 判不了任何规则。")
        return out
    game_span = (min(b.game_time_start for b in in_game),
                 max(b.game_time_end for b in in_game))
    out.game_span = (round(game_span[0], 2), round(game_span[1], 2))

    considered = 0
    for pattern in patterns or []:
        j = _judge_one(pattern, facts, in_game, game_span)
        out.judgements.append(j)
        if j.status == STATUS_NOT_APPLICABLE:
            out.stats.not_applicable += 1
            continue
        considered += 1
        out.stats.considered += 1
        if j.status == STATUS_SATISFIED:
            out.stats.satisfied += 1
        elif j.status == STATUS_DEVIATED:
            out.stats.deviated += 1
        else:
            out.stats.undetermined += 1
            #: 因规则是**条件性的**而判不了的，单独记一笔。
            if j.premise.strip():
                out.stats.conditional += 1

    if out.stats.compliance_rate is None:
        out.notes.append("一条都判不了（画面看不清或规则不适用）—— 不给执行度数字。")
    if out.stats.undetermined:
        out.notes.append("有 %d 条判不了（画面看不清或前提读不到），"
                         "它们**不计入**执行度的分母。" % out.stats.undetermined)
    if out.stats.conditional:
        out.notes.append(
            "其中 %d 条是**条件性规则**（原文前提是「对方位移刚交」"
            "「暴君倒计时最后一段」这类局面条件，画面读不出来）—— "
            "这些**判不了「该做而没做」**，要判得让模型看画面逐条裁决前提。"
            % out.stats.conditional)
    return out


# ---------------------------------------------------------------- 筛子（对外）


@dataclass
class Candidate:
    """一条**这一局要检验**的规则，连同检验它的那几段画面。

    这是「筛子」的产物 —— 模型裁决层（:mod:`app.recognition2.adjudicator`）
    拿这个去问模型：这一段玩家做的这些事，算不算攻略说的那件事。
    """

    pattern: Any
    judgement: RuleJudgement
    beats: List[Beat]
    span: Tuple[float, float]

    @property
    def rule_id(self) -> str:
        return self.judgement.rule_id

    @property
    def code_status(self) -> str:
        """**纯代码比对**的口径（满足 / 偏离 / 判不了 / 不适用）。

        它不参与判定 —— 判定归模型。这个字段只用来**对账**：
        模型说「做到」而代码比对说「没做到」，或者反过来，
        说明两边看到的不是同一件事，报告要把它写出来。
        """
        return str(getattr(self.judgement, "status", "") or "")

    @property
    def window(self) -> Tuple[float, float]:
        """规则的时间窗（局内分钟）。"""
        return self.judgement.window

    def rule_text(self, max_len: int = 600) -> str:
        """给模型看的**规则原文**。

        为什么改成「规则卡」（2026-11-13）
        ----------------------------------
        原来这里手拼五个字段（标题/前提/要求/结果/建议），而**分支没进去** ——
        英雄层攻略的价值恰恰在分支（敌方红开就这样、蓝开就那样）。
        只给第一条，模型只能拿一句不适用于本局的要求去判，然后判「你没做到」。

        现在统一走 :mod:`app.rules.cards`：同一份规则在裁决层与综述层
        长得一样，而且带分支、带「看什么」。
        """
        try:
            from ..rules import cards

            text = cards.card_block(cards.rule_card(self.pattern))
        except Exception:                              # noqa: BLE001
            text = ""
        if not text:
            pat = self.pattern
            parts: List[str] = []
            title = str(getattr(pat, "title", "") or "").strip()
            if title:
                parts.append("规则：%s" % title)
            premise = str(getattr(pat, "premise", "") or "").strip()
            if premise:
                parts.append("前提：%s" % premise)
            action = str(getattr(pat, "action", "") or "").strip()
            if action:
                parts.append("要求：%s" % action)
            text = "\n".join(parts)
        return text[:max_len]


def screen_one(pattern: Any, facts: MatchFacts, in_game: Sequence[Beat],
               game_span: Tuple[float, float], model_premise: bool = False
               ) -> Tuple[Optional[Candidate], Optional[RuleJudgement]]:
    """筛**一条**规则。

    返回 ``(候选, 被筛掉的判定)`` —— 两者只有一个不是 ``None``：

    * 候选不为空 → 这条这一局要检验（带着检验区间与画面段）；
    * 判定不为空 → 这条不适用（带着**为什么**）。

    拆出来的原因：候选的来源有两处（整库筛、模型检索回来的），
    但「一条规则要不要检验」只有**一份**实现 —— 两处各写一遍迟早漂，
    漂了就有一半对局说谎。
    """
    screen = _screen(pattern, facts, in_game, game_span,
                     model_premise=model_premise)
    if screen.beats is None:
        return None, screen.judgement
    # **纯代码的比对结果也留下来** —— 它不参与判定（判定归模型），
    # 但要能跟模型的裁决**对账**：两边不一致时，报告得说出来。
    code_judgement = _judge_one(pattern, facts, in_game, game_span,
                                screen=screen)
    # **把「前提里哪几项对不上」带到候选上**（2026-11-14）：
    # ``_judge_one`` 拿到的是代码口径的判定，它不知道筛子记下的那些
    # 读不到/对不上的前提项 —— 而裁决层要拿这些当线索交给模型。
    if screen.judgement.unknown_facts:
        code_judgement.unknown_facts = list(screen.judgement.unknown_facts)
    if screen.judgement.reason and "前提项对不上" in screen.judgement.reason:
        code_judgement.reason = screen.judgement.reason
    return (Candidate(
        pattern=pattern,
        judgement=code_judgement,
        beats=list(screen.beats),
        span=screen.span or (0.0, 0.0),
    ), None)


def screen_candidates(timeline: Timeline, patterns: Iterable[Any],
                       facts: Optional[MatchFacts] = None,
                       model_premise: bool = False
                       ) -> Tuple[List[Candidate], List[RuleJudgement]]:
    """把规则清单筛成「这一局要检验的候选」。

    返回 `(候选, 被筛掉的)`。被筛掉的那些带**为什么被筛掉**的理由。

    这是**筛子**，不是判官 —— 它只回答「这条规则这一局要不要检验」：
    退役的、不是给这个玩家的、命中排除名单的、时间窗没交上的，都不进候选。

    **前提（``when``）例外**（2026-11-14）：``model_premise=True`` 时，
    前提对不上**不再直接判不适用** —— 代码手上那几个字段本来就是粗的
    （金差读不到、对位是模型判的、压迫是推的），拿它替模型下结论，
    等于让一个读不全事实的筛子说了算。这时只把「哪几项对不上」记成线索，
    由模型判这条打法适不适用这一段。

    **但它顺手做一次纯代码比对**（:func:`_judge_one`），结果挂在候选上
    （:attr:`Candidate.code_status`）—— 不参与判定，只用来跟模型的裁决
    对账。两边不一致时报告要写出来：那是「两边看到的不是同一件事」，
    藏起来等于让玩家自己猜哪边对。

    **它是兜底口径**：新链路里候选由模型自己检索（:mod:`app.recognition2.selector`），
    只有那条路走不通时才整库筛一遍。
    """
    facts = facts or MatchFacts()
    in_game = [b for b in timeline.beats
               if b.game_time_start >= 0 and b.game_time_end >= 0]
    candidates: List[Candidate] = []
    screened: List[RuleJudgement] = []
    if not in_game:
        for pattern in patterns or []:
            j = RuleJudgement(rule_id=str(getattr(pattern, "id", "") or ""))
            j.status = STATUS_NOT_APPLICABLE
            j.reason = "这一局没有带局内时刻的段 —— 判不了任何规则。"
            screened.append(j)
        return candidates, screened
    game_span = (min(b.game_time_start for b in in_game),
                 max(b.game_time_end for b in in_game))
    for pattern in patterns or []:
        candidate, dropped = screen_one(pattern, facts, in_game, game_span,
                                        model_premise=model_premise)
        if candidate is None:
            if dropped is not None:
                screened.append(dropped)
            continue
        candidates.append(candidate)
    return candidates, screened
