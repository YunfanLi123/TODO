"""金铲铲之战（JinChanChan / 掌上英雄联盟）适配器。

与云顶之弈的关系：同源的自走棋，回合结构一样，但**数值口径不同**，
而且这些差异必须体现在判定上，不能只在文案里改字眼：

- 存钱上限 60（金币上限更高，50 就停手的结论在金铲铲里是错的）
- 人口里程碑整体提前一档：3 阶段就要 6 人口，4-1 卡 8 人口
- 连胜/连败 4 连就拿到翻倍收入奖励，前期滚雪球更快

因此本文件定义自己的 :class:`JCCAdapter`（独立类、独立注册、独立
``fact_names`` 列表），复用 :mod:`app.adapters.tft` 里的解析流程函数并传入
本游戏自己的 :class:`~app.adapters.tft.AutoChessProfile`。
共用的是「怎么把一个回合翻译成事实」，不是适配器对象。
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..core.context import MatchContext
from .base import GameAdapter, register
from .tft import (
    AUTOCHESS_ACTION_LABELS,
    AUTOCHESS_ACTION_VALUES,
    AUTOCHESS_FACT_LABELS,
    AUTOCHESS_FACT_NAMES,
    AutoChessProfile,
    build_autochess_context,
)

# --------------------------------------------------------------------------
# 金铲铲专属口径
# --------------------------------------------------------------------------

#: 金铲铲的当前赛季号。规则库按版本绑定，换赛季时规则要重新核一遍。
JCC_VERSION = "S14"

JCC_PROFILE = AutoChessProfile(
    game_id="jcc",
    display_name="金铲铲之战",
    default_version=JCC_VERSION,
    # 金币上限 60：多出来的 10 金币同样产生利息，「攒到 50 就停手」在金铲铲是错的
    gold_cap=60,
    # 人口节奏整体提前：3-1 就要 6 人口，4-1 卡 8 人口，5 阶段即可到 9 人口
    level_milestones={1: 4, 2: 5, 3: 6, 4: 8, 5: 9, 6: 9},
    # 4 连即拿满额连胜奖励，金铲铲前期滚雪球比云顶更快
    streak_bonus_threshold=4,
    # 金铲铲节奏快，40 血就已经进入需要止损的区间
    health_low=35,
    health_mid=65,
)

#: 金铲铲的动作取值集合。与云顶共用同名取值，但金铲铲把「强化主C」
#: 拆得更细（追三星与 D 牌是两种不同收益），这里只登记同一套取值，
#: 新增取值必须同步扩充规则库，否则规则会静默漏判。
#:
#: 同样按动作事实分组：标注页要为每一类动作单独列出可选取值。
JCC_ACTION_VALUES: Dict[str, Dict[str, str]] = {
    kind: dict(values) for kind, values in AUTOCHESS_ACTION_VALUES.items()
}
JCC_ACTION_VALUES["econ_action"].update({
    "upgrade_carry": "强化主C（装备/羁绊）",
    "share_draft": "共享选秀",
    "activate_season_buff": "激活赛季强化",
})

JCC_ACTION_LABELS: Dict[str, str] = {
    value: label
    for group in JCC_ACTION_VALUES.values()
    for value, label in group.items()
}


class JCCAdapter(GameAdapter):
    """金铲铲之战。"""

    game_id = "jcc"
    # profile 里已经带着这一版的版本号，直接暴露出来，
    # 免得上传 / 攻略页还要各维护一份版本默认值。
    default_version = JCC_PROFILE.default_version
    display_name = "金铲铲之战"
    #: 独立列表对象 —— 与云顶不共享同一个 list 对象
    fact_names = list(AUTOCHESS_FACT_NAMES)
    #: 期望动作只能挂在这两类事实上，否则规则会恒判为偏离
    action_fact_names = ["econ_action", "positioning_action"]
    fact_labels = dict(AUTOCHESS_FACT_LABELS)
    action_labels = dict(JCC_ACTION_LABELS)
    action_values_by_fact = {
        k: dict(v) for k, v in JCC_ACTION_VALUES.items()
    }
    #: 标注页要问玩家的数字型事实。与云顶同一组：只问原始量，
    #: 收入 / 节奏 / 压力 / 是否关键回合都是这五个原始量派生的，
    #: 问了派生量等于让玩家白填 —— 填进去的值会被适配器覆盖掉。
    annotate_numeric_facts = [
        "health",     # 剩余生命值
        "level",      # 人口等级
        "gold",       # 当前金币
        "streak",     # 连胜(+) / 连败(-) 场次
        "item_count", # 主 C 身上的装备件数
    ]
    round_unit = "回合"
    profile = JCC_PROFILE

    def normalize(self, raw: Dict[str, Any]) -> MatchContext:
        return build_autochess_context(
            raw,
            self.profile,
            self.fact_names,
            self.action_labels,
        )


#: 与云顶分开注册的两款游戏，REGISTRY["tft"] 与 REGISTRY["jcc"] 是不同实例
JCC_ADAPTER = register(JCCAdapter())
