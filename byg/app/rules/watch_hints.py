# -*- coding: utf-8 -*-
"""**「看什么」从攻略原文里抽出来**（2026-11-13）。

为什么需要这一栏
---------------
裁决层要模型回答「这一段有没有做到这条打法」。模型看的是**抽帧画面**，
它需要知道：判这条**要看画面上的哪个东西**？

* 「残血还去河道接团」→ 要看**血条**和**小地图上敌方的位置**；
* 「野区被反」→ 要看**小地图**上自家野区有没有敌方红点；
* 「该转线」→ 要看**兵线**在哪、塔还站着没有。

这些信息**攻略原文里本来就有**（「看小地图」「血低于三成」），只是以前没抽出来，
模型只能自己猜依据。这一层做的事很窄：**把原文里那句关于「看什么」的话挑出来**，
不新增任何说法 —— 抽不到就空着（不编）。

为什么不用模型抽
---------------
抽一句「原文里提到小地图」是**字符串层面**的事，规则确定、可测、不花钱；
交给模型反而多一次调用、多一处不一致。真需要理解上下文的（比如「他为什么这么说」）
才交给模型。
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple

#: 线索词 → 归一名。
#:
#: **只收「画面上看得见的东西」**：小地图、视野、血条、塔、兵线、龙坑、
#: 技能冷却、经济面板。像「敌方打野为红开」这种是**情报**不是**画面** ——
#: 它进「情形」那一栏，不该进「看什么」（2026-11-13 实测抽错过一次：
#: 前提句里的「敌方打野」「血量」被当成了要看的东西）。
CUES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("小地图", ("小地图", "地图上", "看地图", "红点", "露头", "消失")),
    ("视野", ("视野", "眼位", "插眼", "照亮", "视野盲区", "探草")),
    ("血条", ("血条", "血量", "残血", "半血")),
    ("塔", ("外塔", "一塔", "二塔", "高地", "塔的护盾", "推塔", "守塔")),
    ("兵线", ("兵线", "清线", "压线", "线权", "塔下兵")),
    ("龙坑", ("龙坑", "暴君", "主宰", "打龙", "控龙")),
    ("技能冷却", ("冷却", "技能好了", "CD", "大招", "闪现", "惩击", "召唤师技能")),
    ("经济面板", ("经济差", "金币", "看经济")),
)

#: 「看什么」这句话里通常带一个**看的动作** —— 没有它，多数只是陈述。
LOOK_VERBS = ("看", "注意", "盯", "观察", "留意", "确认", "检查")

#: 这几个线索**本身就是画面**（小地图/视野/血条）：句子里出现它就够，
#: 不必再要求「看的动作」（「血量低于一半」已经是在说看什么了）。
VISUAL_LABELS = ("小地图", "视野", "血条", "经济面板")

#: 一条线索最多几个字（进提示词的「看什么」一行）。
MAX_PHRASE = 26

#: 每条规则最多抽几条（再多就是拿钱请模型读废话）。
MAX_HINTS = 2

_SPLIT_RE = re.compile(r"[。；;！!？?\n]+")
_CLAUSE_RE = re.compile(r"[，,、]+")


def _clauses(text: str) -> List[str]:
    out: List[str] = []
    # 库里 ``source.quoted`` 会把几段「动作：…」拼在一起 ——
    # 不先拆开，「做视野 动作：推塔顺序…」这种半截句会漏进来。
    text = (text or "").replace("动作：", "。").replace("动作:", "。")
    for sentence in _SPLIT_RE.split(text):
        for clause in _CLAUSE_RE.split(sentence):
            clause = clause.strip()
            if clause:
                out.append(clause)
    return out


def _cue_of(clause: str) -> str:
    for label, words in CUES:
        if any(word in clause for word in words):
            return label
    return ""


def extract_watch(text: str, limit: int = MAX_HINTS) -> List[str]:
    """从一段攻略原文里挑出「判这条要看什么」的那几句。

    返回的是**原文里的句子**（截断到 :data:`MAX_PHRASE` 字），不是我们编的说法 ——
    这样溯源时一眼能对回原文。

    判据：句子里有**画面上看得见的东西**（:data:`CUES`），并且满足其一：

    * 句子里有**看的动作**（:data:`LOOK_VERBS`）——「看小地图」「注意血条」；
    * 线索本身就是画面（:data:`VISUAL_LABELS`）——「血量低于一半」也算。

    「敌方打野为红开」这种**情报**不算 —— 它属于「情形」，不是「看什么」。
    """
    out: List[str] = []
    seen: set = set()
    for raw in _clauses(text):
        clause = raw
        for prefix in ("动作：", "动作:", "要求：", "要求:", "做法："):
            if clause.startswith(prefix):
                clause = clause[len(prefix):]
        label = _cue_of(clause)
        if not label or label in seen:
            continue
        if label not in VISUAL_LABELS and not any(v in clause
                                                  for v in LOOK_VERBS):
            continue
        seen.add(label)
        phrase = clause if len(clause) <= MAX_PHRASE else clause[:MAX_PHRASE] + "…"
        out.append(phrase)
        if len(out) >= max(1, limit):
            break
    return out


def has_cue(text: str) -> bool:
    """这段文字里有没有「看什么」类的话（给测试与统计用）。"""
    return any(_cue_of(clause) for clause in _clauses(text))
