"""和狄仁杰、诸葛亮聊天。

这两个人物本来就站在页面上盯着你看 —— 这一层让他们**能回话**。

**聊的是局外的事**：打法怎么用、节奏怎么排、这个版本有什么变化。
依据是**操作模式库**（``rag.retriever.search_modes`` 查出来的打法）。

**不是**把玩家那局的报告再解释一遍 —— 报告页自己已经写清楚了，
在这儿复读一遍只是让人多看一屏字。给模型的只有玩家的情况
（英雄、分路、版本），让他答得具体一点，不给他报告。

两个人各是一个**角色**，不是一个通用问答：

- 狄仁杰（判官）：直给、挑毛病、说人话；
- 诸葛亮（军师）：讲策略、给下一步。

**查不到就说查不到** —— 不编。答案要点明依据是哪一条。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

#: 两个人物。人设写在提示词里 —— 同一个模型，换一段设定就是另一个人。
CHARACTERS: Dict[str, Dict[str, str]] = {
    "drj": {
        "name": "狄仁杰",
        "role": "判官",
        "greeting": "要问哪条打法的判罚？说来。",
        "persona": (
            "你是狄仁杰：断案的判官。说话直、挑毛病、不绕弯，也不安慰人。\n"
            "你回答的是**游戏本身的事**：某条操作模式打法怎么用、什么时候用、"
            "违反了会怎样。你依据下面查到的**操作模式打法**回答。\n"
            "这是**局外拓展** —— 不要复述玩家那局的报告（报告页自己已经写清楚了），"
            "也不要去复盘他刚打完的那一局。\n"
            "先说结论，再说依据；两三句话说清。不寒暄、不说套话。"
        ),
    },
    "zgl": {
        "name": "诸葛亮",
        "role": "军师",
        "greeting": "打法、节奏、版本适配 —— 问吧。",
        "persona": (
            "你是诸葛亮：军师。先讲道理，再说怎么用。\n"
            "你回答的是**游戏本身的事**：某个阶段的打法该怎么打、某个位置该怎么打、"
            "这个版本有什么变化。你依据下面查到的**操作模式打法**回答。\n"
            "这是**局外拓展** —— 不要复述玩家那局的报告（报告页自己已经写清楚了），"
            "也不要去复盘他刚打完的那一局。\n"
            "说人话，别掉书袋；一次只让人改一件事。两三句话。"
        ),
    },
}

DEFAULT_CHARACTER = "zgl"

#: 最多带几条打法进提示词 —— 太多模型会挑花眼，答得又慢又散。
MAX_RULES = 6

#: 历史上带几句对话进提示词（这个模型接口没有多轮角色，只拼文本）。
MAX_HISTORY = 6

SYSTEM_SUFFIX = (
    "\n\n要求：\n"
    "- **只依据下面给的操作模式打法**回答，不要编游戏机制、"
    "不要编玩家没做过的事；\n"
    "- 引用打法时带上它的编号或标题（例如「按②那条」），让人能去库里核对；\n"
    "- 打分库里查不到依据时，直说「库里没有对应打法」，别硬凑；\n"
    "- 这是**局外拓展**：聊打法与思路，不复述玩家那局的报告；\n"
    "- 2~4 句，不超过 180 字。"
)


def _rules_text(result: Any) -> str:
    """检索结果 → 带编号的打法清单。"""
    hits = list(getattr(result, "hits", []) or [])
    if not hits:
        return "（操作模式库里没有查到相关打法）"
    marks = "①②③④⑤⑥⑦⑧⑨⑩"
    lines: List[str] = []
    for i, hit in enumerate(hits[:MAX_RULES]):
        title = str(getattr(hit, "title", "") or "").strip()
        text = str(getattr(hit, "text", "") or "").strip()
        mark = marks[i] if i < len(marks) else str(i + 1)
        lines.append("%s %s：%s" % (mark, title or "（无名打法）", text))
    return "\n".join(lines)


def _history_text(history: Sequence[Dict[str, str]]) -> str:
    """对话历史 → 纯文本（这个模型接口只吃文本块）。"""
    lines: List[str] = []
    for item in list(history or [])[-MAX_HISTORY:]:
        role = str(item.get("role") or "").strip()
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        who = "玩家" if role == "user" else "对方"
        lines.append("%s：%s" % (who, text))
    return "\n".join(lines)


#: 段首标记 —— 编号要跟回答里引用的一致。
_MARKS = "①②③④⑤⑥⑦⑧⑨⑩"


def _query_terms(query: str) -> List[str]:
    """把问句拆成可匹配的词：中文取二元组，其余按词。

    中文不分词，整句拿去匹配等于没匹配 —— 二元组（「打野」「野前」…）
    才碰得到规则文本里的字。
    """
    terms = set()
    for part in re.sub(r"[^\w一-鿿]+", " ", (query or "")).split():
        if re.match(r"^[一-鿿]{2,}$", part):
            for i in range(len(part) - 1):
                terms.add(part[i:i + 2])
        else:
            terms.add(part)
    return sorted(t for t in terms if t)


def _fallback_rules(query: str, *, game: str, top_k: int) -> str:
    """**分块索引没建时的兜底**：直接在规则文档里按词找。

    为什么要有这条：操作模式库的**分块索引**可能是空的（``chunks`` 为 0），
    检索层一条都召不回 —— 那聊天就答不了。规则本身是在的（``documents``），
    按词捞一遍就够用：同一段文字切出来的块，跟索引里存的是同一份 ✓。
    """
    from ..rag.store import ModeStore, make_chunks

    terms = _query_terms(query)
    if not terms:
        return "（问题太短，不知道要查什么）"
    store = ModeStore()
    docs = store.documents(game=game or None)
    #: **按打法去重** —— 一条打法的多个分支各占一个名额的话，
    #: 前几名全是同一条规则的不同分支，等于只给了一条。
    best: Dict[str, Any] = {}
    for doc in docs:
        title = str(doc.get("title") or "")
        for chunk in make_chunks(doc):
            text = str(chunk.get("text") or "")
            blob = title + "\n" + text
            hit = sum(1 for t in terms if t in blob)
            if not hit:
                continue
            # 标题命中更值钱 —— 那是这条打法的名字
            score = hit + sum(1 for t in terms if t in title)
            key = title or str(doc.get("id") or "")
            if score > best.get(key, (0,))[0]:
                best[key] = (score, title, text)
    if not best:
        return "（操作模式库里没有查到相关打法）"
    scored = sorted(best.values(), key=lambda x: -x[0])
    lines = []
    for i, (_score, title, text) in enumerate(scored[:top_k]):
        mark = _MARKS[i] if i < len(_MARKS) else str(i + 1)
        lines.append("%s %s：%s" % (mark, title or "（无名打法）", text[:220]))
    return "\n".join(lines)


def reply(
    character: str,
    message: str,
    *,
    history: Optional[Sequence[Dict[str, str]]] = None,
    game: str = "hok",
    version: str = "S45",
    data_dir: str = "",
    timeout: float = 90.0,
) -> Dict[str, Any]:
    """问一句，回一句。

    返回 ``{"reply": str, "name": str, "role": str}``；出错时 ``reply`` 是
    一句人话说明（不是异常）—— 聊天是给人看的，不是给日志看的。
    """
    from ..collection import settings, vlm_client

    who = CHARACTERS.get(character or "") or CHARACTERS[DEFAULT_CHARACTER]
    text = (message or "").strip()
    if not text:
        return {"reply": who["greeting"], "name": who["name"], "role": who["role"]}

    # 1. 从操作模式库检索依据 —— 答的是「库里那条打法」，不是模型的知识。
    #    **不按本局过滤**：整库检索，聊的是打法本身。
    rules = ""
    try:
        from ..rag.retriever import search_modes

        with settings.use_data_dir(data_dir or None):
            result = search_modes(
                text[:200], game=game, version=version, top_k=MAX_RULES,
            )
        rules = _rules_text(result)
    except Exception as exc:                                  # noqa: BLE001
        rules = "（检索没跑起来：%s）" % exc
    if not rules or "没有查到" in rules:
        # 索引没建 / 召回空 → 直接在规则文档里按词捞一遍。
        try:
            rules = _fallback_rules(text, game=game, top_k=MAX_RULES)
        except Exception as exc:                              # noqa: BLE001
            rules = rules or ("（操作模式库没查到：%s）" % exc)

    # 2. 拼提示词：人设 + 依据 + 对话历史 + 这个问题。
    #    **不给本局的东西** —— 这是局外拓展，复述报告只是让人多看一屏字。
    parts = [
        "【可以引用的操作模式打法】\n%s" % rules,
    ]
    hist = _history_text(history or [])
    if hist:
        parts.append("【刚才的对话】\n%s" % hist)
    parts.append("【玩家的问题】\n%s" % text)
    prompt = "\n\n".join(parts)

    # 3. 调文字模型。
    try:
        with settings.use_data_dir(data_dir or None):
            creds = vlm_client.find_credentials("text", data_dir or None)
            model = settings.model_name("text")
        if not creds or not model:
            return {"reply": "（还没配文字模型，答不了。模型页填一下 API Key 就能聊。）",
                    "name": who["name"], "role": who["role"]}
        resp = vlm_client.call_model(
            model, [vlm_client.text_block(prompt)], creds=creds, slot="text",
            system=who["persona"] + SYSTEM_SUFFIX, max_tokens=500,
            timeout=timeout, retry=1,
        )
        answer = (resp.text or "").strip()
    except Exception as exc:                                  # noqa: BLE001
        return {"reply": "（我这会儿说不出话：%s）" % exc,
                "name": who["name"], "role": who["role"]}

    if not answer:
        answer = "（这句我没答上来，再问一次？）"
    return {"reply": answer, "name": who["name"], "role": who["role"]}
