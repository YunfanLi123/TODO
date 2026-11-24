"""Harness 的**工具层**：模型说「我要看这个」，代码去取。

为什么要有这一层
----------------
模型的动作只有一种：**「我要再看一眼」**（放大某一帧的某个局部、
补一帧画面）。它说清要什么，**代码去取**，取回来再问它。

把它单独抽出来，而不是散在识别器里，是为了：

* **动作解析可以换**：现在是「JSON 里的 ``need_more`` 字段」，
  将来端点支持原生 tool_call 时，换的只是「怎么把 tool_use 块
  翻译成同一个 :class:`ToolCall`」，取料与回填的代码不动；
* **失败可回填**：工具出错时把错误**回填给模型**，而不是抛异常
  （harness 的通用纪律：异常不穿循环）；
* **可测**：dispatch 是纯函数式的（给定请求 → 给定结果），能单测。

参考：Harness Engineering Guide《Tool System》《Agentic Loop》，
以及中文《从 Prompt Engineering 到 Agent Harness》的工具注册中心。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

#: 一次「模型要料」的请求。
@dataclass
class ToolCall:
    #: 工具名（``zoom_frame`` 看画面 / ``search_modes`` 查打法）。
    name: str = "zoom_frame"
    #: 要看的帧号（1 起，对应窗内帧编号）。
    frame: int = 0
    #: 要看的局部：``minimap`` / ``skill_ult`` / ``clock`` / ``whole``。
    zoom: str = "whole"
    #: 模型想确认什么（一句话）。
    about: str = ""
    #: 想确认的是哪一类事实/动作（可空）。
    kind: str = ""
    #: 给这张图贴的**编号说明**（如「图2 = 第3帧的小地图（放大）」）。
    #:
    #: 编号很重要：模型得能说「第 2 张图里有没有龙坑」，
    #: 否则它只能笼统地说「我看到了地图」（借鉴 OS-Genesis 的
    #: set-of-mark：给图编号，模型就能引用）。
    label: str = ""
    #: ``search_modes`` 用的自然语言查询（模型自己写的）。
    query: str = ""
    #: ``search_modes`` 要几条（0 = 用默认）。
    k: int = 0

    def key(self) -> str:
        """同一个请求的指纹 —— 用来发现「模型在原地打转」。"""
        if self.name == "search_modes":
            return "%s|%s" % (self.name, (self.query or "").strip())
        return "%s|%d|%s" % (self.name, self.frame, self.zoom)


@dataclass
class ToolResult:
    """工具执行结果。

    ``ok=False`` 时 ``error`` 是**要回填给模型的人话** ——
    模型看到「这里取不到」，可以自己换个法子，而不是我们替它决定。
    """

    ok: bool = True
    #: 取到的图（PNG/JPEG 字节）；取不到是空。
    image: Optional[bytes] = None
    #: 给模型看的文字说明。
    note: str = ""
    error: str = ""
    #: 工具取回的**结构化数据**（``search_modes`` 带回清单条目）。
    data: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "note": self.note,
            "error": self.error,
            "has_image": bool(self.image),
            "has_data": self.data is not None,
        }


#: 工具表：名字 → 实现。
#:
#: 实现签名：``(call, ctx) -> ToolResult``，``ctx`` 是
#: :class:`app.collection.vlm_recognizer.WindowInput`。
ToolImpl = Callable[[ToolCall, Any], ToolResult]

_REGISTRY: Dict[str, ToolImpl] = {}


def register(name: str) -> Callable[[ToolImpl], ToolImpl]:
    """把一个工具登记进表里。"""
    def deco(fn: ToolImpl) -> ToolImpl:
        _REGISTRY[name] = fn
        return fn
    return deco


def tool_names() -> List[str]:
    return sorted(_REGISTRY)


def dispatch(call: ToolCall, ctx: Any) -> ToolResult:
    """执行一次工具调用。

    **永不抛异常**：失败要变成 :class:`ToolResult`，
    因为错误要**回填给模型**（让它换个法子），
    而不是把整个识别流程炸掉。
    """
    fn = _REGISTRY.get(call.name)
    if fn is None:
        return ToolResult(
            ok=False,
            error="没有这个工具：%s（可用：%s）"
                  % (call.name, "、".join(tool_names()) or "无"))
    try:
        return fn(call, ctx)
    except Exception as exc:                          # noqa: BLE001
        return ToolResult(ok=False, error="取料失败：%s" % exc)


#: 局部区域的中文名（图上要印出来，模型才知道看的是哪块）。
ZOOM_LABELS = {
    "minimap": "小地图",
    "skill_ult": "大招",
    "clock": "计时器",
    "skill_all": "技能按钮",
}


@register("zoom_frame")
def tool_zoom_frame(call: ToolCall, ctx: Any) -> ToolResult:
    """把某一帧的某个局部裁出来、放大。

    ``ctx`` 需要提供 ``zoom_image(zoom, label) -> bytes | None``。
    取不到就**如实说取不到**（错误要回填给模型，让它换个法子），
    绝不返回一张错的局部图。
    """
    label = call.label or ("%s（放大）" % ZOOM_LABELS.get(call.zoom, call.zoom))
    try:
        image = ctx.zoom_image(call.zoom, label)
    except Exception as exc:                          # noqa: BLE001
        return ToolResult(ok=False, error="取料失败：%s" % exc)
    if not image:
        return ToolResult(
            ok=False,
            error="第 %d 帧的「%s」取不到（这一块没有标定，或裁不出来），"
                  "请按整帧判断，不要猜" % (
                      call.frame, ZOOM_LABELS.get(call.zoom, call.zoom)))
    return ToolResult(ok=True, image=image,
                      note="已放大第 %d 帧的%s" % (
                          call.frame, ZOOM_LABELS.get(call.zoom, call.zoom)))


@register("search_modes")
def tool_search_modes(call: ToolCall, ctx: Any) -> ToolResult:
    """模型主动查打法。

    模型在回答里说「我这一段在河道跟人打架，该不该占龙坑」，
    代码去模式库里把**相关的攻略原文**取回来给它判。

    ``ctx`` 需要提供 ``search_modes(query, k)``，返回一个字典：
    ``{"entries": [...], "degradation": {...}}``。

    失败**不抛异常**：查不到就回填一句「没查到相关打法」，
    模型按自己看到的画面判断即可 —— 检索是补充，不是前提。
    """
    query = (call.query or call.about or "").strip()
    if not query:
        return ToolResult(ok=False, error="没有说要看什么打法（query 是空的）")
    searcher = getattr(ctx, "search_modes", None)
    if searcher is None:
        return ToolResult(ok=False, error="这个上下文不支持检索打法")
    try:
        payload = searcher(query, int(call.k or 0))
    except Exception as exc:                          # noqa: BLE001
        return ToolResult(ok=False, error="检索打法失败：%s" % exc)
    if not isinstance(payload, dict):
        return ToolResult(ok=False, error="检索打法返回了意料之外的东西")
    entries = payload.get("entries") or []
    if not entries:
        return ToolResult(
            ok=True,
            note="没查到相关的打法（%s）。按你在画面里看到的判断，"
                 "不要硬套。" % (payload.get("degradation") or {}).get("reason", ""),
            data=payload)
    ids = "、".join(str(e.get("id") or "") for e in entries[:8])
    return ToolResult(
        ok=True,
        note="查到 %d 条相关打法：%s" % (len(entries), ids),
        data=payload)


def parse_need_more(raw: Any) -> List[ToolCall]:
    """把模型回答里的 ``need_more`` 解析成工具调用列表。

    形态有两种，都认：

    1. JSON 里的 ``need_more`` 字段（现在的协议）；
    2. 原生 ``tool_use`` 块（端点将来支持时）——
       工具名 ``zoom_frame``、参数 ``frame`` / ``zoom`` / ``about``。

    解析不出来就返回空列表 —— **不猜**。
    """
    out: List[ToolCall] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                frame = int(item.get("frame"))
            except (TypeError, ValueError):
                continue
            out.append(ToolCall(
                name=str(item.get("name") or "zoom_frame"),
                frame=frame,
                zoom=str(item.get("zoom") or "whole"),
                about=str(item.get("about") or ""),
                kind=str(item.get("kind") or ""),
            ))
    return out


def parse_need_modes(raw: Any) -> List[ToolCall]:
    """把模型回答里的 ``need_modes`` 解析成检索请求。

    形态与 :func:`parse_need_more` 同一层（JSON 字段）——
    将来端点支持原生 ``tool_use`` 时，只改「动作解析」那一层，
    取料与回填的代码不动。

    ``[{"query": "我在河道跟人打架，这时候该去占龙坑吗", "k": 5}]``
    """
    out: List[ToolCall] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query") or item.get("about") or "").strip()
        if not query:
            continue
        try:
            k = int(item.get("k") or 0)
        except (TypeError, ValueError):
            k = 0
        out.append(ToolCall(name="search_modes", query=query, k=k))
    return out


def parse_tool_use_blocks(blocks: Any) -> List[ToolCall]:
    """把 Anthropic 形状的 ``tool_use`` 块翻译成工具调用。

    端点不支持原生工具时这个函数不会命中（返回空列表），
    所以它可以**一直留着** —— 端点哪天支持了，这条路就自动生效。
    """
    out: List[ToolCall] = []
    if not isinstance(blocks, list):
        return out
    for blk in blocks:
        if not isinstance(blk, dict):
            continue
        if blk.get("type") != "tool_use":
            continue
        payload = blk.get("input") or {}
        if not isinstance(payload, dict):
            payload = {}
        name = str(blk.get("name") or "zoom_frame")
        if name == "search_modes":
            query = str(payload.get("query") or "").strip()
            if not query:
                continue
            try:
                k = int(payload.get("k") or 0)
            except (TypeError, ValueError):
                k = 0
            out.append(ToolCall(name=name, query=query, k=k))
            continue
        try:
            frame = int(payload.get("frame"))
        except (TypeError, ValueError):
            continue
        out.append(ToolCall(
            name=name,
            frame=frame,
            zoom=str(payload.get("zoom") or "whole"),
            about=str(payload.get("about") or ""),
            kind=str(payload.get("kind") or ""),
        ))
    return out


@dataclass
class LoopGuard:
    """**打转检测** —— 模型反复要同一块料时的止损闸。

    借鉴 Harness Engineering Guide 的 ``detect_loop``：
    连续 ``window`` 次请求**同一个**（工具, 帧, 区域）就停。

    没有这个闸，一次「看不清就再看看」的循环能把额度烧光
    （2026-10-31 刚踩过：Token Plan 额度用尽）。
    """

    window: int = 3
    _seen: List[str] = field(default_factory=list)

    def push(self, call: ToolCall) -> bool:
        """记一次请求。返回 True = 检测到打转，该停了。"""
        self._seen.append(call.key())
        if len(self._seen) < self.window:
            return False
        recent = self._seen[-self.window:]
        return len(set(recent)) == 1
