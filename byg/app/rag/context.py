"""**检索的上下文装配**：把召回的块变成模型能判的清单条目。

为什么要有这一层
----------------
检索层 (:mod:`app.rag.retriever`) 召回的是**块**（模式级 + 分支级），
但模型要判的是**条目**（一条模式 + 它的分支）。两者形状不一样，
中间这一步没人做的话，模型看到的就会是「一堆文字」而不是「可以逐条裁决的尺子」。

所以这里做三件事：

1. **块 → 清单条目**：用 :func:`app.rules.checklist.entry_from_rule` 拼装，
   与窗口清单**完全同形** —— 检索回来的条目因此能过同一道接地闸
   （:func:`app.collection.vlm_review.parse_review`），**不开旁路**。
2. **记住命中的是第几支**：分支块命中意味着「这一条里最相关的是第 N 支」。
   这条提示要写进提示词，模型才知道该指认哪一支。
3. **渲染成提示词片段**：**逐字原文**，不改写（改写过的转述模型对不上）。

时间窗、英雄、分路这些**硬约束**已经在 :meth:`app.rag.retriever.Retriever.search`
里过滤过了，这里不再放宽 —— 放宽就是拿「10 分钟才该做的事」去纠正 3 分钟的人。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .retriever import RetrievalResult, Retriever

#: 一次检索最多带回几条（**模式**数，不是块数）。
#:
#: 太多会把提示词撑爆，模型也判不完；太少则「查了等于没查」。
DEFAULT_TOP_K = 5

#: 提示词里提示「命中第几支」用的标记。
_BRANCH_HINT = "（这一段最相关的是第 %d 支）"


class ModeSearchService:
    """模式库的检索服务（带上下文装配）。"""

    def __init__(self, game: str = "hok",
                 retriever: Optional[Retriever] = None):
        self.game = game
        self._retriever = retriever
        self._nodes: Optional[Dict[Any, Dict[str, List[str]]]] = None
        self._ensured = False

    def _ensure_library(self) -> None:
        """库是空的 → 现场建一份。

        什么时候会空？刚装好的程序第一次跑 —— 库里还没有文档。
        这时从**应用正在读的那份模式库 JSON** 现场建（文档 + 分块 + 向量），
        建在用户数据目录的 ``mode_library.db`` 里，不碰用户的 ``app.db``。

        建不出来就**降级**（检索没有结果，判定仍走原清单），不报错。
        """
        if self._ensured:
            return
        self._ensured = True
        try:
            from . import bootstrap

            bootstrap.ensure()
        except Exception:                              # noqa: BLE001
            pass

    # -- 检索 ------------------------------------------------------------

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._ensure_library()
            self._retriever = Retriever()
        return self._retriever

    def _node_map(self) -> Dict[Any, Dict[str, List[str]]]:
        """节点 → 该时段攻略要求的全部打法（跨动作事实）。

        与窗口清单**同一份算法**（:func:`app.rules.checklist.node_action_map`）——
        检索回来的条目也要过「玩家做了节点认可的事就不纠正」那道闸。
        """
        if self._nodes is None:
            try:
                from ..rules.checklist import node_action_map

                self._nodes = node_action_map(self.game)
            except Exception:                          # noqa: BLE001
                self._nodes = {}
        return self._nodes

    def search(self, query: str, *, lane: str = "", hero: str = "",
               lo: int = 0, hi: int = 60,
               top_k: int = DEFAULT_TOP_K) -> RetrievalResult:
        """检索与 ``query`` 相关的打法。

        过滤条件（时间窗 / 英雄 / 分路）是**硬约束**，与窗口清单一致。

        **版本不过滤** —— 判定层用哪一版，检索层就跟哪一版；
        写死一个版本号会在换库之后一条都召不回（见
        :meth:`app.rag.retriever.Retriever.search`）。
        """
        return self.retriever.search(query, game=self.game, lane=lane,
                                     hero=hero, lo=lo, hi=hi, top_k=top_k)

    # -- 装配 ------------------------------------------------------------

    def entries(self, result: RetrievalResult) -> List[Dict[str, Any]]:
        """把召回的块变成**清单条目**（与窗口清单同形）。

        同一条模式可能被召回好几块（模式级 + 多个分支级）——
        这里按 ``mode_id`` 去重，保留**命中分支号最大**的那条（分支级命中
        比模式级命中更有指向性）。
        """
        from ..rules.checklist import entry_from_rule
        from ..rules.checklist import lane_of as _lane_of

        nodes = self._node_map()
        by_id: Dict[str, Dict[str, Any]] = {}
        for hit in (result.hits or []):
            mode_id = str(hit.mode_id or "")
            if not mode_id:
                continue
            payload = hit.payload or {}
            if not isinstance(payload, dict) or not payload:
                continue
            key = (_lane_of(payload), tuple(hit.window or ()))
            node = nodes.get(key, {})
            entry = entry_from_rule(payload, node=node)
            if entry is None:
                continue
            prev = by_id.get(entry["id"])
            if prev is None or int(hit.branch_no or 0) > int(prev.get("_branch") or 0):
                entry["_branch"] = int(hit.branch_no or 0)
                entry["_score"] = float(hit.score or 0.0)
                entry["_source"] = hit.source or ""
                by_id[entry["id"]] = entry
        out = list(by_id.values())
        out.sort(key=lambda e: -float(e.get("_score") or 0.0))
        return out

    @staticmethod
    def prompt_block(entries: Sequence[Dict[str, Any]]) -> str:
        """渲染成提示词片段（**逐字原文** + 命中分支提示）。"""
        from ..rules.checklist import checklist_prompt_block

        if not entries:
            return ""
        plain = [{k: v for k, v in e.items() if not k.startswith("_")}
                 for e in entries]
        block = checklist_prompt_block(plain)
        hints: List[str] = []
        for entry in entries:
            branch_no = int(entry.get("_branch") or 0)
            if branch_no >= 1:
                hints.append("【%s】%s" % (entry.get("id") or "",
                                          _BRANCH_HINT % branch_no))
        if hints:
            block += "\n（我们检索时，下面这些条目的某一支与你的描述最接近，"
            block += "但**仍然是你的判断**：\n" + "\n".join(hints) + "）"
        return block
