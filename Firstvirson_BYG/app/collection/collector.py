"""攻略来源采集。

立项设计里的合规立场，在这里落成代码约束：

  1. **只存结构化规则，不存原文全文。** 采集到的正文在抽取成候选后即丢弃，
     数据库里只留出处元信息与必要的短引用。系统是分析方，不是内容生产方。
  2. **自动采集不绕过审核。** 采集只负责「拿到文本」—— 拿到之后走的是
     和用户手工粘贴**完全相同**的抽取 / 准入 / 人工确认流程。这样"从网上
     抓来的"和"用户自己贴的"在规则库里没有任何区别，也就不存在一条
     后门让未经确认的条目混进去。
  3. **来源可追溯、可限制。** 每条规则都能指回具体 URL 与章节；
     默认只允许注册来源，非注册来源必须显式声明。

正文提取用的是启发式而非 DOM 级精确解析 —— 见 :func:`extract_main_text`
的说明。这里刻意不做「完美还原网页」，因为抽取器本来按句工作，
导航残留、广告文案会被自然过滤掉。为一个不需要的精度引入 DOM 解析依赖
反而不划算。
"""

from __future__ import annotations

import abc
import hashlib
import html as html_mod
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set

#: 采集时使用的 User-Agent。
#:
#: 明确表明身份而不是伪装成浏览器 —— 对方站点有权知道请求来自什么，
#: 这既是基本礼貌，也是让站点管理员在出问题时能直接联系到人的前提。
USER_AGENT = (
    "GuideRuleExtractor/0.1 (personal game-coach analysis tool; "
    "contact: local-user)"
)

#: 请求超时（秒）。设上限是为了不让一次采集把流程卡死。
DEFAULT_TIMEOUT = 20.0

#: 单次抓取正文长度上限。攻略页面动辄几万字，超过这个量级多半是
#: 抓错了页面（列表页、登录页），与其让抽取器慢慢处理垃圾，不如早失败。
MAX_CONTENT_CHARS = 200_000

#: 默认允许的来源域名。空集合表示「不限制」，但调用方需要显式传参确认。
DEFAULT_ALLOWED_HOSTS: Set[str] = set()


class CollectionError(RuntimeError):
    pass


class SourceNotAllowed(CollectionError):
    """来源不在允许列表内。"""


# --------------------------------------------------------------------------
# 数据契约
# --------------------------------------------------------------------------


@dataclass
class GuideDocument:
    """采集到的攻略文档。"""

    url: str
    title: str
    text: str
    game: str = ""
    kind: str = "community"          # official / community / user
    fetched_at: str = ""
    #: 正文指纹。同一篇攻略重复采集能识别出来，避免重复入库。
    content_hash: str = ""
    #: 采集过程中的提示（截断、字符集猜测失败等）
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = hashlib.sha256(
                self.text.encode("utf-8")
            ).hexdigest()[:16]
        if not self.fetched_at:
            self.fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    @property
    def sentence_count(self) -> int:
        return len([s for s in re.split(r"[。！？；\n]", self.text) if s.strip()])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "game": self.game,
            "kind": self.kind,
            "fetched_at": self.fetched_at,
            "content_hash": self.content_hash,
            "chars": len(self.text),
            "sentences": self.sentence_count,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------
# 正文提取（启发式）
# --------------------------------------------------------------------------

_DROP_TAGS = re.compile(
    r"<(script|style|noscript|iframe|svg|nav|header|footer|aside|form|head)\b[^>]*>"
    r".*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_COMMENTS = re.compile(r"<!--.*?-->", re.DOTALL)
#: 常见的导航/侧栏/广告容器 class —— 攻略站点的噪音几乎都出在这几类里
_NOISE_CLASS = re.compile(
    r'<[a-z][^>]*(?:class|id)\s*=\s*["\'][^"\']*'
    r"(?:nav|menu|sidebar|side-bar|footer|header|comment|advert|ads?|banner|"
    r"recommend|related|breadcrumb|pagination|toolbar|share|qrcode)"
    r'[^"\']*["\'][^>]*>.*?</[a-z]+>',
    re.IGNORECASE | re.DOTALL,
)
_TITLE_TAG = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_BLOCK_END = re.compile(r"</(p|div|li|tr|h[1-6]|section|article|br)\s*>", re.IGNORECASE)
_INLINE_TAG = re.compile(
    r"<(br|hr)\s*/?>", re.IGNORECASE
)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t　]+")


def extract_main_text(source: str) -> tuple:
    """从 HTML 提取正文，返回 ``(标题, 正文)``。

    启发式实现，不追求完美：先丢掉脚本与导航类容器，再去标签、还原实体、
    把块级标签结束处转成换行以保住句子边界。

    为什么这样够用：抽取器是**按句**工作的，一条导航项「首页/攻略/攻略库」
    既匹配不到动作词也匹配不到数值阈值，会被自然过滤。真正需要保住的是
    正文句子的**边界**（否则会把多句拼成一句，抽不出阈值），所以块级标签
    必须转成换行 —— 这是这里唯一不能省的步骤。
    """
    if not source:
        return "", ""

    text = source
    text = _COMMENTS.sub(" ", text)

    # 标题必须在剥离 head 之前取 —— 否则 <title> 文本会被当成正文，
    # 在抽取器里变成一句噪声
    title_match = _TITLE_TAG.search(source)
    title = ""
    if title_match:
        title = html_mod.unescape(_TAG.sub("", title_match.group(1))).strip()

    text = _DROP_TAGS.sub(" ", text)
    text = _NOISE_CLASS.sub(" ", text)

    text = _INLINE_TAG.sub("\n", text)
    text = _BLOCK_END.sub("\n", text)
    text = _TAG.sub("", text)
    text = html_mod.unescape(text)

    lines = [_WS.sub(" ", line).strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return title, "\n".join(lines)


def normalize_text(text: str) -> str:
    """把纯文本规整成抽取器友好的形态：补标点、压空白。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_WS.sub(" ", line).strip() for line in text.split("\n")]
    out: List[str] = []
    for line in lines:
        if not line:
            continue
        # 很多攻略用换行代替句号，缺了标点抽取器就切不出句子
        if line[-1] not in "。！？；.!?":
            line += "。"
        out.append(line)
    return "\n".join(out)


# --------------------------------------------------------------------------
# 采集器
# --------------------------------------------------------------------------


class GuideCollector(abc.ABC):
    """采集器接口。替换抓取方式（HTTP / 本地文件 / 用户粘贴）不影响下游。"""

    collector_id: str = ""

    @abc.abstractmethod
    def fetch(self, source: str, game: str = "", kind: str = "community") -> GuideDocument:
        """取回一篇攻略的正文。"""


class TextCollector(GuideCollector):
    """直接接收文本。

    用途有两个：用户手工粘贴攻略，以及测试。走的是和 HTTP 采集**完全相同**
    的下游路径，所以「用户贴的」和「网上抓的」在规则库里没有区别。

    输入若是 HTML，会先提取正文再交给下游。这不是额外功能而是必需的 ——
    用户从网页复制一段带标签的内容粘进来是常事，若不剥离，
    ``<p>4-1 回合连胜…`` 这样的标签会一路进到规则标题里。
    """

    collector_id = "text"

    def __init__(self, documents: Optional[Dict[str, str]] = None) -> None:
        self._documents = documents or {}

    def add(self, url: str, text: str) -> None:
        self._documents[url] = text

    def fetch(self, source: str, game: str = "", kind: str = "community") -> GuideDocument:
        known = source in self._documents
        body = self._documents[source] if known else source

        title = source if known else "手工录入的攻略"
        if _looks_like_html(body):
            title, body = extract_main_text(body)

        normalized = normalize_text(body)
        # url 保留 source：它是来源标识，注册进 guide_sources 时需要它
        # 才能让规则指回具体出处。TextCollector 只是抓取方式不同，
        # 不该让下游拿到一个空的 url。
        return GuideDocument(
            url=source,
            title=title,
            text=normalized,
            game=game,
            kind=kind,
        )


def _looks_like_html(text: str) -> bool:
    """粗判是否为 HTML。

    看是否有标签即可 —— 攻略正文里几乎不会出现 ``<p>`` 这类结构，
    误判风险远低于漏判（漏判会让标签污染整条规则）。
    """
    return bool(re.search(r"<[a-zA-Z/!][^>]*>", text[:4000]))


class HttpGuideCollector(GuideCollector):
    """HTTP 抓取。

    只对**明确允许的来源**工作。默认允许列表为空 —— 也就是说必须由调用方
    显式声明要抓哪些站点。理由：系统定位是分析方，它有权主动抓什么不该
    是默认值，写成默认拒绝才能保证这一点不会在赶工期时被忘掉。
    """

    collector_id = "http"

    def __init__(
        self,
        allowed_hosts: Optional[Iterable[str]] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_chars: int = MAX_CONTENT_CHARS,
        user_agent: str = USER_AGENT,
        follow_redirects: bool = True,
    ) -> None:
        self.allowed_hosts: Set[str] = {
            h.strip().lower() for h in (allowed_hosts or ()) if h.strip()
        }
        self.timeout = timeout
        self.max_chars = max_chars
        self.user_agent = user_agent
        self.follow_redirects = follow_redirects

    # -- 合规 ---------------------------------------------------------------

    def is_allowed(self, url: str) -> bool:
        from urllib.parse import urlparse

        host = (urlparse(url).hostname or "").lower()
        if not host:
            return False
        if not self.allowed_hosts:
            return False      # 默认拒绝
        # 允许子域名（example.com 覆盖 wiki.example.com）
        return any(host == h or host.endswith("." + h) for h in self.allowed_hosts)

    def _guard(self, url: str) -> None:
        if not self.is_allowed(url):
            raise SourceNotAllowed(
                f"来源 {url} 不在允许列表内。"
                f"当前允许：{sorted(self.allowed_hosts) or '（空，默认全部拒绝）'}"
            )

    # -- 抓取 ---------------------------------------------------------------

    def fetch(self, source: str, game: str = "", kind: str = "community") -> GuideDocument:
        self._guard(source)

        import httpx

        notes: List[str] = []
        with httpx.Client(
            timeout=self.timeout,
            follow_redirects=self.follow_redirects,
            headers={"User-Agent": self.user_agent},
        ) as client:
            resp = client.get(source)
            resp.raise_for_status()

        # 声明的字符集经常是错的（国内站点写 gbk 实际 utf-8 的不少），
        # 这种情况 httpx 会抛异常；退回 utf-8 再试 gb18030 覆盖中文场景
        try:
            page = resp.text
        except (UnicodeDecodeError, LookupError):
            page = resp.content.decode("utf-8", errors="replace")
            notes.append("字符集声明异常，已按 utf-8 容错解码")

        title, body = extract_main_text(page)
        body = normalize_text(body)

        if len(body) > self.max_chars:
            body = body[: self.max_chars]
            notes.append(f"正文超过 {self.max_chars} 字符，已截断")

        if not body.strip():
            raise CollectionError(
                f"{source} 未提取到正文 —— 可能是 JS 渲染页面或抓到了列表页"
            )

        return GuideDocument(
            url=source,
            title=title or source,
            text=body,
            game=game,
            kind=kind,
            notes=notes,
        )


# --------------------------------------------------------------------------
# 编排：采集 → 抽取 → 审核
# --------------------------------------------------------------------------


@dataclass
class CollectionResult:
    """一次采集的完整交代。"""

    document: GuideDocument
    candidates: List[Any] = field(default_factory=list)     # RuleCandidate
    rejected_subjective: List[str] = field(default_factory=list)
    source_id: Optional[int] = None

    def to_dict(self, include_candidates: bool = True) -> Dict[str, Any]:
        out = {
            "document": self.document.to_dict(),
            "candidate_count": len(self.candidates),
            "rejected_subjective": self.rejected_subjective,
            "source_id": self.source_id,
        }
        if include_candidates:
            out["candidates"] = [c.to_dict() for c in self.candidates]
        return out


def collect_and_extract(
    collector: GuideCollector,
    session,
    source: str,
    game: str,
    version: str,
    title: str = "",
    kind: str = "community",
    min_confidence: float = 0.4,
    register_source: bool = True,
) -> CollectionResult:
    """采集 → 抽取候选，**不入库规则**。

    与手工粘贴走同一条路径：抽取器、主观词拦截、置信度门槛完全一致。
    采集到候选后函数就停 —— 入库必须经过 :func:`confirm_candidates`，
    也就是必须经过人的判断。这条路径上没有旁路。
    """
    from ..adapters.base import get_adapter
    from ..rules.extractor import _detect_subjective, extract_candidates, split_sentences

    adapter = get_adapter(game)
    doc = collector.fetch(source, game=game, kind=kind)

    candidates = extract_candidates(
        text=doc.text,
        game=game,
        version=version,
        known_facts=adapter.known_facts(),
        guide_title=title or doc.title,
        guide_url=doc.url,
        min_confidence=min_confidence,
    )

    result = CollectionResult(document=doc, candidates=candidates)
    result.rejected_subjective = [
        s for s in split_sentences(doc.text) if _detect_subjective(s)
    ]

    if register_source:
        # note 只放一行统计。以前这里塞了整个 document.to_dict()，
        # 于是 url、content_hash、字数在库里重复存了两遍，而且 note
        # 变成了 Python dict 的字符串形式 —— 既脏又没法查询。
        result.source_id = _register_source(
            session, game=game, title=title or doc.title,
            url=doc.url, kind=kind,
            note=f"抽出 {len(candidates)} 条候选"
                 f"（{doc.sentence_count} 句 / {len(doc.text)} 字）",
        )

    return result


def confirm_collected(
    session,
    collected: CollectionResult,
    accept: List[str],
) -> Dict[str, Any]:
    """对**已采集**的候选做人工确认入库。

    ``accept`` 为空表示「一条都不要」，这是刻意设计：调用方必须显式表达
    判断，不能因为忘了传就把候选全部放行。

    这里刻意不重新抓取正文：规则 id 由标题稳定化生成，采集阶段的抽取结果
    已在手边。「确认入库不需要、也不应该依赖原文还在」正是「只存结构化
    规则」在流程上的体现 —— 库里没有攻略全文，但这不妨碍把规则入库。
    """
    from ..adapters.base import get_adapter
    from ..rules.store import import_rules

    game = collected.document.game
    adapter = get_adapter(game)
    facts = adapter.known_facts()
    action_facts = adapter.known_action_facts()

    accepted_titles = set(accept)
    if not accepted_titles:
        return {"imported": [], "failed": [], "skipped": 0,
                "message": "未确认任何候选，本次不入库"}

    prefix = game.upper()[:3]
    imported: List[str] = []
    failed: List[Dict[str, Any]] = []
    skipped = 0

    for cand in collected.candidates:
        if cand.title not in accepted_titles:
            skipped += 1
            continue
        rule_id = _stable_rule_id(prefix, cand)
        rule = cand.to_rule(rule_id)
        sub = import_rules(
            session, [rule], facts, strict=False, action_facts=action_facts
        )
        if sub.accepted:
            imported.append(rule_id)
        else:
            failed.append(
                {"rule_id": rule_id, "title": cand.title, "reasons": sub.rejected}
            )

    return {
        "imported": imported,
        "failed": failed,
        "skipped": skipped,
        "message": (
            f"已入库 {len(imported)} 条规则"
            if imported
            else "没有规则通过准入筛选"
        ),
    }


def _stable_rule_id(prefix: str, cand) -> str:
    """由候选标题生成稳定规则 id。

    与 :func:`app.collection.guide._make_rule_id` 同构 —— 同一句攻略
    重复采集会得到同一个 id，不会产生重复条目。
    """
    digest = hashlib.sha1(cand.title.encode("utf-8")).hexdigest()[:4]
    cat = _slug(cand.category or "misc").upper()[:8]
    return f"{prefix}-{cat}-{digest}"


def _slug(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum():
            keep.append(ch)
        elif ch in " -_/":
            keep.append("-")
    return "".join(keep).strip("-") or "misc"


def _register_source(session, game, title, url, kind, note) -> int:
    from ..rules.store import register_guide_source

    src = register_guide_source(
        session, game=game, title=title, url=url, kind=kind, note=str(note)[:2000]
    )
    return src.id
