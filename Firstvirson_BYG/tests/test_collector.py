"""攻略采集的测试。

重点在**边界与拒绝**，不在 happy path：
采集器默认拒绝一切来源、白名单外一律拦下、确认入库必须显式表态 ——
这些是合规立场在代码里的强制点，比"能不能抓到页面"重要得多。
"""

from __future__ import annotations

import pytest

from app.adapters import registry as _registry  # noqa: F401  触发适配器注册
from app.collection.collector import (
    CollectionError,
    GuideCollector,
    HttpGuideCollector,
    SourceNotAllowed,
    TextCollector,
    collect_and_extract,
    confirm_collected,
    extract_main_text,
    normalize_text,
)
from app.collection.guide_sessions import MAX_SESSIONS, SessionStore


SAMPLE_HTML = """<html><head>
<title>云顶之弈 13.1 运营指南</title>
<script>var ads=1;</script><style>.x{}</style>
</head><body>
<nav>首页 攻略库 联系我们</nav>
<div class="sidebar">相关推荐：某某攻略</div>
<div class="content">
  <p>4-1 回合连胜达到 3 以上并且血量不低于 50 时，应该停手存钱。</p>
  <p>2 阶段金币超过 50 就不要乱花钱，存钱吃利息。</p>
  <p>5 阶段装备达到 4 件并且血量不超过 30 时，应该搜牌锁血。</p>
  <p>这波该不该追三要看对手站位，尽量不要梭哈。</p>
</div>
<div class="comment">评论区：楼主威强</div>
<footer>版权所有</footer>
</body></html>"""


# --------------------------------------------------------------------------
# 正文提取
# --------------------------------------------------------------------------


class TestMainTextExtraction:
    def test_title_extracted(self):
        title, _ = extract_main_text(SAMPLE_HTML)
        assert title == "云顶之弈 13.1 运营指南"

    def test_title_not_duplicated_into_body(self):
        """标题文本不该混进正文 —— 混进去就成了抽取器眼里的一句噪声。"""
        _, body = extract_main_text(SAMPLE_HTML)
        assert "运营指南" not in body

    @pytest.mark.parametrize(
        "noise",
        ["ads", "首页", "攻略库", "联系我们", "相关推荐", "评论区", "版权所有", "楼主威强"],
    )
    def test_noise_removed(self, noise):
        _, body = extract_main_text(SAMPLE_HTML)
        assert noise not in body

    def test_content_preserved(self):
        _, body = extract_main_text(SAMPLE_HTML)
        for key in ("连胜", "停手存钱", "搜牌锁血"):
            assert key in body

    def test_sentence_boundaries_kept(self):
        """块级标签必须转成换行，否则多句会粘成一句，阈值就抽不出来了。"""
        _, body = extract_main_text(SAMPLE_HTML)
        lines = [l for l in body.split("\n") if l.strip()]
        assert len(lines) >= 4

    def test_empty_input(self):
        assert extract_main_text("") == ("", "")

    def test_entities_unescaped(self):
        _, body = extract_main_text("<p>血量 &gt; 50 时存钱 &amp; 观望</p>")
        assert ">" in body
        assert "&amp;" not in body
        assert "&" in body


class TestNormalize:
    def test_adds_missing_punctuation(self):
        """攻略常用换行代替句号，缺标点抽取器就切不出句子。"""
        assert normalize_text("4-1 连胜 3 以上应该存钱") == "4-1 连胜 3 以上应该存钱。"

    def test_keeps_existing_punctuation(self):
        assert normalize_text("已经存钱了。") == "已经存钱了。"

    def test_collapses_whitespace(self):
        assert normalize_text("血量  50　时") == "血量 50 时。"

    def test_drops_blank_lines(self):
        assert normalize_text("a\n\n\nb") == "a。\nb。"


# --------------------------------------------------------------------------
# 来源白名单（合规核心）
# --------------------------------------------------------------------------


class TestSourceAllowlist:
    def test_default_rejects_everything(self):
        c = HttpGuideCollector()
        assert c.allowed_hosts == set()
        assert c.is_allowed("https://example.com") is False
        assert c.is_allowed("https://leagueoflegends.com/wiki") is False

    def test_fetch_blocked_by_default(self):
        c = HttpGuideCollector()
        with pytest.raises(SourceNotAllowed):
            c.fetch("https://example.com/guide")

    def test_explicit_host_allowed(self):
        c = HttpGuideCollector(allowed_hosts=["official.example.com"])
        assert c.is_allowed("https://official.example.com/guide")

    def test_subdomain_allowed(self):
        c = HttpGuideCollector(allowed_hosts=["example.com"])
        assert c.is_allowed("https://wiki.example.com/g")

    def test_lookalike_domain_rejected(self):
        """evilexample.com 不是 example.com —— 后缀匹配必须防住这类。"""
        c = HttpGuideCollector(allowed_hosts=["example.com"])
        assert c.is_allowed("https://notexample.com/g") is False
        assert c.is_allowed("https://example.com.evil.net/g") is False

    def test_case_insensitive(self):
        c = HttpGuideCollector(allowed_hosts=["Official.Example.COM"])
        assert c.is_allowed("https://official.example.com/g")

    def test_empty_host_rejected(self):
        c = HttpGuideCollector(allowed_hosts=["example.com"])
        assert c.is_allowed("not-a-url") is False

    def test_user_agent_declares_identity(self):
        """UA 必须表明身份而不是伪装浏览器。"""
        from app.collection.collector import USER_AGENT

        assert "GuideRuleExtractor" in USER_AGENT
        assert "Mozilla" not in USER_AGENT


# --------------------------------------------------------------------------
# TextCollector 与抽取一致性
# --------------------------------------------------------------------------


class TestTextCollector:
    def test_fetch_known_document(self):
        c = TextCollector({"g1": "4-1 连胜 3 以上应该存钱"})
        doc = c.fetch("g1", game="tft")
        assert "停手" not in doc.text
        assert "存钱" in doc.text

    def test_raw_text_treated_as_body(self):
        c = TextCollector()
        doc = c.fetch("2 阶段金币超过 50 就存钱", game="tft")
        assert doc.url == "2 阶段金币超过 50 就存钱"
        assert "存钱" in doc.text

    def test_content_hash_stable(self):
        c = TextCollector()
        a = c.fetch("同样的文本内容")
        b = TextCollector().fetch("同样的文本内容")
        assert a.content_hash == b.content_hash

    def test_content_hash_differs(self):
        c = TextCollector()
        assert c.fetch("文本甲").content_hash != c.fetch("文本乙").content_hash

    def test_sentence_count(self):
        c = TextCollector()
        doc = c.fetch("第一句。第二句！第三句？")
        assert doc.sentence_count == 3


class TestCollectorInterface:
    def test_abstract(self):
        with pytest.raises(TypeError):
            GuideCollector()


# --------------------------------------------------------------------------
# 采集 → 抽取 → 确认 的完整链路
# --------------------------------------------------------------------------


@pytest.fixture
def seeded_session(tmp_path):
    from app import db as dbmod
    from app.db import get_engine, session_scope
    from app.rules.store import latest_version, load_seed_library

    dbmod._engine = None
    dbmod._SessionFactory = None
    get_engine(str(tmp_path / "collector_test.db"))
    with session_scope() as s:
        if not latest_version(s, "tft"):
            load_seed_library(s, "tft")
    with session_scope() as s:
        nested = s.begin_nested()
        try:
            yield s
        finally:
            if nested.is_active:
                nested.rollback()


class TestCollectExtractConfirm:
    def _collect(self, session, html=SAMPLE_HTML, version="13.2"):
        collector = TextCollector({"guide://x": html})
        return collect_and_extract(
            collector, session,
            source="guide://x", game="tft", version=version,
            title="测试攻略",
        )

    def test_candidates_extracted(self, seeded_session):
        with seeded_session as s:
            result = self._collect(s)
            assert result.candidates, "示例攻略应抽出候选"
            assert len(result.rejected_subjective) >= 1

    def test_collection_ingests_nothing(self, seeded_session):
        """采集本身绝不写模式 —— 未经确认的候选进不了操作模式库。"""
        from app.rules.store import get_rules, latest_version

        with seeded_session as s:
            self._collect(s)
            # tft 的库是空的（只保留模块、资料后补），所以这里断言的是
            # 「采集没有凭空造出一个 13.2」—— 而不是因为基线版本恰好存在
            assert latest_version(s, "tft") != "13.2"
            assert len(get_rules(s, "tft", "13.2")) == 0

    def test_source_registered_without_body(self, seeded_session):
        """来源元信息入库，但攻略原文不留存。"""
        from app.db import GuideSource

        with seeded_session as s:
            result = self._collect(s)
            assert result.source_id is not None
            row = s.get(GuideSource, result.source_id)
            assert row.url == "guide://x"
            assert row.title == "测试攻略"
            body = (row.note or "")
            assert "连胜达到 3 以上" not in body   # 原文片段不入库

    def test_confirm_imports_only_accepted(self, seeded_session):
        from app.rules.store import get_rules

        with seeded_session as s:
            result = self._collect(s)
            titles = [c.title for c in result.candidates][:2]
            assert titles, "需要至少两条候选才能验证选择性入库"

            outcome = confirm_collected(s, result, titles)
            assert outcome["imported"]
            assert outcome["skipped"] == len(result.candidates) - len(titles)

            rules = {r.id for r in get_rules(s, "tft", "13.2")}
            assert set(outcome["imported"]) <= rules

    def test_empty_accept_imports_nothing(self, seeded_session):
        from app.rules.store import get_rules

        with seeded_session as s:
            result = self._collect(s)
            outcome = confirm_collected(s, result, [])
            assert outcome["imported"] == []
            assert "未确认" in outcome["message"]
            assert len(get_rules(s, "tft", "13.2")) == 0

    def test_subjective_lines_never_become_candidates(self, seeded_session):
        """含主观词的句子不该出现在候选列表里 —— 抽取阶段就该拦下。

        这是「自动采集不绕过审核」的第一环：不靠人不去勾选，而是系统
        根本不把这种条目递到人面前。
        """
        with seeded_session as s:
            result = self._collect(s)
            titles = [c.title for c in result.candidates]
            assert not any("该不该" in t or "尽量" in t for t in titles), \
                "主观句不应成为候选"
            assert all(not c.subjective_markers for c in result.candidates)
            assert any("该不该" in x for x in result.rejected_subjective)

    def test_subjective_text_is_reported(self, seeded_session):
        """被拦下的主观句要如实告知，否则采集方会以为漏抓了内容。"""
        with seeded_session as s:
            result = self._collect(s)
            assert len(result.rejected_subjective) >= 1
            assert any("梭哈" in x for x in result.rejected_subjective)

    def test_rule_id_stable_across_collections(self, seeded_session):
        """同一句攻略重复采集必须得到同一个 id，否则规则库会积重复条目。"""
        with seeded_session as s:
            first = self._collect(s)
            second = self._collect(s)
            ids_a = confirm_collected(s, first, [first.candidates[0].title])
            ids_b = confirm_collected(s, second, [second.candidates[0].title])
            assert ids_a["imported"] == ids_b["imported"]


# --------------------------------------------------------------------------
# 会话存储
# --------------------------------------------------------------------------


class TestSessionStore:
    def _result(self):
        from app.collection.collector import CollectionResult, GuideDocument

        return CollectionResult(document=GuideDocument(url="u", title="t", text="x"))

    def test_put_and_get(self):
        s = SessionStore()
        s.put("a", self._result())
        assert s.get("a") is not None

    def test_missing_returns_none(self):
        assert SessionStore().get("nope") is None

    def test_drop(self):
        s = SessionStore()
        s.put("a", self._result())
        assert s.drop("a") is True
        assert s.drop("a") is False

    def test_expiry(self):
        s = SessionStore(ttl=0.01)
        s.put("a", self._result())
        import time

        time.sleep(0.05)
        assert s.get("a") is None

    def test_overflow_evicts_oldest(self):
        s = SessionStore()
        import time

        for i in range(MAX_SESSIONS + 5):
            s.put(f"s{i}", self._result())
            time.sleep(0.001)
        assert s.size() <= MAX_SESSIONS

    def test_no_content_persisted_to_disk(self, tmp_path):
        """会话只在内存 —— 放进数据库会让人误以为系统在长期保存攻略原文。"""
        s = SessionStore()
        s.put("a", self._result())
        assert list(tmp_path.iterdir()) == []
