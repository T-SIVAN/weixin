import json
import zipfile
from io import BytesIO

import pytest

from weixin_lite import batch_analyze
from weixin_lite import llm as llm_module
from weixin_lite.article_analysis import ANALYSIS_PROMPT_VERSION, analysis_cache_key, analyze_paper, build_analysis_prompt
from weixin_lite.downloader import download_open_access
from weixin_lite.exporter import article_html, project_zip, unavailable_dois_csv
from weixin_lite.figure_analysis import analyze_confirmed_figures
from weixin_lite.generator import ArticleGenerationError, build_prompt, generate_article, markdown_to_wechat_html
from weixin_lite.llm import _parse_retry_after, call_openai_compatible, default_api_key, default_base_url, default_model
from weixin_lite.models import AnalysisClaim, BatchProject, FigureAnalysis, PaperAnalysis, PaperInput, generation_candidate_papers, generation_ready_papers, unavailable_papers
from weixin_lite.pdf_reader import PdfContent, extract_figure_legends, extract_numeric_evidence
from weixin_lite.search import (
    build_plain_search_queries,
    build_keyword_query,
    crossref_publication_date,
    crossref_year,
    federated_search,
    is_synthetic_biology_relevant,
    relevance_score,
    year_from,
)
from weixin_lite.llm import LLMError
from weixin_lite.translate import translate_records
from weixin_lite.wechat_publish import WechatDraftConfig, publish_draft


def test_keyword_query_expands_simple_keywords():
    query = build_keyword_query(["TdT", "酶促DNA合成"])

    assert "terminal deoxynucleotidyl transferase" in query
    assert "enzymatic DNA synthesis" in query
    assert "synthetic biology" in query
    assert " AND " in query


def test_plain_search_queries_expand_chinese_keywords_without_boolean_syntax():
    queries = build_plain_search_queries(["酶促DNA合成", "合成生物"])

    assert "enzymatic DNA synthesis" in queries
    assert "synthetic biology" in queries
    assert all(" AND " not in item and " OR " not in item for item in queries)


def test_crossref_receives_plain_queries(monkeypatch):
    seen: list[str] = []
    def fake_crossref(query, limit, since_days=None):
        seen.append(query)
        return []

    monkeypatch.setattr("weixin_lite.search.search_crossref", fake_crossref)

    federated_search(["TdT"], limit=5, sources=["Crossref"])

    assert seen
    assert all(" AND " not in query and " OR " not in query for query in seen)


def test_synthetic_biology_relevance_filters_off_topic_results():
    relevant = PaperInput(
        title_en="Engineered microbes for high-yield biosynthesis of terpenoids",
        abstract_en="Synthetic biology and metabolic engineering improve microbial cell factory production.",
        doi="10.1000/synbio",
    )
    off_topic = PaperInput(
        title_en="Engineered microbes detected in patient infection diagnosis",
        abstract_en="A clinical study reports prognosis and therapy outcomes in patients.",
        doi="10.1000/clinical",
    )

    assert is_synthetic_biology_relevant(relevant, ["engineered microbes"])
    assert relevance_score(relevant, ["engineered microbes"]) > relevance_score(off_topic, ["engineered microbes"])
    assert not is_synthetic_biology_relevant(off_topic, ["engineered microbes"])


def test_year_parser_rejects_future_years():
    assert year_from("published 2024") == "2024"
    assert year_from("published 2027") == ""
    assert year_from({"date-parts": [[2050, 1, 1]]}) == ""
    assert crossref_year({"published-online": {"date-parts": [[2025, 6, 1]]}}) == "2025"


def test_crossref_publication_date_ignores_record_created_date():
    item = {
        "created": {"date-parts": [[2026, 7, 1]]},
        "indexed": {"date-parts": [[2026, 7, 2]]},
    }

    assert crossref_publication_date(item) == ("", "")
    assert crossref_year(item) == ""


def test_crossref_publication_date_prefers_published_fields():
    item = {
        "created": {"date-parts": [[2026, 7, 1]]},
        "issued": {"date-parts": [[2024, 12]]},
        "published-online": {"date-parts": [[2025, 1, 5]]},
    }

    assert crossref_publication_date(item) == ("2025-01-05", "published-online")
    assert crossref_year(item) == "2025"


def test_translation_without_key_marks_pending():
    paper = PaperInput(title_en="A test title", abstract_en="A test abstract.")

    report = translate_records([paper])

    assert paper.title_zh == ""
    assert paper.translation_status == "pending"
    assert paper.abstract_zh == ""
    assert report.pending_count == 1
    assert report.errors


def test_translation_populates_title_only(monkeypatch, tmp_path):
    seen_payloads: list[str] = []

    def fake_call(**kwargs):
        seen_payloads.append(kwargs["user_prompt"])
        return '[{"title_zh": "中文标题", "abstract_zh": "中文摘要"}]'

    monkeypatch.setattr("weixin_lite.translate.call_openai_compatible", fake_call)
    paper = PaperInput(title_en="English title", abstract_en="English abstract.")

    report = translate_records([paper], api_key="test-key", batch_size=1, delay_seconds=0, cache_path=tmp_path / "cache.json")

    assert report.translated_count == 1
    assert paper.title_zh == "中文标题"
    assert paper.translation_status == "translated"
    assert paper.abstract_zh == ""
    assert "abstract_en" not in seen_payloads[0]


def test_translation_empty_records_does_not_call_llm(monkeypatch):
    def fake_call(**kwargs):
        raise AssertionError("LLM should not be called for empty records")

    monkeypatch.setattr("weixin_lite.translate.call_openai_compatible", fake_call)

    report = translate_records([], api_key="test-key")

    assert report.ok
    assert report.records == []
    assert report.translated_count == 0


def test_translation_uses_cache_without_calling_llm(monkeypatch, tmp_path):
    cache_path = tmp_path / "translation_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "doi:10.1000/cache": {
                    "title_en": "Cached title",
                    "title_zh": "缓存标题",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def fake_call(**kwargs):
        raise AssertionError("LLM should not be called for cached records")

    monkeypatch.setattr("weixin_lite.translate.call_openai_compatible", fake_call)
    paper = PaperInput(title_en="Cached title", doi="10.1000/cache")

    report = translate_records([paper], api_key="test-key", cache_path=cache_path)

    assert paper.title_zh == "缓存标题"
    assert paper.translation_status == "cached"
    assert report.cached_count == 1


def test_translation_existing_title_normalizes_failed_status(monkeypatch, tmp_path):
    def fake_call(**kwargs):
        raise AssertionError("LLM should not be called for already translated records")

    monkeypatch.setattr("weixin_lite.translate.call_openai_compatible", fake_call)
    paper = PaperInput(title_en="English title", title_zh="中文标题", translation_status="failed")

    report = translate_records([paper], api_key="test-key", cache_path=tmp_path / "cache.json")

    assert paper.translation_status == "translated"
    assert report.skipped_count == 1


def test_translation_cache_save_merges_existing_entries(monkeypatch, tmp_path):
    cache_path = tmp_path / "translation_cache.json"
    cache_path.write_text(
        json.dumps({"doi:10.1000/old": {"title_zh": "旧缓存"}}, ensure_ascii=False),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "weixin_lite.translate.call_openai_compatible",
        lambda **kwargs: '[{"title_zh": "新标题"}]',
    )
    paper = PaperInput(title_en="New title", doi="10.1000/new")

    report = translate_records([paper], api_key="test-key", cache_path=cache_path, delay_seconds=0)
    data = json.loads(cache_path.read_text(encoding="utf-8"))

    assert report.translated_count == 1
    assert data["doi:10.1000/old"]["title_zh"] == "旧缓存"
    assert data["doi:10.1000/new"]["title_zh"] == "新标题"


def test_translation_cache_save_failure_does_not_discard_translation(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "weixin_lite.translate.call_openai_compatible",
        lambda **kwargs: '[{"title_zh": "中文标题"}]',
    )

    def fail_save(*args, **kwargs):
        raise OSError("disk is busy")

    monkeypatch.setattr("weixin_lite.translate.save_translation_cache", fail_save)
    paper = PaperInput(title_en="English title")

    report = translate_records([paper], api_key="test-key", cache_path=tmp_path / "cache.json", delay_seconds=0)

    assert paper.title_zh == "中文标题"
    assert report.translated_count == 1
    assert any("Translation cache save failed" in error for error in report.errors)


def test_translation_retries_429(monkeypatch, tmp_path):
    calls = {"count": 0}
    monkeypatch.setattr("weixin_lite.translate.time.sleep", lambda seconds: None)

    def fake_call(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise LLMError("rate limited", status_code=429, retry_after=0, transient=True)
        return '[{"title_zh": "重试后成功"}]'

    monkeypatch.setattr("weixin_lite.translate.call_openai_compatible", fake_call)
    paper = PaperInput(title_en="Retry title")

    report = translate_records([paper], api_key="test-key", delay_seconds=0, cache_path=tmp_path / "cache.json")

    assert calls["count"] == 2
    assert paper.title_zh == "重试后成功"
    assert report.translated_count == 1


def test_translation_splits_failed_batch(monkeypatch, tmp_path):
    seen_prompts: list[str] = []

    def fake_call(**kwargs):
        prompt = kwargs["user_prompt"]
        seen_prompts.append(prompt)
        if prompt.startswith("["):
            payload = json.loads(prompt)
            if len(payload) > 1:
                raise LLMError("server overloaded", status_code=500, transient=True)
            return '[{"title_zh": "单篇成功"}]'
        if prompt == "First":
            return '{"title_zh": "第一篇成功"}'
        if prompt == "Second":
            return "第二篇成功"
        raise AssertionError(f"Unexpected prompt: {prompt}")

    monkeypatch.setattr("weixin_lite.translate.call_openai_compatible", fake_call)
    papers = [PaperInput(title_en="First"), PaperInput(title_en="Second")]

    report = translate_records(papers, api_key="test-key", batch_size=2, delay_seconds=0, max_retries=0, cache_path=tmp_path / "cache.json")

    assert seen_prompts == ['[{"title_en": "First"}, {"title_en": "Second"}]', "First", "Second"]
    assert [paper.title_zh for paper in papers] == ["第一篇成功", "第二篇成功"]
    assert report.failed_count == 0
    assert report.ok


def test_translation_mismatched_json_falls_back_to_single_title(monkeypatch, tmp_path):
    seen_prompts: list[str] = []

    def fake_call(**kwargs):
        prompt = kwargs["user_prompt"]
        seen_prompts.append(prompt)
        if prompt.startswith("["):
            return '[{"title_zh": "第一篇"}]'
        return json.dumps({"title_zh": f"{prompt} 中文"}, ensure_ascii=False)

    monkeypatch.setattr("weixin_lite.translate.call_openai_compatible", fake_call)
    papers = [PaperInput(title_en="First"), PaperInput(title_en="Second")]

    report = translate_records(papers, api_key="test-key", batch_size=2, delay_seconds=0, cache_path=tmp_path / "cache.json")

    assert seen_prompts == ['[{"title_en": "First"}, {"title_en": "Second"}]', "First", "Second"]
    assert [paper.title_zh for paper in papers] == ["First 中文", "Second 中文"]
    assert [paper.translation_status for paper in papers] == ["translated", "translated"]
    assert report.translated_count == 2
    assert report.failed_count == 0
    assert report.ok


def test_translation_single_title_fallback_failure_marks_failed(monkeypatch, tmp_path):
    def fake_call(**kwargs):
        if kwargs["user_prompt"].startswith("["):
            raise LLMError("server overloaded", status_code=500, transient=True)
        raise LLMError("quota exhausted", status_code=429, transient=False)

    monkeypatch.setattr("weixin_lite.translate.call_openai_compatible", fake_call)
    paper = PaperInput(title_en="First")

    report = translate_records([paper], api_key="test-key", batch_size=1, delay_seconds=0, max_retries=0, cache_path=tmp_path / "cache.json")

    assert paper.translation_status == "failed"
    assert report.failed_count == 1
    assert report.failed_items


def test_retry_after_naive_http_date_does_not_raise():
    assert _parse_retry_after("Wed, 21 Oct 2015 07:28:00") == 0


def test_provider_defaults_are_configured():
    assert default_base_url("gemini") == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert default_model("gemini") == "gemini-2.5-flash"
    assert default_base_url("deepseek") == "https://api.deepseek.com/v1"
    assert default_model("deepseek") == "deepseek-chat"
    assert default_base_url("siliconflow") == "https://api.siliconflow.cn/v1"


def test_gemini_api_key_uses_gemini_environment(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")

    assert default_api_key("gemini") == "gemini-key"


def test_paywalled_without_oa_url_keeps_doi_only():
    paper = PaperInput(title_en="Paywalled paper", doi="10.1000/paywall")

    downloaded = download_open_access(paper)

    assert downloaded.status == "paywalled"
    assert downloaded.content_bytes == b""


def test_fallback_article_has_required_chinese_sections():
    paper = PaperInput(
        title_en="Template-independent enzymatic DNA synthesis using terminal deoxynucleotidyl transferase",
        title_zh="利用 TdT 的模板非依赖酶促 DNA 合成",
        journal="Nature Biotechnology",
        year="2024",
        doi="10.1000/test",
        abstract_en="This study reports an enzymatic DNA synthesis method.",
        abstract_zh="该研究报道了一种酶促 DNA 合成方法。",
    )
    text = "Fig. 1 The TdT reaction produced DNA up to 100 nt with 90% conversion."
    legends = extract_figure_legends(text)
    evidence = extract_numeric_evidence(text, legends)
    pdf = PdfContent(text=text, legends=legends, evidence=evidence, parser="fixture")

    article = generate_article(paper, pdf=pdf)

    assert "文章核心要点简述" in article.body_markdown
    assert "文章的创新意义" in article.body_markdown
    assert "解读：" not in article.body_markdown
    assert "English abstract" not in article.body_markdown
    assert article.evidence


def test_article_places_screenshot_before_short_note():
    paper = PaperInput(title_en="A test paper", journal="Nature", year="2025")
    figure = FigureAnalysis(
        figure_id="Fig. 3",
        caption="Fig. 3 Growth curve with 90% conversion.",
        page="3",
        image_name="paper-page-3.png",
    )
    pdf = PdfContent(text="Fig. 3 Growth curve with 90% conversion.", legends=[figure], parser="fixture")

    article = generate_article(paper, pdf=pdf)

    image_pos = article.body_markdown.find("![Fig. 3](images/paper-page-3.png)")
    note_pos = article.body_markdown.find("**Fig. 3：原文关键信息截图**")
    assert image_pos >= 0
    assert note_pos > image_pos


def test_confirmed_figure_analysis_falls_back_to_caption_evidence():
    paper = PaperInput(title_en="A test paper")
    figure = FigureAnalysis(
        figure_id="Fig. 2",
        caption="Fig. 2 The engineered strain reached 90% conversion.",
        page="4",
        image_name="fig2.png",
        role="key_result",
        selected=True,
        order=1,
    )
    analysis = PaperAnalysis(
        key_results=[
            AnalysisClaim(
                "工程菌株提高了转化效率",
                page="4",
                figure_id="Fig. 2",
                evidence_text="reached 90% conversion",
            )
        ],
        status="complete",
    )

    figures = analyze_confirmed_figures(paper, analysis, [figure])

    assert figures == [figure]
    assert figure.interpretation
    assert "工程菌株提高了转化效率" in figure.interpretation


def test_confirmed_figures_ignore_unconfirmed_model_figure_notes(monkeypatch):
    def fake_call(**kwargs):
        return json.dumps(
            {
                "title": "确认配图稿",
                "digest": "摘要",
                "intro": "导语",
                "core_points": ["核心分析。"],
                "figure_notes": [{"figure_id": "Fig. 99", "heading": "不应出现", "note": "不应进入正文"}],
                "innovation": ["创新意义。"],
                "limitations": ["证据边界。"],
                "take_home": "总结。",
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr("weixin_lite.generator.call_openai_compatible", fake_call)
    monkeypatch.setattr(
        "weixin_lite.figure_analysis.call_openai_compatible",
        lambda **kwargs: json.dumps(
            {
                "figures": [
                    {
                        "figure_id": "Fig. 2",
                        "heading": "Fig. 2：关键结果图",
                        "note": "确认图解。",
                        "evidence_text": "caption evidence",
                        "page": "4",
                    }
                ]
            },
            ensure_ascii=False,
        ),
    )
    paper = PaperInput(title_en="A test paper")
    figure = FigureAnalysis(
        figure_id="Fig. 2",
        caption="Fig. 2 Result.",
        page="4",
        image_name="fig2.png",
        role="key_result",
        selected=True,
        order=1,
    )
    pdf = PdfContent(
        text="Fig. 2 Result.",
        legends=[figure],
        lead_image=FigureAnalysis("Lead", "Title", page="1", image_name="lead.png"),
    )

    article = generate_article(paper, pdf=pdf, api_key="key", confirmed_figures=[figure])

    assert "![论文首页](images/lead.png)" in article.body_markdown
    assert "![Fig. 2](images/fig2.png)" in article.body_markdown
    assert "确认图解" in article.body_markdown
    assert "Fig. 99" not in article.body_markdown
    assert article.body_markdown.find("![论文首页]") < article.body_markdown.find("导语")
    assert article.body_markdown.find("![Fig. 2]") < article.body_markdown.find("确认图解")


def test_untraceable_confirmed_figure_is_not_rendered():
    paper = PaperInput(title_en="A test paper")
    figure = FigureAnalysis(
        figure_id="Fig. 5",
        caption="",
        page="",
        image_name="fig5.png",
        selected=True,
        order=1,
    )

    article = generate_article(paper, confirmed_figures=[figure])

    assert "fig5.png" not in article.body_markdown


def test_wechat_markdown_html_matches_reference_style_without_duplicate_title():
    markdown = """# 平台标题

引言里有**重点**。

## 文章核心要点简述

1. **核心策略：** 先放原文截图，再写中文说明。

![Fig. 1](images/fig1.png)

**Fig. 1：原文关键信息截图**

原文信息：Nature / DOI: 10.1000/test
"""

    html = markdown_to_wechat_html(markdown)

    assert "平台标题" not in html
    assert "font-size:18px;line-height:2.05;color:#000" in html
    assert "font-size:22px" in html
    assert "width:100%;height:auto;display:block" in html
    assert '<strong style="font-weight:800;color:#000;">重点</strong>' in html
    assert "原文信息" in html


def test_exported_article_html_omits_wechat_platform_header():
    article = generate_article(PaperInput(title_zh="测试文章", journal="Nature Biotechnology"))

    html = article_html(article)

    assert "原创" not in html
    assert "陶小花" not in html
    assert "遇见生物合成" not in html
    assert 'font-size:28px;line-height:1.22' not in html
    assert "文章核心要点简述" in html


def test_llm_429_warning_is_user_friendly(monkeypatch):
    def fail_call(**kwargs):
        raise LLMError(
            "LLM HTTP 429 Too Many Requests: You exceeded your current quota. https://platform.openai.com/docs",
            status_code=429,
            transient=True,
        )

    monkeypatch.setattr("weixin_lite.generator.call_openai_compatible", fail_call)
    article = generate_article(PaperInput(title_zh="测试文章"), api_key="test-key")
    joined = "；".join(article.warnings)

    assert "模型接口额度不足或被限流" in joined
    assert "Too Many Requests" not in joined
    assert "platform.openai.com" not in joined


def test_llm_retries_transient_429_and_respects_retry_after(monkeypatch):
    calls = {"count": 0}
    sleeps: list[float] = []

    def fake_once(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise LLMError("rate_limit_exceeded", status_code=429, retry_after=2.5, transient=True)
        return "ok"

    monkeypatch.setattr(llm_module, "_call_openai_compatible_once", fake_once)
    monkeypatch.setattr(llm_module.time, "sleep", sleeps.append)

    result = call_openai_compatible("key", "https://example.test/v1", "model", "system", "user")

    assert result == "ok"
    assert calls["count"] == 2
    assert sleeps == [2.5]


def test_llm_does_not_retry_exhausted_quota(monkeypatch):
    calls = {"count": 0}

    def fake_once(**kwargs):
        calls["count"] += 1
        raise LLMError(
            "You exceeded your current quota",
            status_code=429,
            transient=False,
            error_code="insufficient_quota",
            quota_exhausted=True,
        )

    monkeypatch.setattr(llm_module, "_call_openai_compatible_once", fake_once)
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: pytest.fail("quota errors must not sleep"))

    with pytest.raises(LLMError) as error:
        call_openai_compatible("key", "https://example.test/v1", "model", "system", "user")

    assert error.value.quota_exhausted is True
    assert calls["count"] == 1


def _complete_analysis() -> PaperAnalysis:
    return PaperAnalysis(
        research_question=[AnalysisClaim("研究问题", page="1", evidence_text="摘要")],
        methods=[AnalysisClaim("研究方法", page="3", figure_id="Fig. 1", evidence_text="方法段")],
        key_results=[AnalysisClaim("关键结果", page="5", figure_id="Fig. 2", evidence_text="结果段")],
        innovation=[AnalysisClaim("创新意义", page="6", evidence_text="讨论段")],
        limitations=[AnalysisClaim("样本有限", page="7", evidence_text="局限段")],
        conclusion=[AnalysisClaim("论文结论", page="8", evidence_text="结论段")],
        status="complete",
        source_hash="source-hash",
        model="test-model",
    )


def test_analysis_models_serialize_and_cache_key_excludes_api_key():
    analysis = _complete_analysis()
    restored = PaperAnalysis.from_dict(analysis.to_dict())
    paper = PaperInput(title_en="Traceable paper", doi="10.1000/trace")
    pdf = PdfContent(text="full text", hash="pdf-hash", quality="high", coverage=["results"])

    first = analysis_cache_key(paper, pdf, {"model": "test-model", "api_key": "secret-a"})
    second = analysis_cache_key(paper, pdf, {"model": "test-model", "api_key": "secret-b"})

    assert restored.complete
    assert restored.key_results[0].figure_id == "Fig. 2"
    assert first == second
    assert "secret" not in first


def test_analysis_prompt_uses_expert_deep_reading_guide():
    prompt = build_analysis_prompt(
        PaperInput(title_en="Traceable paper", doi="10.1000/trace"),
        PdfContent(text="[Page 1]\nAbstract\nKey result.", hash="pdf-hash"),
    )

    assert ANALYSIS_PROMPT_VERSION == "paper-analysis-v2"
    assert "世界顶级学术专家" in prompt
    assert "### 论文的研究目标是什么？想要解决什么实际问题？" in prompt
    assert "### 这个问题对于产业发展有什么重要意义？" in prompt
    assert "### 实验是如何设计的？实验数据和结果如何？" in prompt
    assert "blockquote" in prompt
    assert '"research_question"' in prompt
    assert "只返回符合要求的 JSON" in prompt


def test_analysis_failure_preserves_previous_complete_analysis(monkeypatch):
    previous = _complete_analysis()
    monkeypatch.setattr(
        "weixin_lite.article_analysis.call_openai_compatible",
        lambda **kwargs: (_ for _ in ()).throw(LLMError("server overloaded", status_code=500, transient=True)),
    )

    result = analyze_paper(
        PaperInput(title_en="Paper"),
        PdfContent(text="[Page 1] full text", hash="hash"),
        {"api_key": "key", "model": "test-model"},
        previous_analysis=previous,
    )

    assert result.complete
    assert result.key_results[0].statement == "关键结果"
    assert any("已保留上一次完整分析" in warning for warning in result.warnings)


def test_failed_analysis_cannot_generate_fake_completed_article():
    failed = PaperAnalysis(status="failed", error="analysis unavailable")

    with pytest.raises(ArticleGenerationError, match="analysis unavailable"):
        generate_article(PaperInput(title_en="Paper"), analysis=failed, api_key="key")


def test_quality_generation_uses_one_call_without_hidden_length_repair(monkeypatch):
    calls = {"count": 0}

    def fake_call(**kwargs):
        calls["count"] += 1
        return json.dumps(
            {
                "title": "可追溯解读",
                "digest": "摘要",
                "intro": "导语",
                "core_points": ["核心结果来自第五页。"],
                "figure_notes": [],
                "innovation": ["创新点来自第六页。"],
                "limitations": ["样本有限。"],
                "take_home": "总结。",
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr("weixin_lite.generator.call_openai_compatible", fake_call)

    article = generate_article(
        PaperInput(title_en="Paper"),
        pdf=PdfContent(text="full text", hash="source-hash"),
        api_key="key",
        analysis=_complete_analysis(),
    )

    assert calls["count"] == 1
    assert article.analysis_version == "paper-analysis-v1"
    assert "局限性与解读边界" in article.body_markdown
    assert any("显式补写/精简" in warning for warning in article.warnings)


def test_project_zip_contains_paywalled_and_download_status():
    paper = PaperInput(title_en="A paper", doi="10.1000/a", access_status="paywalled")
    article = generate_article(paper)
    project = BatchProject(topic="secret-key-should-not-appear", papers=[paper], articles=[article])

    data = project_zip(project, {"cover.png": b"image"})
    with zipfile.ZipFile(BytesIO(data)) as zf:
        names = set(zf.namelist())
        payload = zf.read("project.json").decode("utf-8")
        paywalled = zf.read("paywalled_dois.csv").decode("utf-8")
        unavailable = zf.read("unavailable_dois.csv").decode("utf-8")
        latest = json.loads(zf.read("latest_papers.json").decode("utf-8"))

    assert any(name.startswith("articles/") and name.endswith(".html") for name in names)
    assert "images/cover.png" in names
    assert "paywalled_dois.csv" in names
    assert "unavailable_dois.csv" in names
    assert "download_status.json" in names
    assert "10.1000/a" in paywalled
    assert "10.1000/a" in unavailable
    assert latest[0]["doi"] == "10.1000/a"
    assert "sk-" not in payload


def test_generation_ready_requires_open_status_and_parsed_pdf():
    ready = PaperInput(title_en="Ready", doi="10.1000/ready", access_status="open", pdf_name="ready.pdf")
    no_pdf = PaperInput(title_en="No PDF", doi="10.1000/nopdf", access_status="open")
    failed = PaperInput(title_en="Failed", doi="10.1000/failed", access_status="download_failed")
    pdfs = {"ready.pdf": PdfContent(text="Fig. 1 data")}

    assert generation_ready_papers([ready, no_pdf, failed], pdfs) == [ready]
    assert [paper.doi for paper in unavailable_papers([ready, no_pdf, failed], pdfs)] == ["10.1000/nopdf", "10.1000/failed"]


def test_generation_candidates_include_metadata_without_pdf():
    ready = PaperInput(title_en="Ready", doi="10.1000/ready", access_status="open", pdf_name="ready.pdf")
    no_pdf = PaperInput(title_en="No PDF", abstract_en="Metadata is enough.", doi="10.1000/nopdf", access_status="open")
    failed = PaperInput(title_en="Failed", doi="10.1000/failed", access_status="download_failed")
    paywalled = PaperInput(title_en="Paywalled", doi="10.1000/pay", access_status="paywalled")
    unknown = PaperInput(title_en="Unknown", doi="10.1000/unknown", access_status="unknown")
    pdfs = {"ready.pdf": PdfContent(text="Fig. 1 data")}

    assert generation_candidate_papers([ready, no_pdf, failed, paywalled, unknown], pdfs) == [
        ready,
        no_pdf,
        failed,
        paywalled,
        unknown,
    ]


def test_batch_analyze_does_not_duplicate_open_without_pdf(monkeypatch, tmp_path):
    paper = PaperInput(title_en="Open without parsed PDF", doi="10.1000/open", access_status="open")
    input_path = tmp_path / "papers.json"
    output_path = tmp_path / "project.zip"
    input_path.write_text(json.dumps([paper.to_dict()], ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["batch_analyze", "--input", str(input_path), "--output", str(output_path)])

    batch_analyze.main()

    with zipfile.ZipFile(output_path) as zf:
        latest = json.loads(zf.read("latest_papers.json").decode("utf-8"))
    assert [item["doi"] for item in latest] == ["10.1000/open"]


def test_generate_article_without_pdf_uses_metadata():
    paper = PaperInput(
        title_en="A general technology article",
        journal="Example Journal",
        abstract_en="The article explains a new engineering workflow.",
        access_status="paywalled",
    )

    article = generate_article(paper)

    assert "文章核心要点简述" in article.body_markdown
    assert "文章的创新意义" in article.body_markdown
    assert any("题录/摘要" in warning or "摘要级" in warning for warning in article.warnings)


def test_manual_source_text_enters_prompt_and_generation():
    paper = PaperInput(title_en="Manual source article")
    prompt = build_prompt(paper, None, 1200, source_text="This is pasted source material about market adoption.")

    assert "用户提供正文/材料" in prompt
    assert "market adoption" in prompt

    article = generate_article(paper, source_text="This is pasted source material about market adoption.")
    assert any("用户粘贴材料" in warning or "材料级" in warning for warning in article.warnings)


def test_fallback_article_is_not_domain_limited():
    paper = PaperInput(title_en="A general article", abstract_en="A policy and industry analysis.")

    article = generate_article(paper)

    forbidden = ["TdT", "PUP", "DNA/RNA", "酶促 DNA", "酶促DNA", "生物技术"]
    assert all(term not in article.body_markdown for term in forbidden)


def test_unavailable_dois_csv_includes_open_without_pdf_and_error():
    paper = PaperInput(
        title_en="Open but missing PDF",
        doi="10.1000/missing",
        access_status="open",
        download_error="downloaded HTML only",
    )

    csv_text = unavailable_dois_csv([paper])

    assert "10.1000/missing" in csv_text
    assert "downloaded HTML only" in csv_text


def test_wechat_draft_dry_run_builds_payload():
    paper = PaperInput(title_zh="测试论文", doi="10.1000/test", url="https://doi.org/10.1000/test")
    article = generate_article(paper, pdf=PdfContent(text="Fig. 1 produced 90% conversion."))
    config = WechatDraftConfig(author="Codex", show_cover_pic=True)

    result = publish_draft(article, config, dry_run=True)

    payload = result["payload"]
    assert result["dry_run"] is True
    assert payload["articles"][0]["title"] == article.title
    assert payload["articles"][0]["author"] == "Codex"
    assert payload["articles"][0]["show_cover_pic"] == 1
