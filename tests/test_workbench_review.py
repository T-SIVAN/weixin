import io
import json
import urllib.error
import zipfile
from datetime import date

import pytest
from streamlit.testing.v1 import AppTest

import app
from weixin_lite import llm
from weixin_lite.article_analysis import ANALYSIS_FIELDS, analyze_paper
from weixin_lite.figure_analysis import analyze_confirmed_figures
from weixin_lite.models import AnalysisClaim, FigureAnalysis, PaperAnalysis, PaperInput
from weixin_lite.pdf_reader import PdfContent
from weixin_lite.translate import translate_records


def payload():
    return {name: [{"statement": name, "page": "1", "evidence_text": "> observed result"}]
            for name in ANALYSIS_FIELDS}


def google_error(quota_id, delay="28.7s", value="20"):
    return [{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
             "message": "You exceeded your current quota, please check your plan and billing details.",
             "details": [
                 {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                  "violations": [{"quotaId": quota_id, "quotaValue": value}]},
                 {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay},
             ]}}]


@pytest.mark.parametrize("quota_id,value,expected_calls", [
    ("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "20", 2),
    ("GenerateRequestsPerDayPerProjectPerModel-FreeTier", "20", 1),
    ("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "0", 1),
])
def test_real_google_http_error_classification(monkeypatch, quota_id, value, expected_calls):
    calls, sleeps = [], []
    def urlopen(request, **kwargs):
        calls.append(request)
        if len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {},
                                        io.BytesIO(json.dumps(google_error(quota_id, value=value)).encode()))
        return io.BytesIO(b'{"choices":[{"message":{"content":"ok"}}]}')
    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    if expected_calls == 2:
        assert llm.call_openai_compatible("fake", "https://example.test", "gemini", "system", "user") == "ok"
        assert sleeps == [28.7]
    else:
        with pytest.raises(llm.LLMError) as error:
            llm.call_openai_compatible("fake", "https://example.test", "gemini", "system", "user")
        assert error.value.quota_exhausted
        assert not sleeps
    assert len(calls) == expected_calls


def test_google_message_delay_does_not_imply_daily_quota():
    message = "You exceeded your current quota. Please retry in 28.784059496s."
    assert llm._parse_body_retry_delay(message) == 28.784059496
    assert not llm._is_quota_exhaustion(message)
    assert llm._is_quota_exhaustion("insufficient_quota")


def test_long_cooldown_does_not_freeze_request(monkeypatch):
    def fail(**kwargs):
        raise llm.LLMError("wait", status_code=429, transient=True, retry_after=3600)
    monkeypatch.setattr(llm, "_call_openai_compatible_once", fail)
    monkeypatch.setattr(llm.time, "sleep", lambda _: pytest.fail("must return control"))
    with pytest.raises(llm.LLMError):
        llm.call_openai_compatible("key", "url", "model", "system", "user")


def test_analysis_resumes_after_quota_without_repeating_completed_chunks(monkeypatch):
    pdf = PdfContent(text="x" * 37000, hash="long-pdf")
    paper = PaperInput(title_en="Long paper")
    calls, progress = [], []
    def call(**kwargs):
        calls.append(kwargs["user_prompt"])
        if len(calls) == 2:
            raise llm.LLMError("private raw response", status_code=429, quota_exhausted=True)
        return json.dumps(payload())
    monkeypatch.setattr("weixin_lite.article_analysis.call_openai_compatible", call)
    config = {"api_key": "fake", "cache": {}, "progress_callback": lambda done, total: progress.append((done, total))}
    first = analyze_paper(paper, pdf, config)
    assert not first.complete
    assert first.completed_chunks == 1 and first.total_chunks == 3
    assert first.claims
    assert "private raw response" not in first.error
    second = analyze_paper(paper, pdf, config, first)
    assert second.complete and second.completed_chunks == 3
    assert len(calls) == 4
    assert progress[-1] == (3, 3)
    assert analyze_paper(paper, pdf, config).complete
    assert len(calls) == 4


def test_analysis_does_not_reuse_a_different_pdf(monkeypatch):
    previous = PaperAnalysis(status="complete", source_hash="other", research_question=[AnalysisClaim("old")])
    monkeypatch.setattr("weixin_lite.article_analysis.call_openai_compatible",
                        lambda **_: (_ for _ in ()).throw(llm.LLMError("quota")))
    result = analyze_paper(PaperInput(title_en="New"), PdfContent(text="new", hash="new"), {"api_key": "fake"}, previous)
    assert not result.complete
    assert not result.claims


def test_short_sections_share_a_request_without_losing_sections(monkeypatch):
    seen = []
    def call(**kwargs):
        seen.append(kwargs["user_prompt"])
        return json.dumps(payload())
    monkeypatch.setattr("weixin_lite.article_analysis.call_openai_compatible", call)
    pdf = PdfContent(sections={"abstract": "START", "methods": "CONTROL", "discussion": "END"})
    assert analyze_paper(PaperInput(title_en="Short"), pdf, {"api_key": "fake"}).complete
    assert len(seen) == 1 and all(word in seen[0] for word in ("START", "CONTROL", "END"))


def test_visual_review_reuses_images_and_invalidates_changed_bytes(monkeypatch):
    calls = []
    def call(**kwargs):
        calls.append(kwargs["images"])
        return json.dumps({"figures": [{"figure_id": "Fig. 1", "note": "Observed trend", "page": "1", "evidence_text": "control"}]})
    monkeypatch.setattr("weixin_lite.figure_analysis.call_openai_compatible_with_images", call)
    figure = FigureAnalysis("Fig. 1", "Treatment compared with control", page="1", selected=True, image_name="figure.png")
    config = {"provider": "gemini", "api_key": "fake", "model": "gemini-test", "cache": {}, "image_assets": {"figure.png": b"image-a"}}
    paper = PaperInput(title_en="Study")
    assert analyze_confirmed_figures(paper, None, [figure], config)
    assert analyze_confirmed_figures(paper, None, [figure], config)
    assert len(calls) == 1
    config["image_assets"]["figure.png"] = b"image-b"
    assert analyze_confirmed_figures(paper, None, [figure], config)
    assert len(calls) == 2


def test_translation_quota_stops_batch_fanout(monkeypatch, tmp_path):
    calls = []
    def call(**kwargs):
        calls.append(kwargs)
        raise llm.LLMError("quota", status_code=429, quota_exhausted=True)
    monkeypatch.setattr("weixin_lite.translate.call_openai_compatible", call)
    records = [PaperInput(title_en=f"Title {index}") for index in range(8)]
    report = translate_records(records, api_key="fake", batch_size=2, delay_seconds=0, cache_path=tmp_path / "cache.json")
    assert len(calls) == 1
    assert report.failed_count == 2 and report.pending_count == 6


def test_chinese_titles_have_distinct_stable_keys():
    first, second = PaperInput(title="小鼠甲基化"), PaperInput(title="蛋白质合成")
    assert app.paper_key(first) != app.paper_key(second)
    assert app.paper_key(first) == app.paper_key(first)
    assert len(app.merge_papers([first], [second])) == 2


def test_selectable_date_range_label_includes_exact_days():
    assert app.date_range_label(date(2026, 2, 3), date(2026, 9, 18)) == "2026年2月3日 至 2026年9月18日"


def test_reuploaded_pdf_replaces_previous_source_without_losing_metadata():
    old = PaperInput(doi="10.1234/paper", title_en="Study", title_zh="已翻译标题", pdf_name="old.pdf")
    new = PaperInput(doi="10.1234/paper", title_en="Study", pdf_name="revised.pdf")
    merged = app.merge_papers([old], [new])
    assert len(merged) == 1
    assert merged[0].pdf_name == "revised.pdf"
    assert merged[0].title_zh == "已翻译标题"


@pytest.mark.parametrize("batch_response", ["[]", "invalid json"])
def test_translation_quota_also_stops_single_title_fallback(monkeypatch, tmp_path, batch_response):
    calls = []
    def call(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return batch_response
        raise llm.LLMError("quota", status_code=429, quota_exhausted=True)
    monkeypatch.setattr("weixin_lite.translate.call_openai_compatible", call)
    records = [PaperInput(title_en=f"Title {index}") for index in range(8)]
    report = translate_records(records, api_key="fake", batch_size=3, delay_seconds=0,
                               cache_path=tmp_path / "cache.json")
    assert len(calls) == 2
    assert report.failed_count == 1 and report.pending_count == 7


def test_navigation_renders_only_selected_page_and_needs_no_credentials(monkeypatch):
    monkeypatch.setattr("weixin_lite.exporter.export_article_docx", lambda *_: pytest.fail("no eager export"))
    at = AppTest.from_file("app.py", default_timeout=20).run()
    assert not at.exception
    assert any(button.label == "检索文章" and button.disabled for button in at.button)
    assert [widget.label for widget in at.date_input] == ["开始日期", "结束日期"]
    assert not any(button.label == "解析上传 PDF" for button in at.button)
    at.radio(key="workspace-page").set_value("论文分析").run()
    assert not at.exception
    assert not any(button.label == "检索文章" for button in at.button)
    at.radio(key="workspace-page").set_value("成稿导出").run()
    assert not at.exception
    assert any("暂无稿件" in info.value for info in at.info)


def test_paper_workspace_without_key_blocks_model_actions():
    at = AppTest.from_file("app.py", default_timeout=20)
    at.session_state["papers"] = [PaperInput(title_en="Study", pdf_name="study.pdf", access_status="open")]
    at.session_state["pdfs"] = {"study.pdf": PdfContent(text="[Page 1] Study", page_count=1, hash="study")}
    at.session_state["workspace-page"] = "论文分析"
    at.run()
    assert not at.exception
    for label in ("分析全文", "复核选中图表", "生成公众号稿"):
        assert next(button for button in at.button if button.label == label).disabled


def test_repeated_sections_and_preamble_reach_analysis():
    from weixin_lite.pdf_reader import extract_sections
    text = "[Page 1]\nTITLE_EVIDENCE\nMethods\nFIRST_METHOD\n[Page 2]\nResults\nRESULT\nMethods\nSECOND_METHOD"
    sections = extract_sections(text)
    assert "FIRST_METHOD" in sections["methods"] and "SECOND_METHOD" in sections["methods"]
    chunks = "\n".join(PdfContent(text=text, sections=sections).analysis_chunks())
    assert all(value in chunks for value in ("TITLE_EVIDENCE", "SECOND_METHOD", "[Page 2]"))


def test_empty_pdf_text_never_calls_model(monkeypatch):
    monkeypatch.setattr("weixin_lite.article_analysis.call_openai_compatible", lambda **_: pytest.fail("no readable text"))
    result = analyze_paper(PaperInput(title_en="Scan"), PdfContent(text="[Page 1]\n", page_count=1), {"api_key": "fake"})
    assert not result.complete and "OCR" in result.error


@pytest.fixture
def simulated_article(monkeypatch):
    import fitz
    from weixin_lite.pdf_reader import parse_pdf
    from weixin_lite.generator import generate_article
    with fitz.open() as document:
        first = document.new_page()
        first.insert_text((50, 50), "SYNTHETIC TEST PAPER - NOT RESEARCH", fontsize=16)
        first.insert_text((50, 100), "Abstract\nA controlled test compares treatment with baseline.\nMethods\nThree replicates.")
        second = document.new_page()
        second.draw_rect(fitz.Rect(50, 60, 300, 220), color=(0, 0.5, 0.4), fill=(0.8, 0.9, 0.9))
        second.insert_text((50, 250), "Fig. 1 Treatment reached 90% conversion compared with 40% for control.")
        second.insert_text((50, 300), "Results\n90% conversion in three replicates.\nDiscussion\nLimited test conditions.")
        data = document.tobytes()
    pdf = parse_pdf(data, mode="pypdf")
    assert pdf.page_count == 2 and pdf.lead_image and pdf.rendered_images
    paper = PaperInput(title_en="Synthetic pipeline fixture", pdf_name="fixture.pdf", access_status="open")
    monkeypatch.setattr("weixin_lite.article_analysis.call_openai_compatible", lambda **_: json.dumps(payload()))
    analysis = analyze_paper(paper, pdf, {"api_key": "fake", "cache": {}})
    assert analysis.complete
    figure = pdf.legends[0]
    figure.selected = True
    detailed_note = "已验证原图数据。" * 120 + "最后一段仍保留。"
    monkeypatch.setattr("weixin_lite.figure_analysis.call_openai_compatible_with_images", lambda **_: json.dumps({
        "figures": [{"figure_id": figure.figure_id, "note": detailed_note, "page": "2", "evidence_text": "90% conversion"}]}))
    confirmed = analyze_confirmed_figures(paper, analysis, [figure], {
        "api_key": "fake", "provider": "gemini", "image_assets": pdf.rendered_images, "cache": {}})
    assert confirmed and "最后一段仍保留" in figure.interpretation
    article_payload = {name: "有证据的测试内容。" for name in (
        "intro", "research_question", "approach_advantage", "experiment_validation", "quantitative_findings",
        "innovation", "limitations", "take_home")}
    article_payload.update(title="本地流程测试稿", digest="用于验证导出链路的合成测试样例。", core_points=["已核对数据。"])
    monkeypatch.setattr("weixin_lite.generator.call_openai_compatible", lambda **_: json.dumps(article_payload))
    article = generate_article(paper, pdf, api_key="fake", analysis=analysis, confirmed_figures=confirmed)
    return article, pdf


def test_pdf_analysis_review_export_pipeline_without_api(simulated_article):
    from docx import Document
    from weixin_lite.exporter import export_article_docx_bytes, export_article_html, project_zip
    from weixin_lite.models import BatchProject
    from weixin_lite.wechat_publish import WechatDraftConfig, publish_draft
    article, pdf = simulated_article
    html = export_article_html(article, pdf.rendered_images).decode()
    assert "data:image/png;base64," in html
    assert html.index("论文首页") < html.index("文章核心要点")
    docx = export_article_docx_bytes(article, pdf.rendered_images)
    assert len(Document(io.BytesIO(docx)).inline_shapes) >= 2
    draft = publish_draft(article, WechatDraftConfig(), pdf.rendered_images, dry_run=True)
    assert len(draft["referenced_content_images"]) >= 2
    archive = project_zip(BatchProject(topic="Synthetic test", articles=[article]), pdf.rendered_images)
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        assert any(name.endswith(".docx") for name in bundle.namelist())
        assert len([name for name in bundle.namelist() if name.startswith("images/")]) >= 2


def test_editor_saves_before_preparing_word(simulated_article):
    article, pdf = simulated_article
    at = AppTest.from_file("app.py", default_timeout=20)
    at.session_state["articles"] = [article]
    at.session_state["images"] = pdf.rendered_images
    at.session_state["workspace-page"] = "成稿导出"
    at.run()
    assert not at.exception
    next(item for item in at.text_input if item.label == "标题").set_value("编辑后的测试稿")
    next(item for item in at.button if item.label == "保存修改").click().run()
    assert not at.exception
    assert at.session_state["articles"][0].title == "编辑后的测试稿"
    next(item for item in at.button if item.label == "准备下载文件").click().run()
    assert not at.exception
    assert at.session_state["prepared_exports"]["docx"].startswith(b"PK")
