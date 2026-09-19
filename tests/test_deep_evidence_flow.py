import json

import pytest

from weixin_lite.article_analysis import analyze_paper, build_analysis_prompt, paper_analysis_from_payload
from weixin_lite.figure_analysis import analyze_confirmed_figures, prepare_text_evidence_figures
from weixin_lite.generator import build_prompt, render_markdown
from weixin_lite.models import AnalysisClaim, FigureAnalysis, PaperAnalysis, PaperInput
from weixin_lite.pdf_reader import PdfContent


def _analysis() -> PaperAnalysis:
    return PaperAnalysis(
        research_question=[AnalysisClaim("论文要解决稳定性不足的问题", page="1", evidence_text="> goal")],
        background=[AnalysisClaim("该问题影响工艺放大", page="2", evidence_text="> scale-up")],
        methods=[AnalysisClaim("采用双对照实验", page="3", figure_id="Fig. 1", evidence_text="> controls")],
        key_results=[AnalysisClaim("转化率达到 90%", page="4", figure_id="Fig. 2", evidence_text="> 90% conversion")],
        limitations=[AnalysisClaim("样本范围有限", page="5", evidence_text="> limited samples")],
        conclusion=[AnalysisClaim("该策略在当前条件下有效", page="5", evidence_text="> effective")],
        status="complete",
    )


def test_deep_prompts_use_full_evidence_without_a_product_length_cap():
    paper = PaperInput(title_en="Traceable paper", authors=["Personal Author"])
    basic_prompt = build_prompt(paper, None, 1200)
    analysis_prompt = build_analysis_prompt(paper, PdfContent(text="[Page 1] evidence"))

    assert "不设置人为字数上限" in basic_prompt
    assert "2800-4200" not in basic_prompt
    assert "500-1500" not in basic_prompt
    assert "产业发展有什么重要意义" in analysis_prompt
    assert "论文采用了什么方法" in analysis_prompt
    assert "不要整理完整实验清单" in analysis_prompt
    assert "局限性" in analysis_prompt
    assert "Personal Author" not in basic_prompt
    assert "公众号名称、运营作者" in basic_prompt


def test_rendered_deep_article_has_all_distinct_evidence_sections():
    markdown = render_markdown(
        PaperInput(title_zh="测试论文"),
        {
            "title": "测试论文深度解读",
            "intro": "导语来自全文证据。",
            "research_question": "研究目标与产业意义均有页码证据。",
            "method_overview": ["方法概述来自 p.3。"],
            "innovation": ["创新点来自讨论部分。"],
            "limitations": ["样本范围有限。"],
            "take_home": "结论只限定于原文证据。",
        },
        [],
    )

    for heading in (
        "研究问题与现实意义",
        "研究方法概述",
        "文章的创新意义",
        "局限性与解读边界",
        "总结",
    ):
        assert f"## {heading}" in markdown
    assert "实验设计与验证" not in markdown
    assert "关键数据与结果" not in markdown


def test_method_overview_is_limited_to_three_items():
    markdown = render_markdown(
        PaperInput(title_zh="测试论文"),
        {
            "title": "测试",
            "method_overview": ["方法一", "方法二", "方法三", "不应出现的方法四"],
        },
        [],
    )

    assert all(value in markdown for value in ("方法一", "方法二", "方法三"))
    assert "不应出现的方法四" not in markdown


def test_analysis_rejects_missing_industry_or_source_evidence():
    payload = {
        "research_question": [{"statement": "目标", "page": "1", "evidence_text": "evidence"}],
        "background": [],
        "methods": [{"statement": "方法", "page": "2", "evidence_text": "evidence"}],
        "key_results": [{"statement": "结果", "page": "3", "evidence_text": "evidence"}],
        "limitations": [{"statement": "局限", "page": "4", "evidence_text": "evidence"}],
        "conclusion": [{"statement": "结论", "page": "4", "evidence_text": "evidence"}],
    }

    with pytest.raises(ValueError, match="现实/产业意义"):
        paper_analysis_from_payload(payload, source_hash="hash", model="model")

    payload["background"] = [{"statement": "产业意义", "page": "1", "evidence_text": ""}]
    with pytest.raises(ValueError, match="现实/产业意义"):
        paper_analysis_from_payload(payload, source_hash="hash", model="model")


def test_confirmed_assets_require_gemini_visual_review_before_rendering():
    paper = PaperInput(title_en="Traceable paper")
    no_evidence = FigureAnalysis("Fig. 99", "", page="2", image_name="no-evidence.png", selected=True)
    confirmed = FigureAnalysis(
        "Fig. 2",
        "Fig. 2 reports 90% conversion for the treatment group.",
        page="4",
        image_name="confirmed.png",
        selected=True,
    )

    figures = analyze_confirmed_figures(paper, _analysis(), [no_evidence, confirmed])

    assert figures == []
    assert confirmed.vision_status == "blocked"
    assert "Gemini" in confirmed.vision_error


def test_selected_asset_can_use_explicit_text_evidence_fallback():
    selected = FigureAnalysis(
        "Fig. 2",
        "Fig. 2 reports 90% conversion for the treatment group.",
        page="4",
        image_name="selected.png",
        selected=True,
    )
    skipped = FigureAnalysis(
        "Fig. 3",
        "Fig. 3 control.",
        page="5",
        image_name="skipped.png",
        selected=False,
    )

    figures = prepare_text_evidence_figures(_analysis(), [selected, skipped])

    assert figures == [selected]
    assert selected.vision_status == "text_evidence"
    assert "未完成视觉复核" in selected.interpretation
    assert "图展示什么" in selected.interpretation
    assert "人工核对" in selected.vision_error


def test_gemini_vision_receives_only_confirmed_image_bytes(monkeypatch):
    calls = []

    def fake_vision(*, images, **kwargs):
        calls.append({"images": images, "prompt": kwargs["user_prompt"]})
        return json.dumps(
            {
                "figures": [
                    {
                        "figure_id": "Fig. 2",
                        "heading": "Fig. 2：关键结果",
                        "overview": "图中比较处理组与对照组。",
                        "key_findings": "处理组达到 90%。",
                        "conclusion_boundary": "该结果仅支持图示条件下的结论。",
                        "evidence_text": "> 90% conversion",
                        "page": "4",
                    }
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr("weixin_lite.figure_analysis.call_openai_compatible_with_images", fake_vision)
    selected = FigureAnalysis("Fig. 2", "Fig. 2 90% conversion.", page="4", image_name="selected.png", selected=True)
    skipped = FigureAnalysis("Fig. 3", "Fig. 3 control.", page="5", image_name="skipped.png", selected=False)

    figures = analyze_confirmed_figures(
        PaperInput(title_en="Traceable paper"),
        _analysis(),
        [selected, skipped],
        {
            "api_key": "test-key",
            "provider": "gemini",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "model": "gemini-2.5-flash",
            "image_assets": {"selected.png": b"selected-image", "skipped.png": b"skipped-image"},
        },
        pdf=PdfContent(text="[Page 4]\nSELECTED_PAGE_EVIDENCE\n[Page 5]\nSKIPPED_PAGE_EVIDENCE"),
    )

    assert figures == [selected]
    assert calls[0]["images"] == [(b"selected-image", "image/png")]
    assert "SELECTED_PAGE_EVIDENCE" in calls[0]["prompt"]
    assert "SKIPPED_PAGE_EVIDENCE" not in calls[0]["prompt"]
    assert "Fig. 3" not in calls[0]["prompt"]
    assert "视觉复核已完成（Gemini 图像输入）" in selected.interpretation
    assert "图展示什么" in selected.interpretation
    assert "关键数据或趋势" in selected.interpretation
    assert "结论与证据边界" in selected.interpretation
    assert "实验或比较如何设计" not in selected.interpretation


def test_analysis_accepts_pdf_content_restored_from_before_analysis_chunks(monkeypatch):
    class LegacyPdfContent:
        text = "[Page 1] full evidence"
        hash = "legacy-hash"
        legends = []

    payload = {
        field: [{"statement": field, "page": "1", "evidence_text": "> source"}]
        for field in ("research_question", "background", "methods", "key_results", "limitations", "conclusion")
    }
    payload["innovation"] = [{"statement": "innovation", "page": "1", "evidence_text": "> source"}]
    monkeypatch.setattr("weixin_lite.article_analysis.call_openai_compatible", lambda **kwargs: json.dumps(payload))

    result = analyze_paper(PaperInput(title_en="Legacy paper"), LegacyPdfContent(), {"api_key": "key"})

    assert result.complete
