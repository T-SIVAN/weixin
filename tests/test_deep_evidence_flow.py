import json

import pytest

from weixin_lite.article_analysis import build_analysis_prompt, paper_analysis_from_payload
from weixin_lite.figure_analysis import analyze_confirmed_figures
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
    paper = PaperInput(title_en="Traceable paper")
    basic_prompt = build_prompt(paper, None, 1200)
    analysis_prompt = build_analysis_prompt(paper, PdfContent(text="[Page 1] evidence"))

    assert "不设置人为字数上限" in basic_prompt
    assert "2800-4200" not in basic_prompt
    assert "500-1500" not in basic_prompt
    assert "产业发展有什么重要意义" in analysis_prompt
    assert "实验是如何设计的？实验数据和结果如何？" in analysis_prompt
    assert "局限性" in analysis_prompt


def test_rendered_deep_article_has_all_distinct_evidence_sections():
    markdown = render_markdown(
        PaperInput(title_zh="测试论文"),
        {
            "title": "测试论文深度解读",
            "intro": "导语来自全文证据。",
            "research_question": "研究目标与产业意义均有页码证据。",
            "approach_advantage": ["方法与对照优势来自 p.3。"],
            "experiment_validation": ["实验设计来自 p.4。"],
            "quantitative_findings": ["关键数据为 90%，见 Fig. 2。"],
            "innovation": ["创新点来自讨论部分。"],
            "limitations": ["样本范围有限。"],
            "take_home": "结论只限定于原文证据。",
        },
        [],
    )

    for heading in (
        "研究问题与现实意义",
        "方法路径与比较优势",
        "实验设计与验证",
        "关键数据与结果",
        "文章的创新意义",
        "局限性与解读边界",
        "总结",
    ):
        assert f"## {heading}" in markdown


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


def test_gemini_vision_receives_only_confirmed_image_bytes(monkeypatch):
    calls: list[list[tuple[bytes, str]]] = []

    def fake_vision(*, images, **kwargs):
        calls.append(images)
        return json.dumps(
            {
                "figures": [
                    {
                        "figure_id": "Fig. 2",
                        "heading": "Fig. 2：关键结果",
                        "note": "处理组高于对照组。",
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
    )

    assert figures == [selected]
    assert calls == [[(b"selected-image", "image/png")]]
    assert "视觉复核已完成（Gemini 图像输入）" in selected.interpretation
