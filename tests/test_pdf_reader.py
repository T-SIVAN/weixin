import pytest

from weixin_lite.models import EvidenceItem, FigureAnalysis
from weixin_lite.pdf_reader import (
    PdfContent,
    adjust_crop_bbox,
    build_figure_crops,
    choose_key_figures,
    extract_figure_legends,
    extract_numeric_evidence,
    parse_pdf,
    render_pdf_crop,
)


def test_extract_figure_legends_and_numeric_evidence():
    text = """
    [Page 3]
    Fig. 1 Overview of TdT-mediated enzymatic DNA synthesis.
    The reaction generated products up to 120 nt after 30 min at 37 °C.
    [Page 4]
    Table 1 Comparison of variants. Mutant A reached 86% conversion, while wild-type reached 41%.
    """
    legends = extract_figure_legends(text)
    evidence = extract_numeric_evidence(text, legends)

    assert [item.figure_id for item in legends] == ["Fig. 1", "Table 1"]
    assert any(item.value == "120 nt" and item.figure_id == "Fig. 1" for item in evidence)
    assert any(item.value == "86%" and item.figure_id == "Table 1" for item in evidence)


def test_choose_key_figures_marks_manual_check_without_data():
    text = "Fig. 2 Workflow for enzyme engineering and screening without numeric values."
    legends = extract_figure_legends(text)
    selected = choose_key_figures(legends, [])

    assert selected
    assert selected[0].needs_manual_check is True


def test_extract_extended_supplementary_and_scheme_legends():
    text = """
    [Page 8]
    Extended Data Figure 2 Validation across three cohorts with 91% agreement.
    [Page 9]
    Supplementary Table 4 Detailed sample characteristics for 120 participants.
    [Page 10]
    Scheme 1 Proposed reaction mechanism and catalyst cycle.
    """

    legends = extract_figure_legends(text)

    assert [item.figure_id for item in legends] == [
        "Extended Data Fig. 2",
        "Supplementary Table 4",
        "Scheme 1",
    ]
    assert [item.page for item in legends] == ["8", "9", "10"]


def test_parse_pdf_auto_falls_back_to_pypdf(monkeypatch):
    monkeypatch.setattr(
        "weixin_lite.pdf_reader.extract_markdown_with_pymupdf4llm",
        lambda pdf_bytes: (_ for _ in ()).throw(RuntimeError("layout unavailable")),
    )
    monkeypatch.setattr(
        "weixin_lite.pdf_reader.extract_text_with_pypdf",
        lambda pdf_bytes: "[Page 1]\nAbstract\nFallback text with 90% accuracy.",
    )
    monkeypatch.setattr("weixin_lite.pdf_reader.pypdf_page_count", lambda pdf_bytes: 1)
    monkeypatch.setattr("weixin_lite.pdf_reader.build_figure_crops", lambda *args: (None, {}))

    parsed = parse_pdf(b"not-a-real-pdf", mode="auto")

    assert parsed.parser == "pypdf"
    assert parsed.parse_mode == "pypdf"
    assert parsed.page_count == 1
    assert len(parsed.hash) == 64
    assert "fell back to pypdf" in parsed.warning


def test_crop_bbox_adjustment_and_rendering():
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    page = document.new_page(width=300, height=400)
    page.insert_text((30, 60), "Crop test")
    pdf_bytes = document.tobytes()

    adjusted = adjust_crop_bbox((0.1, 0.1, 0.9, 0.9), left=0.05, top=0.05, right=-0.05)
    image = render_pdf_crop(pdf_bytes, 1, adjusted, scale=1.0)

    assert adjusted == pytest.approx((0.15, 0.15, 0.85, 0.9))
    assert image.startswith(b"\x89PNG")
    with pytest.raises(ValueError):
        adjust_crop_bbox((0.4, 0.4, 0.5, 0.5), left=0.2)


def test_low_confidence_figure_crop_requires_manual_confirmation(monkeypatch):
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    document.new_page(width=300, height=400)
    pdf_bytes = document.tobytes()
    figure = FigureAnalysis(
        figure_id="Fig. 1",
        caption="Fig. 1 Overview.",
        page="1",
        selected=True,
    )

    monkeypatch.setattr("weixin_lite.pdf_reader._lead_crop", lambda *args: None)
    monkeypatch.setattr("weixin_lite.pdf_reader._candidate_crop", lambda *args: ((0.0, 0.0, 1.0, 1.0), 0.4))
    monkeypatch.setattr("weixin_lite.pdf_reader.render_pdf_crop", lambda *args, **kwargs: b"png")

    _lead, images = build_figure_crops(pdf_bytes, [figure], "digest")

    assert images == {"digest-fig-1.png": b"png"}
    assert figure.needs_manual_crop is True
    assert figure.selected is False


def test_pdf_and_figure_evidence_serialization_round_trip():
    figure = FigureAnalysis(
        figure_id="Fig. 2",
        caption="Result",
        page="4",
        crop_bbox=(0.1, 0.2, 0.9, 0.8),
        confidence=0.88,
        role="key_result",
        selected=True,
        order=1,
        evidence=[EvidenceItem(claim="Conversion improved", value="90%", page="4", figure_id="Fig. 2")],
    )
    content = PdfContent(
        text="paper",
        legends=[figure],
        all_figures=[figure],
        evidence=figure.evidence,
        hash="abc",
        page_count=5,
        parse_mode="enhanced",
        quality="high",
        coverage=["abstract", "results"],
        lead_image=FigureAnalysis("Lead", "Title", page="1"),
    )

    restored = PdfContent.from_dict(content.to_dict())

    assert restored.legends[0].crop_bbox == (0.1, 0.2, 0.9, 0.8)
    assert restored.legends[0].evidence[0].value == "90%"
    assert restored.lead_image and restored.lead_image.figure_id == "Lead"
    assert restored.coverage == ["abstract", "results"]
