from weixin_lite.pdf_reader import choose_key_figures, extract_figure_legends, extract_numeric_evidence


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
