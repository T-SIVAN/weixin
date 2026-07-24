import zipfile
from io import BytesIO

from weixin_lite.exporter import project_zip
from weixin_lite.generator import generate_article
from weixin_lite.models import BatchProject, PaperInput
from weixin_lite.pdf_reader import PdfContent, extract_figure_legends, extract_numeric_evidence


def test_fallback_article_has_required_sections():
    paper = PaperInput(
        title="Template-independent enzymatic DNA synthesis using terminal deoxynucleotidyl transferase",
        journal="Nature Biotechnology",
        year="2024",
        doi="10.1000/test",
        abstract="This study reports an enzymatic DNA synthesis method.",
    )
    text = "Fig. 1 The TdT reaction produced DNA up to 100 nt with 90% conversion."
    legends = extract_figure_legends(text)
    evidence = extract_numeric_evidence(text, legends)
    pdf = PdfContent(text=text, legends=legends, evidence=evidence, parser="fixture")

    article = generate_article(paper, pdf=pdf)

    assert "文章核心要点简述" in article.body_markdown
    assert "文章的创新意义" in article.body_markdown
    assert article.evidence


def test_project_zip_excludes_api_keys_and_contains_articles():
    paper = PaperInput(title="A paper", doi="10.1000/a")
    article = generate_article(paper)
    project = BatchProject(topic="secret-key-should-not-appear", papers=[paper], articles=[article])

    data = project_zip(project, {"cover.png": b"image"})
    with zipfile.ZipFile(BytesIO(data)) as zf:
        names = set(zf.namelist())
        payload = zf.read("project.json").decode("utf-8")

    assert any(name.startswith("articles/") and name.endswith(".html") for name in names)
    assert "images/cover.png" in names
    assert "sk-" not in payload
