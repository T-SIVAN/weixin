import json
import zipfile
from io import BytesIO

from weixin_lite.downloader import download_open_access
from weixin_lite.exporter import project_zip
from weixin_lite.generator import generate_article
from weixin_lite.models import BatchProject, PaperInput
from weixin_lite.pdf_reader import PdfContent, extract_figure_legends, extract_numeric_evidence
from weixin_lite.search import build_keyword_query, run_keyword_search
from weixin_lite.translate import translate_records


def test_keyword_query_expands_simple_keywords():
    query = build_keyword_query(["TdT", "酶促DNA合成"])

    assert "terminal deoxynucleotidyl transferase" in query
    assert "enzymatic DNA synthesis" in query


def test_translation_fallback_populates_chinese_fields():
    paper = PaperInput(title_en="A test title", abstract_en="A test abstract.")

    translate_records([paper])

    assert "待翻译标题" in paper.title_zh
    assert "待翻译摘要" in paper.abstract_zh


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
    assert "English abstract" not in article.body_markdown
    assert article.evidence


def test_project_zip_contains_paywalled_and_download_status():
    paper = PaperInput(title_en="A paper", doi="10.1000/a", access_status="paywalled")
    article = generate_article(paper)
    project = BatchProject(topic="secret-key-should-not-appear", papers=[paper], articles=[article])

    data = project_zip(project, {"cover.png": b"image"})
    with zipfile.ZipFile(BytesIO(data)) as zf:
        names = set(zf.namelist())
        payload = zf.read("project.json").decode("utf-8")
        paywalled = zf.read("paywalled_dois.csv").decode("utf-8")
        latest = json.loads(zf.read("latest_papers.json").decode("utf-8"))

    assert any(name.startswith("articles/") and name.endswith(".html") for name in names)
    assert "images/cover.png" in names
    assert "paywalled_dois.csv" in names
    assert "download_status.json" in names
    assert "10.1000/a" in paywalled
    assert latest[0]["doi"] == "10.1000/a"
    assert "sk-" not in payload
