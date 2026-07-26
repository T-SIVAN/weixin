import json
import zipfile
from io import BytesIO

from weixin_lite.downloader import download_open_access
from weixin_lite.exporter import project_zip, unavailable_dois_csv
from weixin_lite.generator import generate_article
from weixin_lite.llm import default_base_url, default_model
from weixin_lite.models import BatchProject, PaperInput, generation_ready_papers, unavailable_papers
from weixin_lite.pdf_reader import PdfContent, extract_figure_legends, extract_numeric_evidence
from weixin_lite.search import build_keyword_query, crossref_publication_date, crossref_year, run_keyword_search, year_from
from weixin_lite.translate import translate_records
from weixin_lite.wechat_publish import WechatDraftConfig, publish_draft


def test_keyword_query_expands_simple_keywords():
    query = build_keyword_query(["TdT", "酶促DNA合成"])

    assert "terminal deoxynucleotidyl transferase" in query
    assert "enzymatic DNA synthesis" in query


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


def test_translation_fallback_populates_chinese_fields():
    paper = PaperInput(title_en="A test title", abstract_en="A test abstract.")

    report = translate_records([paper])

    assert "待翻译标题" in paper.title_zh
    assert "待翻译摘要" in paper.abstract_zh
    assert report.errors


def test_provider_defaults_are_configured():
    assert default_base_url("deepseek") == "https://api.deepseek.com/v1"
    assert default_model("deepseek") == "deepseek-chat"
    assert default_base_url("siliconflow") == "https://api.siliconflow.cn/v1"


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
