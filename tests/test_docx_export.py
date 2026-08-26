from __future__ import annotations

import io
import zipfile

import pytest
from docx import Document
from PIL import Image

import app
from weixin_lite.docx_exporter import DocxExportError, export_article_docx
from weixin_lite.exporter import export_article_html, project_zip
from weixin_lite.models import BatchProject, PaperInput, QuickReadArticle


def _png_bytes(color: str = "white") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (48, 32), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _article(markdown: str) -> QuickReadArticle:
    return QuickReadArticle(
        paper=PaperInput(title_zh="测试论文"),
        title="可编辑论文解读",
        digest="摘要",
        body_markdown=markdown,
        body_html="<p>预览正文</p>",
        lead_image_name="lead.png",
    )


def test_docx_embeds_images_and_keeps_text_editable_with_single_lead():
    article = _article(
        "# 可编辑论文解读\n\n"
        "![论文首页](images/lead.png)\n\n"
        "导语中的**关键术语**可编辑。\n\n"
        "## 核心发现\n\n"
        "- 第一项\n"
        "1. 第二项\n\n"
        "![关键结果](images/figure.png)\n\n"
        "![重复首页](images/lead.png)\n"
    )

    payload = export_article_docx(article, {"lead.png": _png_bytes(), "figure.png": _png_bytes("blue")})
    document = Document(io.BytesIO(payload))
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)

    assert "可编辑论文解读" in text
    assert "关键术语" in text
    assert "第一项" in text
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        media = [name for name in archive.namelist() if name.startswith("word/media/")]
    assert len(media) == 2


def test_docx_rejects_missing_image_by_name():
    article = _article("# 可编辑论文解读\n\n![缺图](images/missing.png)")

    with pytest.raises(DocxExportError, match="missing.png"):
        export_article_docx(article, {"lead.png": _png_bytes()})


def test_docx_accepts_standalone_image_with_optional_title_and_chinese_filename():
    article = _article('![关键图](images/实验 图.png "关键结果")')
    payload = export_article_docx(article, {"lead.png": _png_bytes(), "实验 图.png": _png_bytes("blue")})

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        media = [name for name in archive.namelist() if name.startswith("word/media/")]
    assert len(media) == 2


def test_docx_missing_optional_title_image_reports_normalized_asset_name_only():
    article = _article('![关键图](<images/中文 图.png> "关键结果")')

    with pytest.raises(DocxExportError) as error:
        export_article_docx(article, {"lead.png": _png_bytes()})

    assert str(error.value) == "无法导出 Word：正文引用的图片缺失：中文 图.png"


def test_docx_inline_local_image_is_rejected_without_silently_dropping_text():
    article = _article("这段文字中有图 ![结果](images/中文 图.png) ，但不是独占一行。")

    with pytest.raises(DocxExportError, match="Markdown 图片必须独占一行.*中文 图.png"):
        export_article_docx(article, {"lead.png": _png_bytes(), "中文 图.png": _png_bytes("blue")})


def test_project_zip_failure_is_returned_as_actionable_error(monkeypatch):
    project = BatchProject(topic="测试")

    def fail(*_args, **_kwargs):
        raise DocxExportError("无法导出 Word：正文引用的图片缺失：中文 图.png")

    monkeypatch.setattr(app, "project_zip", fail)
    data, error = app.build_project_zip_download(project, {}, [])

    assert data is None
    assert error == "项目包暂不能导出：无法导出 Word：正文引用的图片缺失：中文 图.png。请返回“内容与生成”补齐图片后重试。"


def test_standalone_html_uses_data_uris_and_rejects_missing_assets():
    article = _article("正文")
    article.body_html = '<p>正文</p><img src="images/lead.png" alt="论文首页">'

    html = export_article_html(article, {"lead.png": _png_bytes()}).decode("utf-8")
    assert "data:image/png;base64," in html
    assert 'src="images/lead.png"' not in html

    with pytest.raises(DocxExportError, match="lead.png"):
        export_article_html(article)


def test_project_zip_includes_primary_docx_and_rejects_missing_html_images():
    article = _article("# 可编辑论文解读\n\n![论文首页](images/lead.png)")
    article.body_html = '<img src="images/lead.png" alt="论文首页">'
    archive_bytes = project_zip(BatchProject(topic="测试", articles=[article]), {"lead.png": _png_bytes()})

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        docx_name = next(name for name in archive.namelist() if name.endswith(".docx"))
        with zipfile.ZipFile(io.BytesIO(archive.read(docx_name))) as docx:
            assert any(name.startswith("word/media/") for name in docx.namelist())

    article.body_html = '<img src="images/missing.png" alt="缺图">'
    with pytest.raises(DocxExportError, match="missing.png"):
        project_zip(BatchProject(topic="测试", articles=[article]), {"lead.png": _png_bytes()})
