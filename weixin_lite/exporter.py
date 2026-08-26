from __future__ import annotations

import base64
import csv
import html as html_lib
import io
import json
import mimetypes
import re
import zipfile
from typing import Any
from urllib import parse, request

from .docx_exporter import DocxExportError, export_article_docx, validate_article_images
from .models import BatchProject, DownloadedPaper, PaperInput, QuickReadArticle


def safe_slug(value: str, fallback: str = "article") -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value or "").strip("-")
    return (slug or fallback)[:80]


_IMG_TAG_RE = re.compile(r"<img\b[^>]*>", flags=re.I | re.S)
_SRC_ATTR_RE = re.compile(r"(?<![\w:-])src(\s*=\s*)([\"'])(.*?)\2", flags=re.I | re.S)


def _safe_asset_basename(name: str) -> str:
    decoded = parse.unquote(html_lib.unescape(str(name or ""))).replace("\\", "/")
    basename = decoded.rsplit("/", 1)[-1].strip()
    basename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", basename).rstrip(". ")
    return basename if basename not in {"", ".", ".."} else "image"


def _asset_archive_entries(assets: dict[str, bytes]) -> list[tuple[str, str, bytes]]:
    entries: list[tuple[str, str, bytes]] = []
    used: set[str] = set()
    for source_name in sorted(assets, key=lambda value: str(value).replace("\\", "/").casefold()):
        data = assets[source_name]
        if not data:
            continue
        basename = _safe_asset_basename(source_name)
        stem, dot, suffix = basename.rpartition(".")
        if not stem:
            stem, dot, suffix = basename, "", ""
        candidate = basename
        counter = 2
        while candidate.casefold() in used:
            candidate = f"{stem}-{counter}{dot}{suffix}"
            counter += 1
        used.add(candidate.casefold())
        entries.append((str(source_name).replace("\\", "/"), candidate, data))
    return entries


def _rewrite_packaged_image_references(content: str, asset_names: dict[str, str]) -> str:
    def mapped_src(src: str) -> str | None:
        raw = html_lib.unescape(src).strip()
        path, marker, tail = raw.partition("?")
        if not marker:
            path, marker, tail = raw.partition("#")
        normalized = path.replace("\\", "/")
        for prefix in ("./images/", "/images/", "images/"):
            if normalized.startswith(prefix):
                source_name = parse.unquote(normalized[len(prefix) :])
                archived = asset_names.get(source_name)
                if archived is None:
                    return None
                trailer = f"{marker}{tail}" if marker else ""
                return f"images/{archived}{trailer}"
        return None

    def replace_tag(tag_match: re.Match[str]) -> str:
        tag = tag_match.group(0)
        attr = _SRC_ATTR_RE.search(tag)
        if not attr:
            return tag
        replacement = mapped_src(attr.group(3))
        if replacement is None:
            return tag
        escaped = html_lib.escape(replacement, quote=True)
        return tag[: attr.start(3)] + escaped + tag[attr.end(3) :]

    rewritten = _IMG_TAG_RE.sub(replace_tag, content or "")

    def replace_markdown(match: re.Match[str]) -> str:
        destination = match.group(2).strip()
        wrapped = destination.startswith("<") and destination.endswith(">")
        raw = destination[1:-1] if wrapped else destination
        replacement = mapped_src(raw)
        if replacement is None:
            return match.group(0)
        value = f"<{replacement}>" if wrapped else replacement
        return f"{match.group(1)}{value}{match.group(3)}"

    return re.sub(r"(!\[[^\]]*\]\()([^\r\n)]*)(\))", replace_markdown, rewritten)


ARTICLE_CANVAS_STYLE = (
    "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;"
    "width:100%;max-width:760px;box-sizing:border-box;margin:0 auto;padding:0 14px 44px;"
    "background:#fff;color:#000;"
)
ARTICLE_LEAD_IMAGE_STYLE = "margin:0 0 42px;"
# Kept for callers that imported the old constant. Platform covers are no longer
# inserted into the article body.
ARTICLE_COVER_STYLE = ARTICLE_LEAD_IMAGE_STYLE
WECHAT_CONTENT_STYLE = (
    "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;"
    "background:#fff;color:#000;font-size:18px;line-height:2.05;word-break:break-word;"
)


def _local_image_names(content_html: str) -> set[str]:
    names: set[str] = set()
    for tag_match in _IMG_TAG_RE.finditer(content_html or ""):
        attr = _SRC_ATTR_RE.search(tag_match.group(0))
        if not attr:
            continue
        src = html_lib.unescape(attr.group(3)).strip().replace("\\", "/")
        for prefix in ("./images/", "/images/", "images/"):
            if src.startswith(prefix):
                name = src[len(prefix) :].split("?", 1)[0].split("#", 1)[0]
                names.add(parse.unquote(name))
                break
    return names


def _lead_image_html(article: QuickReadArticle, content_html: str) -> str:
    lead_image_name = str(getattr(article, "lead_image_name", "") or "").strip()
    if not lead_image_name or lead_image_name in _local_image_names(content_html):
        return ""
    escaped_name = html_lib.escape(lead_image_name, quote=True)
    return (
        f'<section style="{ARTICLE_LEAD_IMAGE_STYLE}">'
        f'<img src="images/{escaped_name}" alt="论文首页" '
        'style="width:100%;height:auto;display:block;margin:0 auto;">'
        "</section>\n"
    )


def wechat_content_html(article: QuickReadArticle, content_html: str | None = None, *, include_cover: bool = False) -> str:
    # include_cover remains in the signature for API compatibility. A WeChat
    # platform cover is a payload asset and must never be duplicated in content.
    _ = include_cover
    body = article.body_html if content_html is None else content_html
    lead = _lead_image_html(article, body)
    return f'<section style="{WECHAT_CONTENT_STYLE}">\n{lead}{body}\n</section>'


def article_document_html(
    article: QuickReadArticle,
    content_html: str | None = None,
    *,
    author: str = "",
    account_name: str = "",
) -> str:
    _ = (author, account_name)
    return (
        f'<section style="{ARTICLE_CANVAS_STYLE}">\n'
        f"{wechat_content_html(article, content_html=content_html)}\n"
        "</section>"
    )


def article_html(article: QuickReadArticle) -> str:
    content = article_document_html(article)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html_lib.escape(article.title)}</title>
  <style>
    html, body {{ margin:0; padding:0; background:#fff; }}
    body {{ padding:0 0 60px; }}
    img {{ max-width:100%; }}
  </style>
</head>
<body>
{content}
</body>
</html>
"""


def _local_src_name(src: str) -> str | None:
    raw = html_lib.unescape(src or "").strip()
    path = raw.split("?", 1)[0].split("#", 1)[0].replace("\\", "/")
    for prefix in ("./images/", "/images/", "images/"):
        if path.startswith(prefix):
            name = parse.unquote(path[len(prefix) :])
            if name and ".." not in name.split("/"):
                return name
    return None


def _portable_html(content: str, image_assets: dict[str, bytes]) -> str:
    """Inline local images for a standalone HTML file without changing WeChat paths."""
    missing: set[str] = set()

    def replace_tag(tag_match: re.Match[str]) -> str:
        tag = tag_match.group(0)
        attr = _SRC_ATTR_RE.search(tag)
        if not attr:
            return tag
        name = _local_src_name(attr.group(3))
        if not name:
            return tag
        data = image_assets.get(name)
        if not data:
            missing.add(name)
            return tag
        mime_type = mimetypes.guess_type(name)[0] or "image/png"
        uri = f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"
        return tag[: attr.start(3)] + uri + tag[attr.end(3) :]

    portable = _IMG_TAG_RE.sub(replace_tag, content or "")
    if missing:
        raise DocxExportError("无法导出便携 HTML：正文引用的图片缺失：" + "、".join(sorted(missing)))
    return portable


def _validate_html_image_assets(article: QuickReadArticle, image_assets: dict[str, bytes] | None) -> None:
    """Reject exports that would leave any local article image unresolved."""
    assets = image_assets or {}
    missing = sorted(name for name in _local_image_names(article_html(article)) if not assets.get(name))
    if missing:
        raise DocxExportError("无法导出：正文引用的图片缺失：" + "、".join(missing))


def unavailable_dois_csv(papers: list[PaperInput]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["doi", "title", "journal", "publication_date", "year", "url", "access_status", "download_error"],
    )
    writer.writeheader()
    for paper in papers:
        if paper.access_status != "open" or not paper.pdf_name:
            writer.writerow(
                {
                    "doi": paper.doi,
                    "title": paper.title_en or paper.title,
                    "journal": paper.journal,
                    "publication_date": paper.publication_date,
                    "year": paper.year,
                    "url": paper.url,
                    "access_status": paper.access_status,
                    "download_error": paper.download_error or ("未解析到 PDF 全文。" if paper.access_status == "open" else ""),
                }
            )
    return buffer.getvalue()


def paywalled_csv(papers: list[PaperInput]) -> str:
    return unavailable_dois_csv(papers)


def project_zip(
    project: BatchProject,
    image_assets: dict[str, bytes] | None = None,
    downloads: list[DownloadedPaper] | None = None,
) -> bytes:
    assets = image_assets or {}
    asset_entries = _asset_archive_entries(assets)
    asset_names = {source_name: archived_name for source_name, archived_name, _data in asset_entries}
    download_items = downloads if downloads is not None else project.downloads
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project.json", json.dumps(project.to_dict(), ensure_ascii=False, indent=2))
        unavailable_csv = unavailable_dois_csv(project.papers)
        zf.writestr("unavailable_dois.csv", unavailable_csv)
        zf.writestr("paywalled_dois.csv", unavailable_csv)
        zf.writestr(
            "download_status.json",
            json.dumps([item.to_dict() for item in download_items], ensure_ascii=False, indent=2),
        )
        zf.writestr("latest_papers.json", json.dumps([paper.to_dict() for paper in project.papers], ensure_ascii=False, indent=2))
        for index, article in enumerate(project.articles, start=1):
            slug = safe_slug(article.title, f"article-{index:02d}")
            # Fail early: a ZIP with a text-only final manuscript is not a valid
            # project export, even if its auxiliary Markdown can still be read.
            validate_article_images(article, assets)
            _validate_html_image_assets(article, assets)
            markdown = _rewrite_packaged_image_references(article.body_markdown, asset_names)
            rendered_html = _rewrite_packaged_image_references(article_html(article), asset_names)
            docx_bytes = export_article_docx(article, assets)
            zf.writestr(f"articles/{index:02d}-{slug}.docx", docx_bytes)
            zf.writestr(f"articles/{index:02d}-{slug}.md", markdown)
            zf.writestr(f"articles/{index:02d}-{slug}.html", rendered_html)
            zf.writestr(
                f"evidence/{index:02d}-{slug}.json",
                json.dumps(
                    {
                        "paper": article.paper.to_dict(),
                        "figures": [figure.to_dict() for figure in article.figures],
                        "evidence": [item.to_dict() for item in article.evidence],
                        "warnings": article.warnings,
                        "source_hash": str(getattr(article, "source_hash", "") or ""),
                        "analysis_version": str(getattr(article, "analysis_version", "") or ""),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        for _source_name, archived_name, data in asset_entries:
            zf.writestr(f"images/{archived_name}", data)
    return buffer.getvalue()


def export_article_markdown(article: QuickReadArticle) -> bytes:
    return article.body_markdown.encode("utf-8")


def export_article_html(article: QuickReadArticle, image_assets: dict[str, bytes] | None = None) -> bytes:
    """Export standalone HTML; supplied local assets are embedded as data URIs."""
    content = article_html(article)
    _validate_html_image_assets(article, image_assets)
    content = _portable_html(content, image_assets or {})
    return content.encode("utf-8")


def export_portable_article_html(article: QuickReadArticle, image_assets: dict[str, bytes]) -> bytes:
    """Export a self-contained HTML file for users who need compatibility output."""
    return export_article_html(article, image_assets)


def export_article_docx_bytes(article: QuickReadArticle, image_assets: dict[str, bytes]) -> bytes:
    """Compatibility export surface used by Streamlit and callers of exporter.py."""
    return export_article_docx(article, image_assets)


def post_to_bridge(bridge_url: str, article: QuickReadArticle, token: str = "") -> dict[str, Any]:
    payload = {
        "title": article.title,
        "digest": article.digest,
        "content_html": article_html(article),
        "source_url": article.paper.url,
        "doi": article.paper.doi,
        "warnings": article.warnings,
        "source_hash": str(getattr(article, "source_hash", "") or ""),
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = request.Request(bridge_url, data=data, headers=headers, method="POST")
    with request.urlopen(req, timeout=60) as response:
        text = response.read().decode(response.headers.get_content_charset() or "utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"ok": True, "raw": text}
