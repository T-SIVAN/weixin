from __future__ import annotations

import csv
from datetime import datetime
import html as html_lib
import io
import json
import os
import re
import zipfile
from typing import Any
from urllib import request

from .models import BatchProject, DownloadedPaper, PaperInput, QuickReadArticle


def safe_slug(value: str, fallback: str = "article") -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value or "").strip("-")
    return (slug or fallback)[:80]


ARTICLE_CANVAS_STYLE = (
    "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;"
    "max-width:760px;margin:0 auto;padding:0 10px 44px;background:#fff;color:#000;"
)
ARTICLE_TITLE_STYLE = "margin:0 0 24px;font-size:28px;line-height:1.22;color:#000;font-weight:800;"
ARTICLE_META_STYLE = "margin:0 0 42px;font-size:15px;line-height:1.7;color:#a0a4aa;"
ARTICLE_ORIGINAL_STYLE = (
    "display:inline-block;margin:0 10px 0 0;padding:1px 6px;border-radius:2px;"
    "background:#f3f3f3;color:#a0a4aa;font-size:14px;vertical-align:1px;"
)
ARTICLE_ACCOUNT_STYLE = "color:#576b95;text-decoration:none;margin:0 12px 0 8px;"
ARTICLE_AUTHOR_STYLE = "color:#a0a4aa;margin:0;"
ARTICLE_COVER_STYLE = "margin:0 0 42px;"
WECHAT_CONTENT_STYLE = (
    "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;"
    "background:#fff;color:#000;line-height:2.05;font-size:18px;"
)


def _format_article_time(value: str) -> str:
    try:
        parsed = datetime.fromisoformat((value or "").replace("Z", "+00:00"))
    except ValueError:
        return ""
    return f"{parsed.year}年{parsed.month}月{parsed.day}日 {parsed.hour:02d}:{parsed.minute:02d}"


def _wechat_author(author: str = "") -> str:
    return author or os.getenv("WECHAT_AUTHOR_NAME", "陶小花")


def _wechat_account(account_name: str = "") -> str:
    return account_name or os.getenv("WECHAT_ACCOUNT_NAME", "遇见生物合成")


def article_platform_header_html(article: QuickReadArticle, *, author: str = "", account_name: str = "") -> str:
    published_at = _format_article_time(article.created_at)
    date_html = f'<span>{html_lib.escape(published_at)}</span>' if published_at else ""
    return (
        f'<h1 style="{ARTICLE_TITLE_STYLE}">{html_lib.escape(article.title)}</h1>\n'
        f'<section style="{ARTICLE_META_STYLE}">'
        f'<span style="{ARTICLE_ORIGINAL_STYLE}">原创</span>'
        f'<span style="{ARTICLE_AUTHOR_STYLE}">{html_lib.escape(_wechat_author(author))}</span>'
        f'<span style="{ARTICLE_ACCOUNT_STYLE}">{html_lib.escape(_wechat_account(account_name))}</span>'
        f"{date_html}"
        "</section>"
    )


def wechat_content_html(article: QuickReadArticle, content_html: str | None = None, *, include_cover: bool = False) -> str:
    cover = ""
    if include_cover and article.cover_image_name:
        cover = (
            f'<section style="{ARTICLE_COVER_STYLE}">'
            f'<img src="images/{html_lib.escape(article.cover_image_name)}" alt="publisher title image" '
            'style="width:100%;height:auto;display:block;margin:0 auto;">'
            "</section>\n"
        )
    return f'<section style="{WECHAT_CONTENT_STYLE}">\n{cover}{content_html or article.body_html}\n</section>'


def article_document_html(article: QuickReadArticle, *, author: str = "", account_name: str = "") -> str:
    return (
        f'<section style="{ARTICLE_CANVAS_STYLE}">\n'
        f"{article_platform_header_html(article, author=author, account_name=account_name)}\n"
        f"{wechat_content_html(article, include_cover=True)}\n"
        "</section>"
    )


def article_html(article: QuickReadArticle) -> str:
    content = article_document_html(article)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{article.title}</title>
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
            zf.writestr(f"articles/{index:02d}-{slug}.md", article.body_markdown)
            zf.writestr(f"articles/{index:02d}-{slug}.html", article_html(article))
            zf.writestr(
                f"evidence/{index:02d}-{slug}.json",
                json.dumps(
                    {
                        "paper": article.paper.to_dict(),
                        "figures": [figure.to_dict() for figure in article.figures],
                        "evidence": [item.to_dict() for item in article.evidence],
                        "warnings": article.warnings,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        for name, data in assets.items():
            if data:
                zf.writestr(f"images/{safe_slug(name, 'image')}", data)
    return buffer.getvalue()


def export_article_markdown(article: QuickReadArticle) -> bytes:
    return article.body_markdown.encode("utf-8")


def export_article_html(article: QuickReadArticle) -> bytes:
    return article_html(article).encode("utf-8")


def post_to_bridge(bridge_url: str, article: QuickReadArticle, token: str = "") -> dict[str, Any]:
    payload = {
        "title": article.title,
        "digest": article.digest,
        "content_html": article_html(article),
        "source_url": article.paper.url,
        "doi": article.paper.doi,
        "warnings": article.warnings,
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
