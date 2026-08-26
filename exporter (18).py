from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from typing import Any
from urllib import request

from .models import BatchProject, DownloadedPaper, PaperInput, QuickReadArticle


def safe_slug(value: str, fallback: str = "article") -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value or "").strip("-")
    return (slug or fallback)[:80]


def article_html(article: QuickReadArticle) -> str:
    cover = ""
    if article.cover_image_name:
        cover = f'<p style="margin:0 0 22px;"><img src="images/{article.cover_image_name}" alt="publisher title image" style="width:100%;height:auto;display:block;"></p>\n'
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{article.title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; line-height: 1.95; max-width: 760px; margin: 0 auto 40px; color: #0f172a; background: #fff; }}
    h2 {{ font-size: 24px; line-height: 1.45; margin: 18px 0 18px; font-weight: 800; }}
    h3 {{ font-size: 20px; line-height: 1.55; margin: 26px 0 14px; font-weight: 800; }}
    p {{ font-size: 17px; line-height: 1.95; margin: 16px 0; }}
    img {{ max-width: 100%; height: auto; display: block; margin: 24px auto 10px; }}
    strong {{ color: #000; font-weight: 800; }}
  </style>
</head>
<body>
{cover}{article.body_html}
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
