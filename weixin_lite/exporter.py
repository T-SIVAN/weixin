from __future__ import annotations

import io
import json
import re
import zipfile
from typing import Any
from urllib import request

from .models import BatchProject, QuickReadArticle


def safe_slug(value: str, fallback: str = "article") -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value or "").strip("-")
    return (slug or fallback)[:80]


def article_html(article: QuickReadArticle) -> str:
    cover = ""
    if article.cover_image_name:
        cover = f'<p><img src="images/{article.cover_image_name}" alt="publisher title image"></p>\n'
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{article.title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.85; max-width: 760px; margin: 32px auto; color: #1f2933; }}
    h2 {{ font-size: 22px; margin-top: 20px; }}
    h3 {{ font-size: 18px; margin-top: 24px; border-left: 4px solid #0f766e; padding-left: 10px; }}
    p {{ font-size: 16px; }}
    img {{ max-width: 100%; height: auto; display: block; margin: 16px auto; }}
    strong {{ color: #0f766e; }}
  </style>
</head>
<body>
{cover}{article.body_html}
</body>
</html>
"""


def project_zip(project: BatchProject, image_assets: dict[str, bytes] | None = None) -> bytes:
    assets = image_assets or {}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project.json", json.dumps(project.to_dict(), ensure_ascii=False, indent=2))
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
