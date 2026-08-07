from __future__ import annotations

import re
import urllib.error
import urllib.parse
import urllib.request

from .models import DownloadedPaper, PaperInput


MAX_DOWNLOAD_BYTES = 12_000_000


def safe_name(value: str, suffix: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", value or "paper").strip("-")[:80] or "paper"
    return f"{stem}.{suffix}"


def looks_like_pdf(content_type: str, url: str, data: bytes) -> bool:
    return "pdf" in content_type.lower() or url.lower().split("?", 1)[0].endswith(".pdf") or data.startswith(b"%PDF")


def download_open_access(paper: PaperInput, timeout: int = 45) -> DownloadedPaper:
    key = paper.paper_key
    if not paper.oa_pdf_url:
        return DownloadedPaper(
            paper_key=key,
            source_url="",
            status="paywalled",
            error="未发现合法开放全文链接，仅保留 DOI 和题录。",
        )

    request = urllib.request.Request(
        paper.oa_pdf_url,
        headers={"User-Agent": "weixin-paper-radar/0.2", "Accept": "application/pdf,text/html;q=0.8,*/*;q=0.5"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            data = response.read(MAX_DOWNLOAD_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        return DownloadedPaper(
            paper_key=key,
            source_url=paper.oa_pdf_url,
            status="download_failed",
            error=f"开放全文下载失败：{exc}",
        )

    if len(data) > MAX_DOWNLOAD_BYTES:
        return DownloadedPaper(
            paper_key=key,
            source_url=paper.oa_pdf_url,
            status="download_failed",
            error="开放全文超过 12 MB，已跳过自动下载。",
        )

    suffix = "pdf" if looks_like_pdf(content_type, paper.oa_pdf_url, data) else "html"
    return DownloadedPaper(
        paper_key=key,
        file_name=safe_name(paper.doi or paper.title_en or paper.title, suffix),
        content_type=content_type or ("application/pdf" if suffix == "pdf" else "text/html"),
        source_url=paper.oa_pdf_url,
        status="open",
        content_bytes=data,
    )


def download_many(papers: list[PaperInput]) -> list[DownloadedPaper]:
    return [download_open_access(paper) for paper in papers]
