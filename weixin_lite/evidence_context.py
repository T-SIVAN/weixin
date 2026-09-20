"""Locate figure-specific source passages without assuming captions share a page."""
from __future__ import annotations

import re
import unicodedata

from .models import FigureAnalysis
from .pdf_reader import PdfContent, figure_key


def normalized_excerpt(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    return re.sub(r"\s+", "", text.replace(">", "").replace("**", "")).strip('"\'“”‘’').casefold()


def excerpt_matches(excerpt: str, source: str) -> bool:
    value = normalized_excerpt(excerpt)
    return len(value) >= 8 and value in normalized_excerpt(source)


def figure_context(pdf: PdfContent | None, figure: FigureAnalysis, budget: int = 12000) -> str:
    if pdf is None:
        return ""
    pages = re.split(r"\[Page\s+(\d+)\]", pdf.text, flags=re.I)
    records = [(pages[i], pages[i + 1]) for i in range(1, len(pages) - 1, 2)]
    if not records:
        records = [("?", pdf.text)]
    identifier = re.search(r"(?i)(fig(?:ure)?\.?|table|scheme)\s*([A-Za-z]?\d+)", figure.figure_id)
    if identifier:
        kind = r"fig(?:ure)?s?\.?" if identifier[1].lower().startswith("fig") else re.escape(identifier[1]) + "s?"
        prefix = r"(?:extended\s+data|supplementary)\s+" if re.search(r"(?i)extended|supplementary", figure.figure_id) else ""
        reference = re.compile(rf"(?i)\b{prefix}{kind}\s*{re.escape(identifier[2])}(?!\d)[a-z]?\b")
    else:
        reference = re.compile(re.escape(figure.figure_id), re.I)
    passages: list[tuple[int, str, str]] = []
    for page, text in records:
        paragraphs = re.split(r"\n\s*\n", text)
        for paragraph in paragraphs:
            # Some extractors do not preserve paragraphs; bound windows around references.
            hits = list(reference.finditer(paragraph))
            for hit in hits:
                if identifier and not prefix and re.search(r"(?i)(extended\s+data|supplementary)\s*$", paragraph[max(0, hit.start() - 35):hit.start()]):
                    continue
                snippet = paragraph[max(0, hit.start() - 450):hit.end() + 1600].strip()
                if snippet:
                    passages.append((0, page, snippet))
        if page == str(figure.page):
            passages.append((1, page, text[:3500].strip()))
    seen: set[str] = set()
    output: list[str] = []
    remaining = budget
    for _, page, passage in sorted(passages, key=lambda item: item[0]):
        key = normalized_excerpt(passage)
        if not key or key in seen or remaining < 100:
            continue
        seen.add(key)
        block = f"[Page {page}]\n{passage}"[:remaining]
        output.append(block)
        remaining -= len(block) + 2
    return "\n\n".join(output)


def prose_from_note(note: str) -> str:
    """Render legacy analytical labels as reader-facing paragraphs."""
    note = re.sub(r"(?m)^\*\*证据级别：.*?\*\*\s*", "", note)
    return re.sub(r"\*\*(?:图展示什么|关键数据或趋势|结论与证据边界)：\*\*", "", note).strip()
