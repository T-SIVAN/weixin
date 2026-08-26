from __future__ import annotations

import io
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from .models import EvidenceItem, FigureAnalysis


FIGURE_MARKER_RE = re.compile(
    r"(?is)\b((?:fig(?:ure)?\.?|scheme|table)\s*\d+[a-z]?)\b[\s:.\-–—]+"
)

NUMERIC_RE = re.compile(
    r"(?i)(?:\b\d+(?:\.\d+)?\s*(?:%|bp|nt|kb|Mb|nM|uM|µM|mM|M|mg/L|g/L|"
    r"min|h|s|°C|fold|x|×|copies|bases|residues|cycles)(?![A-Za-z0-9])|"
    r"\b\d+(?:\.\d+)?\s*(?:to|-|–)\s*\d+(?:\.\d+)?\s*(?:bp|nt|%|h|min)(?![A-Za-z0-9]))"
)

SECTION_RE = re.compile(
    r"(?im)^\s*(abstract|introduction|results?|discussion|conclusion|methods?|"
    r"materials and methods|supplementary)\s*$"
)


@dataclass
class PdfContent:
    text: str = ""
    sections: dict[str, str] = field(default_factory=dict)
    legends: list[FigureAnalysis] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    rendered_images: dict[str, bytes] = field(default_factory=dict)
    parser: str = ""
    warning: str = ""

    def prompt_text(self, max_chars: int = 18000) -> str:
        parts: list[str] = [f"Parser: {self.parser}"]
        if self.warning:
            parts.append(f"Warning: {self.warning}")
        if self.legends:
            parts.append("\nFigure/Table captions:")
            for item in self.legends[:30]:
                page = f" page {item.page}" if item.page else ""
                parts.append(f"- {item.figure_id}{page}: {item.caption}")
        if self.evidence:
            parts.append("\nNumeric evidence candidates:")
            for item in self.evidence[:40]:
                parts.append(f"- {item.value} | {item.claim} | {item.label()}")
        if self.sections:
            parts.append("\nMain sections:")
            for name, value in self.sections.items():
                parts.append(f"\n## {name}\n{compact_text(value, 4500)}")
        else:
            parts.append("\nText:")
            parts.append(self.text)
        return compact_text("\n".join(parts), max_chars)


def compact_text(text: Any, max_chars: int = 24000) -> str:
    cleaned = re.sub(r"[ \t]+", " ", str(text or ""))
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 20].rstrip() + "\n...[truncated]"


def selected_page_indexes(page_count: int) -> list[int]:
    if page_count <= 0:
        return []
    indexes = {0, 1, 2, page_count - 2, page_count - 1}
    return sorted(index for index in indexes if 0 <= index < page_count)


def normalize_figure_id(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"(?i)^figure\b", "Fig.", text)
    text = re.sub(r"(?i)^fig\b\.?", "Fig.", text)
    text = re.sub(r"(?i)^table\b", "Table", text)
    text = re.sub(r"(?i)^scheme\b", "Scheme", text)
    return text


def figure_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_figure_id(value).lower())


def extract_page_number_near(text: str, position: int) -> str:
    before = text[max(0, position - 800) : position]
    matches = list(re.finditer(r"\[Page\s+(\d+)\]", before, flags=re.I))
    return matches[-1].group(1) if matches else ""


def trim_caption(segment: str, max_chars: int = 1200) -> str:
    segment = re.sub(r"\[Page\s+\d+\]", " ", segment, flags=re.I)
    segment = re.sub(r"\s+", " ", segment).strip()
    segment = re.sub(
        r"(?i)\s+(references|acknowledg(?:e)?ments|author contributions|competing interests)\s+.*$",
        "",
        segment,
    ).strip()
    return compact_text(segment, max_chars)


def extract_figure_legends(text: str, max_items: int = 60) -> list[FigureAnalysis]:
    matches = list(FIGURE_MARKER_RE.finditer(text or ""))
    legends: list[FigureAnalysis] = []
    seen: set[str] = set()
    for index, match in enumerate(matches):
        fig_id = normalize_figure_id(match.group(1))
        key = figure_key(fig_id)
        if not key or key in seen:
            continue
        next_start = matches[index + 1].start(1) if index + 1 < len(matches) else len(text)
        segment = text[match.start(1) : min(next_start, match.start(1) + 1800)]
        caption = trim_caption(segment)
        if len(caption) < len(fig_id) + 25:
            continue
        seen.add(key)
        legends.append(
            FigureAnalysis(
                figure_id=fig_id,
                caption=caption,
                page=extract_page_number_near(text, match.start(1)),
            )
        )
        if len(legends) >= max_items:
            break
    return legends


def extract_sections(text: str, max_chars_each: int = 6000) -> dict[str, str]:
    matches = list(SECTION_RE.finditer(text or ""))
    if not matches:
        return {}
    sections: dict[str, str] = {}
    for idx, match in enumerate(matches):
        name = match.group(1).lower()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = compact_text(text[start:end], max_chars_each)
        if body and name not in sections:
            sections[name] = body
    return sections


def extract_numeric_evidence(text: str, legends: list[FigureAnalysis], max_items: int = 50) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []
    seen: set[str] = set()

    def add_candidates(source_text: str, source: str, page: str = "", figure_id: str = "") -> None:
        for match in NUMERIC_RE.finditer(source_text or ""):
            value = match.group(0)
            start = max(0, match.start() - 120)
            end = min(len(source_text), match.end() + 160)
            claim = compact_text(source_text[start:end], 320)
            key = f"{value}|{figure_id}|{claim[:80]}"
            if key in seen:
                continue
            seen.add(key)
            evidence.append(
                EvidenceItem(
                    claim=claim,
                    value=value,
                    source=source,
                    page=page,
                    figure_id=figure_id,
                    confidence="high" if figure_id else "medium",
                )
            )
            if len(evidence) >= max_items:
                return

    for legend in legends:
        add_candidates(legend.caption, "caption", legend.page, legend.figure_id)
        if len(evidence) >= max_items:
            return evidence
    add_candidates(text, "full text")
    return evidence[:max_items]


def choose_key_figures(
    legends: list[FigureAnalysis],
    evidence: list[EvidenceItem],
    max_figures: int = 4,
) -> list[FigureAnalysis]:
    if not legends:
        return []
    scores: dict[str, int] = {}
    for legend in legends:
        key = figure_key(legend.figure_id)
        caption_l = legend.caption.lower()
        score = 0
        score += 8 if re.search(r"\bfig\.?\s*1\b|workflow|scheme|overview|design", caption_l) else 0
        score += 5 if re.search(r"activity|efficiency|yield|conversion|accuracy|fidelity|length|synthesis", caption_l) else 0
        score += 4 if re.search(r"mutant|variant|engineering|directed evolution|screen", caption_l) else 0
        score += 3 if re.search(r"comparison|benchmark|control|wild-type|scale", caption_l) else 0
        score += sum(1 for item in evidence if figure_key(item.figure_id) == key)
        scores[key] = score
    ranked = sorted(legends, key=lambda item: (scores.get(figure_key(item.figure_id), 0), item.figure_id), reverse=True)
    selected = ranked[:max_figures]
    for item in selected:
        item.why_selected = "该图包含流程、性能、酶改造或对照数据，是支撑单篇解读的高信号图。"
        item.evidence = [ev for ev in evidence if figure_key(ev.figure_id) == figure_key(item.figure_id)][:6]
        if not item.evidence:
            item.needs_manual_check = True
    return selected


def extract_text_with_pypdf(pdf_bytes: bytes, max_pages: int | None = None) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = reader.pages if max_pages is None else reader.pages[:max_pages]
    chunks: list[str] = []
    for index, page in enumerate(pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            chunks.append(f"[Page {index}]\n{text}")
    return "\n\n".join(chunks)


def render_key_pages(pdf_bytes: bytes, figures: list[FigureAnalysis], max_pages: int = 4) -> dict[str, bytes]:
    try:
        import fitz
    except ImportError:
        return {}

    images: dict[str, bytes] = {}
    digest = hashlib.sha1(pdf_bytes).hexdigest()[:10]
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    wanted: list[int] = []
    for figure in figures:
        try:
            page = int(figure.page) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= page < doc.page_count and page not in wanted:
            wanted.append(page)
        if len(wanted) >= max_pages:
            break
    if not wanted:
        wanted = selected_page_indexes(doc.page_count)[:max_pages]
    for page_index in wanted:
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
        name = f"{digest}-page-{page_index + 1}.png"
        images[name] = pix.tobytes("png")
    return images


def parse_pdf(pdf_bytes: bytes) -> PdfContent:
    parser = "pypdf"
    warning = ""
    try:
        text = extract_text_with_pypdf(pdf_bytes)
    except Exception as exc:
        text = ""
        warning = f"PDF text extraction failed: {exc}"
    text = compact_text(text, 50000)
    sections = extract_sections(text)
    legends = extract_figure_legends(text)
    evidence = extract_numeric_evidence(text, legends)
    key_figures = choose_key_figures(legends, evidence)
    rendered = render_key_pages(pdf_bytes, key_figures)
    for figure in key_figures:
        try:
            page = int(figure.page)
        except (TypeError, ValueError):
            continue
        image_name = f"{hashlib.sha1(pdf_bytes).hexdigest()[:10]}-page-{page}.png"
        figure.page_image_name = image_name if image_name in rendered else ""
        figure.image_name = figure.page_image_name
        figure.needs_manual_crop = bool(figure.image_name)
    return PdfContent(
        text=text,
        sections=sections,
        legends=key_figures,
        evidence=evidence,
        rendered_images=rendered,
        parser=parser,
        warning=warning,
    )
