from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from typing import Any

from .models import EvidenceItem, FigureAnalysis


FIGURE_MARKER_RE = re.compile(
    r"(?im)^\s*((?:(?:extended\s+data|supplementary)\s+)?"
    r"(?:fig(?:ure)?\.?|scheme|table)\s*[A-Za-z]?\d+[A-Za-z]?)"
    r"\s*[\s:.,;\-\u2013\u2014]+"
)
NUMERIC_RE = re.compile(
    r"(?i)(?:\b\d+(?:\.\d+)?\s*(?:%|bp|nt|kb|Mb|nM|uM|\u00b5M|mM|M|mg/L|g/L|"
    r"min|h|s|\u00b0C|fold|x|\u00d7|copies|bases|residues|cycles)(?![A-Za-z0-9])|"
    r"\b\d+(?:\.\d+)?\s*(?:to|-|\u2013)\s*\d+(?:\.\d+)?\s*(?:bp|nt|%|h|min)(?![A-Za-z0-9]))"
)
SECTION_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(abstract|summary|introduction|background|results?|discussion|"
    r"conclusions?|methods?|materials\s+and\s+methods|experimental\s+procedures|"
    r"supplementary(?:\s+information|\s+materials?)?|references)\s*[:.]?\s*$"
)
SECTION_ALIASES = {
    "summary": "abstract",
    "background": "introduction",
    "result": "results",
    "conclusions": "conclusion",
    "method": "methods",
    "materials and methods": "methods",
    "experimental procedures": "methods",
    "supplementary information": "supplementary",
    "supplementary materials": "supplementary",
}
PROMPT_SECTION_ORDER = ("abstract", "introduction", "methods", "results", "discussion", "conclusion")


@dataclass
class PdfContent:
    text: str = ""
    sections: dict[str, str] = field(default_factory=dict)
    legends: list[FigureAnalysis] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    rendered_images: dict[str, bytes] = field(default_factory=dict)
    parser: str = ""
    warning: str = ""
    hash: str = ""
    page_count: int = 0
    parse_mode: str = ""
    quality: str = "unknown"
    coverage: list[str] = field(default_factory=list)
    all_figures: list[FigureAnalysis] = field(default_factory=list)
    lead_image: FigureAnalysis | None = None

    def prompt_text(self, max_chars: int = 24000) -> str:
        parts: list[str] = [
            f"Parser: {self.parser or self.parse_mode}",
            f"Quality: {self.quality}; Pages: {self.page_count}; Coverage: {', '.join(self.coverage)}",
        ]
        if self.warning:
            parts.append(f"Warning: {self.warning}")
        if self.legends:
            parts.append("\nFigure/Table captions:")
            for item in self.legends[:30]:
                page = f" page {item.page}" if item.page else ""
                parts.append(f"- {item.figure_id}{page}: {compact_text(item.caption, 900)}")
        if self.evidence:
            parts.append("\nNumeric evidence candidates:")
            for item in self.evidence[:40]:
                parts.append(f"- {item.value} | {item.claim} | {item.label()}")
        if self.sections:
            parts.append("\nEvidence-balanced sections:")
            section_budget = max(1200, (max_chars - 6000) // max(1, len(self.sections)))
            ordered = list(PROMPT_SECTION_ORDER) + [name for name in self.sections if name not in PROMPT_SECTION_ORDER]
            for name in ordered:
                value = self.sections.get(name)
                if value:
                    parts.append(f"\n## {name}\n{compact_text(value, section_budget)}")
        elif self.text:
            parts.extend(["\nText:", compact_text(self.text, max_chars - 2000)])
        return compact_text("\n".join(parts), max_chars)

    def to_dict(self, include_images: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "text": self.text,
            "sections": dict(self.sections),
            "legends": [item.to_dict() for item in self.legends],
            "evidence": [item.to_dict() for item in self.evidence],
            "parser": self.parser,
            "warning": self.warning,
            "hash": self.hash,
            "page_count": self.page_count,
            "parse_mode": self.parse_mode,
            "quality": self.quality,
            "coverage": list(self.coverage),
            "all_figures": [item.to_dict() for item in self.all_figures],
            "lead_image": self.lead_image.to_dict() if self.lead_image else None,
        }
        if include_images:
            data["rendered_images"] = dict(self.rendered_images)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PdfContent":
        return cls(
            text=str(data.get("text") or ""),
            sections={str(key): str(value) for key, value in (data.get("sections") or {}).items()},
            legends=[FigureAnalysis.from_dict(item) for item in data.get("legends") or [] if isinstance(item, dict)],
            evidence=[EvidenceItem.from_dict(item) for item in data.get("evidence") or [] if isinstance(item, dict)],
            rendered_images=dict(data.get("rendered_images") or {}),
            parser=str(data.get("parser") or ""),
            warning=str(data.get("warning") or ""),
            hash=str(data.get("hash") or ""),
            page_count=int(data.get("page_count") or 0),
            parse_mode=str(data.get("parse_mode") or ""),
            quality=str(data.get("quality") or "unknown"),
            coverage=[str(item) for item in data.get("coverage") or []],
            all_figures=[FigureAnalysis.from_dict(item) for item in data.get("all_figures") or [] if isinstance(item, dict)],
            lead_image=(FigureAnalysis.from_dict(data["lead_image"]) if isinstance(data.get("lead_image"), dict) else None),
        )


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
    prefix = ""
    prefix_match = re.match(r"(?i)^(extended data|supplementary)\s+", text)
    if prefix_match:
        prefix = prefix_match.group(1).title() + " "
        text = text[prefix_match.end() :]
    text = re.sub(r"(?i)^figure\b", "Fig.", text)
    text = re.sub(r"(?i)^fig\b\.?", "Fig.", text)
    text = re.sub(r"(?i)^table\b", "Table", text)
    text = re.sub(r"(?i)^scheme\b", "Scheme", text)
    return prefix + text


def figure_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_figure_id(value).lower())


def extract_page_number_near(text: str, position: int) -> str:
    before = text[max(0, position - 1200) : position]
    matches = list(re.finditer(r"\[Page\s+(\d+)\]", before, flags=re.I))
    return matches[-1].group(1) if matches else ""


def trim_caption(segment: str, max_chars: int = 1400) -> str:
    segment = re.sub(r"\[Page\s+\d+\]", " ", segment, flags=re.I)
    segment = re.sub(r"\s+", " ", segment).strip()
    segment = re.sub(
        r"(?i)\s+(references|acknowledg(?:e)?ments|author contributions|competing interests)\s+.*$",
        "",
        segment,
    ).strip()
    return compact_text(segment, max_chars)


def extract_figure_legends(text: str, max_items: int = 80) -> list[FigureAnalysis]:
    matches = list(FIGURE_MARKER_RE.finditer(text or ""))
    legends: list[FigureAnalysis] = []
    seen: set[str] = set()
    for index, match in enumerate(matches):
        fig_id = normalize_figure_id(match.group(1))
        key = figure_key(fig_id)
        if not key or key in seen:
            continue
        next_start = matches[index + 1].start(1) if index + 1 < len(matches) else len(text)
        segment = text[match.start(1) : min(next_start, match.start(1) + 2200)]
        caption = trim_caption(segment)
        if len(caption) < len(fig_id) + 12:
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


def extract_sections(text: str, max_chars_each: int = 16000) -> dict[str, str]:
    matches = list(SECTION_RE.finditer(text or ""))
    if not matches:
        return {}
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        raw_name = re.sub(r"\s+", " ", match.group(1).lower()).strip()
        name = SECTION_ALIASES.get(raw_name, raw_name)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = compact_text(text[start:end], max_chars_each)
        if body and name not in sections:
            sections[name] = body
    return sections


def extract_numeric_evidence(text: str, legends: list[FigureAnalysis], max_items: int = 80) -> list[EvidenceItem]:
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
            evidence.append(EvidenceItem(claim=claim, value=value, source=source, page=page, figure_id=figure_id, confidence="high" if figure_id else "medium"))
            if len(evidence) >= max_items:
                return

    for legend in legends:
        add_candidates(legend.caption, "caption", legend.page, legend.figure_id)
        if len(evidence) >= max_items:
            return evidence
    for page_match in re.finditer(r"(?s)\[Page\s+(\d+)\](.*?)(?=\[Page\s+\d+\]|\Z)", text or ""):
        add_candidates(page_match.group(2), "full text", page_match.group(1))
        if len(evidence) >= max_items:
            break
    if not evidence:
        add_candidates(text, "full text")
    return evidence[:max_items]


def _figure_role(figure: FigureAnalysis, order: int) -> str:
    caption = figure.caption.lower()
    if re.search(r"overview|schematic|mechanism|pathway|model", caption):
        return "mechanism"
    if re.search(r"workflow|method|design|fabrication|protocol|screen", caption):
        return "method"
    if re.search(r"validation|confirm|replicat|robust|additional|extended|supplementary", caption):
        return "validation"
    return "key_result" if order <= 2 else "validation"


def choose_key_figures(legends: list[FigureAnalysis], evidence: list[EvidenceItem], max_figures: int = 4) -> list[FigureAnalysis]:
    if not legends:
        return []
    scores: dict[str, int] = {}
    for legend in legends:
        key = figure_key(legend.figure_id)
        caption = legend.caption.lower()
        score = 0
        score += 8 if re.search(r"\bfig\.?\s*1\b|workflow|scheme|overview|design|mechanism", caption) else 0
        score += 5 if re.search(r"activity|efficiency|yield|conversion|accuracy|fidelity|length|synthesis|result", caption) else 0
        score += 4 if re.search(r"mutant|variant|engineering|directed evolution|screen", caption) else 0
        score += 3 if re.search(r"comparison|benchmark|control|wild-type|scale|validation", caption) else 0
        score += sum(1 for item in evidence if figure_key(item.figure_id) == key)
        scores[key] = score
    ranked = sorted(legends, key=lambda item: (scores.get(figure_key(item.figure_id), 0), item.figure_id), reverse=True)
    selected = ranked[: max(0, min(max_figures, 4))]
    for index, item in enumerate(selected, start=1):
        item.why_selected = "该图包含流程、方法、关键结果或验证信息，是支撑单篇解读的高信号图。"
        item.evidence = [ev for ev in evidence if figure_key(ev.figure_id) == figure_key(item.figure_id)][:6]
        item.needs_manual_check = not bool(item.evidence)
        item.selected = True
        item.order = index
        item.role = _figure_role(item, index)
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


def pypdf_page_count(pdf_bytes: bytes) -> int:
    from pypdf import PdfReader

    return len(PdfReader(io.BytesIO(pdf_bytes)).pages)


def extract_markdown_with_pymupdf4llm(pdf_bytes: bytes) -> tuple[str, list[dict[str, Any]], int]:
    import fitz
    import pymupdf4llm

    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    chunks = pymupdf4llm.to_markdown(document, page_chunks=True, show_progress=False)
    if not isinstance(chunks, list):
        raise ValueError("PyMuPDF4LLM did not return page chunks")
    text_parts: list[str] = []
    normalized_chunks: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        item = dict(chunk)
        page_number = int((item.get("metadata") or {}).get("page_number") or index)
        page_text = str(item.get("text") or "").strip()
        if page_text:
            text_parts.append(f"[Page {page_number}]\n{page_text}")
        normalized_chunks.append(item)
    return "\n\n".join(text_parts), normalized_chunks, document.page_count


def normalize_crop_bbox(bbox: tuple[float, float, float, float] | list[float]) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError("crop bbox must contain four normalized coordinates")
    left, top, right, bottom = (max(0.0, min(1.0, float(value))) for value in bbox)
    if right - left < 0.02 or bottom - top < 0.02:
        raise ValueError("crop bbox is empty or too small")
    return left, top, right, bottom


def adjust_crop_bbox(
    bbox: tuple[float, float, float, float] | list[float],
    *,
    left: float = 0.0,
    top: float = 0.0,
    right: float = 0.0,
    bottom: float = 0.0,
) -> tuple[float, float, float, float]:
    current = normalize_crop_bbox(bbox)
    return normalize_crop_bbox((current[0] + left, current[1] + top, current[2] + right, current[3] + bottom))


def render_pdf_crop(
    pdf_bytes: bytes,
    page_number: int,
    crop_bbox: tuple[float, float, float, float] | list[float],
    scale: float = 2.0,
) -> bytes:
    import fitz

    normalized = normalize_crop_bbox(crop_bbox)
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    if page_number < 1 or page_number > document.page_count:
        raise ValueError(f"page number {page_number} is outside 1-{document.page_count}")
    page = document.load_page(page_number - 1)
    rect = page.rect
    clip = fitz.Rect(rect.x0 + normalized[0] * rect.width, rect.y0 + normalized[1] * rect.height, rect.x0 + normalized[2] * rect.width, rect.y0 + normalized[3] * rect.height)
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
    return pixmap.tobytes("png")


def adjust_and_render_crop(
    pdf_bytes: bytes,
    page_number: int,
    crop_bbox: tuple[float, float, float, float] | list[float],
    *,
    left: float = 0.0,
    top: float = 0.0,
    right: float = 0.0,
    bottom: float = 0.0,
    scale: float = 2.0,
) -> tuple[tuple[float, float, float, float], bytes]:
    adjusted = adjust_crop_bbox(crop_bbox, left=left, top=top, right=right, bottom=bottom)
    return adjusted, render_pdf_crop(pdf_bytes, page_number, adjusted, scale=scale)


def _normalize_page_rect(rect: Any, page_rect: Any, padding: float = 0.02) -> tuple[float, float, float, float]:
    return normalize_crop_bbox(((rect.x0 - page_rect.x0) / page_rect.width - padding, (rect.y0 - page_rect.y0) / page_rect.height - padding, (rect.x1 - page_rect.x0) / page_rect.width + padding, (rect.y1 - page_rect.y0) / page_rect.height + padding))


def _find_caption_rect(page: Any, figure_id: str) -> Any | None:
    variants = [figure_id, figure_id.replace("Fig.", "Figure"), figure_id.replace("Fig.", "Fig")]
    for variant in variants:
        matches = page.search_for(variant)
        if matches:
            return matches[0]
    return None


def _candidate_crop(page: Any, figure: FigureAnalysis) -> tuple[tuple[float, float, float, float], float]:
    import fitz

    caption_rect = _find_caption_rect(page, figure.figure_id)
    if caption_rect is None:
        return (0.0, 0.0, 1.0, 1.0), 0.25
    media_rects: list[Any] = []
    for info in page.get_image_info():
        rect = fitz.Rect(info.get("bbox"))
        if rect.width >= page.rect.width * 0.18 and rect.height >= page.rect.height * 0.06:
            media_rects.append(rect)
    for drawing in page.get_drawings():
        rect = fitz.Rect(drawing.get("rect"))
        if rect.width >= page.rect.width * 0.28 and rect.height >= page.rect.height * 0.08:
            media_rects.append(rect)
    above = [rect for rect in media_rects if rect.y0 < caption_rect.y0 and caption_rect.y0 - rect.y1 < page.rect.height * 0.55]
    if not above:
        return (0.0, 0.0, 1.0, 1.0), 0.4
    nearest = max(above, key=lambda rect: rect.y1)
    return _normalize_page_rect(nearest | caption_rect, page.rect, padding=0.025), 0.88


def _lead_crop(document: Any, digest: str) -> FigureAnalysis | None:
    if document.page_count <= 0:
        return None
    page = document.load_page(0)
    blocks = sorted(page.get_text("blocks"), key=lambda block: (block[1], block[0]))
    text_blocks = [block for block in blocks if str(block[4]).strip()]
    bottom = page.rect.height * 0.34
    if text_blocks:
        abstract_blocks = [
            block
            for block in text_blocks
            if block[1] < page.rect.height * 0.68 and re.search(r"(?i)\babstract\b|摘要", str(block[4]))
        ]
        if abstract_blocks:
            first_abstract = min(abstract_blocks, key=lambda block: block[1])
            bottom = min(page.rect.height * 0.58, first_abstract[3] + page.rect.height * 0.08)
        else:
            top_blocks = [block for block in text_blocks if block[1] < page.rect.height * 0.48]
            if top_blocks:
                bottom = min(page.rect.height * 0.52, max(block[3] for block in top_blocks) + 22)
    return FigureAnalysis(figure_id="Lead", caption="论文首页标题与题录信息", page="1", image_name=f"{digest}-lead.png", crop_bbox=normalize_crop_bbox((0.035, 0.02, 0.965, max(0.24, bottom / page.rect.height))), confidence=0.92, role="lead", selected=True, order=0)


def build_figure_crops(pdf_bytes: bytes, figures: list[FigureAnalysis], digest: str) -> tuple[FigureAnalysis | None, dict[str, bytes]]:
    try:
        import fitz
    except ImportError:
        return None, {}
    images: dict[str, bytes] = {}
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    lead = _lead_crop(document, digest)
    if lead and lead.crop_bbox:
        images[lead.image_name] = render_pdf_crop(pdf_bytes, 1, lead.crop_bbox)
    for figure in figures:
        try:
            page_number = int(figure.page)
        except (TypeError, ValueError):
            figure.crop_bbox = (0.0, 0.0, 1.0, 1.0)
            figure.confidence = 0.0
            figure.needs_manual_crop = True
            continue
        if not 1 <= page_number <= document.page_count:
            figure.needs_manual_crop = True
            continue
        bbox, confidence = _candidate_crop(document.load_page(page_number - 1), figure)
        figure.crop_bbox = bbox
        figure.confidence = confidence
        figure.needs_manual_crop = confidence < 0.65
        if figure.needs_manual_crop:
            figure.selected = False
        safe_id = re.sub(r"[^a-z0-9]+", "-", figure.figure_id.lower()).strip("-") or "figure"
        figure.image_name = f"{digest}-{safe_id}.png"
        figure.page_image_name = figure.image_name
        try:
            images[figure.image_name] = render_pdf_crop(pdf_bytes, page_number, bbox)
        except Exception:
            figure.image_name = ""
            figure.page_image_name = ""
            figure.needs_manual_crop = True
    return lead, images


def render_key_pages(pdf_bytes: bytes, figures: list[FigureAnalysis], max_pages: int = 4) -> dict[str, bytes]:
    digest = hashlib.sha1(pdf_bytes).hexdigest()[:10]
    images: dict[str, bytes] = {}
    pages: list[int] = []
    for figure in figures:
        try:
            page = int(figure.page)
        except (TypeError, ValueError):
            continue
        if page not in pages:
            pages.append(page)
        if len(pages) >= max_pages:
            break
    if not pages:
        try:
            pages = [index + 1 for index in selected_page_indexes(pypdf_page_count(pdf_bytes))[:max_pages]]
        except Exception:
            return {}
    for page in pages:
        try:
            name = f"{digest}-page-{page}.png"
            images[name] = render_pdf_crop(pdf_bytes, page, (0.0, 0.0, 1.0, 1.0))
        except Exception:
            continue
    return images


def _coverage_and_quality(text: str, sections: dict[str, str], page_count: int) -> tuple[list[str], str]:
    coverage = [name for name in PROMPT_SECTION_ORDER if sections.get(name)]
    chars_per_page = len(re.sub(r"\s+", "", text)) / max(1, page_count)
    if len(coverage) >= 4 and chars_per_page >= 500:
        return coverage, "high"
    if len(coverage) >= 2 or chars_per_page >= 250:
        return coverage, "medium"
    return coverage, "low"


def parse_pdf(pdf_bytes: bytes, mode: str = "auto") -> PdfContent:
    requested_mode = (mode or "auto").strip().lower()
    if requested_mode not in {"auto", "enhanced", "pypdf", "fast"}:
        raise ValueError("mode must be one of: auto, enhanced, pypdf, fast")
    digest = hashlib.sha256(pdf_bytes).hexdigest()
    warning_parts: list[str] = []
    text = ""
    page_count = 0
    parser = ""
    actual_mode = ""
    if requested_mode in {"auto", "enhanced"}:
        try:
            text, _layout, page_count = extract_markdown_with_pymupdf4llm(pdf_bytes)
            if not text.strip():
                raise ValueError("enhanced parser returned no text")
            parser = "pymupdf4llm"
            actual_mode = "enhanced"
        except Exception as exc:
            warning_parts.append(f"Enhanced PDF parsing failed; fell back to pypdf: {exc}")
    if not text:
        try:
            text = extract_text_with_pypdf(pdf_bytes)
            page_count = pypdf_page_count(pdf_bytes)
            parser = "pypdf"
            actual_mode = "pypdf"
        except Exception as exc:
            warning_parts.append(f"PDF text extraction failed: {exc}")
    text = compact_text(text, 240000)
    sections = extract_sections(text)
    all_figures = extract_figure_legends(text)
    evidence = extract_numeric_evidence(text, all_figures)
    selected = choose_key_figures(all_figures, evidence, max_figures=4)
    lead_image: FigureAnalysis | None = None
    rendered: dict[str, bytes] = {}
    if pdf_bytes:
        try:
            lead_image, rendered = build_figure_crops(pdf_bytes, selected, digest[:12])
        except Exception as exc:
            warning_parts.append(f"Figure crop rendering failed: {exc}")
            rendered = render_key_pages(pdf_bytes, selected)
            for figure in selected:
                try:
                    page = int(figure.page)
                except (TypeError, ValueError):
                    continue
                page_name = f"{hashlib.sha1(pdf_bytes).hexdigest()[:10]}-page-{page}.png"
                if page_name in rendered:
                    figure.image_name = page_name
                    figure.page_image_name = page_name
                    figure.crop_bbox = (0.0, 0.0, 1.0, 1.0)
                    figure.confidence = 0.3
                    figure.needs_manual_crop = True
    coverage, quality = _coverage_and_quality(text, sections, page_count)
    if quality == "low":
        warning_parts.append("PDF text coverage is low; scanned pages may require OCR or visual review.")
    return PdfContent(text=text, sections=sections, legends=selected, evidence=evidence, rendered_images=rendered, parser=parser, warning=" ".join(warning_parts), hash=digest, page_count=page_count, parse_mode=actual_mode, quality=quality, coverage=coverage, all_figures=all_figures, lead_image=lead_image)
