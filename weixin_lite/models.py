from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PaperInput:
    title: str
    doi: str = ""
    pmid: str = ""
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    year: str = ""
    abstract: str = ""
    url: str = ""
    oa_pdf_url: str = ""
    source: str = "manual"
    pdf_name: str = ""

    @property
    def display_title(self) -> str:
        return self.title or self.doi or self.pmid or self.pdf_name or "Untitled paper"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceItem:
    claim: str
    value: str = ""
    source: str = ""
    page: str = ""
    figure_id: str = ""
    confidence: str = "medium"

    def label(self) -> str:
        bits = [self.figure_id, f"p.{self.page}" if self.page else "", self.source]
        return " / ".join(bit for bit in bits if bit)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FigureAnalysis:
    figure_id: str
    caption: str
    page: str = ""
    image_name: str = ""
    why_selected: str = ""
    interpretation: str = ""
    evidence: list[EvidenceItem] = field(default_factory=list)
    needs_manual_check: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence"] = [item.to_dict() for item in self.evidence]
        return data


@dataclass
class QuickReadArticle:
    paper: PaperInput
    title: str
    digest: str
    body_markdown: str
    body_html: str
    figures: list[FigureAnalysis] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    cover_image_name: str = ""
    word_count: int = 0
    status: str = "draft"
    warnings: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["paper"] = self.paper.to_dict()
        data["figures"] = [item.to_dict() for item in self.figures]
        data["evidence"] = [item.to_dict() for item in self.evidence]
        return data


@dataclass
class BatchProject:
    topic: str
    papers: list[PaperInput] = field(default_factory=list)
    articles: list[QuickReadArticle] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    version: str = "0.1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "topic": self.topic,
            "created_at": self.created_at,
            "papers": [paper.to_dict() for paper in self.papers],
            "articles": [article.to_dict() for article in self.articles],
        }
