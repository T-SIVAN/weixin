from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from collections.abc import Mapping
from typing import Any, Literal


AccessStatus = Literal["open", "paywalled", "unknown", "download_failed"]
KeywordSource = Literal["dictionary", "model", "original", "fallback"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PaperInput:
    title: str = ""
    doi: str = ""
    pmid: str = ""
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    year: str = ""
    publication_date: str = ""
    publication_date_source: str = ""
    abstract: str = ""
    url: str = ""
    oa_pdf_url: str = ""
    source: str = "manual"
    pdf_name: str = ""
    title_en: str = ""
    title_zh: str = ""
    abstract_en: str = ""
    abstract_zh: str = ""
    keywords: list[str] = field(default_factory=list)
    discovered_at: str = ""
    is_open_access: bool = False
    access_status: AccessStatus = "unknown"
    oa_source: str = ""
    download_error: str = ""
    article_type: str = ""
    journal_priority: int = 9999

    def __post_init__(self) -> None:
        if not self.title and self.title_en:
            self.title = self.title_en
        if not self.title_en and self.title:
            self.title_en = self.title
        if not self.abstract_en and self.abstract:
            self.abstract_en = self.abstract
        if not self.abstract and self.abstract_en:
            self.abstract = self.abstract_en
        if self.oa_pdf_url and self.access_status == "unknown":
            self.is_open_access = True
            self.access_status = "open"
        if not self.discovered_at:
            self.discovered_at = utc_now()

    @property
    def display_title(self) -> str:
        return self.title_zh or self.title_en or self.title or self.doi or self.pmid or self.pdf_name or "Untitled paper"

    @property
    def paper_key(self) -> str:
        return self.doi or self.pmid or self.title_en or self.title or self.pdf_name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PaperInput":
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in data.items() if key in fields})


@dataclass
class ResolvedKeyword:
    original: str
    english_terms: list[str] = field(default_factory=list)
    source: KeywordSource = "original"
    warning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResolvedKeyword":
        source = str(data.get("source") or "original")
        if source not in {"dictionary", "model", "original", "fallback"}:
            source = "original"
        return cls(
            original=str(data.get("original") or ""),
            english_terms=[str(item) for item in data.get("english_terms") or [] if str(item).strip()],
            source=source,  # type: ignore[arg-type]
            warning=str(data.get("warning") or ""),
        )


@dataclass
class SearchQueryPlan:
    keywords: list[ResolvedKeyword] = field(default_factory=list)
    search_mode: str = "strict"
    warnings: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)

    @property
    def original_keywords(self) -> list[str]:
        return [item.original for item in self.keywords if item.original]

    @property
    def search_terms(self) -> list[str]:
        terms: list[str] = []
        for item in self.keywords:
            terms.extend(item.english_terms or [item.original])
        return list(dict.fromkeys(term.strip() for term in terms if term.strip()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "keywords": [item.to_dict() for item in self.keywords],
            "search_mode": self.search_mode,
            "warnings": self.warnings,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchQueryPlan":
        return cls(
            keywords=[ResolvedKeyword.from_dict(item) for item in data.get("keywords") or [] if isinstance(item, dict)],
            search_mode=str(data.get("search_mode") or "strict"),
            warnings=[str(item) for item in data.get("warnings") or []],
            created_at=str(data.get("created_at") or utc_now()),
        )


@dataclass
class SearchRun:
    run_id: str
    keywords: list[str]
    started_at: str
    finished_at: str = ""
    records: list[PaperInput] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    query_plan: SearchQueryPlan | None = None
    source_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    raw_count: int = 0
    filtered_count: int = 0
    warnings: list[str] = field(default_factory=list)
    search_kind: str = "keyword"
    journal_filters: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "keywords": self.keywords,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "records": [record.to_dict() for record in self.records],
            "errors": self.errors,
            "query_plan": self.query_plan.to_dict() if self.query_plan else None,
            "source_counts": self.source_counts,
            "raw_count": self.raw_count,
            "filtered_count": self.filtered_count,
            "warnings": self.warnings,
            "search_kind": self.search_kind,
            "journal_filters": self.journal_filters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchRun":
        return cls(
            run_id=str(data.get("run_id") or ""),
            keywords=[str(item) for item in data.get("keywords") or []],
            started_at=str(data.get("started_at") or ""),
            finished_at=str(data.get("finished_at") or ""),
            records=[PaperInput.from_dict(item) for item in data.get("records") or []],
            errors={str(k): str(v) for k, v in (data.get("errors") or {}).items()},
            query_plan=(
                SearchQueryPlan.from_dict(data["query_plan"])
                if isinstance(data.get("query_plan"), dict)
                else None
            ),
            source_counts={
                str(source): {str(key): int(value) for key, value in counts.items()}
                for source, counts in (data.get("source_counts") or {}).items()
                if isinstance(counts, dict)
            },
            raw_count=int(data.get("raw_count") or 0),
            filtered_count=int(data.get("filtered_count") or len(data.get("records") or [])),
            warnings=[str(item) for item in data.get("warnings") or []],
            search_kind=str(data.get("search_kind") or "keyword"),
            journal_filters=[
                dict(item) for item in data.get("journal_filters") or [] if isinstance(item, dict)
            ],
        )


@dataclass
class DownloadedPaper:
    paper_key: str
    file_name: str = ""
    content_type: str = ""
    source_url: str = ""
    status: AccessStatus = "unknown"
    error: str = ""
    content_bytes: bytes = b""

    def to_dict(self, include_bytes: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if include_bytes:
            return data
        data.pop("content_bytes", None)
        return data


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
    page_image_name: str = ""
    why_selected: str = ""
    interpretation: str = ""
    evidence: list[EvidenceItem] = field(default_factory=list)
    needs_manual_check: bool = False
    needs_manual_crop: bool = False

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
    downloads: list[DownloadedPaper] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    version: str = "0.2.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "topic": self.topic,
            "created_at": self.created_at,
            "papers": [paper.to_dict() for paper in self.papers],
            "articles": [article.to_dict() for article in self.articles],
            "downloads": [download.to_dict() for download in self.downloads],
        }


def is_generation_ready(paper: PaperInput, pdfs: Mapping[str, Any] | None = None) -> bool:
    """Return True only when a paper has a parsed full text available."""
    pdf_cache = pdfs or {}
    return bool(paper.access_status == "open" and paper.pdf_name and paper.pdf_name in pdf_cache)


def generation_ready_papers(papers: list[PaperInput], pdfs: Mapping[str, Any] | None = None) -> list[PaperInput]:
    return [paper for paper in papers if is_generation_ready(paper, pdfs)]


def unavailable_papers(papers: list[PaperInput], pdfs: Mapping[str, Any] | None = None) -> list[PaperInput]:
    return [paper for paper in papers if not is_generation_ready(paper, pdfs)]
