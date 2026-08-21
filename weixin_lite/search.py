from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Literal

from .models import PaperInput, ResolvedKeyword, SearchQueryPlan, SearchRun, utc_now


DEFAULT_KEYWORDS = [
    "synthetic biology",
    "metabolic engineering",
    "biomanufacturing",
    "biosynthesis",
    "engineered microbes",
]
KEYWORD_EXPANSIONS = {
    "tdt": ["terminal deoxynucleotidyl transferase", "TdT"],
    "pup": ["poly(U) polymerase", "PUP", "polynucleotide phosphorylase"],
    "核酸末端转移酶": [
        "terminal deoxynucleotidyl transferase",
        "terminal nucleotidyltransferase",
        "template-independent DNA synthesis",
    ],
    "末端转移酶": [
        "terminal transferase",
        "terminal deoxynucleotidyl transferase",
        "terminal nucleotidyltransferase",
    ],
    "末端脱氧核苷酸转移酶": [
        "terminal deoxynucleotidyl transferase",
        "TdT",
        "template-independent polymerase",
    ],
    "terminal transferase": [
        "terminal transferase",
        "terminal deoxynucleotidyl transferase",
        "terminal nucleotidyltransferase",
    ],
    "酶促dna合成": ["enzymatic DNA synthesis", "template-independent DNA synthesis"],
    "酶促rna合成": ["enzymatic RNA synthesis", "RNA enzymatic synthesis"],
    "enzyme engineering": ["enzyme engineering", "directed evolution", "protein engineering", "enzyme design"],
    "synthetic biology": ["synthetic biology", "synthetic biology-enabled", "SynBio"],
    "合成生物": ["synthetic biology", "SynBio"],
    "metabolic engineering": ["metabolic engineering", "pathway engineering", "metabolic pathway engineering"],
    "代谢工程": ["metabolic engineering", "pathway engineering"],
    "biomanufacturing": ["biomanufacturing", "bio-manufacturing", "biofabrication", "bioproduction"],
    "生物制造": ["biomanufacturing", "bio-manufacturing", "bioproduction"],
    "biosynthesis": ["biosynthesis", "biosynthetic pathway", "de novo biosynthesis"],
    "生物合成": ["biosynthesis", "biosynthetic pathway"],
    "engineered microbes": ["engineered microbes", "engineered microorganism", "engineered strain", "microbial cell factory"],
    "工程菌": ["engineered strain", "engineered microorganism", "microbial cell factory"],
    "底盘细胞": ["chassis cell", "microbial chassis", "synthetic biology chassis"],
    "细胞工厂": ["cell factory", "microbial cell factory"],
    "微生物细胞工厂": ["microbial cell factory", "microbial production platform"],
    "精准发酵": ["precision fermentation", "precision-fermented"],
    "发酵工程": ["fermentation engineering", "bioprocess engineering"],
    "酶工程": ["enzyme engineering", "enzyme design", "directed evolution"],
    "蛋白质工程": ["protein engineering", "protein design", "directed evolution"],
    "天然产物": ["natural product", "natural products", "natural product biosynthesis"],
    "天然产物合成": ["natural product biosynthesis", "biosynthetic gene cluster"],
    "聚羟基脂肪酸酯": ["polyhydroxyalkanoate", "polyhydroxyalkanoates", "PHA biopolymer"],
    "聚羟基烷酸酯": ["polyhydroxyalkanoate", "polyhydroxyalkanoates", "PHA biopolymer"],
    "合成生物学": ["synthetic biology", "SynBio"],
    "基因线路": ["genetic circuit", "synthetic gene circuit"],
    "生物传感器": ["biosensor", "whole-cell biosensor"],
    "通路工程": ["pathway engineering", "metabolic pathway engineering"],
    "菌株工程": ["strain engineering", "engineered strain"],
    "基因组工程": ["genome engineering", "genome-scale engineering"],
    "合成基因组": ["synthetic genome", "genome synthesis"],
    "无细胞系统": ["cell-free system", "cell-free synthetic biology"],
    "生物基材料": ["bio-based material", "biobased material", "biomaterial production"],
    "生物铸造厂": ["biofoundry", "biofoundries", "synthetic biology foundry"],
    "合成代谢通路": ["synthetic metabolic pathway", "engineered metabolic pathway", "pathway engineering"],
    "代谢通量": ["metabolic flux", "flux balance analysis", "metabolic flux analysis"],
    "动态调控": ["dynamic regulation", "dynamic pathway regulation", "dynamic metabolic control"],
    "基因编辑": ["genome editing", "gene editing", "CRISPR engineering"],
    "模块化克隆": ["modular cloning", "MoClo", "Golden Gate assembly"],
    "组合生物合成": ["combinatorial biosynthesis", "combinatorial pathway engineering"],
    "生物合成基因簇": ["biosynthetic gene cluster", "biosynthetic gene clusters", "BGC"],
    "天然产物挖掘": ["natural product discovery", "genome mining", "biosynthetic gene cluster mining"],
    "高通量筛选": ["high-throughput screening", "high throughput screening", "screening platform"],
    "定向进化": ["directed evolution", "laboratory evolution", "adaptive laboratory evolution"],
    "适应性实验室进化": ["adaptive laboratory evolution", "ALE", "laboratory evolution"],
    "工业微生物": ["industrial microorganism", "industrial microbe", "industrial biotechnology"],
    "生物炼制": ["biorefinery", "biorefining", "integrated biorefinery"],
    "生物燃料": ["biofuel", "biofuels", "microbial biofuel production"],
    "生物塑料": ["bioplastic", "bioplastics", "microbial polymer production"],
    "单细胞蛋白": ["single-cell protein", "single cell protein", "microbial protein"],
    "人造肉": ["cultivated meat", "cell-based meat", "cultured meat"],
    "无细胞合成生物学": ["cell-free synthetic biology", "cell-free biosynthesis", "cell-free system"],
    "碳固定": ["carbon fixation", "synthetic carbon fixation", "microbial carbon fixation"],
    "二氧化碳生物转化": ["carbon dioxide bioconversion", "CO2 bioconversion", "microbial CO2 conversion"],
    "甲醇生物转化": ["methanol bioconversion", "methylotrophic biomanufacturing", "methanol-based bioproduction"],
    "一碳生物制造": ["one-carbon biomanufacturing", "C1 biomanufacturing", "one-carbon biotechnology"],
}

SearchMode = Literal["strict", "balanced", "broad"]
KeywordResolver = Callable[[str], list[str]]
SEARCH_MODE_ALIASES = {
    "strict": "strict",
    "precise": "strict",
    "精准": "strict",
    "严格": "strict",
    "balanced": "balanced",
    "均衡": "balanced",
    "broad": "broad",
    "宽松": "broad",
}


@dataclass
class SearchDiagnostics:
    source_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    raw_count: int = 0
    deduplicated_count: int = 0
    filtered_count: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchDiagnostics":
        return cls(
            source_counts={
                str(source): {str(key): int(value) for key, value in counts.items()}
                for source, counts in (data.get("source_counts") or {}).items()
                if isinstance(counts, dict)
            },
            raw_count=int(data.get("raw_count") or 0),
            deduplicated_count=int(data.get("deduplicated_count") or 0),
            filtered_count=int(data.get("filtered_count") or 0),
            warnings=[clean_text(item) for item in data.get("warnings") or [] if clean_text(item)],
            errors={str(key): str(value) for key, value in (data.get("errors") or {}).items()},
        )


@dataclass
class JournalFilter:
    name: str
    aliases: list[str] = field(default_factory=list)
    issn: str = ""
    eissn: str = ""
    publisher_family: str = ""
    priority: int = 9999
    enabled: bool = True

    @property
    def names(self) -> list[str]:
        return list(dict.fromkeys([self.name, *self.aliases]))

    @property
    def issns(self) -> list[str]:
        return [item for item in [self.issn, self.eissn] if clean_text(item)]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JournalFilter":
        return cls(
            name=clean_text(data.get("name")),
            aliases=[clean_text(item) for item in data.get("aliases") or [] if clean_text(item)],
            issn=clean_text(data.get("issn")),
            eissn=clean_text(data.get("eissn")),
            publisher_family=clean_text(data.get("publisher_family")),
            priority=int(data.get("priority") or 9999),
            enabled=bool(data.get("enabled", True)),
        )


SYNBIO_CONTEXT_TERMS = [
    "synthetic biology",
    "metabolic engineering",
    "pathway engineering",
    "biosynthesis",
    "biosynthetic pathway",
    "biomanufacturing",
    "bio-manufacturing",
    "bioproduction",
    "cell factory",
    "microbial cell factory",
    "engineered strain",
    "engineered microorganism",
    "engineered microbe",
    "fermentation",
    "strain engineering",
    "genome engineering",
    "protein engineering",
    "enzyme engineering",
    "directed evolution",
    "enzymatic DNA synthesis",
    "template-independent DNA synthesis",
    "de novo DNA synthesis",
    "oligonucleotide synthesis",
    "nucleic acid synthesis",
    "DNA synthesis",
    "terminal deoxynucleotidyl transferase",
    "terminal nucleotidyltransferase",
    "template-independent polymerase",
]

SYNBIO_PRODUCTION_TERMS = [
    "production",
    "produce",
    "producer",
    "yield",
    "titer",
    "titre",
    "g/l",
    "mg/l",
    "scale-up",
    "fed-batch",
    "bioreactor",
    "industrial",
    "commercial",
    "manufacturing",
]

LOW_RELEVANCE_TERMS = [
    "patient",
    "clinical trial",
    "diagnosis",
    "prognosis",
    "tumor",
    "cancer",
    "metastasis",
    "therapy",
    "vaccine",
    "infection",
]


class SearchError(RuntimeError):
    pass


DEFAULT_JOURNALS_PATH = Path("config/journals.json")
EXCLUDED_ARTICLE_TYPE_TERMS = {
    "book review",
    "brief comment",
    "calendar",
    "comment",
    "correction",
    "discussion",
    "editorial",
    "erratum",
    "expression of concern",
    "interview",
    "letter",
    "news",
    "newspaper article",
    "obituary",
    "reply",
    "retraction",
}


def clean_doi(value: Any) -> str:
    doi = str(value or "").strip()
    doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", doi, flags=re.I)
    return doi.rstrip(" .").lower()


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(value or ""))).strip()


def normalize_journal_text(value: Any) -> str:
    text = clean_text(value).lower().replace("&", "and")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def load_journal_filters(path: str | Path = DEFAULT_JOURNALS_PATH) -> list[JournalFilter]:
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    raw_items = data.get("journals") if isinstance(data, dict) else data
    if not isinstance(raw_items, list):
        raise SearchError(f"Journal config must contain a list: {config_path}")
    journals = [
        JournalFilter.from_dict(item)
        for item in raw_items
        if isinstance(item, dict) and clean_text(item.get("name"))
    ]
    return sorted([journal for journal in journals if journal.enabled], key=lambda item: item.priority)


def journal_matches_name(record_journal: str, journal: JournalFilter) -> bool:
    normalized_record = normalize_journal_text(record_journal)
    if not normalized_record:
        return False
    for name in journal.names:
        normalized_name = normalize_journal_text(name)
        if normalized_name and (normalized_record == normalized_name or normalized_name in normalized_record):
            return True
    return False


def decorate_journal_record(record: PaperInput, journal: JournalFilter) -> PaperInput:
    if not record.journal:
        record.journal = journal.name
    record.journal_priority = journal.priority
    return record


def should_keep_article_type(article_type: str) -> bool:
    normalized = normalize_journal_text(article_type)
    if not normalized:
        return True
    return not any(term.replace("-", " ") in normalized for term in EXCLUDED_ARTICLE_TYPE_TERMS)


def filter_journal_records(records: list[PaperInput], journal: JournalFilter) -> list[PaperInput]:
    filtered: list[PaperInput] = []
    for record in records:
        if record.journal and not journal_matches_name(record.journal, journal):
            continue
        if not should_keep_article_type(record.article_type):
            continue
        filtered.append(decorate_journal_record(record, journal))
    return filtered


def parse_keywords(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = re.split(r"[,，;\n]+", str(value or ""))
    keywords = [clean_text(item) for item in raw_items if clean_text(item)]
    return list(dict.fromkeys(keywords))


def normalize_search_mode(value: Any) -> SearchMode:
    return SEARCH_MODE_ALIASES.get(clean_text(value).lower(), "strict")  # type: ignore[return-value]


def contains_chinese(value: str) -> bool:
    return re.search(r"[\u4e00-\u9fff]", value) is not None


def _parse_llm_terms(content: Any) -> list[str]:
    text = str(content or "").strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.I)
    if fenced:
        text = fenced.group(1).strip()
    candidates: Any = None
    try:
        candidates = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*?\]", text)
        if match:
            try:
                candidates = json.loads(match.group(0))
            except json.JSONDecodeError:
                candidates = None
    if isinstance(candidates, dict):
        candidates = candidates.get("terms") or candidates.get("english_terms")
    if not isinstance(candidates, list):
        candidates = re.split(r"[,;\n]+", text)
    terms = [clean_text(item).strip('"\'') for item in candidates if clean_text(item)]
    return list(dict.fromkeys(term for term in terms if not contains_chinese(term)))[:3]


def openai_compatible_keyword_resolver(
    keyword: str,
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int = 20,
) -> list[str]:
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 80,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Translate a Chinese synthetic-biology literature keyword into 1-3 concise English "
                    "academic search terms. Return only a JSON array of strings."
                ),
            },
            {"role": "user", "content": keyword},
        ],
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with open_url_with_retries(request, timeout=timeout, attempts=2) as response:
        data = json.loads(response.read().decode(response.headers.get_content_charset() or "utf-8"))
    choices = data.get("choices") or []
    if not choices:
        raise SearchError("LLM response contains no choices")
    terms = _parse_llm_terms(((choices[0].get("message") or {}).get("content")))
    if not terms:
        raise SearchError("LLM response contains no usable English terms")
    return terms


def resolve_keyword_plan(
    keywords: str | list[str] | None,
    *,
    search_mode: str = "strict",
    resolver: KeywordResolver | None = None,
    llm_api_key: str = "",
    llm_base_url: str = "",
    llm_model: str = "",
) -> SearchQueryPlan:
    parsed = parse_keywords(keywords or DEFAULT_KEYWORDS) or list(DEFAULT_KEYWORDS)
    resolved: list[ResolvedKeyword] = []
    warnings: list[str] = []
    if resolver is None and llm_api_key and llm_base_url and llm_model:
        resolver = lambda keyword: openai_compatible_keyword_resolver(
            keyword,
            api_key=llm_api_key,
            base_url=llm_base_url,
            model=llm_model,
        )
    for keyword in parsed:
        dictionary_terms = KEYWORD_EXPANSIONS.get(keyword.lower())
        if dictionary_terms:
            resolved.append(ResolvedKeyword(keyword, list(dictionary_terms), "dictionary"))
            continue
        if not contains_chinese(keyword):
            resolved.append(ResolvedKeyword(keyword, [keyword], "original"))
            continue
        if resolver is None:
            warning = f"中文关键词“{keyword}”不在内置词典中，未配置模型，暂以原词检索。"
            warnings.append(warning)
            resolved.append(ResolvedKeyword(keyword, [keyword], "fallback", warning))
            continue
        try:
            terms = list(dict.fromkeys(clean_text(term) for term in resolver(keyword) if clean_text(term)))[:3]
            if not terms:
                raise SearchError("模型未返回英文同义词")
            resolved.append(ResolvedKeyword(keyword, terms, "model"))
        except Exception as exc:
            warning = f"中文关键词“{keyword}”扩展失败：{type(exc).__name__}: {exc}；暂以原词检索。"
            warnings.append(warning)
            resolved.append(ResolvedKeyword(keyword, [keyword], "fallback", warning))
    return SearchQueryPlan(resolved, normalize_search_mode(search_mode), warnings)


def ensure_query_plan(
    keywords: str | list[str] | None = None,
    query_plan: SearchQueryPlan | dict[str, Any] | None = None,
    search_mode: str | None = None,
) -> SearchQueryPlan:
    if isinstance(query_plan, dict):
        query_plan = SearchQueryPlan.from_dict(query_plan)
    if query_plan is not None:
        query_plan.search_mode = normalize_search_mode(search_mode or query_plan.search_mode)
        return query_plan
    return resolve_keyword_plan(keywords, search_mode=search_mode or "strict")


def _quoted(term: str) -> str:
    return f'"{term.replace(chr(34), "").strip()}"'


def _plan_groups(plan: SearchQueryPlan) -> list[list[str]]:
    return [item.english_terms or [item.original] for item in plan.keywords if item.original]


def build_pubmed_query(keywords: str | list[str] | SearchQueryPlan) -> str:
    plan = keywords if isinstance(keywords, SearchQueryPlan) else resolve_keyword_plan(keywords)
    groups = [
        "(" + " OR ".join(f"{_quoted(term)}[Title/Abstract]" for term in terms) + ")"
        for terms in _plan_groups(plan)
    ]
    context = " OR ".join(f"{_quoted(term)}[Title/Abstract]" for term in SYNBIO_CONTEXT_TERMS)
    return "(" + " OR ".join(groups) + f") AND ({context})"


def build_europe_pmc_query(keywords: str | list[str] | SearchQueryPlan, since_days: int | None = None) -> str:
    plan = keywords if isinstance(keywords, SearchQueryPlan) else resolve_keyword_plan(keywords)
    groups = []
    for terms in _plan_groups(plan):
        clauses = [f"TITLE:{_quoted(term)} OR ABSTRACT:{_quoted(term)}" for term in terms]
        groups.append("(" + " OR ".join(clauses) + ")")
    context = " OR ".join(
        f"TITLE:{_quoted(term)} OR ABSTRACT:{_quoted(term)}" for term in SYNBIO_CONTEXT_TERMS
    )
    query = "(" + " OR ".join(groups) + f") AND ({context})"
    if since_days:
        start, end = since_dates(since_days)
        query += f" AND FIRST_PDATE:[{start} TO {end}]"
    return query


def build_openalex_query(keywords: str | list[str] | SearchQueryPlan) -> str:
    plan = keywords if isinstance(keywords, SearchQueryPlan) else resolve_keyword_plan(keywords)
    groups = ["(" + " OR ".join(_quoted(term) for term in terms) + ")" for terms in _plan_groups(plan)]
    context = " OR ".join(_quoted(term) for term in SYNBIO_CONTEXT_TERMS)
    return "(" + " OR ".join(groups) + f") AND ({context})"


def build_crossref_queries(
    keywords: str | list[str] | SearchQueryPlan,
    max_queries: int = 12,
) -> list[str]:
    plan = keywords if isinstance(keywords, SearchQueryPlan) else resolve_keyword_plan(keywords)
    terms = [clean_text(term) for group in _plan_groups(plan) for term in group if clean_text(term)]
    return list(dict.fromkeys(terms))[:max_queries]


def build_pubmed_journal_query(journal: JournalFilter) -> str:
    clauses = [f"{_quoted(name)}[Journal]" for name in journal.names if clean_text(name)]
    clauses.extend(f"{_quoted(issn)}[ISSN]" for issn in journal.issns)
    return " OR ".join(clauses)


def build_europe_pmc_journal_query(journal: JournalFilter, since_days: int | None = None) -> str:
    clauses = [f"JOURNAL:{_quoted(name)}" for name in journal.names if clean_text(name)]
    clauses.extend(f"ISSN:{_quoted(issn)}" for issn in journal.issns)
    query = "(" + " OR ".join(clauses) + ")"
    if since_days:
        start, end = since_dates(since_days)
        query += f" AND FIRST_PDATE:[{start} TO {end}]"
    return query


def crossref_journal_urls(journal: JournalFilter) -> list[str]:
    if journal.issns:
        return [f"https://api.crossref.org/journals/{urllib.parse.quote(issn)}/works" for issn in journal.issns]
    return ["https://api.crossref.org/works"]


def build_keyword_query(keywords: str | list[str]) -> str:
    """Backward-compatible alias for the PubMed title/abstract query."""
    return build_pubmed_query(keywords)


def build_plain_search_queries(keywords: str | list[str], max_queries: int = 8) -> list[str]:
    return build_crossref_queries(keywords, max_queries=max_queries)


def term_in_text(term: str, text: str) -> bool:
    term_l = term.lower()
    if re.search(r"[\u4e00-\u9fff]", term_l):
        return term_l in text
    return re.search(rf"(?<![a-z0-9]){re.escape(term_l)}(?![a-z0-9])", text) is not None


def record_search_text(record: PaperInput) -> str:
    return " ".join(
        [
            record.title_en,
            record.title_zh,
            record.title,
            record.abstract_en,
            record.abstract_zh,
            record.abstract,
            " ".join(record.keywords),
            record.journal,
            record.source,
            record.doi,
            record.pmid,
        ]
    ).lower()


def keyword_terms(keywords: str | list[str] | SearchQueryPlan) -> list[str]:
    if isinstance(keywords, SearchQueryPlan):
        terms = [
            term
            for item in keywords.keywords
            for term in [item.original, *(item.english_terms or [])]
        ]
        return list(dict.fromkeys(clean_text(term) for term in terms if clean_text(term)))
    terms: list[str] = []
    for keyword in parse_keywords(keywords):
        terms.extend(KEYWORD_EXPANSIONS.get(keyword.lower(), [keyword]))
    return list(dict.fromkeys(clean_text(term) for term in terms if clean_text(term)))


FILTER_KEYWORD_STOPWORDS = {
    "about",
    "after",
    "against",
    "analysis",
    "article",
    "based",
    "between",
    "cells",
    "clinical",
    "data",
    "disease",
    "effect",
    "from",
    "gene",
    "genes",
    "human",
    "into",
    "model",
    "models",
    "nature",
    "novel",
    "paper",
    "protein",
    "study",
    "system",
    "systems",
    "that",
    "their",
    "these",
    "this",
    "through",
    "using",
    "with",
}


def filter_records_by_keywords(records: list[PaperInput], keywords: str | list[str]) -> list[PaperInput]:
    terms = keyword_terms(keywords)
    if not terms:
        return list(records)
    filtered: list[PaperInput] = []
    for record in records:
        text = record_search_text(record)
        if any(term_in_text(term, text) for term in terms):
            filtered.append(record)
    return filtered


def suggest_filter_keywords(records: list[PaperInput], limit: int = 12) -> list[str]:
    counts: Counter[str] = Counter()
    for record in records:
        text = record_search_text(record)
        for original, expanded in KEYWORD_EXPANSIONS.items():
            if contains_chinese(original):
                label = original
            else:
                label = clean_text(original)
            if not label:
                continue
            if any(term_in_text(term, text) for term in [original, *expanded]):
                counts[label] += 4
        for keyword in record.keywords:
            keyword = clean_text(keyword)
            if 3 <= len(keyword) <= 40:
                counts[keyword] += 3
        title_text = " ".join([record.title_en, record.title]).lower()
        for token in re.findall(r"\b[a-z][a-z0-9-]{3,}\b", title_text):
            if token in FILTER_KEYWORD_STOPWORDS:
                continue
            counts[token] += 1
    suggestions = [
        keyword
        for keyword, _ in counts.most_common(limit * 3)
        if keyword.lower() not in FILTER_KEYWORD_STOPWORDS
    ]
    defaults = [keyword for keyword in DEFAULT_KEYWORDS if keyword not in suggestions]
    return list(dict.fromkeys([*suggestions, *defaults]))[:limit]


def relevance_score(
    record: PaperInput,
    keywords: str | list[str] | SearchQueryPlan,
    search_mode: str | None = None,
) -> int:
    title = " ".join([record.title_en, record.title]).lower()
    abstract = " ".join([record.abstract_en, record.abstract]).lower()
    text = f"{title} {abstract}"
    terms = keyword_terms(keywords)
    title_keyword_hits = sum(term_in_text(term, title) for term in terms)
    abstract_keyword_hits = sum(term_in_text(term, abstract) for term in terms)
    title_context_hits = sum(term_in_text(term, title) for term in SYNBIO_CONTEXT_TERMS)
    abstract_context_hits = sum(term_in_text(term, abstract) for term in SYNBIO_CONTEXT_TERMS)
    production_hits = sum(term_in_text(term, text) for term in SYNBIO_PRODUCTION_TERMS)
    clinical_hits = sum(term_in_text(term, text) for term in LOW_RELEVANCE_TERMS)
    score = 0
    score += min(title_keyword_hits, 3) * 8
    score += min(abstract_keyword_hits, 3) * 4
    score += min(title_context_hits, 3) * 5
    score += min(abstract_context_hits, 3) * 3
    score += min(production_hits, 3) * 2
    score -= min(clinical_hits, 3) * 12
    if record.abstract_en or record.abstract:
        score += 1
    if record.doi:
        score += 1
    return score


def is_synthetic_biology_relevant(
    record: PaperInput,
    keywords: str | list[str] | SearchQueryPlan,
    search_mode: str | None = None,
) -> bool:
    mode = normalize_search_mode(
        search_mode or (keywords.search_mode if isinstance(keywords, SearchQueryPlan) else "strict")
    )
    title = " ".join([record.title_en, record.title]).lower()
    abstract = " ".join([record.abstract_en, record.abstract]).lower()
    text = f"{title} {abstract}"
    terms = keyword_terms(keywords)
    title_keyword = any(term_in_text(term, title) for term in terms)
    abstract_keyword = any(term_in_text(term, abstract) for term in terms)
    has_keyword = title_keyword or abstract_keyword
    has_context = any(term_in_text(term, text) for term in SYNBIO_CONTEXT_TERMS)
    score = relevance_score(record, keywords, mode)

    if mode == "broad":
        return has_keyword
    if mode == "balanced":
        return ((has_keyword and has_context) or title_keyword) and score >= 0
    strict_match = title_keyword or (abstract_keyword and has_context)
    return strict_match and score >= 6


def open_url_with_retries(req: urllib.request.Request, timeout: int, attempts: int = 3) -> Any:
    for attempt in range(attempts):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            retryable = exc.code in {429, 500, 502, 503, 504}
            if not retryable or attempt == attempts - 1:
                raise
            retry_after = exc.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else 1.5 * (attempt + 1)
            time.sleep(min(wait, 8))
        except (TimeoutError, urllib.error.URLError, ConnectionError) as exc:
            if attempt == attempts - 1:
                raise
            time.sleep(1.0 * (attempt + 1))


def http_json(url: str, params: dict[str, Any] | None = None, timeout: int = 25) -> Any:
    if params:
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v not in (None, "")}, doseq=True)
        url = f"{url}{'&' if '?' in url else '?'}{query}"
    req = urllib.request.Request(url, headers={"User-Agent": "weixin-paper-radar/0.2"})
    with open_url_with_retries(req, timeout=timeout) as response:
        return json.loads(response.read().decode(response.headers.get_content_charset() or "utf-8"))


def http_text(url: str, params: dict[str, Any] | None = None, timeout: int = 25) -> str:
    if params:
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v not in (None, "")}, doseq=True)
        url = f"{url}{'&' if '?' in url else '?'}{query}"
    req = urllib.request.Request(url, headers={"User-Agent": "weixin-paper-radar/0.2"})
    with open_url_with_retries(req, timeout=timeout) as response:
        return response.read().decode(response.headers.get_content_charset() or "utf-8")


MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def valid_year(value: Any) -> str:
    try:
        year = int(value)
    except (TypeError, ValueError):
        return ""
    current_year = date.today().year
    return str(year) if 1800 <= year <= current_year else ""


def valid_iso_date(value: str) -> str:
    text = str(value or "").strip()
    if not re.match(r"^\d{4}(?:-\d{2})?(?:-\d{2})?$", text):
        return ""
    parts = [int(part) for part in text.split("-")]
    try:
        parsed = date(parts[0], parts[1] if len(parts) > 1 else 1, parts[2] if len(parts) > 2 else 1)
    except ValueError:
        return ""
    return text if date(1800, 1, 1) <= parsed <= date.today() else ""


def date_from_parts(parts: Any) -> str:
    if not isinstance(parts, list) or not parts:
        return ""
    first_part = parts[0]
    if not isinstance(first_part, list) or not first_part:
        return ""
    try:
        year = int(first_part[0])
        month = int(first_part[1]) if len(first_part) > 1 else 1
        day = int(first_part[2]) if len(first_part) > 2 else 1
    except (TypeError, ValueError):
        return ""
    try:
        parsed = date(year, month, day)
    except ValueError:
        return ""
    if not date(1800, 1, 1) <= parsed <= date.today():
        return ""
    if len(first_part) > 2:
        return parsed.isoformat()
    if len(first_part) > 1:
        return f"{year:04d}-{month:02d}"
    return f"{year:04d}"


def year_from_date(value: str) -> str:
    match = re.match(r"^(\d{4})", str(value or ""))
    return valid_year(match.group(1)) if match else ""


def year_from(value: Any) -> str:
    if isinstance(value, dict):
        date_parts = value.get("date-parts")
        if isinstance(date_parts, list) and date_parts:
            first_part = date_parts[0]
            if isinstance(first_part, list) and first_part:
                parsed = valid_year(first_part[0])
                if parsed:
                    return parsed
        for key in ("date-time", "timestamp"):
            parsed = year_from(value.get(key))
            if parsed:
                return parsed
        return ""
    if isinstance(value, list):
        for item in value:
            parsed = year_from(item)
            if parsed:
                return parsed
        return ""
    for match in re.finditer(r"(?:18|19|20|21)\d{2}", str(value or "")):
        parsed = valid_year(match.group(0))
        if parsed:
            return parsed
    return ""


def crossref_year(item: dict[str, Any]) -> str:
    return year_from_date(crossref_publication_date(item)[0])


def crossref_publication_date(item: dict[str, Any]) -> tuple[str, str]:
    for key in ("published", "published-online", "published-print", "issued", "posted"):
        value = item.get(key)
        parsed = date_from_parts(value.get("date-parts") if isinstance(value, dict) else None)
        if parsed:
            return parsed, key
    return "", ""


def _pubmed_date_element(node: ET.Element, source: str) -> tuple[str, str]:
    year = valid_year(node.findtext("Year"))
    if not year:
        medline_year = year_from(node.findtext("MedlineDate"))
        return medline_year, "MedlineDate" if medline_year else ""
    month_raw = clean_text(node.findtext("Month"))
    day_raw = clean_text(node.findtext("Day"))
    month = ""
    if month_raw:
        if month_raw.isdigit():
            month = f"{int(month_raw):02d}" if 1 <= int(month_raw) <= 12 else ""
        else:
            month = f"{MONTHS.get(month_raw[:3].lower(), 0):02d}" if MONTHS.get(month_raw[:3].lower()) else ""
    if month and day_raw.isdigit():
        parsed = valid_iso_date(f"{year}-{month}-{int(day_raw):02d}")
        return (parsed, source) if parsed else ("", "")
    if month:
        parsed = valid_iso_date(f"{year}-{month}")
        return (parsed, source) if parsed else ("", "")
    return year, source


def pubmed_publication_date(article: ET.Element) -> tuple[str, str]:
    electronic_dates = [
        node
        for node in article.findall(".//Article/ArticleDate")
        if clean_text(node.attrib.get("DateType")).lower() == "electronic"
    ]
    for node in electronic_dates:
        parsed = _pubmed_date_element(node, "ArticleDate Electronic")
        if parsed[0]:
            return parsed

    pub_date = article.find(".//JournalIssue/PubDate")
    if pub_date is None:
        pub_date = article.find(".//PubDate")
    return _pubmed_date_element(pub_date, "PubDate") if pub_date is not None else ("", "")


def europe_pmc_publication_date(item: dict[str, Any]) -> tuple[str, str]:
    for key in ("firstPublicationDate", "electronicPublicationDate", "printPublicationDate"):
        parsed = valid_iso_date(item.get(key))
        if parsed:
            return parsed, key
    year = valid_year(item.get("pubYear"))
    return (year, "pubYear") if year else ("", "")


def openalex_publication_date(item: dict[str, Any]) -> tuple[str, str]:
    parsed = valid_iso_date(item.get("publication_date"))
    if parsed:
        return parsed, "publication_date"
    year = valid_year(item.get("publication_year"))
    return (year, "publication_year") if year else ("", "")


def parse_authors(items: list[Any], *keys: str) -> list[str]:
    authors: list[str] = []
    for item in items[:8]:
        if isinstance(item, str):
            name = item
        else:
            name = " ".join(str(item.get(key, "")).strip() for key in keys).strip()
            if not name and isinstance(item, dict):
                name = str(item.get("name") or item.get("display_name") or "").strip()
        if name:
            authors.append(clean_text(name))
    return authors


def since_dates(days: int | None) -> tuple[str, str]:
    end = date.today()
    start = end - timedelta(days=max(1, int(days or 365)))
    return start.isoformat(), end.isoformat()


def element_text(node: ET.Element | None) -> str:
    return clean_text("".join(node.itertext())) if node is not None else ""


def search_pubmed(query: str, limit: int, since_days: int | None = None) -> list[PaperInput]:
    params: dict[str, Any] = {"db": "pubmed", "term": query, "retmode": "json", "retmax": limit, "sort": "relevance"}
    if since_days:
        start, end = since_dates(since_days)
        params.update({"datetype": "pdat", "mindate": start, "maxdate": end})
    ids_json = http_text("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params)
    ids = (json.loads(ids_json).get("esearchresult") or {}).get("idlist") or []
    if not ids:
        return []
    time.sleep(0.34)
    xml_text = http_text(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
    )
    root = ET.fromstring(xml_text)
    records: list[PaperInput] = []
    for article in root.findall(".//PubmedArticle"):
        medline = article.find("./MedlineCitation")
        pmid = clean_text(medline.findtext("./PMID") if medline is not None else "")
        title = element_text(article.find(".//ArticleTitle"))
        journal = clean_text(article.findtext(".//Journal/Title"))
        abstract = clean_text(" ".join(element_text(node) for node in article.findall(".//AbstractText")))
        doi = ""
        pmcid = ""
        for node in article.findall(".//ArticleId"):
            if node.attrib.get("IdType") == "doi":
                doi = clean_doi(node.text)
            if node.attrib.get("IdType") == "pmc":
                pmcid = clean_text(node.text)
        authors = []
        for author in article.findall(".//Author")[:8]:
            name = clean_text(f"{author.findtext('ForeName') or ''} {author.findtext('LastName') or ''}")
            if name:
                authors.append(name)
        article_type = "; ".join(
            clean_text(node.text) for node in article.findall(".//PublicationType") if clean_text(node.text)
        )
        publication_date, publication_date_source = pubmed_publication_date(article)
        year = year_from_date(publication_date)
        oa_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/" if pmcid else ""
        if title:
            records.append(
                PaperInput(
                    title=title,
                    title_en=title,
                    doi=doi,
                    pmid=pmid,
                    authors=authors,
                    journal=journal,
                    year=year,
                    publication_date=publication_date,
                    publication_date_source=publication_date_source,
                    abstract=abstract,
                    abstract_en=abstract,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                    oa_pdf_url=oa_url,
                    source="PubMed",
                    is_open_access=bool(oa_url),
                    access_status="open" if oa_url else "unknown",
                    oa_source="PMC" if oa_url else "",
                    article_type=article_type,
                )
            )
    return records


def search_europe_pmc(query: str, limit: int, since_days: int | None = None) -> list[PaperInput]:
    epmc_query = query
    if since_days:
        start, end = since_dates(since_days)
        epmc_query = f"({query}) AND FIRST_PDATE:[{start} TO {end}]"
    data = http_json(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        {"query": epmc_query, "format": "json", "pageSize": limit, "resultType": "core"},
    )
    records: list[PaperInput] = []
    for item in (data.get("resultList") or {}).get("result") or []:
        title = clean_text(item.get("title"))
        if not title:
            continue
        authors = [clean_text(name) for name in str(item.get("authorString") or "").split(",")[:8] if name.strip()]
        pmcid = str(item.get("pmcid") or "")
        oa_url = f"https://europepmc.org/articles/{pmcid}?pdf=render" if pmcid else ""
        publication_date, publication_date_source = europe_pmc_publication_date(item)
        records.append(
            PaperInput(
                title=title,
                title_en=title,
                doi=clean_doi(item.get("doi")),
                pmid=str(item.get("pmid") or ""),
                authors=authors,
                journal=clean_text(item.get("journalTitle")),
                year=year_from_date(publication_date),
                publication_date=publication_date,
                publication_date_source=publication_date_source,
                abstract=clean_text(item.get("abstractText")),
                abstract_en=clean_text(item.get("abstractText")),
                url=f"https://europepmc.org/article/{item.get('source', 'MED')}/{item.get('id')}",
                oa_pdf_url=oa_url,
                source="Europe PMC",
                is_open_access=bool(oa_url),
                access_status="open" if oa_url else "unknown",
                oa_source="Europe PMC" if oa_url else "",
                article_type=clean_text(item.get("pubType") or item.get("pubTypeList")),
            )
        )
    return records


def inverted_abstract(index: Any) -> str:
    if not isinstance(index, dict):
        return ""
    pairs: list[tuple[int, str]] = []
    for word, positions in index.items():
        for pos in positions:
            pairs.append((int(pos), word))
    return clean_text(" ".join(word for _, word in sorted(pairs)))


def search_openalex(
    query: str,
    limit: int,
    email: str = "",
    since_days: int | None = None,
    api_key: str = "",
) -> list[PaperInput]:
    del email  # Retained only for backward-compatible callers; OpenAlex no longer uses mailto.
    api_key = clean_text(api_key or os.getenv("OPENALEX_API_KEY"))
    if not api_key:
        return []
    params: dict[str, Any] = {"search": query, "per-page": limit}
    filters = []
    if since_days:
        start, _ = since_dates(since_days)
        filters.append(f"from_publication_date:{start}")
    if filters:
        params["filter"] = ",".join(filters)
    params["api_key"] = api_key
    data = http_json("https://api.openalex.org/works", params)
    records: list[PaperInput] = []
    for item in data.get("results") or []:
        primary = item.get("primary_location") or {}
        source = primary.get("source") or {}
        oa = item.get("open_access") or {}
        doi = clean_doi(item.get("doi"))
        authors = [
            clean_text(((auth.get("author") or {}).get("display_name") or ""))
            for auth in (item.get("authorships") or [])[:8]
        ]
        pdf_url = clean_text(primary.get("pdf_url") or oa.get("oa_url"))
        is_oa = bool(oa.get("is_oa") or pdf_url)
        publication_date, publication_date_source = openalex_publication_date(item)
        records.append(
            PaperInput(
                title=clean_text(item.get("title")),
                title_en=clean_text(item.get("title")),
                doi=doi,
                authors=[name for name in authors if name],
                journal=clean_text(source.get("display_name")),
                year=year_from_date(publication_date),
                publication_date=publication_date,
                publication_date_source=publication_date_source,
                abstract=inverted_abstract(item.get("abstract_inverted_index")),
                abstract_en=inverted_abstract(item.get("abstract_inverted_index")),
                url=item.get("doi") or item.get("id") or "",
                oa_pdf_url=pdf_url,
                source="OpenAlex",
                is_open_access=is_oa,
                access_status="open" if pdf_url else ("unknown" if is_oa else "paywalled"),
                oa_source="OpenAlex" if pdf_url else "",
                article_type=clean_text(item.get("type") or item.get("type_crossref")),
            )
        )
    return [record for record in records if record.title]


def search_crossref(query: str, limit: int, since_days: int | None = None) -> list[PaperInput]:
    params: dict[str, Any] = {"query.bibliographic": query, "rows": limit}
    if since_days:
        start, _ = since_dates(since_days)
        params["filter"] = f"from-pub-date:{start}"
    data = http_json("https://api.crossref.org/works", params)
    records: list[PaperInput] = []
    for item in (data.get("message") or {}).get("items") or []:
        record = crossref_item_to_record(item, source="Crossref")
        if record:
            records.append(record)
    return records


def search_crossref_many(queries: list[str], limit: int, since_days: int | None = None) -> list[PaperInput]:
    records: list[PaperInput] = []
    if not queries:
        return records
    per_query = max(1, min(20, (limit + len(queries) - 1) // len(queries)))
    for query in queries:
        records.extend(search_crossref(query, per_query, since_days=since_days))
        time.sleep(0.15)
    return dedupe(records)[:limit]


def crossref_item_to_record(item: dict[str, Any], *, source: str = "Crossref") -> PaperInput | None:
    title = clean_text(" ".join(item.get("title") or []))
    if not title:
        return None
    journal = clean_text(" ".join(item.get("container-title") or []))
    publication_date, publication_date_source = crossref_publication_date(item)
    return PaperInput(
        title=title,
        title_en=title,
        doi=clean_doi(item.get("DOI")),
        authors=parse_authors(item.get("author") or [], "given", "family"),
        journal=journal,
        year=year_from_date(publication_date),
        publication_date=publication_date,
        publication_date_source=publication_date_source,
        abstract=clean_text(item.get("abstract")),
        abstract_en=clean_text(item.get("abstract")),
        url=clean_text(item.get("URL")),
        source=source,
        access_status="unknown",
        article_type=clean_text(item.get("type")),
    )


def matching_journal(record: PaperInput, journals: list[JournalFilter]) -> JournalFilter | None:
    for journal in journals:
        if journal_matches_name(record.journal, journal):
            return journal
    return None


def filter_latest_records(records: list[PaperInput], journals: list[JournalFilter]) -> list[PaperInput]:
    filtered: list[PaperInput] = []
    for record in records:
        journal = matching_journal(record, journals)
        if journal is None:
            continue
        if not should_keep_article_type(record.article_type):
            continue
        filtered.append(decorate_journal_record(record, journal))
    return filtered


def search_pubmed_latest(journals: list[JournalFilter], limit: int, since_days: int | None = 7) -> list[PaperInput]:
    query = "(" + " OR ".join(build_pubmed_journal_query(journal) for journal in journals) + ")"
    return filter_latest_records(search_pubmed(query, limit, since_days=since_days), journals)


def search_europe_pmc_latest(journals: list[JournalFilter], limit: int, since_days: int | None = 7) -> list[PaperInput]:
    query = "(" + " OR ".join(build_europe_pmc_journal_query(journal) for journal in journals) + ")"
    return filter_latest_records(search_europe_pmc(query, limit, since_days=None), journals)


def search_openalex_latest(
    journals: list[JournalFilter],
    limit: int,
    since_days: int | None = 7,
    api_key: str = "",
) -> list[PaperInput]:
    api_key = clean_text(api_key or os.getenv("OPENALEX_API_KEY"))
    if not api_key:
        return []
    params: dict[str, Any] = {"per-page": limit, "sort": "publication_date:desc", "api_key": api_key}
    filters: list[str] = []
    if since_days:
        start, _ = since_dates(since_days)
        filters.append(f"from_publication_date:{start}")
    issns = list(dict.fromkeys(issn for journal in journals for issn in journal.issns))
    if issns:
        filters.append("primary_location.source.issn:" + "|".join(issns))
    else:
        params["search"] = " OR ".join(journal.name for journal in journals)
    if filters:
        params["filter"] = ",".join(filters)
    data = http_json("https://api.openalex.org/works", params)
    records: list[PaperInput] = []
    for item in data.get("results") or []:
        primary = item.get("primary_location") or {}
        source = primary.get("source") or {}
        oa = item.get("open_access") or {}
        publication_date, publication_date_source = openalex_publication_date(item)
        pdf_url = clean_text(primary.get("pdf_url") or oa.get("oa_url"))
        records.append(
            PaperInput(
                title=clean_text(item.get("title")),
                title_en=clean_text(item.get("title")),
                doi=clean_doi(item.get("doi")),
                authors=[
                    clean_text(((auth.get("author") or {}).get("display_name") or ""))
                    for auth in (item.get("authorships") or [])[:8]
                    if clean_text(((auth.get("author") or {}).get("display_name") or ""))
                ],
                journal=clean_text(source.get("display_name")),
                year=year_from_date(publication_date),
                publication_date=publication_date,
                publication_date_source=publication_date_source,
                abstract=inverted_abstract(item.get("abstract_inverted_index")),
                abstract_en=inverted_abstract(item.get("abstract_inverted_index")),
                url=item.get("doi") or item.get("id") or "",
                oa_pdf_url=pdf_url,
                source="OpenAlex",
                is_open_access=bool(oa.get("is_oa") or pdf_url),
                access_status="open" if pdf_url else ("unknown" if oa.get("is_oa") else "paywalled"),
                oa_source="OpenAlex" if pdf_url else "",
                article_type=clean_text(item.get("type") or item.get("type_crossref")),
            )
        )
    return filter_latest_records([record for record in records if record.title], journals)


def search_crossref_latest(journals: list[JournalFilter], limit: int, since_days: int | None = 7) -> list[PaperInput]:
    records: list[PaperInput] = []
    if not journals:
        return records
    per_journal = max(3, min(25, (limit + len(journals) - 1) // len(journals) + 3))
    for journal in journals:
        urls = crossref_journal_urls(journal)
        for url in urls:
            params: dict[str, Any] = {"rows": per_journal, "sort": "published", "order": "desc"}
            if not journal.issns:
                params["query.container-title"] = journal.name
            if since_days:
                start, _ = since_dates(since_days)
                params["filter"] = f"from-pub-date:{start}"
            data = http_json(url, params)
            for item in (data.get("message") or {}).get("items") or []:
                record = crossref_item_to_record(item, source="Crossref")
                if record:
                    records.append(record)
            time.sleep(0.05)
    return filter_latest_records(dedupe(records), journals)[:limit]


def dedupe(records: list[PaperInput]) -> list[PaperInput]:
    by_key: dict[str, PaperInput] = {}
    for record in records:
        key = clean_doi(record.doi)
        if not key:
            key = re.sub(r"[^a-z0-9]+", "", (record.title_en or record.title).lower())[:120]
        if not key:
            continue
        existing = by_key.get(key)
        if not existing:
            by_key[key] = record
            continue
        for field_name in (
            "abstract",
            "abstract_en",
            "journal",
            "year",
            "url",
            "oa_pdf_url",
            "pmid",
            "title_zh",
            "abstract_zh",
            "oa_source",
            "download_error",
            "publication_date",
            "publication_date_source",
        ):
            if not getattr(existing, field_name) and getattr(record, field_name):
                setattr(existing, field_name, getattr(record, field_name))
        if len(record.authors) > len(existing.authors):
            existing.authors = record.authors
        if record.is_open_access:
            existing.is_open_access = True
            existing.access_status = "open"
        if record.source not in existing.source:
            existing.source = f"{existing.source}, {record.source}"
    return list(by_key.values())


def mark_paywalled(records: list[PaperInput]) -> list[PaperInput]:
    for record in records:
        if record.oa_pdf_url:
            record.is_open_access = True
            record.access_status = "open"
        elif record.access_status == "unknown":
            record.access_status = "paywalled"
            record.is_open_access = False
    return records


def federated_search(
    keywords: str | list[str] | None = None,
    limit: int = 20,
    sources: list[str] | None = None,
    email: str = "",
    since_days: int | None = None,
    *,
    query_plan: SearchQueryPlan | dict[str, Any] | None = None,
    search_mode: str | None = None,
    openalex_api_key: str = "",
    llm_provider: str = "",
    llm_api_key: str = "",
    llm_base_url: str = "",
    llm_model: str = "",
    keyword_resolver: KeywordResolver | None = None,
    diagnostics: SearchDiagnostics | None = None,
) -> tuple[list[PaperInput], dict[str, str]]:
    if query_plan is None:
        from .llm import default_api_key, default_base_url, default_model, default_provider

        provider = llm_provider or default_provider()
        plan = resolve_keyword_plan(
            keywords,
            search_mode=search_mode or "strict",
            resolver=keyword_resolver,
            llm_api_key=llm_api_key or default_api_key(),
            llm_base_url=llm_base_url or default_base_url(provider),
            llm_model=llm_model or default_model(provider),
        )
    else:
        plan = ensure_query_plan(keywords, query_plan, search_mode)

    mode = normalize_search_mode(search_mode or plan.search_mode)
    plan.search_mode = mode
    per_source = min(100, max(5, int(limit) * 3))
    openalex_key = clean_text(openalex_api_key or os.getenv("OPENALEX_API_KEY"))
    default_sources = ["PubMed", "Europe PMC", "Crossref"]
    if openalex_key:
        default_sources.append("OpenAlex")
    selected = list(dict.fromkeys(default_sources if sources is None else sources))
    diag = diagnostics if diagnostics is not None else SearchDiagnostics()
    diag.source_counts.clear()
    diag.raw_count = 0
    diag.deduplicated_count = 0
    diag.filtered_count = 0
    diag.warnings = list(plan.warnings)
    diag.errors.clear()
    for source in selected:
        diag.source_counts[source] = {"fetched": 0, "deduplicated": 0, "relevant": 0}

    pubmed_query = build_pubmed_query(plan)
    europe_pmc_query = build_europe_pmc_query(plan, since_days=since_days)
    openalex_query = build_openalex_query(plan)
    crossref_queries = build_crossref_queries(plan)
    functions = {
        "PubMed": lambda: search_pubmed(pubmed_query, per_source, since_days=since_days),
        "Europe PMC": lambda: search_europe_pmc(europe_pmc_query, per_source, since_days=None),
        "Crossref": lambda: search_crossref_many(crossref_queries, per_source, since_days=since_days),
    }
    if "OpenAlex" in selected:
        if openalex_key:
            functions["OpenAlex"] = lambda: search_openalex(
                openalex_query,
                per_source,
                email=email,
                since_days=since_days,
                api_key=openalex_key,
            )
        else:
            diag.warnings.append("OpenAlex 未配置 OPENALEX_API_KEY，已跳过且未发起网络请求。")
    if email:
        diag.warnings.append("OpenAlex 邮箱/mailto 配置已弃用，请改用 OPENALEX_API_KEY。")
    unknown_sources = [source for source in selected if source not in functions and source != "OpenAlex"]
    if unknown_sources:
        diag.warnings.append("未知检索源已跳过：" + ", ".join(unknown_sources))

    records_by_source: dict[str, list[PaperInput]] = {}
    errors: dict[str, str] = {}
    runnable = {name: functions[name] for name in selected if name in functions}
    if runnable:
        with ThreadPoolExecutor(max_workers=min(4, len(runnable))) as pool:
            futures = {pool.submit(function): name for name, function in runnable.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    records_by_source[name] = future.result()
                except Exception as exc:
                    errors[name] = f"{type(exc).__name__}: {exc}"
                    records_by_source[name] = []

    records: list[PaperInput] = []
    for source in selected:
        source_records = records_by_source.get(source, [])
        source_deduplicated = dedupe(source_records)
        source_relevant = [
            record for record in source_deduplicated if is_synthetic_biology_relevant(record, plan, mode)
        ]
        counts = diag.source_counts[source]
        counts["fetched"] = len(source_records)
        counts["deduplicated"] = len(source_deduplicated)
        counts["relevant"] = len(source_relevant)
        records.extend(source_records)

    diag.errors.update(errors)
    diag.raw_count = len(records)
    merged = mark_paywalled(dedupe(records))
    diag.deduplicated_count = len(merged)
    relevant = [record for record in merged if is_synthetic_biology_relevant(record, plan, mode)]
    for record in merged:
        record.keywords = plan.original_keywords
    relevant.sort(
        key=lambda item: (
            relevance_score(item, plan, mode),
            item.publication_date or item.year or "",
            item.is_open_access,
            bool(item.abstract_en),
            bool(item.doi),
        ),
        reverse=True,
    )
    diag.filtered_count = len(relevant)
    if diag.raw_count and not relevant:
        diag.warnings.append(f"共抓取 {diag.raw_count} 条记录，但均未通过“{mode}”相关性过滤。")
    elif not diag.raw_count and runnable and not errors:
        diag.warnings.append("已完成检索，但各来源均未返回记录。")
    return relevant[:limit], errors


def journal_latest_search(
    journals: list[JournalFilter],
    limit: int = 100,
    sources: list[str] | None = None,
    since_days: int | None = 7,
    *,
    openalex_api_key: str = "",
    diagnostics: SearchDiagnostics | None = None,
) -> tuple[list[PaperInput], dict[str, str]]:
    active_journals = sorted([journal for journal in journals if journal.enabled], key=lambda item: item.priority)
    openalex_key = clean_text(openalex_api_key or os.getenv("OPENALEX_API_KEY"))
    default_sources = ["PubMed", "Europe PMC", "Crossref"]
    if openalex_key:
        default_sources.append("OpenAlex")
    selected = list(dict.fromkeys(sources or default_sources))
    per_source = max(int(limit) * 2, int(limit) + 20)
    diag = diagnostics if diagnostics is not None else SearchDiagnostics()
    diag.source_counts.clear()
    diag.raw_count = 0
    diag.deduplicated_count = 0
    diag.filtered_count = 0
    diag.warnings = []
    diag.errors.clear()
    for source in selected:
        diag.source_counts[source] = {"fetched": 0, "deduplicated": 0, "relevant": 0}
    if not active_journals:
        diag.warnings.append("未启用任何期刊，已跳过检索。")
        return [], {}

    functions: dict[str, Callable[[], list[PaperInput]]] = {
        "PubMed": lambda: search_pubmed_latest(active_journals, per_source, since_days=since_days),
        "Europe PMC": lambda: search_europe_pmc_latest(active_journals, per_source, since_days=since_days),
        "Crossref": lambda: search_crossref_latest(active_journals, per_source, since_days=since_days),
    }
    if "OpenAlex" in selected:
        if openalex_key:
            functions["OpenAlex"] = lambda: search_openalex_latest(
                active_journals,
                per_source,
                since_days=since_days,
                api_key=openalex_key,
            )
        else:
            diag.warnings.append("OpenAlex 未配置 OPENALEX_API_KEY，已跳过且未发起网络请求。")
    unknown_sources = [source for source in selected if source not in functions and source != "OpenAlex"]
    if unknown_sources:
        diag.warnings.append("未知检索源已跳过：" + ", ".join(unknown_sources))

    records_by_source: dict[str, list[PaperInput]] = {}
    errors: dict[str, str] = {}
    runnable = {name: functions[name] for name in selected if name in functions}
    if runnable:
        with ThreadPoolExecutor(max_workers=min(4, len(runnable))) as pool:
            futures = {pool.submit(function): name for name, function in runnable.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    records_by_source[name] = future.result()
                except Exception as exc:
                    errors[name] = f"{type(exc).__name__}: {exc}"
                    records_by_source[name] = []

    records: list[PaperInput] = []
    for source in selected:
        source_records = records_by_source.get(source, [])
        source_filtered = filter_latest_records(source_records, active_journals)
        source_deduplicated = dedupe(source_filtered)
        counts = diag.source_counts[source]
        counts["fetched"] = len(source_records)
        counts["deduplicated"] = len(source_deduplicated)
        counts["relevant"] = len(source_deduplicated)
        records.extend(source_filtered)

    diag.errors.update(errors)
    diag.raw_count = len(records)
    merged = mark_paywalled(dedupe(records))
    merged.sort(
        key=lambda item: (
            -(item.journal_priority or 9999),
            item.publication_date or item.year or "",
            bool(item.abstract_en),
            bool(item.doi),
        ),
        reverse=True,
    )
    diag.deduplicated_count = len(merged)
    diag.filtered_count = len(merged)
    if not merged and runnable and not errors:
        diag.warnings.append("已完成最新文章检索，但各来源均未返回符合期刊与文章类型条件的记录。")
    return merged[:limit], errors


def run_journal_latest_search(
    journals: list[JournalFilter],
    limit: int = 100,
    sources: list[str] | None = None,
    since_days: int | None = 7,
    *,
    openalex_api_key: str = "",
) -> SearchRun:
    started = utc_now()
    diagnostics = SearchDiagnostics()
    records, errors = journal_latest_search(
        journals,
        limit=limit,
        sources=sources,
        since_days=since_days,
        openalex_api_key=openalex_api_key,
        diagnostics=diagnostics,
    )
    return SearchRun(
        run_id=started.replace(":", "").replace("-", "").split(".")[0],
        keywords=[journal.name for journal in journals if journal.enabled],
        started_at=started,
        finished_at=utc_now(),
        records=records,
        errors=errors,
        source_counts=diagnostics.source_counts,
        raw_count=diagnostics.raw_count,
        filtered_count=diagnostics.filtered_count,
        warnings=list(dict.fromkeys(diagnostics.warnings)),
        search_kind="journal_latest",
        journal_filters=[journal.to_dict() for journal in journals if journal.enabled],
    )


def run_keyword_search(
    keywords: str | list[str],
    limit: int = 20,
    email: str = "",
    since_days: int | None = 30,
    *,
    query_plan: SearchQueryPlan | dict[str, Any] | None = None,
    search_mode: str | None = None,
    openalex_api_key: str = "",
    llm_provider: str = "",
    llm_api_key: str = "",
    llm_base_url: str = "",
    llm_model: str = "",
    keyword_resolver: KeywordResolver | None = None,
) -> SearchRun:
    started = utc_now()
    if query_plan is None:
        from .llm import default_api_key, default_base_url, default_model, default_provider

        provider = llm_provider or default_provider()
        plan = resolve_keyword_plan(
            keywords,
            search_mode=search_mode or "strict",
            resolver=keyword_resolver,
            llm_api_key=llm_api_key or default_api_key(),
            llm_base_url=llm_base_url or default_base_url(provider),
            llm_model=llm_model or default_model(provider),
        )
    else:
        plan = ensure_query_plan(keywords, query_plan, search_mode)
    diagnostics = SearchDiagnostics()
    records, errors = federated_search(
        keywords,
        limit=limit,
        email=email,
        since_days=since_days,
        query_plan=plan,
        search_mode=search_mode,
        openalex_api_key=openalex_api_key,
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        keyword_resolver=keyword_resolver,
        diagnostics=diagnostics,
    )
    return SearchRun(
        run_id=started.replace(":", "").replace("-", "").split(".")[0],
        keywords=plan.original_keywords,
        started_at=started,
        finished_at=utc_now(),
        records=records,
        errors=errors,
        query_plan=plan,
        source_counts=diagnostics.source_counts,
        raw_count=diagnostics.raw_count,
        filtered_count=diagnostics.filtered_count,
        warnings=list(dict.fromkeys(diagnostics.warnings)),
    )


def parse_manual_inputs(text: str) -> list[PaperInput]:
    records: list[PaperInput] = []
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", line, flags=re.I)
        pmid_match = re.search(r"\bPMID[:\s]*(\d{6,10})\b", line, flags=re.I)
        doi = clean_doi(doi_match.group(0)) if doi_match else ""
        pmid = pmid_match.group(1) if pmid_match else ""
        title = clean_text(re.sub(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", "", line, flags=re.I))
        title = re.sub(r"\bPMID[:\s]*\d{6,10}\b", "", title, flags=re.I).strip(" -|,")
        records.append(PaperInput(title=title or doi or pmid, title_en=title or doi or pmid, doi=doi, pmid=pmid, source="manual"))
    return records


def resolve_doi(doi: str) -> PaperInput | None:
    doi = clean_doi(doi)
    if not doi:
        return None
    data = http_json(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
    item = data.get("message") or {}
    title = clean_text(" ".join(item.get("title") or []))
    publication_date, publication_date_source = crossref_publication_date(item)
    return PaperInput(
        title=title or doi,
        title_en=title or doi,
        doi=doi,
        authors=parse_authors(item.get("author") or [], "given", "family"),
        journal=clean_text(" ".join(item.get("container-title") or [])),
        year=year_from_date(publication_date),
        publication_date=publication_date,
        publication_date_source=publication_date_source,
        abstract=clean_text(item.get("abstract")),
        abstract_en=clean_text(item.get("abstract")),
        url=clean_text(item.get("URL") or f"https://doi.org/{doi}"),
        source="Crossref DOI",
    )
