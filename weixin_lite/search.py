from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .models import PaperInput


DEFAULT_TOPIC = (
    '("terminal deoxynucleotidyl transferase" OR TdT OR "poly(U) polymerase" OR PUP) '
    'AND ("enzymatic DNA synthesis" OR "enzymatic RNA synthesis" OR nucleic acid synthesis)'
)


class SearchError(RuntimeError):
    pass


def clean_doi(value: Any) -> str:
    doi = str(value or "").strip()
    doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", doi, flags=re.I)
    return doi.rstrip(" .").lower()


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(value or ""))).strip()


def http_json(url: str, params: dict[str, Any] | None = None, timeout: int = 25) -> Any:
    if params:
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v not in (None, "")}, doseq=True)
        url = f"{url}{'&' if '?' in url else '?'}{query}"
    req = urllib.request.Request(url, headers={"User-Agent": "weixin-paper-reader/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode(response.headers.get_content_charset() or "utf-8"))


def http_text(url: str, params: dict[str, Any] | None = None, timeout: int = 25) -> str:
    if params:
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v not in (None, "")}, doseq=True)
        url = f"{url}{'&' if '?' in url else '?'}{query}"
    req = urllib.request.Request(url, headers={"User-Agent": "weixin-paper-reader/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode(response.headers.get_content_charset() or "utf-8")


def year_from(value: Any) -> str:
    match = re.search(r"(?:19|20|21)\d{2}", str(value or ""))
    return match.group(0) if match else ""


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


def search_pubmed(query: str, limit: int) -> list[PaperInput]:
    ids_xml = http_text(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        {"db": "pubmed", "term": query, "retmode": "json", "retmax": limit, "sort": "relevance"},
    )
    ids = (json.loads(ids_xml).get("esearchresult") or {}).get("idlist") or []
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
        title = clean_text(article.findtext(".//ArticleTitle"))
        journal = clean_text(article.findtext(".//Journal/Title"))
        abstract = clean_text(" ".join(node.text or "" for node in article.findall(".//AbstractText")))
        doi = ""
        for node in article.findall(".//ArticleId"):
            if node.attrib.get("IdType") == "doi":
                doi = clean_doi(node.text)
                break
        authors = []
        for author in article.findall(".//Author")[:8]:
            last = author.findtext("LastName") or ""
            fore = author.findtext("ForeName") or ""
            name = clean_text(f"{fore} {last}")
            if name:
                authors.append(name)
        year = year_from(article.findtext(".//PubDate/Year") or article.findtext(".//PubDate/MedlineDate"))
        if title:
            records.append(
                PaperInput(
                    title=title,
                    doi=doi,
                    pmid=pmid,
                    authors=authors,
                    journal=journal,
                    year=year,
                    abstract=abstract,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                    source="PubMed",
                )
            )
    return records


def search_europe_pmc(query: str, limit: int) -> list[PaperInput]:
    data = http_json(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        {"query": query, "format": "json", "pageSize": limit, "resultType": "core"},
    )
    records: list[PaperInput] = []
    for item in (data.get("resultList") or {}).get("result") or []:
        title = clean_text(item.get("title"))
        if not title:
            continue
        authors = [clean_text(name) for name in str(item.get("authorString") or "").split(",")[:8] if name.strip()]
        pmcid = str(item.get("pmcid") or "")
        records.append(
            PaperInput(
                title=title,
                doi=clean_doi(item.get("doi")),
                pmid=str(item.get("pmid") or ""),
                authors=authors,
                journal=clean_text(item.get("journalTitle")),
                year=year_from(item.get("pubYear")),
                abstract=clean_text(item.get("abstractText")),
                url=f"https://europepmc.org/article/{item.get('source', 'MED')}/{item.get('id')}",
                oa_pdf_url=f"https://europepmc.org/articles/{pmcid}?pdf=render" if pmcid else "",
                source="Europe PMC",
            )
        )
    return records


def search_openalex(query: str, limit: int, email: str = "") -> list[PaperInput]:
    params = {"search": query, "per-page": limit}
    if email:
        params["mailto"] = email
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
        records.append(
            PaperInput(
                title=clean_text(item.get("title")),
                doi=doi,
                authors=[name for name in authors if name],
                journal=clean_text(source.get("display_name")),
                year=year_from(item.get("publication_year")),
                abstract=inverted_abstract(item.get("abstract_inverted_index")),
                url=item.get("doi") or item.get("id") or "",
                oa_pdf_url=clean_text(primary.get("pdf_url") or oa.get("oa_url")),
                source="OpenAlex",
            )
        )
    return [record for record in records if record.title]


def search_crossref(query: str, limit: int) -> list[PaperInput]:
    data = http_json("https://api.crossref.org/works", {"query": query, "rows": limit})
    records: list[PaperInput] = []
    for item in (data.get("message") or {}).get("items") or []:
        title = clean_text(" ".join(item.get("title") or []))
        if not title:
            continue
        journal = clean_text(" ".join(item.get("container-title") or []))
        authors = parse_authors(item.get("author") or [], "given", "family")
        records.append(
            PaperInput(
                title=title,
                doi=clean_doi(item.get("DOI")),
                authors=authors,
                journal=journal,
                year=year_from(item.get("published-print") or item.get("published-online") or item.get("created")),
                abstract=clean_text(item.get("abstract")),
                url=clean_text(item.get("URL")),
                source="Crossref",
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


def dedupe(records: list[PaperInput]) -> list[PaperInput]:
    by_key: dict[str, PaperInput] = {}
    for record in records:
        key = clean_doi(record.doi)
        if not key:
            key = re.sub(r"[^a-z0-9]+", "", record.title.lower())[:120]
        if not key:
            continue
        existing = by_key.get(key)
        if not existing:
            by_key[key] = record
            continue
        for field in ("abstract", "journal", "year", "url", "oa_pdf_url", "pmid"):
            if not getattr(existing, field) and getattr(record, field):
                setattr(existing, field, getattr(record, field))
        if len(record.authors) > len(existing.authors):
            existing.authors = record.authors
        existing.source = f"{existing.source}, {record.source}" if record.source not in existing.source else existing.source
    return list(by_key.values())


def federated_search(
    query: str = DEFAULT_TOPIC,
    limit: int = 20,
    sources: list[str] | None = None,
    email: str = "",
) -> tuple[list[PaperInput], dict[str, str]]:
    selected = sources or ["PubMed", "Europe PMC", "OpenAlex", "Crossref"]
    per_source = max(5, min(50, limit))
    functions = {
        "PubMed": lambda: search_pubmed(query, per_source),
        "Europe PMC": lambda: search_europe_pmc(query, per_source),
        "OpenAlex": lambda: search_openalex(query, per_source, email=email),
        "Crossref": lambda: search_crossref(query, per_source),
    }
    records: list[PaperInput] = []
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(4, len(selected))) as pool:
        futures = {pool.submit(functions[name]): name for name in selected if name in functions}
        for future in as_completed(futures):
            name = futures[future]
            try:
                records.extend(future.result())
            except Exception as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"
    merged = dedupe(records)
    merged.sort(key=lambda item: (item.year or "", bool(item.abstract), bool(item.doi)), reverse=True)
    return merged[:limit], errors


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
        records.append(PaperInput(title=title or doi or pmid, doi=doi, pmid=pmid, source="manual"))
    return records


def resolve_doi(doi: str) -> PaperInput | None:
    doi = clean_doi(doi)
    if not doi:
        return None
    data = http_json(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
    item = data.get("message") or {}
    title = clean_text(" ".join(item.get("title") or []))
    return PaperInput(
        title=title or doi,
        doi=doi,
        authors=parse_authors(item.get("author") or [], "given", "family"),
        journal=clean_text(" ".join(item.get("container-title") or [])),
        year=year_from(item.get("published-print") or item.get("published-online") or item.get("created")),
        abstract=clean_text(item.get("abstract")),
        url=clean_text(item.get("URL") or f"https://doi.org/{doi}"),
        source="Crossref DOI",
    )
