from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Any

from .models import PaperInput, SearchRun, utc_now


DEFAULT_KEYWORDS = ["TdT", "PUP", "酶促DNA合成", "酶促RNA合成", "enzyme engineering"]
KEYWORD_EXPANSIONS = {
    "tdt": ["terminal deoxynucleotidyl transferase", "TdT"],
    "pup": ["poly(U) polymerase", "PUP", "polynucleotide phosphorylase"],
    "酶促dna合成": ["enzymatic DNA synthesis", "template-independent DNA synthesis"],
    "酶促rna合成": ["enzymatic RNA synthesis", "RNA enzymatic synthesis"],
    "enzyme engineering": ["enzyme engineering", "directed evolution", "protein engineering"],
}


class SearchError(RuntimeError):
    pass


def clean_doi(value: Any) -> str:
    doi = str(value or "").strip()
    doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", doi, flags=re.I)
    return doi.rstrip(" .").lower()


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(value or ""))).strip()


def parse_keywords(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = re.split(r"[,，;\n]+", str(value or ""))
    keywords = [clean_text(item) for item in raw_items if clean_text(item)]
    return list(dict.fromkeys(keywords))


def build_keyword_query(keywords: str | list[str]) -> str:
    parsed = parse_keywords(keywords)
    if not parsed:
        parsed = DEFAULT_KEYWORDS
    groups: list[str] = []
    for keyword in parsed:
        expansions = KEYWORD_EXPANSIONS.get(keyword.lower(), [keyword])
        terms = [f'"{term}"' if " " in term else term for term in expansions]
        groups.append("(" + " OR ".join(terms) + ")")
    domain = '("enzymatic DNA synthesis" OR "enzymatic RNA synthesis" OR "nucleic acid synthesis" OR TdT OR PUP)'
    return "(" + " OR ".join(groups) + f") AND {domain}"


def http_json(url: str, params: dict[str, Any] | None = None, timeout: int = 25) -> Any:
    if params:
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v not in (None, "")}, doseq=True)
        url = f"{url}{'&' if '?' in url else '?'}{query}"
    req = urllib.request.Request(url, headers={"User-Agent": "weixin-paper-radar/0.2"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode(response.headers.get_content_charset() or "utf-8"))


def http_text(url: str, params: dict[str, Any] | None = None, timeout: int = 25) -> str:
    if params:
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v not in (None, "")}, doseq=True)
        url = f"{url}{'&' if '?' in url else '?'}{query}"
    req = urllib.request.Request(url, headers={"User-Agent": "weixin-paper-radar/0.2"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
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


def pubmed_publication_date(article: ET.Element) -> tuple[str, str]:
    pub_date = article.find(".//JournalIssue/PubDate") or article.find(".//PubDate")
    if pub_date is None:
        return "", ""
    year = valid_year(pub_date.findtext("Year"))
    if not year:
        medline_year = year_from(pub_date.findtext("MedlineDate"))
        return medline_year, "MedlineDate" if medline_year else ""
    month_raw = clean_text(pub_date.findtext("Month"))
    day_raw = clean_text(pub_date.findtext("Day"))
    month = ""
    if month_raw:
        if month_raw.isdigit():
            month = f"{int(month_raw):02d}" if 1 <= int(month_raw) <= 12 else ""
        else:
            month = f"{MONTHS.get(month_raw[:3].lower(), 0):02d}" if MONTHS.get(month_raw[:3].lower()) else ""
    if month and day_raw.isdigit():
        parsed = valid_iso_date(f"{year}-{month}-{int(day_raw):02d}")
        return parsed, "PubDate" if parsed else (year, "PubDate")
    if month:
        parsed = valid_iso_date(f"{year}-{month}")
        return parsed, "PubDate" if parsed else (year, "PubDate")
    return year, "PubDate"


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


def search_pubmed(query: str, limit: int, since_days: int | None = None) -> list[PaperInput]:
    params: dict[str, Any] = {"db": "pubmed", "term": query, "retmode": "json", "retmax": limit, "sort": "pub date"}
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
        title = clean_text(article.findtext(".//ArticleTitle"))
        journal = clean_text(article.findtext(".//Journal/Title"))
        abstract = clean_text(" ".join(node.text or "" for node in article.findall(".//AbstractText")))
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
        {"query": epmc_query, "format": "json", "pageSize": limit, "resultType": "core", "sort": "P_PDATE_D"},
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


def search_openalex(query: str, limit: int, email: str = "", since_days: int | None = None) -> list[PaperInput]:
    params: dict[str, Any] = {"search": query, "per-page": limit, "sort": "publication_date:desc"}
    filters = []
    if since_days:
        start, _ = since_dates(since_days)
        filters.append(f"from_publication_date:{start}")
    if filters:
        params["filter"] = ",".join(filters)
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
            )
        )
    return [record for record in records if record.title]


def search_crossref(query: str, limit: int, since_days: int | None = None) -> list[PaperInput]:
    params: dict[str, Any] = {"query": query, "rows": limit, "sort": "published", "order": "desc"}
    if since_days:
        start, _ = since_dates(since_days)
        params["filter"] = f"from-pub-date:{start}"
    data = http_json("https://api.crossref.org/works", params)
    records: list[PaperInput] = []
    for item in (data.get("message") or {}).get("items") or []:
        title = clean_text(" ".join(item.get("title") or []))
        if not title:
            continue
        journal = clean_text(" ".join(item.get("container-title") or []))
        publication_date, publication_date_source = crossref_publication_date(item)
        records.append(
            PaperInput(
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
                source="Crossref",
                access_status="unknown",
            )
        )
    return records


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
) -> tuple[list[PaperInput], dict[str, str]]:
    parsed_keywords = parse_keywords(keywords or DEFAULT_KEYWORDS)
    query = build_keyword_query(parsed_keywords)
    selected = sources or ["PubMed", "Europe PMC", "OpenAlex", "Crossref"]
    per_source = max(5, min(50, limit))
    functions = {
        "PubMed": lambda: search_pubmed(query, per_source, since_days=since_days),
        "Europe PMC": lambda: search_europe_pmc(query, per_source, since_days=since_days),
        "OpenAlex": lambda: search_openalex(query, per_source, email=email, since_days=since_days),
        "Crossref": lambda: search_crossref(query, per_source, since_days=since_days),
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
    merged = mark_paywalled(dedupe(records))
    for record in merged:
        record.keywords = parsed_keywords
    merged.sort(key=lambda item: (item.publication_date or item.year or "", item.is_open_access, bool(item.abstract_en), bool(item.doi)), reverse=True)
    return merged[:limit], errors


def run_keyword_search(
    keywords: str | list[str],
    limit: int = 20,
    email: str = "",
    since_days: int | None = 30,
) -> SearchRun:
    started = utc_now()
    parsed = parse_keywords(keywords)
    records, errors = federated_search(parsed, limit=limit, email=email, since_days=since_days)
    return SearchRun(
        run_id=started.replace(":", "").replace("-", "").split(".")[0],
        keywords=parsed,
        started_at=started,
        finished_at=utc_now(),
        records=records,
        errors=errors,
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
