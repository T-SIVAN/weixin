import json
from datetime import date
import sys
from pathlib import Path

from weixin_lite.models import PaperInput, SearchRun
from weixin_lite.search import (
    JournalFilter,
    build_europe_pmc_journal_query,
    build_pubmed_journal_query,
    filter_records_by_date_range,
    filter_records_by_keywords,
    journal_latest_search,
    load_journal_filters,
    run_journal_latest_search,
    recent_year_months,
    search_crossref_latest,
    search_europe_pmc_latest,
    search_openalex_latest,
    search_pubmed_latest,
    year_month_label,
    year_month_range,
    years_months_to_since_days,
    should_keep_article_type,
    suggest_filter_keywords,
)


def test_year_month_lookback_preserves_calendar_month_boundaries():
    end = date(2026, 3, 31)

    assert years_months_to_since_days(0, 1, end_date=end) == (end - date(2026, 2, 28)).days
    assert years_months_to_since_days(1, 2, end_date=end) == (end - date(2025, 1, 31)).days


def test_explicit_year_month_label_and_range_are_calendar_exact():
    assert year_month_label(2026, 2) == "2026年2月"
    assert year_month_range(2026, 2, today=date(2026, 9, 16)) == ("2026-02-01", "2026-02-28")
    assert year_month_range(2026, 9, today=date(2026, 9, 16)) == ("2026-09-01", "2026-09-16")
    assert recent_year_months(3, today=date(2026, 1, 20)) == [
        ("2026年1月", 2026, 1),
        ("2025年12月", 2025, 12),
        ("2025年11月", 2025, 11),
    ]


def test_month_filter_excludes_cross_month_and_missing_publication_dates():
    records = [
        PaperInput(title_en="February", publication_date="2026-02-17"),
        PaperInput(title_en="March", publication_date="2026-03-01"),
        PaperInput(title_en="Unknown"),
    ]

    filtered = filter_records_by_date_range(records, "2026-02-01", "2026-02-28")

    assert [record.title_en for record in filtered] == ["February"]


def test_load_journal_filters_skips_disabled_and_sorts(tmp_path):
    config = tmp_path / "journals.json"
    config.write_text(
        json.dumps(
            {
                "journals": [
                    {"name": "Late", "priority": 20, "enabled": True},
                    {"name": "Disabled", "priority": 1, "enabled": False},
                    {"name": "Early", "priority": 10, "enabled": True},
                ]
            }
        ),
        encoding="utf-8",
    )

    journals = load_journal_filters(config)

    assert [journal.name for journal in journals] == ["Early", "Late"]


def test_journal_query_builders_include_journal_issn_and_date():
    journal = JournalFilter(
        name="Nature Biotechnology",
        aliases=["Nat Biotechnol"],
        issn="1087-0156",
        eissn="1546-1696",
    )

    pubmed = build_pubmed_journal_query(journal)
    epmc = build_europe_pmc_journal_query(journal, since_days=7)
    epmc_month = build_europe_pmc_journal_query(
        journal,
        date_from="2026-02-01",
        date_to="2026-02-28",
    )

    assert '"Nature Biotechnology"[Journal]' in pubmed
    assert '"1087-0156"[ISSN]' in pubmed
    assert 'JOURNAL:"Nature Biotechnology"' in epmc
    assert 'ISSN:"1546-1696"' in epmc
    assert "FIRST_PDATE" in epmc
    assert "FIRST_PDATE:[2026-02-01 TO 2026-02-28]" in epmc_month


def test_all_latest_sources_receive_exact_month_bounds(monkeypatch):
    journal = JournalFilter(name="Nature", issn="0028-0836")
    captured = {}

    def fake_pubmed(query, limit, since_days=None):
        captured["pubmed"] = (query, since_days)
        return []

    def fake_epmc(query, limit, since_days=None):
        captured["epmc"] = (query, since_days)
        return []

    def fake_json(url, params=None):
        captured[url] = dict(params or {})
        if "crossref" in url:
            return {"message": {"items": []}}
        return {"results": []}

    monkeypatch.setattr("weixin_lite.search.search_pubmed", fake_pubmed)
    monkeypatch.setattr("weixin_lite.search.search_europe_pmc", fake_epmc)
    monkeypatch.setattr("weixin_lite.search.http_json", fake_json)
    search_pubmed_latest([journal], 5, since_days=None, date_from="2026-02-01", date_to="2026-02-28")
    search_europe_pmc_latest([journal], 5, since_days=None, date_from="2026-02-01", date_to="2026-02-28")
    search_openalex_latest(
        [journal],
        5,
        since_days=None,
        api_key="key",
        date_from="2026-02-01",
        date_to="2026-02-28",
    )
    search_crossref_latest([journal], 5, since_days=None, date_from="2026-02-01", date_to="2026-02-28")

    assert '"2026-02-01"[Date - Publication]' in captured["pubmed"][0]
    assert '"2026-02-28"[Date - Publication]' in captured["pubmed"][0]
    assert captured["pubmed"][1] is None
    assert "FIRST_PDATE:[2026-02-01 TO 2026-02-28]" in captured["epmc"][0]
    assert captured["epmc"][1] is None
    assert "from_publication_date:2026-02-01" in captured["https://api.openalex.org/works"]["filter"]
    assert "to_publication_date:2026-02-28" in captured["https://api.openalex.org/works"]["filter"]
    crossref_params = captured["https://api.crossref.org/journals/0028-0836/works"]
    assert crossref_params["filter"] == "from-pub-date:2026-02-01,until-pub-date:2026-02-28"


def test_article_type_filter_keeps_research_and_review_but_drops_noise():
    assert should_keep_article_type("journal-article")
    assert should_keep_article_type("Review")
    assert should_keep_article_type("")
    assert not should_keep_article_type("Editorial")
    assert not should_keep_article_type("Correction")
    assert not should_keep_article_type("News")


def test_journal_latest_search_merges_sources_filters_types_and_sorts(monkeypatch):
    journals = [
        JournalFilter(name="Nature", priority=10),
        JournalFilter(name="Cell", priority=40),
    ]

    def fake_pubmed(journals_arg, limit, since_days=None):
        assert since_days == 7
        return [
            PaperInput(
                title_en="Shared paper",
                doi="10.1000/shared",
                journal="Cell",
                publication_date="2026-08-06",
                source="PubMed",
                article_type="Journal Article",
                journal_priority=40,
            ),
            PaperInput(
                title_en="Editorial item",
                doi="10.1000/editorial",
                journal="Nature",
                publication_date="2026-08-07",
                source="PubMed",
                article_type="Editorial",
                journal_priority=10,
            ),
        ]

    def fake_epmc(journals_arg, limit, since_days=None):
        assert since_days == 7
        return [
            PaperInput(
                title_en="Shared paper",
                doi="10.1000/shared",
                journal="Cell",
                abstract_en="More complete abstract.",
                source="Europe PMC",
                article_type="research-article",
                journal_priority=40,
            ),
            PaperInput(
                title_en="Nature review",
                doi="10.1000/nature-review",
                journal="Nature",
                publication_date="2026-08-05",
                source="Europe PMC",
                article_type="Review",
                journal_priority=10,
            ),
        ]

    monkeypatch.setattr("weixin_lite.search.search_pubmed_latest", fake_pubmed)
    monkeypatch.setattr("weixin_lite.search.search_europe_pmc_latest", fake_epmc)

    records, errors = journal_latest_search(
        journals,
        limit=10,
        sources=["PubMed", "Europe PMC"],
        since_days=7,
    )

    assert errors == {}
    assert [record.doi for record in records] == ["10.1000/nature-review", "10.1000/shared"]
    assert records[1].abstract_en == "More complete abstract."
    assert "Europe PMC" in records[1].source


def test_journal_latest_search_passes_explicit_month_and_rechecks_dates(monkeypatch):
    seen = {}

    def fake_pubmed(journals_arg, limit, since_days=None, *, date_from="", date_to=""):
        seen.update(since_days=since_days, date_from=date_from, date_to=date_to)
        return [
            PaperInput(
                title_en="In month",
                doi="10.1000/in-month",
                journal="Nature",
                publication_date="2026-02-14",
                article_type="Journal Article",
            ),
            PaperInput(
                title_en="Wrong month",
                doi="10.1000/wrong-month",
                journal="Nature",
                publication_date="2026-03-01",
                article_type="Journal Article",
            ),
        ]

    monkeypatch.setattr("weixin_lite.search.search_pubmed_latest", fake_pubmed)

    records, errors = journal_latest_search(
        [JournalFilter(name="Nature", priority=10)],
        limit=10,
        sources=["PubMed"],
        since_days=None,
        date_from="2026-02-01",
        date_to="2026-02-28",
    )

    assert errors == {}
    assert seen == {"since_days": None, "date_from": "2026-02-01", "date_to": "2026-02-28"}
    assert [record.doi for record in records] == ["10.1000/in-month"]


def test_keyword_filtering_is_applied_after_latest_search_results_are_kept():
    papers = [
        PaperInput(
            title_en="Metabolic engineering of microbial cell factories",
            abstract_en="A synthetic biology route for biomanufacturing.",
            doi="10.1000/metabolic",
        ),
        PaperInput(
            title_en="Clinical trial of a cardiac device",
            abstract_en="A patient outcome study.",
            doi="10.1000/clinical",
        ),
    ]

    suggestions = suggest_filter_keywords(papers, limit=8)
    filtered = filter_records_by_keywords(papers, ["代谢工程"])

    assert any("metabolic" in keyword.lower() or keyword == "代谢工程" for keyword in suggestions)
    assert [paper.doi for paper in filtered] == ["10.1000/metabolic"]
    assert filter_records_by_keywords(papers, []) == papers


def test_journal_latest_search_skips_openalex_without_key(monkeypatch):
    def fail_openalex(*args, **kwargs):
        raise AssertionError("OpenAlex should not run without key")

    monkeypatch.setattr("weixin_lite.search.search_openalex_latest", fail_openalex)

    records, errors = journal_latest_search(
        [JournalFilter(name="Nature", priority=10)],
        sources=["OpenAlex"],
        openalex_api_key="",
    )

    assert records == []
    assert errors == {}


def test_run_journal_latest_search_serializes_compatible_metadata():
    run = run_journal_latest_search(
        [JournalFilter(name="Nature", priority=10, enabled=True)],
        sources=[],
        limit=5,
        since_days=None,
        date_from="2026-02-01",
        date_to="2026-02-28",
        period_label="2026年2月",
    )
    round_tripped = SearchRun.from_dict(run.to_dict())

    assert round_tripped.search_kind == "journal_latest"
    assert round_tripped.journal_filters[0]["name"] == "Nature"
    assert round_tripped.period_label == "2026年2月"
    assert (round_tripped.date_from, round_tripped.date_to) == ("2026-02-01", "2026-02-28")


def test_daily_search_defaults_to_journal_latest_with_seven_days(monkeypatch, tmp_path):
    from weixin_lite import daily_search

    seen = {}

    def fake_load(path):
        seen["journals_path"] = str(path)
        return [JournalFilter(name="Nature", priority=10)]

    def fake_run(journals, limit=100, sources=None, since_days=7, openalex_api_key=""):
        seen["since_days"] = since_days
        seen["limit"] = limit
        return SearchRun(
            run_id="test",
            keywords=[journal.name for journal in journals],
            started_at="2026-08-07T00:00:00+00:00",
            finished_at="2026-08-07T00:00:01+00:00",
            search_kind="journal_latest",
            journal_filters=[journal.to_dict() for journal in journals],
        )

    output = tmp_path / "latest.json"
    monkeypatch.setattr(daily_search, "load_journal_filters", fake_load)
    monkeypatch.setattr(daily_search, "run_journal_latest_search", fake_run)
    def fail_translate(*args, **kwargs):
        raise AssertionError("translation should be opt-in")

    monkeypatch.setattr(daily_search, "translate_records", fail_translate)
    monkeypatch.setattr(sys, "argv", ["daily_search", "--output", str(output)])

    daily_search.main()

    assert seen["since_days"] == 7
    assert seen["limit"] == 100
    assert json.loads(output.read_text(encoding="utf-8"))["search_kind"] == "journal_latest"


def test_daily_search_since_days_override(monkeypatch, tmp_path):
    from weixin_lite import daily_search

    seen = {}

    def fake_run(journals, limit=100, sources=None, since_days=7, openalex_api_key=""):
        seen["since_days"] = since_days
        return SearchRun(run_id="test", keywords=[], started_at="start")

    monkeypatch.setattr(daily_search, "load_journal_filters", lambda path: [JournalFilter(name="Nature")])
    monkeypatch.setattr(daily_search, "run_journal_latest_search", fake_run)
    monkeypatch.setattr(daily_search, "translate_records", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["daily_search", "--since-days", "3", "--output", str(tmp_path / "latest.json")])

    daily_search.main()

    assert seen["since_days"] == 3


def test_daily_search_translate_is_explicit_and_uses_cache(monkeypatch, tmp_path):
    from weixin_lite import daily_search

    seen = {}

    def fake_run(journals, limit=100, sources=None, since_days=7, openalex_api_key=""):
        return SearchRun(
            run_id="test",
            keywords=[],
            started_at="start",
            records=[PaperInput(title_en="Needs translation")],
        )

    class FakeReport:
        errors: list[str] = []

    def fake_translate(records, **kwargs):
        seen["records"] = records
        seen["cache_path"] = kwargs.get("cache_path")
        seen["batch_size"] = kwargs.get("batch_size")
        records[0].title_zh = "中文"
        return FakeReport()

    cache_path = tmp_path / "cache.json"
    output = tmp_path / "latest.json"
    monkeypatch.setattr(daily_search, "load_journal_filters", lambda path: [JournalFilter(name="Nature")])
    monkeypatch.setattr(daily_search, "run_journal_latest_search", fake_run)
    monkeypatch.setattr(daily_search, "translate_records", fake_translate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "daily_search",
            "--translate",
            "--translation-cache",
            str(cache_path),
            "--batch-size",
            "6",
            "--output",
            str(output),
        ],
    )

    daily_search.main()

    data = json.loads(output.read_text(encoding="utf-8"))
    assert seen["cache_path"] == str(cache_path)
    assert seen["batch_size"] == 6
    assert data["records"][0]["title_zh"] == "中文"
