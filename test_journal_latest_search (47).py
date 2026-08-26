import json
import sys
from pathlib import Path

from weixin_lite.models import PaperInput, SearchRun
from weixin_lite.search import (
    JournalFilter,
    build_europe_pmc_journal_query,
    build_pubmed_journal_query,
    journal_latest_search,
    load_journal_filters,
    run_journal_latest_search,
    should_keep_article_type,
)


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

    assert '"Nature Biotechnology"[Journal]' in pubmed
    assert '"1087-0156"[ISSN]' in pubmed
    assert 'JOURNAL:"Nature Biotechnology"' in epmc
    assert 'ISSN:"1546-1696"' in epmc
    assert "FIRST_PDATE" in epmc


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
    run = run_journal_latest_search([JournalFilter(name="Nature", priority=10, enabled=True)], sources=[], limit=5)
    round_tripped = SearchRun.from_dict(run.to_dict())

    assert round_tripped.search_kind == "journal_latest"
    assert round_tripped.journal_filters[0]["name"] == "Nature"


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
