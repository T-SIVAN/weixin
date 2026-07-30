import json
import xml.etree.ElementTree as ET

from weixin_lite.models import PaperInput, ResolvedKeyword, SearchQueryPlan
from weixin_lite.search import (
    build_europe_pmc_query,
    build_pubmed_query,
    federated_search,
    is_synthetic_biology_relevant,
    pubmed_publication_date,
    resolve_keyword_plan,
    search_pubmed,
)


def test_resolve_keyword_plan_uses_dictionary_for_chinese_synbio_terms():
    plan = resolve_keyword_plan(["精准发酵", "底盘细胞"], search_mode="balanced")

    assert plan.search_mode == "balanced"
    assert plan.keywords[0].source == "dictionary"
    assert "precision fermentation" in plan.keywords[0].english_terms
    assert "chassis cell" in plan.keywords[1].english_terms


def test_resolve_keyword_plan_uses_model_for_unknown_chinese_term():
    plan = resolve_keyword_plan(
        ["新型生物制造"],
        resolver=lambda keyword: ["novel biomanufacturing", "synthetic biology production"],
    )

    assert plan.keywords[0].source == "model"
    assert plan.keywords[0].english_terms == ["novel biomanufacturing", "synthetic biology production"]


def test_resolve_keyword_plan_falls_back_when_model_fails():
    def broken_resolver(keyword):
        raise RuntimeError("offline")

    plan = resolve_keyword_plan(["新型生物制造"], resolver=broken_resolver)

    assert plan.keywords[0].source == "fallback"
    assert plan.keywords[0].english_terms == ["新型生物制造"]
    assert plan.warnings


def test_pubmed_article_date_beats_future_volume_pubdate():
    article = ET.fromstring(
        """
        <PubmedArticle>
          <MedlineCitation>
            <Article>
              <ArticleDate DateType="Electronic">
                <Year>2026</Year><Month>07</Month><Day>23</Day>
              </ArticleDate>
              <Journal>
                <JournalIssue>
                  <PubDate><Year>2027</Year><Month>Jan</Month></PubDate>
                </JournalIssue>
              </Journal>
            </Article>
          </MedlineCitation>
        </PubmedArticle>
        """
    )

    assert pubmed_publication_date(article) == ("2026-07-23", "ArticleDate Electronic")


def test_pubmed_search_keeps_inline_markup_title_and_abstract(monkeypatch):
    ids = json.dumps({"esearchresult": {"idlist": ["1"]}})
    xml = """
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>1</PMID>
          <Article>
            <ArticleTitle>Characterization of <i>Escherichia coli</i> cell factory</ArticleTitle>
            <Abstract><AbstractText>Synthetic <i>biology</i> enables production.</AbstractText></Abstract>
            <ArticleDate DateType="Electronic"><Year>2026</Year><Month>07</Month><Day>23</Day></ArticleDate>
            <Journal><Title>Test Journal</Title><JournalIssue><PubDate><Year>2027</Year></PubDate></JournalIssue></Journal>
          </Article>
        </MedlineCitation>
        <PubmedData><ArticleIdList><ArticleId IdType="doi">10.1000/test</ArticleId></ArticleIdList></PubmedData>
      </PubmedArticle>
    </PubmedArticleSet>
    """

    calls: list[str] = []

    def fake_http_text(url, params, timeout=30):
        calls.append(url)
        return ids if "esearch" in url else xml

    monkeypatch.setattr("weixin_lite.search.http_text", fake_http_text)

    records = search_pubmed("cell factory", 5, since_days=365)

    assert records[0].title_en == "Characterization of Escherichia coli cell factory"
    assert records[0].abstract_en == "Synthetic biology enables production."
    assert records[0].publication_date == "2026-07-23"
    assert records[0].year == "2026"
    assert len(calls) == 2


def test_source_queries_use_title_abstract_fields():
    plan = SearchQueryPlan([ResolvedKeyword("精准发酵", ["precision fermentation"])])

    assert '"precision fermentation"[Title/Abstract]' in build_pubmed_query(plan)
    assert 'TITLE:"precision fermentation"' in build_europe_pmc_query(plan)
    assert "FIRST_PDATE" in build_europe_pmc_query(plan, since_days=365)


def test_federated_search_skips_openalex_without_api_key(monkeypatch):
    called_openalex = False

    def fake_openalex(*args, **kwargs):
        nonlocal called_openalex
        called_openalex = True
        return []

    monkeypatch.setattr("weixin_lite.search.search_openalex", fake_openalex)
    records, errors = federated_search(["精准发酵"], sources=["OpenAlex"], limit=5, openalex_api_key="")

    assert records == []
    assert errors == {}
    assert called_openalex is False


def test_federated_search_oversamples_and_counts_sources(monkeypatch):
    seen: dict[str, tuple[str, int]] = {}

    def make_record(source: str) -> PaperInput:
        return PaperInput(
            title_en="Precision fermentation cell factory for synthetic biology production",
            abstract_en="Metabolic engineering improves biomanufacturing yield.",
            doi=f"10.1000/{source.lower().replace(' ', '-')}",
            source=source,
        )

    def fake_pubmed(query, limit, since_days=None):
        seen["PubMed"] = (query, limit)
        return [make_record("PubMed")]

    def fake_epmc(query, limit, since_days=None):
        seen["Europe PMC"] = (query, limit)
        return [make_record("Europe PMC")]

    monkeypatch.setattr("weixin_lite.search.search_pubmed", fake_pubmed)
    monkeypatch.setattr("weixin_lite.search.search_europe_pmc", fake_epmc)

    records, errors = federated_search(["精准发酵"], sources=["PubMed", "Europe PMC"], limit=5)

    assert errors == {}
    assert len(records) == 2
    assert seen["PubMed"][1] == 15
    assert seen["Europe PMC"][1] == 15
    assert "[Title/Abstract]" in seen["PubMed"][0]
    assert "TITLE:" in seen["Europe PMC"][0]


def test_relevance_modes_allow_strong_title_without_abstract():
    strong_title = PaperInput(title_en="Precision fermentation platform for synthetic biology cell factory")
    weak_clinical = PaperInput(
        title_en="Engineered microbes in patient infection diagnosis",
        abstract_en="Clinical therapy outcomes and prognosis.",
    )
    plan = SearchQueryPlan([ResolvedKeyword("精准发酵", ["precision fermentation"])], search_mode="strict")

    assert is_synthetic_biology_relevant(strong_title, plan, "strict")
    assert not is_synthetic_biology_relevant(weak_clinical, ["engineered microbes"], "strict")
    assert is_synthetic_biology_relevant(strong_title, plan, "broad")
