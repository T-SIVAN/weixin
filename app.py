from __future__ import annotations

import json
import os
import re
from pathlib import Path

import streamlit as st

from weixin_lite.downloader import download_open_access
from weixin_lite.exporter import (
    export_article_html,
    export_article_markdown,
    project_zip,
    unavailable_dois_csv,
)
from weixin_lite.generator import generate_article
from weixin_lite.llm import (
    PROVIDERS,
    default_api_key,
    default_base_url,
    default_model,
    default_provider,
    provider_defaults,
    test_llm_connection,
)
from weixin_lite.models import (
    BatchProject,
    DownloadedPaper,
    PaperInput,
    QuickReadArticle,
    ResolvedKeyword,
    SearchQueryPlan,
    SearchRun,
    generation_ready_papers,
    unavailable_papers,
)
from weixin_lite.pdf_reader import PdfContent, parse_pdf
from weixin_lite.search import (
    DEFAULT_KEYWORDS,
    DEFAULT_JOURNALS_PATH,
    JournalFilter,
    load_journal_filters,
    parse_manual_inputs,
    resolve_doi,
    resolve_keyword_plan,
    run_journal_latest_search,
)
from weixin_lite.translate import DEFAULT_CACHE_PATH, TranslationReport, translate_records
from weixin_lite.wechat_publish import WechatDraftConfig, export_wechat_payload, publish_draft


st.set_page_config(page_title="微信文献快读工具", page_icon="🧬", layout="wide")


LATEST_PATH = Path("data/latest_papers.json")
TRANSLATION_CACHE_PATH = DEFAULT_CACHE_PATH


def init_state() -> None:
    st.session_state.setdefault("papers", [])
    st.session_state.setdefault("pdfs", {})
    st.session_state.setdefault("articles", [])
    st.session_state.setdefault("images", {})
    st.session_state.setdefault("downloads", [])
    st.session_state.setdefault("keywords", ", ".join(DEFAULT_KEYWORDS))
    st.session_state.setdefault("query_plan", None)
    st.session_state.setdefault("search_append", False)
    st.session_state.setdefault("last_translation_report", None)


def paper_key(paper: PaperInput) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", paper.paper_key.lower()).strip("-")[:90]


def merge_papers(existing: list[PaperInput], incoming: list[PaperInput]) -> list[PaperInput]:
    merged = {paper_key(item): item for item in existing if paper_key(item)}
    for item in incoming:
        key = paper_key(item)
        if not key:
            continue
        old = merged.get(key)
        if not old:
            merged[key] = item
            continue
        for field in (
            "title",
            "title_en",
            "title_zh",
            "doi",
            "pmid",
            "journal",
            "year",
            "publication_date",
            "publication_date_source",
            "abstract",
            "abstract_en",
            "abstract_zh",
            "url",
            "oa_pdf_url",
            "pdf_name",
            "oa_source",
            "download_error",
            "article_type",
            "translation_status",
        ):
            if not getattr(old, field) and getattr(item, field):
                setattr(old, field, getattr(item, field))
        if item.journal_priority < old.journal_priority:
            old.journal_priority = item.journal_priority
        if item.is_open_access:
            old.is_open_access = True
            old.access_status = "open"
        if len(item.authors) > len(old.authors):
            old.authors = item.authors
    return list(merged.values())


def load_latest_run() -> SearchRun | None:
    if not LATEST_PATH.exists():
        return None
    try:
        return SearchRun.from_dict(json.loads(LATEST_PATH.read_text(encoding="utf-8")))
    except Exception:
        return None


def save_latest_translation_state(papers: list[PaperInput]) -> None:
    latest = load_latest_run()
    if not latest or not latest.records:
        return
    by_key = {paper_key(paper): paper for paper in papers if paper_key(paper)}
    changed = False
    for record in latest.records:
        source = by_key.get(paper_key(record))
        if not source:
            continue
        for field in ("title_zh", "translation_status"):
            value = getattr(source, field)
            if value and value != getattr(record, field):
                setattr(record, field, value)
                changed = True
    if changed:
        LATEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        LATEST_PATH.write_text(json.dumps(latest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def infer_paper_from_pdf(name: str, pdf: PdfContent) -> PaperInput:
    first = pdf.text[:3000]
    doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", first, flags=re.I)
    title = name.rsplit(".", 1)[0]
    for line in first.splitlines():
        clean = re.sub(r"\s+", " ", line).strip()
        if 30 <= len(clean) <= 220 and not clean.lower().startswith(("abstract", "introduction")):
            title = clean
            break
    return PaperInput(
        title=title,
        title_en=title,
        doi=doi_match.group(0).lower() if doi_match else "",
        pdf_name=name,
        source="PDF upload",
        access_status="open",
        is_open_access=True,
    )


def is_distinct_text(primary: str, secondary: str) -> bool:
    primary_clean = re.sub(r"\s+", " ", primary or "").strip()
    secondary_clean = re.sub(r"\s+", " ", secondary or "").strip()
    return bool(secondary_clean and secondary_clean != primary_clean and secondary_clean not in primary_clean)


def bilingual_title(paper: PaperInput) -> str:
    title_zh = paper.title_zh.strip()
    title_en = (paper.title_en or paper.title).strip()
    if title_zh and is_distinct_text(title_zh, title_en):
        return f"{title_zh}\n{title_en}"
    return title_zh or title_en or paper.doi or paper.pmid or paper.pdf_name


def bilingual_abstract(paper: PaperInput) -> str:
    abstract_zh = paper.abstract_zh.strip()
    abstract_en = (paper.abstract_en or paper.abstract).strip()
    if abstract_zh and is_distinct_text(abstract_zh, abstract_en):
        return f"{abstract_zh}\n\nEnglish abstract:\n{abstract_en}"
    return abstract_zh or abstract_en


def paper_rows(papers: list[PaperInput]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for idx, paper in enumerate(papers, start=1):
        rows.append(
            {
                "#": str(idx),
                "标题": bilingual_title(paper),
                "翻译状态": paper_translation_status(paper),
                "期刊": paper.journal,
                "发表日期": paper.publication_date or paper.year,
                "文章类型": paper.article_type or "未知",
                "期刊优先级": str(paper.journal_priority or ""),
                "DOI": paper.doi,
                "全文状态": paper.access_status,
                "PDF": paper.pdf_name,
                "来源": paper.source,
            }
        )
    return rows


def paper_translation_status(paper: PaperInput) -> str:
    labels = {
        "translated": "已翻译",
        "cached": "缓存命中",
        "failed": "失败",
        "pending": "待翻译",
        "skipped": "跳过",
    }
    if paper.translation_status in labels:
        return labels[paper.translation_status]
    if paper.title_zh.strip() and is_distinct_text(paper.title_zh, paper.title_en or paper.title):
        return "已翻译"
    return "待翻译"


def multiselect_default(key: str, options: list[str]) -> list[str]:
    current = st.session_state.get(key)
    if isinstance(current, tuple):
        current = list(current)
    if isinstance(current, list):
        valid = [item for item in current if item in options]
        if len(valid) != len(current):
            st.session_state[key] = valid or list(options)
            return st.session_state[key]
        return valid
    if current is not None:
        st.session_state[key] = list(options)
    return list(options)


def show_paper_table(papers: list[PaperInput], key: str, *, filters: bool = True) -> None:
    visible = list(papers)
    if filters and papers:
        status_options = sorted({paper_translation_status(paper) for paper in papers})
        source_options = sorted({paper.source for paper in papers if paper.source})
        oa_options = sorted({paper.access_status for paper in papers if paper.access_status})
        status_key = f"{key}_status"
        source_key = f"{key}_source"
        oa_key = f"{key}_oa"
        col_a, col_b, col_c = st.columns(3)
        selected_status = col_a.multiselect(
            "翻译状态",
            status_options,
            default=multiselect_default(status_key, status_options),
            key=status_key,
        )
        selected_sources = col_b.multiselect(
            "来源",
            source_options,
            default=multiselect_default(source_key, source_options),
            key=source_key,
        )
        selected_oa = col_c.multiselect(
            "OA/全文状态",
            oa_options,
            default=multiselect_default(oa_key, oa_options),
            key=oa_key,
        )
        visible = [
            paper
            for paper in papers
            if paper_translation_status(paper) in selected_status
            and (not source_options or paper.source in selected_sources)
            and (not oa_options or paper.access_status in selected_oa)
        ]
    st.dataframe(paper_rows(visible), use_container_width=True, hide_index=True)


def show_translation_report(report: TranslationReport) -> None:
    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    col_a.metric("新翻译", report.translated_count)
    col_b.metric("缓存命中", report.cached_count)
    col_c.metric("待翻译", report.pending_count)
    col_d.metric("失败", report.failed_count)
    col_e.metric("跳过", report.skipped_count)
    if report.pending_count:
        st.info("未配置 API Key 或本次未调用模型，英文标题已保留，可稍后配置模型后重试。")
    if report.errors:
        st.warning("翻译未完全成功：" + "; ".join(dict.fromkeys(report.errors[:5])))
        if any("429" in item or "Too Many Requests" in item for item in report.errors):
            st.info("429 表示模型供应商限流或额度不足。工具已自动退避重试；仍失败时可降低批量、调大间隔或稍后重试失败项。")


def translate_current_papers(
    papers: list[PaperInput],
    *,
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    batch_size: int,
    delay_seconds: float,
    retry_failed_only: bool = False,
) -> TranslationReport:
    progress = st.progress(0)
    status = st.empty()

    def progress_callback(event: dict[str, object]) -> None:
        total = int(event.get("total") or 0)
        completed = int(event.get("completed") or 0)
        if total:
            progress.progress(min(1.0, completed / total))
        report = event.get("report")
        if isinstance(report, TranslationReport):
            status.write(
                f"翻译进度：{completed}/{total}；新翻译 {report.translated_count}，"
                f"缓存 {report.cached_count}，失败 {report.failed_count}，待翻译 {report.pending_count}"
            )

    report = translate_records(
        papers,
        api_key=api_key,
        base_url=base_url,
        model=model,
        provider=provider,
        batch_size=batch_size,
        delay_seconds=delay_seconds,
        cache_path=TRANSLATION_CACHE_PATH,
        progress_callback=progress_callback,
        retry_failed_only=retry_failed_only,
    )
    progress.progress(1.0)
    status.write(
        f"翻译完成：新翻译 {report.translated_count}，缓存 {report.cached_count}，"
        f"失败 {report.failed_count}，待翻译 {report.pending_count}"
    )
    save_latest_translation_state(papers)
    return report


def show_details(papers: list[PaperInput], title: str = "查看摘要和错误") -> None:
    if not papers:
        return
    with st.expander(title):
        for idx, paper in enumerate(papers, start=1):
            st.markdown(f"**{idx}. {paper.title_zh or paper.title_en or paper.title or paper.doi}**")
            title_en = paper.title_en or paper.title
            if is_distinct_text(paper.title_zh, title_en):
                st.caption(title_en)
            abstract = bilingual_abstract(paper)
            if abstract:
                st.write(abstract)
            bits = [f"DOI: {paper.doi}" if paper.doi else "", paper.url, paper.download_error]
            st.caption(" | ".join(bit for bit in bits if bit))


MODE_LABELS = {"strict": "精准", "balanced": "均衡", "broad": "宽松"}
MODE_VALUES = {label: value for value, label in MODE_LABELS.items()}


def plan_to_rows(plan: SearchQueryPlan) -> list[dict[str, str]]:
    return [
        {
            "中文/原词": item.original,
            "英文检索词": ", ".join(item.english_terms),
            "来源": item.source,
            "提示": item.warning,
        }
        for item in plan.keywords
    ]


def rows_to_plan(rows: object, search_mode: str, warnings: list[str] | None = None) -> SearchQueryPlan:
    if hasattr(rows, "to_dict"):
        row_items = rows.to_dict("records")  # type: ignore[call-arg]
    else:
        row_items = rows if isinstance(rows, list) else []
    keywords: list[ResolvedKeyword] = []
    for row in row_items:
        if not isinstance(row, dict):
            continue
        original = str(row.get("中文/原词") or "").strip()
        if not original:
            continue
        english_terms = [
            term.strip()
            for term in re.split(r"[,，;\n]+", str(row.get("英文检索词") or ""))
            if term.strip()
        ]
        source = str(row.get("来源") or "original")
        if source not in {"dictionary", "model", "original", "fallback"}:
            source = "original"
        keywords.append(
            ResolvedKeyword(
                original=original,
                english_terms=english_terms or [original],
                source=source,  # type: ignore[arg-type]
                warning=str(row.get("提示") or ""),
            )
        )
    return SearchQueryPlan(keywords=keywords, search_mode=search_mode, warnings=warnings or [])


def show_search_run_diagnostics(run: SearchRun, *, wrapped: bool = True) -> None:
    def render() -> None:
        if run.query_plan:
            st.caption("实际检索词：" + "；".join(
                f"{item.original} -> {', '.join(item.english_terms or [item.original])}"
                for item in run.query_plan.keywords
            ))
        kept_label = "保留结果" if run.search_kind == "journal_latest" else "相关结果"
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("抓取数", run.raw_count)
        col_b.metric(kept_label, run.filtered_count)
        col_c.metric("来源数", len(run.source_counts))
        if run.source_counts:
            st.dataframe(
                [
                    {
                        "来源": source,
                        "抓取": counts.get("fetched", 0),
                        "去重": counts.get("deduplicated", 0),
                        kept_label: counts.get("relevant", 0),
                    }
                    for source, counts in run.source_counts.items()
                ],
                use_container_width=True,
                hide_index=True,
            )
        for warning in run.warnings:
            st.warning(warning)
        for source, error in run.errors.items():
            st.error(f"{source}: {error}")
    if wrapped:
        with st.expander("查看来源统计和提示"):
            render()
    else:
        render()


def journal_to_rows(journals: list[JournalFilter]) -> list[dict[str, object]]:
    return [
        {
            "启用": journal.enabled,
            "期刊": journal.name,
            "别名": ", ".join(journal.aliases),
            "ISSN": journal.issn,
            "EISSN": journal.eissn,
            "出版集团": journal.publisher_family,
            "优先级": journal.priority,
        }
        for journal in journals
    ]


def rows_to_journals(rows: object) -> list[JournalFilter]:
    if hasattr(rows, "to_dict"):
        row_items = rows.to_dict("records")  # type: ignore[call-arg]
    else:
        row_items = rows if isinstance(rows, list) else []
    journals: list[JournalFilter] = []
    for row in row_items:
        if not isinstance(row, dict):
            continue
        name = str(row.get("期刊") or "").strip()
        if not name:
            continue
        journals.append(
            JournalFilter(
                name=name,
                aliases=[item.strip() for item in str(row.get("别名") or "").split(",") if item.strip()],
                issn=str(row.get("ISSN") or "").strip(),
                eissn=str(row.get("EISSN") or "").strip(),
                publisher_family=str(row.get("出版集团") or "").strip(),
                priority=int(row.get("优先级") or 9999),
                enabled=bool(row.get("启用", True)),
            )
        )
    return journals


def render_article_preview(article: QuickReadArticle) -> None:
    pending: list[str] = []

    def flush() -> None:
        if pending:
            st.markdown("\n".join(pending))
            pending.clear()

    for raw in article.body_markdown.splitlines():
        line = raw.strip()
        if line.startswith("![") and "](" in line and line.endswith(")"):
            flush()
            image_name = line[line.find("(") + 1 : line.rfind(")")].replace("images/", "", 1)
            image_bytes = st.session_state.images.get(image_name)
            if image_bytes:
                st.image(image_bytes, use_container_width=True)
            else:
                st.caption(f"图片未找到：{image_name}")
        else:
            pending.append(raw)
    flush()


def sidebar_settings() -> tuple[str, str, str, str, int, float]:
    st.sidebar.header("翻译/生成模型")
    provider_keys = list(PROVIDERS.keys())
    current_provider = default_provider()
    provider = st.sidebar.selectbox(
        "供应商",
        provider_keys,
        index=provider_keys.index(current_provider) if current_provider in provider_keys else 0,
        format_func=lambda key: PROVIDERS[key].label,
    )
    defaults = provider_defaults(provider)
    api_key = st.sidebar.text_input("API Key", value=default_api_key(), type="password")
    base_url = st.sidebar.text_input("Base URL", value=default_base_url(provider) or defaults.base_url)
    model = st.sidebar.text_input("Model", value=default_model(provider) or defaults.default_model)
    batch_size = st.sidebar.slider("翻译批量", 1, 20, 8)
    delay_seconds = st.sidebar.slider("翻译间隔（秒）", 0.0, 10.0, 1.0, step=0.5)
    st.sidebar.caption("支持 OpenAI-compatible 接口。标题翻译会优先使用本地缓存；遇到 429 会自动退避重试。")
    if st.sidebar.button("测试翻译模型"):
        try:
            raw = test_llm_connection(api_key=api_key, base_url=base_url, model=model)
            st.sidebar.success(f"连接成功：{raw[:80]}")
        except Exception as exc:
            st.sidebar.error(f"连接失败：{type(exc).__name__}: {exc}")
    return provider, api_key, base_url, model, batch_size, delay_seconds


def status_bar() -> None:
    papers: list[PaperInput] = st.session_state.papers
    pdfs: dict[str, PdfContent] = st.session_state.pdfs
    ready = generation_ready_papers(papers, pdfs)
    unavailable = unavailable_papers(papers, pdfs)
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("候选文献", len(papers))
    col_b.metric("已解析全文", len(pdfs))
    col_c.metric("可生成", len(ready))
    col_d.metric("未下载 DOI", len(unavailable))


def search_tab(provider: str, api_key: str, base_url: str, model: str, batch_size: int, delay_seconds: float) -> None:
    st.subheader("每日顶刊最新文章检索")
    latest = load_latest_run()
    if latest and latest.records:
        with st.expander(f"每日历史结果：{latest.finished_at or latest.started_at}"):
            label = "期刊：" if latest.search_kind == "journal_latest" else "关键词："
            st.caption(label + ", ".join(latest.keywords[:12]) + (" ..." if len(latest.keywords) > 12 else ""))
            show_paper_table(latest.records, "latest_history", filters=True)
            show_search_run_diagnostics(latest, wrapped=False)
            if st.button("加入每日结果"):
                st.session_state.papers = merge_papers(st.session_state.papers, latest.records)
                st.success(f"已加入 {len(latest.records)} 条候选。")

    st.divider()
    try:
        default_journals = load_journal_filters(DEFAULT_JOURNALS_PATH)
    except Exception as exc:
        default_journals = []
        st.error(f"期刊配置读取失败：{type(exc).__name__}: {exc}")

    col_a, col_b, col_c, col_d = st.columns([1, 1, 1.3, 1.4])
    limit = col_a.slider("结果数量", 10, 200, 100, step=10)
    since_days = col_b.slider("抓取天数", 1, 30, 7)
    selected_sources = col_c.multiselect(
        "数据源",
        ["PubMed", "Europe PMC", "Crossref", "OpenAlex"],
        default=["PubMed", "Europe PMC", "Crossref"] + (["OpenAlex"] if os.getenv("OPENALEX_API_KEY") else []),
    )
    openalex_api_key = col_d.text_input(
        "OpenAlex API Key",
        value=os.getenv("OPENALEX_API_KEY", ""),
        type="password",
        help="未填写时会跳过 OpenAlex，其他来源照常检索。",
    )
    append_results = st.checkbox("追加到候选", value=bool(st.session_state.search_append))
    st.session_state.search_append = append_results

    edited_journals = st.data_editor(
        journal_to_rows(default_journals),
        use_container_width=True,
        hide_index=True,
        disabled=["期刊", "别名", "ISSN", "EISSN", "出版集团"],
        column_config={
            "启用": st.column_config.CheckboxColumn("启用"),
            "优先级": st.column_config.NumberColumn("优先级", min_value=1, step=1),
        },
    )
    journals = rows_to_journals(edited_journals)
    enabled_count = len([journal for journal in journals if journal.enabled])
    st.caption(f"已启用 {enabled_count} 本期刊；默认抓取最近 {since_days} 天，按期刊优先级和发表日期排序。")

    if st.button("抓取最新文章", type="primary"):
        with st.spinner("正在按期刊检索 PubMed、Europe PMC、OpenAlex、Crossref..."):
            run = run_journal_latest_search(
                journals,
                limit=limit,
                sources=selected_sources,
                since_days=since_days,
                openalex_api_key=openalex_api_key,
            )
        st.session_state.papers = merge_papers(st.session_state.papers, run.records) if append_results else list(run.records)
        if run.errors:
            st.warning("部分检索源失败：" + "; ".join(f"{k}: {v}" for k, v in run.errors.items()))
        if run.warnings:
            st.info("；".join(run.warnings))
        if run.records:
            st.success(f"已抓取 {len(run.records)} 条文章。可继续点击下方按钮翻译标题。")
        else:
            st.info("本次没有符合条件的最新文章。")
        show_paper_table(run.records, "last_search", filters=True)
        show_search_run_diagnostics(run)
        show_details(run.records, "查看本次检索详情")

    papers: list[PaperInput] = st.session_state.papers
    if papers:
        st.divider()
        col_translate, col_retry = st.columns(2)
        if col_translate.button("翻译全部未翻译标题"):
            report = translate_current_papers(
                papers,
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                model=model,
                batch_size=batch_size,
                delay_seconds=delay_seconds,
            )
            st.session_state.last_translation_report = report
        if col_retry.button("重试失败翻译"):
            report = translate_current_papers(
                papers,
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                model=model,
                batch_size=batch_size,
                delay_seconds=delay_seconds,
                retry_failed_only=True,
            )
            st.session_state.last_translation_report = report
        last_report = st.session_state.get("last_translation_report")
        if isinstance(last_report, TranslationReport):
            show_translation_report(last_report)
        show_paper_table(papers, "current_candidates", filters=True)
        show_details(papers, "查看候选摘要和错误")


def ingest_and_generate_tab(api_key: str, base_url: str, model: str) -> None:
    st.subheader("全文与生成")
    papers: list[PaperInput] = st.session_state.papers
    pdfs: dict[str, PdfContent] = st.session_state.pdfs

    if papers:
        st.dataframe(paper_rows(papers), use_container_width=True, hide_index=True)
        if st.button("下载并解析开放全文"):
            progress = st.progress(0)
            downloads: list[DownloadedPaper] = []
            for idx, paper in enumerate(papers, start=1):
                downloaded = download_open_access(paper)
                downloads.append(downloaded)
                if downloaded.status == "open" and downloaded.content_bytes and "pdf" in downloaded.content_type.lower():
                    try:
                        pdf = parse_pdf(downloaded.content_bytes)
                        pdf_name = downloaded.file_name or f"{paper_key(paper)}.pdf"
                        pdfs[pdf_name] = pdf
                        st.session_state.images.update(pdf.rendered_images)
                        paper.pdf_name = pdf_name
                        paper.access_status = "open"
                        paper.download_error = ""
                    except Exception as exc:
                        paper.download_error = f"PDF 解析失败：{exc}"
                        paper.access_status = "download_failed"
                else:
                    paper.access_status = downloaded.status
                    paper.download_error = downloaded.error or "未下载到 PDF 全文。"
                progress.progress(idx / len(papers), text=f"已处理 {idx}/{len(papers)}")
            st.session_state.downloads = downloads
            st.success("开放全文下载和解析完成。未成功解析的论文只会进入 DOI CSV。")
    else:
        st.info("先检索文献、粘贴 DOI，或直接上传 PDF。")

    uploaded = st.file_uploader("上传 PDF", type=["pdf"], accept_multiple_files=True)
    if uploaded and st.button("解析上传 PDF"):
        parsed_papers: list[PaperInput] = []
        progress = st.progress(0)
        for idx, file in enumerate(uploaded, start=1):
            pdf = parse_pdf(file.getvalue())
            pdfs[file.name] = pdf
            st.session_state.images.update(pdf.rendered_images)
            parsed_papers.append(infer_paper_from_pdf(file.name, pdf))
            progress.progress(idx / len(uploaded), text=f"已解析 {idx}/{len(uploaded)}")
        st.session_state.papers = merge_papers(st.session_state.papers, parsed_papers)
        st.success(f"已解析 {len(parsed_papers)} 个 PDF。")

    manual = st.text_area("手动粘贴 DOI / PMID / 标题（每行一篇，可选）", height=90)
    col_d, col_e = st.columns([1, 3])
    if col_d.button("解析手动列表"):
        records = parse_manual_inputs(manual)
        resolved: list[PaperInput] = []
        for record in records:
            if record.doi:
                try:
                    resolved.append(resolve_doi(record.doi) or record)
                except Exception:
                    resolved.append(record)
            else:
                resolved.append(record)
        st.session_state.papers = merge_papers(st.session_state.papers, resolved)
        st.success(f"已加入 {len(resolved)} 条手动文献。")
    if col_e.button("清空当前批次"):
        st.session_state.papers = []
        st.session_state.articles = []
        st.session_state.pdfs = {}
        st.session_state.images = {}
        st.session_state.downloads = []
        st.info("已清空当前批次。")

    st.divider()
    ready = generation_ready_papers(st.session_state.papers, st.session_state.pdfs)
    unavailable = unavailable_papers(st.session_state.papers, st.session_state.pdfs)
    if unavailable:
        st.warning(f"{len(unavailable)} 篇未下载或未解析到全文，已排除生成，只进入 DOI CSV。")
        st.dataframe(paper_rows(unavailable), use_container_width=True, hide_index=True)
    if not ready:
        st.info("暂无可生成论文。需要先下载并解析开放 PDF，或上传 PDF。")
        return

    labels = [f"{idx}. {paper.display_title[:90]}" for idx, paper in enumerate(ready, start=1)]
    selected = st.multiselect("选择生成论文", labels, default=labels[: min(10, len(labels))])
    target_chars = st.slider("目标字数", 700, 1400, 1200, step=50)
    if st.button("生成公众号稿", type="primary"):
        chosen_indexes = [labels.index(label) for label in selected]
        generated: list[QuickReadArticle] = []
        progress = st.progress(0)
        for run_idx, paper_idx in enumerate(chosen_indexes, start=1):
            paper = ready[paper_idx]
            pdf = st.session_state.pdfs[paper.pdf_name]
            generated.append(
                generate_article(
                    paper=paper,
                    pdf=pdf,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    target_chars=target_chars,
                )
            )
            progress.progress(run_idx / len(chosen_indexes), text=f"已生成 {run_idx}/{len(chosen_indexes)}")
        st.session_state.articles = generated
        st.success(f"已生成 {len(generated)} 篇中文公众号稿。")

    for idx, article in enumerate(st.session_state.articles, start=1):
        with st.expander(f"{idx}. {article.title} | {article.word_count} 字", expanded=idx == 1):
            cover = st.file_uploader("上传封面图（可选）", type=["png", "jpg", "jpeg"], key=f"cover-{idx}")
            if cover:
                name = f"cover-{idx}-{cover.name}"
                st.session_state.images[name] = cover.getvalue()
                article.cover_image_name = name
            if article.figures:
                st.caption("原文截图")
                for fig_idx, figure in enumerate(article.figures, start=1):
                    current = figure.image_name
                    if current and current in st.session_state.images:
                        st.image(st.session_state.images[current], caption=f"{figure.figure_id} 自动截图", use_container_width=True)
                    replacement = st.file_uploader(
                        f"替换 {figure.figure_id} 截图",
                        type=["png", "jpg", "jpeg"],
                        key=f"figure-replace-{idx}-{fig_idx}",
                    )
                    if replacement:
                        new_name = f"figure-{idx}-{fig_idx}-{replacement.name}"
                        st.session_state.images[new_name] = replacement.getvalue()
                        if current:
                            article.body_markdown = article.body_markdown.replace(f"images/{current}", f"images/{new_name}")
                            article.body_html = article.body_html.replace(f"images/{current}", f"images/{new_name}")
                        figure.image_name = new_name
                        figure.page_image_name = new_name
            if article.warnings:
                st.warning("；".join(article.warnings))
            render_article_preview(article)
            if article.evidence:
                st.caption("证据追踪")
                st.dataframe([item.to_dict() for item in article.evidence[:12]], use_container_width=True, hide_index=True)
            col_a, col_b = st.columns(2)
            col_a.download_button("Markdown", export_article_markdown(article), file_name=f"{idx:02d}-{article.title}.md")
            col_b.download_button("HTML", export_article_html(article), file_name=f"{idx:02d}-{article.title}.html")


def export_and_publish_tab() -> None:
    st.subheader("导出与发布")
    papers: list[PaperInput] = st.session_state.papers
    articles: list[QuickReadArticle] = st.session_state.articles
    unavailable = unavailable_papers(papers, st.session_state.pdfs)

    if unavailable:
        st.download_button(
            "下载未生成 DOI CSV",
            unavailable_dois_csv(unavailable).encode("utf-8-sig"),
            file_name="unavailable_dois.csv",
            mime="text/csv",
        )
        st.dataframe(paper_rows(unavailable), use_container_width=True, hide_index=True)

    if papers or articles:
        project = BatchProject(
            topic=st.session_state.keywords,
            papers=papers,
            articles=articles,
            downloads=st.session_state.downloads,
        )
        st.download_button(
            "下载项目包",
            project_zip(project, st.session_state.images, st.session_state.downloads),
            file_name="weixin-batch.weixin-project.zip",
            mime="application/zip",
            type="primary",
        )

    if not articles:
        st.info("生成文章后可导出单篇文件或创建公众号草稿。")
        return

    st.divider()
    article_titles = [f"{idx}. {article.title}" for idx, article in enumerate(articles, start=1)]
    selected = st.selectbox("选择文章", article_titles)
    article = articles[article_titles.index(selected)]

    col_a, col_b, col_c = st.columns(3)
    col_a.download_button("单篇 Markdown", export_article_markdown(article), file_name=f"{article.title}.md")
    col_b.download_button("单篇 HTML", export_article_html(article), file_name=f"{article.title}.html")

    st.markdown("#### 公众号草稿")
    app_id = st.text_input("APP_ID")
    app_secret = st.text_input("APP_SECRET", type="password")
    author = st.text_input("作者", value="")
    cover = st.file_uploader("草稿封面图", type=["png", "jpg", "jpeg"], key="draft-cover")
    cover_name = ""
    if cover:
        cover_name = f"draft-cover-{cover.name}"
        st.session_state.images[cover_name] = cover.getvalue()
    show_cover_pic = st.checkbox("正文显示封面", value=False)
    source_url = st.text_input("原文链接", value=article.paper.url)
    need_open_comment = st.checkbox("开启留言", value=False)
    only_fans_can_comment = st.checkbox("仅粉丝可留言", value=False)
    dry_run = st.checkbox("只预览 payload（推荐）", value=True)
    confirmed = st.checkbox("确认真实创建草稿", value=False, disabled=dry_run)

    config = WechatDraftConfig(
        app_id=app_id,
        app_secret=app_secret,
        author=author,
        cover_image_name=cover_name or article.cover_image_name,
        show_cover_pic=show_cover_pic,
        content_source_url=source_url,
        need_open_comment=need_open_comment,
        only_fans_can_comment=only_fans_can_comment,
    )
    col_c.download_button("草稿 Payload", export_wechat_payload(article, config), file_name=f"{article.title}-wechat-payload.json")
    if st.button("创建公众号草稿"):
        if not dry_run and not confirmed:
            st.error("真实创建草稿前请勾选确认。")
            return
        try:
            result = publish_draft(article, config, st.session_state.images, dry_run=dry_run)
            st.success("草稿 dry-run 已生成。" if dry_run else "公众号草稿已创建。")
            st.json(result)
        except Exception as exc:
            st.error(f"草稿发布失败：{type(exc).__name__}: {exc}")


def main() -> None:
    init_state()
    provider, api_key, base_url, model, batch_size, delay_seconds = sidebar_settings()
    st.title("微信文献快读工具")
    st.caption("检索文献、解析开放全文或上传 PDF，再生成中文公众号稿。未下载到全文的论文只导出 DOI CSV。")
    status_bar()

    tab_search, tab_generate, tab_export = st.tabs(["检索与翻译", "全文与生成", "导出与发布"])
    with tab_search:
        search_tab(provider, api_key, base_url, model, batch_size, delay_seconds)
    with tab_generate:
        ingest_and_generate_tab(api_key, base_url, model)
    with tab_export:
        export_and_publish_tab()


if __name__ == "__main__":
    main()
