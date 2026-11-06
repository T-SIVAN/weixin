from __future__ import annotations

import hashlib
import json
import os
import re
from copy import deepcopy
from datetime import date
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

from weixin_lite.downloader import download_open_access
from weixin_lite.article_analysis import ANALYSIS_PROMPT_VERSION, analyze_paper
from weixin_lite.docx_exporter import DocxExportError
from weixin_lite.exporter import (
    export_article_docx_bytes,
    export_article_html,
    export_article_markdown,
    project_zip,
    unavailable_dois_csv,
)
from weixin_lite.figure_analysis import (
    FIGURE_ANALYSIS_PROMPT_VERSION,
    analyze_selected_figures,
    SELECTED_REVIEW_VERSION,
)
from weixin_lite.workspace_ui import checkpoint, initialize_workspace, render_workspace_sidebar
from weixin_lite.workspace_store import preserve_article
from weixin_lite.editor import article_blocks, block_editor_parts, replace_block, rewrite_block, replace_figure_note
from weixin_lite.evidence_context import figure_context, prose_from_note
from weixin_lite.generator import ArticleGenerationError, chineseish_len, generate_article, markdown_to_wechat_html
from weixin_lite.llm import (
    PROVIDERS,
    default_api_key,
    default_base_url,
    default_model,
    default_provider,
    friendly_llm_error,
    provider_defaults,
    test_llm_connection,
)
from weixin_lite.models import (
    BatchProject,
    DownloadedPaper,
    FigureAnalysis,
    PaperInput,
    QuickReadArticle,
    SearchRun,
    generation_ready_papers,
    unavailable_papers,
)
from weixin_lite.pdf_reader import PdfContent, extract_first_page_metadata, parse_pdf
from weixin_lite.search import (
    DEFAULT_JOURNALS_PATH,
    JournalFilter,
    load_journal_filters,
    filter_records_by_keywords,
    suggest_filter_keywords,
    parse_manual_inputs,
    resolve_doi,
    run_journal_latest_search,
)
from weixin_lite.translate import translate_records
from weixin_lite.wechat_publish import WechatDraftConfig, export_wechat_payload, publish_draft


st.set_page_config(page_title="微信文献快读工具", layout="wide")


LATEST_PATH = Path("data/latest_papers.json")


def date_range_label(date_from: date, date_to: date) -> str:
    return (
        f"{date_from.year}年{date_from.month}月{date_from.day}日"
        f" 至 {date_to.year}年{date_to.month}月{date_to.day}日"
    )


def crop_image_bytes(data: bytes, horizontal: tuple[int, int], vertical: tuple[int, int]) -> bytes:
    with Image.open(BytesIO(data)) as image:
        width, height = image.size
        left = round(width * horizontal[0] / 100)
        right = round(width * horizontal[1] / 100)
        top = round(height * vertical[0] / 100)
        bottom = round(height * vertical[1] / 100)
        cropped = image.crop((left, top, max(left + 1, right), max(top + 1, bottom)))
        output = BytesIO()
        cropped.save(output, format="PNG")
        return output.getvalue()


def invalidate_asset_review(figure, message: str) -> None:
    figure.vision_status = "pending"
    figure.review_version = ""
    figure.vision_error = message
    figure.visual_evidence = ""
    figure.interpretation = ""
    figure.review_key = ""
    figure.source_excerpt = ""


def pdf_assets(pdf: PdfContent) -> list:
    """Support Streamlit sessions restored from PdfContent versions before assets."""
    legends = getattr(pdf, "legends", None)
    if legends is not None:
        return list(legends)
    return list(getattr(pdf, "assets", []) or [])


def move_selected_asset(assets, figure_id, direction):
    selected = sorted([item for item in assets if item.selected], key=lambda item: (item.order or 999, item.figure_id))
    position = next(i for i, item in enumerate(selected) if item.figure_id == figure_id)
    target = position + direction
    if 0 <= target < len(selected):
        selected[position], selected[target] = selected[target], selected[position]
    for index, item in enumerate(selected, 1):
        item.order = index
    checkpoint()


def build_project_zip_download(
    project: BatchProject,
    image_assets: dict[str, bytes],
    downloads: list[DownloadedPaper],
) -> tuple[bytes | None, str | None]:
    """Build a project archive without letting a stale image break the page."""
    try:
        return project_zip(project, image_assets, downloads), None
    except (DocxExportError, ValueError) as exc:
        return None, f"项目包暂不能导出：{exc}。请返回“内容与生成”补齐图片后重试。"


def init_state() -> None:
    st.session_state.setdefault("papers", [])
    st.session_state.setdefault("pdfs", {})
    st.session_state.setdefault("articles", [])
    st.session_state.setdefault("images", {})
    st.session_state.setdefault("downloads", [])
    st.session_state.setdefault("search_append", False)
    st.session_state.setdefault("single_analysis", None)
    st.session_state.setdefault("paper_analyses", {})
    st.session_state.setdefault("selected_download_keys", [])
    st.session_state.setdefault("analysis_cache", {})
    st.session_state.setdefault("vision_cache", {})
    st.session_state.setdefault("article_versions", [])
    st.session_state.setdefault("article_cache", {})
    st.session_state.setdefault("editor_drafts", {})


def paper_key(paper: PaperInput) -> str:
    return hashlib.sha256(paper.paper_key.strip().lower().encode("utf-8")).hexdigest()[:24]


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
        ):
            if not getattr(old, field) and getattr(item, field):
                setattr(old, field, getattr(item, field))
        if item.is_open_access:
            old.is_open_access = True
            old.access_status = "open"
        if item.pdf_name:
            old.pdf_name = item.pdf_name
        if len(item.authors) > len(old.authors):
            old.authors = item.authors
        if len(item.affiliations) > len(old.affiliations):
            old.affiliations = item.affiliations
    return list(merged.values())


def load_latest_run() -> SearchRun | None:
    if not LATEST_PATH.exists():
        return None
    try:
        return SearchRun.from_dict(json.loads(LATEST_PATH.read_text(encoding="utf-8")))
    except Exception:
        return None


def infer_paper_from_pdf(name: str, pdf: PdfContent) -> PaperInput:
    first = pdf.text[:3000]
    doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", first, flags=re.I)
    title = name.rsplit(".", 1)[0]
    for line in first.splitlines():
        clean = re.sub(r"\s+", " ", line).strip()
        if 30 <= len(clean) <= 220 and not clean.lower().startswith(("abstract", "introduction")):
            title = clean
            break
    authors, affiliations = extract_first_page_metadata(pdf.text)
    return PaperInput(
        title=title,
        title_en=title,
        authors=authors,
        affiliations=affiliations,
        doi=doi_match.group(0).lower() if doi_match else "",
        pdf_name=name,
        source="PDF upload",
        access_status="open",
        is_open_access=True,
    )


def enrich_paper_from_pdf(paper: PaperInput, pdf: PdfContent | None) -> PaperInput:
    if not pdf:
        return paper
    authors, affiliations = extract_first_page_metadata(pdf.text)
    if not paper.authors and authors:
        paper.authors = authors
    if not paper.affiliations and affiliations:
        paper.affiliations = affiliations
    return paper


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
                "期刊": paper.journal,
                "发表日期": paper.publication_date or paper.year,
                "DOI": paper.doi,
                "全文状态": paper.access_status,
                "PDF": paper.pdf_name,
                "来源": paper.source,
            }
        )
    return rows


def paper_download_rows(papers: list[PaperInput], selected_keys: set[str] | None = None) -> list[dict[str, object]]:
    """Build editable selection rows without changing the paper records."""
    selected = selected_keys or set()
    rows: list[dict[str, object]] = []
    for paper, row in zip(papers, paper_rows(papers)):
        rows.append({"下载解析": paper_key(paper) in selected, **row})
    return rows


def selected_papers_from_rows(papers: list[PaperInput], rows: object) -> list[PaperInput]:
    """Return only papers checked in a Streamlit data editor result."""
    if hasattr(rows, "to_dict"):
        records = rows.to_dict("records")
    elif isinstance(rows, list):
        records = rows
    else:
        records = []
    return [
        paper
        for paper, row in zip(papers, records)
        if isinstance(row, dict) and bool(row.get("下载解析"))
    ]


def merge_download_records(
    existing: list[DownloadedPaper],
    incoming: list[DownloadedPaper],
) -> list[DownloadedPaper]:
    """Replace only reprocessed papers while retaining earlier downloads."""
    merged = {item.paper_key: item for item in existing}
    for item in incoming:
        merged[item.paper_key] = item
    return list(merged.values())


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




def show_search_run_diagnostics(run: SearchRun, *, wrapped: bool = True) -> None:
    def render() -> None:
        if run.query_plan:
            st.caption("实际检索词：" + "；".join(
                f"{item.original} -> {', '.join(item.english_terms or [item.original])}"
                for item in run.query_plan.keywords
            ))
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("抓取数", run.raw_count)
        col_b.metric("相关结果", run.filtered_count)
        col_c.metric("来源数", len(run.source_counts))
        if run.source_counts:
            st.dataframe(
                [
                    {
                        "来源": source,
                        "抓取": counts.get("fetched", 0),
                        "去重": counts.get("deduplicated", 0),
                        "相关": counts.get("relevant", 0),
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


def journal_to_rows(
    journals: list[JournalFilter],
    *,
    default_enabled: bool | None = None,
) -> list[dict[str, object]]:
    return [
        {
            "启用": journal.enabled if default_enabled is None else default_enabled,
            "期刊": journal.name,
            "中文简述": journal.description_zh,
            "影响因子": journal.impact_factor,
            "JIF年度": journal.impact_factor_year,
            "别名": ", ".join(journal.aliases),
            "ISSN": journal.issn,
            "EISSN": journal.eissn,
            "出版集团": journal.publisher_family,
        }
        for journal in sorted(
            journals,
            key=lambda item: (-item.impact_factor, item.priority, item.name.lower()),
        )
    ]


def rows_to_journals(rows: object) -> list[JournalFilter]:
    if hasattr(rows, "to_dict"):
        row_items = rows.to_dict("records")  # type: ignore[call-arg]
    else:
        row_items = rows if isinstance(rows, list) else []
    journals: list[JournalFilter] = []
    for index, row in enumerate(row_items, start=1):
        if not isinstance(row, dict):
            continue
        name = str(row.get("期刊") or "").strip()
        if not name:
            continue
        journals.append(
            JournalFilter(
                name=name,
                description_zh=str(row.get("中文简述") or "").strip(),
                aliases=[item.strip() for item in str(row.get("别名") or "").split(",") if item.strip()],
                issn=str(row.get("ISSN") or "").strip(),
                eissn=str(row.get("EISSN") or "").strip(),
                publisher_family=str(row.get("出版集团") or "").strip(),
                priority=index,
                impact_factor=float(row.get("影响因子") or 0),
                impact_factor_year=int(row.get("JIF年度") or 0),
                enabled=bool(row.get("启用", False)),
            )
        )
    return journals


def render_article_preview(article: QuickReadArticle) -> None:
    try:
        content = export_article_html(article, st.session_state.images).decode("utf-8")
        st.iframe(content, height=720)
    except ValueError as exc:
        st.error(f"预览暂不可用：{exc}")


def sidebar_settings() -> tuple[str, str, str, str, int, float]:
    st.sidebar.subheader("模型设置")
    provider_keys = list(PROVIDERS.keys())
    current_provider = default_provider()
    provider = st.sidebar.selectbox(
        "供应商",
        provider_keys,
        index=provider_keys.index(current_provider) if current_provider in provider_keys else 0,
        format_func=lambda key: PROVIDERS[key].label,
    )
    defaults = provider_defaults(provider)
    api_key = st.sidebar.text_input("API Key", value=default_api_key(provider), type="password", key=f"api-key-{provider}")
    model = st.sidebar.text_input("模型", value=default_model(provider) or defaults.default_model, key=f"model-{provider}")
    with st.sidebar.expander("高级连接与翻译设置"):
        base_url = st.text_input("Base URL", value=default_base_url(provider) or defaults.base_url, key=f"base-url-{provider}")
        batch_size = st.number_input("每批翻译标题数", 1, 8, 5)
        delay_seconds = st.number_input("批次间隔（秒）", 0.0, 60.0, 2.0, step=0.5)
        if st.button("测试连接", disabled=not api_key.strip()):
            try:
                test_llm_connection(api_key=api_key, base_url=base_url, model=model)
                st.success("连接成功")
            except Exception as exc:
                st.error(friendly_llm_error(exc))
    vision = {"provider": provider, "api_key": api_key, "base_url": base_url, "model": model}
    if provider != "gemini":
        with st.sidebar.expander("图表复核 · Gemini"):
            vision = {
                "provider": "gemini",
                "api_key": st.text_input("Gemini API Key", value=default_api_key("gemini"), type="password", key="vision-api-key"),
                "model": st.text_input("视觉模型", value=PROVIDERS["gemini"].default_model, key="vision-model"),
                "base_url": st.text_input("Gemini Base URL", value=PROVIDERS["gemini"].base_url, key="vision-base-url"),
            }
    st.session_state.vision_config = vision
    st.sidebar.caption("API Key 已配置" if api_key.strip() else "尚未配置 API Key")
    return provider, api_key, base_url, model, batch_size, delay_seconds




def search_tab(provider: str, api_key: str, base_url: str, model: str, batch_size: int, delay_seconds: float) -> None:
    st.subheader("文献检索")
    latest = load_latest_run()
    if latest and latest.records:
        with st.expander(f"每日历史结果：{latest.finished_at or latest.started_at}"):
            label = "期刊：" if latest.search_kind == "journal_latest" else "关键词："
            st.caption(label + ", ".join(latest.keywords[:12]) + (" ..." if len(latest.keywords) > 12 else ""))
            if latest.period_label:
                st.caption(
                    f"检索日期：{latest.period_label}"
                    + (f"（{latest.date_from} 至 {latest.date_to}）" if latest.date_from and latest.date_to else "")
                )
            st.dataframe(paper_rows(latest.records), use_container_width=True, hide_index=True)
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

    col_a, col_b, col_c, col_d, col_e = st.columns([0.7, 0.95, 0.95, 1.4, 1.4])
    limit = col_a.slider("结果数量", 10, 200, 100, step=10)
    today = date.today()
    earliest = date(1900, 1, 1)
    selected_date_from = col_b.date_input(
        "开始日期",
        value=today.replace(day=1),
        min_value=earliest,
        max_value=today,
        format="YYYY/MM/DD",
    )
    selected_date_to = col_c.date_input(
        "结束日期",
        value=today,
        min_value=earliest,
        max_value=today,
        format="YYYY/MM/DD",
    )
    invalid_date_range = selected_date_from > selected_date_to
    date_from = selected_date_from.isoformat()
    date_to = selected_date_to.isoformat()
    selected_period_label = date_range_label(selected_date_from, selected_date_to)
    selected_sources = col_d.multiselect(
        "数据源",
        ["PubMed", "Europe PMC", "Crossref", "OpenAlex"],
        default=["PubMed", "Europe PMC", "Crossref"] + (["OpenAlex"] if os.getenv("OPENALEX_API_KEY") else []),
    )
    openalex_api_key = col_e.text_input(
        "OpenAlex API Key",
        value=os.getenv("OPENALEX_API_KEY", ""),
        type="password",
        help="未填写时会跳过 OpenAlex，其他来源照常检索。",
    )
    append_results = st.checkbox("追加到候选", value=bool(st.session_state.search_append))
    st.session_state.search_append = append_results

    edited_journals = st.data_editor(
        journal_to_rows(default_journals, default_enabled=False),
        use_container_width=True,
        hide_index=True,
        column_order=["启用", "期刊", "中文简述", "影响因子", "JIF年度"],
        disabled=["期刊", "中文简述", "影响因子", "JIF年度", "别名", "ISSN", "EISSN", "出版集团"],
        column_config={
            "启用": st.column_config.CheckboxColumn("启用"),
            "中文简述": st.column_config.TextColumn("中文简述", width="large"),
            "影响因子": st.column_config.NumberColumn("影响因子", format="%.1f"),
            "JIF年度": st.column_config.NumberColumn("JIF年度", format="%d"),
        },
        key="journal_picker",
    )
    journals = rows_to_journals(edited_journals)
    enabled_count = len([journal for journal in journals if journal.enabled])
    st.caption(
        f"已启用 {enabled_count} 本期刊；检索日期：{selected_period_label}；"
        "期刊按 2024 JIF 从高到低排列，进入页面默认全部未选。"
    )

    if invalid_date_range:
        st.warning("结束日期不能早于开始日期。")
    if not enabled_count:
        st.info("请至少勾选一本期刊后再开始检索。")
    if st.button(
        "检索文章",
        type="primary",
        disabled=invalid_date_range or not enabled_count or not selected_sources,
    ):
        with st.spinner("正在检索所选日期范围内的文章..."):
            run = run_journal_latest_search(
                journals,
                limit=limit,
                sources=selected_sources,
                since_days=None,
                openalex_api_key=openalex_api_key,
                date_from=date_from,
                date_to=date_to,
                period_label=selected_period_label,
            )
        st.session_state.papers = merge_papers(st.session_state.papers, run.records) if append_results else list(run.records)
        if run.errors:
            st.warning("部分检索源失败：" + "; ".join(f"{k}: {v}" for k, v in run.errors.items()))
        if run.warnings:
            st.info("；".join(run.warnings))
        st.success(f"检索完成：{len(run.records)} 篇。")
        st.session_state.last_search_run = run

    papers: list[PaperInput] = st.session_state.papers
    if papers:
        st.divider()
        filters = st.multiselect("关键词筛选", suggest_filter_keywords(papers), key="result-keywords")
        custom_filter = st.text_input("补充关键词", placeholder="输入关键词，多个词用逗号分隔")
        visible = filter_records_by_keywords(papers, filters + [item.strip() for item in custom_filter.split(",") if item.strip()])
        st.caption(f"显示 {len(visible)} / {len(papers)} 篇")
        st.dataframe(paper_rows(visible), use_container_width=True, hide_index=True)
        if st.button("翻译筛选结果的标题", disabled=not visible or not api_key.strip()):
            with st.spinner("正在翻译标题..."):
                report = translate_records(visible, api_key=api_key, base_url=base_url, model=model,
                                           provider=provider, batch_size=batch_size, delay_seconds=delay_seconds)
            st.session_state.translation_notice = f"翻译 {report.translated_count} 条，缓存 {report.cached_count} 条，失败 {report.failed_count} 条。"
            if report.errors:
                st.session_state.translation_notice += " " + report.errors[0][:300]
            st.rerun()
        if st.session_state.get("translation_notice"):
            st.info(st.session_state.translation_notice)
        show_details(visible, "摘要与全文状态")
        if st.session_state.get("last_search_run"):
            show_search_run_diagnostics(st.session_state.last_search_run)


def render_paper_inputs() -> None:
    papers: list[PaperInput] = st.session_state.papers
    pdfs: dict[str, PdfContent] = st.session_state.pdfs

    if papers:
        paper_keys = [paper_key(paper) for paper in papers]
        selected_keys = set(st.session_state.selected_download_keys) & set(paper_keys)
        selector_version = hashlib.sha256("|".join(paper_keys).encode("utf-8")).hexdigest()[:12]
        st.markdown("##### 选择需要下载并解析的文章")
        st.caption("只处理勾选的文章；未勾选的检索结果仍保留在列表中，不会下载全文。")
        edited_rows = st.data_editor(
            paper_download_rows(papers, selected_keys),
            use_container_width=True,
            hide_index=True,
            disabled=["#", "标题", "期刊", "发表日期", "DOI", "全文状态", "PDF", "来源"],
            column_config={
                "下载解析": st.column_config.CheckboxColumn(
                    "下载解析",
                    help="仅勾选需要下载 PDF、截取首页与关键图表的文章。",
                    default=False,
                )
            },
            key=f"download-paper-selector-{selector_version}",
        )
        selected_papers = selected_papers_from_rows(papers, edited_rows)
        st.session_state.selected_download_keys = [paper_key(paper) for paper in selected_papers]
        st.caption(f"已选择 {len(selected_papers)} / {len(papers)} 篇。")
        if st.button("下载并解析已选全文", disabled=not selected_papers):
            progress = st.progress(0)
            downloads: list[DownloadedPaper] = []
            for idx, paper in enumerate(selected_papers, start=1):
                downloaded = download_open_access(paper)
                downloads.append(downloaded)
                if downloaded.status == "open" and downloaded.content_bytes and "pdf" in downloaded.content_type.lower():
                    try:
                        pdf = parse_pdf(downloaded.content_bytes)
                        pdf_name = downloaded.file_name or f"{paper_key(paper)}.pdf"
                        pdfs[pdf_name] = pdf
                        st.session_state.images.update(pdf.rendered_images)
                        enrich_paper_from_pdf(paper, pdf)
                        paper.pdf_name = pdf_name
                        paper.access_status = "open"
                        paper.download_error = ""
                    except Exception as exc:
                        paper.download_error = f"PDF 解析失败：{exc}"
                        paper.access_status = "download_failed"
                else:
                    paper.access_status = downloaded.status
                    paper.download_error = downloaded.error or "未下载到 PDF 全文。"
                progress.progress(idx / len(selected_papers), text=f"已处理 {idx}/{len(selected_papers)}")
            st.session_state.downloads = merge_download_records(st.session_state.downloads, downloads)
            st.success(f"已完成 {len(selected_papers)} 篇已选文章的下载与解析。未成功解析的论文只会进入 DOI CSV。")
    else:
        st.info("先检索文献、粘贴 DOI，或直接上传 PDF。")

    uploaded = st.file_uploader("上传 PDF", type=["pdf"], accept_multiple_files=True)
    if uploaded and st.button("解析上传 PDF"):
        parsed_papers: list[PaperInput] = []
        progress = st.progress(0)
        for idx, file in enumerate(uploaded, start=1):
            content = file.getvalue()
            storage_name = hashlib.sha256(content).hexdigest()[:12] + "-" + file.name
            try:
                pdf = pdfs.get(storage_name) or parse_pdf(content)
                pdfs[storage_name] = pdf
                st.session_state.images.update(pdf.rendered_images)
                record = infer_paper_from_pdf(file.name, pdf)
                record.pdf_name = storage_name
                parsed_papers.append(record)
                st.session_state.downloads = merge_download_records(st.session_state.downloads, [
                    DownloadedPaper(paper_key=record.paper_key, file_name=storage_name,
                                    content_type="application/pdf", status="open", content_bytes=content)
                ])
            except Exception as exc:
                st.error(f"{file.name} 解析失败：{type(exc).__name__}，请确认文件是有效且未加密的 PDF。")
            progress.progress(idx / len(uploaded), text=f"已解析 {idx}/{len(uploaded)}")
        st.session_state.papers = merge_papers(st.session_state.papers, parsed_papers)
        st.success(f"已解析 {len(parsed_papers)} 个 PDF。")

    manual = st.text_area("手动粘贴 DOI / PMID / 标题（每行一篇，可选）", height=90)
    col_d, col_e = st.columns([1, 3])
    if col_d.button("解析手动列表", disabled=not manual.strip()):
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
        st.session_state.selected_download_keys = []
        st.session_state.paper_analyses = {}
        st.session_state.single_analysis = None
        st.session_state.analysis_cache = {}
        st.session_state.vision_cache = {}
        for key in list(st.session_state):
            if key.startswith(("asset-", "draft-", "editor-", "download-paper-selector-")):
                del st.session_state[key]
        st.info("已清空当前批次。")


def ingest_and_generate_tab(provider: str, api_key: str, base_url: str, model: str) -> None:
    st.subheader("论文分析")
    with st.expander("导入论文", expanded=not st.session_state.pdfs):
        render_paper_inputs()
    st.divider()
    ready = generation_ready_papers(st.session_state.papers, st.session_state.pdfs)
    if not ready:
        st.info("暂无可生成论文。需要先下载并解析开放 PDF，或上传 PDF。")
        return
    labels = [f"{idx}. {paper.display_title[:90]}" for idx, paper in enumerate(ready, start=1)]
    selected_label = st.selectbox("活动文章", labels)
    paper = ready[labels.index(selected_label)]
    pdf = st.session_state.pdfs[paper.pdf_name]
    active_key = paper_key(paper)
    analyses = st.session_state.paper_analyses
    analysis = analyses.get(active_key)
    if analysis is not None and (
        analysis.source_hash != getattr(pdf, "hash", "")
        or analysis.version != ANALYSIS_PROMPT_VERSION
    ):
        analysis = None
        analyses.pop(active_key, None)
    legacy_analysis = st.session_state.single_analysis
    if analysis is None and legacy_analysis is not None:
        legacy_hash = str(getattr(legacy_analysis, "source_hash", "") or "")
        pdf_hash = str(getattr(pdf, "hash", "") or "")
        if legacy_hash and legacy_hash == pdf_hash and legacy_analysis.version == ANALYSIS_PROMPT_VERSION:
            analysis = legacy_analysis
            analyses[active_key] = legacy_analysis
    st.caption(f"{getattr(pdf, 'page_count', 0)} 页 · {len(pdf_assets(pdf))} 个候选图表 · {getattr(pdf, 'quality', 'unknown')}")
    if pdf.warning:
        st.warning(pdf.warning)
    lead = getattr(pdf, "lead_image", None)
    if lead and lead.image_name in st.session_state.images:
        with st.expander("论文首页"):
            st.image(st.session_state.images[lead.image_name], width="stretch")
    st.markdown("#### 1. 论文概览分析")
    st.caption("全文阶段只提炼研究主线和简要方法；详细数据与结果只分析您随后勾选的图片。")
    if st.button("继续概览分析" if analysis and not analysis.complete else "分析论文概览", type="primary", disabled=not api_key.strip()):
        progress = st.progress(0, text="准备全文证据")
        with st.spinner("正在提炼研究目标、方法概述与创新边界..."):
            analysis = analyze_paper(
                paper,
                pdf,
                {"api_key": api_key, "base_url": base_url, "model": model, "cache": st.session_state.analysis_cache,
                 "progress_callback": lambda done, total: progress.progress(done / total, text=f"已分析 {done}/{total} 段"),
                 "checkpoint": checkpoint},
                analysis,
            )
            analyses[active_key] = analysis
            st.session_state.single_analysis = analysis
    if analysis and analysis.complete:
        st.success("论文概览分析完成。")
        with st.expander("查看概览与原文证据"):
            for field, label in (("research_question", "研究问题"), ("background", "背景与意义"), ("methods", "方法概述"), ("innovation", "创新"), ("limitations", "局限性"), ("conclusion", "结论")):
                st.markdown(f"**{label}**")
                for claim in getattr(analysis, field):
                    st.markdown(claim.statement)
                    st.caption(f"p.{claim.page} {claim.figure_id} · {claim.evidence_text}")
    elif analysis:
        st.error(analysis.error or "分析未完成。")
        st.caption(f"已完成 {getattr(analysis, 'completed_chunks', 0)}/{getattr(analysis, 'total_chunks', 0)} 段；保留在本次会话，继续概览分析时复用。")

    st.markdown("##### 2. 确认关键图、表格与线路图")
    assets = pdf_assets(pdf)
    stale_reviews = False
    for figure in assets:
        if figure.vision_status == "reviewed" and getattr(figure, "review_version", "") not in {FIGURE_ANALYSIS_PROMPT_VERSION, SELECTED_REVIEW_VERSION}:
            invalidate_asset_review(figure, "图解规则已更新，请按三段式重新执行 Gemini 视觉复核。")
            stale_reviews = True
    if stale_reviews:
        st.info("检测到旧版图解，已取消其复核状态；重新复核后才能进入稿件。")
    if not assets:
        st.warning(
            "没有识别到可选图表。常见原因是 PDF 为扫描件、图注排版特殊，或本次会话仍保留旧版解析结果。"
            "请重新上传并解析 PDF，或在下方手动添加截图。"
        )
    thumbnails = st.container(key="asset-gallery").columns(3)
    for index, figure in enumerate(assets, start=1):
        with thumbnails[(index - 1) % 3]:
            if figure.image_name in st.session_state.images:
                st.image(st.session_state.images[figure.image_name], width="stretch")
            figure.selected = st.checkbox(f"{figure.figure_id} · p.{figure.page}", value=figure.selected,
                                          key=f"asset-select-{active_key}-{index}")
            with st.popover("查看大图", icon=":material/zoom_in:"):
                if figure.image_name in st.session_state.images:
                    st.image(st.session_state.images[figure.image_name], width="stretch")
                st.write(figure.caption)
    for index, figure in enumerate(assets, start=1):
        if not figure.selected:
            continue
        review_label = {
            "reviewed": "已复核",
            "text_evidence": "文本证据直出",
        }.get(figure.vision_status, "待复核")
        with st.expander(f"{figure.figure_id} · p.{figure.page} · {review_label}", expanded=False):
            st.caption(figure.caption[:900])
            if figure.image_name in st.session_state.images:
                st.image(st.session_state.images[figure.image_name], use_container_width=True)
                if st.checkbox("调整裁剪", key=f"asset-crop-toggle-{active_key}-{index}"):
                    horizontal = st.slider(
                        "水平保留范围",
                        0,
                        100,
                        (0, 100),
                        key=f"asset-crop-x-{paper_key(paper)}-{index}",
                    )
                    vertical = st.slider(
                        "垂直保留范围",
                        0,
                        100,
                        (0, 100),
                        key=f"asset-crop-y-{paper_key(paper)}-{index}",
                    )
                    preview = crop_image_bytes(st.session_state.images[figure.image_name], horizontal, vertical)
                    st.image(preview, caption="裁剪预览", use_container_width=True)
                    if st.button("应用裁剪", key=f"asset-crop-apply-{paper_key(paper)}-{index}"):
                        cropped_name = f"crop-{paper_key(paper)}-{index}-{hashlib.sha256(preview).hexdigest()[:12]}.png"
                        st.session_state.images[cropped_name] = preview
                        figure.image_name = cropped_name
                        figure.page_image_name = cropped_name
                        figure.crop_bbox = (horizontal[0] / 100, vertical[0] / 100, horizontal[1] / 100, vertical[1] / 100)
                        figure.needs_manual_crop = False
                        invalidate_asset_review(figure, "截图已裁剪，请重新执行 Gemini 视觉复核后再生成。")
                        checkpoint()
                        st.rerun()
            replacement = st.file_uploader("替换截图", type=["png", "jpg", "jpeg"], key=f"asset-replace-{active_key}-{index}")
            if replacement and st.button("应用替换", key=f"asset-replace-apply-{active_key}-{index}"):
                replacement_data = replacement.getvalue()
                replacement_name = f"replacement-{active_key}-{index}-{hashlib.sha256(replacement_data).hexdigest()[:12]}.png"
                if replacement_name != figure.image_name:
                    try:
                        replacement_data = crop_image_bytes(replacement_data, (0, 100), (0, 100))
                        st.session_state.images[replacement_name] = replacement_data
                        figure.image_name = replacement_name
                        figure.page_image_name = replacement_name
                        invalidate_asset_review(figure, "截图已替换，请重新复核。")
                        checkpoint()
                        st.rerun()
                    except Exception:
                        st.error("图片无法读取，请使用有效的 PNG 或 JPEG 图片。")
            if figure.editable_table:
                st.caption(f"可编辑表格候选：{len(figure.editable_table.rows)} 行；置信度 {figure.editable_table.confidence:.2f}")
            if figure.vision_status == "reviewed" and figure.interpretation:
                st.success("Gemini 视觉复核完成")
                st.markdown(figure.interpretation)
            elif figure.vision_status == "text_evidence" and figure.interpretation:
                st.info("已按图注和正文证据生成，未完成视觉复核")
                st.markdown(figure.interpretation)
            if figure.vision_error:
                st.warning(figure.vision_error)
            with st.popover("原文证据", icon=":material/menu_book:"):
                st.write(figure_context(pdf, figure) or figure.caption)
                if figure.source_excerpt:
                    st.info(figure.source_excerpt)
            caption = st.text_area("图注", figure.caption, key=f"asset-caption-{active_key}-{index}")
            if st.button("保存图注", key=f"asset-caption-save-{active_key}-{index}"):
                if caption.strip() != figure.caption:
                    figure.caption = caption.strip()
                    invalidate_asset_review(figure, "图注已更新，生成时将重新解读。")
                    checkpoint()
                    st.rerun()
    with st.expander("手动添加图表截图", expanded=not assets):
        manual_image = st.file_uploader(
            "上传图、表格或线路图截图",
            type=["png", "jpg", "jpeg"],
            key=f"asset-manual-image-{active_key}",
        )
        manual_col_a, manual_col_b = st.columns(2)
        manual_id = manual_col_a.text_input(
            "图号或名称",
            value=f"手动图 {len(assets) + 1}",
            key=f"asset-manual-id-{active_key}",
        )
        manual_page = manual_col_b.number_input(
            "所在页码",
            min_value=1,
            max_value=max(1, int(getattr(pdf, "page_count", 1) or 1)),
            value=1,
            key=f"asset-manual-page-{active_key}",
        )
        manual_caption = st.text_area(
            "图注或说明（可选）",
            key=f"asset-manual-caption-{active_key}",
            height=90,
        )
        if st.button(
            "添加为候选",
            disabled=manual_image is None or not manual_id.strip(),
            key=f"asset-manual-add-{active_key}",
        ):
            try:
                manual_data = crop_image_bytes(manual_image.getvalue(), (0, 100), (0, 100))
                manual_name = (
                    f"manual-{active_key}-{hashlib.sha256(manual_data).hexdigest()[:12]}.png"
                )
                figure = FigureAnalysis(
                    figure_id=manual_id.strip(),
                    caption=manual_caption.strip() or "用户手动添加的关键图表截图。",
                    page=str(manual_page),
                    image_name=manual_name,
                    page_image_name=manual_name,
                    why_selected="用户手动添加。",
                    needs_manual_check=True,
                    confidence=1.0,
                    role="key_result",
                    selected=True,
                    order=len(assets) + 1,
                )
                if not hasattr(pdf, "legends") or pdf.legends is None:
                    pdf.legends = []
                pdf.legends.append(figure)
                if hasattr(pdf, "all_figures") and pdf.all_figures is not None:
                    pdf.all_figures.append(figure)
                st.session_state.images[manual_name] = manual_data
                checkpoint()
                st.success("截图已加入候选并勾选。")
                st.rerun()
            except Exception:
                st.error("图片无法读取，请使用有效的 PNG 或 JPEG 图片。")
    selected_assets = sorted(
        [item for item in assets if item.selected],
        key=lambda item: (item.order or 999, item.figure_id),
    )
    st.caption(f"已选 {len(selected_assets)} 个图表，建议 2–4 个。")
    for position, item in enumerate(selected_assets):
        label_col, up_col, down_col = st.columns([6, 1, 1])
        label_col.write(f"{position + 1}. {item.figure_id}")
        up_col.button("", icon=":material/arrow_upward:", help="上移", disabled=position == 0,
                      key=f"selected-order-up-{active_key}-{position}",
                      on_click=move_selected_asset, args=(assets, item.figure_id, -1))
        down_col.button("", icon=":material/arrow_downward:", help="下移", disabled=position == len(selected_assets) - 1,
                        key=f"selected-order-down-{active_key}-{position}",
                        on_click=move_selected_asset, args=(assets, item.figure_id, 1))
    if len(selected_assets) > 4:
        st.warning("每篇最多确认 4 个关键图表，请取消多余选择。")
    st.markdown("#### 3. 生成稿件")
    mode_label = st.radio("图解方式", ["文字证据", "文字 + 视觉"], horizontal=True, key="generation-mode")
    mode = "vision" if mode_label == "文字 + 视觉" else "text"
    text_config = {"api_key": api_key, "base_url": base_url, "model": model, "provider": provider}
    review_config = dict(st.session_state.get("vision_config", {})) if mode == "vision" else text_config.copy()
    review_config.update(image_assets=st.session_state.images, cache=st.session_state.vision_cache,
                         checkpoint=checkpoint)
    if mode == "vision" and not review_config.get("api_key"):
        st.info("请配置 Gemini，或选择文字证据模式。")
    ready = bool(analysis and analysis.complete and api_key.strip()
                 and review_config.get("api_key") and 1 <= len(selected_assets) <= 4)
    if st.button("生成稿件", type="primary", disabled=not ready):
        try:
            checkpoint()
            with st.spinner("正在解读已选图片，已成功的步骤将自动保存…"):
                confirmed = analyze_selected_figures(paper, analysis, selected_assets, review_config, pdf=pdf, mode=mode)
            if len(confirmed) != len(selected_assets):
                failed = next((item for item in selected_assets if item not in confirmed), None)
                st.error((failed.vision_error if failed else "") or "部分图表未完成，已保存成功结果。")
                if mode == "vision":
                    st.info("可以切换为文字证据模式后再次生成。")
            else:
                fingerprint = hashlib.sha256(json.dumps({
                    "version": "article-prose-v1", "paper": paper.to_dict(), "analysis": analysis.to_dict(),
                    "figures": [item.to_dict() for item in confirmed], "model": model, "base_url": base_url,
                }, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                cached = st.session_state.article_cache.get(fingerprint)
                with st.spinner("正在组织公众号稿件…"):
                    article = QuickReadArticle.from_dict(cached) if cached else generate_article(
                        paper, pdf, api_key, base_url, model, analysis=analysis,
                        confirmed_figures=confirmed, image_assets=st.session_state.images,
                        provider=provider, allow_text_evidence_figures=mode == "text")
                    export_article_html(article, st.session_state.images)
                st.session_state.article_cache[fingerprint] = article.to_dict()
                preserve_article(st.session_state, deepcopy(article))
                checkpoint()
                st.success("稿件已生成，可在成稿导出中修改；上一版已保留。")
        except (ArticleGenerationError, ValueError) as exc:
            st.error(str(exc))
            checkpoint()

    if st.session_state.articles:
        st.caption(f"已有 {len(st.session_state.articles)} 篇稿件，可在“成稿导出”中编辑和下载。")


def edit_article(article: QuickReadArticle) -> None:
    revision = hashlib.sha256(article.body_markdown.encode()).hexdigest()[:12]
    key = paper_key(article.paper) + "-" + revision
    blocks = article_blocks(article.body_markdown)
    targets = [("section", index, label) for index, (label, _) in enumerate(blocks)
               if label != "关键图证据解读"]
    targets += [("figure", index, figure.figure_id) for index, figure in enumerate(article.figures)]
    with st.expander("编辑稿件", expanded=True):
        left, right = st.columns(2)
        with left:
            choice = st.selectbox("编辑位置", range(len(targets)), format_func=lambda i: targets[i][2], key=f"editor-target-{key}")
            kind, index, label = targets[choice]
            prefix, original = block_editor_parts(blocks[index][1]) if kind == "section" else ("", prose_from_note(article.figures[index].interpretation))
            draft_key = f"{key}-{kind}-{index}"
            initial = st.session_state.editor_drafts.get(draft_key, original)
            if prefix and initial.startswith(prefix):
                initial = initial[len(prefix):]
            if kind == "figure" and article.figures[index].image_name in st.session_state.images:
                st.image(st.session_state.images[article.figures[index].image_name], width="stretch")
            text = st.text_area("正文", initial, height=320, key=f"block-plain-{draft_key}")
            st.session_state.editor_drafts[draft_key] = text
            try:
                if kind == "section":
                    candidate = replace_block(article, index, prefix + text)
                else:
                    figure = deepcopy(article.figures[index])
                    figure.interpretation = text
                    candidate = replace_figure_note(article, index, figure)
            except ValueError as exc:
                st.warning(str(exc))
                candidate = None
            if st.button("保存本节", icon=":material/save:", key=f"editor-save-{draft_key}", disabled=candidate is None):
                try:
                    export_article_html(candidate, st.session_state.images)
                    preserve_article(st.session_state, candidate)
                    checkpoint()
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
            request = st.text_input("改写要求", value="更自然流畅，保留关键证据，减少重复。", key=f"editor-request-{draft_key}")
            config = st.session_state.get("text_config", {})
            if st.button("仅重写本节" if kind == "section" else "仅重写这张图", icon=":material/refresh:",
                         disabled=not config.get("api_key"), key=f"editor-rewrite-{draft_key}"):
                try:
                    paper_id = paper_key(article.paper)
                    analysis = st.session_state.paper_analyses.get(paper_id)
                    pdf = st.session_state.pdfs.get(article.paper.pdf_name)
                    if analysis is None or pdf is None:
                        raise ValueError("请先恢复此论文的分析项目。")
                    with st.spinner("正在重写所选内容…"):
                        if kind == "section":
                            revised = rewrite_block(candidate or article, index, request, analysis, pdf, config)
                        else:
                            figure = deepcopy(article.figures[index])
                            review_config = dict(st.session_state.get("vision_config", {})) if figure.review_mode == "vision" else dict(config)
                            review_config.update(image_assets=st.session_state.images, cache=st.session_state.vision_cache,
                                                 instruction=request, checkpoint=checkpoint)
                            result = analyze_selected_figures(article.paper, analysis, [figure], review_config,
                                                              pdf=pdf, mode=figure.review_mode or "text", force=True)
                            if not result:
                                raise ValueError(figure.vision_error)
                            revised = replace_figure_note(article, index, result[0])
                        export_article_html(revised, st.session_state.images)
                    preserve_article(st.session_state, revised)
                    checkpoint()
                    st.rerun()
                except (ValueError, RuntimeError) as exc:
                    st.error(friendly_llm_error(exc) if not isinstance(exc, ValueError) else str(exc))
        with right:
            st.markdown("##### 当前修改预览")
            render_article_preview(candidate or article)

        versions = [(i, item) for i, item in enumerate(st.session_state.article_versions)
                    if item["paper"].get("doi") == article.paper.doi and
                    PaperInput.from_dict(item["paper"]).paper_key == article.paper.paper_key]
        if versions:
            selected = st.selectbox("历史版本", range(len(versions)),
                                   format_func=lambda i: f"版本 {i + 1} · {versions[i][1]['title']}", key=f"editor-history-{key}")
            historical = QuickReadArticle.from_dict(versions[selected][1])
            with st.popover("查看历史版本", icon=":material/history:"):
                st.markdown(historical.body_markdown)
            if st.button("恢复此版本", icon=":material/restore:", key=f"editor-restore-{key}"):
                preserve_article(st.session_state, historical)
                checkpoint()
                st.rerun()
    with st.expander("标题与全文编辑"):
        with st.form(f"editor-{key}"):
            title = st.text_input("标题", value=article.title, max_chars=32)
            digest = st.text_area("摘要", value=article.digest, max_chars=120, height=90)
            body = st.text_area("全文（Markdown）", value=article.body_markdown, height=400)
            submitted = st.form_submit_button("保存全文")
        if submitted:
            if not title.strip() or not body.strip():
                st.error("标题和正文不能为空。")
            else:
                candidate = deepcopy(article)
                candidate.title, candidate.digest, candidate.body_markdown = title.strip(), digest.strip(), body
                candidate.body_html = markdown_to_wechat_html(body, candidate.figures)
                candidate.word_count = chineseish_len(body)
                try:
                    export_article_html(candidate, st.session_state.images)
                    preserve_article(st.session_state, candidate)
                    checkpoint()
                    st.rerun()
                except ValueError as exc:
                    st.error(f"修改未保存：{exc}")


def export_fingerprint(article: QuickReadArticle, images: dict[str, bytes]) -> str:
    payload = {"article": article.to_dict(),
               "images": {name: hashlib.sha256(data).hexdigest() for name, data in images.items()}}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def export_and_publish_tab() -> None:
    st.subheader("成稿导出")
    papers = st.session_state.papers
    articles = st.session_state.articles
    unavailable = unavailable_papers(papers, st.session_state.pdfs)
    if unavailable:
        with st.expander(f"未获得全文 · {len(unavailable)} 篇"):
            st.download_button("下载 DOI 清单", unavailable_dois_csv(unavailable).encode("utf-8-sig"),
                               file_name="unavailable_dois.csv", mime="text/csv")
            st.dataframe(paper_rows(unavailable), use_container_width=True, hide_index=True)
    if not articles:
        st.info("暂无稿件。完成论文分析和图表复核后，在论文分析页生成稿件。")
    else:
        titles = tuple(article.title for article in articles)
        index = st.selectbox("选择稿件", range(len(articles)), format_func=lambda i: titles[i])
        article = articles[index]
        edit_article(article)
        st.caption(f"{article.word_count} 字 · {len(article.figures)} 个关键图表")
        if article.warnings:
            with st.expander("来源与核对事项"):
                for warning in article.warnings:
                    st.write(warning)
        fingerprint = export_fingerprint(article, st.session_state.images)
        if st.button("准备下载文件", type="primary"):
            try:
                with st.spinner("正在生成 Word 与便携文件..."):
                    st.session_state.prepared_exports = {
                        "key": fingerprint,
                        "docx": export_article_docx_bytes(article, st.session_state.images),
                        "html": export_article_html(article, st.session_state.images),
                        "md": export_article_markdown(article),
                    }
            except (DocxExportError, ValueError) as exc:
                st.error(str(exc))
        prepared = st.session_state.get("prepared_exports", {})
        if prepared.get("key") == fingerprint:
            file_title = re.sub(r'[<>:"/\\\\|?*]', "_", article.title)
            st.download_button("下载 Word", prepared["docx"], file_name=f"{file_title}.docx",
                               mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                               type="primary")
            with st.expander("其他格式"):
                st.download_button("便携 HTML（内嵌图片）", prepared["html"], file_name=f"{file_title}.html", mime="text/html")
                st.download_button("Markdown", prepared["md"], file_name=f"{file_title}.md")
        render_article_preview(article)
        with st.expander("微信公众号草稿"):
            render_publish_settings(article)
    if papers or articles:
        with st.expander("项目归档"):
            if st.button("打包当前项目"):
                project = BatchProject(topic="论文解读", papers=papers, articles=articles, downloads=st.session_state.downloads)
                payload, error = build_project_zip_download(project, st.session_state.images, st.session_state.downloads)
                if error:
                    st.error(error)
                elif payload is not None:
                    st.download_button("下载项目 ZIP", payload, file_name="weixin-project.zip", mime="application/zip")


def render_publish_settings(article: QuickReadArticle) -> None:
    app_id = st.text_input("APP_ID")
    app_secret = st.text_input("APP_SECRET", type="password")
    cover = st.file_uploader("草稿封面图", type=["png", "jpg", "jpeg"], key="draft-cover")
    cover_name = ""
    if cover:
        cover_name = f"draft-cover-{cover.name}"
        st.session_state.images[cover_name] = cover.getvalue()
    source_url = st.text_input("原文链接", value=article.paper.url, key=f"draft-url-{paper_key(article.paper)}")
    dry_run = st.checkbox("仅预览草稿，不发布", value=True)
    config = WechatDraftConfig(
        app_id=app_id, app_secret=app_secret, author="",
        cover_image_name=cover_name or article.cover_image_name,
        content_source_url=source_url,
    )
    st.download_button("草稿 JSON", export_wechat_payload(article, config), file_name="wechat-payload.json")
    if st.button("预览草稿" if dry_run else "创建公众号草稿", disabled=not dry_run and not (app_id and app_secret and config.cover_image_name)):
        try:
            result = publish_draft(article, config, st.session_state.images, dry_run=dry_run)
            st.success("草稿预览已生成。" if dry_run else "公众号草稿已创建。")
            st.json(result)
        except Exception as exc:
            st.error(f"草稿操作失败：{exc}")


def main() -> None:
    initialize_workspace()
    init_state()
    st.markdown("""<style>
    .block-container {max-width:1200px;padding-top:2rem;padding-bottom:3rem;}
    h1 {font-size:1.65rem !important;} h2 {font-size:1.35rem !important;}
    h3,h4 {font-size:1.1rem !important;}
    [data-testid='stMetricValue'] {font-size:1.4rem;}
    [data-testid='stSidebar'] {border-right:1px solid #dedee3;}
    button p {white-space:normal;overflow-wrap:anywhere;}
    .st-key-asset-gallery [data-testid='stImage'] {height:200px;}
    .st-key-asset-gallery [data-testid='stImage'] img {height:200px;object-fit:contain;width:100%;}
    </style>""", unsafe_allow_html=True)
    st.sidebar.title("文献工作台")
    render_workspace_sidebar()
    page = st.sidebar.radio("工作区", ["文献检索", "论文分析", "成稿导出"], key="workspace-page")
    st.sidebar.divider()
    provider, api_key, base_url, model, batch_size, delay_seconds = sidebar_settings()
    st.title("微信文献快读工具")
    st.caption(f"候选文献 {len(st.session_state.papers)} · 已解析全文 {len(st.session_state.pdfs)} · 稿件 {len(st.session_state.articles)}")
    st.session_state.text_config = {"provider": provider, "api_key": api_key, "base_url": base_url, "model": model}
    try:
        if page == "文献检索":
            search_tab(provider, api_key, base_url, model, batch_size, delay_seconds)
        elif page == "论文分析":
            ingest_and_generate_tab(provider, api_key, base_url, model)
        else:
            export_and_publish_tab()
    finally:
        checkpoint()


if __name__ == "__main__":
    main()
