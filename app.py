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
from weixin_lite.article_analysis import analyze_paper
from weixin_lite.docx_exporter import DocxExportError
from weixin_lite.exporter import (
    export_article_docx_bytes,
    export_article_html,
    export_article_markdown,
    project_zip,
    unavailable_dois_csv,
)
from weixin_lite.figure_analysis import analyze_confirmed_figures
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
    PaperInput,
    QuickReadArticle,
    SearchRun,
    generation_ready_papers,
    unavailable_papers,
)
from weixin_lite.pdf_reader import PdfContent, parse_pdf
from weixin_lite.search import (
    DEFAULT_JOURNALS_PATH,
    JournalFilter,
    load_journal_filters,
    filter_records_by_keywords,
    suggest_filter_keywords,
    parse_manual_inputs,
    resolve_doi,
    recent_year_months,
    run_journal_latest_search,
    year_month_range,
)
from weixin_lite.translate import translate_records
from weixin_lite.wechat_publish import WechatDraftConfig, export_wechat_payload, publish_draft


st.set_page_config(page_title="微信文献快读工具", layout="wide")


LATEST_PATH = Path("data/latest_papers.json")


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
    figure.vision_error = message
    figure.visual_evidence = ""
    figure.interpretation = ""


def pdf_assets(pdf: PdfContent) -> list:
    """Support Streamlit sessions restored from PdfContent versions before assets."""
    assets = getattr(pdf, "assets", None)
    return list(assets if assets is not None else getattr(pdf, "legends", []) or [])


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
                    f"抓取月份：{latest.period_label}"
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

    col_a, col_b, col_c, col_d = st.columns([0.8, 1.0, 1.4, 1.4])
    limit = col_a.slider("结果数量", 10, 200, 100, step=10)
    month_choices = recent_year_months(today=date.today())
    month_labels = [item[0] for item in month_choices]
    selected_month_label = col_b.selectbox("抓取月份", month_labels, index=0)
    selected_month = month_choices[month_labels.index(selected_month_label)]
    date_from, date_to = year_month_range(selected_month[1], selected_month[2], today=date.today())
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
        journal_to_rows(default_journals, default_enabled=False),
        use_container_width=True,
        hide_index=True,
        column_order=["启用", "期刊", "影响因子", "JIF年度"],
        disabled=["期刊", "影响因子", "JIF年度", "别名", "ISSN", "EISSN", "出版集团"],
        column_config={
            "启用": st.column_config.CheckboxColumn("启用"),
            "影响因子": st.column_config.NumberColumn("影响因子", format="%.1f"),
            "JIF年度": st.column_config.NumberColumn("JIF年度", format="%d"),
        },
        key="journal_picker",
    )
    journals = rows_to_journals(edited_journals)
    enabled_count = len([journal for journal in journals if journal.enabled])
    st.caption(
        f"已启用 {enabled_count} 本期刊；抓取月份：{selected_month_label}"
        f"（{date_from} 至 {date_to}）；期刊按 2024 JIF 从高到低排列，进入页面默认全部未选。"
    )

    if not enabled_count:
        st.info("请至少勾选一本期刊后再开始检索。")
    if st.button("检索文章", type="primary", disabled=not enabled_count or not selected_sources):
        with st.spinner("正在检索所选月份的文章..."):
            run = run_journal_latest_search(
                journals,
                limit=limit,
                sources=selected_sources,
                since_days=None,
                openalex_api_key=openalex_api_key,
                date_from=date_from,
                date_to=date_to,
                period_label=selected_month_label,
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
    if analysis is not None and analysis.source_hash != getattr(pdf, "hash", ""):
        analysis = None
        analyses.pop(active_key, None)
    legacy_analysis = st.session_state.single_analysis
    if analysis is None and legacy_analysis is not None:
        legacy_hash = str(getattr(legacy_analysis, "source_hash", "") or "")
        pdf_hash = str(getattr(pdf, "hash", "") or "")
        if legacy_hash and legacy_hash == pdf_hash:
            analysis = legacy_analysis
            analyses[active_key] = legacy_analysis
    st.caption(f"{getattr(pdf, 'page_count', 0)} 页 · {len(pdf_assets(pdf))} 个候选图表 · {getattr(pdf, 'quality', 'unknown')}")
    if pdf.warning:
        st.warning(pdf.warning)
    lead = getattr(pdf, "lead_image", None)
    if lead and lead.image_name in st.session_state.images:
        with st.expander("论文首页"):
            st.image(st.session_state.images[lead.image_name], width="stretch")
    st.markdown("#### 1. 全文分析")
    if st.button("继续分析" if analysis and not analysis.complete else "分析全文", type="primary", disabled=not api_key.strip()):
        progress = st.progress(0, text="准备全文证据")
        with st.spinner("正在按章节分析全文证据..."):
            analysis = analyze_paper(
                paper,
                pdf,
                {"api_key": api_key, "base_url": base_url, "model": model, "cache": st.session_state.analysis_cache,
                 "progress_callback": lambda done, total: progress.progress(done / total, text=f"已分析 {done}/{total} 段")},
                analysis,
            )
            analyses[active_key] = analysis
            st.session_state.single_analysis = analysis
    if analysis and analysis.complete:
        st.success("全文结构化分析完成。")
        with st.expander("查看分析与原文证据"):
            for field, label in (("research_question", "研究问题"), ("background", "背景与意义"), ("methods", "方法"), ("key_results", "关键结果"), ("innovation", "创新"), ("limitations", "局限性"), ("conclusion", "结论")):
                st.markdown(f"**{label}**")
                for claim in getattr(analysis, field):
                    st.markdown(claim.statement)
                    st.caption(f"p.{claim.page} {claim.figure_id} · {claim.evidence_text}")
    elif analysis:
        st.error(analysis.error or "分析未完成。")
        st.caption(f"已完成 {getattr(analysis, 'completed_chunks', 0)}/{getattr(analysis, 'total_chunks', 0)} 段；保留在本次会话，继续分析时复用。")

    st.markdown("##### 2. 确认关键图、表格与线路图")
    assets = pdf_assets(pdf)
    for index, figure in enumerate(assets, start=1):
        with st.expander(f"{figure.figure_id} · p.{figure.page} · {'已复核' if figure.vision_status == 'reviewed' else '待复核'}", expanded=False):
            figure.selected = st.checkbox("选入最终稿", value=figure.selected, key=f"asset-select-{paper_key(paper)}-{index}")
            figure.order = st.number_input("顺序", 1, max(4, len(assets)), min(max(1, int(figure.order or index)), max(4, len(assets))), key=f"asset-order-{paper_key(paper)}-{index}")
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
                        st.rerun()
                    except Exception:
                        st.error("图片无法读取，请使用有效的 PNG 或 JPEG 图片。")
            if figure.editable_table:
                st.caption(f"可编辑表格候选：{len(figure.editable_table.rows)} 行；置信度 {figure.editable_table.confidence:.2f}")
            if figure.vision_status == "reviewed" and figure.interpretation:
                st.success("Gemini 视觉复核完成")
                st.markdown(figure.interpretation)
            if figure.vision_error:
                st.warning(figure.vision_error)
    selected_assets = sorted(
        [item for item in assets if item.selected],
        key=lambda item: (item.order or 999, item.figure_id),
    )
    st.caption(f"已选 {len(selected_assets)} 个图表，建议 2–4 个。")
    if len(selected_assets) > 4:
        st.warning("每篇最多确认 4 个关键图表，请取消多余选择。")
    vision_config = st.session_state.get("vision_config", {})
    if st.button("复核选中图表", disabled=not analysis or not analysis.complete or not 1 <= len(selected_assets) <= 4 or not vision_config.get("api_key")):
        with st.spinner("正在复核图中曲线、表格、箭头关系与关键数据..."):
            reviewed = analyze_confirmed_figures(
                paper, analysis, selected_assets,
                {**vision_config, "image_assets": st.session_state.images, "cache": st.session_state.vision_cache},
            )
        if reviewed:
            st.success(f"已复核 {len(reviewed)} 项资产。")
        else:
            st.error("没有资产完成 Gemini 视觉复核；未复核资产不会进入最终稿。")

    st.markdown("#### 3. 生成稿件")
    can_generate = bool(api_key.strip() and analysis and analysis.complete and 1 <= len(selected_assets) <= 4 and all(item.vision_status == "reviewed" for item in selected_assets))
    if st.button("生成公众号稿", type="primary", disabled=not can_generate):
        if not analysis or not analysis.complete:
            st.error("请先完成结构化全文分析。")
        elif not selected_assets:
            st.error("请至少选择 1 张关键图、表格或线路图，并完成 Gemini 视觉复核。")
        elif any(item.vision_status != "reviewed" for item in selected_assets):
            pending = "、".join(item.figure_id for item in selected_assets if item.vision_status != "reviewed")
            st.error(f"以下已选资产尚未完成 Gemini 视觉复核：{pending}。请先执行第 3 步。")
        else:
            try:
                article = generate_article(paper, pdf, api_key, base_url, model, analysis=analysis, confirmed_figures=selected_assets, image_assets=st.session_state.images, provider=provider)
                article = deepcopy(article)
                st.session_state.articles = [article] + [item for item in st.session_state.articles if paper_key(item.paper) != active_key]
                st.success("深度稿已生成。")
            except ArticleGenerationError as exc:
                st.error(str(exc))

    if st.session_state.articles:
        st.caption(f"已有 {len(st.session_state.articles)} 篇稿件，可在“成稿导出”中编辑和下载。")


def edit_article(article: QuickReadArticle) -> None:
    key = paper_key(article.paper) + "-" + article.source_hash[:12]
    with st.expander("编辑稿件"):
        with st.form(f"editor-{key}"):
            title = st.text_input("标题", value=article.title, max_chars=32)
            digest = st.text_area("摘要", value=article.digest, max_chars=120, height=90)
            body = st.text_area("正文（Markdown）", value=article.body_markdown, height=400)
            submitted = st.form_submit_button("保存修改")
        if submitted:
            if not title.strip() or not body.strip():
                st.error("标题和正文不能为空。")
            else:
                candidate = deepcopy(article)
                candidate.title = title.strip()
                candidate.digest = digest.strip()
                candidate.body_markdown = body
                candidate.body_html = markdown_to_wechat_html(body, candidate.figures)
                candidate.word_count = chineseish_len(body)
                try:
                    export_article_html(candidate, st.session_state.images)
                except ValueError as exc:
                    st.error(f"修改未保存：{exc}")
                else:
                    article.title = candidate.title
                    article.digest = candidate.digest
                    article.body_markdown = candidate.body_markdown
                    article.body_html = candidate.body_html
                    article.word_count = candidate.word_count
                    st.success("修改已保存。")


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
    init_state()
    st.markdown("""<style>
    .block-container {max-width:1200px;padding-top:2rem;padding-bottom:3rem;}
    h1 {font-size:1.65rem !important;} h2 {font-size:1.35rem !important;}
    h3,h4 {font-size:1.1rem !important;}
    [data-testid='stMetricValue'] {font-size:1.4rem;}
    [data-testid='stSidebar'] {border-right:1px solid #dedee3;}
    button p {white-space:normal;overflow-wrap:anywhere;}
    </style>""", unsafe_allow_html=True)
    st.sidebar.title("文献工作台")
    page = st.sidebar.radio("工作区", ["文献检索", "论文分析", "成稿导出"], key="workspace-page")
    st.sidebar.divider()
    provider, api_key, base_url, model, batch_size, delay_seconds = sidebar_settings()
    st.title("微信文献快读工具")
    st.caption(f"候选文献 {len(st.session_state.papers)} · 已解析全文 {len(st.session_state.pdfs)} · 稿件 {len(st.session_state.articles)}")
    if page == "文献检索":
        search_tab(provider, api_key, base_url, model, batch_size, delay_seconds)
    elif page == "论文分析":
        ingest_and_generate_tab(provider, api_key, base_url, model)
    else:
        export_and_publish_tab()


if __name__ == "__main__":
    main()
