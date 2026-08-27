from __future__ import annotations

import json
import os
import re
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
from weixin_lite.generator import ArticleGenerationError, generate_article
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
from weixin_lite.translate import translate_records
from weixin_lite.wechat_publish import WechatDraftConfig, export_wechat_payload, publish_draft


st.set_page_config(page_title="微信文献快读工具", page_icon="🧬", layout="wide")


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
    st.session_state.setdefault("keywords", ", ".join(DEFAULT_KEYWORDS))
    st.session_state.setdefault("query_plan", None)
    st.session_state.setdefault("search_append", False)
    st.session_state.setdefault("single_analysis", None)
    st.session_state.setdefault("analysis_cache", {})
    st.session_state.setdefault("vision_cache", {})


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
        ):
            if not getattr(old, field) and getattr(item, field):
                setattr(old, field, getattr(item, field))
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
    api_key = st.sidebar.text_input("API Key", value=default_api_key(provider), type="password")
    base_url = st.sidebar.text_input("Base URL", value=default_base_url(provider) or defaults.base_url)
    model = st.sidebar.text_input("Model", value=default_model(provider) or defaults.default_model)
    batch_size = st.sidebar.slider("翻译批量", 1, 8, 3)
    delay_seconds = st.sidebar.slider("翻译间隔（秒）", 0.0, 10.0, 1.5, step=0.5)
    st.sidebar.caption("支持 OpenAI-compatible 接口。遇到 429 时请调小批量、调大间隔，或换额度更高的供应商。")
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
    st.subheader("每日顶刊最新文章检索与标题翻译")
    latest = load_latest_run()
    if latest and latest.records:
        with st.expander(f"每日历史结果：{latest.finished_at or latest.started_at}"):
            label = "期刊：" if latest.search_kind == "journal_latest" else "关键词："
            st.caption(label + ", ".join(latest.keywords[:12]) + (" ..." if len(latest.keywords) > 12 else ""))
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

    if st.button("抓取最新文章并翻译标题", type="primary"):
        with st.spinner("正在按期刊检索 PubMed、Europe PMC、OpenAlex、Crossref，并翻译标题..."):
            run = run_journal_latest_search(
                journals,
                limit=limit,
                sources=selected_sources,
                since_days=since_days,
                openalex_api_key=openalex_api_key,
            )
            report = None
            if run.records:
                report = translate_records(
                    run.records,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    provider=provider,
                    batch_size=batch_size,
                    delay_seconds=delay_seconds,
                )
        st.session_state.papers = merge_papers(st.session_state.papers, run.records) if append_results else list(run.records)
        if run.errors:
            st.warning("部分检索源失败：" + "; ".join(f"{k}: {v}" for k, v in run.errors.items()))
        if run.warnings:
            st.info("；".join(run.warnings))
        if report and report.errors:
            st.warning("翻译未完全成功：" + "; ".join(report.errors))
            if any("429" in item or "Too Many Requests" in item for item in report.errors):
                st.info("429 是模型供应商限流/额度问题。建议把侧边栏“翻译批量”调为 1，把“翻译间隔”调到 5-10 秒，或切换 DeepSeek/SiliconFlow/custom。")
        elif report:
            st.success(f"已翻译 {report.translated_count} 条标题。")
        else:
            st.info("本次没有符合条件的最新文章，已跳过标题翻译。")
        st.dataframe(paper_rows(run.records), use_container_width=True, hide_index=True)
        show_search_run_diagnostics(run)
        show_details(run.records, "查看本次检索详情")

    papers: list[PaperInput] = st.session_state.papers
    if papers:
        st.divider()
        st.dataframe(paper_rows(papers), use_container_width=True, hide_index=True)
        show_details(papers, "查看候选摘要和错误")


def ingest_and_generate_tab(provider: str, api_key: str, base_url: str, model: str) -> None:
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
    if not ready:
        st.info("暂无可生成论文。需要先下载并解析开放 PDF，或上传 PDF。")
        return
    labels = [f"{idx}. {paper.display_title[:90]}" for idx, paper in enumerate(ready, start=1)]
    selected_label = st.selectbox("活动文章", labels)
    paper = ready[labels.index(selected_label)]
    pdf = st.session_state.pdfs[paper.pdf_name]
    st.markdown("#### 单篇深度工作台")
    st.caption("全文分段取证，不设置分析或成稿字数上限；确认图表后必须经 Gemini 视觉复核。")
    if st.button("1. 执行结构化全文分析", type="primary"):
        with st.spinner("正在按章节分析全文证据..."):
            st.session_state.single_analysis = analyze_paper(
                paper, pdf, {"api_key": api_key, "base_url": base_url, "model": model, "cache": st.session_state.analysis_cache}, st.session_state.single_analysis
            )
    analysis = st.session_state.single_analysis
    if analysis and analysis.complete:
        st.success("全文结构化分析完成。")
        st.dataframe([claim.to_dict() for claim in analysis.claims], use_container_width=True, hide_index=True)
    elif analysis:
        st.error(analysis.error or "分析未完成。")

    st.markdown("##### 2. 确认关键图、表格与线路图")
    for index, figure in enumerate(pdf.assets, start=1):
        with st.expander(f"{index}. {figure.figure_id} | {figure.asset_kind} | p.{figure.page}", expanded=index <= 2):
            figure.selected = st.checkbox("选入最终稿", value=figure.selected, key=f"asset-select-{paper_key(paper)}-{index}")
            figure.order = st.number_input("顺序", 1, 4, int(figure.order or min(index, 4)), key=f"asset-order-{paper_key(paper)}-{index}")
            st.caption(figure.caption[:900])
            if figure.image_name in st.session_state.images:
                st.image(st.session_state.images[figure.image_name], use_container_width=True)
                with st.expander("微调裁剪", expanded=False):
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
                        cropped_name = f"crop-{paper_key(paper)}-{index}.png"
                        st.session_state.images[cropped_name] = preview
                        figure.image_name = cropped_name
                        figure.page_image_name = cropped_name
                        figure.crop_bbox = (horizontal[0] / 100, vertical[0] / 100, horizontal[1] / 100, vertical[1] / 100)
                        figure.needs_manual_crop = False
                        invalidate_asset_review(figure, "截图已裁剪，请重新执行 Gemini 视觉复核后再生成。")
                        st.rerun()
            if figure.editable_table:
                st.caption(f"可编辑表格候选：{len(figure.editable_table.rows)} 行；置信度 {figure.editable_table.confidence:.2f}")
            if figure.vision_error:
                st.warning(figure.vision_error)
    selected_assets = [item for item in pdf.assets if item.selected]
    if st.button("3. Gemini 视觉复核已选资产", type="primary"):
        with st.spinner("正在复核图中曲线、表格、箭头关系与关键数据..."):
            reviewed = analyze_confirmed_figures(
                paper, analysis, selected_assets,
                {"provider": provider, "api_key": api_key, "base_url": base_url, "model": model, "image_assets": st.session_state.images},
            )
        if reviewed:
            st.success(f"已复核 {len(reviewed)} 项资产。")
        else:
            st.error("没有资产完成 Gemini 视觉复核；未复核资产不会进入最终稿。")

    if st.button("4. 生成无字数上限公众号稿", type="primary"):
        if not analysis or not analysis.complete:
            st.error("请先完成结构化全文分析。")
        else:
            try:
                article = generate_article(paper, pdf, api_key, base_url, model, analysis=analysis, confirmed_figures=selected_assets, image_assets=st.session_state.images, provider=provider)
                st.session_state.articles = [article] + [item for item in st.session_state.articles if item.title != article.title]
                st.success("深度稿已生成。")
            except ArticleGenerationError as exc:
                st.error(str(exc))

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
                        invalidate_asset_review(figure, "截图已替换，请重新执行 Gemini 视觉复核后再生成。")
            if article.warnings:
                st.warning("；".join(article.warnings))
            render_article_preview(article)
            if article.evidence:
                st.caption("证据追踪")
                st.dataframe([item.to_dict() for item in article.evidence[:12]], use_container_width=True, hide_index=True)
            col_a, col_b, col_c = st.columns(3)
            try:
                col_a.download_button("Word（可编辑）", export_article_docx_bytes(article, st.session_state.images), file_name=f"{idx:02d}-{article.title}.docx")
            except DocxExportError as exc:
                col_a.error(str(exc))
            col_b.download_button("Markdown", export_article_markdown(article), file_name=f"{idx:02d}-{article.title}.md")
            try:
                col_c.download_button(
                    "HTML",
                    export_article_html(article, st.session_state.images),
                    file_name=f"{idx:02d}-{article.title}.html",
                )
            except ValueError as exc:
                col_c.error(f"HTML 暂不能导出：{exc}")


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
        project_data, project_error = build_project_zip_download(
            project,
            st.session_state.images,
            st.session_state.downloads,
        )
        if project_error:
            st.error(project_error)
        elif project_data is not None:
            st.download_button(
                "下载项目包",
                project_data,
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

    col_a, col_b, col_c, col_d = st.columns(4)
    try:
        col_a.download_button("单篇 Word（可编辑）", export_article_docx_bytes(article, st.session_state.images), file_name=f"{article.title}.docx")
    except DocxExportError as exc:
        col_a.error(str(exc))
    col_b.download_button("单篇 Markdown", export_article_markdown(article), file_name=f"{article.title}.md")
    try:
        col_c.download_button(
            "单篇 HTML",
            export_article_html(article, st.session_state.images),
            file_name=f"{article.title}.html",
        )
    except ValueError as exc:
        col_c.error(f"HTML 暂不能导出：{exc}")

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
    col_d.download_button("草稿 Payload", export_wechat_payload(article, config), file_name=f"{article.title}-wechat-payload.json")
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
        ingest_and_generate_tab(provider, api_key, base_url, model)
    with tab_export:
        export_and_publish_tab()


if __name__ == "__main__":
    main()
