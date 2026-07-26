from __future__ import annotations

import json
import re
from pathlib import Path

import streamlit as st

from weixin_lite.downloader import download_open_access
from weixin_lite.exporter import export_article_html, export_article_markdown, post_to_bridge, project_zip
from weixin_lite.generator import generate_article
from weixin_lite.models import BatchProject, DownloadedPaper, PaperInput, QuickReadArticle, SearchRun
from weixin_lite.pdf_reader import PdfContent, parse_pdf
from weixin_lite.search import DEFAULT_KEYWORDS, parse_manual_inputs, resolve_doi, run_keyword_search
from weixin_lite.translate import translate_records


st.set_page_config(page_title="微信公众号文献雷达", page_icon="🧬", layout="wide")


LATEST_PATH = Path("data/latest_papers.json")


def init_state() -> None:
    st.session_state.setdefault("papers", [])
    st.session_state.setdefault("pdfs", {})
    st.session_state.setdefault("articles", [])
    st.session_state.setdefault("images", {})
    st.session_state.setdefault("downloads", [])
    st.session_state.setdefault("keywords", ", ".join(DEFAULT_KEYWORDS))


def paper_key(paper: PaperInput) -> str:
    base = paper.paper_key
    return re.sub(r"[^a-zA-Z0-9]+", "-", base.lower()).strip("-")[:90]


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
        data = json.loads(LATEST_PATH.read_text(encoding="utf-8"))
        return SearchRun.from_dict(data)
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


def paper_rows(papers: list[PaperInput]) -> list[dict[str, str]]:
    rows = []
    for idx, paper in enumerate(papers, start=1):
        rows.append(
            {
                "#": str(idx),
                "英文标题": paper.title_en or paper.title,
                "中文标题": paper.title_zh,
                "英文摘要": (paper.abstract_en or paper.abstract)[:280],
                "中文摘要": paper.abstract_zh[:280],
                "期刊": paper.journal,
                "年份": paper.year,
                "DOI": paper.doi,
                "全文状态": paper.access_status,
                "来源": paper.source,
            }
        )
    return rows


def sidebar_settings() -> tuple[str, str, str]:
    st.sidebar.header("模型设置")
    api_key = st.sidebar.text_input("OpenAI-compatible API Key", type="password")
    base_url = st.sidebar.text_input("Base URL", value="https://api.openai.com/v1")
    model = st.sidebar.text_input("Model", value="gpt-4o-mini")
    st.sidebar.caption("Key 只保存在当前浏览器会话，不写入导出包。GitHub Actions 可用仓库 Secret OPENAI_API_KEY。")
    return api_key, base_url, model


def radar_tab(api_key: str, base_url: str, model: str) -> None:
    st.subheader("关键词文献雷达")
    latest = load_latest_run()
    if latest and latest.records:
        st.info(f"已读取每日自动检索结果：{latest.finished_at or latest.started_at}，关键词：{', '.join(latest.keywords)}")
        st.dataframe(paper_rows(latest.records), use_container_width=True, hide_index=True)
        if st.button("加入每日最新结果到候选池"):
            st.session_state.papers = merge_papers(st.session_state.papers, latest.records)
            st.success(f"已加入 {len(latest.records)} 条每日检索结果。")
    else:
        st.caption("尚未发现 data/latest_papers.json。GitHub Actions 首次运行后这里会自动显示每日结果。")

    st.divider()
    keywords = st.text_input("只填关键词", value=st.session_state.keywords, placeholder="例如：TdT, PUP, 酶促DNA合成")
    st.session_state.keywords = keywords
    col_a, col_b, col_c = st.columns(3)
    limit = col_a.slider("检索数量", 10, 20, 20)
    since_days = col_b.slider("时间范围（天）", 1, 365, 30)
    email = col_c.text_input("OpenAlex 邮箱（可选）", value="")
    if st.button("检索并翻译标题/摘要", type="primary"):
        with st.spinner("正在检索 PubMed、Europe PMC、OpenAlex、Crossref，并翻译标题/摘要..."):
            run = run_keyword_search(keywords, limit=limit, email=email, since_days=since_days)
            translate_records(run.records, api_key=api_key, base_url=base_url, model=model)
        st.session_state.papers = merge_papers(st.session_state.papers, run.records)
        if run.errors:
            st.warning("部分来源失败：" + "; ".join(f"{k}: {v}" for k, v in run.errors.items()))
        st.success(f"已加入 {len(run.records)} 条候选。")
        st.dataframe(paper_rows(run.records), use_container_width=True, hide_index=True)


def ingest_tab() -> None:
    st.subheader("开放全文下载与文件上传")
    papers: list[PaperInput] = st.session_state.papers
    if papers:
        st.dataframe(paper_rows(papers), use_container_width=True, hide_index=True)
        if st.button("自动下载合法开放全文并解析"):
            progress = st.progress(0)
            downloads: list[DownloadedPaper] = []
            for idx, paper in enumerate(papers, start=1):
                downloaded = download_open_access(paper)
                downloads.append(downloaded)
                if downloaded.status == "open" and downloaded.content_bytes and "pdf" in downloaded.content_type.lower():
                    try:
                        pdf = parse_pdf(downloaded.content_bytes)
                        pdf_name = downloaded.file_name or f"{paper_key(paper)}.pdf"
                        st.session_state.pdfs[pdf_name] = pdf
                        st.session_state.images.update(pdf.rendered_images)
                        paper.pdf_name = pdf_name
                        paper.access_status = "open"
                    except Exception as exc:
                        paper.download_error = f"PDF 解析失败：{exc}"
                elif downloaded.status != "open":
                    paper.access_status = downloaded.status
                    paper.download_error = downloaded.error
                progress.progress(idx / len(papers), text=f"已处理 {idx}/{len(papers)}")
            st.session_state.downloads = downloads
            st.success("开放全文下载/解析完成；需付费或失败文章已保留 DOI。")
    else:
        st.info("先在“关键词文献雷达”里加入候选，或直接上传 PDF。")

    uploaded = st.file_uploader("上传 PDF 文件进入批量分析", type=["pdf"], accept_multiple_files=True)
    if uploaded and st.button("解析上传 PDF"):
        parsed_papers: list[PaperInput] = []
        progress = st.progress(0)
        for idx, file in enumerate(uploaded, start=1):
            pdf = parse_pdf(file.getvalue())
            st.session_state.pdfs[file.name] = pdf
            st.session_state.images.update(pdf.rendered_images)
            parsed_papers.append(infer_paper_from_pdf(file.name, pdf))
            progress.progress(idx / len(uploaded), text=f"已解析 {idx}/{len(uploaded)}")
        st.session_state.papers = merge_papers(st.session_state.papers, parsed_papers)
        st.success(f"已解析 {len(parsed_papers)} 个 PDF。")

    manual = st.text_area("手动粘贴 DOI / PMID / 标题（每行一篇，可选）", height=90)
    col_d, col_e = st.columns([1, 3])
    if col_d.button("解析手动列表"):
        records = parse_manual_inputs(manual)
        resolved = []
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


def generation_tab(api_key: str, base_url: str, model: str) -> None:
    st.subheader("批量生成中文公众号稿")
    papers: list[PaperInput] = st.session_state.papers
    if not papers:
        st.info("先加入检索结果或上传 PDF。")
        return
    labels = [f"{idx}. {paper.display_title[:90]}" for idx, paper in enumerate(papers, start=1)]
    selected = st.multiselect("选择本次生成的论文", labels, default=labels[: min(10, len(labels))])
    target_chars = st.slider("目标字数", 700, 1400, 1200, step=50)
    if st.button("生成所选公众号稿", type="primary"):
        chosen_indexes = [labels.index(label) for label in selected]
        generated: list[QuickReadArticle] = []
        progress = st.progress(0)
        for run_idx, paper_idx in enumerate(chosen_indexes, start=1):
            paper = papers[paper_idx]
            pdf = st.session_state.pdfs.get(paper.pdf_name)
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

    articles: list[QuickReadArticle] = st.session_state.articles
    for idx, article in enumerate(articles, start=1):
        with st.expander(f"{idx}. {article.title} | {article.word_count} 字", expanded=idx == 1):
            cover = st.file_uploader("上传该文献的期刊网页标题图/截图（可选）", type=["png", "jpg", "jpeg"], key=f"cover-{idx}")
            if cover:
                name = f"cover-{idx}-{cover.name}"
                st.session_state.images[name] = cover.getvalue()
                article.cover_image_name = name
            if article.warnings:
                st.warning("；".join(article.warnings))
            st.markdown(article.body_markdown)
            if article.evidence:
                st.caption("证据追踪")
                st.dataframe([item.to_dict() for item in article.evidence[:12]], use_container_width=True, hide_index=True)
            col_a, col_b = st.columns(2)
            col_a.download_button("下载 Markdown", export_article_markdown(article), file_name=f"{idx:02d}-{article.title}.md")
            col_b.download_button("下载 HTML", export_article_html(article), file_name=f"{idx:02d}-{article.title}.html")


def export_tab() -> None:
    st.subheader("导出与公众号对接")
    articles: list[QuickReadArticle] = st.session_state.articles
    papers: list[PaperInput] = st.session_state.papers
    if not articles:
        st.info("生成文章后可导出发布包。")
        return
    project = BatchProject(
        topic=st.session_state.keywords,
        papers=papers,
        articles=articles,
        downloads=st.session_state.downloads,
    )
    st.download_button(
        "下载本批次项目包 (.weixin-project.zip)",
        project_zip(project, st.session_state.images, st.session_state.downloads),
        file_name="weixin-batch.weixin-project.zip",
        mime="application/zip",
        type="primary",
    )
    st.caption("项目包包含 Markdown、公众号 HTML、证据 JSON、图片、付费 DOI 列表和下载状态；不包含任何 API Key 或 PDF 缓存。")

    with st.expander("同步到公众号草稿桥接服务（可选）"):
        st.write("Streamlit Cloud 通常没有固定出口 IP，微信接口可能触发 IP 白名单限制。这里默认对接你自己的固定 IP 桥接服务，只创建草稿，不自动发布。")
        bridge_url = st.text_input("Bridge URL", placeholder="https://your-domain.example.com/v1/drafts")
        bridge_token = st.text_input("Bridge Token", type="password")
        article_titles = [f"{idx}. {article.title}" for idx, article in enumerate(articles, start=1)]
        selected = st.selectbox("选择同步文章", article_titles)
        if st.button("发送到草稿桥接服务"):
            if not bridge_url:
                st.error("请先填写 Bridge URL。")
            else:
                article = articles[article_titles.index(selected)]
                try:
                    result = post_to_bridge(bridge_url, article, bridge_token)
                    st.success("桥接服务已返回结果。")
                    st.json(result)
                except Exception as exc:
                    st.error(f"同步失败：{exc}")


def main() -> None:
    init_state()
    api_key, base_url, model = sidebar_settings()
    st.title("微信公众号文献雷达与中文快读生成器")
    st.caption("一个完整流程：只填关键词检索，网站展示标题/摘要中英对照，开放全文自动下载或上传 PDF，最后批量生成中文公众号稿并导出。")

    radar_tab(api_key, base_url, model)
    st.divider()
    ingest_tab()
    st.divider()
    generation_tab(api_key, base_url, model)
    st.divider()
    export_tab()


if __name__ == "__main__":
    main()
