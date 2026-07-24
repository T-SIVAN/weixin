from __future__ import annotations

import json
import re
from dataclasses import replace

import streamlit as st

from weixin_lite.exporter import export_article_html, export_article_markdown, post_to_bridge, project_zip
from weixin_lite.generator import generate_article
from weixin_lite.models import BatchProject, PaperInput, QuickReadArticle
from weixin_lite.pdf_reader import PdfContent, parse_pdf
from weixin_lite.search import DEFAULT_TOPIC, federated_search, parse_manual_inputs, resolve_doi


st.set_page_config(page_title="微信公众号文献快读生成器", page_icon="🧬", layout="wide")


def init_state() -> None:
    st.session_state.setdefault("papers", [])
    st.session_state.setdefault("pdfs", {})
    st.session_state.setdefault("articles", [])
    st.session_state.setdefault("images", {})
    st.session_state.setdefault("topic", DEFAULT_TOPIC)


def paper_key(paper: PaperInput) -> str:
    base = paper.doi or paper.pmid or paper.title or paper.pdf_name
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
        for field in ("title", "doi", "pmid", "journal", "year", "abstract", "url", "oa_pdf_url", "pdf_name"):
            if not getattr(old, field) and getattr(item, field):
                setattr(old, field, getattr(item, field))
        if len(item.authors) > len(old.authors):
            old.authors = item.authors
    return list(merged.values())


def infer_paper_from_pdf(name: str, pdf: PdfContent) -> PaperInput:
    first = pdf.text[:3000]
    doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", first, flags=re.I)
    title = name.rsplit(".", 1)[0]
    for line in first.splitlines():
        clean = re.sub(r"\s+", " ", line).strip()
        if 30 <= len(clean) <= 220 and not clean.lower().startswith(("abstract", "introduction")):
            title = clean
            break
    return PaperInput(title=title, doi=doi_match.group(0).lower() if doi_match else "", pdf_name=name, source="PDF")


def render_paper_table(papers: list[PaperInput]) -> None:
    rows = []
    for idx, paper in enumerate(papers, start=1):
        rows.append(
            {
                "#": idx,
                "Title": paper.display_title,
                "Journal": paper.journal,
                "Year": paper.year,
                "DOI": paper.doi,
                "Source": paper.source,
                "PDF": "yes" if paper.pdf_name else "",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def sidebar_settings() -> tuple[str, str, str]:
    st.sidebar.header("生成设置")
    api_key = st.sidebar.text_input("OpenAI-compatible API Key", type="password")
    base_url = st.sidebar.text_input("Base URL", value="https://api.openai.com/v1")
    model = st.sidebar.text_input("Model", value="gpt-4o-mini")
    st.sidebar.caption("Key 只保存在当前浏览器会话，不会写入导出包。")
    return api_key, base_url, model


def import_tab() -> None:
    st.subheader("批量导入论文")
    topic = st.text_area("主题检索式", value=st.session_state.topic, height=90)
    st.session_state.topic = topic
    col_a, col_b, col_c = st.columns([1, 1, 1])
    limit = col_a.slider("本次候选数量", 10, 20, 10)
    sources = col_b.multiselect("检索来源", ["PubMed", "Europe PMC", "OpenAlex", "Crossref"], default=["PubMed", "Europe PMC", "OpenAlex", "Crossref"])
    email = col_c.text_input("OpenAlex 邮箱（可选）", value="")
    if st.button("检索并加入候选", type="primary"):
        with st.spinner("正在多源检索和去重..."):
            records, errors = federated_search(topic, limit=limit, sources=sources, email=email)
        st.session_state.papers = merge_papers(st.session_state.papers, records)
        if errors:
            st.warning("部分来源失败：" + "; ".join(f"{k}: {v}" for k, v in errors.items()))
        st.success(f"已加入 {len(records)} 条候选，去重后共 {len(st.session_state.papers)} 条。")

    manual = st.text_area("手动粘贴 DOI / PMID / 标题（每行一篇）", height=120)
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
    if col_e.button("清空候选"):
        st.session_state.papers = []
        st.session_state.articles = []
        st.session_state.pdfs = {}
        st.session_state.images = {}
        st.info("已清空当前批次。")

    uploaded = st.file_uploader("上传 PDF（可多选；有 PDF 才能做关键数据与关键图例分析）", type=["pdf"], accept_multiple_files=True)
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

    if st.session_state.papers:
        render_paper_table(st.session_state.papers)


def generation_tab(api_key: str, base_url: str, model: str) -> None:
    st.subheader("批量生成公众号稿")
    papers: list[PaperInput] = st.session_state.papers
    if not papers:
        st.info("先在“批量导入论文”里加入 10-20 篇候选。")
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
            article = generate_article(
                paper=paper,
                pdf=pdf,
                api_key=api_key,
                base_url=base_url,
                model=model,
                target_chars=target_chars,
            )
            generated.append(article)
            progress.progress(run_idx / len(chosen_indexes), text=f"已生成 {run_idx}/{len(chosen_indexes)}")
        st.session_state.articles = generated
        st.success(f"已生成 {len(generated)} 篇公众号稿。")

    articles: list[QuickReadArticle] = st.session_state.articles
    if not articles:
        return
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
    if not articles:
        st.info("生成文章后可在这里导出发布包或同步到草稿桥接服务。")
        return
    project = BatchProject(topic=st.session_state.topic, papers=st.session_state.papers, articles=articles)
    st.download_button(
        "下载本批次项目包 (.weixin-project.zip)",
        project_zip(project, st.session_state.images),
        file_name="weixin-batch.weixin-project.zip",
        mime="application/zip",
        type="primary",
    )
    st.caption("项目包包含 Markdown、公众号 HTML、图像和证据 JSON；不包含任何 API Key。")

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
    st.title("微信公众号文献快读生成器")
    st.caption("面向 TdT、PUP 与酶促 DNA/RNA 合成方向；批量生成 500-1500 字、带关键数据和图例证据的单篇快读。")
    tabs = st.tabs(["批量导入论文", "生成公众号稿", "导出/公众号"])
    with tabs[0]:
        import_tab()
    with tabs[1]:
        generation_tab(api_key, base_url, model)
    with tabs[2]:
        export_tab()


if __name__ == "__main__":
    main()
