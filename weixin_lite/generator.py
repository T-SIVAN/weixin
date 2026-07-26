from __future__ import annotations

import html
import re
from typing import Any

from .llm import LLMError, call_openai_compatible, parse_json_object
from .models import FigureAnalysis, PaperInput, QuickReadArticle
from .pdf_reader import PdfContent, compact_text


TARGET_MIN = 500
TARGET_MAX = 1500


SYSTEM_PROMPT = """你是一个严谨的中文生物技术公众号作者，专门写单篇文献快读。
写作风格参考用户给出的微信公众号文章：中文主体、短小标题、编号要点、关键数据和关键图例穿插分析。
必须遵守：
1. 正文只写中文，不加入英文段落，不做中英对照。
2. 每篇 500-1500 个中文字符，优先 1000-1300 字。
3. 必须包含“文章核心要点简述”和“文章的创新意义”两个小标题。
4. 必须分析关键数据和关键图例：图在比较什么、关键趋势或数字是什么、支持什么结论。
5. 所有数字必须来自摘要、正文、图注或证据候选；不得编造影响因子、数据、单位、作者机构。
6. 如果没有全文或图注，必须明确写“图例分析需开放全文或上传 PDF 后补充”，不能假装读过图。
7. 标题不超过 32 个中文字符，digest 不超过 120 字。
只返回 JSON，不要返回 Markdown 代码块。"""


def chineseish_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def build_prompt(paper: PaperInput, pdf: PdfContent | None, target_chars: int) -> str:
    authors = ", ".join(paper.authors[:6])
    metadata = f"""
英文题名：{paper.title_en or paper.title}
中文题名：{paper.title_zh}
作者：{authors}
期刊：{paper.journal}
年份：{paper.year}
DOI：{paper.doi}
链接：{paper.url}
英文摘要：{paper.abstract_en or paper.abstract}
中文摘要：{paper.abstract_zh}
开放全文状态：{paper.access_status}
""".strip()
    pdf_text = pdf.prompt_text() if pdf else "未提供 PDF 或开放全文。只能基于题录和摘要生成摘要级快读。"
    return f"""
请为下面这篇论文生成可直接发布到微信公众号的中文单篇快读。
目标长度：约 {target_chars} 个中文字符，硬限制 {TARGET_MIN}-{TARGET_MAX}。

输出 JSON Schema：
{{
  "title": "不超过32个中文字符的公众号标题",
  "digest": "不超过120字摘要",
  "intro": "约80-120字，用中文说明这篇文章解决什么问题",
  "core_points": ["2-3条核心要点，每条包含必要数据和证据边界"],
  "figure_analyses": [
    {{"figure_id": "Fig. 1", "interpretation": "这张图/表在比较什么、关键趋势/数据是什么、支撑什么结论"}}
  ],
  "innovation": ["2-3条创新意义"],
  "take_home": "一句话总结"
}}

文献信息：
{metadata}

全文/图注/数据证据：
{pdf_text}
""".strip()


def short_title(paper: PaperInput) -> str:
    journal = (paper.journal or "文献").split()[0][:10]
    base = paper.title_zh or paper.title_en or paper.title or paper.doi or "单篇快读"
    base = re.sub(r"[:：].*$", "", base)
    base = re.sub(r"\s+", "", base)
    return f"{journal}|{base[:20]}"[:32]


def fallback_article(paper: PaperInput, pdf: PdfContent | None) -> dict[str, Any]:
    figures = pdf.legends[:3] if pdf else []
    evidence = pdf.evidence[:6] if pdf else []
    title = paper.title_zh or paper.title_en or paper.title or paper.doi
    points = [
        f"这篇文章围绕“{title}”展开，适合作为 {paper.journal or '相关期刊'} 的单篇快读。当前生成稿严格基于题录、摘要和可读取全文证据，不补写未出现的数据。",
    ]
    if evidence:
        values = "、".join(item.value for item in evidence[:4])
        points.append(f"全文或图注中可追踪到的关键数据包括 {values}。这些数字会在证据表中保留页码、图号或图注来源，发布前可逐项核对。")
    else:
        points.append("当前没有读取到可追踪的关键数字，因此正文不加入未经证实的性能指标。")
    if figures:
        points.append(f"可用于图例分析的线索集中在 {', '.join(fig.figure_id for fig in figures)}。这些图适合穿插在正文中解释实验设计、性能比较或酶工程结果。")
    else:
        points.append("图例分析需开放全文或上传 PDF 后补充；当前只能生成摘要级快读。")
    return {
        "title": short_title(paper),
        "digest": f"{paper.journal or '文献'}快读：{title[:70]}",
        "intro": f"本文聚焦 {title}，与 TdT、PUP 或酶促 DNA/RNA 合成方向的技术进展相关。",
        "core_points": points[:3],
        "figure_analyses": [
            {
                "figure_id": fig.figure_id,
                "interpretation": "根据图注，该图是论文中的关键结果展示；需结合原图坐标轴、分组和图例细节确认后发布。",
            }
            for fig in figures
        ],
        "innovation": [
            "把酶促核酸合成问题落到可验证的反应体系、底物选择或酶工程策略上。",
            "为 TdT/PUP 相关酶的改造和应用提供了可追溯的实验线索。",
        ],
        "take_home": "这篇文章可作为酶促 DNA/RNA 合成方向的中文快读；正式发布前应优先核对全文图例和关键数字。",
    }


def render_markdown(paper: PaperInput, data: dict[str, Any], figures: list[FigureAnalysis]) -> str:
    lines: list[str] = [f"# {data.get('title') or short_title(paper)}", ""]
    intro = str(data.get("intro") or "").strip()
    if intro:
        lines.extend([intro, ""])
    lines.append("## 文章核心要点简述")
    lines.append("")
    for idx, point in enumerate(data.get("core_points") or [], start=1):
        lines.append(f"{idx}. {str(point).strip()}")
    figure_map = {figure.figure_id.lower(): figure for figure in figures}
    for item in data.get("figure_analyses") or []:
        fig_id = str(item.get("figure_id") or "").strip()
        interpretation = str(item.get("interpretation") or "").strip()
        figure = figure_map.get(fig_id.lower())
        lines.append("")
        if figure and figure.image_name:
            lines.append(f"![{fig_id}](images/{figure.image_name})")
        lines.append(f"**{fig_id or '关键图例'} 解读：** {interpretation}")
    lines.append("")
    lines.append("## 文章的创新意义")
    lines.append("")
    for idx, point in enumerate(data.get("innovation") or [], start=1):
        lines.append(f"{idx}. {str(point).strip()}")
    if data.get("take_home"):
        lines.append("")
        lines.append(str(data["take_home"]).strip())
    meta = " / ".join(bit for bit in [paper.journal, paper.year, f"DOI: {paper.doi}" if paper.doi else "", paper.url] if bit)
    if meta:
        lines.append("")
        lines.append(f"原文信息：{meta}")
    return "\n".join(lines).strip()


def markdown_to_wechat_html(markdown: str) -> str:
    html_lines: list[str] = []
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            html_lines.append("<p><br></p>")
        elif line.startswith("# "):
            html_lines.append(f"<h2>{html.escape(line[2:])}</h2>")
        elif line.startswith("## "):
            html_lines.append(f"<h3>{html.escape(line[3:])}</h3>")
        elif line.startswith("!["):
            alt = html.escape(line[line.find("[") + 1 : line.find("]")])
            src = html.escape(line[line.find("(") + 1 : line.rfind(")")])
            html_lines.append(f'<p><img src="{src}" alt="{alt}"></p>')
        elif re.match(r"^\d+\.\s+", line):
            html_lines.append(f"<p>{html.escape(line)}</p>")
        else:
            escaped = html.escape(line)
            escaped = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", escaped)
            html_lines.append(f"<p>{escaped}</p>")
    return "\n".join(html_lines)


def attach_interpretations(figures: list[FigureAnalysis], data: dict[str, Any]) -> None:
    by_key = {figure.figure_id.lower(): figure for figure in figures}
    for item in data.get("figure_analyses") or []:
        figure = by_key.get(str(item.get("figure_id") or "").lower())
        if figure:
            figure.interpretation = str(item.get("interpretation") or "")


def generate_article(
    paper: PaperInput,
    pdf: PdfContent | None = None,
    api_key: str = "",
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
    target_chars: int = 1200,
) -> QuickReadArticle:
    warnings: list[str] = []
    if api_key.strip():
        try:
            raw = call_openai_compatible(
                api_key=api_key,
                base_url=base_url,
                model=model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=build_prompt(paper, pdf, target_chars),
            )
            data = parse_json_object(raw)
        except Exception as exc:
            warnings.append(f"LLM 生成失败，已使用保守模板：{exc}")
            data = fallback_article(paper, pdf)
    else:
        warnings.append("未填写 LLM API Key，已生成摘要级模板稿。")
        data = fallback_article(paper, pdf)

    figures = pdf.legends[:4] if pdf else []
    attach_interpretations(figures, data)
    markdown = render_markdown(paper, data, figures)
    count = chineseish_len(markdown)
    if api_key.strip() and (count < TARGET_MIN or count > TARGET_MAX):
        try:
            repair_prompt = (
                f"请把下面公众号稿改写到 {TARGET_MIN}-{TARGET_MAX} 个中文字符。"
                "只保留中文正文，不新增未给出的数字，保留两个固定小标题。\n\n"
                f"{markdown}"
            )
            markdown = call_openai_compatible(
                api_key=api_key,
                base_url=base_url,
                model=model,
                system_prompt="你是严谨的中文科技编辑，只做长度和表达修正。",
                user_prompt=repair_prompt,
                timeout=90,
            )
            count = chineseish_len(markdown)
        except LLMError as exc:
            warnings.append(f"长度自动修正失败：{exc}")
    if count < TARGET_MIN or count > TARGET_MAX:
        warnings.append(f"当前字数 {count}，超出 500-1500 发布目标，请人工微调或换更强模型重生成。")
    evidence = pdf.evidence[:30] if pdf else []
    if not pdf or not pdf.legends:
        warnings.append("未获得可靠图注；图例分析需开放全文或上传 PDF 后补充。")
    return QuickReadArticle(
        paper=paper,
        title=str(data.get("title") or short_title(paper))[:32],
        digest=str(data.get("digest") or "")[:120],
        body_markdown=compact_text(markdown, 9000),
        body_html=markdown_to_wechat_html(markdown),
        figures=figures,
        evidence=evidence,
        word_count=count,
        warnings=warnings,
    )
