from __future__ import annotations

import html
import re
from typing import Any

from .llm import LLMError, call_openai_compatible, parse_json_object
from .models import FigureAnalysis, PaperInput, QuickReadArticle
from .pdf_reader import PdfContent, compact_text


TARGET_MIN = 500
TARGET_MAX = 1500


SYSTEM_PROMPT = """你是一个严谨、克制、面向中文读者的公众号文章解读作者。
你可以解读任意文章内容，包括学术论文、新闻稿、综述、技术文章、政策报告、产业文章和普通长文。
写作要求：
1. 正文只写中文，不做中英对照，不输出 Markdown 代码块。
2. 按证据强度写作：有全文时可做深度解读；只有题录、摘要或粘贴材料时，只做摘要级/材料级解读，并明确边界。
3. 必须包含“文章核心要点简述”和“文章的创新意义”两个小标题。
4. 数字、结论、机构、作者和技术细节只能来自题录、摘要、全文、图注、证据或用户提供材料，不得编造。
5. 有图片或图注时，只写图下短说明；没有图片或图注时，不强行写图解，也不要假装读过图。
6. 标题不超过 32 个中文字符，digest 不超过 120 字。
7. 目标正文 500-1500 个中文字符，优先 1000-1300 字。
只返回 JSON，不要返回 Markdown 代码块。"""


def chineseish_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _source_material(source_text: str = "", extra_text: str = "") -> str:
    chunks = []
    if source_text.strip():
        chunks.append("用户提供正文/材料：\n" + source_text.strip())
    if extra_text.strip():
        chunks.append("用户补充说明：\n" + extra_text.strip())
    return "\n\n".join(chunks)


def source_level(pdf: PdfContent | None = None, source_text: str = "", extra_text: str = "") -> str:
    if pdf and (source_text.strip() or extra_text.strip()):
        return "混合来源"
    if pdf:
        return "全文 PDF"
    if source_text.strip() or extra_text.strip():
        return "手动文本"
    return "摘要/题录"


def build_prompt(
    paper: PaperInput,
    pdf: PdfContent | None,
    target_chars: int,
    source_text: str = "",
    extra_text: str = "",
) -> str:
    authors = ", ".join(paper.authors[:6])
    metadata = f"""
英文题名：{paper.title_en or paper.title}
中文题名：{paper.title_zh}
作者：{authors}
期刊/来源：{paper.journal}
年份：{paper.year}
发表日期：{paper.publication_date}
DOI：{paper.doi}
PMID：{paper.pmid}
链接：{paper.url}
英文摘要：{paper.abstract_en or paper.abstract}
中文摘要：{paper.abstract_zh}
文章类型：{paper.article_type}
全文状态：{paper.access_status}
""".strip()
    pdf_text = pdf.prompt_text() if pdf else ""
    manual_text = _source_material(source_text, extra_text)
    if pdf_text:
        evidence_block = "全文/图注/数据证据：\n" + pdf_text
        depth = "深度解读稿"
    elif manual_text:
        evidence_block = manual_text
        depth = "材料级解读稿"
    else:
        evidence_block = "未提供 PDF 全文或额外正文；只能基于题录、摘要、DOI、期刊和链接生成摘要级解读。"
        depth = "摘要级解读稿"

    return f"""
请为下面这篇文章生成可直接发布到微信公众号的中文单篇解读。
生成层级：{depth}
目标长度：约 {target_chars} 个中文字符，建议控制在 {TARGET_MIN}-{TARGET_MAX}。

输出 JSON Schema：
{{
  "title": "不超过32个中文字符的公众号标题",
  "digest": "不超过120字摘要",
  "intro": "约80-120字，用中文说明这篇文章讨论什么问题，以及当前解读依据是什么",
  "core_points": ["2-3条核心要点，每条必须基于给定材料"],
  "figure_notes": [
    {{"figure_id": "Fig. 1", "heading": "一句话概括图中信息", "note": "1-2句中文短说明，只写可核对信息"}}
  ],
  "innovation": ["2-3条创新意义、启发或价值"],
  "take_home": "一句话总结"
}}

文章信息：
{metadata}

可用材料：
{evidence_block}
""".strip()


def short_title(paper: PaperInput) -> str:
    journal = (paper.journal or "文章").split()[0][:10]
    base = paper.title_zh or paper.title_en or paper.title or paper.doi or paper.pmid or paper.pdf_name or "单篇解读"
    base = re.sub(r"[:：|].*$", "", base)
    base = re.sub(r"\s+", "", base)
    return f"{journal}|{base[:20]}"[:32]


def fallback_article(
    paper: PaperInput,
    pdf: PdfContent | None,
    source_text: str = "",
    extra_text: str = "",
) -> dict[str, Any]:
    figures = pdf.legends[:3] if pdf else []
    evidence = pdf.evidence[:6] if pdf else []
    title = paper.title_zh or paper.title_en or paper.title or paper.doi or paper.pmid or "这篇文章"
    level = source_level(pdf, source_text, extra_text)
    has_manual = bool(source_text.strip() or extra_text.strip())

    points = [
        f"这篇文章围绕“{title}”展开，当前草稿基于{level}生成，适合作为公众号单篇解读的初稿。",
    ]
    if paper.abstract_zh or paper.abstract_en or paper.abstract:
        abstract = _clean_text(paper.abstract_zh or paper.abstract_en or paper.abstract)
        points.append(f"从摘要可见，文章重点讨论：{abstract[:180]}。正式发布前建议结合原文核对关键表述。")
    elif has_manual:
        material = _clean_text(source_text or extra_text)
        points.append(f"用户提供材料显示，文章重点可概括为：{material[:180]}。")
    else:
        points.append("当前没有摘要、全文或额外正文，因此只能形成题录级导语，不能展开为可靠的深度分析。")

    if evidence:
        values = "、".join(item.value for item in evidence[:4])
        points.append(f"全文或图注中可追踪到的关键数据包括：{values}；这些数字应在发布前逐项核对来源。")
    elif not pdf:
        points.append("当前没有解析到全文证据，文中的判断应限定在题录、摘要或用户粘贴材料范围内。")

    figure_notes = [
        {
            "figure_id": fig.figure_id,
            "heading": f"{fig.figure_id}：原文关键信息截图",
            "note": "原文截图置于上方，正文只围绕图中可见流程、比较或趋势做简短说明。",
        }
        for fig in figures
    ]

    innovation = [
        "把文章中的问题、方法或观察结果整理成中文读者更容易把握的主线。",
        "保留证据边界，避免在材料不足时把摘要级信息包装成全文级结论。",
    ]
    if has_manual:
        innovation.append("结合用户补充材料，可把原文信息转化为更贴近目标读者的解读稿。")

    return {
        "title": short_title(paper),
        "digest": f"{paper.journal or '文章'}解读：{str(title)[:70]}",
        "intro": f"本文解读 {title}。当前依据为{level}；若未提供全文，以下内容属于摘要级或材料级整理，发布前建议核对原文。",
        "core_points": points[:3],
        "figure_notes": figure_notes,
        "innovation": innovation[:3],
        "take_home": "这是一篇可开放生成的中文解读稿；材料越完整，结论和图文分析越可靠。",
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
    figure_items = data.get("figure_notes") or data.get("figure_analyses") or []
    for figure_index, item in enumerate(figure_items, start=1):
        fig_id = str(item.get("figure_id") or "").strip()
        heading = str(item.get("heading") or f"{figure_index}. {fig_id or '原文截图'}").strip()
        note = str(item.get("note") or item.get("interpretation") or "").strip()
        figure = figure_map.get(fig_id.lower())
        lines.append("")
        if figure and figure.image_name:
            lines.append(f"![{fig_id}](images/{figure.image_name})")
        elif fig_id:
            lines.append(f"> {fig_id} 需要上传原图截图后发布。")
        lines.append(f"**{heading}**")
        if note:
            lines.append("")
            lines.append(note)
    lines.append("")
    lines.append("## 文章的创新意义")
    lines.append("")
    for idx, point in enumerate(data.get("innovation") or [], start=1):
        lines.append(f"{idx}. {str(point).strip()}")
    if data.get("take_home"):
        lines.append("")
        lines.append(str(data["take_home"]).strip())
    meta = " / ".join(bit for bit in [paper.journal, paper.publication_date or paper.year, f"DOI: {paper.doi}" if paper.doi else "", paper.url] if bit)
    if meta:
        lines.append("")
        lines.append(f"原文信息：{meta}")
    return "\n".join(lines).strip()


WECHAT_P_STYLE = "margin:20px 0;font-size:18px;line-height:2.05;color:#000;text-align:justify;"
WECHAT_HEADING_STYLE = "margin:44px 0 22px;font-size:22px;line-height:1.45;color:#000;font-weight:800;"
WECHAT_SUBHEAD_STYLE = "margin:30px 0 18px;font-size:20px;line-height:1.55;color:#000;font-weight:800;"
WECHAT_IMAGE_WRAP_STYLE = "margin:38px 0 34px;"
WECHAT_STRONG_STYLE = "font-weight:800;color:#000;"
WECHAT_NOTE_STYLE = "margin:12px 0 22px;font-size:15px;line-height:1.8;color:#8a8f98;"
WECHAT_META_STYLE = "margin:38px 0 0;font-size:14px;line-height:1.8;color:#8a8f98;"


def _wechat_inline(text: str) -> str:
    escaped = html.escape(text)
    return re.sub(
        r"\*\*(.*?)\*\*",
        rf'<strong style="{WECHAT_STRONG_STYLE}">\1</strong>',
        escaped,
    )


def markdown_to_wechat_html(markdown: str) -> str:
    html_lines: list[str] = []
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            continue
        elif line.startswith("# "):
            # The WeChat platform owns the article title. Export and local preview
            # add the title back outside the rich-text body to avoid draft duplicates.
            continue
        elif line.startswith("## "):
            html_lines.append(f'<p style="{WECHAT_HEADING_STYLE}">{html.escape(line[3:])}</p>')
        elif line.startswith("!["):
            alt = html.escape(line[line.find("[") + 1 : line.find("]")])
            src = html.escape(line[line.find("(") + 1 : line.rfind(")")])
            html_lines.append(
                f'<section style="{WECHAT_IMAGE_WRAP_STYLE}">'
                f'<img src="{src}" alt="{alt}" style="width:100%;height:auto;display:block;margin:0 auto;">'
                "</section>"
            )
        elif line.startswith("> "):
            html_lines.append(f'<p style="{WECHAT_NOTE_STYLE}">{html.escape(line[2:])}</p>')
        elif line.startswith("**") and line.endswith("**") and len(line) <= 120:
            html_lines.append(f'<p style="{WECHAT_SUBHEAD_STYLE}">{_wechat_inline(line)}</p>')
        elif re.match(r"^\d+\.\s+", line):
            html_lines.append(f'<p style="{WECHAT_P_STYLE}">{_wechat_inline(line)}</p>')
        elif line.startswith("原文信息："):
            html_lines.append(f'<p style="{WECHAT_META_STYLE}">{_wechat_inline(line)}</p>')
        else:
            html_lines.append(f'<p style="{WECHAT_P_STYLE}">{_wechat_inline(line)}</p>')
    return "\n".join(html_lines)


def attach_interpretations(figures: list[FigureAnalysis], data: dict[str, Any]) -> None:
    by_key = {figure.figure_id.lower(): figure for figure in figures}
    for item in (data.get("figure_notes") or data.get("figure_analyses") or []):
        figure = by_key.get(str(item.get("figure_id") or "").lower())
        if figure:
            figure.interpretation = str(item.get("note") or item.get("interpretation") or "")


def generate_article(
    paper: PaperInput,
    pdf: PdfContent | None = None,
    api_key: str = "",
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
    target_chars: int = 1200,
    source_text: str = "",
    extra_text: str = "",
) -> QuickReadArticle:
    warnings: list[str] = []
    level = source_level(pdf, source_text, extra_text)
    if pdf:
        warnings.append("已基于 PDF 全文/图注生成深度解读稿；发布前仍建议核对关键数据和截图。")
    elif source_text.strip() or extra_text.strip():
        warnings.append("未提供 PDF 全文，已基于用户粘贴材料生成材料级解读；不要把它表述为全文级结论。")
    else:
        warnings.append("未提供 PDF 全文或额外正文，已基于题录/摘要生成摘要级解读；证据边界较窄。")

    if api_key.strip():
        try:
            raw = call_openai_compatible(
                api_key=api_key,
                base_url=base_url,
                model=model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=build_prompt(paper, pdf, target_chars, source_text=source_text, extra_text=extra_text),
            )
            data = parse_json_object(raw)
        except Exception as exc:
            warnings.append(f"LLM 生成失败，已使用通用保守模板：{exc}")
            data = fallback_article(paper, pdf, source_text=source_text, extra_text=extra_text)
    else:
        warnings.append("未填写 LLM API Key，已生成通用占位级模板稿；深度解读需要配置模型后重新生成。")
        data = fallback_article(paper, pdf, source_text=source_text, extra_text=extra_text)

    figures = pdf.legends[:4] if pdf else []
    attach_interpretations(figures, data)
    markdown = render_markdown(paper, data, figures)
    count = chineseish_len(markdown)
    if api_key.strip() and (count < TARGET_MIN or count > TARGET_MAX):
        try:
            repair_prompt = (
                f"请把下面公众号稿改写到 {TARGET_MIN}-{TARGET_MAX} 个中文字符。"
                "只保留中文正文，不新增未给出的事实、数字或结论，保留两个固定小标题。\n\n"
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
        warnings.append(f"当前字数 {count}，不在 500-1500 发布目标内；可人工微调或换更强模型重生成。")
    evidence = pdf.evidence[:30] if pdf else []
    if not pdf or not pdf.legends:
        warnings.append(f"内容来源：{level}；未获得可靠图注，图例分析需开放全文或上传 PDF 后补充。")
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
