from __future__ import annotations

import html
import re
from typing import Any

from .figure_analysis import analyze_confirmed_figures, figure_heading
from .llm import LLMError, call_openai_compatible, parse_json_object
from .models import FigureAnalysis, PaperAnalysis, PaperInput, QuickReadArticle
from .pdf_reader import PdfContent, compact_text


TARGET_MIN = 2800
TARGET_MAX = 4200
ANALYSIS_TARGET_MIN = TARGET_MIN
ANALYSIS_TARGET_MAX = TARGET_MAX


class ArticleGenerationError(RuntimeError):
    """Raised when a quality-first article cannot be generated safely."""


SYSTEM_PROMPT = """你是一个严谨、克制、面向中文读者的公众号文章解读作者。
你可以解读任意文章内容，包括学术论文、新闻稿、综述、技术文章、政策报告、产业文章和普通长文。
写作要求：
1. 正文只写中文，不做中英对照，不输出 Markdown 代码块。
2. 按证据强度写作：有全文时可做深度解读；只有题录、摘要或粘贴材料时，只做摘要级/材料级解读，并明确边界。
3. 必须形成完整长文结构：文章核心要点简述、研究问题与现实意义、方法路径与比较优势、实验设计与验证、关键数据与结果、关键图证据解读、文章的创新意义、局限性与解读边界、总结。
4. 数字、结论、机构、作者和技术细节只能来自题录、摘要、全文、图注、证据或用户提供材料，不得编造。
5. 有确认配图时，正文图解由系统另行插入；不要自行增加未确认图片或虚构图中细节。
6. 标题不超过 32 个中文字符，digest 不超过 120 字。
7. 当全文证据足够时，目标正文为 2800-4200 个中文字符；每个章节都要使用提供材料中可追溯的细节。证据不足时，明确说明无法展开的原因，不用泛泛常识填充。
只返回 JSON，不要返回 Markdown 代码块。"""


def chineseish_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def normalized_target_chars(target_chars: int) -> int:
    """Keep every generation path on the same evidence-first length contract."""
    return max(TARGET_MIN, min(TARGET_MAX, int(target_chars or 3200)))


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
    target_chars = normalized_target_chars(target_chars)
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
目标长度：约 {target_chars} 个中文字符，证据充分时建议控制在 {TARGET_MIN}-{TARGET_MAX}。

输出 JSON Schema：
{{
  "title": "不超过32个中文字符的公众号标题",
  "digest": "不超过120字摘要",
  "intro": "导语，交代解读依据和研究主线",
  "research_question": "研究目标、现实问题和产业/应用意义",
  "approach_advantage": ["方法路径、新思路及相对既有方案的优势，每条基于材料"],
  "experiment_validation": ["实验或验证设计、对照和样本信息"],
  "quantitative_findings": ["关键定量数据、趋势和结果，每条基于材料"],
  "figure_notes": [],
  "innovation": ["创新意义、启发或价值"],
  "limitations": ["局限性与证据边界"],
  "take_home": "一句话总结"
}}

文章信息：
{metadata}

可用材料：
{evidence_block}
""".strip()


def _analysis_claims(analysis: PaperAnalysis) -> str:
    labels = {
        "research_question": "研究问题",
        "background": "背景",
        "methods": "方法",
        "key_results": "关键结果",
        "innovation": "创新点",
        "limitations": "局限性",
        "conclusion": "结论",
    }
    lines: list[str] = []
    for field_name, label in labels.items():
        lines.append(f"\n## {label}")
        for claim in getattr(analysis, field_name):
            source = " / ".join(bit for bit in [f"p.{claim.page}" if claim.page else "", claim.figure_id] if bit)
            lines.append(f"- {claim.statement} [{source}] 证据：{claim.evidence_text}")
    return "\n".join(lines).strip()


def build_analysis_article_prompt(
    paper: PaperInput,
    analysis: PaperAnalysis,
    figures: list[FigureAnalysis],
    target_chars: int,
) -> str:
    target_chars = normalized_target_chars(target_chars)
    figure_lines = [
        f"- {item.figure_id} | role={item.role} | page={item.page} | caption={item.caption}"
        for item in figures
    ]
    return f"""
请根据已经审核为可追溯的结构化分析，生成一篇可直接排版为微信公众号正文的中文深度解读稿。
目标长度：{target_chars} 个中文字符，允许范围 {ANALYSIS_TARGET_MIN}-{ANALYSIS_TARGET_MAX}。
不得加入结构化分析之外的事实、数字或结论。只使用确认配图，并按给定顺序生成图下注释。

输出 JSON Schema：
{{
  "title": "不超过32个中文字符的公众号标题",
  "digest": "不超过120字摘要",
  "intro": "导语，交代问题和研究价值",
  "research_question": "完整说明研究问题、现实痛点与产业/转化意义，必须关联结构化分析证据",
  "approach_advantage": ["方法、模型或机制及其相对优势，每条用可追溯证据展开"],
  "experiment_validation": ["实验设计、对照、样本和验证路径，每条用可追溯证据展开"],
  "quantitative_findings": ["关键数字、趋势、结果及其含义，每条用可追溯证据展开"],
  "figure_notes": [],
  "innovation": ["创新意义"],
  "limitations": ["论文局限性和解读边界"],
  "take_home": "总结"
}}

写作要求：每一节都要写成面向读者的完整段落，而不是一句泛泛概括。可用“p.X”或“Fig. X”标记已有证据位置；没有证据的章节必须说明边界。关键图由系统在对应的“关键图证据解读”部分插入，你无需另行编造图解。

论文：{paper.title_zh or paper.title_en or paper.title}
期刊：{paper.journal}
DOI：{paper.doi}

结构化分析（每条均附页码或图号）：
{_analysis_claims(analysis)}

确认配图：
{chr(10).join(figure_lines) if figure_lines else '无确认配图；不要生成 figure_notes。'}
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

    summary = " ".join(points)
    return {
        "title": short_title(paper),
        "digest": f"{paper.journal or '文章'}解读：{str(title)[:70]}",
        "intro": f"本文解读 {title}。当前依据为{level}；若未提供全文，以下内容属于摘要级或材料级整理，发布前建议核对原文。",
        "core_points": points[:3],
        "research_question": summary,
        "approach_advantage": ["当前材料不足以可靠展开方法路径和比较优势；需要全文方法与结果部分支持。"],
        "experiment_validation": ["当前未获得完整实验设计、对照或样本信息，不能把摘要级材料写成实验验证结论。"],
        "quantitative_findings": points[2:3] or ["当前没有可核对的关键定量数据。"],
        "figure_notes": figure_notes,
        "innovation": innovation[:3],
        "limitations": [f"证据边界：当前仅基于{level}，未获得或未完整解析全文时，不应延伸为全文级判断。"],
        "take_home": "这是一篇可开放生成的中文解读稿；材料越完整，结论和图文分析越可靠。",
    }


def render_markdown(
    paper: PaperInput,
    data: dict[str, Any],
    figures: list[FigureAnalysis],
    lead_image: FigureAnalysis | None = None,
    *,
    confirmed_figure_notes: bool = False,
) -> str:
    lines: list[str] = [f"# {data.get('title') or short_title(paper)}", ""]
    if lead_image and lead_image.image_name:
        lines.extend([f"![论文首页](images/{lead_image.image_name})", ""])
    intro = str(data.get("intro") or "").strip()
    if intro:
        lines.extend([intro, ""])
    def add_section(heading: str, value: Any, *, numbered: bool = False) -> None:
        values = value if isinstance(value, list) else [value]
        cleaned = [str(item).strip() for item in values if item is not None and str(item).strip()]
        if not cleaned:
            return
        lines.extend([f"## {heading}", ""])
        for index, item in enumerate(cleaned, start=1):
            lines.append(f"{index}. {item}" if numbered else item)
            lines.append("")

    # Legacy template data may include core_points. Do not repeat the complete
    # research-question section when the evidence-driven schema has no summary.
    add_section("文章核心要点简述", data.get("core_points"), numbered=True)
    add_section("研究问题与现实意义", data.get("research_question"))
    add_section("方法路径与比较优势", data.get("approach_advantage"), numbered=True)
    add_section("实验设计与验证", data.get("experiment_validation"), numbered=True)
    add_section("关键数据与结果", data.get("quantitative_findings") or data.get("core_points"), numbered=True)
    figure_map = {figure.figure_id.lower(): figure for figure in figures}
    if confirmed_figure_notes:
        figure_items = [
            {
                "figure_id": figure.figure_id,
                "heading": figure.why_selected or figure_heading(figure),
                "note": figure.interpretation,
            }
            for figure in figures
            if figure.image_name and figure.interpretation
        ]
    else:
        figure_items = data.get("figure_notes") or data.get("figure_analyses") or []
    if figure_items:
        lines.extend(["", "## 关键图证据解读", ""])
    for figure_index, item in enumerate(figure_items, start=1):
        fig_id = str(item.get("figure_id") or "").strip()
        heading = str(item.get("heading") or f"{figure_index}. {fig_id or '原文截图'}").strip()
        note = str(item.get("note") or item.get("interpretation") or "").strip()
        figure = figure_map.get(fig_id.lower())
        lines.append("")
        if confirmed_figure_notes and not (figure and figure.image_name and note):
            continue
        if figure and figure.image_name:
            lines.append(f"![{fig_id}](images/{figure.image_name})")
        elif fig_id and not confirmed_figure_notes:
            lines.append(f"> {fig_id} 需要上传原图截图后发布。")
        lines.append(f"**{heading}**")
        if note:
            lines.append("")
            lines.append(note)
    add_section("文章的创新意义", data.get("innovation"), numbered=True)
    add_section("局限性与解读边界", data.get("limitations"), numbered=True)
    if data.get("take_home"):
        lines.extend(["", "## 总结", "", str(data["take_home"]).strip()])
    meta = " / ".join(bit for bit in [paper.journal, paper.publication_date or paper.year, f"DOI: {paper.doi}" if paper.doi else "", paper.url] if bit)
    if meta:
        lines.append("")
        lines.append(f"原文信息：{meta}")
    return "\n".join(lines).strip()


def render_article_with_confirmed_figures(
    paper: PaperInput,
    data: dict[str, Any],
    figures: list[FigureAnalysis],
    lead_image: FigureAnalysis | None = None,
) -> str:
    return render_markdown(
        paper,
        data,
        figures,
        lead_image=lead_image,
        confirmed_figure_notes=True,
    )


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


def _friendly_llm_failure(exc: Exception, action: str) -> str:
    message = str(exc).lower()
    status_code = exc.status_code if isinstance(exc, LLMError) else None
    if status_code == 429 or "too many requests" in message or "quota" in message:
        return f"{action}：模型接口额度不足或被限流，请检查 API Key 余额/套餐，或稍后重试。"
    if status_code in {401, 403}:
        return f"{action}：模型接口鉴权失败，请检查 API Key、Base URL 和模型权限。"
    if isinstance(exc, LLMError) and exc.transient:
        return f"{action}：模型接口暂时不可用，请稍后重试。"
    return f"{action}：模型调用失败，已保留当前可用结果。"


def generate_article(
    paper: PaperInput,
    pdf: PdfContent | None = None,
    api_key: str = "",
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
    target_chars: int = 3200,
    source_text: str = "",
    extra_text: str = "",
    analysis: PaperAnalysis | None = None,
    confirmed_figures: list[FigureAnalysis] | None = None,
    target_profile: str = "adaptive",
    image_assets: dict[str, bytes] | None = None,
) -> QuickReadArticle:
    warnings: list[str] = []
    level = source_level(pdf, source_text, extra_text)
    target_chars = normalized_target_chars(target_chars)
    quality_first = analysis is not None
    if quality_first and not analysis.complete:
        raise ArticleGenerationError(
            analysis.error or "结构化论文分析尚未完成，已停止生成以避免产生无证据的完成稿。"
        )
    if quality_first:
        warnings.append("已基于带页码/图号的结构化分析生成深度稿。")
    elif pdf:
        warnings.append("已基于 PDF 全文/图注生成深度解读稿；发布前仍建议核对关键数据和截图。")
    elif source_text.strip() or extra_text.strip():
        warnings.append("未提供 PDF 全文，已基于用户粘贴材料生成材料级解读；不要把它表述为全文级结论。")
    else:
        warnings.append("未提供 PDF 全文或额外正文，已基于题录/摘要生成摘要级解读；证据边界较窄。")

    using_confirmed_figures = confirmed_figures is not None
    if using_confirmed_figures:
        figures = analyze_confirmed_figures(
            paper,
            analysis,
            sorted(confirmed_figures or [], key=lambda item: (item.order or 999, item.figure_id))[:4],
            {"api_key": api_key, "base_url": base_url, "model": model, "image_assets": image_assets or {}},
        )
    else:
        figures = list(pdf.legends[:4]) if pdf else []

    if quality_first and not api_key.strip():
        raise ArticleGenerationError("未配置模型 API Key，无法把结构化分析生成质量优先公众号稿。")

    if api_key.strip():
        try:
            raw = call_openai_compatible(
                api_key=api_key,
                base_url=base_url,
                model=model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=(
                    build_analysis_article_prompt(paper, analysis, figures, target_chars)
                    if analysis
                    else build_prompt(paper, pdf, target_chars, source_text=source_text, extra_text=extra_text)
                ),
            )
            data = parse_json_object(raw)
            if analysis and not all(
                data.get(field)
                for field in (
                    "intro",
                    "research_question",
                    "approach_advantage",
                    "experiment_validation",
                    "quantitative_findings",
                    "innovation",
                    "limitations",
                    "take_home",
                )
            ):
                raise ValueError("模型返回的深度稿缺少必要章节")
        except Exception as exc:
            if quality_first:
                raise ArticleGenerationError(
                    _friendly_llm_failure(exc, "质量优先稿件生成失败，结构化分析已保留")
                ) from exc
            warnings.append(_friendly_llm_failure(exc, "LLM 生成失败，已使用通用保守模板"))
            data = fallback_article(paper, pdf, source_text=source_text, extra_text=extra_text)
    else:
        warnings.append("未填写 LLM API Key，已生成通用占位级模板稿；深度解读需要配置模型后重新生成。")
        data = fallback_article(paper, pdf, source_text=source_text, extra_text=extra_text)

    if not using_confirmed_figures:
        attach_interpretations(figures, data)
    lead_image = pdf.lead_image if pdf else None
    if using_confirmed_figures:
        markdown = render_article_with_confirmed_figures(paper, data, figures, lead_image=lead_image)
    else:
        markdown = render_markdown(paper, data, figures, lead_image=lead_image)
    count = chineseish_len(markdown)
    minimum = TARGET_MIN
    maximum = TARGET_MAX
    if count < minimum or count > maximum:
        warnings.append(f"当前字数 {count}，不在 {minimum}-{maximum} 发布目标内；请使用显式补写/精简操作，不会自动发起第二次模型调用。")
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
        analysis_version=analysis.version if analysis else "",
        source_hash=(analysis.source_hash if analysis else (pdf.hash if pdf else "")),
        lead_image_name=lead_image.image_name if lead_image else "",
    )
