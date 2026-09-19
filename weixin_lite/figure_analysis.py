from __future__ import annotations

import json
import hashlib
import mimetypes
import re
from collections.abc import Mapping, MutableMapping
from typing import Any

from .llm import call_openai_compatible, call_openai_compatible_with_images, friendly_llm_error, parse_json_object
from .models import AnalysisClaim, FigureAnalysis, PaperAnalysis, PaperInput
from .pdf_reader import PdfContent, compact_text, figure_key


FIGURE_ANALYSIS_PROMPT_VERSION = "figure-analysis-v4-selected-three-part"
ROLE_LABELS = {
    "lead": "论文首页",
    "mechanism": "机制图",
    "method": "方法图",
    "key_result": "关键结果图",
    "validation": "验证图",
}


FIGURE_ANALYSIS_SYSTEM_PROMPT = """你是严谨的科研论文配图解读助手。
只根据用户提供的图号、页码、图注、全文证据、结构化分析，以及（仅当明确提供）实际图像，为已确认配图生成中文图解。
每张图只返回三个内容部分：overview 说明图展示什么并可用一句话交代必要的方法背景；key_findings 提炼图中可核对的关键数据或趋势；conclusion_boundary 说明图支持的结论和证据边界。不要单列或展开实验设计。
没有提供图像时，绝不声称看到了曲线、坐标轴、显著性标记或多面板细节；必须标明为“图注/文本证据级解读（未完成视觉复核）”。
提供图像且完成视觉复核时，也只能描述可见内容和给定证据共同支持的结论，不得补造数值。
每张图必须返回 figure_id、heading、overview、key_findings、conclusion_boundary、evidence_text、page、visible_elements、visual_evidence。没有证据的图不要编造。
当 asset_kind 为 table 时，额外返回 table_headers、table_rows、table_confidence；只能转录确实看得清的单元格，不能猜测缺失数字。
当 asset_kind 为 scheme 或 flowchart 时，overview 必须明确流程步骤、箭头关系、输入和输出；看不清时写明证据边界。
只返回 JSON，不要返回 Markdown 代码块。"""


def _claims_for_figure(analysis: PaperAnalysis | None, figure: FigureAnalysis) -> list[AnalysisClaim]:
    if not analysis:
        return []
    key = figure_key(figure.figure_id)
    if not key:
        return []
    return [claim for claim in analysis.claims if figure_key(claim.figure_id) == key]


def _role_label(figure: FigureAnalysis) -> str:
    return ROLE_LABELS.get((figure.role or "").strip(), "关键图")


def figure_heading(figure: FigureAnalysis) -> str:
    label = _role_label(figure)
    return f"{figure.figure_id}：{label}"


def _evidence_text(figure: FigureAnalysis, claims: list[AnalysisClaim]) -> str:
    parts: list[str] = []
    for claim in claims[:3]:
        source = " / ".join(bit for bit in [f"p.{claim.page}" if claim.page else "", claim.figure_id] if bit)
        parts.append(f"{claim.statement}（{source}）：{claim.evidence_text}")
    for item in figure.evidence[:4]:
        label = item.label()
        parts.append(f"{item.claim}{f'（{label}）' if label else ''}")
    if figure.caption:
        parts.append(figure.caption)
    return compact_text("\n".join(part for part in parts if part), 1800)


def _has_figure_evidence(figure: FigureAnalysis, analysis: PaperAnalysis | None) -> bool:
    return bool(figure.caption.strip() or figure.evidence or _claims_for_figure(analysis, figure))


def _evidence_parts(figure: FigureAnalysis, analysis: PaperAnalysis | None) -> tuple[str, str, str, str]:
    claims = _claims_for_figure(analysis, figure)
    caption = compact_text(figure.caption, 420)
    result_claims = [claim for claim in claims if claim in (analysis.key_results if analysis else [])]
    method_claims = [claim for claim in claims if claim in (analysis.methods if analysis else [])]
    evidence_items = [item for item in figure.evidence if item.claim]

    shown = caption or (claims[0].statement if claims else "原文未提供可复述的图注内容")
    if method_claims:
        design = "；".join(claim.statement for claim in method_claims[:2])
    elif evidence_items:
        design = "；".join(item.claim for item in evidence_items[:2])
    else:
        design = "图注和已提取正文未给出完整实验设计或对照信息，不能据此补充。"
    if result_claims:
        trend = "；".join(claim.statement for claim in result_claims[:2])
    else:
        numeric = [item for item in evidence_items if item.value]
        trend = "；".join(f"{item.claim}：{item.value}" for item in numeric[:2]) or "图注/文本证据未提供可核对的定量结果或趋势。"
    boundary = (
        "；".join(claim.statement for claim in claims[:2])
        if claims
        else f"这张{_role_label(figure)}可作为原文证据定位，但其具体结论仍需结合图注和正文核对。"
    )
    return shown, design, trend, boundary


def _three_part_note(
    figure: FigureAnalysis,
    analysis: PaperAnalysis | None,
    *,
    visual_reviewed: bool,
    model_note: str = "",
    model_findings: str = "",
    model_boundary: str = "",
) -> str:
    shown, _design, trend, boundary = _evidence_parts(figure, analysis)
    source_label = "视觉复核已完成（Gemini 图像输入）" if visual_reviewed else "图注/文本证据级解读（未完成视觉复核）"
    if model_note.strip():
        shown = model_note.strip()
    if model_findings.strip():
        trend = model_findings.strip()
    if model_boundary.strip():
        boundary = model_boundary.strip()
    return (
        f"**证据级别：{source_label}**\n\n"
        f"**图展示什么：**{shown}\n\n"
        f"**关键数据或趋势：**{trend}\n\n"
        f"**结论与证据边界：**{boundary}"
    )


def _fallback_figure_note(figure: FigureAnalysis, analysis: PaperAnalysis | None = None) -> str:
    if not _has_figure_evidence(figure, analysis):
        figure.interpretation = ""
        figure.needs_manual_check = True
        return ""
    figure.interpretation = _three_part_note(figure, analysis, visual_reviewed=False)
    figure.needs_manual_check = True
    return figure.interpretation


def _build_prompt(
    paper: PaperInput,
    analysis: PaperAnalysis | None,
    figures: list[FigureAnalysis],
    *,
    visual_review_available: bool,
    pdf: PdfContent | None = None,
) -> str:
    items = []
    for figure in figures:
        claims = _claims_for_figure(analysis, figure)
        evidence = _evidence_text(figure, claims)
        items.append(
            {
                "figure_id": figure.figure_id,
                "role": figure.role or "key_result",
                "page": figure.page,
                "caption": compact_text(figure.caption, 1200),
                "evidence_text": evidence,
                "confidence": figure.confidence,
                "needs_manual_crop": figure.needs_manual_crop,
                "asset_kind": figure.asset_kind,
                "editable_table": figure.editable_table.to_dict() if figure.editable_table else None,
                "page_context": _page_context(pdf, figure),
            }
        )
    return f"""
请为下面已确认选入公众号正文的论文配图生成逐图中文分析。

论文：{paper.title_zh or paper.title_en or paper.title}
DOI：{paper.doi}

输出 JSON Schema：
{{
  "figures": [
    {{"figure_id":"Fig. 1", "heading":"图文小标题", "overview":"图展示什么及一句必要的方法背景", "key_findings":"可核对的关键数据或趋势", "conclusion_boundary":"图支持的结论与证据边界", "evidence_text":"可核对证据", "visible_elements":"可见的曲线、坐标轴、箭头或表格结构", "visual_evidence":"图像中可直接观察到的趋势或数值", "page":"1", "table_headers":["列名"], "table_rows":[["单元格"]], "table_confidence":0.0}}
  ]
}}

要求：
1. 只分析下面 confirmed_figures 中列出的图。
2. 只填写 overview、key_findings、conclusion_boundary 三部分；不要单列实验设计，方法背景最多一句。
3. 不要添加图注和证据中没有的数字或实验细节。
4. 当前视觉复核状态：{'已提供确认图像，可结合图像复核' if visual_review_available else '未提供或不可使用图像；只能基于图注和文本证据，必须明确未完成视觉复核'}。
5. confidence 低或 needs_manual_crop=true 时，提醒发布前人工核对截图。

confirmed_figures：
{json.dumps(items, ensure_ascii=False, indent=2)}
""".strip()


def _page_context(pdf: PdfContent | None, figure: FigureAnalysis, max_chars: int = 5000) -> str:
    if pdf is None or not str(getattr(pdf, "text", "") or "").strip():
        return ""
    try:
        page_number = int(figure.page)
    except (TypeError, ValueError):
        return ""
    text = str(pdf.text)
    match = re.search(
        rf"(?is)\[Page\s+{page_number}\]\s*(.*?)(?=\[Page\s+\d+\]|\Z)",
        text,
    )
    return compact_text(match.group(1), max_chars) if match else ""


def _apply_payload(
    figures: list[FigureAnalysis],
    payload: dict[str, Any],
    analysis: PaperAnalysis | None,
    *,
    visual_reviewed: bool,
) -> bool:
    by_key = {figure_key(figure.figure_id): figure for figure in figures}
    applied = False
    for item in payload.get("figures") or []:
        if not isinstance(item, dict):
            continue
        figure = by_key.get(figure_key(item.get("figure_id")))
        if not figure:
            continue
        note = str(item.get("overview") or item.get("note") or item.get("interpretation") or "").strip()
        findings = str(item.get("key_findings") or "").strip()
        boundary = str(item.get("conclusion_boundary") or "").strip()
        page = str(item.get("page") or figure.page or "").strip()
        evidence = str(item.get("evidence_text") or item.get("evidence") or "").strip()
        if not note or not (page or figure.figure_id) or not evidence or not _has_figure_evidence(figure, analysis):
            continue
        heading = str(item.get("heading") or "").strip()
        if heading:
            figure.why_selected = heading
        figure.visual_evidence = compact_text(
            "\n".join(
                value
                for value in (
                    str(item.get("visible_elements") or "").strip(),
                    str(item.get("visual_evidence") or "").strip(),
                )
                if value
            ),
            1200,
        )
        if figure.asset_kind == "table" and figure.editable_table:
            headers = item.get("table_headers")
            rows = item.get("table_rows")
            confidence = item.get("table_confidence")
            if isinstance(headers, list) and isinstance(rows, list) and isinstance(confidence, (int, float)) and confidence >= 0.7:
                normalized_headers = [str(value).strip() for value in headers if str(value).strip()]
                normalized_rows = [
                    [str(value).strip() for value in row][: len(normalized_headers)]
                    for row in rows
                    if isinstance(row, list) and normalized_headers
                ]
                if normalized_headers and normalized_rows:
                    figure.editable_table.headers = normalized_headers
                    figure.editable_table.rows = normalized_rows
                    figure.editable_table.confidence = float(confidence)
        figure.interpretation = _three_part_note(
            figure,
            analysis,
            visual_reviewed=visual_reviewed,
            model_note=note,
            model_findings=findings,
            model_boundary=boundary,
        )
        figure.review_version = FIGURE_ANALYSIS_PROMPT_VERSION
        figure.needs_manual_check = not visual_reviewed
        applied = True
    return applied


def _is_gemini_vision(config: Mapping[str, Any], model: str, base_url: str) -> bool:
    provider = str(config.get("provider") or "").strip().lower()
    return provider == "gemini" or model.strip().lower().startswith("gemini") or "generativelanguage.googleapis.com" in base_url.lower()


def _image_inputs(figures: list[FigureAnalysis], image_assets: Mapping[str, Any]) -> list[tuple[bytes, str]]:
    inputs: list[tuple[bytes, str]] = []
    for figure in figures:
        image = image_assets.get(figure.image_name)
        if not isinstance(image, bytes) or not image:
            return []
        mime_type = mimetypes.guess_type(figure.image_name)[0] or "image/png"
        inputs.append((image, mime_type))
    return inputs


def _review_cache_key(paper, analysis, figure, assets, model, base_url, pdf=None) -> str:
    fingerprint = json.dumps({
        "version": FIGURE_ANALYSIS_PROMPT_VERSION,
        "model": model,
        "base_url": base_url,
        "crop": figure.crop_bbox,
        "prompt": _build_prompt(paper, analysis, [figure], visual_review_available=True, pdf=pdf),
        "image": hashlib.sha256(assets[figure.image_name]).hexdigest(),
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()


def analyze_confirmed_figures(
    paper: PaperInput,
    analysis: PaperAnalysis | None,
    figures: list[FigureAnalysis],
    model_config: Mapping[str, Any] | None = None,
    *,
    pdf: PdfContent | None = None,
) -> list[FigureAnalysis]:
    confirmed = [
        figure
        for figure in sorted(figures, key=lambda item: (item.order or 999, item.figure_id))
        if figure.selected and figure.image_name and _has_figure_evidence(figure, analysis)
    ][:4]
    if not confirmed:
        return []

    config = model_config or {}
    if pdf is None:
        configured_pdf = config.get("pdf")
        if configured_pdf is not None:
            pdf = configured_pdf
    api_key = str(config.get("api_key") or "")
    base_url = str(config.get("base_url") or "https://api.openai.com/v1")
    model = str(config.get("model") or "gpt-4o-mini")
    image_assets = config.get("image_assets")
    assets = image_assets if isinstance(image_assets, Mapping) else {}
    images = _image_inputs(confirmed, assets)
    visual_review_available = bool(images) and _is_gemini_vision(config, model, base_url)
    if not visual_review_available or not api_key.strip():
        for figure in confirmed:
            figure.vision_status = "blocked"
            figure.vision_error = "最终图表分析需要配置支持视觉输入的 Gemini 模型，并提供确认截图。"
        return []
    cache = config.get("cache")
    cache = cache if isinstance(cache, MutableMapping) else {}
    keys = {}
    pending = []
    for figure in confirmed:
        key = _review_cache_key(paper, analysis, figure, assets, model, base_url, pdf)
        keys[figure.figure_id] = key
        if key in cache and _apply_payload([figure], cache[key], analysis, visual_reviewed=True):
            figure.vision_status = "reviewed"
            figure.vision_error = ""
        else:
            # Old interpretations must not certify a changed image or an empty response.
            figure.interpretation = ""
            figure.visual_evidence = ""
            figure.review_version = ""
            figure.vision_status = "pending"
            pending.append(figure)
    if not pending:
        return confirmed
    try:
        call_kwargs = {
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
            "system_prompt": FIGURE_ANALYSIS_SYSTEM_PROMPT,
            "user_prompt": _build_prompt(
                paper,
                analysis,
                pending,
                visual_review_available=visual_review_available,
                pdf=pdf,
            ),
            "temperature": 0.1,
        }
        raw = call_openai_compatible_with_images(images=_image_inputs(pending, assets), **call_kwargs)
        payload = parse_json_object(raw)
        _apply_payload(pending, payload, analysis, visual_reviewed=True)
    except Exception as exc:
        for figure in pending:
            figure.vision_status = "failed"
            figure.vision_error = "Gemini 视觉复核暂停：" + friendly_llm_error(exc)
        return [figure for figure in confirmed if figure not in pending]
    for figure in pending:
        if figure.interpretation:
            figure.vision_status = "reviewed"
            figure.vision_error = ""
            cache[keys[figure.figure_id]] = {"figures": [
                item for item in payload.get("figures", [])
                if isinstance(item, dict) and figure_key(item.get("figure_id")) == figure_key(figure.figure_id)
            ]}
            cache[_review_cache_key(paper, analysis, figure, assets, model, base_url, pdf)] = cache[keys[figure.figure_id]]
        else:
            figure.vision_status = "failed"
            figure.vision_error = "Gemini 未返回可追溯的图表解读。"
    return [figure for figure in confirmed if figure.vision_status == "reviewed"]
