from __future__ import annotations

import json
import mimetypes
from collections.abc import Mapping
from typing import Any

from .llm import call_openai_compatible, call_openai_compatible_with_images, parse_json_object
from .models import AnalysisClaim, FigureAnalysis, PaperAnalysis, PaperInput
from .pdf_reader import compact_text, figure_key


FIGURE_ANALYSIS_PROMPT_VERSION = "figure-analysis-v2"
ROLE_LABELS = {
    "lead": "论文首页",
    "mechanism": "机制图",
    "method": "方法图",
    "key_result": "关键结果图",
    "validation": "验证图",
}


FIGURE_ANALYSIS_SYSTEM_PROMPT = """你是严谨的科研论文配图解读助手。
只根据用户提供的图号、页码、图注、全文证据、结构化分析，以及（仅当明确提供）实际图像，为已确认配图生成中文图解。
每张图的 note 必须包含四个清晰部分：图展示什么、实验或比较如何设计、关键数据或趋势、该图支持的结论与证据边界。
没有提供图像时，绝不声称看到了曲线、坐标轴、显著性标记或多面板细节；必须标明为“图注/文本证据级解读（未完成视觉复核）”。
提供图像且完成视觉复核时，也只能描述可见内容和给定证据共同支持的结论，不得补造数值。
每张图必须返回 figure_id、heading、note、evidence_text、page。没有证据的图不要编造。
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
    evidence = _evidence_text(figure, claims)
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


def _four_part_note(
    figure: FigureAnalysis,
    analysis: PaperAnalysis | None,
    *,
    visual_reviewed: bool,
    model_note: str = "",
) -> str:
    shown, design, trend, boundary = _evidence_parts(figure, analysis)
    source_label = "视觉复核已完成（Gemini 图像输入）" if visual_reviewed else "图注/文本证据级解读（未完成视觉复核）"
    if model_note.strip():
        # Keep a model's useful prose, but make the required evidence frame
        # explicit so readers can distinguish observation from text evidence.
        shown = f"{shown}。补充解读：{compact_text(model_note, 700)}"
    return (
        f"**证据级别：{source_label}**\n\n"
        f"**图展示什么：**{shown}\n\n"
        f"**实验或比较如何设计：**{design}\n\n"
        f"**关键数据或趋势：**{trend}\n\n"
        f"**该图支持的结论与证据边界：**{boundary}"
    )


def _fallback_figure_note(figure: FigureAnalysis, analysis: PaperAnalysis | None = None) -> str:
    if not _has_figure_evidence(figure, analysis):
        figure.interpretation = ""
        figure.needs_manual_check = True
        return ""
    figure.interpretation = _four_part_note(figure, analysis, visual_reviewed=False)
    figure.needs_manual_check = True
    return figure.interpretation


def _build_prompt(
    paper: PaperInput,
    analysis: PaperAnalysis | None,
    figures: list[FigureAnalysis],
    *,
    visual_review_available: bool,
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
            }
        )
    return f"""
请为下面已确认选入公众号正文的论文配图生成逐图中文分析。

论文：{paper.title_zh or paper.title_en or paper.title}
DOI：{paper.doi}

输出 JSON Schema：
{{
  "figures": [
    {{"figure_id":"Fig. 1", "heading":"图文小标题", "note":"中文图解分析", "evidence_text":"可核对证据", "page":"1"}}
  ]
}}

要求：
1. 只分析下面 confirmed_figures 中列出的图。
2. note 必须分为“图展示什么 / 实验或比较如何设计 / 关键数据或趋势 / 该图支持的结论与证据边界”四部分，每部分均须可由图像或文本证据追溯。
3. 不要添加图注和证据中没有的数字或实验细节。
4. 当前视觉复核状态：{'已提供确认图像，可结合图像复核' if visual_review_available else '未提供或不可使用图像；只能基于图注和文本证据，必须明确未完成视觉复核'}。
5. confidence 低或 needs_manual_crop=true 时，提醒发布前人工核对截图。

confirmed_figures：
{json.dumps(items, ensure_ascii=False, indent=2)}
""".strip()


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
        note = str(item.get("note") or item.get("interpretation") or "").strip()
        page = str(item.get("page") or figure.page or "").strip()
        evidence = str(item.get("evidence_text") or item.get("evidence") or "").strip()
        if not note or not (page or figure.figure_id) or not evidence or not _has_figure_evidence(figure, analysis):
            continue
        heading = str(item.get("heading") or "").strip()
        if heading:
            figure.why_selected = heading
        figure.interpretation = _four_part_note(
            figure,
            analysis,
            visual_reviewed=visual_reviewed,
            model_note=note,
        )
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


def analyze_confirmed_figures(
    paper: PaperInput,
    analysis: PaperAnalysis | None,
    figures: list[FigureAnalysis],
    model_config: Mapping[str, Any] | None = None,
) -> list[FigureAnalysis]:
    confirmed = [
        figure
        for figure in sorted(figures, key=lambda item: (item.order or 999, item.figure_id))
        if figure.selected and figure.image_name and _has_figure_evidence(figure, analysis)
    ][:4]
    if not confirmed:
        return []

    config = model_config or {}
    api_key = str(config.get("api_key") or "")
    base_url = str(config.get("base_url") or "https://api.openai.com/v1")
    model = str(config.get("model") or "gpt-4o-mini")
    image_assets = config.get("image_assets")
    assets = image_assets if isinstance(image_assets, Mapping) else {}
    images = _image_inputs(confirmed, assets)
    visual_review_available = bool(images) and _is_gemini_vision(config, model, base_url)
    if api_key.strip():
        try:
            call_kwargs = {
                "api_key": api_key,
                "base_url": base_url,
                "model": model,
                "system_prompt": FIGURE_ANALYSIS_SYSTEM_PROMPT,
                "user_prompt": _build_prompt(
                    paper,
                    analysis,
                    confirmed,
                    visual_review_available=visual_review_available,
                ),
                "temperature": 0.1,
            }
            raw = (
                call_openai_compatible_with_images(images=images, **call_kwargs)
                if visual_review_available
                else call_openai_compatible(**call_kwargs)
            )
            payload = parse_json_object(raw)
            _apply_payload(confirmed, payload, analysis, visual_reviewed=visual_review_available)
        except Exception:
            # A failed vision/text request must not masquerade as image review.
            visual_review_available = False

    for figure in confirmed:
        if not figure.interpretation:
            _fallback_figure_note(figure, analysis)
    return [figure for figure in confirmed if figure.interpretation]
