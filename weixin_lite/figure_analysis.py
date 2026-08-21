from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .llm import call_openai_compatible, parse_json_object
from .models import AnalysisClaim, FigureAnalysis, PaperAnalysis, PaperInput
from .pdf_reader import compact_text, figure_key


FIGURE_ANALYSIS_PROMPT_VERSION = "figure-analysis-v1"
ROLE_LABELS = {
    "lead": "论文首页",
    "mechanism": "机制图",
    "method": "方法图",
    "key_result": "关键结果图",
    "validation": "验证图",
}


FIGURE_ANALYSIS_SYSTEM_PROMPT = """你是严谨的科研论文配图解读助手。
只根据用户提供的图号、页码、图注、全文证据和结构化分析，为已确认配图生成中文图解。
不要声称你看到了图片中未提供的视觉细节；如果证据来自图注，请明确基于图注和原文证据。
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


def _fallback_figure_note(figure: FigureAnalysis, analysis: PaperAnalysis | None = None) -> str:
    claims = _claims_for_figure(analysis, figure)
    evidence = _evidence_text(figure, claims)
    if not evidence:
        figure.interpretation = ""
        figure.needs_manual_check = True
        return ""
    role = _role_label(figure)
    source = "图注、页码和全文证据"
    if claims:
        claim_text = "；".join(claim.statement for claim in claims[:2])
        note = f"这张{role}对应原文中的{claim_text}。基于{source}，它适合放在正文中承接相关分析，帮助读者把图中的证据与文章主线对应起来。"
    else:
        caption = compact_text(figure.caption, 220)
        note = f"这张{role}的可核对信息主要来自图注：{caption}。当前未做额外视觉判断，发布前建议人工核对截图是否完整清晰。"
    figure.interpretation = note
    return note


def _build_prompt(paper: PaperInput, analysis: PaperAnalysis | None, figures: list[FigureAnalysis]) -> str:
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
2. note 写 1-3 句，解释这张图在文章逻辑中的作用。
3. 不要添加图注和证据中没有的数字或实验细节。
4. confidence 低或 needs_manual_crop=true 时，提醒发布前人工核对截图。

confirmed_figures：
{json.dumps(items, ensure_ascii=False, indent=2)}
""".strip()


def _apply_payload(figures: list[FigureAnalysis], payload: dict[str, Any]) -> bool:
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
        if not note or not (page or figure.figure_id) or not evidence:
            continue
        heading = str(item.get("heading") or "").strip()
        if heading:
            figure.why_selected = heading
        figure.interpretation = note
        applied = True
    return applied


def analyze_confirmed_figures(
    paper: PaperInput,
    analysis: PaperAnalysis | None,
    figures: list[FigureAnalysis],
    model_config: Mapping[str, Any] | None = None,
) -> list[FigureAnalysis]:
    confirmed = [
        figure
        for figure in sorted(figures, key=lambda item: (item.order or 999, item.figure_id))
        if figure.selected and figure.image_name and (figure.caption or figure.evidence or figure.page)
    ][:4]
    if not confirmed:
        return []

    config = model_config or {}
    api_key = str(config.get("api_key") or "")
    base_url = str(config.get("base_url") or "https://api.openai.com/v1")
    model = str(config.get("model") or "gpt-4o-mini")
    if api_key.strip():
        try:
            raw = call_openai_compatible(
                api_key=api_key,
                base_url=base_url,
                model=model,
                system_prompt=FIGURE_ANALYSIS_SYSTEM_PROMPT,
                user_prompt=_build_prompt(paper, analysis, confirmed),
                temperature=0.1,
            )
            payload = parse_json_object(raw)
            _apply_payload(confirmed, payload)
        except Exception:
            pass

    for figure in confirmed:
        if not figure.interpretation:
            _fallback_figure_note(figure, analysis)
    return [figure for figure in confirmed if figure.interpretation]
