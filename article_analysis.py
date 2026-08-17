from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, MutableMapping
from typing import Any

from .llm import call_openai_compatible, parse_json_object
from .models import AnalysisClaim, PaperAnalysis, PaperInput
from .pdf_reader import PdfContent


ANALYSIS_PROMPT_VERSION = "paper-analysis-v1"
ANALYSIS_FIELDS = (
    "research_question",
    "background",
    "methods",
    "key_results",
    "innovation",
    "limitations",
    "conclusion",
)

ANALYSIS_SYSTEM_PROMPT = """你是严谨的科研论文分析员。只根据提供的全文、图注和证据提取结论。
每一条分析都必须给出原文页码 page 或图号 figure_id，最好同时给出简短 evidence_text。
没有可追溯来源的判断不得输出；材料不足时返回空数组，不得补写常识或生成占位结论。
只返回符合要求的 JSON，不要输出 Markdown 代码块。"""


def analysis_cache_key(
    paper: PaperInput,
    pdf: PdfContent,
    model_config: Mapping[str, Any] | None = None,
) -> str:
    config = model_config or {}
    payload = {
        "version": ANALYSIS_PROMPT_VERSION,
        "source_hash": pdf.hash or hashlib.sha256(pdf.text.encode("utf-8")).hexdigest(),
        "paper_key": paper.paper_key,
        "model": str(config.get("model") or "gpt-4o-mini"),
        "base_url": str(config.get("base_url") or "https://api.openai.com/v1").rstrip("/"),
        "quality": pdf.quality,
        "coverage": pdf.coverage,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_analysis_prompt(paper: PaperInput, pdf: PdfContent) -> str:
    return f"""
请对下面单篇论文执行完整、可追溯的结构化分析。

论文信息：
题名：{paper.title_en or paper.title}
中文题名：{paper.title_zh}
期刊：{paper.journal}
DOI：{paper.doi}

输出 JSON Schema：
{{
  "research_question": [{{"statement":"研究问题", "page":"1", "figure_id":"", "evidence_text":"原文证据", "confidence":"high"}}],
  "background": [{{"statement":"背景", "page":"1", "figure_id":"", "evidence_text":"原文证据", "confidence":"medium"}}],
  "methods": [{{"statement":"方法", "page":"3", "figure_id":"Fig. 1", "evidence_text":"原文证据", "confidence":"high"}}],
  "key_results": [{{"statement":"关键结果", "page":"5", "figure_id":"Fig. 2", "evidence_text":"原文证据", "confidence":"high"}}],
  "innovation": [{{"statement":"创新点", "page":"7", "figure_id":"", "evidence_text":"原文证据", "confidence":"medium"}}],
  "limitations": [{{"statement":"局限性", "page":"8", "figure_id":"", "evidence_text":"原文证据", "confidence":"medium"}}],
  "conclusion": [{{"statement":"结论", "page":"8", "figure_id":"", "evidence_text":"原文证据", "confidence":"high"}}]
}}

全文证据包：
{pdf.prompt_text(max_chars=30000)}
""".strip()


def _claim_from_payload(item: Any) -> AnalysisClaim | None:
    if isinstance(item, str):
        return None
    if not isinstance(item, dict):
        return None
    statement = str(item.get("statement") or item.get("claim") or "").strip()
    if not statement:
        return None
    claim = AnalysisClaim(
        statement=statement,
        page=str(item.get("page") or "").strip(),
        figure_id=str(item.get("figure_id") or "").strip(),
        evidence_text=str(item.get("evidence_text") or item.get("evidence") or "").strip(),
        confidence=str(item.get("confidence") or "medium").strip().lower(),
    )
    return claim if claim.traceable else None


def paper_analysis_from_payload(
    payload: dict[str, Any],
    *,
    source_hash: str,
    model: str,
) -> PaperAnalysis:
    values: dict[str, Any] = {}
    dropped = 0
    for field_name in ANALYSIS_FIELDS:
        claims: list[AnalysisClaim] = []
        for item in payload.get(field_name) or []:
            claim = _claim_from_payload(item)
            if claim:
                claims.append(claim)
            else:
                dropped += 1
        values[field_name] = claims
    analysis = PaperAnalysis(
        **values,
        status="complete",
        source_hash=source_hash,
        model=model,
        version=ANALYSIS_PROMPT_VERSION,
    )
    if dropped:
        analysis.warnings.append(f"已忽略 {dropped} 条缺少页码/图号或正文的不可追溯判断。")
    if not analysis.research_question or not analysis.key_results or not analysis.conclusion:
        raise ValueError("结构化分析缺少研究问题、关键结果或结论的可追溯证据")
    return analysis


def _cached_analysis(cache: MutableMapping[str, Any] | None, key: str) -> PaperAnalysis | None:
    if cache is None or key not in cache:
        return None
    value = cache[key]
    if isinstance(value, PaperAnalysis):
        return PaperAnalysis.from_dict(value.to_dict())
    if isinstance(value, dict):
        return PaperAnalysis.from_dict(value)
    return None


def analyze_paper(
    paper: PaperInput,
    pdf: PdfContent,
    model_config: Mapping[str, Any] | None = None,
    previous_analysis: PaperAnalysis | None = None,
) -> PaperAnalysis:
    config = model_config or {}
    api_key = str(config.get("api_key") or "")
    base_url = str(config.get("base_url") or "https://api.openai.com/v1")
    model = str(config.get("model") or "gpt-4o-mini")
    cache = config.get("cache")
    cache_mapping = cache if isinstance(cache, MutableMapping) else None
    key = analysis_cache_key(paper, pdf, config)
    cached = _cached_analysis(cache_mapping, key)
    if cached and cached.complete:
        return cached
    source_hash = pdf.hash or hashlib.sha256(pdf.text.encode("utf-8")).hexdigest()
    if not api_key.strip():
        return PaperAnalysis(
            status="failed",
            error="未配置模型 API Key，无法执行全文结构化分析。",
            source_hash=source_hash,
            model=model,
            version=ANALYSIS_PROMPT_VERSION,
        )
    try:
        raw = call_openai_compatible(
            api_key=api_key,
            base_url=base_url,
            model=model,
            system_prompt=ANALYSIS_SYSTEM_PROMPT,
            user_prompt=build_analysis_prompt(paper, pdf),
            temperature=0.1,
        )
        analysis = paper_analysis_from_payload(parse_json_object(raw), source_hash=source_hash, model=model)
    except Exception as exc:
        if previous_analysis and previous_analysis.complete:
            preserved = PaperAnalysis.from_dict(previous_analysis.to_dict())
            preserved.warnings.append(f"重新分析失败，已保留上一次完整分析：{exc}")
            return preserved
        return PaperAnalysis(
            status="failed",
            error=f"论文分析失败：{exc}",
            source_hash=source_hash,
            model=model,
            version=ANALYSIS_PROMPT_VERSION,
        )
    if cache_mapping is not None:
        cache_mapping[key] = analysis.to_dict()
    return analysis
