from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, MutableMapping
from typing import Any

from .llm import call_openai_compatible, friendly_llm_error, parse_json_object
from .models import AnalysisClaim, PaperAnalysis, PaperInput
from .pdf_reader import PdfContent


ANALYSIS_PROMPT_VERSION = "paper-analysis-v5-overview"
ANALYSIS_FIELDS = (
    "research_question",
    "background",
    "methods",
    "key_results",
    "innovation",
    "limitations",
    "conclusion",
)


def _legacy_safe_analysis_chunks(pdf: PdfContent, max_chars: int = 18000) -> list[str]:
    """Read every available section even when Streamlit restored an older PdfContent."""
    chunk_builder = getattr(pdf, "analysis_chunks", None)
    if callable(chunk_builder):
        return list(chunk_builder(max_chars=max_chars))

    text = str(getattr(pdf, "text", "") or "")
    sections = getattr(pdf, "sections", {}) or {}
    sources = {"full_text": text} if text.strip() else (sections if isinstance(sections, dict) else {})
    legends = getattr(pdf, "all_figures", None) or getattr(pdf, "legends", []) or []
    captions = "\n".join(
        f"- {getattr(item, 'figure_id', '')} p.{getattr(item, 'page', '')}: {getattr(item, 'caption', '')}"
        for item in legends
    )
    chunks: list[str] = []
    for name, source in sources.items():
        value = str(source or "").strip()
        for offset in range(0, len(value), max_chars):
            chunks.append(f"## {name}\n{value[offset:offset + max_chars]}\n\nFigure/Table captions:\n{captions}")
    return chunks or ["未能从 PDF 中提取正文。"]

ANALYSIS_SYSTEM_PROMPT = """你是该领域的世界顶级学术专家，正在为中文读者提炼一篇论文的可靠概览。
你必须只根据提供的全文、图注和证据提取结论，帮助读者理解研究主线，但不要穷举实验、对照、样本、参数或全部定量结果。
遇到相对新颖或专业的技术概念，首次出现时在 statement 中用 **术语** 标出，并给出通俗解释；学术名词可保留英文补充。
每一条分析都必须给出原文页码 page 或图号 figure_id，并在 evidence_text 中放入可核对的原文短引文或原文细节；可引用时使用 blockquote 风格的 `> 原文`。
没有可追溯来源的判断不得输出；材料不足时返回空数组，不得补写常识或生成占位结论。
概览必须覆盖研究目标、现实或产业意义、方法原理与主要流程、创新点、局限性和总体结论。methods 只保留 1 至 3 条简要方法概述；key_results 最多保留 1 至 2 条支撑论文总体结论的概括性结果，不展开具体实验和数据，详细结果留给用户选中的图片分析。
只返回符合要求的 JSON，不要输出 Markdown 代码块。"""


ANALYSIS_READING_GUIDE = """
论文概览要求：
你现在作为该领域的世界顶级学术专家，想详细阅读并深入这篇论文。
充分阅读当前证据，但输出只保留论文主线。当前分段没有的证据返回空数组，由后续分段合并。
如果技术概念相对新颖，请给出通俗解释。不要整理完整实验清单，也不要逐项罗列关键数据。

请围绕以下问题组织概览，但仍按下方 JSON Schema 输出：
### 论文的研究目标是什么？想要解决什么实际问题？
对应 research_question；说明论文要解决的核心科学/技术问题，以及现实痛点。
### 这个问题对于产业发展有什么重要意义？
对应 background 或 innovation；分析其对产业、转化、生产、诊疗、平台化或工程应用的价值。
### 论文采用了什么方法？
对应 methods；只概述核心原理、主要流程和相对优势，不写逐步实验设计、对照、样本或参数清单。
### 论文的创新、局限与总体结论是什么？
对应 innovation、limitations 和 conclusion；key_results 只保留少量总体结论，不展开具体数字或全部结果。

格式约束：
- 使用中文书写，学术名词可以用英文补充。
- 关键术语首次出现时用 **加粗**。
- evidence_text 中引用原文时使用 blockquote 风格，例如 `> original sentence`，并保持短引文。
- 概览不负责逐图分析；只有确实用于总体结论时才填写 figure_id。
- 只返回符合要求的 JSON，不要输出 Markdown 代码块。
""".strip()


def analysis_cache_key(
    paper: PaperInput,
    pdf: PdfContent,
    model_config: Mapping[str, Any] | None = None,
) -> str:
    config = model_config or {}
    payload = {
        "version": ANALYSIS_PROMPT_VERSION,
        "source_hash": getattr(pdf, "hash", "") or hashlib.sha256(str(getattr(pdf, "text", "")).encode("utf-8")).hexdigest(),
        "paper_key": paper.paper_key,
        "model": str(config.get("model") or "gpt-4o-mini"),
        "base_url": str(config.get("base_url") or "https://api.openai.com/v1").rstrip("/"),
        "quality": getattr(pdf, "quality", "unknown"),
        "coverage": getattr(pdf, "coverage", []),
        "evidence_hash": hashlib.sha256(json.dumps(_legacy_safe_analysis_chunks(pdf), ensure_ascii=False).encode("utf-8")).hexdigest(),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_analysis_prompt(paper: PaperInput, pdf: PdfContent, evidence_chunk: str | None = None) -> str:
    return f"""
请对下面单篇论文执行简洁、可追溯的结构化概览分析。

论文信息：
题名：{paper.title_en or paper.title}
中文题名：{paper.title_zh}
期刊：{paper.journal}
DOI：{paper.doi}

{ANALYSIS_READING_GUIDE}

输出 JSON Schema：
{{
  "research_question": [{{"statement":"研究问题", "page":"1", "figure_id":"", "evidence_text":"原文证据", "confidence":"high"}}],
  "background": [{{"statement":"背景", "page":"1", "figure_id":"", "evidence_text":"原文证据", "confidence":"medium"}}],
  "methods": [{{"statement":"方法概述，不超过三条", "page":"3", "figure_id":"", "evidence_text":"原文证据", "confidence":"high"}}],
  "key_results": [{{"statement":"支撑总体结论的概括性结果，最多两条", "page":"5", "figure_id":"", "evidence_text":"原文证据", "confidence":"high"}}],
  "innovation": [{{"statement":"创新点", "page":"7", "figure_id":"", "evidence_text":"原文证据", "confidence":"medium"}}],
  "limitations": [{{"statement":"局限性", "page":"8", "figure_id":"", "evidence_text":"原文证据", "confidence":"medium"}}],
  "conclusion": [{{"statement":"结论", "page":"8", "figure_id":"", "evidence_text":"原文证据", "confidence":"high"}}]
}}

全文证据包：
{evidence_chunk if evidence_chunk is not None else pdf.prompt_text()}
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
    # A page/figure reference without a source excerpt is not sufficient to
    # support a publishable claim. Keep the evidence adjacent to its locator.
    return claim if claim.traceable and claim.evidence_text else None


def paper_analysis_from_payload(
    payload: dict[str, Any],
    *,
    source_hash: str,
    model: str,
    require_core: bool = True,
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
    required = {
        "研究问题": analysis.research_question,
        "现实/产业意义": analysis.background,
        "方法": analysis.methods,
        "创新点": analysis.innovation,
        "局限性": analysis.limitations,
        "结论": analysis.conclusion,
    }
    missing = [label for label, claims in required.items() if not claims]
    if require_core and missing:
        raise ValueError("结构化分析缺少以下可追溯证据：" + "、".join(missing))
    return analysis


def merge_analyses(analyses: list[PaperAnalysis], *, source_hash: str, model: str) -> PaperAnalysis:
    values: dict[str, list[AnalysisClaim]] = {name: [] for name in ANALYSIS_FIELDS}
    seen: set[tuple[str, str, str]] = set()
    for partial in analyses:
        for name in ANALYSIS_FIELDS:
            for claim in getattr(partial, name):
                key = (name, claim.page, claim.statement)
                if key not in seen:
                    seen.add(key)
                    values[name].append(claim)
    limits = {
        "research_question": 3,
        "background": 3,
        "methods": 3,
        "key_results": 2,
        "innovation": 3,
        "limitations": 3,
        "conclusion": 2,
    }
    values = {name: claims[: limits[name]] for name, claims in values.items()}
    merged = PaperAnalysis(**values, status="complete", source_hash=source_hash, model=model, version=ANALYSIS_PROMPT_VERSION)
    required = (merged.research_question, merged.background, merged.methods, merged.innovation, merged.limitations, merged.conclusion)
    if not all(required):
        raise ValueError("结构化分析缺少完整的可追溯证据，无法生成最终稿。")
    return merged


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
    source_hash = getattr(pdf, "hash", "") or hashlib.sha256(str(getattr(pdf, "text", "")).encode("utf-8")).hexdigest()
    readable_text = str(getattr(pdf, "text", "") or "") or " ".join(str(value) for value in (getattr(pdf, "sections", {}) or {}).values())
    if not re.sub(r"\[Page\s+\d+\]", "", readable_text).strip():
        return PaperAnalysis(status="failed", error="未提取到可分析正文，请上传文字型 PDF 或先完成 OCR。", source_hash=source_hash, model=model)
    if not api_key.strip():
        return PaperAnalysis(
            status="failed",
            error="未配置模型 API Key，无法执行论文概览分析。",
            source_hash=source_hash,
            model=model,
            version=ANALYSIS_PROMPT_VERSION,
        )
    chunks: list[str] = []
    for chunk in _legacy_safe_analysis_chunks(pdf):
        if chunks and len(chunks[-1]) + len(chunk) + 2 <= 18000:
            chunks[-1] += "\n\n" + chunk
        else:
            chunks.append(chunk)
    partials: list[PaperAnalysis] = []
    progress = config.get("progress_callback")
    try:
        for index, chunk in enumerate(chunks):
            chunk_key = key + f":chunk:{index}:" + hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            partial = _cached_analysis(cache_mapping, chunk_key)
            if partial is not None:
                partials.append(partial)
                if callable(progress):
                    progress(index + 1, len(chunks))
                continue
            raw = call_openai_compatible(
                api_key=api_key,
                base_url=base_url,
                model=model,
                system_prompt=ANALYSIS_SYSTEM_PROMPT,
                user_prompt=build_analysis_prompt(paper, pdf, chunk),
                temperature=0.1,
            )
            partial = paper_analysis_from_payload(parse_json_object(raw), source_hash=source_hash, model=model, require_core=False)
            partials.append(partial)
            if cache_mapping is not None:
                cache_mapping[chunk_key] = partial.to_dict()
            if callable(progress):
                progress(index + 1, len(chunks))
        analysis = merge_analyses(partials, source_hash=source_hash, model=model)
        analysis.completed_chunks = len(chunks)
        analysis.total_chunks = len(chunks)
    except Exception as exc:
        if (
            previous_analysis
            and previous_analysis.complete
            and previous_analysis.source_hash == source_hash
            and previous_analysis.version == ANALYSIS_PROMPT_VERSION
        ):
            preserved = PaperAnalysis.from_dict(previous_analysis.to_dict())
            preserved.warnings.append("重新分析失败，已保留上一次完整分析：" + friendly_llm_error(exc))
            return preserved
        return PaperAnalysis(
            status="failed",
            error="论文概览分析暂停：" + friendly_llm_error(exc),
            completed_chunks=len(partials),
            total_chunks=len(chunks),
            **{name: [claim for partial in partials for claim in getattr(partial, name)] for name in ANALYSIS_FIELDS},
            source_hash=source_hash,
            model=model,
            version=ANALYSIS_PROMPT_VERSION,
        )
    if cache_mapping is not None:
        cache_mapping[key] = analysis.to_dict()
    return analysis
