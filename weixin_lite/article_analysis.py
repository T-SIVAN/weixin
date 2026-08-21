from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, MutableMapping
from typing import Any

from .llm import call_openai_compatible, parse_json_object
from .models import AnalysisClaim, PaperAnalysis, PaperInput
from .pdf_reader import PdfContent


ANALYSIS_PROMPT_VERSION = "paper-analysis-v2"
ANALYSIS_FIELDS = (
    "research_question",
    "background",
    "methods",
    "key_results",
    "innovation",
    "limitations",
    "conclusion",
)

ANALYSIS_SYSTEM_PROMPT = """你是该领域的世界顶级学术专家，正在详细阅读并深入解读一篇论文。
你必须只根据提供的全文、图注和证据提取结论，多引用论文中的细节内容、关键数据和实验结果，帮助中文读者理解论文主线。
遇到相对新颖或专业的技术概念，首次出现时在 statement 中用 **术语** 标出，并给出通俗解释；学术名词可保留英文补充。
每一条分析都必须给出原文页码 page 或图号 figure_id，并在 evidence_text 中放入可核对的原文短引文或原文细节；可引用时使用 blockquote 风格的 `> 原文`。
没有可追溯来源的判断不得输出；材料不足时返回空数组，不得补写常识或生成占位结论。
总体分析应足够深入，覆盖研究目标、产业意义、创新思路、方法优势、实验验证、实验设计和关键数据。
只返回符合要求的 JSON，不要输出 Markdown 代码块。"""


ANALYSIS_READING_GUIDE = """
深度解读要求：
你现在作为该领域的世界顶级学术专家，想详细阅读并深入这篇论文。
首先，请用约 1000-3000 字信息量的深度来阅读论文；在 JSON 的各字段中分散承载这些内容，而不是另起 Markdown 正文。
讲述过程中，请多引用论文中的细节内容、关键数据和实验结果；如果技术概念相对新颖，请给出通俗解释。

请围绕以下六个三级标题式问题组织分析，但仍按下方 JSON Schema 输出：
### 论文的研究目标是什么？想要解决什么实际问题？
对应 research_question；说明论文要解决的核心科学/技术问题，以及现实痛点。
### 这个问题对于产业发展有什么重要意义？
对应 background 或 innovation；分析其对产业、转化、生产、诊疗、平台化或工程应用的价值。
### 论文提出了哪些新的思路、方法或模型？
对应 methods 和 innovation；提炼新方法、新模型、新系统或新机制。
### 跟之前的方法相比有什么特点和优势？
对应 innovation 和 key_results；尽可能引用对照、性能、效率、成本、准确性、规模化等细节。
### 论文通过什么实验来验证所提出方法的有效性？
对应 methods 和 key_results；说明验证路径、关键实验、样本或对照设计。
### 实验是如何设计的？实验数据和结果如何？
对应 key_results、limitations 和 conclusion；引用关键数据、实验结果、页码和图号。

格式约束：
- 使用中文书写，学术名词可以用英文补充。
- 关键术语首次出现时用 **加粗**。
- evidence_text 中引用原文时使用 blockquote 风格，例如 `> original sentence`，并保持短引文。
- 适当关联可用图表；涉及图时必须填写 figure_id。
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

{ANALYSIS_READING_GUIDE}

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
