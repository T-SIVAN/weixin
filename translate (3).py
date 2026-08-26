from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .llm import LLMError, call_openai_compatible, default_api_key, default_base_url, default_model, default_provider, parse_json_array
from .models import PaperInput, utc_now


TRANSLATE_SYSTEM = """你是生命科学文献标题翻译助手。
把英文论文标题忠实翻译为中文，不扩写、不加入评价、不改变数字和专业术语。
返回 JSON 数组，每个对象只包含 title_zh。"""

TRANSLATE_SINGLE_TITLE_SYSTEM = """你是生命科学文献标题翻译助手。
只翻译用户给出的一个英文论文标题，不翻译摘要，不解释。
返回 JSON 对象，格式为 {"title_zh": "中文标题"}。"""

DEFAULT_CACHE_PATH = Path("data/translation_cache.json")


def fallback_translation(text: str, kind: str) -> str:
    if not text:
        return ""
    return f"（未配置翻译模型，待翻译{kind}）{text}"


def normalize_title_key(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "").strip().lower())
    cleaned = re.sub(r"[^\w\s:/.-]+", "", cleaned)
    return cleaned


def translation_cache_key(record: PaperInput) -> str:
    doi = (record.doi or "").strip().lower()
    if doi:
        return f"doi:{doi}"
    return f"title:{normalize_title_key(record.title_en or record.title)}"


def load_translation_cache(cache_path: str | Path | None = None) -> dict[str, dict[str, str]]:
    path = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    cache: dict[str, dict[str, str]] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            title_zh = str(value.get("title_zh") or "").strip()
            if title_zh:
                cache[str(key)] = {str(k): str(v) for k, v in value.items() if isinstance(k, str)}
        elif str(value).strip():
            cache[str(key)] = {"title_zh": str(value).strip()}
    return cache


def save_translation_cache(cache: dict[str, dict[str, str]], cache_path: str | Path | None = None) -> None:
    path = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = load_translation_cache(path)
    merged.update(cache)
    tmp = path.with_name(f"{path.name}.{time.time_ns()}.tmp")
    try:
        tmp.write_text(json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


@dataclass
class TranslationReport:
    records: list[PaperInput]
    provider: str = "openai"
    base_url: str = ""
    model: str = ""
    errors: list[str] = field(default_factory=list)
    translated_count: int = 0
    cached_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    pending_count: int = 0
    failed_items: list[dict[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and self.failed_count == 0


@dataclass
class _TranslationCandidate:
    index: int
    record: PaperInput
    title: str
    key: str


def _emit_progress(progress_callback: Callable[[dict[str, Any]], None] | None, **event: Any) -> None:
    if progress_callback:
        progress_callback(event)


def _split_batches(
    candidates: list[_TranslationCandidate],
    *,
    batch_size: int,
    max_batch_chars: int,
) -> list[list[_TranslationCandidate]]:
    batches: list[list[_TranslationCandidate]] = []
    current: list[_TranslationCandidate] = []
    current_chars = 0
    limit = max(1, batch_size)
    char_limit = max(200, max_batch_chars)
    for candidate in candidates:
        title_chars = len(candidate.title)
        if current and (len(current) >= limit or current_chars + title_chars > char_limit):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(candidate)
        current_chars += title_chars
    if current:
        batches.append(current)
    return batches


def _sleep_for_retry(exc: Exception, attempt: int, delay_seconds: float) -> None:
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is not None:
        wait_seconds = max(0.0, float(retry_after))
    else:
        base_delay = delay_seconds if delay_seconds > 0 else 1.0
        wait_seconds = min(60.0, base_delay * (2 ** attempt))
    if wait_seconds > 0:
        time.sleep(wait_seconds)


def _is_transient_error(exc: Exception) -> bool:
    if isinstance(exc, LLMError):
        return exc.transient
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and (status_code == 429 or status_code >= 500):
        return True
    message = str(exc).lower()
    return any(term in message for term in ("429", "too many requests", "timeout", "timed out", "temporarily"))


def _call_translation_batch(
    chunk: list[_TranslationCandidate],
    *,
    api_key: str,
    base_url: str,
    model: str,
    delay_seconds: float,
    max_retries: int,
) -> list[Any]:
    payload = [{"title_en": candidate.title} for candidate in chunk]
    last_error: Exception | None = None
    attempts = max(1, max_retries + 1)
    for attempt in range(attempts):
        try:
            raw = call_openai_compatible(
                api_key=api_key,
                base_url=base_url,
                model=model,
                system_prompt=TRANSLATE_SYSTEM,
                user_prompt=json.dumps(payload, ensure_ascii=False),
                temperature=0.1,
                timeout=120,
            )
            return parse_json_array(raw)
        except Exception as exc:
            last_error = exc
            if attempt >= attempts - 1 or not _is_transient_error(exc):
                break
            _sleep_for_retry(exc, attempt, delay_seconds)
    assert last_error is not None
    raise last_error


def _extract_title_translation(raw: str) -> str:
    text = str(raw or "").strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.I)
    if fenced:
        text = fenced.group(1).strip()
    parsed: Any = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*?\}|\[[\s\S]*?\]", text)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None
    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]
    if isinstance(parsed, dict):
        return str(parsed.get("title_zh") or "").strip()
    return text.strip().strip('"').strip("'")


def _call_translation_title(
    candidate: _TranslationCandidate,
    *,
    api_key: str,
    base_url: str,
    model: str,
    delay_seconds: float,
    max_retries: int,
) -> dict[str, str]:
    last_error: Exception | None = None
    attempts = max(1, max_retries + 1)
    for attempt in range(attempts):
        try:
            raw = call_openai_compatible(
                api_key=api_key,
                base_url=base_url,
                model=model,
                system_prompt=TRANSLATE_SINGLE_TITLE_SYSTEM,
                user_prompt=candidate.title,
                temperature=0.1,
                timeout=120,
            )
            title_zh = _extract_title_translation(raw)
            if not title_zh:
                raise ValueError("模型未返回有效 title_zh")
            return {"title_zh": title_zh}
        except Exception as exc:
            last_error = exc
            if attempt >= attempts - 1 or not _is_transient_error(exc):
                break
            _sleep_for_retry(exc, attempt, delay_seconds)
    assert last_error is not None
    raise last_error


def _mark_failed(report: TranslationReport, candidate: _TranslationCandidate, error: str) -> None:
    candidate.record.translation_status = "failed"
    report.failed_count += 1
    report.failed_items.append(
        {
            "key": candidate.key,
            "doi": candidate.record.doi,
            "title": candidate.title,
            "error": error,
        }
    )


def _apply_translation(
    report: TranslationReport,
    cache: dict[str, dict[str, str]],
    candidate: _TranslationCandidate,
    item: Any,
) -> bool:
    if not isinstance(item, dict):
        return False
    title_zh = str(item.get("title_zh") or "").strip()
    if not title_zh:
        return False
    candidate.record.title_zh = title_zh
    candidate.record.translation_status = "translated"
    report.translated_count += 1
    cache[candidate.key] = {
        "title_en": candidate.title,
        "title_zh": title_zh,
        "doi": candidate.record.doi,
        "updated_at": utc_now(),
    }
    return True


def _translate_chunk(
    report: TranslationReport,
    cache: dict[str, dict[str, str]],
    chunk: list[_TranslationCandidate],
    *,
    api_key: str,
    base_url: str,
    model: str,
    delay_seconds: float,
    max_retries: int,
) -> None:
    try:
        translated = _call_translation_batch(
            chunk,
            api_key=api_key,
            base_url=base_url,
            model=model,
            delay_seconds=delay_seconds,
            max_retries=max_retries,
        )
    except Exception:
        for candidate in chunk:
            try:
                item = _call_translation_title(
                    candidate,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    delay_seconds=delay_seconds,
                    max_retries=max_retries,
                )
                if not _apply_translation(report, cache, candidate, item):
                    _mark_failed(report, candidate, "模型未返回有效 title_zh")
            except Exception as single_exc:
                message = f"{type(single_exc).__name__}: {single_exc}"
                report.errors.append(message)
                _mark_failed(report, candidate, message)
        return

    if len(translated) != len(chunk):
        for candidate in chunk:
            try:
                item = _call_translation_title(
                    candidate,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    delay_seconds=delay_seconds,
                    max_retries=max_retries,
                )
                if not _apply_translation(report, cache, candidate, item):
                    _mark_failed(report, candidate, "模型未返回有效 title_zh")
            except Exception as single_exc:
                message = f"{type(single_exc).__name__}: {single_exc}"
                report.errors.append(message)
                _mark_failed(report, candidate, message)
        return
    for idx, candidate in enumerate(chunk):
        item = translated[idx] if idx < len(translated) else {}
        if not _apply_translation(report, cache, candidate, item):
            try:
                item = _call_translation_title(
                    candidate,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    delay_seconds=delay_seconds,
                    max_retries=max_retries,
                )
                if not _apply_translation(report, cache, candidate, item):
                    _mark_failed(report, candidate, "模型未返回有效 title_zh")
            except Exception as single_exc:
                message = f"{type(single_exc).__name__}: {single_exc}"
                report.errors.append(message)
                _mark_failed(report, candidate, message)


def translate_records(
    records: list[PaperInput],
    api_key: str = "",
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    batch_size: int = 8,
    delay_seconds: float = 1.0,
    max_retries: int = 2,
    cache_path: str | Path | None = None,
    max_batch_chars: int = 4000,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    retry_failed_only: bool = False,
) -> TranslationReport:
    report = TranslationReport(records=records, provider=provider, base_url=base_url, model=model)
    if not records:
        return report

    cache = load_translation_cache(cache_path)
    candidates: list[_TranslationCandidate] = []
    for idx, record in enumerate(records):
        title = (record.title_en or record.title or "").strip()
        key = translation_cache_key(record)
        if not title:
            record.translation_status = "skipped"
            report.skipped_count += 1
            continue
        if retry_failed_only and record.translation_status != "failed":
            report.skipped_count += 1
            continue
        if record.title_zh.strip() and not retry_failed_only:
            record.translation_status = "translated"
            report.skipped_count += 1
            continue
        cached = cache.get(key, {}).get("title_zh", "").strip()
        if cached:
            record.title_zh = cached
            record.translation_status = "cached"
            report.cached_count += 1
            continue
        candidates.append(_TranslationCandidate(index=idx, record=record, title=title, key=key))

    if not candidates:
        _emit_progress(progress_callback, stage="done", completed=0, total=0, report=report)
        return report

    if not api_key.strip():
        for candidate in candidates:
            candidate.record.translation_status = "pending"
            report.pending_count += 1
        report.errors.append("未配置 API Key，已保留英文标题并标记为待翻译。")
        _emit_progress(progress_callback, stage="pending", completed=0, total=len(candidates), report=report)
        return report

    batches = _split_batches(candidates, batch_size=batch_size, max_batch_chars=max_batch_chars)
    completed = 0
    for batch_no, chunk in enumerate(batches, start=1):
        _emit_progress(
            progress_callback,
            stage="batch_start",
            batch=batch_no,
            batches=len(batches),
            completed=completed,
            total=len(candidates),
            report=report,
        )
        _translate_chunk(
            report,
            cache,
            chunk,
            api_key=api_key,
            base_url=base_url,
            model=model,
            delay_seconds=delay_seconds,
            max_retries=max_retries,
        )
        completed += len(chunk)
        _emit_progress(
            progress_callback,
            stage="batch_done",
            batch=batch_no,
            batches=len(batches),
            completed=completed,
            total=len(candidates),
            report=report,
        )
        if batch_no < len(batches) and delay_seconds > 0:
            time.sleep(delay_seconds)

    if report.translated_count:
        try:
            save_translation_cache(cache, cache_path)
        except Exception as exc:
            report.errors.append(f"Translation cache save failed: {type(exc).__name__}: {exc}")
    return report


def load_records(path: Path) -> tuple[list[PaperInput], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "records" in data:
        records = [PaperInput.from_dict(item) for item in data.get("records") or []]
        return records, data
    records = [PaperInput.from_dict(item) for item in data]
    return records, {"records": [record.to_dict() for record in records]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate search result titles into Chinese.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--provider", default=default_provider())
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--translation-cache", default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--max-batch-chars", type=int, default=4000)
    args = parser.parse_args()

    input_path = Path(args.input)
    records, original = load_records(input_path)
    base_url = args.base_url or default_base_url(args.provider)
    model = args.model or default_model(args.provider)
    report = translate_records(
        records,
        api_key=default_api_key(),
        base_url=base_url,
        model=model,
        provider=args.provider,
        batch_size=args.batch_size,
        delay_seconds=args.delay_seconds,
        cache_path=args.translation_cache,
        max_batch_chars=args.max_batch_chars,
    )
    original["records"] = [record.to_dict() for record in records]
    original["translation"] = {
        "provider": report.provider,
        "base_url": report.base_url,
        "model": report.model,
        "translated_count": report.translated_count,
        "cached_count": report.cached_count,
        "failed_count": report.failed_count,
        "skipped_count": report.skipped_count,
        "pending_count": report.pending_count,
        "failed_items": report.failed_items,
        "errors": report.errors,
    }
    output_path = Path(args.output) if args.output else input_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(original, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
