from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .llm import call_openai_compatible, default_api_key, default_base_url, default_model, default_provider, parse_json_array
from .models import PaperInput, SearchRun


TRANSLATE_SYSTEM = """你是合成生物学文献标题翻译助手。
把英文论文标题忠实翻译为中文，不扩写、不加入评价、不改变数字和专业术语。
返回 JSON 数组，每个对象只包含 title_zh。"""


def fallback_translation(text: str, kind: str) -> str:
    if not text:
        return ""
    return f"（未配置翻译模型，待翻译{kind}）{text}"


@dataclass
class TranslationReport:
    records: list[PaperInput]
    provider: str = "openai"
    base_url: str = ""
    model: str = ""
    errors: list[str] = field(default_factory=list)
    translated_count: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def translate_records(
    records: list[PaperInput],
    api_key: str = "",
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    batch_size: int = 3,
    delay_seconds: float = 1.5,
    max_retries: int = 2,
) -> TranslationReport:
    errors: list[str] = []
    translated_count = 0
    if not records:
        return TranslationReport(records=records, provider=provider, base_url=base_url, model=model)
    if not api_key.strip():
        for record in records:
            record.title_zh = record.title_zh or fallback_translation(record.title_en or record.title, "标题")
        errors.append("未填写 API Key，已保留英文标题并标记待翻译。")
        return TranslationReport(records=records, provider=provider, base_url=base_url, model=model, errors=errors)

    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        payload = [
            {
                "title_en": record.title_en or record.title,
            }
            for record in chunk
        ]
        translated = []
        last_error: Exception | None = None
        for attempt in range(max(1, max_retries + 1)):
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
                translated = parse_json_array(raw)
                break
            except Exception as exc:
                last_error = exc
                message = str(exc).lower()
                if "429" not in message and "too many requests" not in message:
                    break
                time.sleep(min(20.0, delay_seconds * (attempt + 1) * 2))
        if last_error and not translated:
            errors.append(
                f"{provider} batch {start // batch_size + 1} failed at {base_url}: {type(last_error).__name__}: {last_error}"
            )
        for idx, record in enumerate(chunk):
            item = translated[idx] if idx < len(translated) and isinstance(translated[idx], dict) else {}
            if item.get("title_zh"):
                translated_count += 1
            record.title_zh = str(item.get("title_zh") or record.title_zh or fallback_translation(record.title_en or record.title, "标题"))
        if start + batch_size < len(records) and delay_seconds > 0:
            time.sleep(delay_seconds)
    return TranslationReport(
        records=records,
        provider=provider,
        base_url=base_url,
        model=model,
        errors=errors,
        translated_count=translated_count,
    )


def load_records(path: Path) -> tuple[list[PaperInput], dict]:
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
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--delay-seconds", type=float, default=1.5)
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
    )
    original["records"] = [record.to_dict() for record in records]
    original["translation"] = {
        "provider": report.provider,
        "base_url": report.base_url,
        "model": report.model,
        "translated_count": report.translated_count,
        "errors": report.errors,
    }
    output_path = Path(args.output) if args.output else input_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(original, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
