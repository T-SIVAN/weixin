from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from .llm import call_openai_compatible, default_api_key, default_base_url, default_model, default_provider, parse_json_array
from .models import PaperInput, SearchRun


TRANSLATE_SYSTEM = """你是科研文献标题和摘要翻译助手。
只把英文标题和英文摘要忠实翻译为中文，不扩写、不加入评价、不改变数字和专业术语。
返回 JSON 数组，每个对象只包含 title_zh 和 abstract_zh。"""


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
    batch_size: int = 8,
) -> TranslationReport:
    errors: list[str] = []
    translated_count = 0
    if not api_key.strip():
        for record in records:
            record.title_zh = record.title_zh or fallback_translation(record.title_en or record.title, "标题")
            record.abstract_zh = record.abstract_zh or fallback_translation(record.abstract_en or record.abstract, "摘要")
        errors.append("未填写 API Key，已保留英文并标记待翻译。")
        return TranslationReport(records=records, provider=provider, base_url=base_url, model=model, errors=errors)

    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        payload = [
            {
                "title_en": record.title_en or record.title,
                "abstract_en": record.abstract_en or record.abstract,
            }
            for record in chunk
        ]
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
        except Exception as exc:
            errors.append(
                f"{provider} batch {start // batch_size + 1} failed at {base_url}: {type(exc).__name__}: {exc}"
            )
            translated = []
        for idx, record in enumerate(chunk):
            item = translated[idx] if idx < len(translated) and isinstance(translated[idx], dict) else {}
            if item.get("title_zh") or item.get("abstract_zh"):
                translated_count += 1
            record.title_zh = str(item.get("title_zh") or record.title_zh or fallback_translation(record.title_en or record.title, "标题"))
            record.abstract_zh = str(item.get("abstract_zh") or record.abstract_zh or fallback_translation(record.abstract_en or record.abstract, "摘要"))
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
    parser = argparse.ArgumentParser(description="Translate search result titles and abstracts into Chinese.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--provider", default=default_provider())
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="")
    args = parser.parse_args()

    input_path = Path(args.input)
    records, original = load_records(input_path)
    base_url = args.base_url or default_base_url(args.provider)
    model = args.model or default_model(args.provider)
    report = translate_records(records, api_key=default_api_key(), base_url=base_url, model=model, provider=args.provider)
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
