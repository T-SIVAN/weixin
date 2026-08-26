from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Any


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    label: str
    base_url: str
    default_model: str


PROVIDERS: dict[str, ProviderConfig] = {
    "openai": ProviderConfig("openai", "OpenAI", "https://api.openai.com/v1", "gpt-4o-mini"),
    "deepseek": ProviderConfig("deepseek", "DeepSeek", "https://api.deepseek.com/v1", "deepseek-chat"),
    "siliconflow": ProviderConfig("siliconflow", "SiliconFlow", "https://api.siliconflow.cn/v1", "Qwen/Qwen2.5-7B-Instruct"),
    "custom": ProviderConfig("custom", "Custom OpenAI-compatible", "", ""),
}


def normalize_provider(provider: str = "") -> str:
    key = (provider or "").strip().lower()
    return key if key in PROVIDERS else "openai"


def default_provider() -> str:
    return normalize_provider(os.getenv("LLM_PROVIDER") or os.getenv("OPENAI_PROVIDER") or "openai")


def default_api_key() -> str:
    return os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or ""


def default_base_url(provider: str = "") -> str:
    configured = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or ""
    if configured:
        return configured
    return PROVIDERS[normalize_provider(provider or default_provider())].base_url or PROVIDERS["openai"].base_url


def default_model(provider: str = "") -> str:
    configured = os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or ""
    if configured:
        return configured
    return PROVIDERS[normalize_provider(provider or default_provider())].default_model or PROVIDERS["openai"].default_model


def provider_defaults(provider: str = "") -> ProviderConfig:
    return PROVIDERS[normalize_provider(provider)]


def call_openai_compatible(
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    timeout: int = 120,
) -> str:
    if not api_key.strip():
        raise LLMError("Missing API key")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise LLMError(f"LLM request failed: {exc}") from exc
    try:
        return body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected LLM response: {body}") from exc


def test_llm_connection(api_key: str, base_url: str, model: str) -> str:
    return call_openai_compatible(
        api_key=api_key,
        base_url=base_url,
        model=model,
        system_prompt="You are a translation health-check endpoint. Reply with JSON only.",
        user_prompt='Return exactly {"ok":true,"zh":"测试成功"}.',
        temperature=0,
        timeout=30,
    )


def strip_json_fence(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_json_fence(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM response is not a JSON object")
    return parsed


def parse_json_array(text: str) -> list[Any]:
    cleaned = strip_json_fence(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, list):
        raise ValueError("LLM response is not a JSON array")
    return parsed
