from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any


class LLMError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
        transient: bool = False,
        error_code: str = "",
        quota_exhausted: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.transient = transient
        self.error_code = error_code
        self.quota_exhausted = quota_exhausted


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
    max_attempts: int = 3,
) -> str:
    attempts = max(1, min(int(max_attempts), 3))
    for attempt in range(attempts):
        try:
            return _call_openai_compatible_once(
                api_key=api_key,
                base_url=base_url,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                timeout=timeout,
            )
        except LLMError as exc:
            if exc.quota_exhausted or not exc.transient or attempt + 1 >= attempts:
                raise
            delay = exc.retry_after if exc.retry_after is not None else float(2**attempt)
            time.sleep(max(0.0, delay))
    raise LLMError("LLM retry loop ended unexpectedly")


def _call_openai_compatible_once(
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
    except urllib.error.HTTPError as exc:
        retry_after = _parse_retry_after(exc.headers.get("Retry-After") if exc.headers else None)
        detail, error_code = _read_http_error_detail(exc)
        quota_exhausted = _is_quota_exhaustion(detail, error_code)
        transient = (exc.code == 429 or exc.code >= 500) and not quota_exhausted
        message = f"LLM HTTP {exc.code} {exc.reason}: {detail}".strip()
        raise LLMError(
            message,
            status_code=exc.code,
            retry_after=retry_after,
            transient=transient,
            error_code=error_code,
            quota_exhausted=quota_exhausted,
        ) from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        transient = isinstance(reason, (TimeoutError, socket.timeout)) or "timed out" in str(reason).lower()
        raise LLMError(f"LLM network error: {reason}", transient=transient) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise LLMError(f"LLM request timed out after {timeout}s", transient=True) from exc
    except json.JSONDecodeError as exc:
        raise LLMError(f"LLM returned invalid JSON response: {exc}") from exc
    except Exception as exc:
        raise LLMError(f"LLM request failed: {exc}") from exc
    try:
        return body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected LLM response: {body}") from exc


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    from datetime import datetime, timezone

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())


def _is_quota_exhaustion(message: str, error_code: str = "") -> bool:
    value = f"{error_code} {message}".lower()
    return any(
        marker in value
        for marker in (
            "insufficient_quota",
            "current quota",
            "billing quota",
            "run out of credits",
            "no balance left",
            "exceeded your quota",
        )
    )


def _read_http_error_detail(exc: urllib.error.HTTPError) -> tuple[str, str]:
    try:
        raw = exc.read().decode("utf-8", errors="replace")
    except Exception:
        raw = ""
    if not raw:
        return "no response body", ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw[:500], ""
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error), str(error.get("code") or error.get("type") or "")
        if error:
            return str(error), ""
        if data.get("message"):
            return str(data["message"]), str(data.get("code") or "")
    return raw[:500], ""


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
