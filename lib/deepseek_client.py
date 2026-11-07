"""Direct DeepSeek chat-completions client for the production director.

The director calls ``https://api.deepseek.com/chat/completions`` with
``DEEPSEEK_API_KEY``. It does not send the request through OpenRouter.

Public docs checked on 2026-11-07, without a live call:

- https://api-docs.deepseek.com/ lists ``deepseek-flash`` for the first call
  and ``https://api.deepseek.com`` as the base URL.
- https://api-docs.deepseek.com/api/list-models/ shows ``deepseek-flash`` and
  ``deepseek-v4-pro``.
- https://api-docs.deepseek.com/guides/json_mode/ requires
  ``response_format={"type": "json_object"}``, the word "json" in the prompt,
  and a non-empty content check. Truncated ``finish_reason=length`` output is
  rejected.

No other host, and no model id containing ``/``, is accepted.
"""

from __future__ import annotations

import json
import os
from typing import Any

import requests


CHAT_URL = "https://api.deepseek.com/chat/completions"
ALLOWED_MODELS = ("deepseek-flash", "deepseek-v4-pro")
DEFAULT_MODEL = "deepseek-flash"


class DeepSeekDirectorError(ValueError):
    """The director could not complete a safe DeepSeek call."""


def redact_secret(text: str, secret: str | None) -> str:
    if not text:
        return text
    if secret:
        text = text.replace(secret, "[REDACTED]")
    return text


def resolve_model(explicit: str | None = None) -> str:
    model = (explicit or os.environ.get("DEEPSEEK_MODEL") or DEFAULT_MODEL).strip()
    if "/" in model or model not in ALLOWED_MODELS:
        raise DeepSeekDirectorError(
            f"Unsupported DeepSeek model {model!r}. "
            f"Use one of: {', '.join(ALLOWED_MODELS)}. "
            "OpenRouter model ids are not a substitute for the direct API."
        )
    return model


def resolve_chat_url() -> str:
    raw = os.environ.get("DEEPSEEK_API_BASE", "").strip()
    allowed = {
        "",
        "https://api.deepseek.com",
        "https://api.deepseek.com/",
        CHAT_URL,
    }
    if raw not in allowed:
        raise DeepSeekDirectorError(
            "DEEPSEEK_API_BASE must be https://api.deepseek.com. "
            "Other hosts, including OpenRouter, are refused."
        )
    return CHAT_URL


def complete_json(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    api_key: str | None = None,
    timeout: float = 90,
    transport=None,
) -> dict[str, Any]:
    """POST one JSON-mode chat completion and return the parsed object.

    ``transport`` replaces ``requests.post`` in tests. It is not a model hook.
    """

    key = api_key if api_key is not None else os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise DeepSeekDirectorError(
            "DEEPSEEK_API_KEY is not set. Export it in the environment. "
            "The director does not read the key from the brief or from OpenRouter."
        )
    chosen = resolve_model(model)
    url = resolve_chat_url()
    if not url.startswith("https://api.deepseek.com/"):
        raise DeepSeekDirectorError("Refusing a non-DeepSeek chat URL")
    body = {
        "model": chosen,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "max_tokens": 4096,
        "stream": False,
    }
    post = transport or requests.post
    try:
        response = post(
            url,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise DeepSeekDirectorError(
            "DeepSeek request failed: " + redact_secret(str(exc), key)
        ) from exc

    status = getattr(response, "status_code", 200)
    if status >= 400:
        detail = _error_detail(response)
        raise DeepSeekDirectorError(
            f"DeepSeek HTTP {status}: " + redact_secret(detail, key)
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise DeepSeekDirectorError("DeepSeek returned a non-JSON response") from exc
    return _parse_completion(payload, key)


def _error_detail(response) -> str:
    try:
        payload = response.json()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])[:500]
        if payload.get("message"):
            return str(payload["message"])[:500]
    text = getattr(response, "text", "") or ""
    return text[:500] or "request failed"


def _parse_completion(payload: dict[str, Any], key: str) -> dict[str, Any]:
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices:
        raise DeepSeekDirectorError("DeepSeek response did not include choices")
    choice = choices[0] if isinstance(choices[0], dict) else {}
    if choice.get("finish_reason") == "length":
        raise DeepSeekDirectorError(
            "DeepSeek stopped because max_tokens was reached. "
            "Refusing truncated JSON."
        )
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise DeepSeekDirectorError(
            "DeepSeek JSON mode returned empty content. "
            "The API can replace non-JSON output with spaces; nothing was applied."
        )
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        excerpt = redact_secret(content.strip()[:120], key)
        raise DeepSeekDirectorError(
            f"DeepSeek content was not JSON ({exc.msg}): {excerpt}"
        ) from exc
    if not isinstance(parsed, dict):
        raise DeepSeekDirectorError("DeepSeek JSON root must be an object")
    parsed["_deepseek_model"] = payload.get("model") or message.get("model")
    return parsed
