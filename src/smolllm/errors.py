from __future__ import annotations

import hashlib
import re
from contextlib import suppress
from typing import Protocol

import httpx

from .log import logger

_MAX_ERROR_DETAIL = 500
_CREDENTIAL_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[^\s,\"'}]+"),
    re.compile(r"(?i)\b(?:api[_ -]?key|access[_ -]?token|token|key)" + r"[\"']?\s*[:=]\s*[\"']?[^\s,\"'}]+"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{12,}\b"),
    re.compile(r"\b(?:sk-|gh[opusr]_)[0-9A-Za-z_-]{10,}\b"),
)
_PERMANENT_KEY_REASONS = (
    "API_KEY_INVALID",
    "API_KEY_SERVICE_BLOCKED",
    "API_KEY_HTTP_REFERRER_BLOCKED",
    "API_KEY_IP_ADDRESS_BLOCKED",
    "API_KEY_ANDROID_APP_BLOCKED",
    "API_KEY_IOS_APP_BLOCKED",
    "CONSUMER_SUSPENDED",
    "CONSUMER_INVALID",
    "INVALID_API_KEY",
    "KEY_REVOKED",
    "API_KEY_REVOKED",
    "API_KEY_EXPIRED",
    "ACCOUNT_DEACTIVATED",
    "ORGANIZATION_DEACTIVATED",
)


class KeyEvictor(Protocol):
    def evict_key(self, key: str) -> bool: ...


def redact_credentials(text: str) -> str:
    for pattern in _CREDENTIAL_PATTERNS:
        text = pattern.sub("[REDACTED_CREDENTIAL]", text)
    return text


def brief_error_detail(text: str) -> str:
    detail = " ".join(redact_credentials(text).split())
    if not detail:
        return "no detail"
    if len(detail) <= _MAX_ERROR_DETAIL:
        return detail
    return f"{detail[: _MAX_ERROR_DETAIL - 3]}..."


def render_exception(exc: Exception) -> str:
    status: int | str = "n/a"
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
    return f"{type(exc).__name__} status={status} detail={brief_error_detail(str(exc))}"


def _permanent_error_text(exc: Exception) -> str:
    parts = [str(exc)]
    if isinstance(exc, httpx.HTTPStatusError):
        with suppress(httpx.ResponseNotRead):
            parts.append(exc.response.text)
    return " ".join(parts).upper()


def is_permanent_key_error(exc: Exception) -> bool:
    text = _permanent_error_text(exc)
    return any(reason in text for reason in _PERMANENT_KEY_REASONS)


def evict_permanent_key(evictor: KeyEvictor, key: str, exc: Exception) -> bool:
    if not key or not is_permanent_key_error(exc) or not evictor.evict_key(key):
        return False
    fingerprint = hashlib.sha256(key.encode()).hexdigest()[:12]
    logger.warning(f"Evicted permanently rejected API key for process lifetime fingerprint=sha256:{fingerprint}")
    return True
