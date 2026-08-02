from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Protocol, cast

import httpx

from .log import logger

_MAX_ERROR_DETAIL = 500
_CREDENTIAL_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[^\s,\"'}]+"),
    re.compile(r"(?i)\b(?:api[_ -]?key|access[_ -]?token|token|key)" + r"[\"']?\s*[:=]\s*[\"']?[^\s,\"'}]+"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{12,}\b"),
    re.compile(r"\b(?:sk-|gh[opusr]_)[0-9A-Za-z_-]{10,}\b"),
)
_PERMANENT_KEY_REASONS = frozenset(
    {
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
    }
)
_ERROR_CODE_FIELDS = frozenset({"code", "reason", "status", "type"})
_PROVIDER_REASON_CODES_ATTR = "_smolllm_provider_reason_codes"


class PairEvictor(Protocol):
    def evict_pair(self, key: str, url: str) -> bool: ...


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


def extract_error_reason_codes(value: object) -> frozenset[str]:
    codes: set[str] = set()

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            mapping = cast(Mapping[object, object], item)
            for key, child in mapping.items():
                if isinstance(key, str) and key.lower() in _ERROR_CODE_FIELDS and isinstance(child, str):
                    normalized = re.sub(r"[^A-Z0-9]+", "_", child.upper()).strip("_")
                    if normalized:
                        codes.add(normalized)
                visit(child)
        elif isinstance(item, list | tuple):
            sequence = cast(list[object] | tuple[object, ...], item)
            for child in sequence:
                visit(child)
        elif isinstance(item, str):
            upper = item.upper()
            codes.update(reason for reason in _PERMANENT_KEY_REASONS if reason in upper)

    visit(value)
    return frozenset(codes)


def provider_http_status_error(
    message: str,
    *,
    request: httpx.Request,
    response: httpx.Response,
    reason_codes: frozenset[str],
) -> httpx.HTTPStatusError:
    error = httpx.HTTPStatusError(message, request=request, response=response)
    error.__dict__[_PROVIDER_REASON_CODES_ATTR] = reason_codes
    return error


def _render_single_exception(exc: BaseException) -> str:
    status: int | str = "n/a"
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
    return f"{type(exc).__name__} status={status} detail={brief_error_detail(str(exc))}"


def render_exception(exc: Exception) -> str:
    rendered: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        rendered.append(_render_single_exception(current))
        current = current.__cause__ or current.__context__
    return " caused_by=".join(rendered)


def is_permanent_key_error(exc: Exception) -> bool:
    from .types import StreamError

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, httpx.HTTPStatusError):
            reason_codes: object = current.__dict__.get(_PROVIDER_REASON_CODES_ATTR)
            if isinstance(reason_codes, frozenset) and reason_codes & _PERMANENT_KEY_REASONS:
                return True
        elif isinstance(current, StreamError) and current.reason_codes & _PERMANENT_KEY_REASONS:
            return True
        current = current.__cause__ or current.__context__
    return False


def evict_permanent_pair(evictor: PairEvictor, key: str, url: str, exc: Exception) -> bool:
    if not key or not url or not is_permanent_key_error(exc) or not evictor.evict_pair(key, url):
        return False
    fingerprint = hashlib.sha256(key.encode()).hexdigest()[:12]
    logger.warning(f"Evicted permanently rejected API key for process lifetime fingerprint=sha256:{fingerprint}")
    return True
