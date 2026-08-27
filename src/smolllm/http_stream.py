from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from time import perf_counter
from typing import cast

import httpx

from .errors import brief_error_detail, extract_error_reason_codes, provider_http_status_error
from .log import logger
from .stream import (
    ToolCallAccumulator,
    decode_sse_chunk,
    extract_delta,
    extract_finish_reason,
    extract_model,
    update_usage,
)
from .types import StreamHandler
from .utils import ThinkTagFilter


async def handle_http_error(response: httpx.Response) -> None:
    if response.status_code >= 400:
        error_text = await response.aread()
        decoded = error_text.decode(errors="replace")
        detail = brief_error_detail(decoded)
        try:
            payload = cast(object, response.json())
        except ValueError:
            payload = decoded
        raise provider_http_status_error(
            f"HTTP Error {response.status_code}: {detail}",
            request=response.request,
            response=response,
            reason_codes=extract_error_reason_codes(payload),
        )


def usage_tokens_from_payload(payload: object) -> tuple[int | None, int | None] | None:
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if isinstance(prompt_tokens, int) or isinstance(completion_tokens, int):
        return (
            prompt_tokens if isinstance(prompt_tokens, int) else None,
            completion_tokens if isinstance(completion_tokens, int) else None,
        )
    return None


def _without_stream_usage(data: dict[str, object]) -> dict[str, object]:
    retry_data = dict(data)
    retry_data.pop("stream_options", None)
    return retry_data


def _can_retry_without_stream_usage(data: dict[str, object], exc: httpx.HTTPStatusError) -> bool:
    return bool(data.get("stream_options")) and exc.response.status_code == 400


async def iter_stream_lines(
    client: httpx.AsyncClient,
    url: str,
    data: dict[str, object],
    timeout: float,
    *,
    headers: dict[str, str] | None = None,
) -> AsyncIterator[str]:
    try:
        async with client.stream(
            "POST",
            url,
            json=data,
            timeout=timeout,
            headers=headers,
            auth=None,
        ) as response:
            await handle_http_error(response)
            async for line in response.aiter_lines():
                yield line
            return
    except httpx.HTTPStatusError as exc:
        if not _can_retry_without_stream_usage(data, exc):
            raise
        logger.warning("Provider rejected stream_options; retrying stream without usage inclusion")

    retry_data = _without_stream_usage(data)
    async with client.stream(
        "POST",
        url,
        json=retry_data,
        timeout=timeout,
        headers=headers,
        auth=None,
    ) as response:
        await handle_http_error(response)
        async for line in response.aiter_lines():
            yield line


@dataclass(slots=True)
class StreamOutcome:
    """Everything a consumed stream yielded besides the chunks themselves."""

    text: str
    reasoning: str
    ttft_ms: int | None
    resolved_model: str | None
    finish_reason: str | None
    tool_calls: list[dict[str, object]] = field(default_factory=list)


async def process_stream_response(
    lines: AsyncIterator[str],
    stream_handler: StreamHandler | None,
    start_time: float,
    *,
    usage: dict[str, int] | None = None,
) -> StreamOutcome:
    """Consume an SSE stream into displayed text plus its terminal metadata."""
    from .display import ResponseDisplay

    first_token_time: float | None = None
    resolved_model: str | None = None
    finish_reason: str | None = None
    think_filter = ThinkTagFilter()
    tool_calls = ToolCallAccumulator()
    with ResponseDisplay(stream_handler) as display:
        async for line in lines:
            raw = decode_sse_chunk(line)
            if raw is None:
                continue
            if usage is not None:
                update_usage(raw, usage)
            if resolved_model is None:
                resolved_model = extract_model(raw)
            if (reason := extract_finish_reason(raw)) is not None:
                finish_reason = reason
            tool_calls.feed(raw)
            if chunk := extract_delta(raw):
                chunk = think_filter.feed(chunk)
                if chunk:
                    if first_token_time is None:
                        first_token_time = perf_counter()
                    await display.update(chunk)
        if final_chunk := think_filter.flush():
            await display.update(final_chunk)
        assembled_calls = tool_calls.result()
        text, reasoning = display.finalize(allow_empty=bool(assembled_calls))

    ttft_ms: int | None = None
    if first_token_time is not None:
        ttft_ms = max(0, int((first_token_time - start_time) * 1000))

    return StreamOutcome(
        text=text,
        reasoning=reasoning,
        ttft_ms=ttft_ms,
        resolved_model=resolved_model,
        finish_reason=finish_reason,
        tool_calls=assembled_calls,
    )
