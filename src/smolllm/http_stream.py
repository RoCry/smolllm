from __future__ import annotations

from collections.abc import AsyncIterator
from time import perf_counter

import httpx

from .display import ResponseDisplay
from .log import logger
from .stream import decode_sse_chunk, extract_delta, extract_finish_reason, extract_model, update_usage
from .types import StreamHandler
from .utils import ThinkTagFilter


async def handle_http_error(response: httpx.Response) -> None:
    if response.status_code >= 400:
        error_text = await response.aread()
        raise httpx.HTTPStatusError(
            f"HTTP Error {response.status_code}: {error_text.decode()}",
            request=response.request,
            response=response,
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
) -> AsyncIterator[str]:
    try:
        async with client.stream("POST", url, json=data, timeout=timeout) as response:
            await handle_http_error(response)
            async for line in response.aiter_lines():
                yield line
            return
    except httpx.HTTPStatusError as exc:
        if not _can_retry_without_stream_usage(data, exc):
            raise
        logger.warning("Provider rejected stream_options; retrying stream without usage inclusion")

    retry_data = _without_stream_usage(data)
    async with client.stream("POST", url, json=retry_data, timeout=timeout) as response:
        await handle_http_error(response)
        async for line in response.aiter_lines():
            yield line


async def process_stream_response(
    lines: AsyncIterator[str],
    stream_handler: StreamHandler | None,
    start_time: float,
    *,
    usage: dict[str, int] | None = None,
) -> tuple[str, str, int | None, str | None, str | None]:
    """Returns (text, reasoning, ttft_ms, resolved_model, finish_reason)."""
    first_token_time: float | None = None
    resolved_model: str | None = None
    finish_reason: str | None = None
    think_filter = ThinkTagFilter()
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
            if chunk := extract_delta(raw):
                chunk = think_filter.feed(chunk)
                if chunk:
                    if first_token_time is None:
                        first_token_time = perf_counter()
                    await display.update(chunk)
        if final_chunk := think_filter.flush():
            await display.update(final_chunk)
        text, reasoning = display.finalize()

    ttft_ms: int | None = None
    if first_token_time is not None:
        ttft_ms = max(0, int((first_token_time - start_time) * 1000))

    return text, reasoning, ttft_ms, resolved_model, finish_reason
