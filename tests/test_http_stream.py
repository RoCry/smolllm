from __future__ import annotations

import json

import httpx
import pytest

from smolllm.http_stream import iter_stream_lines, usage_tokens_from_payload


@pytest.mark.asyncio
async def test_iter_stream_lines_does_not_retry_stream_options_on_rate_limit() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(429, text="rate limited", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            _ = [line async for line in iter_stream_lines(client, "https://example.test/chat", _stream_data(), 10)]

    assert len(requests) == 1
    assert "stream_options" in requests[0]


@pytest.mark.asyncio
async def test_iter_stream_lines_retries_stream_options_on_bad_request() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            return httpx.Response(400, text="unknown field stream_options", request=request)
        return httpx.Response(200, text="data: [DONE]\n\n", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        lines = [line async for line in iter_stream_lines(client, "https://example.test/chat", _stream_data(), 10)]

    assert lines == ["data: [DONE]", ""]
    assert len(requests) == 2
    assert "stream_options" in requests[0]
    assert "stream_options" not in requests[1]


def test_usage_tokens_from_payload_accepts_partial_fields() -> None:
    assert usage_tokens_from_payload({"usage": {"prompt_tokens": 11}}) == (11, None)
    assert usage_tokens_from_payload({"usage": {"completion_tokens": 7}}) == (None, 7)


def _stream_data() -> dict[str, object]:
    return {"model": "m", "stream": True, "stream_options": {"include_usage": True}}
