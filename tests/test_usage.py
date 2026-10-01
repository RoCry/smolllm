"""End-to-end usage reporting through the public API (mocked transport)."""

from __future__ import annotations

import json

import httpx
import pytest

import smolllm.core as core
from smolllm.core import ask_llm, stream_llm
from smolllm.types import RequestEvent, StreamError

MODEL = "testprov/m1"
BASE_URL = "http://test.local/v1"


def _install_transport(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    def fake_prepare(url: str, api_key: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(core, "prepare_client_and_auth", fake_prepare)


def _sse(*chunks: dict[str, object]) -> bytes:
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def _omlx_keepalive_body() -> bytes:
    return _sse(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "model": "keepalive",
            "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "model": "Qwen3.8-27B-4bit",
            "choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": "stop"}],
        },
    )


@pytest.mark.asyncio
async def test_ask_llm_non_stream_uses_reported_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "m1",
                "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 34},
            },
        )

    _install_transport(monkeypatch, handler)
    resp = await ask_llm("hi", model=MODEL, api_key="k", base_url=BASE_URL, stream=False)
    assert resp.text == "hello"
    assert resp.usage is not None
    assert resp.usage.estimated is False
    assert (resp.usage.input_tokens, resp.usage.output_tokens) == (12, 34)


@pytest.mark.asyncio
async def test_ask_llm_stream_uses_final_usage_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    body = _sse(
        {"model": "m1", "choices": [{"delta": {"content": "hel"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 34}},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert b"include_usage" in request.content
        return httpx.Response(200, content=body)

    _install_transport(monkeypatch, handler)
    resp = await ask_llm("hi", model=MODEL, api_key="k", base_url=BASE_URL)
    assert resp.text == "hello"
    assert resp.usage is not None
    assert resp.usage.estimated is False
    assert (resp.usage.input_tokens, resp.usage.output_tokens) == (12, 34)


@pytest.mark.asyncio
async def test_stream_model_comes_from_last_frame_carrying_one(monkeypatch: pytest.MonkeyPatch) -> None:
    # A relay leg that only reasoned and failed names itself in the early frames;
    # the finish frame and usage frame name the leg that answered.
    body = _sse(
        {"model": "failed/leg", "choices": [{"delta": {"reasoning_content": "hmm"}, "finish_reason": None}]},
        {"model": "good/leg", "choices": [{"delta": {"content": "ok"}, "finish_reason": None}]},
        {"model": "good/leg", "choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"model": "good/leg", "choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1}},
    )
    _install_transport(monkeypatch, lambda request: httpx.Response(200, content=body))

    asked = await ask_llm("hi", model=MODEL, api_key="k", base_url=BASE_URL)
    streamed = await stream_llm("hi", model=MODEL, api_key="k", base_url=BASE_URL)
    _ = [chunk async for chunk in streamed]

    for response in (asked, streamed):
        assert response.resolved_model == "good/leg"
        assert response.usage is not None
        assert (response.usage.input_tokens, response.usage.output_tokens, response.usage.estimated) == (3, 1, False)


@pytest.mark.asyncio
async def test_truncated_stream_exposes_finish_reason_and_reported_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = _sse(
        {"model": "resolved/model", "choices": [{"delta": {"content": "partial"}, "finish_reason": "length"}]},
        {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 34}},
    )
    _install_transport(monkeypatch, lambda request: httpx.Response(200, content=body))
    events: list[RequestEvent] = []

    with pytest.raises(StreamError) as caught:
        await ask_llm("hi", model=MODEL, api_key="k", base_url=BASE_URL, hook=events.append)

    error = caught.value
    assert error.finish_reason == "length"
    assert error.resolved_model == "resolved/model"
    assert error.actual_model == "resolved/model"
    assert error.usage is not None
    assert (error.usage.input_tokens, error.usage.output_tokens, error.usage.estimated) == (12, 34, False)
    assert len(events) == 1
    assert events[0].error is error
    assert events[0].usage is error.usage


@pytest.mark.asyncio
async def test_truncated_non_stream_estimates_missing_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "resolved/model",
                "choices": [{"message": {"content": "partial"}, "finish_reason": "length"}],
            },
        )

    _install_transport(monkeypatch, handler)
    with pytest.raises(StreamError) as caught:
        await ask_llm("hi", model=MODEL, api_key="k", base_url=BASE_URL, stream=False)

    assert caught.value.finish_reason == "length"
    assert caught.value.usage is not None
    assert caught.value.usage.output_tokens > 0
    assert caught.value.usage.estimated is True


@pytest.mark.asyncio
async def test_stream_missing_terminal_reason_exposes_estimated_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    body = _sse({"model": "resolved/model", "choices": [{"delta": {"content": "partial"}}]})
    _install_transport(monkeypatch, lambda request: httpx.Response(200, content=body))

    with pytest.raises(StreamError) as caught:
        await ask_llm("hi", model=MODEL, api_key="k", base_url=BASE_URL)

    assert caught.value.finish_reason is None
    assert caught.value.actual_model == "resolved/model"
    assert caught.value.usage is not None
    assert caught.value.usage.output_tokens > 0
    assert caught.value.usage.estimated is True


@pytest.mark.asyncio
async def test_ask_llm_stream_without_usage_falls_back_to_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    body = _sse({"model": "m1", "choices": [{"delta": {"content": "hello"}, "finish_reason": "stop"}]})
    _install_transport(monkeypatch, lambda request: httpx.Response(200, content=body))
    resp = await ask_llm("hi", model=MODEL, api_key="k", base_url=BASE_URL)
    assert resp.usage is not None
    assert resp.usage.estimated is True


@pytest.mark.asyncio
async def test_ask_llm_ignores_omlx_keepalive_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_transport(monkeypatch, lambda request: httpx.Response(200, content=_omlx_keepalive_body()))

    resp = await ask_llm("hi", model=MODEL, api_key="k", base_url=BASE_URL)

    assert resp.text == "hello"
    assert resp.resolved_model == "Qwen3.8-27B-4bit"
    assert resp.actual_model == "Qwen3.8-27B-4bit"


@pytest.mark.asyncio
async def test_ask_llm_retries_400_without_stream_options(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bytes] = []
    body = _sse({"model": "m1", "choices": [{"delta": {"content": "hello"}, "finish_reason": "stop"}]})

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.content)
        if b"stream_options" in request.content:
            return httpx.Response(400, json={"error": {"message": "unknown field stream_options"}})
        return httpx.Response(200, content=body)

    _install_transport(monkeypatch, handler)
    resp = await ask_llm("hi", model=MODEL, api_key="k", base_url=BASE_URL)
    assert resp.text == "hello"
    assert len(calls) == 2
    assert b"stream_options" not in calls[1]
    assert resp.usage is not None
    assert resp.usage.estimated is True


@pytest.mark.asyncio
async def test_stream_llm_uses_final_usage_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    body = _sse(
        {"model": "m1", "choices": [{"delta": {"content": "hel"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 34}},
    )
    _install_transport(monkeypatch, lambda request: httpx.Response(200, content=body))
    resp = await stream_llm("hi", model=MODEL, api_key="k", base_url=BASE_URL)
    text = "".join([chunk.content async for chunk in resp])
    assert text == "hello"
    assert resp.actual_model == "m1"
    assert resp.usage is not None
    assert resp.usage.estimated is False
    assert (resp.usage.input_tokens, resp.usage.output_tokens) == (12, 34)


@pytest.mark.asyncio
async def test_stream_llm_ignores_omlx_keepalive_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_transport(monkeypatch, lambda request: httpx.Response(200, content=_omlx_keepalive_body()))

    resp = await stream_llm("hi", model=MODEL, api_key="k", base_url=BASE_URL)
    text = "".join([chunk.content async for chunk in resp])

    assert text == "hello"
    assert resp.resolved_model == "Qwen3.8-27B-4bit"
    assert resp.actual_model == "Qwen3.8-27B-4bit"


@pytest.mark.asyncio
async def test_stream_llm_retries_400_without_stream_options(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bytes] = []
    body = _sse({"model": "m1", "choices": [{"delta": {"content": "hello"}, "finish_reason": "stop"}]})

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.content)
        if b"stream_options" in request.content:
            return httpx.Response(400, json={"error": {"message": "unknown field stream_options"}})
        return httpx.Response(200, content=body)

    _install_transport(monkeypatch, handler)
    resp = await stream_llm("hi", model=MODEL, api_key="k", base_url=BASE_URL)
    text = "".join([chunk.content async for chunk in resp])
    assert text == "hello"
    assert len(calls) == 2
    assert resp.usage is not None
    assert resp.usage.estimated is True
