"""End-to-end usage reporting through the public API (mocked transport)."""

from __future__ import annotations

import json

import httpx
import pytest

import smolllm.core as core
from smolllm.core import ask_llm, stream_llm

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
async def test_ask_llm_stream_without_usage_falls_back_to_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    body = _sse({"model": "m1", "choices": [{"delta": {"content": "hello"}, "finish_reason": "stop"}]})
    _install_transport(monkeypatch, lambda request: httpx.Response(200, content=body))
    resp = await ask_llm("hi", model=MODEL, api_key="k", base_url=BASE_URL)
    assert resp.usage is not None
    assert resp.usage.estimated is True


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
    assert resp.usage is not None
    assert resp.usage.estimated is False
    assert (resp.usage.input_tokens, resp.usage.output_tokens) == (12, 34)


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
