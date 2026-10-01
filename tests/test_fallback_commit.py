"""Answer content commits a call to its model: no client-side fallback after it was delivered."""

from __future__ import annotations

import json

import httpx
import pytest

import smolllm.core as core
from smolllm.core import ask_llm, stream_llm
from smolllm.types import StreamChunk, StreamError

BASE_URL = "http://test.local/v1"
MODELS = ["testprov/first", "testprov/second"]


def _sse(*chunks: dict[str, object]) -> bytes:
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def _error_frame(model: str) -> dict[str, object]:
    return {
        "model": model,
        "choices": [{"delta": {}, "finish_reason": "error"}],
        "error": {"message": "upstream died", "type": "upstream_error"},
    }


def _answer(model: str) -> bytes:
    return _sse(
        {"model": model, "choices": [{"delta": {"content": "second answer"}, "finish_reason": "stop"}]},
    )


def _install(monkeypatch: pytest.MonkeyPatch, bodies: dict[str, bytes]) -> list[str]:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        requested.append(model)
        return httpx.Response(200, content=bodies[model])

    def fake_prepare(url: str, api_key: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(core, "prepare_client_and_auth", fake_prepare)
    return requested


def _content_then_error() -> dict[str, bytes]:
    return {
        "first": _sse(
            {"model": "first", "choices": [{"delta": {"content": "first partial"}, "finish_reason": None}]},
            _error_frame("first"),
        ),
        "second": _answer("second"),
    }


def _reasoning_then_error() -> dict[str, bytes]:
    return {
        "first": _sse(
            {"model": "first", "choices": [{"delta": {"reasoning_content": "thinking"}, "finish_reason": None}]},
            _error_frame("first"),
        ),
        "second": _answer("second"),
    }


@pytest.mark.asyncio
async def test_ask_llm_handler_content_commits_to_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    requested = _install(monkeypatch, _content_then_error())
    received: list[StreamChunk] = []

    async def handler(chunk: StreamChunk) -> None:
        received.append(chunk)

    with pytest.raises(StreamError, match="upstream died"):
        await ask_llm("hi", model=MODELS, api_key="k", base_url=BASE_URL, handler=handler)

    assert requested == ["first"]
    assert "".join(chunk.content for chunk in received) == "first partial"


@pytest.mark.asyncio
async def test_ask_llm_without_handler_still_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    requested = _install(monkeypatch, _content_then_error())

    response = await ask_llm("hi", model=MODELS, api_key="k", base_url=BASE_URL)

    assert requested == ["first", "second"]
    assert response.text == "second answer"
    assert response.resolved_model == "second"


@pytest.mark.asyncio
async def test_ask_llm_handler_reasoning_only_does_not_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    requested = _install(monkeypatch, _reasoning_then_error())
    received: list[StreamChunk] = []

    async def handler(chunk: StreamChunk) -> None:
        received.append(chunk)

    response = await ask_llm("hi", model=MODELS, api_key="k", base_url=BASE_URL, handler=handler)

    assert requested == ["first", "second"]
    assert response.text == "second answer"
    assert [(chunk.reasoning, chunk.content) for chunk in received] == [("thinking", ""), ("", "second answer")]


@pytest.mark.asyncio
async def test_ask_llm_non_stream_handler_content_commits_to_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # A truncated non-streamed answer has already been handed to the handler.
    truncated = json.dumps(
        {"model": "first", "choices": [{"message": {"content": "cut"}, "finish_reason": "length"}]}
    ).encode()
    requested = _install(monkeypatch, {"first": truncated, "second": truncated})
    received: list[StreamChunk] = []

    async def handler(chunk: StreamChunk) -> None:
        received.append(chunk)

    with pytest.raises(StreamError, match="Truncated"):
        await ask_llm("hi", model=MODELS, api_key="k", base_url=BASE_URL, handler=handler, stream=False)

    assert requested == ["first"]
    assert [chunk.content for chunk in received] == ["cut"]


@pytest.mark.asyncio
async def test_stream_llm_content_commits_reasoning_does_not(monkeypatch: pytest.MonkeyPatch) -> None:
    requested = _install(monkeypatch, _content_then_error())
    committed = await stream_llm("hi", model=MODELS, api_key="k", base_url=BASE_URL)
    with pytest.raises(StreamError) as raised:
        _ = [chunk async for chunk in committed]
    assert raised.value.partial == "first partial"
    assert requested == ["first"]

    requested = _install(monkeypatch, _reasoning_then_error())
    retried = await stream_llm("hi", model=MODELS, api_key="k", base_url=BASE_URL)
    chunks = [chunk async for chunk in retried]
    assert requested == ["first", "second"]
    assert [(chunk.reasoning, chunk.content) for chunk in chunks] == [("thinking", ""), ("", "second answer")]
    assert retried.resolved_model == "second"
