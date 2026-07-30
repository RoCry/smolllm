"""Bare model names (no provider prefix) resolve against explicit base_url/api_key only."""

from __future__ import annotations

import json

import httpx
import pytest

import smolllm.core as core
import smolllm.embeddings as embeddings
from smolllm import RequestEvent, ask_llm, embed_llm
from smolllm.providers import parse_model_string

BASE_URL = "http://test.local"


def _completion(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "m1",
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        },
    )


def _install_transport(monkeypatch: pytest.MonkeyPatch, handler, *, module=core) -> None:
    def fake_prepare(url: str, api_key: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(module, "prepare_client_and_auth", fake_prepare)


@pytest.mark.asyncio
async def test_bare_model_builds_generic_v1_url(monkeypatch: pytest.MonkeyPatch) -> None:
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        assert json.loads(request.content)["model"] == "m1"
        return _completion(request)

    _install_transport(monkeypatch, handler)
    resp = await ask_llm("hi", model="m1", api_key="k", base_url=BASE_URL, stream=False)
    assert resp.text == "hello"
    assert urls == ["http://test.local/v1/chat/completions"]
    assert resp.provider == ""
    assert resp.usage is not None
    assert resp.usage.provider == ""


@pytest.mark.asyncio
async def test_bare_model_hash_base_url_used_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return _completion(request)

    _install_transport(monkeypatch, handler)
    resp = await ask_llm("hi", model="m1", api_key="k", base_url="http://test.local/custom/endpoint#", stream=False)
    assert resp.text == "hello"
    assert urls == ["http://test.local/custom/endpoint"]


@pytest.mark.asyncio
async def test_bare_model_without_base_url_raises() -> None:
    with pytest.raises(
        ValueError,
        match=r"^Bare model 'gpt-4' requires base_url\. Pass base_url or use provider/model format$",
    ):
        await ask_llm("hi", model="gpt-4", api_key="k")


@pytest.mark.asyncio
async def test_bare_model_without_api_key_raises() -> None:
    with pytest.raises(
        ValueError,
        match=r"^Bare model 'gpt-4' requires api_key\. Pass api_key or use provider/model format$",
    ):
        await ask_llm("hi", model="gpt-4", base_url=BASE_URL)


def test_empty_provider_raises() -> None:
    with pytest.raises(ValueError, match="Empty provider"):
        parse_model_string("/gpt-4", base_url=BASE_URL)


def test_empty_model_name_raises() -> None:
    with pytest.raises(ValueError, match="Empty model name"):
        parse_model_string("testprov/")


def test_empty_model_string_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        parse_model_string("", base_url=BASE_URL)


@pytest.mark.asyncio
async def test_bare_gemini_is_a_literal_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    urls: list[str] = []
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        payloads.append(json.loads(request.content))
        return _completion(request)

    _install_transport(monkeypatch, handler)
    resp = await ask_llm("hi", model="gemini", api_key="k", base_url=BASE_URL, stream=False)
    assert payloads[0]["model"] == "gemini"
    assert resp.model_name == "gemini"
    assert resp.provider == ""
    # Generic URL grammar — the gemini provider special case must not match.
    assert urls == ["http://test.local/v1/chat/completions"]


@pytest.mark.asyncio
async def test_chain_bare_leg_failure_falls_back_to_prefixed(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[RequestEvent] = []
    payload_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        payload_models.append(model)
        if model == "bare-model":
            return httpx.Response(503, json={"error": "unavailable"})
        return _completion(request)

    _install_transport(monkeypatch, handler)
    resp = await ask_llm(
        "hi",
        model="bare-model,testprov/m1",
        api_key="k",
        base_url=BASE_URL,
        stream=False,
        hook=events.append,
    )
    assert resp.text == "hello"
    assert resp.model == "testprov/m1"
    assert payload_models == ["bare-model", "m1"]
    assert [event.usage.provider for event in events] == ["", "testprov"]
    assert events[0].error is not None
    assert events[1].error is None


@pytest.mark.asyncio
async def test_bare_model_effort_suffix_is_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return _completion(request)

    _install_transport(monkeypatch, handler)
    resp = await ask_llm("hi", model="gpt-4!low", api_key="k", base_url=BASE_URL, stream=False)
    assert payloads[0]["model"] == "gpt-4"
    assert payloads[0]["reasoning_effort"] == "low"
    assert resp.model == "gpt-4"


@pytest.mark.asyncio
async def test_embed_llm_bare_model(monkeypatch: pytest.MonkeyPatch) -> None:
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}], "usage": {"prompt_tokens": 3}})

    _install_transport(monkeypatch, handler, module=embeddings)
    resp = await embed_llm("hi", model="embed-model", api_key="k", base_url=BASE_URL)
    assert urls == ["http://test.local/v1/embeddings"]
    assert resp.provider == ""
    assert resp.model_name == "embed-model"
    assert resp.dimensions == 2
