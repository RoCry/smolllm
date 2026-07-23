from __future__ import annotations

import json
import typing

import httpx
import pytest

import smolllm.core as core
from smolllm import LLMFunction, RequestEvent, ask_llm

MODEL = "testprov/m1"
BASE_URL = "http://test.local/v1"


@pytest.mark.asyncio
async def test_ask_llm_closes_an_internally_created_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[httpx.AsyncClient] = []

    def prepare_client(_url: str, _api_key: str) -> httpx.AsyncClient:
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={
                        "model": "resolved",
                        "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
                    },
                )
            )
        )
        clients.append(client)
        return client

    monkeypatch.setattr(core, "prepare_client_and_auth", prepare_client)

    response = await ask_llm(
        "question",
        model=MODEL,
        api_key="secret",
        base_url=BASE_URL,
        stream=False,
    )

    assert response.text == "answer"
    assert len(clients) == 1
    assert clients[0].is_closed is True


@pytest.mark.asyncio
async def test_ask_llm_closes_internal_client_when_pre_request_work_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[httpx.AsyncClient] = []

    def prepare_client(_url: str, _api_key: str) -> httpx.AsyncClient:
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: httpx.Response(200)))
        clients.append(client)
        return client

    estimates = iter((1, RuntimeError("estimate failed")))

    def estimate_tokens(_text: str) -> int:
        result = next(estimates)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(core, "prepare_client_and_auth", prepare_client)
    monkeypatch.setattr(core, "estimate_tokens", estimate_tokens)

    with pytest.raises(RuntimeError, match="estimate failed"):
        await ask_llm(
            "question",
            model=MODEL,
            api_key="secret",
            base_url=BASE_URL,
            stream=False,
        )

    assert len(clients) == 1
    assert clients[0].is_closed is True


@pytest.mark.asyncio
async def test_ask_llm_reuses_caller_owned_client_across_calls() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        answer = "expanded" if len(requests) == 1 else "synthesized"
        return httpx.Response(
            200,
            json={
                "model": f"resolved-{answer}",
                "choices": [{"message": {"content": answer}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            },
        )

    client = httpx.AsyncClient(
        auth=httpx.BasicAuth("wrong", "auth"),
        transport=httpx.MockTransport(handler),
    )
    try:
        expansion = await ask_llm(
            "expand",
            model=MODEL,
            api_key="secret",
            base_url=BASE_URL,
            stream=False,
            client=client,
        )
        assert client.is_closed is False

        synthesis = await ask_llm(
            "synthesize",
            model=MODEL,
            api_key="secret",
            base_url=BASE_URL,
            stream=False,
            client=client,
        )
        assert client.is_closed is False
    finally:
        await client.aclose()

    assert (expansion.text, synthesis.text) == ("expanded", "synthesized")
    assert (expansion.resolved_model, synthesis.resolved_model) == (
        "resolved-expanded",
        "resolved-synthesized",
    )
    assert [request.headers["authorization"] for request in requests] == ["Bearer secret", "Bearer secret"]


@pytest.mark.asyncio
async def test_injected_client_preserves_stream_fallback_hooks_and_usage() -> None:
    requests: list[dict[str, object]] = []
    authorization_headers: list[str] = []
    events: list[RequestEvent] = []
    stream_body = (
        'data: {"model":"upstream/good","choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}\n\n'
        'data: {"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":7}}\n\n'
        "data: [DONE]\n\n"
    ).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        authorization_headers.append(request.headers["authorization"])
        if payload["model"] == "fail":
            return httpx.Response(503, json={"error": "unavailable"})
        if "stream_options" in payload:
            return httpx.Response(400, json={"error": "unknown field stream_options"})
        return httpx.Response(200, content=stream_body)

    client = httpx.AsyncClient(
        auth=httpx.BasicAuth("wrong", "auth"),
        transport=httpx.MockTransport(handler),
    )
    try:
        response = await ask_llm(
            "question",
            model=["testprov/fail", "testprov/good"],
            api_key="secret",
            base_url=BASE_URL,
            hook=events.append,
            client=client,
        )
        assert client.is_closed is False
    finally:
        await client.aclose()

    assert response.text == "answer"
    assert response.model == "testprov/good"
    assert response.resolved_model == "upstream/good"
    assert response.finish_reason == "stop"
    assert response.usage is not None
    assert (response.usage.input_tokens, response.usage.output_tokens, response.usage.estimated) == (11, 7, False)
    assert [type(event.error).__name__ if event.error else None for event in events] == ["HTTPStatusError", None]
    assert [payload["model"] for payload in requests] == ["fail", "good", "good"]
    assert authorization_headers == ["Bearer secret"] * 3
    assert "stream_options" in requests[1]
    assert "stream_options" not in requests[2]


def test_llm_function_runtime_type_hints_are_resolvable() -> None:
    hints = typing.get_type_hints(LLMFunction.__call__)

    assert hints["client"] == httpx.AsyncClient | None
