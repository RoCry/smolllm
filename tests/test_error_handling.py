from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import httpx
import pytest

import smolllm.core as core
from smolllm import ask_llm, stream_llm
from smolllm.balancer import SimpleBalancer
from smolllm.errors import brief_error_detail
from smolllm.types import RequestEvent, StreamError

MODEL = "testprov/m1"
BASE_URL = "http://test.local/v1"


@pytest.mark.parametrize(
    ("payload", "secret"),
    [
        ("Authorization: Bearer eyJhbGciOiJIUzI1Ni.secret.signature", "eyJhbGciOiJIUzI1Ni.secret.signature"),
        ('{"api_key":"sk-proj-super-secret-value"}', "sk-proj-super-secret-value"),
        ("request failed for AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ1234567", "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ1234567"),
    ],
)
def test_error_details_redact_credential_shapes(payload: str, secret: str) -> None:
    detail = brief_error_detail(payload)

    assert secret not in detail
    assert "[REDACTED_CREDENTIAL]" in detail


@pytest.mark.asyncio
async def test_suspended_key_is_redacted_evicted_and_logged_once(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    dead_key = "AIzaSy" + "D" * 33
    live_key = "live-key"
    key_pool = f"{dead_key},{live_key}"
    selected_keys: list[str] = []
    test_balancer = SimpleBalancer()
    test_balancer.pair_usage[(dead_key, BASE_URL)] = 0
    test_balancer.pair_usage[(live_key, BASE_URL)] = 2

    def handler(request: httpx.Request) -> httpx.Response:
        selected_key = request.headers["authorization"].removeprefix("Bearer ")
        selected_keys.append(selected_key)
        if selected_key == dead_key:
            return httpx.Response(
                403,
                json={
                    "error": {
                        "code": 403,
                        "message": (f"Permission denied: Consumer 'api_key:{dead_key}' has been suspended"),
                        "status": "PERMISSION_DENIED",
                        "details": [{"reason": "CONSUMER_SUSPENDED"}],
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            },
        )

    def prepare_client(_url: str, api_key: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers={"authorization": f"Bearer {api_key}"},
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr(core, "balancer", test_balancer)
    monkeypatch.setattr(core, "prepare_client_and_auth", prepare_client)
    caplog.set_level(logging.WARNING, logger="smolllm")

    with pytest.raises(httpx.HTTPStatusError) as raised:
        await ask_llm(
            "question",
            model=MODEL,
            api_key=key_pool,
            base_url=BASE_URL,
            stream=False,
        )

    first = await ask_llm(
        "question",
        model=MODEL,
        api_key=key_pool,
        base_url=BASE_URL,
        stream=False,
    )
    second = await ask_llm(
        "question",
        model=MODEL,
        api_key=key_pool,
        base_url=BASE_URL,
        stream=False,
    )

    assert first.text == second.text == "ok"
    assert selected_keys == [dead_key, live_key, live_key]
    assert dead_key not in str(raised.value)
    assert dead_key not in caplog.text
    assert "api_key:" not in caplog.text.lower()
    assert caplog.text.count("Evicted permanently rejected API key") == 1


@pytest.mark.asyncio
async def test_empty_transport_error_has_class_status_and_detail(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("", request=request)

    def prepare_client(_url: str, api_key: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers={"authorization": f"Bearer {api_key}"},
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr(core, "prepare_client_and_auth", prepare_client)
    caplog.set_level(logging.WARNING, logger="smolllm")

    with pytest.raises(httpx.ConnectTimeout):
        await ask_llm(
            "question",
            model=MODEL,
            api_key="key",
            base_url=BASE_URL,
            stream=False,
        )

    assert "ConnectTimeout status=n/a detail=no detail" in caplog.text


def test_balancer_fails_fast_when_every_key_is_evicted() -> None:
    balancer = SimpleBalancer()
    assert balancer.evict_pair("dead", BASE_URL) is True
    assert balancer.evict_pair("dead", BASE_URL) is False

    with pytest.raises(RuntimeError, match="No active API keys"):
        balancer.choose_pair("dead", BASE_URL)


def test_eviction_is_scoped_to_key_url_pair() -> None:
    balancer = SimpleBalancer()

    assert balancer.evict_pair("shared-placeholder", "https://rejected.example/v1") is True

    assert balancer.choose_pair("shared-placeholder", "https://healthy.example/v1") == (
        "shared-placeholder",
        "https://healthy.example/v1",
    )


@pytest.mark.asyncio
async def test_caller_hook_text_cannot_evict_healthy_key(monkeypatch: pytest.MonkeyPatch) -> None:
    test_balancer = SimpleBalancer()
    selected_keys: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        selected_keys.append(request.headers["authorization"].removeprefix("Bearer "))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
        )

    def prepare_client(_url: str, api_key: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers={"authorization": f"Bearer {api_key}"},
            transport=httpx.MockTransport(handler),
        )

    def hook(event: RequestEvent) -> None:
        if event.error is None:
            raise RuntimeError("observer rejected event: API_KEY_INVALID")

    monkeypatch.setattr(core, "balancer", test_balancer)
    monkeypatch.setattr(core, "prepare_client_and_auth", prepare_client)

    with pytest.raises(RuntimeError, match="observer rejected"):
        await ask_llm(
            "question",
            model=MODEL,
            api_key="healthy-key",
            base_url=BASE_URL,
            stream=False,
            hook=hook,
        )

    response = await ask_llm(
        "question",
        model=MODEL,
        api_key="healthy-key",
        base_url=BASE_URL,
        stream=False,
    )

    assert response.text == "ok"
    assert selected_keys == ["healthy-key", "healthy-key"]


def _stream_body(content: str) -> str:
    return f'data: {{"choices":[{{"delta":{{"content":"{content}"}},"finish_reason":"stop"}}]}}\n\n'


def _suspended_stream_body() -> str:
    return 'data: {"error":{"message":"Permission denied","details":[{"reason":"CONSUMER_SUSPENDED"}]}}\n\n'


@pytest.mark.asyncio
async def test_default_streaming_ask_evicts_suspended_key(monkeypatch: pytest.MonkeyPatch) -> None:
    dead_key = "dead-key"
    live_key = "live-key"
    key_pool = f"{dead_key},{live_key}"
    selected_keys: list[str] = []
    test_balancer = SimpleBalancer()
    test_balancer.pair_usage[(dead_key, BASE_URL)] = 0
    test_balancer.pair_usage[(live_key, BASE_URL)] = 2

    def handler(request: httpx.Request) -> httpx.Response:
        selected_key = request.headers["authorization"].removeprefix("Bearer ")
        selected_keys.append(selected_key)
        body = _suspended_stream_body() if selected_key == dead_key else _stream_body("ok")
        return httpx.Response(200, text=body)

    def prepare_client(_url: str, api_key: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers={"authorization": f"Bearer {api_key}"},
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr(core, "balancer", test_balancer)
    monkeypatch.setattr(core, "prepare_client_and_auth", prepare_client)

    with pytest.raises(StreamError, match="Permission denied"):
        await ask_llm("question", model=MODEL, api_key=key_pool, base_url=BASE_URL)

    response = await ask_llm("question", model=MODEL, api_key=key_pool, base_url=BASE_URL)

    assert response.text == "ok"
    assert selected_keys == [dead_key, live_key]


@pytest.mark.asyncio
async def test_stream_llm_evicts_suspended_key(monkeypatch: pytest.MonkeyPatch) -> None:
    dead_key = "dead-key"
    live_key = "live-key"
    key_pool = f"{dead_key},{live_key}"
    selected_keys: list[str] = []
    test_balancer = SimpleBalancer()
    test_balancer.pair_usage[(dead_key, BASE_URL)] = 0
    test_balancer.pair_usage[(live_key, BASE_URL)] = 2

    def handler(request: httpx.Request) -> httpx.Response:
        selected_key = request.headers["authorization"].removeprefix("Bearer ")
        selected_keys.append(selected_key)
        body = _suspended_stream_body() if selected_key == dead_key else _stream_body("ok")
        return httpx.Response(200, text=body)

    def prepare_client(_url: str, api_key: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers={"authorization": f"Bearer {api_key}"},
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr(core, "balancer", test_balancer)
    monkeypatch.setattr(core, "prepare_client_and_auth", prepare_client)

    failed = await stream_llm("question", model=MODEL, api_key=key_pool, base_url=BASE_URL)
    with pytest.raises(StreamError, match="Permission denied"):
        _ = [chunk async for chunk in failed]

    succeeded = await stream_llm("question", model=MODEL, api_key=key_pool, base_url=BASE_URL)

    chunks = [str(chunk) async for chunk in succeeded]
    assert "".join(chunks) == "ok"
    assert selected_keys == [dead_key, live_key]


class _PartialThenTimeout(httpx.AsyncByteStream):
    def __init__(self, request: httpx.Request) -> None:
        self.request = request

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield _stream_body("partial").encode()
        raise httpx.ConnectTimeout("", request=self.request)


@pytest.mark.asyncio
async def test_partial_stream_logs_causal_transport_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_PartialThenTimeout(request))

    def prepare_client(_url: str, api_key: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers={"authorization": f"Bearer {api_key}"},
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr(core, "prepare_client_and_auth", prepare_client)
    caplog.set_level(logging.WARNING, logger="smolllm")
    response = await stream_llm("question", model=MODEL, api_key="key", base_url=BASE_URL)
    stream = response.__aiter__()

    first = await anext(stream)
    with pytest.raises(StreamError) as raised:
        await anext(stream)

    assert first.content == "partial"
    assert raised.value.partial == "partial"
    assert "ConnectTimeout status=n/a detail=no detail" in caplog.text
