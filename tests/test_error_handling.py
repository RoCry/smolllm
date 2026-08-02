from __future__ import annotations

import logging

import httpx
import pytest

import smolllm.core as core
from smolllm import ask_llm
from smolllm.balancer import SimpleBalancer
from smolllm.errors import brief_error_detail

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
    test_balancer.pair_usage[(live_key, BASE_URL)] = 1

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
    assert balancer.evict_key("dead") is True
    assert balancer.evict_key("dead") is False

    with pytest.raises(RuntimeError, match="No active API keys"):
        balancer.choose_pair("dead", BASE_URL)
