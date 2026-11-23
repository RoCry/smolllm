"""Replay smolayer's recorded Chat Completions SSE and check it against the shared expectations.

``tests/fixtures/smolayer/`` is a byte-identical copy of ``smolllm-go/smolllm/testdata/smolayer/``
(see its README): the expected files are derived from the recorded bytes with jq, not by smolllm.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

import smolllm.core as core
from smolllm import ask_llm, stream_llm

FIXTURES = Path(__file__).parent / "fixtures" / "smolayer"
NAMES = sorted(p.name.removesuffix(".sse") for p in FIXTURES.glob("*.sse"))
BASE_URL = "http://test.local/v1"
MODEL = "testprov/m1"


def _expected(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.expected.json").read_text())


def _serve(monkeypatch: pytest.MonkeyPatch, bodies: dict[str, bytes]) -> list[dict[str, Any]]:
    """Answer each request with the body registered for its model; return the requests seen."""
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body)
        return httpx.Response(200, content=bodies[body["model"]])

    monkeypatch.setattr(
        core, "prepare_client_and_auth", lambda _u, _k: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    return sent


def _fixture_bytes(name: str) -> bytes:
    return (FIXTURES / f"{name}.sse").read_bytes()


def _assert_tool_calls(actual: list[dict[str, object]], want: dict[str, Any]) -> None:
    got = [(c["id"], cast(dict[str, str], c["function"])) for c in actual]
    assert [(call_id, fn["name"], fn["arguments"]) for call_id, fn in got] == [
        (c["id"], c["name"], c["arguments"]) for c in want["tool_calls"]
    ]


def test_fixture_set_is_the_shared_one() -> None:
    assert NAMES == ["openai_reasoning_tool_call", "reasoning_tool_call", "text", "tool_call"]


@pytest.mark.asyncio
@pytest.mark.parametrize("name", NAMES)
async def test_ask_llm_stream_matches_expectations(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    want = _expected(name)
    _ = _serve(monkeypatch, {"m1": _fixture_bytes(name)})

    response = await ask_llm("hi", model=MODEL, api_key="k", base_url=BASE_URL)

    assert response.text == want["content"]
    assert response.reasoning == want["reasoning"]
    assert response.finish_reason == want["finish_reason"]
    assert response.reasoning_items == want["reasoning_items"]
    _assert_tool_calls(response.tool_calls, want)
    assert response.usage is not None and not response.usage.estimated
    # smolllm Python keeps no cache/reasoning split: prompt and completion counts only.
    assert response.usage.input_tokens == want["usage"]["prompt_tokens"]
    assert response.usage.output_tokens == want["usage"]["completion_tokens"]


@pytest.mark.asyncio
@pytest.mark.parametrize("name", NAMES)
async def test_stream_llm_matches_expectations(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    want = _expected(name)
    _ = _serve(monkeypatch, {"m1": _fixture_bytes(name)})

    response = await stream_llm("hi", model=MODEL, api_key="k", base_url=BASE_URL)
    content = "".join([chunk.content async for chunk in response])

    assert content == want["content"]
    assert response.reasoning == want["reasoning"]
    assert response.finish_reason == want["finish_reason"]
    assert response.reasoning_items == want["reasoning_items"]
    _assert_tool_calls(response.tool_calls, want)
    assert response.usage is not None and not response.usage.estimated
    assert response.usage.input_tokens == want["usage"]["prompt_tokens"]
    assert response.usage.output_tokens == want["usage"]["completion_tokens"]


@pytest.mark.asyncio
async def test_tool_loop_sends_reasoning_items_back_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Step 1 streams a tool call with reasoning items; step 2 replays them on the assistant message."""
    want = _expected("reasoning_tool_call")
    sent = _serve(monkeypatch, {"m1": _fixture_bytes("reasoning_tool_call")})

    first = await ask_llm("weather in Paris?", model=MODEL, api_key="k", base_url=BASE_URL)
    assert first.reasoning_items == want["reasoning_items"]

    _ = await ask_llm(
        [
            {"role": "user", "content": "weather in Paris?"},
            {
                "role": "assistant",
                "content": first.text or None,
                "tool_calls": first.tool_calls,
                "reasoning_items": first.reasoning_items,
            },
            {"role": "tool", "tool_call_id": want["tool_calls"][0]["id"], "content": "18C and sunny"},
        ],
        model=MODEL,
        api_key="k",
        base_url=BASE_URL,
    )

    replayed = sent[1]["messages"][1]
    assert replayed["reasoning_items"] == want["reasoning_items"]
    assert replayed["tool_calls"][0]["id"] == want["tool_calls"][0]["id"]


@pytest.mark.asyncio
async def test_reasoning_items_of_a_failed_leg_are_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A token minted for a leg that failed names a provider that did not answer; only the winner's survive."""

    def sse(*chunks: dict[str, object]) -> bytes:
        return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks).encode() + b"data: [DONE]\n\n"

    failing = sse(
        {"model": "a", "choices": [{"delta": {"reasoning_items": ["rsn_FROM_A"]}}]},
        {"model": "a", "choices": [{"delta": {}, "finish_reason": "error"}], "error": {"message": "upstream reset"}},
    )
    winning = sse(
        {"model": "b", "choices": [{"delta": {"reasoning_items": ["rsn_FROM_B"]}}]},
        {"model": "b", "choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]},
    )
    _ = _serve(monkeypatch, {"a": failing, "b": winning})

    response = await ask_llm("hi", model=["testprov/a", "testprov/b"], api_key="k", base_url=BASE_URL)

    assert response.text == "ok"
    assert response.reasoning_items == ["rsn_FROM_B"]


@pytest.mark.asyncio
async def test_non_streamed_message_reasoning_items_are_exposed(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "m",
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "reasoning_items": ["rsn_X"],
                            "tool_calls": [
                                {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )

    monkeypatch.setattr(
        core, "prepare_client_and_auth", lambda _u, _k: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    response = await ask_llm("hi", model=MODEL, api_key="k", base_url=BASE_URL, stream=False)
    assert response.reasoning_items == ["rsn_X"]
