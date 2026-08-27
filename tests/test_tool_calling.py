from __future__ import annotations

import httpx
import pytest

import smolllm.core as core
from smolllm import ask_llm, stream_llm
from smolllm.response import extract_tool_calls
from smolllm.stream import ToolCallAccumulator, decode_sse_chunk

MODEL = "testprov/m1"
BASE_URL = "http://test.local/v1"


# --------------------------------------------------------------- response side


def test_extract_tool_calls_from_message() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"SF"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    calls = extract_tool_calls(payload)
    assert len(calls) == 1
    assert calls[0]["id"] == "call_1"
    assert calls[0]["function"] == {"name": "get_weather", "arguments": '{"city":"SF"}'}


def test_extract_tool_calls_absent_is_empty() -> None:
    assert extract_tool_calls({"choices": [{"message": {"content": "hi"}}]}) == []


def test_null_content_with_tool_calls_is_not_an_error() -> None:
    """The empty-content guard must not fire on a legal tool-call response."""
    payload = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
                }
            }
        ]
    }
    text, reasoning = core._extract_text_from_response(payload)
    assert text == ""
    assert reasoning == ""


def test_missing_content_without_tool_calls_still_raises() -> None:
    with pytest.raises(ValueError, match="missing content"):
        core._extract_text_from_response({"choices": [{"message": {}}]})


# --------------------------------------------------------------- accumulator


def test_accumulator_merges_argument_fragments() -> None:
    acc = ToolCallAccumulator()
    for frame in (
        '{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function",'
        '"function":{"name":"get_weather","arguments":""}}]}}]}',
        '{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"ci"}}]}}]}',
        '{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ty\\":\\"SF\\"}"}}]}}]}',
    ):
        chunk = decode_sse_chunk(f"data: {frame}")
        assert chunk is not None
        acc.feed(chunk)

    calls = acc.result()
    assert len(calls) == 1
    assert calls[0] == {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city":"SF"}'},
    }


def test_accumulator_keeps_parallel_calls_separate_and_ordered() -> None:
    acc = ToolCallAccumulator()
    acc.feed(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 1, "id": "b", "type": "function", "function": {"name": "second"}},
                            {"index": 0, "id": "a", "type": "function", "function": {"name": "first"}},
                        ]
                    }
                }
            ]
        }
    )
    acc.feed({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}}]})

    calls = acc.result()
    assert [c["id"] for c in calls] == ["a", "b"]
    assert calls[0]["function"] == {"name": "first", "arguments": "{}"}


def test_accumulator_without_index_starts_new_call_on_id() -> None:
    """Some providers omit `index`; a fresh `id` means a new call, otherwise append."""
    acc = ToolCallAccumulator()
    acc.feed({"choices": [{"delta": {"tool_calls": [{"id": "a", "type": "function", "function": {"name": "f"}}]}}]})
    acc.feed({"choices": [{"delta": {"tool_calls": [{"function": {"arguments": '{"x":1}'}}]}}]})
    acc.feed({"choices": [{"delta": {"tool_calls": [{"id": "b", "type": "function", "function": {"name": "g"}}]}}]})

    calls = acc.result()
    assert [c["id"] for c in calls] == ["a", "b"]
    assert calls[0]["function"]["arguments"] == '{"x":1}'


def test_accumulator_is_empty_for_plain_text_stream() -> None:
    acc = ToolCallAccumulator()
    chunk = decode_sse_chunk('data: {"choices":[{"delta":{"content":"hi"}}]}')
    assert chunk is not None
    acc.feed(chunk)
    assert acc.result() == []


# --------------------------------------------------------------- end to end


def _mock_client(handler: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))  # pyright: ignore[reportArgumentType]


@pytest.mark.asyncio
async def test_ask_llm_returns_tool_calls_non_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        import json

        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "resolved",
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": '{"city":"SF"}'},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )

    monkeypatch.setattr(core, "prepare_client_and_auth", lambda _u, _k: _mock_client(respond))

    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
    response = await ask_llm(
        "weather in SF?",
        model=MODEL,
        api_key="secret",
        base_url=BASE_URL,
        stream=False,
        extra_body={"tools": tools, "tool_choice": "auto"},
    )

    assert response.text == ""
    assert response.finish_reason == "tool_calls"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["function"]["name"] == "get_weather"
    assert bool(response) is True, "a tool-call response is a useful response"
    assert sent[0]["tools"] == tools
    assert sent[0]["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_ask_llm_accumulates_tool_calls_when_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = [
        'data: {"model":"resolved","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
        '"type":"function","function":{"name":"get_weather","arguments":""}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"city"}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\":\\"SF\\"}"}}]}}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
        "data: [DONE]",
    ]

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="\n\n".join(frames))

    monkeypatch.setattr(core, "prepare_client_and_auth", lambda _u, _k: _mock_client(respond))

    response = await ask_llm(
        "weather in SF?",
        model=MODEL,
        api_key="secret",
        base_url=BASE_URL,
        extra_body={"tools": [{"type": "function", "function": {"name": "get_weather"}}]},
    )

    assert response.text == ""
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0]["function"]["arguments"] == '{"city":"SF"}'


@pytest.mark.asyncio
async def test_stream_llm_exposes_tool_calls_after_the_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = [
        'data: {"model":"resolved","choices":[{"delta":{"content":"looking it up"}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function",'
        '"function":{"name":"get_weather","arguments":"{}"}}]}}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
        "data: [DONE]",
    ]

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="\n\n".join(frames))

    monkeypatch.setattr(core, "prepare_client_and_auth", lambda _u, _k: _mock_client(respond))

    response = await stream_llm("weather in SF?", model=MODEL, api_key="secret", base_url=BASE_URL)
    chunks = [chunk.content async for chunk in response]

    assert "".join(chunks) == "looking it up"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["id"] == "call_1"
    assert response.finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_stream_llm_survives_a_tool_calls_only_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """No text and no reasoning is legal when the model answered with tool calls."""
    frames = [
        'data: {"model":"resolved","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
        '"type":"function","function":{"name":"f","arguments":"{}"}}]}}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
        "data: [DONE]",
    ]

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="\n\n".join(frames))

    monkeypatch.setattr(core, "prepare_client_and_auth", lambda _u, _k: _mock_client(respond))

    response = await stream_llm("go", model=MODEL, api_key="secret", base_url=BASE_URL)
    chunks = [chunk.content async for chunk in response]

    assert chunks == []
    assert len(response.tool_calls) == 1


@pytest.mark.asyncio
async def test_tool_role_messages_round_trip_to_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """The caller executes the tool and replays the assistant + tool turns."""
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        import json

        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "It is sunny."}, "finish_reason": "stop"}]},
        )

    monkeypatch.setattr(core, "prepare_client_and_auth", lambda _u, _k: _mock_client(respond))

    conversation = [
        {"role": "user", "content": "weather in SF?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"SF"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temp_c":21}'},
    ]

    response = await ask_llm(
        conversation,  # pyright: ignore[reportArgumentType]
        model=MODEL,
        api_key="secret",
        base_url=BASE_URL,
        stream=False,
    )

    assert response.text == "It is sunny."
    messages = sent[0]["messages"]
    assert messages == conversation, "tool and tool_calls messages must pass through untouched"


@pytest.mark.asyncio
async def test_empty_response_without_tool_calls_still_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]})

    monkeypatch.setattr(core, "prepare_client_and_auth", lambda _u, _k: _mock_client(respond))

    with pytest.raises(ValueError, match="empty response"):
        _ = await ask_llm("hi", model=MODEL, api_key="secret", base_url=BASE_URL, stream=False)
