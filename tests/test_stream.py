from __future__ import annotations

import pytest

from smolllm.stream import decode_sse_chunk, extract_model, process_chunk_line
from smolllm.types import StreamChunk, StreamError


@pytest.mark.asyncio
async def test_process_chunk_line_returns_content() -> None:
    line = 'data: {"choices":[{"delta":{"content":"hi"}}]}'
    result = await process_chunk_line(line)
    assert isinstance(result, StreamChunk)
    assert result.content == "hi"
    assert result.reasoning == ""


@pytest.mark.asyncio
async def test_process_chunk_line_returns_reasoning_content() -> None:
    line = 'data: {"choices":[{"delta":{"reasoning_content":"thinking..."}}]}'
    result = await process_chunk_line(line)
    assert isinstance(result, StreamChunk)
    assert result.content == ""
    assert result.reasoning == "thinking..."


@pytest.mark.asyncio
async def test_process_chunk_line_returns_reasoning_ollama() -> None:
    """Ollama uses 'reasoning' instead of 'reasoning_content'."""
    line = 'data: {"choices":[{"delta":{"reasoning":"thinking...","content":""}}]}'
    result = await process_chunk_line(line)
    assert isinstance(result, StreamChunk)
    assert result.reasoning == "thinking..."


@pytest.mark.asyncio
async def test_process_chunk_line_both_content_and_reasoning() -> None:
    line = 'data: {"choices":[{"delta":{"content":"answer","reasoning_content":"thought"}}]}'
    result = await process_chunk_line(line)
    assert isinstance(result, StreamChunk)
    assert result.content == "answer"
    assert result.reasoning == "thought"


@pytest.mark.asyncio
async def test_process_chunk_line_raises_on_error_message() -> None:
    line = 'data: {"error":{"message":"boom"}}'
    with pytest.raises(StreamError, match="boom"):
        await process_chunk_line(line)


@pytest.mark.asyncio
async def test_process_chunk_line_raises_on_error_without_message() -> None:
    line = 'data: {"error":{"type":"stream_error"}}'
    with pytest.raises(StreamError, match="Stream error"):
        await process_chunk_line(line)


@pytest.mark.asyncio
async def test_process_chunk_line_updates_usage_from_usage_chunk() -> None:
    usage: dict[str, int] = {}
    line = 'data: {"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}'
    result = await process_chunk_line(line, usage=usage)
    assert result is None
    assert usage == {"prompt_tokens": 11, "completion_tokens": 7}


def test_stream_error_partial() -> None:
    err = StreamError("fail", partial="abc")
    assert err.partial == "abc"


def test_decode_sse_chunk_skips_non_data_lines() -> None:
    assert decode_sse_chunk("") is None
    assert decode_sse_chunk("data: [DONE]") is None
    assert decode_sse_chunk(": keep-alive") is None


def test_extract_model_from_chunk() -> None:
    chunk = decode_sse_chunk('data: {"model":"jake/kimi-2.6","choices":[{"delta":{"content":"hi"}}]}')
    assert chunk is not None
    assert extract_model(chunk) == "jake/kimi-2.6"


def test_extract_model_from_full_response() -> None:
    """extract_model also works on a non-streaming response body."""
    assert extract_model({"model": "gemini/gemini-flash", "choices": []}) == "gemini/gemini-flash"


def test_extract_model_absent_or_blank() -> None:
    assert extract_model({"choices": []}) is None
    assert extract_model({"model": ""}) is None
    assert extract_model("not a mapping") is None
