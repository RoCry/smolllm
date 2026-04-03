from __future__ import annotations

import pytest

from smolllm.stream import process_chunk_line
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


def test_stream_error_partial() -> None:
    err = StreamError("fail", partial="abc")
    assert err.partial == "abc"
