from __future__ import annotations

import pytest

from smolllm.stream import process_chunk_line
from smolllm.types import StreamError


@pytest.mark.asyncio
async def test_process_chunk_line_returns_content() -> None:
    line = 'data: {"choices":[{"delta":{"content":"hi"}}]}'
    assert await process_chunk_line(line) == "hi"


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
