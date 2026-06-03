from __future__ import annotations

from smolllm.core import _is_truncated
from smolllm.stream import decode_sse_chunk, extract_finish_reason


def test_extract_finish_reason_from_terminal_chunk() -> None:
    chunk = decode_sse_chunk('data: {"choices":[{"delta":{},"finish_reason":"stop"}]}')
    assert chunk is not None
    assert extract_finish_reason(chunk) == "stop"


def test_extract_finish_reason_length() -> None:
    chunk = decode_sse_chunk('data: {"choices":[{"delta":{},"finish_reason":"length"}]}')
    assert chunk is not None
    assert extract_finish_reason(chunk) == "length"


def test_extract_finish_reason_null_on_content_chunk() -> None:
    chunk = decode_sse_chunk('data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}')
    assert chunk is not None
    assert extract_finish_reason(chunk) is None


def test_extract_finish_reason_absent() -> None:
    chunk = decode_sse_chunk('data: {"choices":[{"delta":{"content":"hi"}}]}')
    assert chunk is not None
    assert extract_finish_reason(chunk) is None


def test_extract_finish_reason_from_full_response() -> None:
    """Non-streaming responses expose finish_reason at the same path."""
    assert extract_finish_reason({"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]}) == "stop"


def test_extract_finish_reason_no_choices() -> None:
    assert extract_finish_reason({"choices": []}) is None


def test_length_finish_reason_is_truncated() -> None:
    assert _is_truncated("length", has_content=True, stream=True) is True
    assert _is_truncated("length", has_content=True, stream=False) is True


def test_stop_finish_reason_is_complete() -> None:
    assert _is_truncated("stop", has_content=True, stream=True) is False
    assert _is_truncated("stop", has_content=True, stream=False) is False


def test_missing_finish_reason_in_stream_with_content_is_truncated() -> None:
    """A stream that yielded content but never sent a terminal finish_reason
    chunk was cut short (dropped connection / upstream terminated early)."""
    assert _is_truncated(None, has_content=True, stream=True) is True


def test_missing_finish_reason_non_stream_is_complete() -> None:
    """A non-streaming body arrives whole; absent finish_reason is not truncation."""
    assert _is_truncated(None, has_content=True, stream=False) is False


def test_no_content_is_not_truncated() -> None:
    """Empty responses are handled by the separate empty-response path."""
    assert _is_truncated(None, has_content=False, stream=True) is False
    assert _is_truncated("length", has_content=False, stream=True) is False
