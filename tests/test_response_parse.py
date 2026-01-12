from __future__ import annotations

import pytest

from smolllm.core import _extract_text_from_response


def test_extract_text_from_response_message() -> None:
    payload = {"choices": [{"message": {"content": "hi"}}]}
    assert _extract_text_from_response(payload) == "hi"


def test_extract_text_from_response_delta() -> None:
    payload = {"choices": [{"delta": {"content": "hi"}}]}
    assert _extract_text_from_response(payload) == "hi"


def test_extract_text_from_response_text() -> None:
    payload = {"choices": [{"text": "hi"}]}
    assert _extract_text_from_response(payload) == "hi"


def test_extract_text_from_response_missing_content() -> None:
    with pytest.raises(ValueError, match="missing content"):
        _extract_text_from_response({"choices": [{}]})
