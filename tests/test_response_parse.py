from __future__ import annotations

import pytest

from smolllm.core import _extract_text_from_response


def test_extract_text_from_response_message() -> None:
    payload = {"choices": [{"message": {"content": "hi"}}]}
    text, reasoning = _extract_text_from_response(payload)
    assert text == "hi"
    assert reasoning == ""


def test_extract_text_from_response_delta() -> None:
    payload = {"choices": [{"delta": {"content": "hi"}}]}
    text, reasoning = _extract_text_from_response(payload)
    assert text == "hi"
    assert reasoning == ""


def test_extract_text_from_response_text() -> None:
    payload = {"choices": [{"text": "hi"}]}
    text, reasoning = _extract_text_from_response(payload)
    assert text == "hi"
    assert reasoning == ""


def test_extract_text_from_response_missing_content() -> None:
    with pytest.raises(ValueError, match="missing content"):
        _extract_text_from_response({"choices": [{}]})


def test_extract_reasoning_content() -> None:
    payload = {"choices": [{"message": {"content": "answer", "reasoning_content": "thought"}}]}
    text, reasoning = _extract_text_from_response(payload)
    assert text == "answer"
    assert reasoning == "thought"


def test_extract_reasoning_ollama() -> None:
    """Ollama uses 'reasoning' field."""
    payload = {"choices": [{"message": {"content": "answer", "reasoning": "thought"}}]}
    text, reasoning = _extract_text_from_response(payload)
    assert text == "answer"
    assert reasoning == "thought"


def test_extract_think_tags_fallback() -> None:
    payload = {"choices": [{"message": {"content": "<think>reasoning here</think>actual answer"}}]}
    text, reasoning = _extract_text_from_response(payload)
    assert text == "actual answer"
    assert reasoning == "reasoning here"


def test_no_think_tag_extraction_when_reasoning_present() -> None:
    """If reasoning_content is present, don't also extract <think> tags."""
    payload = {"choices": [{"message": {"content": "<think>inline</think>answer", "reasoning_content": "explicit"}}]}
    text, reasoning = _extract_text_from_response(payload)
    assert text == "<think>inline</think>answer"
    assert reasoning == "explicit"
