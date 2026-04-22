from __future__ import annotations

import asyncio

import httpx

from smolllm.request import prepare_client_and_auth, prepare_request_data


def test_prepare_request_data_respects_stream_flag() -> None:
    _, data = prepare_request_data(
        "hi",
        None,
        "test-model",
        "openai",
        "https://api.openai.com",
        stream=False,
    )
    assert data["stream"] is False


def test_prepare_request_data_defaults_stream_true() -> None:
    _, data = prepare_request_data(
        "hi",
        None,
        "test-model",
        "openai",
        "https://api.openai.com",
    )
    assert data["stream"] is True


def test_prepare_request_data_with_reasoning_effort() -> None:
    _, data = prepare_request_data(
        "hi",
        None,
        "test-model",
        "openai",
        "https://api.openai.com",
        reasoning_effort="none",
    )
    assert data["reasoning_effort"] == "none"


def test_prepare_request_data_normalizes_reasoning_effort() -> None:
    _, data = prepare_request_data(
        "hi",
        None,
        "test-model",
        "openai",
        "https://api.openai.com",
        reasoning_effort=" Minimal ",
    )
    assert data["reasoning_effort"] == "minimal"


def test_prepare_request_data_rejects_empty_reasoning_effort() -> None:
    try:
        prepare_request_data(
            "hi",
            None,
            "test-model",
            "openai",
            "https://api.openai.com",
            reasoning_effort="   ",
        )
    except ValueError as exc:
        assert "reasoning_effort" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty reasoning_effort")


def test_prepare_request_data_rejects_unknown_reasoning_effort() -> None:
    try:
        prepare_request_data(
            "hi",
            None,
            "test-model",
            "openai",
            "https://api.openai.com",
            reasoning_effort="minimum",
        )
    except ValueError as exc:
        assert "reasoning_effort" in str(exc)
        assert "openai" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported reasoning_effort")


def test_prepare_request_data_rejects_ollama_unsupported_reasoning_effort() -> None:
    try:
        prepare_request_data(
            "hi",
            None,
            "test-model",
            "ollama",
            "http://localhost:11434",
            reasoning_effort="minimal",
        )
    except ValueError as exc:
        assert "reasoning_effort" in str(exc)
        assert "ollama" in str(exc)
    else:
        raise AssertionError("Expected ValueError for Ollama-specific unsupported reasoning_effort")


def test_prepare_client_and_auth_http_uses_default_transport(monkeypatch) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("prepare_client_and_auth should not override the transport for http URLs")

    monkeypatch.setattr(httpx, "AsyncHTTPTransport", fail_if_called)

    client = prepare_client_and_auth("http://example.com/v1/chat/completions", "token")
    try:
        assert client.headers["authorization"] == "Bearer token"
    finally:
        asyncio.run(client.aclose())


def test_url_skips_version_prefix_when_base_url_has_version() -> None:
    """When base_url already ends with /v1, don't prepend /v1 again."""
    url, _ = prepare_request_data("hi", None, "m", "openai", "https://example.com/v1")
    assert url == "https://example.com/v1/chat/completions"


def test_url_appends_v1_when_base_url_has_no_version() -> None:
    url, _ = prepare_request_data("hi", None, "m", "openai", "https://api.openai.com")
    assert url == "https://api.openai.com/v1/chat/completions"


def test_url_version_suffix_with_higher_version() -> None:
    url, _ = prepare_request_data("hi", None, "m", "openai", "https://example.com/v3")
    assert url == "https://example.com/v3/chat/completions"


def test_url_anthropic_with_version_suffix() -> None:
    url, _ = prepare_request_data("hi", None, "m", "anthropic", "https://proxy.example.com/v1")
    assert url == "https://proxy.example.com/v1/chat/completions"


def test_url_anthropic_without_version_suffix() -> None:
    url, _ = prepare_request_data("hi", None, "m", "anthropic", "https://api.anthropic.com")
    assert url == "https://api.anthropic.com/v1/chat/completions"


def test_url_gemini_with_version_suffix() -> None:
    url, _ = prepare_request_data("hi", None, "m", "gemini", "https://proxy.example.com/v1")
    assert url == "https://proxy.example.com/v1/chat/completions"


def test_url_gemini_without_version_suffix() -> None:
    url, _ = prepare_request_data("hi", None, "m", "gemini", "https://generativelanguage.googleapis.com")
    assert url == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
