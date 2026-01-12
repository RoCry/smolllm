from __future__ import annotations

from smolllm.request import prepare_request_data


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
