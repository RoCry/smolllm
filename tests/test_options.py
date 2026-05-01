from __future__ import annotations

import pytest

from smolllm import RequestEvent, Usage
from smolllm.request import prepare_request_data
from smolllm.utils import preview_api_key


def test_temperature_passed_through_to_payload() -> None:
    _, data = prepare_request_data(
        "hi",
        None,
        "test-model",
        "openai",
        "https://api.openai.com",
        temperature=0.7,
    )
    assert data["temperature"] == 0.7
    assert "top_p" not in data


def test_top_p_passed_through_to_payload() -> None:
    _, data = prepare_request_data(
        "hi",
        None,
        "test-model",
        "openai",
        "https://api.openai.com",
        top_p=0.9,
    )
    assert data["top_p"] == 0.9
    assert "temperature" not in data


def test_temperature_and_top_p_together() -> None:
    _, data = prepare_request_data(
        "hi",
        None,
        "test-model",
        "openai",
        "https://api.openai.com",
        temperature=0.5,
        top_p=0.95,
    )
    assert data["temperature"] == 0.5
    assert data["top_p"] == 0.95


def test_temperature_default_omitted() -> None:
    _, data = prepare_request_data("hi", None, "m", "openai", "https://api.openai.com")
    assert "temperature" not in data
    assert "top_p" not in data


@pytest.mark.parametrize("bad", [-0.1, 2.5, 3.0])
def test_temperature_out_of_range_raises(bad: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        prepare_request_data("hi", None, "m", "openai", "https://api.openai.com", temperature=bad)


@pytest.mark.parametrize("bad", [-0.1, 1.5, 2.0])
def test_top_p_out_of_range_raises(bad: float) -> None:
    with pytest.raises(ValueError, match="top_p"):
        prepare_request_data("hi", None, "m", "openai", "https://api.openai.com", top_p=bad)


@pytest.mark.parametrize("good", [0.0, 1.0, 1.5, 2.0])
def test_temperature_in_range_accepted(good: float) -> None:
    _, data = prepare_request_data("hi", None, "m", "openai", "https://api.openai.com", temperature=good)
    assert data["temperature"] == good


@pytest.mark.parametrize("good", [0.0, 0.5, 1.0])
def test_top_p_in_range_accepted(good: float) -> None:
    _, data = prepare_request_data("hi", None, "m", "openai", "https://api.openai.com", top_p=good)
    assert data["top_p"] == good


def test_preview_api_key_truncates_long_keys() -> None:
    assert preview_api_key("sk-1234567890ab") == "sk-12...90ab"


def test_preview_api_key_short_key_returned_as_is() -> None:
    assert preview_api_key("short") == "short"
    assert preview_api_key("123456789") == "123456789"


def test_usage_dataclass_shape() -> None:
    usage = Usage(
        provider="openai",
        model="openai/gpt-4o",
        model_name="gpt-4o",
        api_key_hint="sk-12...90ab",
        input_tokens=10,
        output_tokens=20,
        duration_ms=123,
    )
    assert usage.provider == "openai"
    assert usage.model == "openai/gpt-4o"
    assert usage.model_name == "gpt-4o"
    assert usage.api_key_hint == "sk-12...90ab"
    assert usage.input_tokens == 10
    assert usage.output_tokens == 20
    assert usage.duration_ms == 123
    assert usage.ttft_ms is None


def test_request_event_dataclass_shape() -> None:
    usage = Usage(
        provider="openai",
        model="openai/gpt-4o",
        model_name="gpt-4o",
        api_key_hint="sk-12...90ab",
        input_tokens=1,
        output_tokens=2,
        duration_ms=3,
        ttft_ms=4,
    )
    err = RuntimeError("boom")
    event = RequestEvent(usage=usage, error=err, timestamp=1700000000.0)
    assert event.usage is usage
    assert event.error is err
    assert event.timestamp == 1700000000.0
