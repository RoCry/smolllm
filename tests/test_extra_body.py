from __future__ import annotations

import pytest

from smolllm.request import prepare_request_data


def _data(**kwargs: object) -> dict[str, object]:
    _, data = prepare_request_data(
        "hi",
        None,
        "test-model",
        "openai",
        "https://api.openai.com",
        **kwargs,  # pyright: ignore[reportArgumentType]
    )
    return data


def test_extra_body_fields_reach_the_payload() -> None:
    data = _data(extra_body={"response_format": {"type": "json_object"}, "service_tier": "flex"})
    assert data["response_format"] == {"type": "json_object"}
    assert data["service_tier"] == "flex"


def test_extra_body_wins_over_library_defaults() -> None:
    """Caller-set fields are merged last; a modelled field can be overridden."""
    data = _data(temperature=0.1, extra_body={"temperature": 1.5})
    assert data["temperature"] == 1.5


def test_extra_body_is_absent_when_not_passed() -> None:
    data = _data()
    assert "response_format" not in data


@pytest.mark.parametrize("key", ["stream", "stream_options", "messages", "model"])
def test_reserved_keys_fail_fast(key: str) -> None:
    with pytest.raises(ValueError, match=f"extra_body may not set {key!r}"):
        _ = _data(extra_body={key: "anything"})


def test_reserved_key_check_names_every_offender() -> None:
    with pytest.raises(ValueError) as excinfo:
        _ = _data(extra_body={"model": "x", "stream": False})
    message = str(excinfo.value)
    assert "'model'" in message and "'stream'" in message


def test_extra_body_does_not_mutate_the_callers_dict() -> None:
    extra = {"tools": [{"type": "function", "function": {"name": "f"}}]}
    before = repr(extra)
    _ = _data(extra_body=extra)
    assert repr(extra) == before
