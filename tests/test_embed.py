from __future__ import annotations

import pytest

from smolllm.request import prepare_embedding_request_data


def test_embedding_url_appends_v1_for_unversioned_base() -> None:
    url, _ = prepare_embedding_request_data("hi", model_name="m", provider_name="ollama", base_url="http://x:11434")
    assert url == "http://x:11434/v1/embeddings"


def test_embedding_url_respects_existing_version_suffix() -> None:
    url, _ = prepare_embedding_request_data("hi", model_name="m", provider_name="openai", base_url="https://x.com/v1")
    assert url == "https://x.com/v1/embeddings"


def test_embedding_url_hash_suffix_is_used_verbatim() -> None:
    url, _ = prepare_embedding_request_data(
        "hi", model_name="m", provider_name="custom", base_url="https://x.com/api/embed#"
    )
    assert url == "https://x.com/api/embed"


def test_embedding_payload_omits_dimensions_when_none() -> None:
    _, data = prepare_embedding_request_data("hi", model_name="m", provider_name="openai", base_url="https://x.com")
    assert data == {"model": "m", "input": "hi"}


def test_embedding_payload_includes_dimensions_when_provided() -> None:
    _, data = prepare_embedding_request_data(
        "hi", model_name="m", provider_name="openai", base_url="https://x.com", dimensions=256
    )
    assert data["dimensions"] == 256


def test_embedding_payload_accepts_list_input() -> None:
    _, data = prepare_embedding_request_data(
        ["a", "b"], model_name="m", provider_name="openai", base_url="https://x.com"
    )
    assert data["input"] == ["a", "b"]


def test_embedding_rejects_empty_string() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        prepare_embedding_request_data("", model_name="m", provider_name="openai", base_url="https://x.com")


def test_embedding_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        prepare_embedding_request_data([], model_name="m", provider_name="openai", base_url="https://x.com")


def test_embedding_rejects_blank_list_entry() -> None:
    with pytest.raises(ValueError, match="non-empty strings"):
        prepare_embedding_request_data(["a", ""], model_name="m", provider_name="openai", base_url="https://x.com")


def test_embedding_rejects_non_positive_dimensions() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        prepare_embedding_request_data(
            "hi", model_name="m", provider_name="openai", base_url="https://x.com", dimensions=0
        )
