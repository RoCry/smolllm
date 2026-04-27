from __future__ import annotations

import base64
import mimetypes
import re
from collections.abc import Sequence

import httpx

from .types import Message, PromptType

_OPENAI_REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")
_OLLAMA_REASONING_EFFORTS = ("none", "low", "medium", "high")


def _has_version_suffix(url: str) -> bool:
    """Check if URL already ends with a version path like /v1, /v2, etc."""
    return bool(re.search(r"/v\d+$", url.rstrip("/")))


def _guess_mime_type(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        raise ValueError(f"Can't guess mime type for: '{image_path}'")
    return mime_type


def _image_path_to_llm_data_str(image_path: str) -> str:
    # check if image_path is already a data string, e.g. data:image/png;base64,...
    if image_path.startswith("data:"):
        return image_path
    mime_type = _guess_mime_type(image_path)
    with open(image_path, "rb") as img_file:
        image_data = base64.b64encode(img_file.read()).decode()
    return f"data:{mime_type};base64,{image_data}"


def _normalize_reasoning_effort(reasoning_effort: str | None, *, provider_name: str) -> str | None:
    if reasoning_effort is None:
        return None
    normalized = reasoning_effort.strip().lower()
    if not normalized:
        raise ValueError("reasoning_effort must not be empty")
    allowed = _OLLAMA_REASONING_EFFORTS if provider_name == "ollama" else _OPENAI_REASONING_EFFORTS
    if normalized not in allowed:
        expected = ", ".join(allowed)
        raise ValueError(
            f"Unsupported reasoning_effort={reasoning_effort!r} for provider={provider_name!r}; expected one of: {expected}"
        )
    return normalized


def _prepare_openai_request(
    prompt: PromptType,
    system_prompt: str | None,
    model_name: str,
    image_paths: Sequence[str],
    stream: bool,
    reasoning_effort: str | None,
) -> dict[str, object]:
    messages: list[Message] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if not isinstance(prompt, str):
        if image_paths:
            raise ValueError(
                "Image paths are not supported with list prompt, you could put the images in the prompt instead"
            )
        messages.extend(prompt)
    else:
        if image_paths:
            content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
            for image_path in image_paths:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_path_to_llm_data_str(image_path)},
                    }
                )
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

    payload: dict[str, object] = {
        "messages": messages,
        "model": model_name,
        "stream": stream,
    }
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    return payload


def _build_endpoint_url(base_url: str, provider_name: str, suffix: str) -> str:
    """Build the full request URL by appending an OpenAI-style endpoint suffix.

    suffix is the path after the version segment, e.g. "chat/completions" or
    "embeddings".
    """
    versioned = _has_version_suffix(base_url)
    if provider_name == "anthropic":
        # [OpenAI SDK compatibility (beta) - Anthropic](https://docs.anthropic.com/en/api/openai-sdk)
        stripped = base_url.rstrip("/")
        return f"{stripped}/{suffix}" if versioned else f"{stripped}/v1/{suffix}"
    if provider_name == "gemini":
        # [OpenAI compatibility | Gemini API](https://ai.google.dev/gemini-api/docs/openai)
        stripped = base_url.rstrip("/")
        return f"{stripped}/{suffix}" if versioned else f"{stripped}/v1beta/openai/{suffix}"
    if base_url.endswith("#"):
        return base_url[:-1]
    if base_url.endswith("/"):
        return f"{base_url}{suffix}"
    if versioned:
        return f"{base_url}/{suffix}"
    return f"{base_url}/v1/{suffix}"


def prepare_request_data(
    prompt: PromptType,
    system_prompt: str | None,
    model_name: str,
    provider_name: str,
    base_url: str,
    image_paths: Sequence[str] | None = None,
    stream: bool = True,
    reasoning_effort: str | None = None,
) -> tuple[str, dict[str, object]]:
    """Prepare request URL, data and headers for the API call"""
    image_path_list = list(image_paths) if image_paths else []
    normalized_reasoning_effort = _normalize_reasoning_effort(reasoning_effort, provider_name=provider_name)
    url = _build_endpoint_url(base_url, provider_name, "chat/completions")
    data = _prepare_openai_request(
        prompt,
        system_prompt,
        model_name,
        image_path_list,
        stream,
        normalized_reasoning_effort,
    )
    return url, data


def prepare_embedding_request_data(
    inputs: str | Sequence[str],
    model_name: str,
    provider_name: str,
    base_url: str,
    dimensions: int | None = None,
) -> tuple[str, dict[str, object]]:
    """Prepare request URL and payload for an OpenAI-compatible embeddings call.

    `dimensions` is forwarded to providers that support output truncation
    (OpenAI text-embedding-3+, Ollama recent versions, and Matryoshka models
    such as qwen3-embedding). Providers that don't recognise the field
    typically ignore it.
    """
    if isinstance(inputs, str):
        if not inputs:
            raise ValueError("input must not be empty")
        payload_input: object = inputs
    else:
        items = list(inputs)
        if not items:
            raise ValueError("input list must not be empty")
        if any(not isinstance(item, str) or not item for item in items):
            raise ValueError("input list entries must be non-empty strings")
        payload_input = items

    if dimensions is not None and dimensions <= 0:
        raise ValueError("dimensions must be a positive integer")

    url = _build_endpoint_url(base_url, provider_name, "embeddings")
    payload: dict[str, object] = {"model": model_name, "input": payload_input}
    if dimensions is not None:
        payload["dimensions"] = dimensions
    return url, payload


def prepare_client_and_auth(
    url: str,
    api_key: str,
) -> httpx.AsyncClient:
    """Prepare HTTP client and handle authentication"""
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}",
    }

    unsecure = url.startswith("http://")

    # Let httpx pick the correct local address family. Forcing an IPv4 bind
    # breaks reachable HTTP endpoints on some networks, including `.local`
    # hosts discovered over mDNS.
    return httpx.AsyncClient(headers=headers, verify=not unsecure)
