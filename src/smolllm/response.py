from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from .utils import extract_think_tags


def _first_choice(payload: object) -> Mapping[str, object] | None:
    if not isinstance(payload, Mapping):
        return None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, Mapping):
        return None
    return cast(Mapping[str, object], first)


def _tool_calls_in(choice: Mapping[str, object]) -> list[dict[str, object]]:
    """Read tool calls out of a choice's ``message`` or ``delta`` container."""
    for container_key in ("message", "delta"):
        container = choice.get(container_key)
        if not isinstance(container, Mapping):
            continue
        calls = cast(Mapping[str, object], container).get("tool_calls")
        if isinstance(calls, list) and calls:
            return [cast(dict[str, object], call) for call in calls if isinstance(call, dict)]
    return []


def extract_tool_calls(payload: object) -> list[dict[str, object]]:
    """Return the provider's tool calls verbatim, or an empty list.

    Dicts are passed through unmodified — smolllm neither validates the argument
    JSON nor normalizes schemas; the caller executes the calls.
    """
    choice = _first_choice(payload)
    if choice is None:
        return []
    return _tool_calls_in(choice)


def extract_text_from_response(payload: object) -> tuple[str, str]:
    """Extract text and reasoning from a non-streaming response.

    Returns (text, reasoning).
    """
    if not isinstance(payload, dict):
        raise TypeError("Response payload must be a mapping")

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Response payload missing choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise TypeError("Response choice must be a mapping")

    reasoning = ""
    content: str | None = None

    for container_key in ("message", "delta"):
        container = first.get(container_key)
        if not isinstance(container, dict):
            continue
        if not reasoning:
            for rk in ("reasoning_content", "reasoning"):
                rc = container.get(rk)
                if isinstance(rc, str) and rc:
                    reasoning = rc
                    break
        if content is None:
            c = container.get("content")
            if isinstance(c, str):
                content = c

    if content is None:
        text = first.get("text")
        if isinstance(text, str):
            content = text

    if content is None:
        # A tool-call turn legally carries null content; anything else is malformed.
        if _tool_calls_in(cast(Mapping[str, object], first)):
            return "", reasoning
        raise ValueError("Response choice missing content")

    if not reasoning and content:
        extracted_reasoning, clean_content = extract_think_tags(content)
        if extracted_reasoning:
            reasoning = extracted_reasoning
            content = clean_content

    return content, reasoning
