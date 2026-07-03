from __future__ import annotations

from .utils import extract_think_tags


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
        raise ValueError("Response choice missing content")

    if not reasoning and content:
        extracted_reasoning, clean_content = extract_think_tags(content)
        if extracted_reasoning:
            reasoning = extracted_reasoning
            content = clean_content

    return content, reasoning
