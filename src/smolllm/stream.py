from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

from .log import logger
from .types import StreamChunk, StreamError


def extract_delta(chunk: Mapping[str, object]) -> StreamChunk | None:
    """Pull content/reasoning out of a decoded OpenAI-shaped chunk."""
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    if not isinstance(choices[0], Mapping):
        raise TypeError("Chunk choice must be a mapping")
    choice = cast(Mapping[str, object], choices[0])
    delta_candidate = choice.get("delta")
    if delta_candidate is None:
        return None
    if not isinstance(delta_candidate, Mapping):
        raise TypeError("Chunk delta must be a mapping")
    delta = cast(Mapping[str, object], delta_candidate)

    content = ""
    content_candidate = delta.get("content")
    if isinstance(content_candidate, str):
        content = content_candidate

    reasoning = ""
    # "reasoning_content" (DeepSeek, vLLM, LiteLLM) or "reasoning" (Ollama)
    for key in ("reasoning_content", "reasoning"):
        reasoning_candidate = delta.get(key)
        if isinstance(reasoning_candidate, str) and reasoning_candidate:
            reasoning = reasoning_candidate
            break

    if not content and not reasoning:
        return None
    return StreamChunk(content=content, reasoning=reasoning)


def decode_sse_chunk(line: str) -> dict[str, object] | None:
    """Decode one SSE ``data:`` frame into a chunk dict.

    Returns None for blank, ``[DONE]``, or non-``data:`` lines. Raises StreamError
    on error frames and ValueError on malformed JSON.
    """
    line = line.strip()
    if not line or line == "data: [DONE]" or not line.startswith("data: "):
        return None
    payload = line[6:]
    try:
        chunk_raw_obj = cast(object, json.loads(payload))
    except json.JSONDecodeError as exc:
        message = f"Malformed streaming chunk: {payload}"
        logger.error(message)
        raise ValueError(message) from exc
    if not isinstance(chunk_raw_obj, dict):
        raise TypeError("Streaming chunk must decode into a mapping")
    chunk_raw = cast(dict[object, object], chunk_raw_obj)
    chunk: dict[str, object] = {}
    for key_obj, value in chunk_raw.items():
        if not isinstance(key_obj, str):
            raise TypeError("Streaming chunk keys must be strings")
        chunk[key_obj] = value
    if "error" in chunk:
        error_obj = chunk.get("error")
        if isinstance(error_obj, Mapping):
            message = error_obj.get("message")
            if isinstance(message, str) and message.strip():
                raise StreamError(message.strip())
        raise StreamError("Stream error")
    return chunk


def extract_model(payload: object) -> str | None:
    """Read the upstream-reported model from a decoded chunk/response dict.

    Works for both streaming chunks and full responses (both carry a top-level
    ``model``). Returns None when absent or not a non-empty string.
    """
    if isinstance(payload, Mapping):
        model = payload.get("model")
        if isinstance(model, str) and model:
            return model
    return None


async def process_chunk_line(line: str) -> StreamChunk | None:
    """Process a single SSE line, returning a StreamChunk or None."""
    chunk = decode_sse_chunk(line)
    return extract_delta(chunk) if chunk is not None else None
