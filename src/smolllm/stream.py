from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

from .log import logger


def _handle_chunk(chunk: Mapping[str, object]) -> str | None:
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
    content_candidate = delta.get("content")
    if content_candidate is None:
        return None
    if not isinstance(content_candidate, str):
        raise TypeError("Chunk content must be a string")
    return content_candidate


async def process_chunk_line(line: str) -> str | None:
    """Process a single chunk of data from the stream"""
    line = line.strip()
    if not line or line == "data: [DONE]" or not line.startswith("data: "):
        return None
    payload = line[6:]
    try:
        chunk_raw_obj = cast(object, json.loads(payload))
        if not isinstance(chunk_raw_obj, dict):
            raise TypeError("Streaming chunk must decode into a mapping")
        chunk_raw = cast(dict[object, object], chunk_raw_obj)
        chunk: dict[str, object] = {}
        for key_obj, value in chunk_raw.items():
            if not isinstance(key_obj, str):
                raise TypeError("Streaming chunk keys must be strings")
            chunk[key_obj] = value
        return _handle_chunk(chunk)
    except json.JSONDecodeError as exc:
        message = f"Malformed streaming chunk: {payload}"
        logger.error(message)
        raise ValueError(message) from exc
