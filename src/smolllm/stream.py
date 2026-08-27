from __future__ import annotations

import json
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import cast

from .errors import brief_error_detail, extract_error_reason_codes
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


def extract_tool_call_deltas(chunk: Mapping[str, object]) -> list[Mapping[str, object]]:
    """Pull the raw ``delta.tool_calls`` fragments out of a decoded chunk."""
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return []
    delta = cast(Mapping[str, object], choice).get("delta")
    if not isinstance(delta, Mapping):
        return []
    calls = cast(Mapping[str, object], delta).get("tool_calls")
    if not isinstance(calls, list):
        return []
    return [cast(Mapping[str, object], call) for call in calls if isinstance(call, Mapping)]


@dataclass(slots=True)
class ToolCallAccumulator:
    """Reassembles streamed tool-call deltas into whole calls.

    Providers stream a tool call across many frames: the first carries ``id`` and
    the function name, later ones append fragments of the argument JSON. Fragments
    are never pushed to stream handlers — a caller can only act on a complete call,
    so the assembled list is exposed once the stream ends.
    """

    _slots: dict[int, dict[str, object]] = field(default_factory=dict)
    _last_slot: int | None = None

    def feed(self, chunk: Mapping[str, object]) -> None:
        for delta in extract_tool_call_deltas(chunk):
            self._merge(delta)

    def _slot_for(self, delta: Mapping[str, object]) -> int:
        index = delta.get("index")
        if isinstance(index, int):
            return index
        # Providers that omit `index` start a new call whenever they send a fresh
        # `id`; everything else continues the call already in progress.
        if delta.get("id") or self._last_slot is None:
            return max(self._slots, default=-1) + 1
        return self._last_slot

    def _merge(self, delta: Mapping[str, object]) -> None:
        slot = self._slot_for(delta)
        self._last_slot = slot
        call = self._slots.setdefault(slot, {})

        for key in ("id", "type"):
            value = delta.get(key)
            if isinstance(value, str) and value:
                call[key] = value

        function = delta.get("function")
        if not isinstance(function, Mapping):
            return
        merged = cast(dict[str, object], call.setdefault("function", {}))
        name = cast(Mapping[str, object], function).get("name")
        if isinstance(name, str) and name:
            merged["name"] = name
        arguments = cast(Mapping[str, object], function).get("arguments")
        if isinstance(arguments, str):
            previous = merged.get("arguments")
            merged["arguments"] = (previous if isinstance(previous, str) else "") + arguments

    def result(self) -> list[dict[str, object]]:
        """The assembled calls, ordered by their provider-assigned index."""
        calls: list[dict[str, object]] = []
        for _, call in sorted(self._slots.items()):
            assembled = dict(call)
            function = assembled.get("function")
            if isinstance(function, dict):
                merged = cast(dict[str, object], dict(function))
                merged.setdefault("arguments", "")
                assembled["function"] = merged
            calls.append(assembled)
        return calls


def update_usage(chunk: Mapping[str, object], usage: MutableMapping[str, int]) -> None:
    """Merge token counts from a decoded chunk's ``usage`` object into ``usage``.

    Providers may split prompt/completion counts across chunks, so fields merge
    individually rather than replacing the whole mapping.
    """
    usage_obj = chunk.get("usage")
    if not isinstance(usage_obj, Mapping):
        return
    prompt_tokens = usage_obj.get("prompt_tokens")
    completion_tokens = usage_obj.get("completion_tokens")
    if isinstance(prompt_tokens, int):
        usage["prompt_tokens"] = prompt_tokens
    if isinstance(completion_tokens, int):
        usage["completion_tokens"] = completion_tokens


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
        message = brief_error_detail(f"Malformed streaming chunk: {payload}")
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
        reason_codes = extract_error_reason_codes(error_obj)
        if isinstance(error_obj, Mapping):
            message = error_obj.get("message")
            if isinstance(message, str) and message.strip():
                raise StreamError(brief_error_detail(message), reason_codes=reason_codes)
        raise StreamError("Stream error", reason_codes=reason_codes)
    return chunk


def extract_finish_reason(payload: Mapping[str, object]) -> str | None:
    """Read ``choices[0].finish_reason`` from a decoded chunk or full response.

    Works for both streaming chunks and non-streaming bodies (both carry it at the
    same path). Returns None when absent or null — i.e. content chunks mid-stream,
    or a stream that ended without its mandatory terminal finish_reason frame.
    """
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return None
    finish_reason = choice.get("finish_reason")
    if isinstance(finish_reason, str) and finish_reason:
        return finish_reason
    return None


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


async def process_chunk_line(line: str, *, usage: MutableMapping[str, int] | None = None) -> StreamChunk | None:
    """Process a single SSE line, returning a StreamChunk or None."""
    chunk = decode_sse_chunk(line)
    if chunk is None:
        return None
    if usage is not None:
        update_usage(chunk, usage)
    return extract_delta(chunk)
