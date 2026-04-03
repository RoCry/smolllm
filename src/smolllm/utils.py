from __future__ import annotations

import re

from .types import StreamChunk

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def extract_think_tags(text: str) -> tuple[str, str]:
    """Extract <think> blocks from text.

    Returns (reasoning, clean_content) where reasoning is all think-block
    content concatenated and clean_content has the tags stripped.
    """
    reasoning_parts: list[str] = []
    clean = _THINK_RE.sub(lambda m: (reasoning_parts.append(m.group(1).strip()), "")[1], text)
    return "\n\n".join(reasoning_parts), clean.strip()


class ThinkTagFilter:
    """Stateful streaming filter that reclassifies content inside <think> tags as reasoning.

    Auto-disables if the backend already provides reasoning_content in the delta.
    Handles tag splits across chunk boundaries via buffering.
    """

    def __init__(self) -> None:
        self._inside_think = False
        self._buffer = ""
        self._disabled = False

    def feed(self, chunk: StreamChunk) -> StreamChunk:
        """Process a StreamChunk, reclassifying <think> content as reasoning."""
        # If backend already provides reasoning_content, pass through as-is
        if chunk.reasoning:
            self._disabled = True
            return chunk
        if self._disabled:
            return chunk

        text = self._buffer + chunk.content
        self._buffer = ""

        reasoning_parts: list[str] = []
        content_parts: list[str] = []

        while text:
            if self._inside_think:
                end = text.find("</think>")
                if end == -1:
                    # Might have a partial closing tag at the end
                    for i in range(min(len("</think>") - 1, len(text)), 0, -1):
                        if "</think>"[:i] == text[-i:]:
                            self._buffer = text[-i:]
                            text = text[:-i]
                            break
                    if text:
                        reasoning_parts.append(text)
                    break
                reasoning_parts.append(text[:end])
                text = text[end + len("</think>") :]
                self._inside_think = False
            else:
                start = text.find("<think>")
                if start == -1:
                    # Might have a partial opening tag at the end
                    for i in range(min(len("<think>") - 1, len(text)), 0, -1):
                        if "<think>"[:i] == text[-i:]:
                            self._buffer = text[-i:]
                            text = text[:-i]
                            break
                    if text:
                        content_parts.append(text)
                    break
                if start > 0:
                    content_parts.append(text[:start])
                text = text[start + len("<think>") :]
                self._inside_think = True

        return StreamChunk(
            content="".join(content_parts),
            reasoning="".join(reasoning_parts),
        )

    def flush(self) -> StreamChunk:
        """Flush any buffered content at end of stream."""
        if not self._buffer:
            return StreamChunk()
        buf = self._buffer
        self._buffer = ""
        if self._inside_think:
            return StreamChunk(reasoning=buf)
        return StreamChunk(content=buf)


def strip_backticks(text: str) -> str:
    """Strip backticks from the beginning and end of a string"""
    # must be starts with ``` and ends with ```
    if not text.startswith("```") or not text.endswith("```"):
        return text

    lines = text.split("\n")

    # Remove first line if it contains backticks
    if lines[0].startswith("```"):
        lines = lines[1:]

    # Remove last line if it contains backticks
    if lines[-1].endswith("```"):
        if lines[-1] == "```":
            lines = lines[:-1]
        else:
            lines[-1] = lines[-1][:-3]

    return "\n".join(lines)
