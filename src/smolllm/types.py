from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict, override


@dataclass(slots=True)
class LLMResponse:
    """High-level response container with provider metadata."""

    text: str
    model: str  # e.g. "gemini/gemini-2.0-flash"
    model_name: str  # e.g. "gemini-2.0-flash"
    provider: str | None = None

    @override
    def __str__(self) -> str:
        return self.text

    def __bool__(self) -> bool:
        return bool(self.text and self.text.strip())


@dataclass(slots=True)
class StreamResponse:
    """Wrapper for streaming responses with model metadata."""

    stream: AsyncIterator[str]
    model: str  # e.g. "openrouter/google/gemini-2.5-flash"
    model_name: str  # e.g. "gemini-2.5-flash"
    provider: str | None = None

    def __aiter__(self) -> AsyncIterator[str]:
        return self.stream

    async def __anext__(self) -> str:
        return await self.stream.__anext__()


StreamHandler = Callable[[str], Awaitable[None]]


class LLMFunction(Protocol):
    async def __call__(
        self,
        prompt: PromptType,
        *,
        system_prompt: str | None = ...,
        model: str | Sequence[str] | None = ...,
        api_key: str | None = ...,
        base_url: str | None = ...,
        handler: StreamHandler | None = ...,
        timeout: float = ...,
        remove_backticks: bool = ...,
        image_paths: Sequence[str] | None = ...,
    ) -> LLMResponse:
        """Protocol describing the callable shape expected for LLM functions."""
        ...


MessageRole = Literal["user", "assistant", "system"]


class Message(TypedDict):
    role: MessageRole
    content: str | Sequence[dict[str, object]]


PromptType = str | Sequence[Message]
