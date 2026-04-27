from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict, override

ModelInput = str | Sequence[str] | set[str] | dict[str, float | int]


class StreamError(RuntimeError):
    """Raised when a streaming response fails mid-flight."""

    def __init__(self, message: str, *, partial: str | None = None) -> None:
        self.partial = partial
        super().__init__(message)


@dataclass(slots=True)
class StreamChunk:
    """A single chunk from a streaming response, separating content and reasoning.

    Note: ``str(chunk)`` returns only ``.content`` — reasoning must be read
    via ``.reasoning`` explicitly.  This keeps ``print(chunk, end="")``
    backward-compatible with code that expects plain answer text.
    """

    content: str = ""
    reasoning: str = ""

    @override
    def __str__(self) -> str:
        # Only content — reasoning is opt-in via .reasoning
        return self.content

    def __bool__(self) -> bool:
        return bool(self.content or self.reasoning)


@dataclass(slots=True)
class LLMResponse:
    """High-level response container with provider metadata."""

    text: str
    model: str  # e.g. "gemini/gemini-2.0-flash"
    model_name: str  # e.g. "gemini-2.0-flash"
    provider: str | None = None
    reasoning: str = ""

    @override
    def __str__(self) -> str:
        return self.text

    def __bool__(self) -> bool:
        return bool(self.text and self.text.strip())


@dataclass(slots=True)
class EmbedResponse:
    """Container for embedding vectors returned by an embeddings call.

    `embeddings` is always a list of vectors, even when a single string was
    passed in — index `[0]` for the single-input case.
    """

    embeddings: list[list[float]]
    model: str  # e.g. "ollama/qwen3-embedding:4b"
    model_name: str  # e.g. "qwen3-embedding:4b"
    dimensions: int  # actual length of the returned vectors
    provider: str | None = None
    prompt_tokens: int | None = None

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> list[float]:
        return self.embeddings[idx]


@dataclass(slots=True)
class StreamResponse:
    """Wrapper for streaming responses with model metadata.

    Yields ``StreamChunk`` objects during iteration.  After the stream
    completes, accumulated ``reasoning`` is available on this object
    (mirrors ``LLMResponse.reasoning``).
    """

    stream: AsyncIterator[StreamChunk]
    model: str  # e.g. "openrouter/google/gemini-2.5-flash"
    model_name: str  # e.g. "gemini-2.5-flash"
    provider: str | None = None
    reasoning: str = ""

    def __aiter__(self) -> AsyncIterator[StreamChunk]:
        return self

    async def __anext__(self) -> StreamChunk:
        chunk = await self.stream.__anext__()
        if chunk.reasoning:
            self.reasoning += chunk.reasoning
        return chunk

    async def display(self, handler: StreamHandler | None = None) -> str:
        """Consume the stream with rich terminal display (like ask_llm).

        Returns the final response text. Reasoning is accumulated on
        ``self.reasoning``.
        """
        from .display import ResponseDisplay

        with ResponseDisplay(handler) as disp:
            async for chunk in self:
                await disp.update(chunk)
            text, _ = disp.finalize()
        return text


StreamHandler = Callable[[StreamChunk], Awaitable[None]]


class LLMFunction(Protocol):
    async def __call__(
        self,
        prompt: PromptType,
        *,
        system_prompt: str | None = ...,
        model: ModelInput | None = ...,
        api_key: str | None = ...,
        base_url: str | None = ...,
        handler: StreamHandler | None = ...,
        timeout: float = ...,
        remove_backticks: bool = ...,
        image_paths: Sequence[str] | None = ...,
        stream: bool = ...,
        reasoning_effort: str | None = ...,
    ) -> LLMResponse:
        """Protocol describing the callable shape expected for LLM functions."""
        ...


MessageRole = Literal["user", "assistant", "system"]


class Message(TypedDict):
    role: MessageRole
    content: str | Sequence[dict[str, object]]


PromptType = str | Sequence[Message]
