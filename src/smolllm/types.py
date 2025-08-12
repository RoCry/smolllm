from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    TypeAlias,
    Union,
)


@dataclass
class LLMResponse:
    """Response from LLM with metadata about which model was used"""

    text: str
    # e.g. "gemini/gemini-2.0-flash"
    model: str
    # e.g. "gemini-2.0-flash"
    model_name: str
    # e.g. gemini
    provider: Optional[str] = None

    def __str__(self) -> str:
        """Allow str() conversion for backwards compatibility"""
        return self.text

    def __bool__(self) -> bool:
        """Check if response has content"""
        return bool(self.text and self.text.strip())


@dataclass
class StreamResponse:
    """Wrapper for streaming responses with model metadata"""

    stream: AsyncIterator[str]
    # e.g. "openrouter/google/gemini-2.5-flash"
    model: str
    # e.g. "gemini-2.5-flash"
    model_name: str
    # e.g. openrouter
    provider: Optional[str] = None

    def __aiter__(self):
        """Return self to make this a proper async iterator"""
        return self

    async def __anext__(self):
        """Forward to underlying stream"""
        return await self.stream.__anext__()


StreamHandler: TypeAlias = Callable[[str], None]
LLMFunction: TypeAlias = Callable[
    [str, Optional[str], Any], Awaitable[LLMResponse]
]  # (prompt, system_prompt, **kwargs) -> LLMResponse

MessageRole = Literal["user", "assistant"]
Message = Dict[str, Union[str, List[Dict[str, Any]]]]
PromptType = Union[str, List[Message]]
