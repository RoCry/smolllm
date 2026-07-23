"""
smolllm - A minimal LLM library for easy interaction with various LLM providers
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import ask_llm as ask_llm
    from .core import stream_llm as stream_llm
    from .embeddings import embed_llm as embed_llm

from .types import EmbedResponse as EmbedResponse
from .types import Hook as Hook
from .types import LLMFunction as LLMFunction
from .types import LLMResponse as LLMResponse
from .types import Message as Message
from .types import MessageRole as MessageRole
from .types import ModelInput as ModelInput
from .types import PromptType as PromptType
from .types import RequestEvent as RequestEvent
from .types import StreamChunk as StreamChunk
from .types import StreamError as StreamError
from .types import StreamHandler as StreamHandler
from .types import StreamResponse as StreamResponse
from .types import Usage as Usage

__version__ = "0.8.0"
_LAZY_EXPORTS = {
    "ask_llm": ".core",
    "stream_llm": ".core",
    "embed_llm": ".embeddings",
}
__all__ = [
    *_LAZY_EXPORTS,
    "EmbedResponse",
    "Hook",
    "LLMFunction",
    "ModelInput",
    "RequestEvent",
    "StreamChunk",
    "StreamHandler",
    "PromptType",
    "Message",
    "MessageRole",
    "LLMResponse",
    "StreamResponse",
    "StreamError",
    "Usage",
]


def __getattr__(name: str) -> object:
    try:
        module_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
