"""
smolllm - A minimal LLM library for easy interaction with various LLM providers
"""

from .core import ask_llm, embed_llm, stream_llm
from .types import (
    EmbedResponse,
    Hook,
    LLMFunction,
    LLMResponse,
    Message,
    MessageRole,
    ModelInput,
    PromptType,
    RequestEvent,
    StreamChunk,
    StreamError,
    StreamHandler,
    StreamResponse,
    Usage,
)

__version__ = "0.7.0"
__all__ = [
    "ask_llm",
    "embed_llm",
    "stream_llm",
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
