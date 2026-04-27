"""
smolllm - A minimal LLM library for easy interaction with various LLM providers
"""

from .core import ask_llm, embed_llm, stream_llm
from .types import (
    EmbedResponse,
    LLMFunction,
    LLMResponse,
    Message,
    MessageRole,
    ModelInput,
    PromptType,
    StreamChunk,
    StreamError,
    StreamHandler,
    StreamResponse,
)

__version__ = "0.5.1"
__all__ = [
    "ask_llm",
    "embed_llm",
    "stream_llm",
    "EmbedResponse",
    "LLMFunction",
    "ModelInput",
    "StreamChunk",
    "StreamHandler",
    "PromptType",
    "Message",
    "MessageRole",
    "LLMResponse",
    "StreamResponse",
    "StreamError",
]
