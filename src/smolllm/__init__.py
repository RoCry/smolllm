"""
smolllm - A minimal LLM library for easy interaction with various LLM providers
"""

from .core import ask_llm, stream_llm
from .types import (
    LLMFunction,
    LLMResponse,
    Message,
    MessageRole,
    PromptType,
    StreamHandler,
    StreamResponse,
)

__version__ = "0.4.0"
__all__ = [
    "ask_llm",
    "stream_llm",
    "LLMFunction",
    "StreamHandler",
    "PromptType",
    "Message",
    "MessageRole",
    "LLMResponse",
    "StreamResponse",
]
