"""
smolllm - A minimal LLM library for easy interaction with various LLM providers
"""

from .core import ask_llm, stream_llm
from .types import (
    LLMFunction,
    LLMResponse,
    Message,
    MessageRole,
    ModelInput,
    PromptType,
    StreamError,
    StreamHandler,
    StreamResponse,
)

__version__ = "0.5.1"
__all__ = [
    "ask_llm",
    "stream_llm",
    "LLMFunction",
    "ModelInput",
    "StreamHandler",
    "PromptType",
    "Message",
    "MessageRole",
    "LLMResponse",
    "StreamResponse",
    "StreamError",
]
