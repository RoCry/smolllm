"""
smolllm - A minimal LLM library for easy interaction with various LLM providers
"""

from .core import ask_llm, stream_llm
from .types import LLMFunction, Message, MessageRole, PromptType, StreamHandler

__version__ = "0.3.1"
__all__ = [
    "ask_llm",
    "stream_llm",
    "LLMFunction",
    "StreamHandler",
    "PromptType",
    "Message",
    "MessageRole",
]
