"""
smolllm - A minimal LLM library for easy interaction with various LLM providers
"""

from .core import ask_llm
from .types import LLMFunction, StreamHandler

__version__ = "0.1.10"
__all__ = ["ask_llm", "LLMFunction", "StreamHandler"]
