from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod

from .types import ModelInput


class ModelSelector(ABC):
    """Base class for model selection strategies."""

    @abstractmethod
    def next_model(self) -> str | None:
        """Return next model to try, or None if exhausted."""


class SequentialSelector(ModelSelector):
    """Tries models in order. Used for str and list[str]."""

    def __init__(self, models: list[str]):
        self._models = iter(models)

    def next_model(self) -> str | None:
        return next(self._models, None)


class RandomSelector(ModelSelector):
    """Random selection with optional weights. Used for set and dict."""

    def __init__(self, models: set[str] | dict[str, float | int]):
        if isinstance(models, set):
            self._weights = {m: 1.0 for m in models}
        else:
            self._weights = dict(models)
        self._remaining = set(self._weights.keys())

    def next_model(self) -> str | None:
        if not self._remaining:
            return None
        models = list(self._remaining)
        weights = [self._weights[m] for m in models]
        chosen = random.choices(models, weights=weights, k=1)[0]
        self._remaining.remove(chosen)
        return chosen


def create_selector(model: ModelInput | None) -> ModelSelector:
    """Create appropriate model selector based on input type."""
    candidate: ModelInput | None = model if model is not None else os.getenv("SMOLLLM_MODEL")
    if candidate is None:
        raise ValueError("Model string not found. Set SMOLLLM_MODEL environment variable or pass model parameter")

    if isinstance(candidate, set):
        sanitized = {m.strip() for m in candidate if m.strip()}
        if not sanitized:
            raise ValueError("Model set must contain at least one non-empty entry")
        return RandomSelector(sanitized)

    if isinstance(candidate, dict):
        sanitized = {k.strip(): v for k, v in candidate.items() if k.strip()}
        if not sanitized:
            raise ValueError("Model dict must contain at least one non-empty entry")
        if any(w <= 0 for w in sanitized.values()):
            raise ValueError("Model weights must be positive")
        return RandomSelector(sanitized)

    if isinstance(candidate, str):
        models = [m.strip() for m in candidate.split(",") if m.strip()]
        if not models:
            raise ValueError("Model string must contain at least one non-empty entry")
        return SequentialSelector(models)

    models = [item.strip() for item in candidate]
    if not models or any(not item for item in models):
        raise ValueError("Model sequence entries must be non-empty strings")
    return SequentialSelector(models)
