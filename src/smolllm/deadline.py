from __future__ import annotations

from time import monotonic

DEFAULT_TIMEOUT_S = 600.0


class Deadline:
    """One wall-clock deadline bounding a whole call: every fallback leg, every retry, every chunk.

    Mirrors smolllm-go's `Timeout`. A per-chunk read timeout alone cannot bound a call: a server that
    trickles keepalive bytes, or a model looping without a terminal finish_reason, keeps it alive forever.
    """

    __slots__ = ("_at", "timeout")

    def __init__(self, timeout: float) -> None:
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")
        self.timeout = timeout
        self._at = monotonic() + timeout

    def remaining(self) -> float:
        """Seconds left; raises TimeoutError once the deadline has passed."""
        left = self._at - monotonic()
        if left <= 0:
            raise TimeoutError(f"call exceeded its {self.timeout:g}s timeout")
        return left
