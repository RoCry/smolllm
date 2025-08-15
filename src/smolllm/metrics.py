"""Metrics and timing utilities for LLM responses."""


def estimate_tokens(text: str) -> int:
    """Estimate token count using a simple heuristic (4 chars per token)."""
    return len(text) // 4


def format_metrics(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    total_time: float,
    ttft_ms: int = None,
) -> str:
    """Format metrics for logging with emojis.

    Args:
        model_name: The name of the model
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        total_time: Total time in seconds
        ttft_ms: Time to first token in milliseconds (optional, for streaming)

    Returns:
        Formatted metrics string with emojis
    """
    total_tokens = input_tokens + output_tokens
    tok_per_sec = int(output_tokens / total_time) if total_time > 0 else 0

    metrics = f"📊{model_name} {total_tokens}tok (↑{input_tokens} ↓{output_tokens})"
    if ttft_ms is not None:
        metrics += f" | 🚀{_format_time(ttft_ms)}"

    metrics += f" | 🐎{tok_per_sec}tok/s | ⌛{_format_time(total_time * 1000.0)}"

    return metrics


def _format_time(ms: float) -> str:
    return f"{ms / 1000.0:.1f}s" if ms >= 1000 else f"{ms}ms"
