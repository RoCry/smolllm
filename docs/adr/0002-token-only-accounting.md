# Usage stops at tokens; cost is the caller's concern

The library reports per-attempt token usage (`Usage` with `Estimated` flag, request hook) and goes no further. A pricing catalog is a data-maintenance liability, not code — hand-maintained price tables go stale silently, and a library that embeds one carries that staleness risk forever. Cost estimation belongs to callers that aggregate usage (e.g. smolllm-server), sourcing prices from LiteLLM's model_prices data (the same upstream llm-api-price syncs hourly).

Guidance for callers that do price: fail closed — unknown model/rate → cost is null, never guessed; estimated tokens → cost marked estimated (`~`).
