# Extreme minimalism: chat + embeddings over the OpenAI-compat wire, nothing else

SmolLLM's value is its resilience layer — multi-key/URL balancing, model fallback chains, token metering — over the OpenAI-compatible wire protocol. Broad multi-modality clients pay for their breadth with hand-rolled native transports and hardcoded per-provider dispatch, and still tend to lack the three things a multi-provider client needs most: key balancing, fallback, and retries. Depth over breadth is the bet.

Decision: scope is frozen at chat + embeddings; no new modalities, no native provider transports, and **no new API surface without a real in-house consumer to exercise it** (a feature nobody calls cannot be tested or experienced, only maintained). Features that pass design review but lack a consumer get a design note in `docs/DEFERRED.md`, not code.

## Consequences

- Tool calling, `extra_body` escape hatch, JSON mode: designs agreed, implementation deferred (see docs/DEFERRED.md).
- Rejected outright (don't re-propose): modality expansion (image gen/ASR/rerank/classify), native provider transports, prompt-based tool emulation, automatic JSON repair, embedded pricing catalogs.
