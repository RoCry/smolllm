# Extreme minimalism: chat + embeddings over the OpenAI-compat wire, nothing else

SmolLLM's value is its resilience layer — multi-key/URL balancing, model fallback chains, token metering — over the OpenAI-compatible wire protocol. Broad multi-modality clients pay for their breadth with hand-rolled native transports and hardcoded per-provider dispatch, and still tend to lack the three things a multi-provider client needs most: key balancing, fallback, and retries. Depth over breadth is the bet.

Decision: scope is frozen at chat + embeddings; no new modalities, no native provider transports, and **no new API surface without a real in-house consumer to exercise it** (a feature nobody calls cannot be tested or experienced, only maintained). Features that pass design review but lack a consumer get a design note in `docs/DEFERRED.md`, not code.

## Consequences

- Tool calling, `extra_body` escape hatch, JSON mode: designs agreed, implementation deferred (see docs/DEFERRED.md).
- Rejected outright (don't re-propose): modality expansion (image gen/ASR/rerank/classify), native provider transports, prompt-based tool emulation, automatic JSON repair, embedded pricing catalogs.
- A provider or parameter combination reachable only over a non-OpenAI-compat wire (e.g. OpenAI `/v1/responses` for gpt-6.x tools + `reasoning_effort`) is fronted by the smolayer gateway, which translates to the OpenAI-compat wire; clients never translate (smolllm-go#13 → smolayer#26).
- The Responses wire stays in smolayer (decision 2026-11-23): smolllm speaks only Chat Completions. smolayer owns every other wire and every subscription login, on the Mac only; smolllm-server is the single front door (aliases, fallback, key balancing, usage ledger). What smolayer's Responses state needs on the Chat Completions side travels as opaque fields a client echoes back, e.g. `reasoning_items` on the assistant message (smolllm#10); the client never translates or reads them.
