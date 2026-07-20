# SmolLLM

Minimal client for many LLM providers over the OpenAI-compatible wire protocol: one interface, API-key/endpoint balancing, model fallback. This glossary covers the Python lib; smolllm-go mirrors the same language.

## Language

**Provider**:
A named OpenAI-compatible endpoint (e.g. `openai`, `groq`); credentials and base URL resolve from env by name.
_Avoid_: vendor, backend.

**Model spec**:
The user-facing model string `provider/model[!effort]`; comma-separated specs form a fallback chain.

**Fallback chain**:
Ordered or weighted candidate models; on failure the call advances to the next candidate.
_Avoid_: confusing with retry.

**Retry**:
Re-attempt of the *same* model after a transient failure. Distinct from fallback (which switches models).

**Balancer pair**:
One (API key, base URL) combination for a provider; the least-used pair is chosen per call.

**Output budget**:
The `max_tokens` cap a caller declares on completion length. Application knowledge — the library never injects one; unset means the provider's own default applies.
_Avoid_: implying a library-side default exists.

**Estimated usage**:
Token counts derived by heuristic when the provider omits usage; always marked (`~` prefix, `Estimated` flag).

**Reasoning**:
Model thinking text, kept in a channel separate from content.
_Avoid_: mixing reasoning into content.

**FinishReason**:
Verbatim provider string explaining why generation ended; never normalized.

**Request hook**:
Per-attempt observation callback receiving usage or error; the library's only telemetry surface.

**Escape hatch**:
A pass-through (`extra_body`) letting callers set raw request fields the library does not model. Deferred — see docs/DEFERRED.md.
_Avoid_: raw options, extra params.
