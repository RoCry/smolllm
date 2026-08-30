# SmolLLM

Minimal client for many LLM providers over the OpenAI-compatible wire protocol: one interface, API-key/endpoint balancing, model fallback. This glossary covers the Python lib; smolllm-go mirrors the same language.

## Language

**Provider**:
A named OpenAI-compatible endpoint (e.g. `openai`, `groq`); credentials and base URL resolve from env by name.
_Avoid_: vendor, backend.

**Model spec**:
The user-facing model string `provider/model[!effort]`, or a bare `model[!effort]` without `/`; comma-separated specs form a fallback chain and may mix both forms.

**Actual model**:
Best available identity of the model that produced a response: the server-reported model when present, otherwise the winning model spec. Exposed as `LLMResponse.actual_model`; this records the server's claim, not independent verification. Raw `model` and `resolved_model` remain available for routing and audit.

**Bare model**:
A model spec with no provider: base URL and API key must be passed explicitly (no env fallback); provider identity stays empty in responses, usage, and hook events.

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
A pass-through (`extra_body`) letting callers set raw request fields the library does not model, merged last so the caller wins. The fields the library machinery reads back (`stream`, `stream_options`, `messages`, `model`) are rejected.
_Avoid_: raw options, extra params.

**Tool call**:
A provider-issued request to run a named function, surfaced verbatim as a dict. The caller executes it and replays the assistant and `tool` messages; the library runs no agentic loop and never inspects or repairs the argument JSON.
_Avoid_: function call, tool use, implying smolllm executes anything.
