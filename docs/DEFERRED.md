# Deferred designs

Conclusions from a 2026-07-11 design review. Each feature passed review but has **no real in-house consumer today**, so per [ADR-0001](adr/0001-extreme-minimalism.md) it stays unimplemented. When a real use case appears, implement per the recorded design — the thinking is done.

## `extra_body` escape hatch

**Status**: deferred — no current caller needs unmodeled request fields.

**Design**:
- Python: `extra_body: dict[str, Any] | None = None` on `ask_llm`/`stream_llm`; Go: `WithExtraBody(map[string]any)`.
- Shallow-merged into the payload **last** — caller wins over library defaults.
- Reserved keys fail fast (`ValueError`/panic): `stream`, `stream_options`, `messages`, `model` — library machinery (stream parser, usage collection, routing) depends on them.
- No `extra_headers` until a real header-based need appears (YAGNI).
- Name matches the openai SDK's `extra_body` — zero learning curve.

**Why it's first in line**: ~20 LOC; permanently ends pressure to model new request fields (`seed`, `service_tier`, `response_format`, provider-private params).

## Tool calling

**Status**: deferred — nothing in-house issues tool calls through smolllm, so it cannot be tested or experienced for real.

**Design** (response-side only; request side rides `extra_body={"tools": [...]}` — signatures stay untouched):
- Accept `tool`-role messages and assistant messages carrying `tool_calls` (Go `Prompt.Validate()` currently rejects them; must be relaxed).
- Surface raw `tool_calls` (list of dicts — no typed ToolCall class) plus `finish_reason` on responses.
- Streaming: accumulate tool-call deltas internally, expose after stream end; no partial-JSON pushes to handlers.
- Empty-content guard fix: a response with `tool_calls` and empty content is legal (today it raises).
- **No agentic loop** — the caller executes tools and appends messages. Because smolllm speaks only the OpenAI-compat wire, the assistant turn replays losslessly; clients with native Gemini/Anthropic transports must reconstruct replay messages to preserve opaque state (thought signatures) — smolllm is immune by architecture.
- **No prompt-based tool emulation** (sentinel protocols + injected system prompts for tool-less models): a whole subsystem with poor ROI now that mainstream models have native tools.
- **No JSON-schema normalization** (e.g. auto-injecting `items:{}` into array schemas some providers reject); fail fast — the provider error is clear enough.
- smolllm-server: currently hard-rejects `tools`; after Go support, forward them (emitting tool_calls at stream end is acceptable to mainstream OpenAI clients).

## JSON mode / `response_format`

**Status**: absorbed — no implementation ever needed; former roadmap item deleted.

`extra_body={"response_format": {...}}` is complete support once the escape hatch exists; the response side is unchanged (content is still text). Ship a docs example only. **Never auto-repair model JSON** — repair heuristics (balancing brace counts on truncated output) fabricate valid-but-wrong data; fence-stripping (`remove_backticks`) is the maximum.

## `SMOLLLM_MAX_TOKENS` env default

**Status**: deferred — exactly one app has been bitten (mindmap truncated at a relay-injected 4096 default, rescued by chain fallback at the cost of a wasted full-length attempt); it now declares `max_tokens` itself.

**Context** (2026-07 survey of primary sources): the OpenAI-compat ecosystem treats max output tokens as optional and never injected — official SDKs and proxies (openai-python, LiteLLM, OpenRouter, one-api) pass it through untouched. The only injection point anywhere is the OpenAI→Anthropic conversion boundary, where the target API *requires* the field (one-api injects 4096, new-api 8192). Provider defaults when the field is omitted vary wildly (Kimi K3: 131072; DeepSeek V3-era: 4096; some relays: 4096). smolllm follows the ecosystem: the library and smolllm-server never invent a value. Output budget is application knowledge — the library cannot know whether truncation is acceptable, and a safe universal value would need a per-model capability catalog (out of scope per [ADR-0001](adr/0001-extreme-minimalism.md)). Oversized values are not free either: strict providers 400 when the value exceeds the model's output cap or when `input + max_tokens` exceeds the context window.

**Design**: env var `SMOLLLM_MAX_TOKENS` read as the default for an unset `max_tokens`; an explicit argument always wins. Go reads the same name during option resolution. Opt-in via user env only — never a hardcoded library constant.

**Un-defer trigger**: a second in-house app wastes a real run on `finish_reason="length"` truncation.

## Health-aware balancer

**Status**: deferred — `/stats` failures data does not yet show a dead-key pattern.

**Design**: add a failure-cooldown circuit breaker to the key/URL balancer, temporarily removing failing pairs from selection. Implement only when `/stats` per-bucket failure data shows a dead-key pattern; that evidence is the un-defer trigger.

Personal key pools are small, so cooldown could empty the whole pool; the required balancer state machine is not small relative to the library; and 429 already falls through to the next model, covering the common quota-exhaustion case.

## Cost accounting

**Status**: step 2 (cost layer after the token ledger) — delegated to callers; unchanged per [ADR-0002](adr/0002-token-only-accounting.md).

The smolllm-server token-only usage ledger (`/stats`) is the substrate cost accounting would sit on. When cost is implemented: LiteLLM `model_prices` JSON as the price source (fetched/vendored, never hand-maintained); fail closed (unknown model → null cost, never guessed); propagate the `~` estimation marker from token counts to cost; v1 prices input/output rates only (no cache/reasoning tiers).
