# smolayer Chat Completions SSE fixtures

Shared conformance set: smolllm-go (`smolllm/testdata/smolayer/`) and smolllm
(`tests/fixtures/smolayer/`) keep byte-identical copies (`diff -r` them when touching one).

`*.sse` are the bytes smolayer (the gateway, `127.0.0.1:8765`) emitted on `POST /v1/chat/completions`
with `stream: true`, recorded live 2026-11-23, with `stream_options.include_usage` and one
`get_weather(city)` tool where the name says tool:

| fixture | model | prompt |
|---|---|---|
| `text` | `codex/gpt-6.1-terra` | `Reply with exactly: pong` |
| `tool_call` | `codex/gpt-6.1-terra` | `What is the weather in Paris? Use the tool.` |
| `reasoning_tool_call` | `codex/gpt-6.1-terra!medium` | `Work out by hand what 17389 * 4271 is, then call get_weather for Paris. Reply with the product and the weather.` |
| `openai_reasoning_tool_call` | `openai/gpt-6.1-terra!low` | same as above |

`*.expected.json` is derived from the raw SSE by `derive-expected.sh` (jq, run inside this directory), not by any smolllm
code. Wire counts: `usage.prompt_tokens` includes `cached_tokens`; a client that keeps no cache or
reasoning split (smolllm Python) checks only `prompt_tokens`/`completion_tokens`.
`reasoning_items` are opaque `rsn_` tokens, compared byte for byte.
