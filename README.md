# SmolLLM

A minimal Python library for interacting with various LLM providers, featuring automatic API key load balancing and streaming responses.

## Installation

```bash
pip install smolllm
uv add "smolllm @ ../smolllm"
```

## Quick Start

```python
import asyncio
from smolllm import ask_llm

async def main():
    response = await ask_llm(
        "Say hello world",
        model="gemini/gemini-2.0-flash"
    )
    print(response)
    print(response.actual_model)

if __name__ == "__main__":
    asyncio.run(main())
```

`response.model` is the winning requested model spec. `response.resolved_model`
is the optional model identity reported by the server. Use
`response.actual_model` for the best available identity: reported when present,
otherwise requested.

## Reasoning Effort

For reasoning-capable models, pass `reasoning_effort` through to the provider.
Useful for Ollama reasoning models where omitting it can make replies much slower.

```python
response = await ask_llm(
    "Reply with one word.",
    model="ollama/qwen3.5:0.8b",
    base_url="http://rocry-ubuntu.local:11434/v1",
    reasoning_effort="none",
)
```

## Images

`image_paths` accepts file paths or `data:` URLs. Each image becomes an
`image_url` part on the last user message: the prompt string, or the last
entry of a message list. Files are base64-encoded with a mime type guessed
from the extension.

```python
response = await ask_llm(
    "What is in this picture?",
    model="openai/gpt-4o",
    image_paths=["photo.jpg", "data:image/png;base64,iVBORw0..."],
)
```

With a message list the last message must be a user turn with content, or a
`ValueError` is raised.

## Tool Calling

Pass tools through the `extra_body` escape hatch; assembled tool calls come back on
the response. smolllm never executes a tool — you run the loop and replay the results
as `assistant` + `tool` messages. See [examples/tool_calling.py](examples/tool_calling.py).

```python
response = await ask_llm(messages, extra_body={"tools": TOOLS})
for call in response.tool_calls:
    name = call["function"]["name"]
    args = json.loads(call["function"]["arguments"])
```

`extra_body` also carries any other raw request field the library does not model
(`response_format`, `service_tier`, provider-private params); it is merged last, so
you win over library defaults. Fields the library reads back — `stream`,
`stream_options`, `messages`, `model` — are rejected.

## Provider Configuration

Format: `provider/model_name` (e.g., `openai/gpt-4`, `gemini/gemini-2.0-flash`)

A bare model name (no `/`) targets an endpoint directly — pass `base_url` and `api_key` explicitly (no env fallback):

```python
response = await ask_llm(
    "hi",
    model="gpt-4",
    base_url="https://my-proxy.example/v1",
    api_key="sk-xxx",
)
```

### API Keys

The library looks for API keys in environment variables following the pattern: `{PROVIDER}_API_KEY`

Example:
```bash
# .env
OPENAI_API_KEY=sk-xxx
GEMINI_API_KEY=key1,key2  # Multiple keys supported
```

Key/endpoint pairs rejected with a permanent credential status (for example
`CONSUMER_SUSPENDED` or `API_KEY_INVALID`) are removed from the active pool for
the rest of the process lifetime. Error logs redact credential-shaped content;
eviction is identified only by a one-way key fingerprint.

### Custom Base URLs

Override default API endpoints using: `{PROVIDER}_BASE_URL`

Example:
```bash
OPENAI_BASE_URL=https://custom.openai.com/v1
OLLAMA_BASE_URL=http://localhost:11434/v1
```

### Advanced Configuration

You can combine multiple keys and base URLs in several ways:

1. One key with multiple base URLs:
```bash
OLLAMA_API_KEY=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1,http://other-server:11434/v1
```

2. Multiple keys with one base URL:
```bash
GEMINI_API_KEY=key1,key2
GEMINI_BASE_URL=https://api.gemini.com/v1
```

3. Paired keys and base URLs:
```bash
# Must have equal number of keys and URLs
# The library will randomly select matching pairs
GEMINI_API_KEY=key1,key2
GEMINI_BASE_URL=https://api.gemini.com/v1,https://api.gemini.com/v2
```

## Environment Setup Best Practices

When using SmolLLM in your project, you should handle environment variables at your application level:

1. Create a `.env` file:
```bash
# .env
OPENAI_API_KEY=sk-xxx
GEMINI_API_KEY=xxx,xxx2
ANTHROPIC_API_KEY=sk-xxx
```

2. Load the file at the application/process boundary:
```bash
uv run --env-file .env app.py
```

## Tips

- Keep sensitive API keys in `.env` (add to .gitignore)
- Create `.env.example` for documentation
- For production, consider using your platform's secret management system
- When using multiple keys, separate with commas (no spaces)
