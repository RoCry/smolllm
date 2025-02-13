# SmolLLM

A minimal Python library for interacting with various LLM providers, featuring automatic API key load balancing and streaming responses.

## Installation

```bash
pip install smolllm
uv add "smolllm @ ../smolllm"
```

## Quick Start

```python
from dotenv import load_dotenv
import asyncio
from smolllm import ask_llm

# Load environment variables at your application startup
load_dotenv()

async def main():
    response = await ask_llm(
        "Say hello world",
        model="gemini/gemini-2.0-flash"
    )
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

## Provider Configuration

Format: `provider/model_name` (e.g., `openai/gpt-4`, `gemini/gemini-2.0-flash`)

### API Keys

The library looks for API keys in environment variables following the pattern: `{PROVIDER}_API_KEY`

Example:
```bash
# .env
OPENAI_API_KEY=sk-xxx
GEMINI_API_KEY=key1,key2  # Multiple keys supported
```

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

2. Load environment variables before using SmolLLM:
```python
from dotenv import load_dotenv
import os

# Load at your application startup
load_dotenv()

# Now you can use SmolLLM
from smolllm import ask_llm
```

## Tips

- Keep sensitive API keys in `.env` (add to .gitignore)
- Create `.env.example` for documentation
- For production, consider using your platform's secret management system
- When using multiple keys, separate with commas (no spaces)
