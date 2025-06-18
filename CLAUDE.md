# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SmolLLM is a minimal Python library for interacting with 45+ LLM providers, featuring automatic API key load balancing and streaming responses. The library provides a unified interface (`ask_llm` and `stream_llm`) for all providers.

## Development Commands

### Testing
```bash
make test                    # Run all tests using pytest
uv run pytest -s -v tests/*  # Run tests with verbose output
```

### Building and Release
```bash
make build                   # Build the package
make test-release           # Test release to Test PyPI
make bump-patch             # Bump patch version (0.3.0 -> 0.3.1)
make bump-minor             # Bump minor version (0.3.0 -> 0.4.0)
make bump-major             # Bump major version (0.3.0 -> 1.0.0)
```

### Development Setup
```bash
make install-dev            # Install all development dependencies
make clean                  # Clean build artifacts
make update-providers       # Update provider configurations
```

### Linting
```bash
# The project uses Ruff with 120-char line length
# Configuration is in pyproject.toml
ruff check src/             # Check linting
ruff format src/            # Format code
```

## Architecture

### Core Modules
- `src/smolllm/core.py` - Main API functions (`ask_llm`, `stream_llm`)
- `src/smolllm/providers.py` - Provider management and configuration loading
- `src/smolllm/balancer.py` - API key load balancing across multiple keys/endpoints
- `src/smolllm/stream.py` - Streaming response handling with async generators
- `src/smolllm/types.py` - Type definitions (Provider, LLMConfig, Message types)
- `src/smolllm/provider_config.py` - Provider endpoint configuration management
- `src/smolllm/request.py` - HTTP request handling and retry logic

### Key Design Patterns
1. **Provider Format**: All models use `provider/model_name` format (e.g., `openai/gpt-4`)
2. **Environment Variables**: API keys follow `{PROVIDER}_API_KEY` pattern, base URLs follow `{PROVIDER}_BASE_URL`
3. **Load Balancing**: Supports multiple API keys per provider with comma separation
4. **Configuration**: Provider configurations stored in `providers.json` with 45+ providers

### Testing Approach
- Tests located in `/tests/` directory
- Use pytest for test execution
- Example tests in `test_strip_backticks.py` show unit testing patterns

### Version Management
- Version is stored in `src/smolllm/__init__.py`
- Use Makefile commands for version bumping which automatically:
  - Updates version
  - Runs tests
  - Creates git commit and tag
  - Pushes to remote

### Provider Configuration
- Provider list maintained in `providers.json`
- Use `tools/update_providers.py` to update provider configurations
- Each provider includes API URL and documentation link