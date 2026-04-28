from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from time import perf_counter
from typing import Literal

import httpx

from .balancer import balancer
from .display import ResponseDisplay
from .log import logger
from .metrics import estimate_tokens, format_metrics
from .providers import Provider, parse_model_spec, parse_model_string
from .request import prepare_client_and_auth, prepare_embedding_request_data, prepare_request_data
from .stream import process_chunk_line
from .types import (
    EmbedResponse,
    LLMResponse,
    ModelInput,
    PromptType,
    StreamError,
    StreamHandler,
    StreamResponse,
)
from .utils import ThinkTagFilter, extract_think_tags, strip_backticks


class ModelSelector(ABC):
    """Base class for model selection strategies."""

    @abstractmethod
    def next_model(self) -> str | None:
        """Return next model to try, or None if exhausted."""


class SequentialSelector(ModelSelector):
    """Tries models in order. Used for str and list[str]."""

    def __init__(self, models: list[str]):
        self._models = iter(models)

    def next_model(self) -> str | None:
        return next(self._models, None)


class RandomSelector(ModelSelector):
    """Random selection with optional weights. Used for set and dict."""

    def __init__(self, models: set[str] | dict[str, float | int]):
        if isinstance(models, set):
            self._weights = {m: 1.0 for m in models}
        else:
            self._weights = dict(models)
        self._remaining = set(self._weights.keys())

    def next_model(self) -> str | None:
        if not self._remaining:
            return None
        models = list(self._remaining)
        weights = [self._weights[m] for m in models]
        chosen = random.choices(models, weights=weights, k=1)[0]
        self._remaining.remove(chosen)
        return chosen


def _create_selector(model: ModelInput | None) -> ModelSelector:
    """Create appropriate model selector based on input type."""
    candidate: ModelInput | None = model if model is not None else os.getenv("SMOLLLM_MODEL")
    if candidate is None:
        raise ValueError("Model string not found. Set SMOLLLM_MODEL environment variable or pass model parameter")

    # set -> random equal weights
    if isinstance(candidate, set):
        sanitized = {m.strip() for m in candidate if m.strip()}
        if not sanitized:
            raise ValueError("Model set must contain at least one non-empty entry")
        return RandomSelector(sanitized)

    # dict -> weighted random
    if isinstance(candidate, dict):
        sanitized = {k.strip(): v for k, v in candidate.items() if k.strip()}
        if not sanitized:
            raise ValueError("Model dict must contain at least one non-empty entry")
        if any(w <= 0 for w in sanitized.values()):
            raise ValueError("Model weights must be positive")
        return RandomSelector(sanitized)

    # str -> comma-separated sequential
    if isinstance(candidate, str):
        models = [m.strip() for m in candidate.split(",") if m.strip()]
        if not models:
            raise ValueError("Model string must contain at least one non-empty entry")
        return SequentialSelector(models)

    # Sequence -> sequential
    models = [item.strip() for item in candidate]
    if not models or any(not item for item in models):
        raise ValueError("Model sequence entries must be non-empty strings")
    return SequentialSelector(models)


def _get_env_var(
    provider_name: str,
    var_type: Literal["API_KEY", "BASE_URL"],
    default: str | None = None,
) -> str:
    """Get environment variable for a provider with fallback to default"""
    env_key = f"{provider_name.upper()}_{var_type}"
    value: str | None = os.getenv(env_key, default)
    if not value and var_type == "API_KEY" and provider_name == "ollama":
        return "ollama"
    if not value:
        raise ValueError(
            f"{var_type} not found. Set {env_key} environment variable or pass {var_type.lower()} parameter"
        )
    return value


# returns url, data for the request, client
async def _prepare_llm_call(
    prompt: PromptType,
    *,
    system_prompt: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    image_paths: Sequence[str] | None = None,
    stream: bool = True,
    reasoning_effort: str | None = None,
) -> tuple[str, dict[str, object], httpx.AsyncClient, Provider, str]:
    """Common setup logic for LLM API calls

    Returns:
        tuple of (url, data, client, provider, model_name)
    """
    if not model:
        model = os.getenv("SMOLLLM_MODEL")
    if not model:
        raise ValueError("Model string not found. Set SMOLLLM_MODEL environment variable or pass model parameter")
    provider, model_name = parse_model_string(model, base_url=base_url)

    base_url = base_url or _get_env_var(provider.name, "BASE_URL", provider.base_url)
    api_key = api_key or _get_env_var(provider.name, "API_KEY")

    api_key, base_url = balancer.choose_pair(api_key, base_url)
    image_list = list(image_paths) if image_paths else None
    url, data = prepare_request_data(
        prompt,
        system_prompt,
        model_name,
        provider.name,
        base_url,
        image_list,
        stream=stream,
        reasoning_effort=reasoning_effort,
    )
    client = prepare_client_and_auth(url, api_key)

    api_key_preview = api_key[:5] + "..." + api_key[-4:]
    logger.info(f"Sending {url} model={model_name} api_key={api_key_preview} ~tokens={estimate_tokens(str(data))}")

    return url, data, client, provider, model_name


async def _handle_http_error(response: httpx.Response) -> None:
    if response.status_code >= 400:
        error_text = await response.aread()
        raise httpx.HTTPStatusError(
            f"HTTP Error {response.status_code}: {error_text.decode()}",
            request=response.request,
            response=response,
        )


def _extract_text_from_response(payload: object) -> tuple[str, str]:
    """Extract text and reasoning from a non-streaming response.

    Returns (text, reasoning).
    """
    if not isinstance(payload, dict):
        raise TypeError("Response payload must be a mapping")

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Response payload missing choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise TypeError("Response choice must be a mapping")

    reasoning = ""
    content: str | None = None

    for container_key in ("message", "delta"):
        container = first.get(container_key)
        if not isinstance(container, dict):
            continue
        # "reasoning_content" (DeepSeek, vLLM, LiteLLM) or "reasoning" (Ollama)
        if not reasoning:
            for rk in ("reasoning_content", "reasoning"):
                rc = container.get(rk)
                if isinstance(rc, str) and rc:
                    reasoning = rc
                    break
        if content is None:
            c = container.get("content")
            if isinstance(c, str):
                content = c

    if content is None:
        text = first.get("text")
        if isinstance(text, str):
            content = text

    if content is None:
        raise ValueError("Response choice missing content")

    # Fall back to extracting <think> tags if backend didn't provide reasoning_content
    if not reasoning and content:
        extracted_reasoning, clean_content = extract_think_tags(content)
        if extracted_reasoning:
            reasoning = extracted_reasoning
            content = clean_content

    return content, reasoning


async def ask_llm(
    prompt: PromptType,
    *,
    system_prompt: str | None = None,
    model: ModelInput | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    handler: StreamHandler | None = None,
    timeout: float = 120.0,
    remove_backticks: bool = False,
    image_paths: Sequence[str] | None = None,
    stream: bool = True,
    reasoning_effort: str | None = None,
) -> LLMResponse:
    """
    Args:
        model: provider/model_name (e.g., "openai/gpt-4" or "gemini"), fallback to SMOLLLM_MODEL
              Can be: str, list[str] (fallback order), set[str] (random), dict[str, weight] (weighted random)
        api_key: Optional API key, fallback to ${PROVIDER}_API_KEY
        base_url: Custom base URL for API endpoint, fallback to ${PROVIDER}_BASE_URL
        handler: Optional callback for handling streaming responses
        remove_backticks: Whether to remove backticks from the response, e.g. ```markdown\nblabla\n``` -> blabla
        image_paths: Optional list of image paths to include with the prompt
        stream: Whether to request a streaming response
        reasoning_effort: Optional reasoning effort passed through to the provider (e.g. "none", "medium", "xhigh")

    Returns:
        LLMResponse object containing the text response, model used, and provider
    """
    selector = _create_selector(model)
    last_error: Exception | None = None
    while (m := selector.next_model()) is not None:
        try:
            model_spec, effort_override = parse_model_spec(m)
            effective_effort = effort_override if effort_override is not None else reasoning_effort
            url, data, client, provider, model_name = await _prepare_llm_call(
                prompt,
                system_prompt=system_prompt,
                model=model_spec,
                api_key=api_key,
                base_url=base_url,
                image_paths=image_paths,
                stream=stream,
                reasoning_effort=effective_effort,
            )

            input_tokens = estimate_tokens(str(data))
            start_time = perf_counter()
            ttft_ms: int | None = None

            reasoning = ""
            async with client:
                if stream:
                    async with client.stream("POST", url, json=data, timeout=timeout) as response:
                        await _handle_http_error(response)
                        resp, reasoning, ttft_ms = await _process_stream_response(response, handler, start_time)
                else:
                    response = await client.post(url, json=data, timeout=timeout)
                    await _handle_http_error(response)
                    await response.aread()
                    payload = response.json()
                    resp, reasoning = _extract_text_from_response(payload)
                    if handler is not None:
                        from .types import StreamChunk

                        await handler(StreamChunk(content=resp, reasoning=reasoning))

            if not resp and not reasoning:
                raise ValueError(f"Received empty response from model {m}")
            if remove_backticks:
                resp = strip_backticks(resp)

            total_time = perf_counter() - start_time
            output_tokens = estimate_tokens(resp + reasoning)

            logger.info(format_metrics(model_name, input_tokens, output_tokens, total_time, ttft_ms))

            return LLMResponse(
                text=resp, model=model_spec, model_name=model_name, provider=provider.name, reasoning=reasoning
            )
        except Exception as e:
            last_error = e
            logger.warning(f"Failed to get response from model {m}: {e}")
            continue
    if last_error:
        raise last_error
    raise ValueError("No valid models found")


async def stream_llm(
    prompt: PromptType,
    *,
    system_prompt: str | None = None,
    model: ModelInput | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 120.0,
    image_paths: Sequence[str] | None = None,
    reasoning_effort: str | None = None,
) -> StreamResponse:
    """Similar to ask_llm but yields chunks of text as they arrive.

    Args:
        model: provider/model_name (e.g., "openai/gpt-4" or "gemini"), fallback to SMOLLLM_MODEL
              Can be: str, list[str] (fallback order), set[str] (random), dict[str, weight] (weighted random)
        api_key: Optional API key, fallback to ${PROVIDER}_API_KEY
        base_url: Custom base URL for API endpoint, fallback to ${PROVIDER}_BASE_URL
        image_paths: Optional list of image paths to include with the prompt
        reasoning_effort: Optional reasoning effort passed through to the provider (e.g. "none", "medium", "xhigh")

    Returns:
        StreamResponse object with stream iterator and model information

    Note:
        If streaming fails mid-way, retries with fallback models. Already-yielded
        chunks cannot be retracted; callers should handle partial output.
    """
    selector = _create_selector(model)

    # Track which model succeeded for the response metadata
    successful_model: list[str | None] = [None]
    successful_model_name: list[str | None] = [None]
    successful_provider: list[str | None] = [None]

    async def _stream_with_fallback():
        nonlocal selector
        last_error: Exception | None = None

        while (m := selector.next_model()) is not None:
            accumulated_content: list[str] = []
            accumulated_reasoning: list[str] = []
            try:
                model_spec, effort_override = parse_model_spec(m)
                effective_effort = effort_override if effort_override is not None else reasoning_effort
                url, data, client, provider, model_name = await _prepare_llm_call(
                    prompt,
                    system_prompt=system_prompt,
                    model=model_spec,
                    api_key=api_key,
                    base_url=base_url,
                    image_paths=image_paths,
                    reasoning_effort=effective_effort,
                )

                input_tokens = estimate_tokens(str(data))
                start_time = perf_counter()
                first_token_time: float | None = None
                think_filter = ThinkTagFilter()

                try:
                    async with client:
                        async with client.stream("POST", url, json=data, timeout=timeout) as response:
                            await _handle_http_error(response)
                            async for line in response.aiter_lines():
                                if chunk := await process_chunk_line(line):
                                    chunk = think_filter.feed(chunk)
                                    if chunk:
                                        if first_token_time is None:
                                            first_token_time = perf_counter()
                                        if chunk.content:
                                            accumulated_content.append(chunk.content)
                                        if chunk.reasoning:
                                            accumulated_reasoning.append(chunk.reasoning)
                                        yield chunk
                            # Flush any buffered content
                            if final_chunk := think_filter.flush():
                                if final_chunk.content:
                                    accumulated_content.append(final_chunk.content)
                                if final_chunk.reasoning:
                                    accumulated_reasoning.append(final_chunk.reasoning)
                                yield final_chunk
                except Exception as e:
                    if accumulated_content or accumulated_reasoning:
                        raise StreamError(
                            "Stream interrupted",
                            partial="".join(accumulated_content),
                        ) from e
                    raise

                # Success - log metrics and record model info
                if accumulated_content or accumulated_reasoning:
                    full_response = "".join(accumulated_content) + "".join(accumulated_reasoning)
                    output_tokens = estimate_tokens(full_response)
                    total_time = perf_counter() - start_time
                    ttft_ms: int | None = None
                    if first_token_time is not None:
                        ttft_ms = max(0, int((first_token_time - start_time) * 1000))
                    logger.info(format_metrics(model_name, input_tokens, output_tokens, total_time, ttft_ms))
                else:
                    raise StreamError("Stream completed without content")

                successful_model[0] = model_spec
                successful_model_name[0] = model_name
                successful_provider[0] = provider.name
                return  # Stream completed successfully

            except Exception as e:
                if isinstance(e, StreamError) and e.partial:
                    last_error = e
                    logger.warning(f"Stream failed for model {m}, not retrying (partial output): {e}")
                    raise
                last_error = e
                logger.warning(f"Stream failed for model {m}, trying fallback: {e}")
                continue

        # All models exhausted
        if last_error:
            raise last_error
        raise ValueError("No valid models found")

    return StreamResponse(
        stream=_stream_with_fallback(),
        model=successful_model[0] or "unknown",
        model_name=successful_model_name[0] or "unknown",
        provider=successful_provider[0],
    )


async def _process_stream_response(
    response: httpx.Response,
    stream_handler: StreamHandler | None,
    start_time: float,
) -> tuple[str, str, int | None]:
    """Returns (text, reasoning, ttft_ms)."""
    first_token_time: float | None = None
    think_filter = ThinkTagFilter()
    with ResponseDisplay(stream_handler) as display:
        async for line in response.aiter_lines():
            if chunk := await process_chunk_line(line):
                chunk = think_filter.feed(chunk)
                if chunk:
                    if first_token_time is None:
                        first_token_time = perf_counter()
                    await display.update(chunk)
        # Flush any buffered content from the think tag filter
        if final_chunk := think_filter.flush():
            await display.update(final_chunk)
        text, reasoning = display.finalize()

    ttft_ms: int | None = None
    if first_token_time is not None:
        ttft_ms = max(0, int((first_token_time - start_time) * 1000))

    return text, reasoning, ttft_ms


def _parse_embedding_payload(payload: object) -> tuple[list[list[float]], int | None]:
    """Pull the vector list and prompt-token usage out of an OpenAI-shaped response."""
    if not isinstance(payload, dict):
        raise TypeError("Embedding response payload must be a mapping")
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise ValueError("Embedding response missing data")

    vectors: list[list[float]] = []
    for entry in data:
        if not isinstance(entry, dict):
            raise TypeError("Embedding entry must be a mapping")
        vec = entry.get("embedding")
        if not isinstance(vec, list) or not vec:
            raise ValueError("Embedding entry missing vector")
        vectors.append([float(x) for x in vec])

    usage = payload.get("usage")
    prompt_tokens: int | None = None
    if isinstance(usage, dict):
        pt = usage.get("prompt_tokens")
        if isinstance(pt, int):
            prompt_tokens = pt

    return vectors, prompt_tokens


async def embed_llm(
    inputs: str | Sequence[str],
    *,
    model: ModelInput | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    dimensions: int | None = None,
    timeout: float = 120.0,
) -> EmbedResponse:
    """Generate embedding vectors via an OpenAI-compatible /embeddings endpoint.

    Args:
        inputs: A single string or list of strings to embed.
        model: provider/model_name (e.g. "ollama/qwen3-embedding:4b"); same
            selector semantics as ``ask_llm`` (str / list / set / dict).
            Falls back to ``SMOLLLM_MODEL``.
        dimensions: Optional output dimensionality. Forwarded to providers
            that support truncation (OpenAI text-embedding-3+, recent Ollama
            with Matryoshka models like qwen3-embedding). Ignored otherwise.

    Returns:
        ``EmbedResponse`` whose ``embeddings`` is always a list of vectors —
        index ``[0]`` for the single-input case.
    """
    selector = _create_selector(model)
    last_error: Exception | None = None
    while (m := selector.next_model()) is not None:
        try:
            model_spec, _ = parse_model_spec(m)
            provider, model_name = parse_model_string(model_spec, base_url=base_url)
            resolved_base = base_url or _get_env_var(provider.name, "BASE_URL", provider.base_url)
            resolved_key = api_key or _get_env_var(provider.name, "API_KEY")
            chosen_key, chosen_url = balancer.choose_pair(resolved_key, resolved_base)
            url, data = prepare_embedding_request_data(
                inputs,
                model_name=model_name,
                provider_name=provider.name,
                base_url=chosen_url,
                dimensions=dimensions,
            )
            client = prepare_client_and_auth(url, chosen_key)

            key_preview = chosen_key[:5] + "..." + chosen_key[-4:]
            logger.info(f"Embedding {url} model={model_name} api_key={key_preview} dimensions={dimensions}")

            start_time = perf_counter()
            async with client:
                response = await client.post(url, json=data, timeout=timeout)
                await _handle_http_error(response)
                await response.aread()
                payload = response.json()

            vectors, prompt_tokens = _parse_embedding_payload(payload)
            elapsed = perf_counter() - start_time
            actual_dim = len(vectors[0])
            logger.info(
                f"Embedded model={model_name} count={len(vectors)} dim={actual_dim} "
                f"prompt_tokens={prompt_tokens} elapsed={elapsed:.2f}s"
            )

            return EmbedResponse(
                embeddings=vectors,
                model=model_spec,
                model_name=model_name,
                dimensions=actual_dim,
                provider=provider.name,
                prompt_tokens=prompt_tokens,
            )
        except Exception as e:
            last_error = e
            logger.warning(f"Failed to embed with model {m}: {e}")
            continue

    if last_error:
        raise last_error
    raise ValueError("No valid models found")
