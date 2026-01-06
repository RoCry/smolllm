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
from .providers import Provider, parse_model_string
from .request import prepare_client_and_auth, prepare_request_data
from .stream import process_chunk_line
from .types import LLMResponse, ModelInput, PromptType, StreamHandler, StreamResponse
from .utils import strip_backticks


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
    url, data = prepare_request_data(prompt, system_prompt, model_name, provider.name, base_url, image_list)
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

    Returns:
        LLMResponse object containing the text response, model used, and provider
    """
    selector = _create_selector(model)
    last_error: Exception | None = None
    while (m := selector.next_model()) is not None:
        try:
            url, data, client, provider, model_name = await _prepare_llm_call(
                prompt,
                system_prompt=system_prompt,
                model=m,
                api_key=api_key,
                base_url=base_url,
                image_paths=image_paths,
            )

            input_tokens = estimate_tokens(str(data))
            start_time = perf_counter()

            async with client.stream("POST", url, json=data, timeout=timeout) as response:
                await _handle_http_error(response)
                resp, ttft_ms = await _process_stream_response(response, handler, start_time)
                if not resp:
                    raise ValueError(f"Received empty response from model {m}")
                if remove_backticks:
                    resp = strip_backticks(resp)

                total_time = perf_counter() - start_time
                output_tokens = estimate_tokens(resp)

                logger.info(format_metrics(model_name, input_tokens, output_tokens, total_time, ttft_ms))

                return LLMResponse(text=resp, model=m, model_name=model_name, provider=provider.name)
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
) -> StreamResponse:
    """Similar to ask_llm but yields chunks of text as they arrive.

    Args:
        model: provider/model_name (e.g., "openai/gpt-4" or "gemini"), fallback to SMOLLLM_MODEL
              Can be: str, list[str] (fallback order), set[str] (random), dict[str, weight] (weighted random)
        api_key: Optional API key, fallback to ${PROVIDER}_API_KEY
        base_url: Custom base URL for API endpoint, fallback to ${PROVIDER}_BASE_URL
        image_paths: Optional list of image paths to include with the prompt

    Returns:
        StreamResponse object with stream iterator and model information
    """
    selector = _create_selector(model)
    last_error: Exception | None = None
    while (m := selector.next_model()) is not None:
        try:
            url, data, client, provider, model_name = await _prepare_llm_call(
                prompt,
                system_prompt=system_prompt,
                model=m,
                api_key=api_key,
                base_url=base_url,
                image_paths=image_paths,
            )

            input_tokens = estimate_tokens(str(data))

            async def _stream():
                accumulated_response: list[str] = []
                start_time = perf_counter()
                first_token_time: float | None = None

                async with client.stream("POST", url, json=data, timeout=timeout) as response:
                    await _handle_http_error(response)
                    async for line in response.aiter_lines():
                        if chunk_data := await process_chunk_line(line):
                            if first_token_time is None:
                                first_token_time = perf_counter()
                            accumulated_response.append(chunk_data)
                            yield chunk_data

                # Log metrics after streaming completes
                if accumulated_response:
                    full_response = "".join(accumulated_response)
                    output_tokens = estimate_tokens(full_response)
                    total_time = perf_counter() - start_time
                    ttft_ms: int | None = None
                    if first_token_time is not None:
                        ttft_ms = max(0, int((first_token_time - start_time) * 1000))

                    logger.info(format_metrics(model_name, input_tokens, output_tokens, total_time, ttft_ms))

            return StreamResponse(stream=_stream(), model=m, model_name=model_name, provider=provider.name)
        except Exception as e:
            last_error = e
            logger.warning(f"Failed to stream from model {m}: {e}")
            continue
    if last_error:
        raise last_error
    raise ValueError("No valid models found")


async def _process_stream_response(
    response: httpx.Response,
    stream_handler: StreamHandler | None,
    start_time: float,
) -> tuple[str, int | None]:
    first_token_time: float | None = None
    with ResponseDisplay(stream_handler) as display:
        async for line in response.aiter_lines():
            if delta := await process_chunk_line(line):
                if first_token_time is None:
                    first_token_time = perf_counter()
                await display.update(delta)
        final_response = display.finalize()

    ttft_ms: int | None = None
    if first_token_time is not None:
        ttft_ms = max(0, int((first_token_time - start_time) * 1000))

    return final_response, ttft_ms
