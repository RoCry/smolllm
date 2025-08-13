import os
from typing import Any, Dict, List, Literal, Optional, Tuple

import httpx

from .balancer import balancer
from .display import ResponseDisplay
from .log import logger
from .metrics import estimate_tokens, format_metrics
from .providers import Provider, parse_model_string
from .request import prepare_client_and_auth, prepare_request_data
from .stream import process_chunk_line
from .types import LLMResponse, PromptType, StreamHandler, StreamResponse
from .utils import strip_backticks


def _parse_models(model: Optional[str | List[str]]) -> List[str]:
    if model and isinstance(model, list):
        return model
    if not model:
        model = os.getenv("SMOLLLM_MODEL")
    if not model:
        raise ValueError("Model string not found. Set SMOLLLM_MODEL environment variable or pass model parameter")
    return [m.strip() for m in model.split(",")]


def _get_env_var(
    provider_name: str,
    var_type: Literal["API_KEY", "BASE_URL"],
    default: Optional[str] = None,
) -> str:
    """Get environment variable for a provider with fallback to default"""
    env_key = f"{provider_name.upper()}_{var_type}"
    value = os.getenv(env_key, default)
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
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
) -> Tuple[str, Dict[str, Any], httpx.AsyncClient, Provider, str]:
    """Common setup logic for LLM API calls

    Returns:
        tuple of (url, data, client, provider, model_name)
    """
    if not model:
        model = os.getenv("SMOLLLM_MODEL")
    if not model:
        raise ValueError("Model string not found. Set SMOLLLM_MODEL environment variable or pass model parameter")
    provider, model_name = parse_model_string(model)

    base_url = base_url or _get_env_var(provider.name, "BASE_URL", provider.base_url)
    api_key = api_key or _get_env_var(provider.name, "API_KEY")

    api_key, base_url = balancer.choose_pair(api_key, base_url)
    url, data = prepare_request_data(prompt, system_prompt, model_name, provider.name, base_url, image_paths)
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
    system_prompt: Optional[str] = None,
    model: Optional[str | List[str]] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    handler: Optional[StreamHandler] = None,
    timeout: float = 120.0,
    remove_backticks: bool = False,
    image_paths: Optional[List[str]] = None,
) -> LLMResponse:
    """
    Args:
        model: provider/model_name (e.g., "openai/gpt-4" or "gemini"), fallback to SMOLLLM_MODEL
              Can also specify multiple models as comma-separated list (e.g., "gemini/gemini-2.0-flash,gemini/gemini-2.5-pro")
        api_key: Optional API key, fallback to ${PROVIDER}_API_KEY
        base_url: Custom base URL for API endpoint, fallback to ${PROVIDER}_BASE_URL
        handler: Optional callback for handling streaming responses
        remove_backticks: Whether to remove backticks from the response, e.g. ```markdown\nblabla\n``` -> blabla
        image_paths: Optional list of image paths to include with the prompt

    Returns:
        LLMResponse object containing the text response, model used, and provider
    """
    last_error = None
    for m in _parse_models(model):
        try:
            url, data, client, provider, model_name = await _prepare_llm_call(
                prompt,
                system_prompt=system_prompt,
                model=m,
                api_key=api_key,
                base_url=base_url,
                image_paths=image_paths,
            )

            import time

            input_tokens = estimate_tokens(str(data))
            start_time = time.perf_counter()

            async with client.stream("POST", url, json=data, timeout=timeout) as response:
                await _handle_http_error(response)
                resp = await _process_stream_response(response, handler)
                if not resp or not resp.strip():
                    raise ValueError(f"Received empty response from model {m}")
                if remove_backticks:
                    resp = strip_backticks(resp)

                total_time = time.perf_counter() - start_time
                output_tokens = estimate_tokens(resp)

                logger.info(format_metrics(model_name, input_tokens, output_tokens, total_time))

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
    system_prompt: Optional[str] = None,
    model: Optional[str | List[str]] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 120.0,
    image_paths: Optional[List[str]] = None,
) -> StreamResponse:
    """Similar to ask_llm but yields chunks of text as they arrive.

    Args:
        model: provider/model_name (e.g., "openai/gpt-4" or "gemini"), fallback to SMOLLLM_MODEL
              Can also specify multiple models as comma-separated list (e.g., "gemini/gemini-2.0-flash,gemini/gemini-2.5-pro")
        api_key: Optional API key, fallback to ${PROVIDER}_API_KEY
        base_url: Custom base URL for API endpoint, fallback to ${PROVIDER}_BASE_URL
        image_paths: Optional list of image paths to include with the prompt

    Returns:
        StreamResponse object with stream iterator and model information
    """
    last_error = None
    for m in _parse_models(model):
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
                import time

                accumulated_response = []
                start_time = time.perf_counter()
                first_token_time = None

                async with client.stream("POST", url, json=data, timeout=timeout) as response:
                    await _handle_http_error(response)
                    async for line in response.aiter_lines():
                        if chunk_data := await process_chunk_line(line):
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            accumulated_response.append(chunk_data)
                            yield chunk_data

                # Log metrics after streaming completes
                if accumulated_response:
                    full_response = "".join(accumulated_response)
                    output_tokens = estimate_tokens(full_response)
                    total_time = time.perf_counter() - start_time
                    ttft_ms = int((first_token_time - start_time) * 1000) if first_token_time else 0

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
    stream_handler: Optional[StreamHandler],
) -> str:
    with ResponseDisplay(stream_handler) as display:
        async for line in response.aiter_lines():
            if delta := await process_chunk_line(line):
                await display.update(delta)
        return display.finalize()
