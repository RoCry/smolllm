import os
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Tuple

import httpx

from .balancer import balancer
from .display import ResponseDisplay
from .log import logger
from .providers import parse_model_string
from .request import prepare_client_and_auth, prepare_request_data
from .stream import process_chunk_line
from .types import PromptType, StreamHandler
from .utils import strip_backticks


def _parse_models(model_str: Optional[str]) -> List[str]:
    if not model_str:
        model_str = os.getenv("SMOLLLM_MODEL")
    if not model_str:
        raise ValueError("Model string not found. Set SMOLLLM_MODEL environment variable or pass model parameter")
    return [m.strip() for m in model_str.split(",")]


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
) -> Tuple[str, Dict[str, Any], httpx.AsyncClient]:
    """Common setup logic for LLM API calls"""
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
    logger.info(f"Sending {url} model={model_name} api_key={api_key_preview} len(data)={len(str(data))}")

    return url, data, client


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
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    handler: Optional[StreamHandler] = None,
    timeout: float = 120.0,
    remove_backticks: bool = False,
    image_paths: Optional[List[str]] = None,
) -> str:
    """
    Args:
        model: provider/model_name (e.g., "openai/gpt-4" or "gemini"), fallback to SMOLLLM_MODEL
              Can also specify multiple models as comma-separated list (e.g., "gemini/gemini-2.0-flash,gemini/gemini-2.5-pro")
        api_key: Optional API key, fallback to ${PROVIDER}_API_KEY
        base_url: Custom base URL for API endpoint, fallback to ${PROVIDER}_BASE_URL
        handler: Optional callback for handling streaming responses
        remove_backticks: Whether to remove backticks from the response, e.g. ```markdown\nblabla\n``` -> blabla
        image_paths: Optional list of image paths to include with the prompt
    """
    last_error = None
    for m in _parse_models(model):
        try:
            url, data, client = await _prepare_llm_call(
                prompt,
                system_prompt=system_prompt,
                model=m,
                api_key=api_key,
                base_url=base_url,
                image_paths=image_paths,
            )

            async with client.stream("POST", url, json=data, timeout=timeout) as response:
                await _handle_http_error(response)
                resp = await _process_stream_response(response, handler)
                if remove_backticks:
                    resp = strip_backticks(resp)
                return resp
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
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 120.0,
    image_paths: Optional[List[str]] = None,
) -> AsyncGenerator[str, None]:
    """Similar to ask_llm but yields chunks of text as they arrive.

    Args:
        model: provider/model_name (e.g., "openai/gpt-4" or "gemini"), fallback to SMOLLLM_MODEL
              Can also specify multiple models as comma-separated list (e.g., "gemini/gemini-2.0-flash,gemini/gemini-2.5-pro")
        api_key: Optional API key, fallback to ${PROVIDER}_API_KEY
        base_url: Custom base URL for API endpoint, fallback to ${PROVIDER}_BASE_URL
        image_paths: Optional list of image paths to include with the prompt
    """
    last_error = None
    for m in _parse_models(model):
        try:
            url, data, client = await _prepare_llm_call(
                prompt,
                system_prompt=system_prompt,
                model=m,
                api_key=api_key,
                base_url=base_url,
                image_paths=image_paths,
            )

            async with client.stream("POST", url, json=data, timeout=timeout) as response:
                await _handle_http_error(response)

                async for line in response.aiter_lines():
                    if data := await process_chunk_line(line):
                        yield data
                return  # Successfully completed streaming
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
