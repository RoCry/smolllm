import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx

from .balancer import balancer
from .display import ResponseDisplay
from .log import logger
from .providers import parse_model_string
from .request import prepare_client_and_auth, prepare_request_data
from .stream import handle_chunk
from .types import PromptType, StreamHandler
from .utils import strip_backticks


async def _prepare_llm_call(
    prompt: PromptType,
    *,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
) -> Tuple[str, Dict[str, Any], httpx.AsyncClient, str]:
    """Common setup logic for LLM API calls"""
    if not model:
        model = os.getenv("SMOLLLM_MODEL")
    if not model:
        raise ValueError(
            "Model string not found. Set SMOLLLM_MODEL environment variable or pass model parameter"
        )
    provider, model_name = parse_model_string(model)

    if not base_url:
        env_key = f"{provider.name.upper()}_BASE_URL"
        base_url = os.getenv(env_key) or provider.base_url
        if not base_url:
            raise ValueError(
                f"Base URL not found. Set {env_key} environment variable or pass base_url parameter"
            )

    if not api_key:
        env_key = f"{provider.name.upper()}_API_KEY"
        api_key = os.getenv(env_key)
        if not api_key:
            # special case for convenience: ollama
            if provider.name == "ollama":
                api_key = "ollama"
            else:
                raise ValueError(
                    f"API key not found. Set {env_key} environment variable or pass api_key parameter"
                )

    if not base_url:
        raise ValueError("Base URL is required")

    api_key, base_url = balancer.choose_pair(api_key, base_url)
    url, data = prepare_request_data(
        prompt, system_prompt, model_name, provider.name, base_url, image_paths
    )
    client = prepare_client_and_auth(url, api_key)

    api_key_preview = api_key[:5] + "..." + api_key[-4:]

    # Log information about the request
    log_message = (
        f"Sending {url} model={model_name} api_key={api_key_preview}, len={len(prompt)}"
    )
    if image_paths:
        image_sizes = [os.path.getsize(path) for path in image_paths]
        log_message += f", with {len(image_paths)} image(s) ({sum(image_sizes)} bytes)"
    logger.info(log_message)

    return url, data, client, provider.name


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
        api_key: Optional API key, fallback to ${PROVIDER}_API_KEY
        base_url: Custom base URL for API endpoint, fallback to ${PROVIDER}_BASE_URL
        handler: Optional callback for handling streaming responses
        remove_backticks: Whether to remove backticks from the response, e.g. ```markdown\nblabla\n``` -> blabla
        image_paths: Optional list of image paths to include with the prompt
    """
    try:
        url, data, client, provider_name = await _prepare_llm_call(
            prompt,
            system_prompt=system_prompt,
            model=model,
            api_key=api_key,
            base_url=base_url,
            image_paths=image_paths,
        )

        async with client.stream("POST", url, json=data, timeout=timeout) as response:
            if response.status_code >= 400:
                error_text = await response.aread()
                raise httpx.HTTPStatusError(
                    f"HTTP Error {response.status_code}: {error_text.decode()}",
                    request=response.request,
                    response=response,
                )
            resp = await process_stream_response(response, handler)
            if remove_backticks:
                resp = strip_backticks(resp)
            return resp
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


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
        api_key: Optional API key, fallback to ${PROVIDER}_API_KEY
        base_url: Custom base URL for API endpoint, fallback to ${PROVIDER}_BASE_URL
        image_paths: Optional list of image paths to include with the prompt
    """
    try:
        url, data, client, provider_name = await _prepare_llm_call(
            prompt,
            system_prompt=system_prompt,
            model=model,
            api_key=api_key,
            base_url=base_url,
            image_paths=image_paths,
        )

        async with client.stream("POST", url, json=data, timeout=timeout) as response:
            if response.status_code >= 400:
                error_text = await response.aread()
                raise httpx.HTTPStatusError(
                    f"HTTP Error {response.status_code}: {error_text.decode()}",
                    request=response.request,
                    response=response,
                )

            async for line in response.aiter_lines():
                line = line.strip()
                if not line or line == "data: [DONE]" or not line.startswith("data: "):
                    continue

                try:
                    chunk = json.loads(line[6:])  # Remove "data: " prefix
                    if delta := handle_chunk(chunk):
                        yield delta
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


async def process_stream_response(
    response: httpx.Response,
    stream_handler: Optional[StreamHandler],
) -> str:
    with ResponseDisplay(stream_handler) as display:
        async for line in response.aiter_lines():
            line = line.strip()
            if not line or line == "data: [DONE]" or not line.startswith("data: "):
                continue

            try:
                chunk = json.loads(line[6:])  # Remove "data: " prefix
                if delta := handle_chunk(chunk):
                    await display.update(delta)
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")

        return display.finalize()
