import json
import os
from typing import List, Optional

import httpx

from .balancer import balancer
from .display import ResponseDisplay
from .log import logger
from .providers import parse_model_string
from .request import prepare_client_and_auth, prepare_request_data
from .stream import handle_chunk
from .types import StreamHandler


async def ask_llm(
    prompt: str,
    *,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    handler: Optional[StreamHandler] = None,
    timeout: float = 60.0,
    image_paths: Optional[List[str]] = None,
) -> str:
    """
    Args:
        model: provider/model_name (e.g., "openai/gpt-4" or "gemini"), fallback to SMOLLLM_MODEL
        api_key: Optional API key, fallback to ${PROVIDER}_API_KEY
        base_url: Custom base URL for API endpoint, fallback to ${PROVIDER}_BASE_URL
        stream_handler: Optional callback for handling streaming responses
        image_paths: Optional list of image paths to include with the prompt
    """
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
    client = prepare_client_and_auth(url, provider.name, api_key)

    api_key_preview = api_key[:5] + "..." + api_key[-4:]

    # Log information about the request
    log_message = (
        f"Sending {url} model={model_name} api_key={api_key_preview}, len={len(prompt)}"
    )
    if image_paths:
        image_sizes = [os.path.getsize(path) for path in image_paths]
        log_message += f", with {len(image_paths)} image(s) ({sum(image_sizes)} bytes)"
    logger.info(log_message)

    try:
        async with client.stream("POST", url, json=data, timeout=timeout) as response:
            if response.status_code >= 400:
                error_text = await response.aread()
                raise httpx.HTTPStatusError(
                    f"HTTP Error {response.status_code}: {error_text.decode()}",
                    request=response.request,
                    response=response,
                )
            return await process_stream_response(response, handler, provider.name)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


async def process_stream_response(
    response: httpx.Response,
    stream_handler: Optional[StreamHandler],
    provider_name: str,
) -> str:
    with ResponseDisplay(stream_handler) as display:
        async for line in response.aiter_lines():
            line = line.strip()
            if not line or line == "data: [DONE]" or not line.startswith("data: "):
                continue

            try:
                chunk = json.loads(line[6:])  # Remove "data: " prefix
                if delta := await handle_chunk(chunk, provider_name):
                    await display.update(delta)
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")

        return display.finalize()
