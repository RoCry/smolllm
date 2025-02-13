import os
import json
from typing import Optional
import httpx

from .types import StreamHandler
from .providers import parse_model_string
from .balancer import balancer
from .stream import handle_chunk
from .request import prepare_request_data, prepare_client_and_auth
from .log import logger
from .display import ResponseDisplay


async def ask_llm(
    prompt: str,
    *,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    handler: Optional[StreamHandler] = None,
    timeout: float = 60.0,
) -> str:
    """
    Send a prompt to an LLM and get the response

    Args:
        prompt: The main prompt text
        system_prompt: Optional system prompt
        model: Model identifier (e.g., "openai/gpt-4" or "google")
        api_key: Optional API key (will fall back to environment variable)
        base_url: Custom base URL for API endpoint
        stream_handler: Optional callback for handling streaming responses
        timeout: Request timeout in seconds
    """
    if not model:
        model = os.getenv("SMOLLLM_MODEL")
    if not model:
        raise ValueError(
            "Model string not found. Set SMOLLLM_MODEL environment variable or pass model parameter"
        )
    provider, model_name = parse_model_string(model)

    if not base_url:
        env_base_url = os.getenv(f"{provider.name.upper()}_BASE_URL")
        base_url = env_base_url or provider.base_url

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
    url, data, headers = prepare_request_data(
        prompt, system_prompt, model_name, provider.name, base_url
    )
    client = prepare_client_and_auth(url, provider.name, api_key, headers)

    api_key_preview = api_key[:5] + "..." + api_key[-4:]
    logger.info(f"Sending {url} model={model_name} api_key={api_key_preview}")

    try:
        async with client.stream(
            "POST", url, headers=headers, json=data, timeout=timeout
        ) as response:
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
                    display.update(delta)
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")

        return display.finalize()
