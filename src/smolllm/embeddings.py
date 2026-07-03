from __future__ import annotations

import time
from collections.abc import Sequence
from time import perf_counter

from .balancer import balancer
from .env import get_env_var
from .http_stream import handle_http_error
from .log import logger
from .metrics import estimate_tokens
from .model_selector import create_selector
from .providers import parse_model_spec, parse_model_string
from .request import prepare_client_and_auth, prepare_embedding_request_data
from .types import EmbedResponse, Hook, ModelInput, RequestEvent, Usage
from .utils import preview_api_key


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
    hook: Hook | None = None,
) -> EmbedResponse:
    """Generate embedding vectors via an OpenAI-compatible /embeddings endpoint."""
    selector = create_selector(model)
    last_error: Exception | None = None
    while (m := selector.next_model()) is not None:
        attempt_provider = ""
        attempt_model_spec = m
        attempt_model_name = ""
        attempt_api_key = ""
        input_tokens = 0
        start_time = perf_counter()
        try:
            model_spec, _ = parse_model_spec(m)
            attempt_model_spec = model_spec
            provider, model_name = parse_model_string(model_spec, base_url=base_url)
            attempt_provider = provider.name
            attempt_model_name = model_name
            resolved_base = base_url or get_env_var(provider.name, "BASE_URL", provider.base_url)
            resolved_key = api_key or get_env_var(provider.name, "API_KEY")
            chosen_key, chosen_url = balancer.choose_pair(resolved_key, resolved_base)
            attempt_api_key = chosen_key
            url, data = prepare_embedding_request_data(
                inputs,
                model_name=model_name,
                provider_name=provider.name,
                base_url=chosen_url,
                dimensions=dimensions,
            )
            client = prepare_client_and_auth(url, chosen_key)

            logger.info(
                f"Embedding {url} model={model_name} api_key={preview_api_key(chosen_key)} dimensions={dimensions}"
            )

            input_tokens = estimate_tokens(str(data))
            start_time = perf_counter()
            async with client:
                response = await client.post(url, json=data, timeout=timeout)
                await handle_http_error(response)
                await response.aread()
                payload = response.json()

            vectors, prompt_tokens = _parse_embedding_payload(payload)
            elapsed = perf_counter() - start_time
            actual_dim = len(vectors[0])
            logger.info(
                f"Embedded model={model_name} count={len(vectors)} dim={actual_dim} "
                f"prompt_tokens={prompt_tokens} elapsed={elapsed:.2f}s"
            )

            usage = Usage(
                provider=provider.name,
                model=model_spec,
                model_name=model_name,
                api_key_hint=preview_api_key(chosen_key),
                input_tokens=prompt_tokens if prompt_tokens is not None else input_tokens,
                output_tokens=0,
                duration_ms=int(elapsed * 1000),
                ttft_ms=None,
                estimated=prompt_tokens is None,
            )
            if hook is not None:
                hook(RequestEvent(usage=usage, error=None, timestamp=time.time()))

            return EmbedResponse(
                embeddings=vectors,
                model=model_spec,
                model_name=model_name,
                dimensions=actual_dim,
                provider=provider.name,
                prompt_tokens=prompt_tokens,
                usage=usage,
            )
        except Exception as e:
            last_error = e
            logger.warning(f"Failed to embed with model {m}: {e}")
            if hook is not None:
                duration_ms = int((perf_counter() - start_time) * 1000)
                fail_usage = Usage(
                    provider=attempt_provider,
                    model=attempt_model_spec,
                    model_name=attempt_model_name,
                    api_key_hint=preview_api_key(attempt_api_key) if attempt_api_key else "",
                    input_tokens=input_tokens,
                    output_tokens=0,
                    duration_ms=duration_ms,
                    ttft_ms=None,
                )
                hook(RequestEvent(usage=fail_usage, error=e, timestamp=time.time()))
            continue

    if last_error:
        raise last_error
    raise ValueError("No valid models found")
