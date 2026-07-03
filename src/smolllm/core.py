from __future__ import annotations

import os
import time
from collections.abc import Sequence
from time import perf_counter

import httpx

from .balancer import balancer
from .embeddings import embed_llm as embed_llm
from .env import get_env_var
from .http_stream import handle_http_error, iter_stream_lines, process_stream_response, usage_tokens_from_payload
from .log import logger
from .metrics import estimate_tokens, format_metrics
from .model_selector import RandomSelector as RandomSelector
from .model_selector import SequentialSelector as SequentialSelector
from .model_selector import create_selector
from .providers import Provider, parse_model_spec, parse_model_string
from .request import prepare_client_and_auth, prepare_request_data
from .response import extract_text_from_response as _extract_text_from_response
from .stream import decode_sse_chunk, extract_delta, extract_finish_reason, extract_model, update_usage
from .types import (
    Hook,
    LLMResponse,
    ModelInput,
    PromptType,
    RequestEvent,
    StreamError,
    StreamHandler,
    StreamResponse,
    Usage,
)
from .utils import ThinkTagFilter, preview_api_key, strip_backticks

_create_selector = create_selector


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
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    stop: str | Sequence[str] | None = None,
    seed: int | None = None,
    include_stream_usage: bool = True,
) -> tuple[str, dict[str, object], httpx.AsyncClient, Provider, str, str]:
    """Common setup logic for LLM API calls."""
    if not model:
        model = os.getenv("SMOLLLM_MODEL")
    if not model:
        raise ValueError("Model string not found. Set SMOLLLM_MODEL environment variable or pass model parameter")
    provider, model_name = parse_model_string(model, base_url=base_url)

    base_url = base_url or get_env_var(provider.name, "BASE_URL", provider.base_url)
    api_key = api_key or get_env_var(provider.name, "API_KEY")

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
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop,
        seed=seed,
        include_stream_usage=include_stream_usage,
    )
    client = prepare_client_and_auth(url, api_key)

    effort_log = f" reasoning_effort={reasoning_effort}" if reasoning_effort is not None else ""
    logger.info(
        f"Sending {url} model={model_name}{effort_log} api_key={preview_api_key(api_key)} "
        f"~tokens={estimate_tokens(str(data))}"
    )

    return url, data, client, provider, model_name, api_key


def _resolve_usage_tokens(
    reported: tuple[int | None, int | None] | None,
    *,
    estimated_input_tokens: int,
    response_text: str,
) -> tuple[int, int, bool]:
    output_tokens = estimate_tokens(response_text)
    if reported is None:
        return estimated_input_tokens, output_tokens, True

    prompt_tokens, completion_tokens = reported
    return (
        prompt_tokens if prompt_tokens is not None else estimated_input_tokens,
        completion_tokens if completion_tokens is not None else output_tokens,
        prompt_tokens is None or completion_tokens is None,
    )


def _is_truncated(finish_reason: str | None, *, has_content: bool, stream: bool) -> bool:
    """Whether a response was cut short rather than completing naturally.

    - ``finish_reason == "length"``: the provider hit the output-token cap.
    - streaming with content but no terminal ``finish_reason``: the SSE stream
      ended without its mandatory final frame, i.e. the connection dropped or the
      upstream terminated the stream early.

    A non-streaming body arrives as one complete JSON object, so a missing
    ``finish_reason`` there is not evidence of truncation. Empty responses are
    handled separately by the empty-response check.
    """
    if not has_content:
        return False
    if finish_reason == "length":
        return True
    return stream and finish_reason is None


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
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    stop: str | Sequence[str] | None = None,
    seed: int | None = None,
    hook: Hook | None = None,
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
        max_tokens: Optional maximum number of output tokens
        stop: Optional stop sequence or stop sequence list
        seed: Optional deterministic sampling seed for providers that support it

    Returns:
        LLMResponse object containing the text response, model used, and provider
    """
    selector = _create_selector(model)
    last_error: Exception | None = None
    while (m := selector.next_model()) is not None:
        # Pre-attempt placeholders for usage tracking on early failures
        attempt_provider = ""
        attempt_model_spec = m
        attempt_model_name = ""
        attempt_api_key = ""
        input_tokens = 0
        start_time = perf_counter()
        ttft_ms: int | None = None
        try:
            model_spec, effort_override = parse_model_spec(m)
            attempt_model_spec = model_spec
            effective_effort = effort_override if effort_override is not None else reasoning_effort
            url, data, client, provider, model_name, used_api_key = await _prepare_llm_call(
                prompt,
                system_prompt=system_prompt,
                model=model_spec,
                api_key=api_key,
                base_url=base_url,
                image_paths=image_paths,
                stream=stream,
                reasoning_effort=effective_effort,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop,
                seed=seed,
            )
            attempt_provider = provider.name
            attempt_model_name = model_name
            attempt_api_key = used_api_key

            input_tokens = estimate_tokens(str(data))
            start_time = perf_counter()
            ttft_ms = None

            reasoning = ""
            resolved_model: str | None = None
            finish_reason: str | None = None
            provider_usage: tuple[int | None, int | None] | None = None
            async with client:
                if stream:
                    stream_usage: dict[str, int] = {}
                    resp, reasoning, ttft_ms, resolved_model, finish_reason = await process_stream_response(
                        iter_stream_lines(client, url, data, timeout),
                        handler,
                        start_time,
                        usage=stream_usage,
                    )
                    prompt_tokens = stream_usage.get("prompt_tokens")
                    completion_tokens = stream_usage.get("completion_tokens")
                    if prompt_tokens is not None or completion_tokens is not None:
                        provider_usage = (prompt_tokens, completion_tokens)
                else:
                    response = await client.post(url, json=data, timeout=timeout)
                    await handle_http_error(response)
                    await response.aread()
                    payload = response.json()
                    resp, reasoning = _extract_text_from_response(payload)
                    resolved_model = extract_model(payload)
                    finish_reason = extract_finish_reason(payload) if isinstance(payload, dict) else None
                    provider_usage = usage_tokens_from_payload(payload)
                    if handler is not None:
                        from .types import StreamChunk

                        await handler(StreamChunk(content=resp, reasoning=reasoning))

            if not resp and not reasoning:
                raise ValueError(f"Received empty response from model {m}")
            if _is_truncated(finish_reason, has_content=bool(resp or reasoning), stream=stream):
                raise StreamError(f"Truncated response from model {m} (finish_reason={finish_reason})")
            if remove_backticks:
                resp = strip_backticks(resp)

            total_time = perf_counter() - start_time
            input_tokens, output_tokens, estimated = _resolve_usage_tokens(
                provider_usage,
                estimated_input_tokens=input_tokens,
                response_text=resp + reasoning,
            )

            logger.info(format_metrics(model_name, input_tokens, output_tokens, total_time, ttft_ms))

            usage = Usage(
                provider=provider.name,
                model=model_spec,
                model_name=model_name,
                api_key_hint=preview_api_key(used_api_key),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=int(total_time * 1000),
                ttft_ms=ttft_ms,
                estimated=estimated,
            )
            if hook is not None:
                hook(RequestEvent(usage=usage, error=None, timestamp=time.time()))

            return LLMResponse(
                text=resp,
                model=model_spec,
                model_name=model_name,
                provider=provider.name,
                resolved_model=resolved_model,
                reasoning=reasoning,
                usage=usage,
                finish_reason=finish_reason,
            )
        except Exception as e:
            last_error = e
            logger.warning(f"Failed to get response from model {m}: {e}")
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
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    stop: str | Sequence[str] | None = None,
    seed: int | None = None,
    hook: Hook | None = None,
) -> StreamResponse:
    """Similar to ask_llm but yields chunks of text as they arrive.

    Args:
        model: provider/model_name (e.g., "openai/gpt-4" or "gemini"), fallback to SMOLLLM_MODEL
              Can be: str, list[str] (fallback order), set[str] (random), dict[str, weight] (weighted random)
        api_key: Optional API key, fallback to ${PROVIDER}_API_KEY
        base_url: Custom base URL for API endpoint, fallback to ${PROVIDER}_BASE_URL
        image_paths: Optional list of image paths to include with the prompt
        reasoning_effort: Optional reasoning effort passed through to the provider (e.g. "none", "medium", "xhigh")
        max_tokens: Optional maximum number of output tokens
        stop: Optional stop sequence or stop sequence list
        seed: Optional deterministic sampling seed for providers that support it

    Returns:
        StreamResponse object with stream iterator and model information

    Note:
        If streaming fails mid-way, retries with fallback models. Already-yielded
        chunks cannot be retracted; callers should handle partial output.
    """
    selector = _create_selector(model)

    # Back-patched with the winning model's metadata once a stream succeeds; the
    # value is unknown until then because the generator below runs lazily.
    response_ref: list[StreamResponse | None] = [None]

    async def _stream_with_fallback():
        nonlocal selector
        last_error: Exception | None = None

        while (m := selector.next_model()) is not None:
            accumulated_content: list[str] = []
            accumulated_reasoning: list[str] = []
            resolved_model: str | None = None
            finish_reason: str | None = None
            attempt_provider = ""
            attempt_model_spec = m
            attempt_model_name = ""
            attempt_api_key = ""
            input_tokens = 0
            start_time = perf_counter()
            try:
                model_spec, effort_override = parse_model_spec(m)
                attempt_model_spec = model_spec
                effective_effort = effort_override if effort_override is not None else reasoning_effort
                url, data, client, provider, model_name, used_api_key = await _prepare_llm_call(
                    prompt,
                    system_prompt=system_prompt,
                    model=model_spec,
                    api_key=api_key,
                    base_url=base_url,
                    image_paths=image_paths,
                    reasoning_effort=effective_effort,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop=stop,
                    seed=seed,
                )
                attempt_provider = provider.name
                attempt_model_name = model_name
                attempt_api_key = used_api_key

                input_tokens = estimate_tokens(str(data))
                start_time = perf_counter()
                first_token_time: float | None = None
                think_filter = ThinkTagFilter()
                stream_usage: dict[str, int] = {}

                try:
                    async with client:
                        async for line in iter_stream_lines(client, url, data, timeout):
                            raw = decode_sse_chunk(line)
                            if raw is None:
                                continue
                            update_usage(raw, stream_usage)
                            if resolved_model is None:
                                resolved_model = extract_model(raw)
                            if (reason := extract_finish_reason(raw)) is not None:
                                finish_reason = reason
                            if chunk := extract_delta(raw):
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
                    prompt_tokens = stream_usage.get("prompt_tokens")
                    completion_tokens = stream_usage.get("completion_tokens")
                    provider_usage = None
                    if prompt_tokens is not None or completion_tokens is not None:
                        provider_usage = (prompt_tokens, completion_tokens)
                    input_tokens, output_tokens, estimated = _resolve_usage_tokens(
                        provider_usage,
                        estimated_input_tokens=input_tokens,
                        response_text=full_response,
                    )
                    total_time = perf_counter() - start_time
                    ttft_ms: int | None = None
                    if first_token_time is not None:
                        ttft_ms = max(0, int((first_token_time - start_time) * 1000))
                    logger.info(format_metrics(model_name, input_tokens, output_tokens, total_time, ttft_ms))
                else:
                    raise StreamError("Stream completed without content")

                usage = Usage(
                    provider=provider.name,
                    model=model_spec,
                    model_name=model_name,
                    api_key_hint=preview_api_key(used_api_key),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration_ms=int(total_time * 1000),
                    ttft_ms=ttft_ms,
                    estimated=estimated,
                )
                if (sr := response_ref[0]) is not None:
                    sr.model = model_spec
                    sr.model_name = model_name
                    sr.provider = provider.name
                    sr.resolved_model = resolved_model
                    sr.usage = usage
                    sr.finish_reason = finish_reason
                if hook is not None:
                    hook(RequestEvent(usage=usage, error=None, timestamp=time.time()))
                return  # Stream completed successfully

            except Exception as e:
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
                if isinstance(e, StreamError) and e.partial:
                    last_error = e
                    logger.warning(f"Stream failed for model {m}, not retrying (partial output): {e}")
                    raise
                last_error = e
                logger.warning(f"Stream failed for model {m}, trying fallback: {e}")
                continue
        if last_error:
            raise last_error
        raise ValueError("No valid models found")

    # Placeholders; back-patched in _stream_with_fallback once a model wins.
    response_ref[0] = StreamResponse(
        stream=_stream_with_fallback(),
        model="unknown",
        model_name="unknown",
        provider=None,
    )
    return response_ref[0]
