from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from time import monotonic

import httpx
import pytest

import smolllm.core as core
from smolllm import ask_llm
from smolllm.deadline import Deadline
from smolllm.http_stream import iter_stream_lines


class _Trickle(httpx.AsyncByteStream):
    """A server that never finishes: keepalive comments forever, faster than any per-chunk timeout."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        while True:
            yield b": keepalive\n"
            await asyncio.sleep(0.02)


def _trickle_transport(requests: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, stream=_Trickle())

    return httpx.MockTransport(handler)


def test_deadline_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="positive"):
        Deadline(0)


def test_deadline_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [100.0]
    monkeypatch.setattr("smolllm.deadline.monotonic", lambda: now[0])
    deadline = Deadline(5)
    assert deadline.remaining() == 5
    now[0] = 105.0
    with pytest.raises(TimeoutError, match="5s timeout"):
        deadline.remaining()


@pytest.mark.asyncio
async def test_trickling_stream_cannot_outlive_the_deadline() -> None:
    async with httpx.AsyncClient(transport=_trickle_transport([])) as client:
        start = monotonic()
        with pytest.raises(TimeoutError):
            async for _ in iter_stream_lines(client, "https://example.test/chat", {"stream": True}, Deadline(0.2)):
                pass
    assert monotonic() - start < 1


@pytest.mark.asyncio
async def test_expired_deadline_ends_the_fallback_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[httpx.Request] = []
    monkeypatch.setattr(
        core, "prepare_client_and_auth", lambda _u, _k: httpx.AsyncClient(transport=_trickle_transport(requests))
    )

    with pytest.raises(TimeoutError):
        await ask_llm("q", model="testprov/m1,testprov/m2", api_key="k", base_url="http://t.local/v1", timeout=0.2)

    assert len(requests) == 1  # the second model is never started once the whole-call budget is spent
