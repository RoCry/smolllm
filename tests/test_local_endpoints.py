"""LAN endpoints bypass every proxy; proxy error pages shrink to their title.

2026-10-31: smolllm-server on rocry-studio.local was down. httpx sent the call
through the macOS system proxy (Surge), which answered a plain "connection
refused" with the HTML 503 page below, and callers logged 500 chars of CSS.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from smolllm.errors import brief_error_detail
from smolllm.request import is_local_host, prepare_client_and_auth

# Head of the page Surge returned on 2026-10-31 (captured from the mnemo log).
SURGE_PAGE = (
    '<!doctype html> <html> <head> <meta charset="utf-8"> <meta name="viewport" '
    'content="width=device-width, initial-scale=1, maximum-scale=1"> <meta http-equiv="X-UA-Compatible" '
    'content="IE=edge,chrome=1"> <title>Unable to connect</title> <style> * { margin: 0; '
    "-webkit-font-smoothing: antialiased; } body { font-family: Helvetica, sans-serif; font-size: 14px; } "
    "</style></head><body>...</body></html>"
)


def test_a_proxy_error_page_reads_as_its_title() -> None:
    assert brief_error_detail(SURGE_PAGE) == "HTML page: Unable to connect"


@pytest.mark.parametrize(
    ("host", "local"),
    [
        ("rocry-studio.local", True),
        ("localhost", True),
        ("127.0.0.1", True),
        ("192.168.31.143", True),
        ("10.0.0.5", True),
        ("::1", True),
        ("api.deepseek.com", False),
        ("8.8.8.8", False),
        ("local.example.com", False),
    ],
)
def test_local_hosts(host: str, local: bool) -> None:
    assert is_local_host(host) is local


class _Ok(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *_: object) -> None:
        pass


@pytest.mark.asyncio
async def test_a_loopback_call_ignores_a_configured_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    server = HTTPServer(("127.0.0.1", 0), _Ok)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    # A proxy that refuses everything: going through it would fail the call.
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    url = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
    try:
        async with prepare_client_and_auth(url, "key") as client:
            response = await client.get(url)
    finally:
        server.shutdown()
    assert response.status_code == 200
