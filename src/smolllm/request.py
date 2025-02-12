from typing import Dict, Any
import httpx


def prepare_request_data(
    prompt: str,
    system_prompt: str | None,
    model_name: str,
    provider_name: str,
    base_url: str,
) -> tuple[str, Dict[str, Any], Dict[str, str]]:
    """Prepare request URL, data and headers for the API call"""
    base_url = base_url.rstrip("/")

    if provider_name == "gemini":
        url = f"{base_url}/v1beta/models/{model_name}:streamGenerateContent?alt=sse"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
        }
        if system_prompt:
            data["system_instruction"] = {"parts": [{"text": system_prompt}]}
    else:
        url = f"{base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        data = {
            "messages": messages,
            "model": model_name,
            "stream": True,
        }

    return url, data, headers


def prepare_client_and_auth(
    url: str, provider_name: str, api_key: str, headers: Dict[str, str]
) -> httpx.AsyncClient:
    """Prepare HTTP client and handle authentication"""
    # Handle authentication
    if provider_name == "gemini":
        headers["x-goog-api-key"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"

    # Prepare client
    unsecure = url.startswith("http://")
    transport = httpx.AsyncHTTPTransport(local_address="0.0.0.0") if unsecure else None

    return httpx.AsyncClient(verify=not unsecure, transport=transport)
