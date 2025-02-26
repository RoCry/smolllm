from typing import Dict, Any
import httpx


def prepare_request_data(
    prompt: str,
    system_prompt: str | None,
    model_name: str,
    provider_name: str,
    base_url: str,
) -> tuple[str, Dict[str, Any]]:
    """Prepare request URL, data and headers for the API call"""
    base_url = base_url.rstrip("/")

    if provider_name == "anthropic":
        url = f"{base_url}/v1/messages"
        data = {
            "model": model_name,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        if system_prompt:
            data["system"] = system_prompt
    elif provider_name == "gemini":
        url = f"{base_url}/v1beta/models/{model_name}:streamGenerateContent?alt=sse"
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
        }
        if system_prompt:
            data["system_instruction"] = {"parts": [{"text": system_prompt}]}
    else:
        # all openai compatible api
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        data = {
            "messages": messages,
            "model": model_name,
            "stream": True,
        }

        # Handle URL based on suffix
        if base_url.endswith("#"):
            url = base_url[:-1]  # Remove the # and use exact URL
        elif base_url.endswith("/"):
            url = f"{base_url}chat/completions"  # Skip v1 prefix
        else:
            url = f"{base_url}/v1/chat/completions"  # Default pattern

    return url, data


def prepare_client_and_auth(
    url: str,
    provider_name: str,
    api_key: str,
) -> httpx.AsyncClient:
    """Prepare HTTP client and handle authentication"""
    # Handle authentication
    headers = {"content-type": "application/json"}
    if provider_name == "anthropic":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
    elif provider_name == "gemini":
        headers["x-goog-api-key"] = api_key
    else:
        headers["authorization"] = f"Bearer {api_key}"

    # Prepare client
    unsecure = url.startswith("http://")
    transport = httpx.AsyncHTTPTransport(local_address="0.0.0.0") if unsecure else None

    return httpx.AsyncClient(headers=headers, verify=not unsecure, transport=transport)
