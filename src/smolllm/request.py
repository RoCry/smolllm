import base64
import mimetypes
from typing import Any, Dict, List, Optional

import httpx

from .types import PromptType


def get_mime_type(image_path: str) -> str:
    """Get the MIME type of a file based on its extension"""
    mime_type, _ = mimetypes.guess_type(image_path)
    return mime_type or "application/octet-stream"


def encode_image(image_path: str) -> tuple[str, str]:
    """Encode image to base64 and get mime type"""
    mime_type = get_mime_type(image_path)
    with open(image_path, "rb") as img_file:
        image_data = base64.b64encode(img_file.read()).decode()
    return image_data, mime_type


def _prepare_anthropic_request(
    prompt: PromptType,
    system_prompt: str | None,
    model_name: str,
    image_paths: List[str],
) -> Dict[str, Any]:
    content = []

    if isinstance(prompt, str):
        content.append({"type": "text", "text": prompt})
    else:
        # Convert message history to Anthropic format
        messages = []
        for msg in prompt:
            content = [{"type": "text", "text": msg["content"]}]
            messages.append({"role": msg["role"], "content": content})
        return {
            "model": model_name,
            "max_tokens": 4096,
            "messages": messages,
            "stream": True,
            "system": system_prompt if system_prompt else None,
        }

    for image_path in image_paths:
        image_data, mime_type = encode_image(image_path)
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": image_data,
                },
            }
        )

    data = {
        "model": model_name,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": content}],
        "stream": True,
    }
    if system_prompt:
        data["system"] = system_prompt
    return data


def _prepare_gemini_request(
    prompt: PromptType,
    system_prompt: str | None,
    image_paths: List[str],
) -> Dict[str, Any]:
    if isinstance(prompt, list):
        # Gemini doesn't support chat history directly, concatenate messages
        text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in prompt)
        parts = [{"text": text}]
    else:
        parts = [{"text": prompt}]

    for image_path in image_paths:
        image_data, mime_type = encode_image(image_path)
        parts.append(
            {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": image_data,
                }
            }
        )

    data = {"contents": [{"parts": parts}]}
    if system_prompt:
        data["system_instruction"] = {"parts": [{"text": system_prompt}]}
    return data


def _prepare_openai_request(
    prompt: PromptType,
    system_prompt: str | None,
    model_name: str,
    image_paths: List[str],
) -> Dict[str, Any]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if isinstance(prompt, list):
        messages.extend(prompt)
    else:
        if image_paths:
            content = [{"type": "text", "text": prompt}]
            for image_path in image_paths:
                image_data, mime_type = encode_image(image_path)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                    }
                )
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

    return {
        "messages": messages,
        "model": model_name,
        "stream": True,
    }


def prepare_request_data(
    prompt: PromptType,
    system_prompt: str | None,
    model_name: str,
    provider_name: str,
    base_url: str,
    image_paths: Optional[List[str]] = None,
) -> tuple[str, Dict[str, Any]]:
    """Prepare request URL, data and headers for the API call"""
    base_url = base_url.rstrip("/")
    image_paths = image_paths or []

    if provider_name == "anthropic":
        url = f"{base_url}/v1/messages"
        data = _prepare_anthropic_request(
            prompt, system_prompt, model_name, image_paths
        )
    elif provider_name == "gemini":
        url = f"{base_url}/v1beta/models/{model_name}:streamGenerateContent?alt=sse"
        data = _prepare_gemini_request(prompt, system_prompt, image_paths)
    else:
        # Handle URL based on suffix
        if base_url.endswith("#"):
            url = base_url[:-1]
        elif base_url.endswith("/"):
            url = f"{base_url}chat/completions"
        else:
            url = f"{base_url}/v1/chat/completions"
        data = _prepare_openai_request(prompt, system_prompt, model_name, image_paths)

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
