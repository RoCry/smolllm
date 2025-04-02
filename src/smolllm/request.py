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
        if image_paths:
            raise ValueError(
                "Image paths are not supported with list prompt, you could put the images in the prompt instead"
            )
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
        # [OpenAI SDK compatibility (beta) - Anthropic](https://docs.anthropic.com/en/api/openai-sdk)
        url = f"{base_url}/v1"
    elif provider_name == "gemini":
        # [OpenAI compatibility | Gemini API](https://ai.google.dev/gemini-api/docs/openai)
        url = f"{base_url}/v1beta/openai/chat/completions"
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
    api_key: str,
) -> httpx.AsyncClient:
    """Prepare HTTP client and handle authentication"""
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}",
    }

    # Prepare client
    unsecure = url.startswith("http://")
    transport = httpx.AsyncHTTPTransport(local_address="0.0.0.0") if unsecure else None

    return httpx.AsyncClient(headers=headers, verify=not unsecure, transport=transport)
