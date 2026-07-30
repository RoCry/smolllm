from __future__ import annotations

import os
from dataclasses import dataclass

from .env import get_env_var
from .provider_config import PROVIDER_CONFIG


@dataclass
class Provider:
    name: str
    base_url: str


def generate_provider_map() -> dict[str, Provider]:
    """Generate provider mapping from static configuration"""
    return {name: Provider(name=name, base_url=config["base_url"]) for name, config in PROVIDER_CONFIG.items()}


PROVIDERS = generate_provider_map()


# Split "provider/model!effort" into ("provider/model", "effort"); no suffix → effort is None.
# An empty value (e.g. "model!") parses as ("model", None) so trailing separators are no-ops.
def parse_model_spec(spec: str) -> tuple[str, str | None]:
    model, sep, effort = spec.partition("!")
    if not sep:
        return model.strip(), None
    effort = effort.strip()
    return model.strip(), effort or None


# Parse a model string into (Provider, model_name), splitting on the FIRST "/", e.g.
# "gemini/gemini-2.0-flash" -> (Provider(gemini), "gemini-2.0-flash")
# A string without "/" is a bare model name with NO provider: base_url must be
# passed explicitly and the returned Provider carries an empty name.
def parse_model_string(model_str: str, *, base_url: str | None = None) -> tuple[Provider, str]:
    if "/" not in model_str:
        if not model_str:
            raise ValueError("Model string must not be empty")
        if not base_url:
            raise ValueError(f"Bare model '{model_str}' requires base_url. Pass base_url or use provider/model format")
        return Provider(name="", base_url=base_url), model_str

    provider_name, model_name = model_str.split("/", 1)
    if not provider_name:
        raise ValueError(f"Empty provider in model string '{model_str}'")
    if not model_name:
        raise ValueError(f"Empty model name in model string '{model_str}'")

    if (provider := PROVIDERS.get(provider_name)) is not None:
        return provider, model_name
    if base_url:
        return Provider(name=provider_name, base_url=base_url), model_name
    # no predefined provider, try to get it from env
    key = f"{provider_name.upper()}_BASE_URL"
    if env_base_url := os.getenv(key):
        return Provider(name=provider_name, base_url=env_base_url), model_name
    raise ValueError(f"Unknown provider name={provider_name}. Pass base_url or set {key}.")


def resolve_credentials(
    provider: Provider,
    *,
    model_name: str,
    base_url: str | None,
    api_key: str | None,
) -> tuple[str, str]:
    """Resolve the final (base_url, api_key) for a parsed model.

    Bare mode (empty provider name) is explicit-only: no env lookup ever runs
    without a provider name.
    """
    if not provider.name:
        if not api_key:
            raise ValueError(f"Bare model '{model_name}' requires api_key. Pass api_key or use provider/model format")
        return provider.base_url, api_key
    return (
        base_url or get_env_var(provider.name, "BASE_URL", provider.base_url),
        api_key or get_env_var(provider.name, "API_KEY"),
    )
