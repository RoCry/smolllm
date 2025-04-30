import os
from dataclasses import dataclass
from typing import Dict, Optional

from .provider_config import PROVIDER_CONFIG


@dataclass
class Provider:
    name: str
    base_url: str
    default_model_name: Optional[str] = None

    # Guess default model name if not provided
    def guess_default_model_name(self) -> Optional[str]:
        if self.default_model_name:
            return self.default_model_name

        if self.name == "gemini":
            return "gemini-2.0-flash"

        return None


def generate_provider_map() -> Dict[str, Provider]:
    """Generate provider mapping from static configuration"""
    return {name: Provider(name=name, base_url=config["base_url"]) for name, config in PROVIDER_CONFIG.items()}


PROVIDERS = generate_provider_map()


# try parse provider and model name from model_str, e.g.
# "gemini/gemini-2.0-flash" -> ("gemini", "gemini-2.0-flash")
# "gemini" -> ("gemini", "gemini-2.0-flash") // /w default model for the provider
def parse_model_string(model_str: str) -> tuple[Provider, str]:
    model_name = None

    if "/" in model_str:
        provider_name, model_name = model_str.split("/", 1)
    else:
        # Use the model string as provider name and get its default model
        provider_name = model_str

    provider = PROVIDERS.get(provider_name)
    if not provider:
        # no predefined provider, try to get it from env
        key = f"{provider_name.upper()}_BASE_URL"
        base_url = os.getenv(key)
        if base_url:
            provider = Provider(name=provider_name, base_url=base_url)
        else:
            raise ValueError(f"Unknown provider name={provider_name} and {key} is not set")
    model_name = model_name or provider.guess_default_model_name()
    if not model_name:
        raise ValueError(f"Model name not found for provider: {provider_name}")

    return provider, model_name
