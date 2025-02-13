from dataclasses import dataclass
import json
from typing import Optional


@dataclass
class Provider:
    name: str
    base_url: str
    default_model_name: Optional[str] = None

    # Guess default model name if not provided
    # this is designed for random choosing from multiple providers
    def guess_default_model_name(self) -> Optional[str]:
        if self.default_model_name:
            return self.default_model_name

        if self.name == "gemini":
            return "gemini-2.0-flash"

        return None


def generate_provider_map() -> dict[str, Provider]:
    with open("providers.json", "r") as f:
        raw_providers = json.load(f)

    provider_map = {}
    for name, config in raw_providers.items():
        provider_map[name] = Provider(
            name=name,
            base_url=config["api"]["url"],
        )

    # add more
    provider_map["tencent_cloud"] = Provider(
        name="tencent_cloud",
        base_url="https://api.lkeap.cloud.tencent.com",
    )

    return provider_map


PROVIDERS = generate_provider_map()


# try parse provider and model name from model_str, e.g.
# "gemini/gemini-2.0-flash" -> ("gemini", "gemini-2.0-flash")
# "gemini" -> ("gemini", "gemini-2.0-flash") // /w default model for the provider
def parse_model_string(model_str: str) -> tuple[Provider, str]:
    """Parse model string into provider and model name"""
    if "/" in model_str:
        provider_name, model_name = model_str.split("/", 1)
    else:
        # Use the model string as provider name and get its default model
        provider_name = model_str

    provider = PROVIDERS.get(provider_name)
    if not provider:
        raise ValueError(f"Unknown provider: {provider_name}")
    model_name = model_name or provider.guess_default_model_name()
    if not model_name:
        raise ValueError(f"Model name not found for provider: {provider_name}")

    return provider, model_name
