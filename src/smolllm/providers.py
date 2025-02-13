from dataclasses import dataclass
import json
import os
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


def parse_model_string(model_str: Optional[str] = None) -> tuple[Provider, str]:
    """Parse model string into provider and model name"""
    if not model_str:
        model_str = os.getenv("SMOLLLM_MODEL")

    model_name = None
    if "/" in model_str:
        provider_name, model_name = model_str.split("/", 1)
        if provider_name not in PROVIDERS:
            # special case: huihui_ai/deepseek-r1-abliterated:14b
            model_name = model_str
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
