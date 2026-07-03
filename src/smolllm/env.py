from __future__ import annotations

import os
from typing import Literal


def get_env_var(
    provider_name: str,
    var_type: Literal["API_KEY", "BASE_URL"],
    default: str | None = None,
) -> str:
    """Get environment variable for a provider with fallback to default."""
    env_key = f"{provider_name.upper()}_{var_type}"
    value: str | None = os.getenv(env_key, default)
    if not value and var_type == "API_KEY" and provider_name == "ollama":
        return "ollama"
    if not value:
        raise ValueError(
            f"{var_type} not found. Set {env_key} environment variable or pass {var_type.lower()} parameter"
        )
    return value
