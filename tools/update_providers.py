#!/usr/bin/env python3
"""
Script to update provider configurations.
Usage: python -m tools.update_providers
"""

import json
from pathlib import Path
from typing import Any, Dict


def read_providers_json() -> Dict[str, Any]:
    """Read and parse providers.json"""
    json_path = Path(__file__).parent.parent / "providers.json"
    with open(json_path) as f:
        return json.load(f)


def generate_config_code(providers: Dict[str, Any]) -> str:
    """Generate Python code for provider configurations"""
    lines = [
        '"""',
        "Provider configurations - DO NOT EDIT MANUALLY",
        "Use tools/update_providers.py to update this file",
        'via https://github.com/CherryHQ/cherry-studio/blob/main/src/renderer/src/config/providers.ts"""',
        "",
        "PROVIDER_CONFIG = {",
    ]

    # Sort providers for consistent output
    for name in sorted(providers.keys()):
        config = providers[name]
        lines.extend(
            [
                f'    "{name}": {{',
                f'        "base_url": "{config["api"]["url"]}",',
                "    },",
            ]
        )

    lines.append("}")
    return "\n".join(lines)


def update_provider_config():
    """Update the provider_config.py file"""
    providers = read_providers_json()
    code = generate_config_code(providers)

    config_path = Path(__file__).parent.parent / "src" / "smolllm" / "provider_config.py"
    with open(config_path, "w") as f:
        f.write(code)

    print(f"Updated {config_path}")


if __name__ == "__main__":
    update_provider_config()
