#!/usr/bin/env python3
import json
from pathlib import Path

def json_to_python():
    """Convert providers.json to Python code"""
    with open("providers.json") as f:
        data = json.load(f)
    
    lines = ["PROVIDER_CONFIG = {"]
    for name, config in data.items():
        lines.append(f'    "{name}": {{')
        lines.append(f'        "base_url": "{config["api"]["url"]}",')
        lines.append("    },")
    lines.append("}")
    
    output = "\n".join(lines)
    print(output)  # You can redirect this to update providers.py

if __name__ == "__main__":
    json_to_python() 