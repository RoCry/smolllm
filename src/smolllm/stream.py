from typing import Optional


def handle_chunk(chunk: dict) -> Optional[str]:
    choices = chunk.get("choices")
    if not choices:
        return None
    choice = choices[0]
    content = choice.get("delta", {}).get("content", "")
    if content:
        return content
    return None
