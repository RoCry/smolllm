from typing import Optional


async def handle_chunk(chunk: dict, provider_name: str) -> Optional[str]:
    """Handle streaming chunks from different providers"""
    if provider_name == "gemini":
        candidates = chunk.get("candidates", [])
        if candidates and "content" in candidates[0]:
            return candidates[0]["content"]["parts"][0]["text"]
    else:  # openai-compatible
        choice = chunk.get("choices", [{}])[0]
        if choice.get("finish_reason") is not None:
            return None
        return choice.get("delta", {}).get("content", "")
    return None
