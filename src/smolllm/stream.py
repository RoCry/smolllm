from typing import Optional


async def handle_chunk(chunk: dict, provider_name: str) -> Optional[str]:
    """Handle streaming chunks from different providers"""
    if provider_name == "gemini":
        candidates = chunk.get("candidates", [])
        if candidates and "content" in candidates[0]:
            return candidates[0]["content"]["parts"][0]["text"]
    elif provider_name == "anthropic":
        # Handle content_block_delta which contains the actual text
        if chunk.get("type") == "content_block_delta":
            delta = chunk.get("delta", {})
            if delta.get("type") == "text_delta":
                return delta.get("text", "")
        # Handle message_stop and other events that indicate completion
        elif chunk.get("type") in ["message_stop", "content_block_stop"]:
            return None
        # Other event types (message_start, content_block_start, ping, message_delta)
        # don't contain response text, so we return None
        return None
    else:  # openai-compatible
        choice = chunk.get("choices", [{}])[0]
        if choice.get("finish_reason") is not None:
            return None
        return choice.get("delta", {}).get("content", "")
    return None
