import json
from typing import Optional

from .log import logger


def _handle_chunk(chunk: dict) -> Optional[str]:
    choices = chunk.get("choices")
    if not choices:
        return None
    choice = choices[0]
    content = choice.get("delta", {}).get("content")
    return content


async def process_chunk_line(line: str) -> Optional[str]:
    """Process a single chunk of data from the stream"""
    line = line.strip()
    if not line or line == "data: [DONE]" or not line.startswith("data: "):
        return None
    try:
        chunk = json.loads(line[6:])  # Remove "data: " prefix
        return _handle_chunk(chunk)
    except Exception as e:
        # acceptable errors here, just log them
        logger.error(f"Error processing chunk: {e}")
        return None
