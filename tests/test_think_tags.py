from __future__ import annotations

from smolllm.types import StreamChunk
from smolllm.utils import ThinkTagFilter, extract_think_tags

# --- extract_think_tags (non-streaming) ---


def test_extract_basic() -> None:
    reasoning, content = extract_think_tags("<think>reasoning</think>answer")
    assert reasoning == "reasoning"
    assert content == "answer"


def test_extract_multiple_blocks() -> None:
    text = "<think>first</think>middle<think>second</think>end"
    reasoning, content = extract_think_tags(text)
    assert reasoning == "first\n\nsecond"
    assert content == "middleend"


def test_extract_no_think_tags() -> None:
    reasoning, content = extract_think_tags("just plain text")
    assert reasoning == ""
    assert content == "just plain text"


def test_extract_multiline_think() -> None:
    text = "<think>\nline1\nline2\n</think>answer"
    reasoning, content = extract_think_tags(text)
    assert "line1" in reasoning
    assert "line2" in reasoning
    assert content == "answer"


# --- ThinkTagFilter (streaming) ---


def test_filter_basic() -> None:
    f = ThinkTagFilter()
    result = f.feed(StreamChunk(content="<think>thought</think>answer"))
    assert result.reasoning == "thought"
    assert result.content == "answer"


def test_filter_split_across_chunks() -> None:
    f = ThinkTagFilter()
    r1 = f.feed(StreamChunk(content="<think>tho"))
    assert r1.reasoning == "tho"
    assert r1.content == ""

    r2 = f.feed(StreamChunk(content="ught</think>answer"))
    assert r2.reasoning == "ught"
    assert r2.content == "answer"


def test_filter_tag_split_at_boundary() -> None:
    """Opening tag split across two chunks."""
    f = ThinkTagFilter()
    r1 = f.feed(StreamChunk(content="<thi"))
    # The partial tag should be buffered, not emitted as content
    assert r1.content == ""
    assert r1.reasoning == ""

    r2 = f.feed(StreamChunk(content="nk>reasoning</think>content"))
    assert r2.reasoning == "reasoning"
    assert r2.content == "content"


def test_filter_closing_tag_split() -> None:
    """Closing tag split across two chunks."""
    f = ThinkTagFilter()
    r1 = f.feed(StreamChunk(content="<think>thought</th"))
    assert r1.reasoning == "thought"

    r2 = f.feed(StreamChunk(content="ink>answer"))
    assert r2.reasoning == ""
    assert r2.content == "answer"


def test_filter_flush() -> None:
    """Content inside <think> without closing tag is emitted as reasoning immediately;
    only partial tag fragments get buffered for flush."""
    f = ThinkTagFilter()
    r1 = f.feed(StreamChunk(content="<think>partial"))
    # "partial" is emitted as reasoning since no closing tag boundary to buffer
    assert r1.reasoning == "partial"

    # Feed content ending with a partial closing tag
    r2 = f.feed(StreamChunk(content="more</thi"))
    assert r2.reasoning == "more"
    # "</thi" is buffered
    result = f.flush()
    assert result.reasoning == "</thi"


def test_filter_flush_empty() -> None:
    f = ThinkTagFilter()
    result = f.flush()
    assert not result


def test_filter_passthrough_when_backend_provides_reasoning() -> None:
    """If backend already sends reasoning, filter disables itself."""
    f = ThinkTagFilter()
    chunk = StreamChunk(content="<think>inline</think>text", reasoning="backend reasoning")
    result = f.feed(chunk)
    # Should pass through unchanged
    assert result.content == "<think>inline</think>text"
    assert result.reasoning == "backend reasoning"

    # Subsequent chunks should also pass through
    chunk2 = StreamChunk(content="<think>more</think>stuff")
    result2 = f.feed(chunk2)
    assert result2.content == "<think>more</think>stuff"
    assert result2.reasoning == ""


def test_filter_no_think_tags() -> None:
    f = ThinkTagFilter()
    result = f.feed(StreamChunk(content="just content"))
    assert result.content == "just content"
    assert result.reasoning == ""
