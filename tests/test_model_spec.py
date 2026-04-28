from smolllm.providers import parse_model_spec


def test_no_suffix():
    assert parse_model_spec("groq/qwen/qwen3-32b") == ("groq/qwen/qwen3-32b", None)


def test_with_effort():
    assert parse_model_spec("groq/qwen/qwen3-32b!none") == ("groq/qwen/qwen3-32b", "none")


def test_with_medium():
    assert parse_model_spec("openai/gpt-5!medium") == ("openai/gpt-5", "medium")


def test_trailing_separator_no_value():
    assert parse_model_spec("groq/qwen/qwen3-32b!") == ("groq/qwen/qwen3-32b", None)


def test_strips_whitespace():
    assert parse_model_spec("  groq/qwen/qwen3-32b  ! none ") == ("groq/qwen/qwen3-32b", "none")


def test_only_provider():
    assert parse_model_spec("gemini") == ("gemini", None)


def test_only_provider_with_effort():
    assert parse_model_spec("gemini!low") == ("gemini", "low")
