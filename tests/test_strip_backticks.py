from smolllm.utils import strip_backticks

def test_strip_backticks():
    assert strip_backticks("```python\nprint('Hello, world!')```") == "print('Hello, world!')"
