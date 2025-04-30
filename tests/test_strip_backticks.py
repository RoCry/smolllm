from smolllm.utils import strip_backticks

keep_original_cases = [
    """```markdown
# Hello, world!
```
above are part of block quote
""",
]


def test_strip_backticks():
    assert strip_backticks("```python\nprint('Hello, world!')```") == "print('Hello, world!')"
    assert strip_backticks("```python\nprint('Hello, world!')\n```") == "print('Hello, world!')"

    assert strip_backticks("```\nplain text\n```") == "plain text"
    assert strip_backticks("```\nplain text```") == "plain text"

    assert strip_backticks("```markdown\n# Hello, world!\n```") == "# Hello, world!"
    assert strip_backticks("```markdown\n# Hello, world!```") == "# Hello, world!"

    # needs keep part of the backticks
    for case in keep_original_cases:
        assert strip_backticks(case) == case
