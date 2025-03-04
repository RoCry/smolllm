from typing import Any, Awaitable, Callable, Optional, TypeAlias

StreamHandler: TypeAlias = Callable[[str], None]
LLMFunction: TypeAlias = Callable[
    [str, Optional[str], Any], Awaitable[str]
]  # (prompt, system_prompt, **kwargs) -> response
