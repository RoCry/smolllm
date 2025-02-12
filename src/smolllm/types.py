from typing import Callable, Awaitable, Optional, TypeAlias, Any

StreamHandler: TypeAlias = Callable[[str], None]
LLMFunction: TypeAlias = Callable[
    [str, Optional[str], Any], Awaitable[str]
]  # (prompt, system_prompt, **kwargs) -> response
