from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    TypeAlias,
    Union,
)

StreamHandler: TypeAlias = Callable[[str], None]
LLMFunction: TypeAlias = Callable[
    [str, Optional[str], Any], Awaitable[str]
]  # (prompt, system_prompt, **kwargs) -> response

MessageRole = Literal["user", "assistant"]
Message = Dict[str, Union[str, List[Dict[str, Any]]]]
PromptType = Union[str, List[Message]]
