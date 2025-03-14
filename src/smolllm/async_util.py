"""Async utilities for SmolLLM"""

import asyncio
from functools import wraps
from typing import Any, Callable, TypeVar

T = TypeVar('T')

def run_async(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to run an async function in a synchronous context.
    
    Args:
        func: The async function to run
        
    Returns:
        A synchronous function that runs the async function
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new loop if we're already in one
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(func(*args, **kwargs))
                finally:
                    new_loop.close()
                    asyncio.set_event_loop(loop)
            else:
                return loop.run_until_complete(func(*args, **kwargs))
        except RuntimeError:
            # If there's no event loop, create one
            return asyncio.run(func(*args, **kwargs))
    return wrapper 