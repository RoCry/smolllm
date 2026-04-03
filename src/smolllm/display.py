from __future__ import annotations

import sys
from types import TracebackType

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.rule import Rule
from rich.text import Text

from .types import StreamChunk, StreamHandler


class ResponseDisplay:
    def __init__(self, stream_handler: StreamHandler | None = None):
        self.stream_handler: StreamHandler | None = stream_handler
        self.final_response: str = ""
        self.reasoning_response: str = ""
        self.live: Live | None = None
        self.is_interactive: bool = sys.stdout.isatty() and sys.stderr.isatty()

    def __enter__(self) -> ResponseDisplay:
        if self.is_interactive:
            self.live = Live(
                Group(Rule(style="grey50"), Text(""), Rule(style="grey50")),
                refresh_per_second=1,
                vertical_overflow="visible",
                console=Console(stderr=True),
            ).__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self.live:
            self.live.__exit__(exc_type, exc_val, exc_tb)

    async def update(self, chunk: StreamChunk) -> None:
        """Update display with new content"""
        if self.stream_handler:
            await self.stream_handler(chunk)

        self.reasoning_response += chunk.reasoning
        self.final_response += chunk.content
        if self.is_interactive:
            self._update_display(with_cursor=True)

    def finalize(self) -> tuple[str, str]:
        """Show final response without cursor. Returns (text, reasoning)."""
        if self.is_interactive:
            self._update_display(with_cursor=False)
        text = self.final_response.strip()
        reasoning = self.reasoning_response.strip()
        if not text and not reasoning:
            raise ValueError("LLM returned an empty response")
        return text, reasoning

    def _update_display(self, with_cursor: bool = True) -> None:
        """Internal method to update the live display"""
        if not self.live:
            return

        parts: list[Rule | Markdown | Text] = [Rule(style="grey50")]

        # Show reasoning in dim italic if present
        if self.reasoning_response:
            thinking_header = Text("Thinking...", style="dim italic")
            parts.append(thinking_header)
            try:
                parts.append(Markdown(self.reasoning_response, style="dim italic"))
            except Exception:
                parts.append(Text(self.reasoning_response, style="dim italic"))
            parts.append(Rule(style="dim"))

        content = self.final_response + ("\n\n▌" if with_cursor else "")
        try:
            parts.append(Markdown(content))
        except Exception:
            text = Text(content, style="blink") if with_cursor else Text(content)
            parts.append(text)

        parts.append(Rule(style="grey50"))
        self.live.update(Group(*parts))
