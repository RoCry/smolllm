import sys
from typing import Optional

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.rule import Rule
from rich.text import Text

from .types import StreamHandler


class ResponseDisplay:
    def __init__(self, stream_handler: Optional[StreamHandler] = None):
        self.stream_handler = stream_handler
        self.final_response = ""
        self.live: Optional[Live] = None
        # Check if we're in an interactive terminal
        self.is_interactive = sys.stdout.isatty() and sys.stderr.isatty()

    def __enter__(self):
        if self.is_interactive:
            self.live = Live(
                Group(Rule(style="grey50"), Text(""), Rule(style="grey50")),
                refresh_per_second=1,
                vertical_overflow="visible",
                console=Console(stderr=True),  # Send rich output to stderr
            ).__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.live:
            self.live.__exit__(exc_type, exc_val, exc_tb)

    async def update(self, delta: str):
        """Update display with new content"""
        if self.stream_handler:
            await self.stream_handler(delta)

        self.final_response += delta
        if self.is_interactive:
            self._update_display(with_cursor=True)

    def finalize(self):
        """Show final response without cursor"""
        if self.is_interactive:
            self._update_display(with_cursor=False)
        result = self.final_response.strip()
        if not result:
            raise ValueError("LLM returned an empty response")
        return result

    def _update_display(self, with_cursor: bool = True):
        """Internal method to update the live display"""
        if not self.live:
            return

        content = self.final_response + ("\n\n▌" if with_cursor else "")
        try:
            group = Group(Rule(style="grey50"), Markdown(content), Rule(style="grey50"))
        except Exception:
            # Fallback to plain text if markdown parsing fails
            group = Group(
                Rule(style="grey50"),
                Text(content, style="blink" if with_cursor else None),
                Rule(style="grey50"),
            )

        self.live.update(group)
