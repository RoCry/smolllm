from rich.live import Live
from rich.text import Text
from rich.markdown import Markdown
from rich.console import Group
from rich.rule import Rule
from typing import Optional
from .types import StreamHandler


class ResponseDisplay:
    def __init__(self, stream_handler: Optional[StreamHandler] = None):
        self.stream_handler = stream_handler
        self.final_response = ""
        self.live: Optional[Live] = None

    def __enter__(self):
        self.live = Live(
            Group(Rule(style="grey50"), Text(""), Rule(style="grey50")),
            refresh_per_second=4,
            vertical_overflow="visible",
        ).__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.live:
            self.live.__exit__(exc_type, exc_val, exc_tb)

    def update(self, delta: str):
        """Update display with new content"""
        if self.stream_handler:
            self.stream_handler(delta)

        self.final_response += delta
        self._update_display(with_cursor=True)

    def finalize(self):
        """Show final response without cursor"""
        self._update_display(with_cursor=False)
        return self.final_response.strip()

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
