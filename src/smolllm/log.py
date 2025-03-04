import logging
import os

from rich.console import Console
from rich.logging import RichHandler

# Get log level from environment or default to INFO
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Set root logger to INFO to suppress debug logs from other libraries
logging.getLogger().setLevel(logging.INFO)

# Create console that outputs to stderr with full width
console = Console(
    stderr=True,
    width=None,  # This will use the full terminal width
)

# Configure rich handler to output to stderr
rich_handler = RichHandler(
    rich_tracebacks=True,
    console=console,  # Use stderr console
    show_time=False,  # removes timestamp for cleaner output
)
rich_handler.setFormatter(logging.Formatter("%(message)s"))

# Configure our package logger
logger = logging.getLogger("smolllm")
logger.setLevel(getattr(logging, log_level, logging.INFO))
logger.handlers = [rich_handler]

# Prevent propagation to root logger to avoid duplicate logs
logger.propagate = False
