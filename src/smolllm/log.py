import os
import logging
from rich.logging import RichHandler

# Get log level from environment or default to INFO
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Set root logger to INFO to suppress debug logs from other libraries
logging.getLogger().setLevel(logging.INFO)

# Configure rich handler
rich_handler = RichHandler(rich_tracebacks=True)
rich_handler.setFormatter(logging.Formatter("%(message)s"))

# Configure our package logger
logger = logging.getLogger("smolllm")
logger.setLevel(getattr(logging, log_level, logging.INFO))
logger.handlers = [rich_handler]

# Prevent propagation to root logger to avoid duplicate logs
logger.propagate = False
