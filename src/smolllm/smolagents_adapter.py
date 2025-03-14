"""Adapter for SmolLLM to be used with smolagents"""

import asyncio
import inspect
from typing import Any, Dict, List, Optional

from smolagents.models import Model, ChatMessage
from smolagents import Tool

from .log import logger

class SmolllmModel(Model):
    