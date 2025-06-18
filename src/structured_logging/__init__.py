"""
Structured Logging Library

A flexible Python library for structured JSON logging with context management.
"""

__version__ = "0.1.0"

from .config import LoggerConfig, get_default_config, set_default_config
from .context import (
    get_custom_context,
    get_request_id,
    get_user_context,
    request_context,
    set_custom_context,
    set_request_id,
    set_user_context,
    update_custom_context,
)
from .formatter import StructuredFormatter
from .logger import get_logger, log_with_context

__all__ = [
    "LoggerConfig",
    "get_default_config",
    "set_default_config",
    "request_context",
    "get_request_id",
    "set_request_id",
    "get_user_context",
    "set_user_context",
    "get_custom_context",
    "set_custom_context",
    "update_custom_context",
    "StructuredFormatter",
    "get_logger",
    "log_with_context",
]
