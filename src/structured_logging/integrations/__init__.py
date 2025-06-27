"""
Framework integrations for structured logging
"""

from .config import FastAPILoggingConfig
from .fastapi import (
    FASTAPI_AVAILABLE,
    FastAPILoggingMiddleware,
    add_structured_logging,
    create_fastapi_logger_config,
)
from .flask import (
    FLASK_AVAILABLE,
    FlaskLoggingMiddleware,
    create_flask_logger_config,
)

__all__ = [
    "FastAPILoggingConfig",
    "FastAPILoggingMiddleware",
    "add_structured_logging",
    "create_fastapi_logger_config",
    "create_flask_logger_config",
    "FlaskLoggingMiddleware",
    "FASTAPI_AVAILABLE",
    "FLASK_AVAILABLE",
]