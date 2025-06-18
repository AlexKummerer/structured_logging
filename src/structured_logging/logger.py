import logging
import sys
from typing import Any, Dict, Optional

from .config import LoggerConfig, get_default_config
from .context import get_custom_context, get_request_id, get_user_context
from .formatter import StructuredFormatter


def get_logger(name: str, config: Optional[LoggerConfig] = None) -> logging.Logger:
    """Create a structured logger with the given name"""
    logger = logging.getLogger(name)

    if not logger.handlers:
        config = config or get_default_config()
        logger.setLevel(getattr(logging, config.log_level.upper()))

        handler = logging.StreamHandler(sys.stdout)
        formatter = StructuredFormatter(config)
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.propagate = True

    return logger


def log_with_context(
    logger: logging.Logger,
    level: str,
    message: str,
    config: Optional[LoggerConfig] = None,
    **extra: Any,
) -> None:
    """Log with automatic context injection"""
    config = config or get_default_config()
    context: Dict[str, Any] = {}

    if config.include_request_id:
        req_id = get_request_id()
        if req_id:
            context["ctx_request_id"] = req_id

    if config.include_user_context:
        user_ctx = get_user_context()
        for key, value in user_ctx.items():
            if value is not None:
                context[f"ctx_{key}"] = value

    custom_ctx = get_custom_context()
    for key, value in custom_ctx.items():
        if value is not None:
            context[f"ctx_{key}"] = value

    for key, value in extra.items():
        if value is not None:
            context[f"ctx_{key}"] = value

    getattr(logger, level)(message, extra=context)
