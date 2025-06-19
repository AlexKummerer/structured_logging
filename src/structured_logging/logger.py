import logging
import sys
from typing import Any, Dict, Optional
from functools import lru_cache

from .config import LoggerConfig, get_default_config
from .context import get_custom_context, get_request_id, get_user_context
from .formatter import StructuredFormatter, CSVFormatter, PlainTextFormatter


# Performance optimization: Cache formatter instances
_formatter_cache: Dict[str, logging.Formatter] = {}


def get_logger(name: str, config: Optional[LoggerConfig] = None) -> logging.Logger:
    """Create a structured logger with the given name"""
    logger = logging.getLogger(name)

    if not logger.handlers:
        config = config or get_default_config()
        logger.setLevel(getattr(logging, config.log_level.upper()))

        handler = logging.StreamHandler(sys.stdout)
        
        # Performance optimization: Use cached formatter if available
        cache_key = f"{config.formatter_type}_{config.include_timestamp}_{config.include_request_id}_{config.include_user_context}"
        
        if cache_key not in _formatter_cache:
            # Select formatter based on config
            if config.formatter_type == "csv":
                formatter = CSVFormatter(config)
            elif config.formatter_type == "plain":
                formatter = PlainTextFormatter(config)
            else:  # default to json
                formatter = StructuredFormatter(config)
            
            _formatter_cache[cache_key] = formatter
        else:
            formatter = _formatter_cache[cache_key]
            
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
    """Log with automatic context injection - optimized version"""
    config = config or get_default_config()
    context: Dict[str, Any] = {}

    # Optimized: Batch context retrieval to minimize context variable lookups
    if config.include_request_id or config.include_user_context:
        # Single context variable access
        req_id = get_request_id() if config.include_request_id else None
        user_ctx = get_user_context() if config.include_user_context else {}
        
        if req_id:
            context["ctx_request_id"] = req_id
        
        if user_ctx:
            # Optimized: Use dict comprehension for better performance
            context.update(
                {f"ctx_{k}": v for k, v in user_ctx.items() if v is not None}
            )

    # Custom context - only access if likely to have data
    custom_ctx = get_custom_context()
    if custom_ctx:
        context.update(
            {f"ctx_{k}": v for k, v in custom_ctx.items() if v is not None}
        )

    # Extra context - optimized dict comprehension
    if extra:
        context.update(
            {f"ctx_{k}": v for k, v in extra.items() if v is not None}
        )

    getattr(logger, level)(message, extra=context)
