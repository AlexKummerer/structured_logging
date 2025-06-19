"""
Structured Logging Library

A flexible Python library for structured JSON logging with context management and async support.
"""

__version__ = "0.4.0"

from .config import LoggerConfig, get_default_config, set_default_config, FormatterType, OutputType
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
from .formatter import StructuredFormatter, CSVFormatter, PlainTextFormatter
from .logger import get_logger, log_with_context, get_filter_metrics, reset_filter_metrics
from .filtering import (
    FilterConfig, 
    FilterEngine, 
    LevelFilter, 
    ContextFilter, 
    CustomFilter, 
    SamplingFilter, 
    FilterResult
)
from .handlers import (
    FileHandlerConfig,
    RotatingFileHandler,
    TimedRotatingFileHandler,
    create_file_logger
)
from .integrations import (
    FastAPILoggingConfig,
    FastAPILoggingMiddleware,
    add_structured_logging,
    create_fastapi_logger_config,
    create_flask_logger_config,
    FlaskLoggingMiddleware
)

# Async logging support
from .async_config import (
    AsyncLoggerConfig,
    get_default_async_config,
    set_default_async_config,
)
from .async_context import (
    async_request_context,
    aget_request_id,
    aget_user_context,
    aget_custom_context,
    aset_request_id,
    aset_user_context,
    aset_custom_context,
)
from .async_logger import (
    AsyncLogger,
    get_async_logger,
    alog_with_context,
    shutdown_all_async_loggers,
)

__all__ = [
    # Sync API
    "LoggerConfig",
    "FormatterType",
    "OutputType",
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
    "CSVFormatter",
    "PlainTextFormatter",
    "get_logger",
    "log_with_context",
    "get_filter_metrics",
    "reset_filter_metrics",
    "FilterConfig",
    "FilterEngine",
    "LevelFilter",
    "ContextFilter",
    "CustomFilter",
    "SamplingFilter",
    "FilterResult",
    "FileHandlerConfig",
    "RotatingFileHandler",
    "TimedRotatingFileHandler",
    "create_file_logger",
    "FastAPILoggingConfig",
    "FastAPILoggingMiddleware", 
    "add_structured_logging",
    "create_fastapi_logger_config",
    "create_flask_logger_config",
    "FlaskLoggingMiddleware",
    
    # Async API
    "AsyncLoggerConfig",
    "get_default_async_config",
    "set_default_async_config",
    "async_request_context",
    "aget_request_id",
    "aget_user_context", 
    "aget_custom_context",
    "aset_request_id",
    "aset_user_context",
    "aset_custom_context",
    "AsyncLogger",
    "get_async_logger",
    "alog_with_context",
    "shutdown_all_async_loggers",
]
