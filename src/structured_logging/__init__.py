"""
Structured Logging Library

A flexible Python library for structured JSON logging with context management and async support.
"""

__version__ = "0.5.0"

# Async logging support
from .async_config import (
    AsyncLoggerConfig,
    get_default_async_config,
    set_default_async_config,
)
from .async_context import (
    aget_custom_context,
    aget_request_id,
    aget_user_context,
    aset_custom_context,
    aset_request_id,
    aset_user_context,
    async_request_context,
)
from .async_logger import (
    AsyncLogger,
    alog_with_context,
    get_async_logger,
    shutdown_all_async_loggers,
)
from .config import (
    FormatterType,
    LoggerConfig,
    OutputType,
    get_default_config,
    set_default_config,
)
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
from .filtering import (
    ContextFilter,
    CustomFilter,
    FilterConfig,
    FilterEngine,
    FilterResult,
    LevelFilter,
    SamplingFilter,
)
from .formatter import CSVFormatter, PlainTextFormatter, StructuredFormatter
from .handlers import (
    FileHandlerConfig,
    RotatingFileHandler,
    TimedRotatingFileHandler,
    create_file_logger,
)
from .integrations import (
    FastAPILoggingConfig,
    FastAPILoggingMiddleware,
    FlaskLoggingMiddleware,
    add_structured_logging,
    create_fastapi_logger_config,
    create_flask_logger_config,
)
from .logger import (
    get_filter_metrics,
    get_logger,
    log_with_context,
    reset_filter_metrics,
)
from .network_handlers import (
    HTTPConfig,
    HTTPHandler,
    NetworkHandlerConfig,
    SocketConfig,
    SocketHandler,
    SyslogConfig,
    SyslogHandler,
)
from .serializers import (
    EnhancedJSONEncoder,
    LazyDict,
    LazySerializable,
    LazySerializationManager,
    SerializationConfig,
    SmartConverter,
    TypeDetector,
    TypeRegistry,
    create_lazy_serializable,
    enhanced_json_dumps,
    get_lazy_serialization_stats,
    register_custom_serializer,
    reset_lazy_serialization_stats,
    serialize_for_logging,
    serialize_for_logging_lazy_aware,
    should_use_lazy_serialization,
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
    # Network handlers
    "NetworkHandlerConfig",
    "SyslogConfig", 
    "SyslogHandler",
    "HTTPConfig",
    "HTTPHandler", 
    "SocketConfig",
    "SocketHandler",
    # Enhanced serialization
    "SerializationConfig",
    "TypeRegistry",
    "TypeDetector",
    "SmartConverter",
    "EnhancedJSONEncoder",
    "enhanced_json_dumps",
    "register_custom_serializer",
    "serialize_for_logging",
    "serialize_for_logging_lazy_aware",
    # Lazy serialization
    "LazySerializable",
    "LazyDict",
    "LazySerializationManager",
    "create_lazy_serializable",
    "should_use_lazy_serialization",
    "get_lazy_serialization_stats",
    "reset_lazy_serialization_stats",
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
