import logging
import sys
from typing import Any, Dict, Optional

from .config import LoggerConfig, get_default_config
from .context import get_custom_context, get_request_id, get_user_context
from .filtering import FilterEngine
from .formatter import CSVFormatter, PlainTextFormatter, StructuredFormatter
from .handlers import RotatingFileHandler
from .network import HTTPHandler, SocketHandler, SyslogConfig, SyslogHandler

# Performance optimization: Cache formatter instances
_formatter_cache: Dict[str, logging.Formatter] = {}

# Filter engine cache
_filter_engines: Dict[int, FilterEngine] = {}


def get_logger(name: str, config: Optional[LoggerConfig] = None) -> logging.Logger:
    """Create a structured logger with the given name"""
    logger = logging.getLogger(name)

    if not logger.handlers:
        config = config or get_default_config()
        logger.setLevel(getattr(logging, config.log_level.upper()))

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

        # Add console handler if required
        if "console" in config.output_type:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # Add file handler if required
        if "file" in config.output_type and config.file_config:
            file_handler = RotatingFileHandler(config.file_config)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Add network handler if required
        if "network" in config.output_type and config.network_config:
            # Determine handler type based on config
            if hasattr(config.network_config, "facility"):  # SyslogConfig
                network_handler = SyslogHandler(config.network_config)
            elif hasattr(config.network_config, "url"):  # HTTPConfig
                network_handler = HTTPHandler(config.network_config)
            elif hasattr(config.network_config, "protocol"):  # SocketConfig
                network_handler = SocketHandler(config.network_config)
            else:
                # Default to syslog
                syslog_config = SyslogConfig(
                    host=config.network_config.host, port=config.network_config.port
                )
                network_handler = SyslogHandler(syslog_config)

            network_handler.setFormatter(formatter)
            logger.addHandler(network_handler)

        logger.propagate = True

    return logger


def log_with_context(
    logger: logging.Logger,
    level: str,
    message: str,
    config: Optional[LoggerConfig] = None,
    **extra: Any,
) -> bool:
    """Log with automatic context injection and filtering - optimized version"""
    config = config or get_default_config()
    context: Dict[str, Any] = {}

    # Optimized: Batch context retrieval to minimize context variable lookups
    if config.include_request_id or config.include_user_context:
        # Single context variable access
        req_id = get_request_id() if config.include_request_id else None
        user_ctx = get_user_context() if config.include_user_context else {}

        if req_id:
            context["request_id"] = req_id

        if user_ctx:
            # Optimized: Use dict comprehension for better performance
            context.update({k: v for k, v in user_ctx.items() if v is not None})

    # Custom context - only access if likely to have data
    custom_ctx = get_custom_context()
    if custom_ctx:
        context.update({k: v for k, v in custom_ctx.items() if v is not None})

    # Extra context - optimized dict comprehension
    if extra:
        context.update({k: v for k, v in extra.items() if v is not None})

    # Apply filtering if configured
    if config.filter_config and config.filter_config.enabled:
        # Get or create filter engine
        filter_id = id(config.filter_config)
        if filter_id not in _filter_engines:
            _filter_engines[filter_id] = FilterEngine(config.filter_config)

        filter_engine = _filter_engines[filter_id]

        # Create LogRecord for filtering
        record = logging.LogRecord(
            name=logger.name,
            level=getattr(logging, level.upper()),
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None,
        )

        # Apply filters
        filter_result = filter_engine.should_log(record, context)
        if not filter_result.should_log:
            return False  # Log was filtered out

    # Convert context back to ctx_ prefixed format for compatibility
    ctx_context = {f"ctx_{k}": v for k, v in context.items()}

    getattr(logger, level)(message, extra=ctx_context)
    return True


def get_filter_metrics(
    config: Optional[LoggerConfig] = None,
) -> Optional[Dict[str, Any]]:
    """Get filtering metrics for the current configuration"""
    config = config or get_default_config()

    if not config.filter_config or not config.filter_config.enabled:
        return None

    filter_id = id(config.filter_config)
    if filter_id in _filter_engines:
        return _filter_engines[filter_id].get_metrics()

    return None


def reset_filter_metrics(config: Optional[LoggerConfig] = None) -> None:
    """Reset filtering metrics for the current configuration"""
    config = config or get_default_config()

    if not config.filter_config or not config.filter_config.enabled:
        return

    filter_id = id(config.filter_config)
    if filter_id in _filter_engines:
        _filter_engines[filter_id].reset_metrics()
