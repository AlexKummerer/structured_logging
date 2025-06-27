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


def _get_formatter_cache_key(config: LoggerConfig) -> str:
    """Generate cache key for formatter"""
    return f"{config.formatter_type}_{config.include_timestamp}_{config.include_request_id}_{config.include_user_context}"


def _get_or_create_formatter(config: LoggerConfig) -> logging.Formatter:
    """Get formatter from cache or create new one"""
    cache_key = _get_formatter_cache_key(config)
    
    if cache_key not in _formatter_cache:
        if config.formatter_type == "csv":
            formatter = CSVFormatter(config)
        elif config.formatter_type == "plain":
            formatter = PlainTextFormatter(config)
        else:  # default to json
            formatter = StructuredFormatter(config)
        _formatter_cache[cache_key] = formatter
    
    return _formatter_cache[cache_key]


def _add_console_handler(logger: logging.Logger, config: LoggerConfig, formatter: logging.Formatter) -> None:
    """Add console handler if required"""
    if "console" in config.output_type:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)


def _add_file_handler(logger: logging.Logger, config: LoggerConfig, formatter: logging.Formatter) -> None:
    """Add file handler if required"""
    if "file" in config.output_type and config.file_config:
        file_handler = RotatingFileHandler(config.file_config)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def _create_network_handler(network_config: Any) -> logging.Handler:
    """Create appropriate network handler based on config type"""
    if hasattr(network_config, "facility"):  # SyslogConfig
        return SyslogHandler(network_config)
    elif hasattr(network_config, "url"):  # HTTPConfig
        return HTTPHandler(network_config)
    elif hasattr(network_config, "protocol"):  # SocketConfig
        return SocketHandler(network_config)
    else:
        # Default to syslog
        syslog_config = SyslogConfig(host=network_config.host, port=network_config.port)
        return SyslogHandler(syslog_config)


def _add_network_handler(logger: logging.Logger, config: LoggerConfig, formatter: logging.Formatter) -> None:
    """Add network handler if required"""
    if "network" in config.output_type and config.network_config:
        network_handler = _create_network_handler(config.network_config)
        network_handler.setFormatter(formatter)
        logger.addHandler(network_handler)


def get_logger(name: str, config: Optional[LoggerConfig] = None) -> logging.Logger:
    """Create a structured logger with the given name"""
    logger = logging.getLogger(name)

    if not logger.handlers:
        config = config or get_default_config()
        logger.setLevel(getattr(logging, config.log_level.upper()))
        
        formatter = _get_or_create_formatter(config)
        
        _add_console_handler(logger, config, formatter)
        _add_file_handler(logger, config, formatter)
        _add_network_handler(logger, config, formatter)
        
        logger.propagate = True

    return logger


def _collect_request_context(config: LoggerConfig) -> Dict[str, Any]:
    """Collect request-related context"""
    context = {}
    
    if config.include_request_id:
        req_id = get_request_id()
        if req_id:
            context["request_id"] = req_id
            
    if config.include_user_context:
        user_ctx = get_user_context()
        if user_ctx:
            context.update({k: v for k, v in user_ctx.items() if v is not None})
            
    return context


def _collect_all_context(config: LoggerConfig, extra: Dict[str, Any]) -> Dict[str, Any]:
    """Collect all context including request, custom, and extra"""
    context = _collect_request_context(config)
    
    # Custom context
    custom_ctx = get_custom_context()
    if custom_ctx:
        context.update({k: v for k, v in custom_ctx.items() if v is not None})
    
    # Extra context
    if extra:
        context.update({k: v for k, v in extra.items() if v is not None})
        
    return context


def _apply_filters(logger: logging.Logger, level: str, message: str, config: LoggerConfig, context: Dict[str, Any]) -> bool:
    """Apply filtering if configured"""
    if not (config.filter_config and config.filter_config.enabled):
        return True
        
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
    return filter_result.should_log


def log_with_context(
    logger: logging.Logger,
    level: str,
    message: str,
    config: Optional[LoggerConfig] = None,
    **extra: Any,
) -> bool:
    """Log with automatic context injection and filtering - optimized version"""
    config = config or get_default_config()
    
    # Collect all context
    context = _collect_all_context(config, extra)
    
    # Apply filtering
    if not _apply_filters(logger, level, message, config, context):
        return False
    
    # Convert context to ctx_ prefixed format and log
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
