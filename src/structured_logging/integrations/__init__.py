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

# OpenTelemetry integration with conditional import
try:
    from .opentelemetry import (
        HAS_OPENTELEMETRY,
        LoggingSpan,
        OpenTelemetryConfig,
        OpenTelemetryHandler,
        OpenTelemetryLogger,
        add_otel_handler_to_logger,
        configure_otel_logging,
        create_otel_logger,
        logged_span,
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
        "HAS_OPENTELEMETRY",
        "OpenTelemetryConfig",
        "OpenTelemetryHandler",
        "OpenTelemetryLogger",
        "LoggingSpan",
        "create_otel_logger",
        "add_otel_handler_to_logger",
        "configure_otel_logging",
        "logged_span",
    ]
except ImportError:
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