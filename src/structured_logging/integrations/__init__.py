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

# Framework integrations with conditional imports
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

# OpenTelemetry integration
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
    
    __all__.extend([
        "HAS_OPENTELEMETRY",
        "OpenTelemetryConfig",
        "OpenTelemetryHandler",
        "OpenTelemetryLogger",
        "LoggingSpan",
        "create_otel_logger",
        "add_otel_handler_to_logger",
        "configure_otel_logging",
        "logged_span",
    ])
except ImportError:
    HAS_OPENTELEMETRY = False

# Django integration
try:
    from .django import (
        HAS_DJANGO,
        DjangoLoggingConfig,
        DjangoStructuredLoggingMiddleware,
        DjangoLoggingCommand,
        setup_django_logging,
        get_django_logger,
    )
    
    __all__.extend([
        "HAS_DJANGO",
        "DjangoLoggingConfig",
        "DjangoStructuredLoggingMiddleware",
        "DjangoLoggingCommand",
        "setup_django_logging",
        "get_django_logger",
    ])
except ImportError:
    HAS_DJANGO = False

# aiohttp integration
try:
    from .aiohttp import (
        HAS_AIOHTTP,
        AioHTTPLoggingConfig,
        WebSocketLogger,
        LoggingClientSession,
        aiohttp_structured_logging_middleware,
        setup_aiohttp_logging,
        log_route,
    )
    
    __all__.extend([
        "HAS_AIOHTTP",
        "AioHTTPLoggingConfig",
        "WebSocketLogger",
        "LoggingClientSession",
        "aiohttp_structured_logging_middleware",
        "setup_aiohttp_logging",
        "log_route",
    ])
except ImportError:
    HAS_AIOHTTP = False

# Celery integration
try:
    from .celery import (
        HAS_CELERY,
        CeleryLoggingConfig,
        StructuredLoggingTask,
        setup_celery_logging,
        get_task_logger,
        log_task,
        log_task_chain,
        log_task_group,
    )
    
    __all__.extend([
        "HAS_CELERY",
        "CeleryLoggingConfig",
        "StructuredLoggingTask",
        "setup_celery_logging",
        "get_task_logger",
        "log_task",
        "log_task_chain",
        "log_task_group",
    ])
except ImportError:
    HAS_CELERY = False

# SQLAlchemy integration
try:
    from .sqlalchemy import (
        HAS_SQLALCHEMY,
        SQLAlchemyLoggingConfig,
        SQLAlchemyQueryLogger,
        SQLAlchemyConnectionLogger,
        SQLAlchemyTransactionLogger,
        SQLAlchemyORMLogger,
        setup_sqlalchemy_logging,
        log_query,
        get_query_logger,
        DatabaseOperation,
    )
    
    __all__.extend([
        "HAS_SQLALCHEMY",
        "SQLAlchemyLoggingConfig",
        "SQLAlchemyQueryLogger",
        "SQLAlchemyConnectionLogger",
        "SQLAlchemyTransactionLogger",
        "SQLAlchemyORMLogger",
        "setup_sqlalchemy_logging",
        "log_query",
        "get_query_logger",
        "DatabaseOperation",
    ])
except ImportError:
    HAS_SQLALCHEMY = False