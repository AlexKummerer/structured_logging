"""
OpenTelemetry integration for structured logging

This module provides integration between structured logging and OpenTelemetry (OTel),
enabling automatic trace correlation, span context injection, and structured log
attributes mapping.

Features:
- Automatic trace and span ID injection into logs
- Context propagation from OTel spans
- Resource attribute mapping
- OTel LogRecord compatibility
- Structured attribute conversion
- Custom attribute filtering and transformation
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# OpenTelemetry imports with availability checking
try:
    from opentelemetry import trace
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.util.types import Attributes

    HAS_OPENTELEMETRY = True
except ImportError:
    trace = None
    LoggingInstrumentor = None
    Resource = None
    ResourceAttributes = None
    Status = None
    StatusCode = None
    Attributes = None
    HAS_OPENTELEMETRY = False

from ..logger import get_logger


@dataclass
class OpenTelemetryConfig:
    """Configuration for OpenTelemetry integration"""

    # Trace correlation
    enable_trace_correlation: bool = True  # Inject trace/span IDs into logs
    trace_id_field: str = "trace_id"  # Field name for trace ID
    span_id_field: str = "span_id"  # Field name for span ID
    trace_flags_field: str = "trace_flags"  # Field name for trace flags

    # Resource attributes
    include_resource_attributes: bool = True  # Include OTel resource attributes
    resource_attribute_prefix: str = "resource."  # Prefix for resource attributes
    allowed_resource_attributes: Optional[List[str]] = None  # Whitelist specific attrs

    # Span attributes
    include_span_attributes: bool = True  # Include current span attributes
    span_attribute_prefix: str = "span."  # Prefix for span attributes
    max_span_attributes: int = 20  # Limit number of span attributes

    # Log attributes
    map_log_attributes: bool = True  # Map log record attrs to OTel attributes
    attribute_mapping: Dict[str, str] = field(default_factory=dict)  # Custom mappings
    exclude_attributes: List[str] = field(
        default_factory=lambda: ["args", "created", "exc_info", "exc_text"]
    )

    # Performance
    enable_instrumentation: bool = True  # Use OTel LoggingInstrumentor
    set_span_status_on_error: bool = True  # Set span status for ERROR/CRITICAL logs
    max_attribute_length: int = 1000  # Truncate long attribute values

    # Filtering
    only_when_tracing: bool = False  # Only add OTel data when actively tracing
    minimum_log_level: str = "INFO"  # Minimum level for OTel integration


class OpenTelemetryHandler(logging.Handler):
    """
    Logging handler that integrates with OpenTelemetry

    This handler automatically:
    - Injects trace and span IDs into log records
    - Adds resource and span attributes as structured fields
    - Maps log records to OTel semantic conventions
    - Correlates logs with active traces and spans
    - Sets span status based on log levels
    """

    def __init__(self, config: OpenTelemetryConfig):
        if not HAS_OPENTELEMETRY:
            raise ImportError(
                "opentelemetry is required for OpenTelemetry integration. "
                "Install with: pip install structured-logging[otel]"
            )

        super().__init__()
        self.config = config

        # Cache resource attributes for performance
        self._resource_attributes = {}
        if self.config.include_resource_attributes:
            self._cache_resource_attributes()

        # Set up instrumentation if enabled
        if self.config.enable_instrumentation:
            self._setup_instrumentation()

    def _cache_resource_attributes(self) -> None:
        """Cache resource attributes for efficient access"""
        try:
            # Get the global tracer provider's resource
            tracer_provider = trace.get_tracer_provider()
            if hasattr(tracer_provider, "resource") and tracer_provider.resource:
                resource = tracer_provider.resource
                prefix = self.config.resource_attribute_prefix

                # Filter and map resource attributes
                for key, value in resource.attributes.items():
                    # Apply whitelist if configured
                    if self.config.allowed_resource_attributes:
                        if key not in self.config.allowed_resource_attributes:
                            continue

                    # Add with prefix
                    field_name = f"{prefix}{key}"
                    self._resource_attributes[field_name] = self._serialize_value(value)

        except Exception:
            # Silently handle any resource access issues
            pass

    def _setup_instrumentation(self) -> None:
        """Set up OpenTelemetry logging instrumentation"""
        try:
            # Configure the logging instrumentation
            LoggingInstrumentor().instrument(
                set_logging_format=False,  # Don't change log format
                log_hook=self._log_hook,  # Custom log processing
            )
        except Exception:
            # Instrumentation setup failed, continue without it
            pass

    def _log_hook(self, span, record) -> None:
        """Hook called by OTel instrumentation for each log record"""
        if span and span.is_recording():
            # Set span status based on log level
            if self.config.set_span_status_on_error and record.levelno >= logging.ERROR:
                span.set_status(Status(StatusCode.ERROR, record.getMessage()))

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record with OpenTelemetry correlation

        Enriches the log record with:
        - Trace and span IDs
        - Resource attributes
        - Span attributes
        - Mapped log attributes
        """
        try:
            # Check minimum log level
            min_level = getattr(logging, self.config.minimum_log_level.upper())
            if record.levelno < min_level:
                return

            # Get current span context
            current_span = trace.get_current_span()
            span_context = current_span.get_span_context() if current_span else None

            # Skip if only_when_tracing is True and no active span
            if self.config.only_when_tracing and (
                not span_context or not span_context.is_valid
            ):
                return

            # Add trace correlation
            if self.config.enable_trace_correlation and span_context:
                self._add_trace_correlation(record, span_context)

            # Add resource attributes
            if self.config.include_resource_attributes:
                self._add_resource_attributes(record)

            # Add span attributes
            if self.config.include_span_attributes and current_span:
                self._add_span_attributes(record, current_span)

            # Map log attributes
            if self.config.map_log_attributes:
                self._map_log_attributes(record)

        except Exception:
            # Don't let OTel integration break logging
            pass

    def _add_trace_correlation(self, record: logging.LogRecord, span_context) -> None:
        """Add trace and span IDs to log record"""
        # Format trace ID as hex string
        trace_id = format(span_context.trace_id, "032x")
        span_id = format(span_context.span_id, "016x")

        # Add to record
        setattr(record, self.config.trace_id_field, trace_id)
        setattr(record, self.config.span_id_field, span_id)
        setattr(record, self.config.trace_flags_field, span_context.trace_flags)

    def _add_resource_attributes(self, record: logging.LogRecord) -> None:
        """Add cached resource attributes to log record"""
        for key, value in self._resource_attributes.items():
            setattr(record, key, value)

    def _add_span_attributes(self, record: logging.LogRecord, span) -> None:
        """Add current span attributes to log record"""
        try:
            if not span.is_recording():
                return

            # Get span attributes (if accessible)
            attributes = getattr(span, "attributes", {})
            if not attributes:
                return

            prefix = self.config.span_attribute_prefix
            count = 0

            for key, value in attributes.items():
                if count >= self.config.max_span_attributes:
                    break

                field_name = f"{prefix}{key}"
                setattr(record, field_name, self._serialize_value(value))
                count += 1

        except Exception:
            # Span attribute access failed, continue
            pass

    def _map_log_attributes(self, record: logging.LogRecord) -> None:
        """Map log record attributes using configured mappings"""
        # Apply custom attribute mappings
        for old_name, new_name in self.config.attribute_mapping.items():
            if hasattr(record, old_name):
                value = getattr(record, old_name)
                setattr(record, new_name, value)

        # Clean up excluded attributes
        for attr_name in self.config.exclude_attributes:
            if hasattr(record, attr_name):
                delattr(record, attr_name)

    def _serialize_value(self, value: Any) -> Any:
        """Serialize value for logging with length limits"""
        if value is None:
            return None
        elif isinstance(value, (str, int, float, bool)):
            # Truncate long strings
            if isinstance(value, str) and len(value) > self.config.max_attribute_length:
                return value[: self.config.max_attribute_length] + "..."
            return value
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value[:10]]  # Limit list size
        elif isinstance(value, dict):
            # Limit dict size and apply recursively
            return {
                k: self._serialize_value(v)
                for k, v in list(value.items())[:10]
                if k not in self.config.exclude_attributes
            }
        else:
            # Convert to string and truncate
            str_value = str(value)
            if len(str_value) > self.config.max_attribute_length:
                return str_value[: self.config.max_attribute_length] + "..."
            return str_value


class OpenTelemetryLogger:
    """
    Structured logger with built-in OpenTelemetry integration

    This class provides a high-level interface for structured logging
    with automatic OpenTelemetry correlation and context management.
    """

    def __init__(
        self,
        name: str,
        otel_config: Optional[OpenTelemetryConfig] = None,
        logger_config: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.otel_config = otel_config or OpenTelemetryConfig()

        # Create underlying structured logger
        self.logger = get_logger(name, logger_config)

        # Add OpenTelemetry handler
        if HAS_OPENTELEMETRY:
            self.otel_handler = OpenTelemetryHandler(self.otel_config)
            self.logger.addHandler(self.otel_handler)

    def _log_with_span_context(
        self, level: str, message: str, **kwargs
    ) -> Optional[Any]:
        """Log with automatic span context injection"""
        # Get current span for context
        current_span = trace.get_current_span() if HAS_OPENTELEMETRY else None

        # Add span name and operation if available
        if current_span and current_span.is_recording():
            # Add span information to extra fields
            kwargs.setdefault("span_name", getattr(current_span, "name", "unknown"))
            kwargs.setdefault("span_kind", getattr(current_span, "kind", "unknown"))

        # Log using structured logger
        log_method = getattr(self.logger, level.lower())
        return log_method(message, extra=kwargs)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with OTel context"""
        self._log_with_span_context("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message with OTel context"""
        self._log_with_span_context("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with OTel context"""
        self._log_with_span_context("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message with OTel context"""
        self._log_with_span_context("ERROR", message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message with OTel context"""
        self._log_with_span_context("CRITICAL", message, **kwargs)

    def exception(self, message: str, **kwargs) -> None:
        """Log exception with OTel context"""
        kwargs["exc_info"] = True
        self._log_with_span_context("ERROR", message, **kwargs)


# Convenience functions for easy integration


def create_otel_logger(
    name: str,
    enable_trace_correlation: bool = True,
    include_resource_attributes: bool = True,
    include_span_attributes: bool = True,
    **config_kwargs,
) -> OpenTelemetryLogger:
    """
    Create a structured logger with OpenTelemetry integration

    Args:
        name: Logger name
        enable_trace_correlation: Include trace/span IDs in logs
        include_resource_attributes: Include OTel resource attributes
        include_span_attributes: Include current span attributes
        **config_kwargs: Additional OpenTelemetryConfig parameters

    Returns:
        Configured OpenTelemetryLogger instance
    """
    config = OpenTelemetryConfig(
        enable_trace_correlation=enable_trace_correlation,
        include_resource_attributes=include_resource_attributes,
        include_span_attributes=include_span_attributes,
        **config_kwargs,
    )

    return OpenTelemetryLogger(name, config)


def add_otel_handler_to_logger(
    logger: logging.Logger, config: Optional[OpenTelemetryConfig] = None
) -> OpenTelemetryHandler:
    """
    Add OpenTelemetry handler to existing logger

    Args:
        logger: Existing logger instance
        config: OpenTelemetry configuration

    Returns:
        The created OpenTelemetryHandler
    """
    if not HAS_OPENTELEMETRY:
        raise ImportError(
            "opentelemetry is required for OpenTelemetry integration. "
            "Install with: pip install structured-logging[otel]"
        )

    config = config or OpenTelemetryConfig()
    handler = OpenTelemetryHandler(config)
    logger.addHandler(handler)
    return handler


def configure_otel_logging(
    service_name: str,
    service_version: Optional[str] = None,
    environment: Optional[str] = None,
    **config_kwargs,
) -> OpenTelemetryConfig:
    """
    Configure OpenTelemetry logging with service information

    Args:
        service_name: Name of the service
        service_version: Version of the service
        environment: Deployment environment
        **config_kwargs: Additional configuration parameters

    Returns:
        Configured OpenTelemetryConfig
    """
    # Set up resource attribute mappings for common service attributes
    attribute_mapping = config_kwargs.setdefault("attribute_mapping", {})

    # Map service information if provided
    if service_name:
        attribute_mapping["service"] = service_name
    if service_version:
        attribute_mapping["version"] = service_version
    if environment:
        attribute_mapping["environment"] = environment

    return OpenTelemetryConfig(**config_kwargs)


# Context managers for span-aware logging


class LoggingSpan:
    """
    Context manager that creates a span and logs entry/exit

    Combines OpenTelemetry span creation with automatic logging
    of span lifecycle events.
    """

    def __init__(
        self,
        logger: OpenTelemetryLogger,
        span_name: str,
        tracer_name: Optional[str] = None,
        **span_kwargs,
    ):
        self.logger = logger
        self.span_name = span_name
        self.tracer_name = tracer_name or logger.name
        self.span_kwargs = span_kwargs
        self.span = None
        self.start_time = None

    def __enter__(self):
        if not HAS_OPENTELEMETRY:
            return self

        # Create and start span
        tracer = trace.get_tracer(self.tracer_name)
        self.span = tracer.start_span(self.span_name, **self.span_kwargs)
        self.start_time = time.time()

        # Log span start
        self.logger.info(
            f"Started span: {self.span_name}",
            span_name=self.span_name,
            span_operation="start",
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.span:
            return

        duration = time.time() - self.start_time if self.start_time else 0

        # Log span completion
        if exc_type:
            # Log error and set span status
            self.logger.error(
                f"Span failed: {self.span_name}",
                span_name=self.span_name,
                span_operation="error",
                span_duration_seconds=duration,
                error_type=exc_type.__name__ if exc_type else None,
                error_message=str(exc_val) if exc_val else None,
            )
            self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
        else:
            # Log successful completion
            self.logger.info(
                f"Completed span: {self.span_name}",
                span_name=self.span_name,
                span_operation="complete",
                span_duration_seconds=duration,
            )
            self.span.set_status(Status(StatusCode.OK))

        # End the span
        self.span.end()


def logged_span(
    logger: OpenTelemetryLogger, span_name: str, **span_kwargs
) -> LoggingSpan:
    """
    Create a logging span context manager

    Args:
        logger: OpenTelemetryLogger instance
        span_name: Name for the span
        **span_kwargs: Additional span creation arguments

    Returns:
        LoggingSpan context manager
    """
    return LoggingSpan(logger, span_name, **span_kwargs)

