"""
Django integration for structured logging

This module provides Django middleware and utilities for automatic structured
logging of HTTP requests, database queries, and Django-specific events.

Features:
- Automatic request/response logging middleware
- Database query logging
- User and session context injection
- Django signal integration
- Error and exception logging
- Performance metrics collection
- CSRF and security event logging
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union

# Django imports with availability checking
try:
    from django.conf import settings
    from django.contrib.auth.models import AnonymousUser
    from django.core.exceptions import ImproperlyConfigured
    from django.db import connection
    from django.http import HttpRequest, HttpResponse
    from django.utils.deprecation import MiddlewareMixin
    import django

    # Django signals
    from django.core.signals import (
        request_started,
        request_finished,
        got_request_exception,
    )
    from django.db.backends.signals import connection_created
    from django.contrib.auth.signals import (
        user_logged_in,
        user_logged_out,
        user_login_failed,
    )

    HAS_DJANGO = True
    DJANGO_VERSION = django.VERSION
except ImportError:
    settings = None
    AnonymousUser = None
    ImproperlyConfigured = Exception
    HttpRequest = None
    HttpResponse = None
    MiddlewareMixin = object
    connection = None
    HAS_DJANGO = False
    DJANGO_VERSION = None

from ...context import request_context
from ...logger import get_logger


@dataclass
class DjangoLoggingConfig:
    """Configuration for Django logging integration"""

    # Request/Response logging
    log_request_body: bool = False  # Log request body (be careful with sensitive data)
    log_response_body: bool = False  # Log response body
    log_request_headers: bool = True  # Log request headers
    log_response_headers: bool = False  # Log response headers
    request_body_max_size: int = 10000  # Max size for request body logging
    response_body_max_size: int = 10000  # Max size for response body logging

    # Header filtering
    sensitive_headers: Set[str] = field(
        default_factory=lambda: {
            "authorization",
            "cookie",
            "x-api-key",
            "x-auth-token",
            "x-csrf-token",
            "x-forwarded-for",
            "x-real-ip",
        }
    )
    header_prefix: str = "header_"  # Prefix for header fields in logs

    # User context
    include_user_info: bool = True  # Include user information in logs
    user_fields: List[str] = field(
        default_factory=lambda: ["id", "username", "email", "is_staff", "is_superuser"]
    )

    # Database logging
    log_database_queries: bool = True  # Log database queries
    slow_query_threshold: float = 1.0  # Threshold for slow query warnings (seconds)
    log_query_params: bool = (
        False  # Include query parameters (careful with sensitive data)
    )

    # Performance
    include_response_time: bool = True  # Include request processing time
    include_db_query_count: bool = True  # Include database query count
    include_db_query_time: bool = True  # Include total database query time

    # Django signals
    log_auth_events: bool = True  # Log authentication events
    log_request_signals: bool = False  # Log request start/finish signals
    log_db_connections: bool = False  # Log database connection events

    # Error handling
    log_exceptions: bool = True  # Log unhandled exceptions
    include_traceback: bool = True  # Include exception traceback

    # Filtering
    excluded_paths: Set[str] = field(
        default_factory=lambda: {
            "/health",
            "/healthz",
            "/ping",
            "/favicon.ico",
            "/static/",
            "/media/",
            "/_debug/",
        }
    )
    excluded_methods: Set[str] = field(default_factory=set)
    only_errors: bool = False  # Only log requests that result in errors

    # Field naming
    use_django_field_names: bool = (
        False  # Use Django-style field names (e.g., 'user_id' vs 'user.id')
    )
    field_name_mapping: Dict[str, str] = field(default_factory=dict)


class DjangoStructuredLoggingMiddleware(MiddlewareMixin):
    """
    Django middleware for automatic structured logging

    This middleware automatically logs HTTP requests and responses with
    structured context including user information, headers, performance
    metrics, and database query statistics.
    """

    def __init__(self, get_response=None):
        if not HAS_DJANGO:
            raise ImportError(
                "django is required for Django integration. "
                "Install with: pip install django"
            )

        super().__init__(get_response)

        # Load configuration from Django settings or use defaults
        self.config = self._load_config()

        # Create logger
        logger_name = getattr(settings, "STRUCTURED_LOGGING_LOGGER_NAME", "django")
        self.logger = get_logger(logger_name)

        # Setup signal handlers if configured
        self._setup_signal_handlers()

        # Track request-specific data
        self._request_data = {}

    def _load_config(self) -> DjangoLoggingConfig:
        """Load configuration from Django settings"""
        config_dict = getattr(settings, "STRUCTURED_LOGGING_CONFIG", {})

        # Convert Django settings to DjangoLoggingConfig
        if isinstance(config_dict, dict):
            return DjangoLoggingConfig(**config_dict)
        elif isinstance(config_dict, DjangoLoggingConfig):
            return config_dict
        else:
            return DjangoLoggingConfig()

    def _setup_signal_handlers(self) -> None:
        """Setup Django signal handlers based on configuration"""
        if self.config.log_auth_events:
            user_logged_in.connect(self._handle_user_login)
            user_logged_out.connect(self._handle_user_logout)
            user_login_failed.connect(self._handle_login_failed)

        if self.config.log_request_signals:
            request_started.connect(self._handle_request_started)
            request_finished.connect(self._handle_request_finished)

        if self.config.log_exceptions:
            got_request_exception.connect(self._handle_request_exception)

        if self.config.log_db_connections:
            connection_created.connect(self._handle_db_connection)

    def process_request(self, request: HttpRequest) -> None:
        """Process incoming request and setup logging context"""
        # Check if path should be excluded
        if self._should_exclude_request(request):
            request._skip_logging = True
            return

        # Store request start time
        request._start_time = time.time()

        # Reset query tracking
        if self.config.log_database_queries:
            self._reset_query_tracking(request)

        # Extract request information
        request_data = self._extract_request_data(request)

        # Setup logging context
        context_data = {
            "request_id": self._generate_request_id(),
            "method": request.method,
            "path": request.path,
            "user_id": request_data.get("user_id"),
            "session_id": request_data.get("session_id"),
        }

        # Store context for later use
        request._logging_context = request_context(**context_data)
        request._logging_context.__enter__()

        # Log request if not using signals
        if not self.config.log_request_signals:
            self.logger.info(
                f"Request started: {request.method} {request.path}", extra=request_data
            )

    def process_response(
        self, request: HttpRequest, response: HttpResponse
    ) -> HttpResponse:
        """Process response and log request completion"""
        # Skip if request was excluded
        if getattr(request, "_skip_logging", False):
            return response

        try:
            # Calculate response time
            response_time = time.time() - getattr(request, "_start_time", time.time())

            # Extract response data
            response_data = self._extract_response_data(
                request, response, response_time
            )

            # Determine log level based on status code
            if response.status_code >= 500:
                log_level = "error"
            elif response.status_code >= 400:
                log_level = "warning"
            else:
                log_level = "info"

            # Skip if only logging errors and this isn't an error
            if self.config.only_errors and response.status_code < 400:
                return response

            # Log the request/response
            getattr(self.logger, log_level)(
                f"Request completed: {request.method} {request.path} - {response.status_code}",
                extra=response_data,
            )

        finally:
            # Clean up context
            if hasattr(request, "_logging_context"):
                request._logging_context.__exit__(None, None, None)

        return response

    def process_exception(self, request: HttpRequest, exception: Exception) -> None:
        """Process unhandled exceptions"""
        if getattr(request, "_skip_logging", False):
            return

        if self.config.log_exceptions:
            response_time = time.time() - getattr(request, "_start_time", time.time())

            exception_data = {
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "response_time_seconds": response_time,
                "path": request.path,
                "method": request.method,
            }

            # Add query statistics
            if self.config.log_database_queries:
                exception_data.update(self._get_query_statistics(request))

            self.logger.error(
                f"Unhandled exception in {request.method} {request.path}: {type(exception).__name__}",
                extra=exception_data,
                exc_info=self.config.include_traceback,
            )

    def _should_exclude_request(self, request: HttpRequest) -> bool:
        """Check if request should be excluded from logging"""
        # Check excluded paths
        for excluded_path in self.config.excluded_paths:
            if request.path.startswith(excluded_path):
                return True

        # Check excluded methods
        if request.method in self.config.excluded_methods:
            return True

        return False

    def _extract_request_data(self, request: HttpRequest) -> Dict[str, Any]:
        """Extract data from request for logging"""
        data = {
            "method": request.method,
            "path": request.path,
            "query_string": request.META.get("QUERY_STRING", ""),
            "remote_addr": self._get_client_ip(request),
            "user_agent": request.META.get("HTTP_USER_AGENT", ""),
            "referer": request.META.get("HTTP_REFERER", ""),
        }

        # Add headers if configured
        if self.config.log_request_headers:
            headers = self._extract_headers(request)
            for key, value in headers.items():
                data[f"{self.config.header_prefix}{key}"] = value

        # Add user information
        if self.config.include_user_info and hasattr(request, "user"):
            user_data = self._extract_user_data(request.user)
            data.update(user_data)

        # Add session ID
        if hasattr(request, "session") and request.session:
            data["session_id"] = request.session.session_key

        # Add request body if configured
        if self.config.log_request_body and request.body:
            data["request_body"] = self._extract_body(
                request.body, request.content_type, self.config.request_body_max_size
            )

        return data

    def _extract_response_data(
        self, request: HttpRequest, response: HttpResponse, response_time: float
    ) -> Dict[str, Any]:
        """Extract data from response for logging"""
        data = {
            "status_code": response.status_code,
            "response_time_seconds": response_time,
        }

        # Add response headers if configured
        if self.config.log_response_headers:
            for key, value in response.items():
                clean_key = key.lower().replace("-", "_")
                if clean_key not in self.config.sensitive_headers:
                    data[f"response_{self.config.header_prefix}{clean_key}"] = value

        # Add response body if configured
        if self.config.log_response_body and hasattr(response, "content"):
            data["response_body"] = self._extract_body(
                response.content,
                response.get("Content-Type", ""),
                self.config.response_body_max_size,
            )

        # Add database query statistics
        if self.config.log_database_queries:
            data.update(self._get_query_statistics(request))

        return data

    def _extract_headers(self, request: HttpRequest) -> Dict[str, str]:
        """Extract and clean request headers"""
        headers = {}

        for key, value in request.META.items():
            if key.startswith("HTTP_"):
                # Convert HTTP_HEADER_NAME to header_name
                header_name = key[5:].lower().replace("_", "-")

                # Skip sensitive headers
                if header_name in self.config.sensitive_headers:
                    headers[header_name.replace("-", "_")] = "[REDACTED]"
                else:
                    headers[header_name.replace("-", "_")] = value

        return headers

    def _extract_user_data(self, user) -> Dict[str, Any]:
        """Extract user information for logging"""
        if not user or isinstance(user, AnonymousUser):
            return {"user_id": None, "user_anonymous": True}

        user_data = {"user_anonymous": False}

        for field in self.config.user_fields:
            if hasattr(user, field):
                value = getattr(user, field)
                # Convert field name if using Django style
                if self.config.use_django_field_names:
                    key = f"user_{field}"
                else:
                    key = f"user.{field}"

                # Apply field name mapping
                key = self.config.field_name_mapping.get(key, key)
                user_data[key] = value

        return user_data

    def _extract_body(
        self, body: bytes, content_type: str, max_size: int
    ) -> Union[str, Dict[str, Any]]:
        """Extract and process request/response body"""
        if not body:
            return None

        # Limit body size
        if len(body) > max_size:
            return f"[Body too large: {len(body)} bytes]"

        # Try to decode as JSON
        if "application/json" in content_type:
            try:
                return json.loads(body.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        # Try to decode as string
        try:
            return body.decode("utf-8")
        except UnicodeDecodeError:
            return f"[Binary data: {len(body)} bytes]"

    def _get_client_ip(self, request: HttpRequest) -> str:
        """Get client IP address from request"""
        # Check X-Forwarded-For header
        x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
        if x_forwarded_for:
            ip = x_forwarded_for.split(",")[0].strip()
        else:
            # Fallback to REMOTE_ADDR
            ip = request.META.get("REMOTE_ADDR", "")

        return ip

    def _reset_query_tracking(self, request: HttpRequest) -> None:
        """Reset database query tracking for request"""
        request._queries_count = 0
        request._queries_time = 0.0
        request._slow_queries = []

        # Store original execute method
        if not hasattr(connection, "_original_execute"):
            connection._original_execute = connection.execute

        # Wrap execute method to track queries
        def tracked_execute(sql, params=None):
            start_time = time.time()
            try:
                return connection._original_execute(sql, params)
            finally:
                query_time = time.time() - start_time
                request._queries_count += 1
                request._queries_time += query_time

                # Track slow queries
                if query_time > self.config.slow_query_threshold:
                    query_info = {
                        "sql": sql if self.config.log_query_params else "[Query]",
                        "time": query_time,
                    }
                    if self.config.log_query_params and params:
                        query_info["params"] = params
                    request._slow_queries.append(query_info)

        connection.execute = tracked_execute

    def _get_query_statistics(self, request: HttpRequest) -> Dict[str, Any]:
        """Get database query statistics for request"""
        stats = {}

        if self.config.include_db_query_count:
            stats["db_query_count"] = getattr(request, "_queries_count", 0)

        if self.config.include_db_query_time:
            stats["db_query_time_seconds"] = getattr(request, "_queries_time", 0.0)

        # Include slow queries if any
        slow_queries = getattr(request, "_slow_queries", [])
        if slow_queries:
            stats["slow_queries"] = slow_queries
            stats["slow_query_count"] = len(slow_queries)

        return stats

    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        import uuid

        return str(uuid.uuid4())

    # Signal handlers

    def _handle_user_login(self, sender, request, user, **kwargs):
        """Handle user login signal"""
        self.logger.info(
            f"User logged in: {user.username}",
            extra={
                "event": "user_login",
                "user_id": user.id,
                "username": user.username,
                "ip_address": self._get_client_ip(request),
            },
        )

    def _handle_user_logout(self, sender, request, user, **kwargs):
        """Handle user logout signal"""
        self.logger.info(
            f"User logged out: {user.username}",
            extra={
                "event": "user_logout",
                "user_id": user.id,
                "username": user.username,
                "ip_address": self._get_client_ip(request),
            },
        )

    def _handle_login_failed(self, sender, credentials, request, **kwargs):
        """Handle failed login attempt"""
        username = credentials.get("username", "unknown")
        self.logger.warning(
            f"Failed login attempt for user: {username}",
            extra={
                "event": "login_failed",
                "username": username,
                "ip_address": self._get_client_ip(request) if request else None,
            },
        )

    def _handle_request_started(self, sender, **kwargs):
        """Handle request started signal"""
        self.logger.debug(
            "Request started signal received", extra={"event": "request_started"}
        )

    def _handle_request_finished(self, sender, **kwargs):
        """Handle request finished signal"""
        self.logger.debug(
            "Request finished signal received", extra={"event": "request_finished"}
        )

    def _handle_request_exception(self, sender, request, **kwargs):
        """Handle request exception signal"""
        # Exception logging is handled in process_exception
        pass

    def _handle_db_connection(self, sender, connection, **kwargs):
        """Handle database connection created signal"""
        self.logger.info(
            "Database connection created",
            extra={
                "event": "db_connection_created",
                "vendor": connection.vendor,
                "alias": connection.alias,
            },
        )


# Convenience functions


def setup_django_logging(
    logger_name: Optional[str] = None, config: Optional[DjangoLoggingConfig] = None
) -> None:
    """
    Setup Django logging configuration in settings

    This should be called in Django settings.py:

    ```python
    from structured_logging.integrations.django import setup_django_logging

    setup_django_logging(
        logger_name='my_app',
        config=DjangoLoggingConfig(
            log_database_queries=True,
            include_user_info=True
        )
    )
    ```
    """
    if not HAS_DJANGO:
        raise ImportError("Django is required for Django integration")

    # Set logger name in settings
    if logger_name:
        settings.STRUCTURED_LOGGING_LOGGER_NAME = logger_name

    # Set configuration in settings
    if config:
        settings.STRUCTURED_LOGGING_CONFIG = config


def get_django_logger(name: Optional[str] = None) -> Any:
    """
    Get a structured logger configured for Django

    Args:
        name: Logger name (defaults to Django settings or 'django')

    Returns:
        Configured structured logger
    """
    if not name:
        name = getattr(settings, "STRUCTURED_LOGGING_LOGGER_NAME", "django")

    return get_logger(name)


# Management command support


class DjangoLoggingCommand:
    """
    Base class for Django management commands with structured logging

    Usage:
    ```python
    from django.core.management.base import BaseCommand
    from structured_logging.integrations.django import DjangoLoggingCommand

    class Command(DjangoLoggingCommand, BaseCommand):
        def handle(self, *args, **options):
            self.log_info("Command started")
            # Command implementation
            self.log_success("Command completed")
    ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_django_logger(f"management.{self.__class__.__module__}")

    def log_debug(self, message: str, **extra):
        """Log debug message"""
        self.logger.debug(message, extra=extra)

    def log_info(self, message: str, **extra):
        """Log info message"""
        self.logger.info(message, extra=extra)
        if hasattr(self, "stdout"):
            self.stdout.write(message)

    def log_warning(self, message: str, **extra):
        """Log warning message"""
        self.logger.warning(message, extra=extra)
        if hasattr(self, "stdout"):
            self.stdout.write(self.style.WARNING(message))

    def log_error(self, message: str, **extra):
        """Log error message"""
        self.logger.error(message, extra=extra)
        if hasattr(self, "stderr"):
            self.stderr.write(self.style.ERROR(message))

    def log_success(self, message: str, **extra):
        """Log success message"""
        self.logger.info(message, extra={"status": "success", **extra})
        if hasattr(self, "stdout"):
            self.stdout.write(self.style.SUCCESS(message))
