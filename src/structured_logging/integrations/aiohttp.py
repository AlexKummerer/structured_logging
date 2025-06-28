"""
aiohttp integration for structured logging

This module provides aiohttp middleware and utilities for automatic structured
logging of HTTP requests, responses, and WebSocket connections in async applications.

Features:
- Async request/response logging middleware
- WebSocket connection logging
- Client request logging
- Error and exception handling
- Performance metrics collection
- Distributed tracing support
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Union

# aiohttp imports with availability checking
try:
    from aiohttp import web
    from aiohttp.web import (
        Application,
        Request,
        Response,
        StreamResponse,
        middleware,
    )
    from aiohttp.web_exceptions import HTTPException
    from aiohttp.hdrs import METH_ALL
    import aiohttp

    HAS_AIOHTTP = True
    AIOHTTP_VERSION = aiohttp.__version__
except ImportError:
    web = None
    Application = None
    Request = None
    Response = None
    StreamResponse = None
    middleware = lambda x: x
    HTTPException = Exception
    METH_ALL = []
    HAS_AIOHTTP = False
    AIOHTTP_VERSION = None

from ...async_context import async_request_context
from ...async_logger import get_async_logger


@dataclass
class AioHTTPLoggingConfig:
    """Configuration for aiohttp logging integration"""

    # Request/Response logging
    log_request_body: bool = False  # Log request body
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
            "set-cookie",
        }
    )
    header_prefix: str = "header_"  # Prefix for header fields in logs

    # WebSocket logging
    log_websocket_frames: bool = False  # Log WebSocket frames
    websocket_frame_max_size: int = 1000  # Max size for frame logging

    # Performance
    include_response_time: bool = True  # Include request processing time
    include_request_size: bool = True  # Include request content length
    include_response_size: bool = True  # Include response content length

    # Error handling
    log_exceptions: bool = True  # Log unhandled exceptions
    include_traceback: bool = True  # Include exception traceback
    log_http_exceptions: bool = True  # Log aiohttp HTTP exceptions

    # Filtering
    excluded_paths: Set[str] = field(
        default_factory=lambda: {
            "/health",
            "/healthz",
            "/ping",
            "/metrics",
            "/favicon.ico",
            "/robots.txt",
        }
    )
    excluded_methods: Set[str] = field(default_factory=set)
    excluded_status_codes: Set[int] = field(default_factory=set)
    only_errors: bool = False  # Only log requests that result in errors

    # Distributed tracing
    trace_id_header: str = "x-trace-id"  # Header for trace ID
    generate_trace_id: bool = True  # Generate trace ID if not present

    # Client logging
    log_client_requests: bool = True  # Log outgoing client requests
    client_request_timeout: float = 30.0  # Default client timeout


@middleware
async def aiohttp_structured_logging_middleware(
    request: Request, handler: Callable[[Request], Awaitable[StreamResponse]]
) -> StreamResponse:
    """
    aiohttp middleware for automatic structured logging

    This middleware automatically logs HTTP requests and responses with
    structured context including headers, performance metrics, and errors.
    """
    if not HAS_AIOHTTP:
        raise ImportError(
            "aiohttp is required for aiohttp integration. "
            "Install with: pip install aiohttp"
        )

    # Get configuration from app
    config = request.app.get("structured_logging_config", AioHTTPLoggingConfig())
    logger = request.app.get("structured_logger") or get_async_logger("aiohttp")

    # Check if request should be excluded
    if _should_exclude_request(request, config):
        return await handler(request)

    # Generate or extract trace ID
    trace_id = request.headers.get(config.trace_id_header)
    if not trace_id and config.generate_trace_id:
        import uuid

        trace_id = str(uuid.uuid4())

    # Start timing
    start_time = time.time()

    # Extract request data
    request_data = await _extract_request_data(request, config)

    # Setup logging context
    context_data = {
        "trace_id": trace_id,
        "method": request.method,
        "path": request.path,
        "remote": request.remote,
    }

    # Log request start
    async with async_request_context(**context_data):
        await logger.info(
            f"Request started: {request.method} {request.path}", extra=request_data
        )

        response = None
        exception = None

        try:
            # Call handler
            response = await handler(request)
            return response

        except HTTPException as http_exc:
            # Handle aiohttp HTTP exceptions
            exception = http_exc
            if config.log_http_exceptions:
                await logger.warning(
                    f"HTTP exception: {http_exc.status} - {http_exc.reason}",
                    extra={
                        "status_code": http_exc.status,
                        "reason": http_exc.reason,
                        "exception_type": type(http_exc).__name__,
                    },
                )
            raise

        except Exception as exc:
            # Handle other exceptions
            exception = exc
            if config.log_exceptions:
                await logger.error(
                    f"Unhandled exception in {request.method} {request.path}",
                    extra={
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    },
                    exc_info=config.include_traceback,
                )
            raise

        finally:
            # Calculate response time
            response_time = time.time() - start_time

            # Extract response data
            if response:
                response_data = await _extract_response_data(
                    request, response, response_time, config
                )

                # Skip if only logging errors and this isn't an error
                if config.only_errors and response.status < 400:
                    return

                # Skip if status code is excluded
                if response.status in config.excluded_status_codes:
                    return

                # Determine log level
                if response.status >= 500:
                    log_level = "error"
                elif response.status >= 400:
                    log_level = "warning"
                else:
                    log_level = "info"

                # Log response
                await getattr(logger, log_level)(
                    f"Request completed: {request.method} {request.path} - {response.status}",
                    extra=response_data,
                )

            elif exception:
                # Log failed request
                await logger.error(
                    f"Request failed: {request.method} {request.path}",
                    extra={
                        "response_time_seconds": response_time,
                        "exception_type": type(exception).__name__,
                    },
                )


def setup_aiohttp_logging(
    app: Application,
    logger_name: Optional[str] = None,
    config: Optional[AioHTTPLoggingConfig] = None,
) -> None:
    """
    Setup aiohttp application with structured logging

    Args:
        app: aiohttp application
        logger_name: Logger name (defaults to 'aiohttp')
        config: Logging configuration

    Example:
        ```python
        from aiohttp import web
        from structured_logging.integrations.aiohttp import setup_aiohttp_logging

        app = web.Application()
        setup_aiohttp_logging(app)
        ```
    """
    if not HAS_AIOHTTP:
        raise ImportError("aiohttp is required for aiohttp integration")

    # Store configuration
    app["structured_logging_config"] = config or AioHTTPLoggingConfig()

    # Create logger
    app["structured_logger"] = get_async_logger(logger_name or "aiohttp")

    # Add middleware
    app.middlewares.append(aiohttp_structured_logging_middleware)


# WebSocket logging


class WebSocketLogger:
    """
    Logger for WebSocket connections

    Provides structured logging for WebSocket lifecycle events,
    messages, and errors.
    """

    def __init__(
        self,
        ws: web.WebSocketResponse,
        request: Request,
        config: Optional[AioHTTPLoggingConfig] = None,
    ):
        self.ws = ws
        self.request = request
        self.config = config or AioHTTPLoggingConfig()
        self.logger = get_async_logger("aiohttp.websocket")
        self.start_time = time.time()
        self.frames_sent = 0
        self.frames_received = 0
        self.bytes_sent = 0
        self.bytes_received = 0

    async def log_connection(self) -> None:
        """Log WebSocket connection established"""
        await self.logger.info(
            f"WebSocket connected: {self.request.path}",
            extra={
                "websocket_event": "connected",
                "path": self.request.path,
                "remote": str(self.request.remote),
                "headers": _extract_headers(self.request, self.config),
            },
        )

    async def log_message_sent(self, message: Any, msg_type: str = "text") -> None:
        """Log outgoing WebSocket message"""
        self.frames_sent += 1

        if isinstance(message, (str, bytes)):
            size = len(message)
            self.bytes_sent += size
        else:
            size = 0

        extra = {
            "websocket_event": "message_sent",
            "message_type": msg_type,
            "message_size": size,
            "frames_sent": self.frames_sent,
            "bytes_sent": self.bytes_sent,
        }

        if (
            self.config.log_websocket_frames
            and size <= self.config.websocket_frame_max_size
        ):
            extra["message_preview"] = str(message)[
                : self.config.websocket_frame_max_size
            ]

        await self.logger.debug("WebSocket message sent", extra=extra)

    async def log_message_received(self, message: Any, msg_type: str = "text") -> None:
        """Log incoming WebSocket message"""
        self.frames_received += 1

        if isinstance(message, (str, bytes)):
            size = len(message)
            self.bytes_received += size
        else:
            size = 0

        extra = {
            "websocket_event": "message_received",
            "message_type": msg_type,
            "message_size": size,
            "frames_received": self.frames_received,
            "bytes_received": self.bytes_received,
        }

        if (
            self.config.log_websocket_frames
            and size <= self.config.websocket_frame_max_size
        ):
            extra["message_preview"] = str(message)[
                : self.config.websocket_frame_max_size
            ]

        await self.logger.debug("WebSocket message received", extra=extra)

    async def log_error(self, error: Exception) -> None:
        """Log WebSocket error"""
        await self.logger.error(
            f"WebSocket error: {type(error).__name__}",
            extra={
                "websocket_event": "error",
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
            exc_info=self.config.include_traceback,
        )

    async def log_close(
        self, code: Optional[int] = None, reason: Optional[str] = None
    ) -> None:
        """Log WebSocket connection closed"""
        duration = time.time() - self.start_time

        await self.logger.info(
            f"WebSocket closed: {self.request.path}",
            extra={
                "websocket_event": "closed",
                "close_code": code,
                "close_reason": reason,
                "duration_seconds": duration,
                "frames_sent": self.frames_sent,
                "frames_received": self.frames_received,
                "bytes_sent": self.bytes_sent,
                "bytes_received": self.bytes_received,
            },
        )


# Client logging


class LoggingClientSession:
    """
    aiohttp ClientSession wrapper with structured logging

    Automatically logs all outgoing HTTP requests with timing,
    headers, and response information.
    """

    def __init__(
        self,
        session: aiohttp.ClientSession,
        logger_name: str = "aiohttp.client",
        config: Optional[AioHTTPLoggingConfig] = None,
    ):
        self.session = session
        self.logger = get_async_logger(logger_name)
        self.config = config or AioHTTPLoggingConfig()

    async def request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make HTTP request with logging"""
        start_time = time.time()

        # Extract request data
        request_data = {
            "client_method": method,
            "client_url": url,
            "client_timeout": kwargs.get("timeout", self.config.client_request_timeout),
        }

        # Log headers if configured
        if self.config.log_request_headers and "headers" in kwargs:
            headers = _clean_headers(kwargs["headers"], self.config)
            request_data["client_headers"] = headers

        # Log request
        await self.logger.info(f"Client request: {method} {url}", extra=request_data)

        response = None
        try:
            # Make request
            response = await self.session.request(method, url, **kwargs)
            return response

        except asyncio.TimeoutError:
            await self.logger.error(
                f"Client request timeout: {method} {url}",
                extra={
                    "client_timeout": kwargs.get(
                        "timeout", self.config.client_request_timeout
                    ),
                    "response_time_seconds": time.time() - start_time,
                },
            )
            raise

        except Exception as exc:
            await self.logger.error(
                f"Client request failed: {method} {url}",
                extra={
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "response_time_seconds": time.time() - start_time,
                },
                exc_info=self.config.include_traceback,
            )
            raise

        finally:
            if response:
                response_time = time.time() - start_time

                # Log response
                log_level = "error" if response.status >= 400 else "info"
                await getattr(self.logger, log_level)(
                    f"Client response: {response.status} from {method} {url}",
                    extra={
                        "client_status": response.status,
                        "client_response_time_seconds": response_time,
                        "client_content_length": response.content_length,
                    },
                )

    # Delegate all other methods to the wrapped session
    def __getattr__(self, name):
        return getattr(self.session, name)


# Helper functions


def _should_exclude_request(request: Request, config: AioHTTPLoggingConfig) -> bool:
    """Check if request should be excluded from logging"""
    # Check excluded paths
    for excluded_path in config.excluded_paths:
        if request.path.startswith(excluded_path):
            return True

    # Check excluded methods
    if request.method in config.excluded_methods:
        return True

    return False


async def _extract_request_data(
    request: Request, config: AioHTTPLoggingConfig
) -> Dict[str, Any]:
    """Extract data from request for logging"""
    data = {
        "method": request.method,
        "path": request.path,
        "query_string": request.query_string,
        "remote": str(request.remote),
        "user_agent": request.headers.get("User-Agent", ""),
        "referer": request.headers.get("Referer", ""),
    }

    # Add headers if configured
    if config.log_request_headers:
        data["headers"] = _extract_headers(request, config)

    # Add request size
    if config.include_request_size:
        data["request_size"] = request.content_length or 0

    # Add request body if configured
    if config.log_request_body and request.body_exists:
        body = await _extract_body(request, config.request_body_max_size)
        if body is not None:
            data["request_body"] = body

    return data


async def _extract_response_data(
    request: Request,
    response: StreamResponse,
    response_time: float,
    config: AioHTTPLoggingConfig,
) -> Dict[str, Any]:
    """Extract data from response for logging"""
    data = {
        "status_code": response.status,
        "response_time_seconds": response_time,
    }

    # Add response headers if configured
    if config.log_response_headers:
        headers = {}
        for key, value in response.headers.items():
            if key.lower() not in config.sensitive_headers:
                headers[key.lower().replace("-", "_")] = value
        data["response_headers"] = headers

    # Add response size
    if config.include_response_size and hasattr(response, "content_length"):
        data["response_size"] = response.content_length or 0

    # Add response body if configured
    if config.log_response_body and isinstance(response, Response):
        if hasattr(response, "body") and response.body:
            body = _parse_body(
                response.body, response.content_type, config.response_body_max_size
            )
            if body is not None:
                data["response_body"] = body

    return data


def _extract_headers(request: Request, config: AioHTTPLoggingConfig) -> Dict[str, str]:
    """Extract and clean request headers"""
    headers = {}

    for key, value in request.headers.items():
        key_lower = key.lower()

        # Skip or redact sensitive headers
        if key_lower in config.sensitive_headers:
            headers[key_lower.replace("-", "_")] = "[REDACTED]"
        else:
            headers[key_lower.replace("-", "_")] = value

    return headers


def _clean_headers(
    headers: Dict[str, str], config: AioHTTPLoggingConfig
) -> Dict[str, str]:
    """Clean headers for logging"""
    cleaned = {}

    for key, value in headers.items():
        key_lower = key.lower()
        if key_lower in config.sensitive_headers:
            cleaned[key] = "[REDACTED]"
        else:
            cleaned[key] = value

    return cleaned


async def _extract_body(request: Request, max_size: int) -> Optional[Union[str, dict]]:
    """Extract request body for logging"""
    if not request.body_exists:
        return None

    try:
        # Read body (this consumes it, so we need to restore it)
        body = await request.read()

        # Restore body for handler
        request._payload = aiohttp.streams.StreamReader()
        request._payload.feed_data(body)
        request._payload.feed_eof()

        return _parse_body(body, request.content_type, max_size)

    except Exception:
        return None


def _parse_body(
    body: bytes, content_type: str, max_size: int
) -> Optional[Union[str, dict]]:
    """Parse body content for logging"""
    if not body:
        return None

    # Check size limit
    if len(body) > max_size:
        return f"[Body too large: {len(body)} bytes]"

    # Try to parse JSON
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


# Decorators for route handlers


def log_route(
    name: Optional[str] = None,
    log_request: bool = True,
    log_response: bool = True,
    **extra_fields,
):
    """
    Decorator for adding structured logging to individual route handlers

    Args:
        name: Custom name for the operation
        log_request: Whether to log request details
        log_response: Whether to log response details
        **extra_fields: Additional fields to include in logs

    Example:
        ```python
        @log_route(name="get_user", operation="read")
        async def get_user(request):
            user_id = request.match_info['id']
            # Handler implementation
            return web.json_response(user_data)
        ```
    """

    def decorator(handler):
        async def wrapped(request: Request) -> StreamResponse:
            logger = get_async_logger("aiohttp.route")
            start_time = time.time()

            # Log request
            if log_request:
                await logger.info(
                    f"Route handler started: {name or handler.__name__}",
                    extra={
                        "handler": handler.__name__,
                        "method": request.method,
                        "path": request.path,
                        **extra_fields,
                    },
                )

            try:
                # Call handler
                response = await handler(request)

                # Log response
                if log_response:
                    await logger.info(
                        f"Route handler completed: {name or handler.__name__}",
                        extra={
                            "handler": handler.__name__,
                            "status": response.status,
                            "response_time_seconds": time.time() - start_time,
                            **extra_fields,
                        },
                    )

                return response

            except Exception as exc:
                await logger.error(
                    f"Route handler failed: {name or handler.__name__}",
                    extra={
                        "handler": handler.__name__,
                        "exception_type": type(exc).__name__,
                        "response_time_seconds": time.time() - start_time,
                        **extra_fields,
                    },
                    exc_info=True,
                )
                raise

        return wrapped

    return decorator
