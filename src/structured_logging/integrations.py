"""
Framework integrations for structured logging
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set

try:
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import StreamingResponse
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.types import ASGIApp, Receive, Scope, Send

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from .config import LoggerConfig
from .context import request_context
from .logger import get_logger, log_with_context


@dataclass
class FastAPILoggingConfig:
    """Configuration for FastAPI logging middleware"""

    # Logger configuration
    logger_config: Optional[LoggerConfig] = None
    logger_name: str = "fastapi"

    # Request logging
    log_requests: bool = True
    log_request_body: bool = False
    log_request_headers: bool = True
    max_request_body_size: int = 1024  # bytes

    # Response logging
    log_responses: bool = True
    log_response_body: bool = False
    log_response_headers: bool = False
    max_response_body_size: int = 1024  # bytes

    # Filtering
    exclude_paths: Set[str] = field(
        default_factory=lambda: {"/health", "/metrics", "/favicon.ico"}
    )
    exclude_methods: Set[str] = field(default_factory=set)
    log_only_errors: bool = False
    min_duration_ms: Optional[float] = None

    # Sensitive data protection
    sensitive_headers: Set[str] = field(
        default_factory=lambda: {"authorization", "cookie", "x-api-key", "x-auth-token"}
    )
    sensitive_query_params: Set[str] = field(
        default_factory=lambda: {"password", "token", "api_key", "secret"}
    )
    mask_sensitive_data: bool = True

    # Performance
    capture_user_agent: bool = True
    capture_ip_address: bool = True
    capture_route_info: bool = True


class FastAPILoggingMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for automatic request/response logging"""

    def __init__(self, app: ASGIApp, config: FastAPILoggingConfig):
        super().__init__(app)
        self.config = config
        self.logger = get_logger(config.logger_name, config.logger_config)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and response with logging"""

        # Skip excluded paths
        if request.url.path in self.config.exclude_paths:
            return await call_next(request)

        # Skip excluded methods
        if request.method in self.config.exclude_methods:
            return await call_next(request)

        # Generate request ID
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Extract request info
        request_info = await self._extract_request_info(request)
        request_info["request_id"] = request_id

        # Log request
        if self.config.log_requests:
            self.logger.info(
                f"Request started: {request.method} {request.url.path}",
                extra={f"ctx_{k}": v for k, v in request_info.items() if v is not None},
            )

        # Process request
        exception_occurred = False
        response = None

        try:
            response = await call_next(request)
        except Exception as e:
            exception_occurred = True

            # Log exception
            error_info = request_info.copy()
            error_info.update({"exception": str(e), "exception_type": type(e).__name__})

            self.logger.error(
                f"Request failed: {request.method} {request.url.path} - {str(e)}",
                extra={f"ctx_{k}": v for k, v in error_info.items() if v is not None},
            )
            raise  # Re-raise the exception

        finally:
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Extract response info if response exists
            response_info = {}
            if response is not None:
                response_info = await self._extract_response_info(response)

            # Determine if we should log this request
            should_log = self._should_log_request(
                response, duration_ms, exception_occurred
            )

            if should_log and self.config.log_responses:
                # Determine log level based on status code
                log_level = self._get_log_level_for_response(
                    response, exception_occurred
                )

                # Create complete log info
                complete_info = request_info.copy()
                complete_info.update(response_info)
                complete_info["duration_ms"] = duration_ms

                # Create log message
                if exception_occurred:
                    message = f"Request failed: {request.method} {request.url.path}"
                else:
                    status_code = getattr(response, "status_code", "unknown")
                    message = f"Request completed: {request.method} {request.url.path} - {status_code}"

                # Log response using the appropriate level
                log_method = getattr(self.logger, log_level)
                log_method(
                    message,
                    extra={
                        f"ctx_{k}": v for k, v in complete_info.items() if v is not None
                    },
                )

        return response

    async def _extract_request_info(self, request: Request) -> Dict[str, Any]:
        """Extract relevant information from request"""
        info = {
            "method": request.method,
            "path": request.url.path,
            "query_params": (
                dict(request.query_params) if request.query_params else None
            ),
        }

        # Add IP address
        if self.config.capture_ip_address:
            client_ip = (
                getattr(request.client, "host", None) if request.client else None
            )
            if client_ip:
                info["client_ip"] = client_ip

        # Add user agent
        if self.config.capture_user_agent:
            user_agent = request.headers.get("user-agent")
            if user_agent:
                info["user_agent"] = user_agent

        # Add route info
        if self.config.capture_route_info:
            route = getattr(request, "route", None)
            if route:
                info["route_name"] = getattr(route, "name", None)
                info["route_path"] = getattr(route, "path", None)

        # Add headers (filtered)
        if self.config.log_request_headers:
            headers = self._filter_sensitive_headers(dict(request.headers))
            if headers:
                info["headers"] = headers

        # Add request body
        if self.config.log_request_body:
            body = await self._extract_request_body(request)
            if body:
                info["request_body"] = body

        # Filter sensitive query params
        if self.config.mask_sensitive_data and info.get("query_params"):
            info["query_params"] = self._filter_sensitive_query_params(
                info["query_params"]
            )

        return {k: v for k, v in info.items() if v is not None}

    async def _extract_response_info(self, response: Response) -> Dict[str, Any]:
        """Extract relevant information from response"""
        info = {
            "status_code": getattr(response, "status_code", None),
        }

        # Add response headers (filtered)
        if self.config.log_response_headers:
            headers = getattr(response, "headers", {})
            if headers:
                filtered_headers = self._filter_sensitive_headers(dict(headers))
                if filtered_headers:
                    info["response_headers"] = filtered_headers

        # Add response body (limited)
        if self.config.log_response_body:
            body = await self._extract_response_body(response)
            if body:
                info["response_body"] = body

        return {k: v for k, v in info.items() if v is not None}

    async def _extract_request_body(self, request: Request) -> Optional[str]:
        """Extract and limit request body"""
        try:
            # Only read body for certain content types
            content_type = request.headers.get("content-type", "")
            if not any(ct in content_type.lower() for ct in ["json", "text", "form"]):
                return None

            body = await request.body()
            if not body:
                return None

            # Limit body size
            if len(body) > self.config.max_request_body_size:
                body = body[: self.config.max_request_body_size]
                body_str = body.decode("utf-8", errors="ignore") + "...[truncated]"
            else:
                body_str = body.decode("utf-8", errors="ignore")

            # Try to parse as JSON for better formatting
            try:
                parsed = json.loads(body_str.replace("...[truncated]", ""))
                return parsed
            except (json.JSONDecodeError, ValueError):
                return body_str

        except Exception:
            return None

    async def _extract_response_body(self, response: Response) -> Optional[str]:
        """Extract and limit response body"""
        try:
            # Skip for streaming responses
            if isinstance(response, StreamingResponse):
                return None

            # Only process certain content types
            content_type = response.headers.get("content-type", "")
            if not any(ct in content_type.lower() for ct in ["json", "text"]):
                return None

            # This is tricky with FastAPI responses - we can't easily read the body
            # without consuming it. For now, we'll skip response body logging
            # unless specifically implemented by the user
            return None

        except Exception:
            return None

    def _filter_sensitive_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Filter out sensitive headers"""
        if not self.config.mask_sensitive_data:
            return headers

        filtered = {}
        for key, value in headers.items():
            if key.lower() in self.config.sensitive_headers:
                filtered[key] = "[MASKED]"
            else:
                filtered[key] = value

        return filtered

    def _filter_sensitive_query_params(self, params: Dict[str, str]) -> Dict[str, str]:
        """Filter out sensitive query parameters"""
        if not self.config.mask_sensitive_data:
            return params

        filtered = {}
        for key, value in params.items():
            if key.lower() in self.config.sensitive_query_params:
                filtered[key] = "[MASKED]"
            else:
                filtered[key] = value

        return filtered

    def _should_log_request(
        self, response: Optional[Response], duration_ms: float, exception_occurred: bool
    ) -> bool:
        """Determine if request should be logged"""

        # Always log exceptions
        if exception_occurred:
            return True

        # Log only errors if configured
        if self.config.log_only_errors:
            if response and hasattr(response, "status_code"):
                return response.status_code >= 400
            return False

        # Check minimum duration
        if self.config.min_duration_ms is not None:
            return duration_ms >= self.config.min_duration_ms

        return True

    def _get_log_level_for_response(
        self, response: Optional[Response], exception_occurred: bool
    ) -> str:
        """Determine appropriate log level for response"""
        if exception_occurred:
            return "error"

        if response and hasattr(response, "status_code"):
            status_code = response.status_code
            if status_code >= 500:
                return "error"
            elif status_code >= 400:
                return "warning"
            else:
                return "info"

        return "info"


def add_structured_logging(
    app: FastAPI, config: Optional[FastAPILoggingConfig] = None
) -> FastAPI:
    """
    Add structured logging middleware to FastAPI app

    Args:
        app: FastAPI application instance
        config: Optional logging configuration

    Returns:
        The FastAPI app with logging middleware added
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is not installed. Install it with: pip install fastapi"
        )

    if config is None:
        config = FastAPILoggingConfig()

    app.add_middleware(FastAPILoggingMiddleware, config=config)
    return app


def create_fastapi_logger_config(
    output_type: str = "console",
    log_level: str = "INFO",
    filename: Optional[str] = None,
    enable_filtering: bool = False,
    sample_rate: float = 1.0,
) -> LoggerConfig:
    """
    Create a logger configuration optimized for FastAPI

    Args:
        output_type: Where to output logs ("console", "file", "both")
        log_level: Minimum log level
        filename: Log filename (for file output)
        enable_filtering: Enable log filtering/sampling
        sample_rate: Sampling rate for filtering (0.0-1.0)

    Returns:
        Configured LoggerConfig instance
    """
    from .filtering import FilterConfig, LevelFilter, SamplingFilter
    from .handlers import FileHandlerConfig

    # Create filter config if enabled
    filter_config = None
    if enable_filtering:
        filters = [LevelFilter(min_level=log_level)]
        if sample_rate < 1.0:
            filters.append(
                SamplingFilter(
                    sample_rate=sample_rate,
                    strategy="level_based",  # Better for web apps
                    max_per_second=1000,  # Reasonable default for web apps
                )
            )

        filter_config = FilterConfig(
            enabled=True, filters=filters, collect_metrics=True
        )

    # Create file config if needed
    file_config = None
    if output_type in ["file", "both"]:
        file_config = FileHandlerConfig(
            filename=filename or "fastapi.log",
            max_bytes=50 * 1024 * 1024,  # 50MB for web apps
            backup_count=10,
            compress_rotated=True,
            archive_old_logs=True,
        )

    return LoggerConfig(
        log_level=log_level,
        include_timestamp=True,
        include_request_id=True,
        include_user_context=True,
        formatter_type="json",  # JSON is best for web apps
        filter_config=filter_config,
        output_type=output_type,
        file_config=file_config,
    )


# Flask integration (basic implementation)
def create_flask_logger_config() -> LoggerConfig:
    """Create a logger configuration optimized for Flask"""
    return create_fastapi_logger_config()  # Same config works for Flask


class FlaskLoggingMiddleware:
    """Basic Flask middleware for request logging"""

    def __init__(self, app, config: Optional[FastAPILoggingConfig] = None):
        if config is None:
            config = FastAPILoggingConfig()

        self.config = config
        self.logger = get_logger(config.logger_name, config.logger_config)

        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        """Initialize Flask app with logging"""
        app.before_request(self._before_request)
        app.after_request(self._after_request)

    def _before_request(self):
        """Log request start"""
        try:
            from flask import g, request

            g.request_start_time = time.time()
            g.request_id = str(uuid.uuid4())

            if self.config.log_requests:
                with request_context(custom_fields={"request_id": g.request_id}):
                    log_with_context(
                        self.logger,
                        "info",
                        f"Request started: {request.method} {request.path}",
                        self.config.logger_config,
                        method=request.method,
                        path=request.path,
                        remote_addr=request.remote_addr,
                        user_agent=request.headers.get("User-Agent"),
                    )
        except Exception:
            pass  # Don't break the request if logging fails

    def _after_request(self, response):
        """Log request completion"""
        try:
            from flask import g, request

            duration_ms = (time.time() - g.request_start_time) * 1000

            if self.config.log_responses:
                with request_context(custom_fields={"request_id": g.request_id}):
                    log_level = (
                        "error"
                        if response.status_code >= 500
                        else "warning" if response.status_code >= 400 else "info"
                    )

                    log_with_context(
                        self.logger,
                        log_level,
                        f"Request completed: {request.method} {request.path} - {response.status_code}",
                        self.config.logger_config,
                        method=request.method,
                        path=request.path,
                        status_code=response.status_code,
                        duration_ms=duration_ms,
                    )
        except Exception:
            pass  # Don't break the response if logging fails

        return response
