"""
FastAPI integration for structured logging
"""

import json
import time
import uuid
from typing import Any, Callable, Dict, Optional

try:
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import StreamingResponse
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.types import ASGIApp

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Define dummy types for type checking
    FastAPI = Any
    Request = Any
    Response = Any
    StreamingResponse = Any
    BaseHTTPMiddleware = object
    ASGIApp = Any

from ..config import LoggerConfig
from ..filtering import FilterConfig, LevelFilter, SamplingFilter
from ..handlers import FileHandlerConfig
from ..logger import get_logger
from .config import FastAPILoggingConfig
from .utils import filter_sensitive_headers, filter_sensitive_query_params


class FastAPILoggingMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for automatic request/response logging"""

    def __init__(self, app: ASGIApp, config: FastAPILoggingConfig):
        super().__init__(app)
        self.config = config
        self.logger = get_logger(config.logger_name, config.logger_config)

    def _should_skip_request(self, request: Request) -> bool:
        """Check if request should be skipped for logging"""
        return (
            request.url.path in self.config.exclude_paths
            or request.method in self.config.exclude_methods
        )

    def _log_request_start(self, request: Request, request_info: Dict[str, Any]) -> None:
        """Log the start of a request"""
        if self.config.log_requests:
            self.logger.info(
                f"Request started: {request.method} {request.url.path}",
                extra={f"ctx_{k}": v for k, v in request_info.items() if v is not None},
            )

    def _log_request_exception(self, request: Request, exception: Exception, request_info: Dict[str, Any]) -> None:
        """Log request exception"""
        error_info = request_info.copy()
        error_info.update({"exception": str(exception), "exception_type": type(exception).__name__})
        
        self.logger.error(
            f"Request failed: {request.method} {request.url.path} - {str(exception)}",
            extra={f"ctx_{k}": v for k, v in error_info.items() if v is not None},
        )

    def _create_completion_message(self, request: Request, response: Optional[Response], exception_occurred: bool) -> str:
        """Create log message for request completion"""
        if exception_occurred:
            return f"Request failed: {request.method} {request.url.path}"
        else:
            status_code = getattr(response, "status_code", "unknown")
            return f"Request completed: {request.method} {request.url.path} - {status_code}"

    async def _log_request_completion(
        self,
        request: Request,
        response: Optional[Response],
        request_info: Dict[str, Any],
        duration_ms: float,
        exception_occurred: bool
    ) -> None:
        """Log request completion with response details"""
        response_info = {}
        if response is not None:
            response_info = await self._extract_response_info(response)

        should_log = self._should_log_request(response, duration_ms, exception_occurred)
        
        if should_log and self.config.log_responses:
            log_level = self._get_log_level_for_response(response, exception_occurred)
            
            complete_info = request_info.copy()
            complete_info.update(response_info)
            complete_info["duration_ms"] = duration_ms
            
            message = self._create_completion_message(request, response, exception_occurred)
            
            log_method = getattr(self.logger, log_level)
            log_method(
                message,
                extra={f"ctx_{k}": v for k, v in complete_info.items() if v is not None},
            )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and response with logging"""
        if self._should_skip_request(request):
            return await call_next(request)

        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        request_info = await self._extract_request_info(request)
        request_info["request_id"] = request_id
        
        self._log_request_start(request, request_info)
        
        exception_occurred = False
        response = None
        
        try:
            response = await call_next(request)
        except Exception as e:
            exception_occurred = True
            self._log_request_exception(request, e, request_info)
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            await self._log_request_completion(
                request, response, request_info, duration_ms, exception_occurred
            )
        
        return response

    def _get_basic_request_info(self, request: Request) -> Dict[str, Any]:
        """Extract basic request information"""
        return {
            "method": request.method,
            "path": request.url.path,
            "query_params": (
                dict(request.query_params) if request.query_params else None
            ),
        }

    def _add_client_info(self, request: Request, info: Dict[str, Any]) -> None:
        """Add client information to request info"""
        if self.config.capture_ip_address:
            client_ip = (
                getattr(request.client, "host", None) if request.client else None
            )
            if client_ip:
                info["client_ip"] = client_ip

        if self.config.capture_user_agent:
            user_agent = request.headers.get("user-agent")
            if user_agent:
                info["user_agent"] = user_agent

    def _add_route_info(self, request: Request, info: Dict[str, Any]) -> None:
        """Add route information to request info"""
        if self.config.capture_route_info:
            route = getattr(request, "route", None)
            if route:
                info["route_name"] = getattr(route, "name", None)
                info["route_path"] = getattr(route, "path", None)

    def _add_headers_info(self, request: Request, info: Dict[str, Any]) -> None:
        """Add filtered headers to request info"""
        if self.config.log_request_headers:
            headers = filter_sensitive_headers(
                dict(request.headers),
                self.config.sensitive_headers,
                self.config.mask_sensitive_data,
            )
            if headers:
                info["headers"] = headers

    async def _add_body_info(self, request: Request, info: Dict[str, Any]) -> None:
        """Add request body to info"""
        if self.config.log_request_body:
            body = await self._extract_request_body(request)
            if body:
                info["request_body"] = body

    def _filter_query_params(self, info: Dict[str, Any]) -> None:
        """Filter sensitive query parameters"""
        if self.config.mask_sensitive_data and info.get("query_params"):
            info["query_params"] = filter_sensitive_query_params(
                info["query_params"],
                self.config.sensitive_query_params,
                self.config.mask_sensitive_data,
            )

    async def _extract_request_info(self, request: Request) -> Dict[str, Any]:
        """Extract relevant information from request"""
        info = self._get_basic_request_info(request)
        
        self._add_client_info(request, info)
        self._add_route_info(request, info)
        self._add_headers_info(request, info)
        await self._add_body_info(request, info)
        self._filter_query_params(info)
        
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
                filtered_headers = filter_sensitive_headers(
                    dict(headers),
                    self.config.sensitive_headers,
                    self.config.mask_sensitive_data,
                )
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


def _create_filter_config_for_fastapi(log_level: str, enable_filtering: bool, sample_rate: float) -> Optional[FilterConfig]:
    """Create filter configuration for FastAPI logging"""
    if not enable_filtering:
        return None
        
    filters = [LevelFilter(min_level=log_level)]
    if sample_rate < 1.0:
        filters.append(
            SamplingFilter(
                sample_rate=sample_rate,
                strategy="level_based",  # Better for web apps
                max_per_second=1000,  # Reasonable default for web apps
            )
        )

    return FilterConfig(enabled=True, filters=filters, collect_metrics=True)


def _create_file_config_for_fastapi(output_type: str, filename: Optional[str]) -> Optional[FileHandlerConfig]:
    """Create file handler configuration for FastAPI logging"""
    if output_type not in ["file", "both"]:
        return None
        
    return FileHandlerConfig(
        filename=filename or "fastapi.log",
        max_bytes=50 * 1024 * 1024,  # 50MB for web apps
        backup_count=10,
        compress_rotated=True,
        archive_old_logs=True,
    )


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
    filter_config = _create_filter_config_for_fastapi(log_level, enable_filtering, sample_rate)
    file_config = _create_file_config_for_fastapi(output_type, filename)
    
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