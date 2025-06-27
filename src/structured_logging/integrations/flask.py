"""
Flask integration for structured logging
"""

import time
import uuid
from typing import Any, Optional

try:
    from flask import Flask, g, request

    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    # Define dummy types for type checking
    Flask = Any
    g = None
    request = None

from ..config import LoggerConfig
from ..context import request_context
from ..logger import get_logger, log_with_context
from .config import FastAPILoggingConfig
from .fastapi import create_fastapi_logger_config


def create_flask_logger_config() -> LoggerConfig:
    """Create a logger configuration optimized for Flask"""
    return create_fastapi_logger_config()  # Same config works for Flask


class FlaskLoggingMiddleware:
    """Basic Flask middleware for request logging"""

    def __init__(self, app=None, config: Optional[FastAPILoggingConfig] = None):
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