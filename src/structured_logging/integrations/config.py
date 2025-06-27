"""
Configuration classes for framework integrations
"""

from dataclasses import dataclass, field
from typing import Optional, Set

from ..config import LoggerConfig


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