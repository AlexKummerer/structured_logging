"""
Configuration classes for network logging handlers
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class NetworkHandlerConfig:
    """Base configuration for network handlers"""

    # Connection settings
    host: str = "localhost"
    port: int = 514
    timeout: float = 5.0

    # Reliability settings
    max_retries: int = 3
    retry_delay: float = 1.0
    fallback_to_local: bool = True
    local_fallback_file: Optional[str] = "network_fallback.log"

    # Buffer settings
    buffer_size: int = 1024
    batch_size: int = 1
    flush_interval: float = 1.0

    # Security settings
    use_ssl: bool = False
    verify_ssl: bool = True
    ssl_cert_file: Optional[str] = None
    ssl_key_file: Optional[str] = None


@dataclass
class SyslogConfig(NetworkHandlerConfig):
    """Configuration for Syslog handler"""

    port: int = 514  # Standard syslog port
    facility: int = 16  # Local use facility (local0)
    rfc_format: str = "3164"  # RFC 3164 or 5424
    hostname: Optional[str] = None
    app_name: str = "python-app"
    process_id: Optional[int] = None

    # Message formatting
    include_timestamp: bool = True
    include_hostname: bool = True
    include_process_id: bool = True


@dataclass
class HTTPConfig(NetworkHandlerConfig):
    """Configuration for HTTP handler"""

    url: str = "http://localhost:8080/logs"
    method: str = "POST"

    # Authentication
    auth_type: str = "none"  # none, basic, bearer, api_key
    username: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None
    api_key: Optional[str] = None
    api_key_header: str = "X-API-Key"

    # Request settings
    headers: Dict[str, str] = field(default_factory=dict)
    batch_size: int = 10
    max_batch_time: float = 5.0

    # Content settings
    content_type: str = "application/json"
    compress_payload: bool = True

    # HTTP specific
    user_agent: str = "StructuredLogging/0.6.0"


@dataclass
class SocketConfig(NetworkHandlerConfig):
    """Configuration for Socket handler (TCP/UDP)"""

    protocol: str = "tcp"  # tcp or udp
    port: int = 5140

    # TCP specific
    keep_alive: bool = True
    tcp_nodelay: bool = True
    connection_pool_size: int = 5

    # UDP specific
    udp_buffer_size: int = 65507  # Max UDP payload

    # Message formatting
    message_delimiter: str = "\n"
    encoding: str = "utf-8"