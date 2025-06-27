"""
Network logging handlers for remote logging capabilities

This package provides handlers for sending logs to remote systems via various protocols.
"""

from .base import BaseNetworkHandler
from .config import HTTPConfig, NetworkHandlerConfig, SocketConfig, SyslogConfig
from .http import HTTPHandler
from .socket import SocketHandler
from .syslog import SyslogHandler

__all__ = [
    # Configuration
    "NetworkHandlerConfig",
    "SyslogConfig",
    "HTTPConfig",
    "SocketConfig",
    # Base handler
    "BaseNetworkHandler",
    # Handlers
    "SyslogHandler",
    "HTTPHandler",
    "SocketHandler",
]