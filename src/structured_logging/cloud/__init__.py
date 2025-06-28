"""
Cloud platform logging handlers for structured logging

Provides integration with major cloud logging services:
- AWS CloudWatch
- Google Cloud Logging
- Azure Monitor
"""

from .base import CloudHandlerConfig, CloudLogHandler

# Import cloud handlers when available
try:
    from .aws import CloudWatchConfig, CloudWatchHandler
    
    __all__ = [
        "CloudHandlerConfig",
        "CloudLogHandler",
        "CloudWatchConfig", 
        "CloudWatchHandler",
    ]
except ImportError:
    # AWS dependencies not installed
    __all__ = [
        "CloudHandlerConfig",
        "CloudLogHandler",
    ]