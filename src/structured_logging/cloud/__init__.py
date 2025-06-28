"""
Cloud platform logging handlers for structured logging

Provides integration with major cloud logging services:
- AWS CloudWatch
- Google Cloud Logging
- Azure Monitor
"""

from .base import CloudHandlerConfig, CloudLogHandler

# Import cloud handlers when available
__all__ = [
    "CloudHandlerConfig",
    "CloudLogHandler",
]

# AWS CloudWatch
try:
    from .aws import CloudWatchConfig, CloudWatchHandler
    __all__.extend(["CloudWatchConfig", "CloudWatchHandler"])
except ImportError:
    pass

# Google Cloud Logging
try:
    from .gcp import (
        GoogleCloudConfig,
        GoogleCloudHandler,
        StackdriverConfig,
        StackdriverHandler,
    )
    __all__.extend([
        "GoogleCloudConfig",
        "GoogleCloudHandler",
        "StackdriverConfig",
        "StackdriverHandler",
    ])
except ImportError:
    pass