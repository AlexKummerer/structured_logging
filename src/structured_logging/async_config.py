"""
Async logging configuration
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AsyncLoggerConfig:
    """Configuration for async logging operations"""
    
    # Queue configuration
    queue_size: int = 1000  # Maximum number of queued log entries
    max_workers: int = 2    # Number of background log processors
    
    # Batching configuration
    batch_size: int = 50           # Number of logs to batch together
    flush_interval: float = 1.0    # Maximum time to wait before flushing (seconds)
    
    # Performance tuning
    queue_timeout: float = 0.1     # Timeout for queue operations (seconds)
    shutdown_timeout: float = 5.0  # Timeout for graceful shutdown (seconds)
    
    # Memory management
    max_memory_mb: Optional[int] = None  # Maximum memory usage (MB)
    drop_on_overflow: bool = False       # Drop logs if queue is full
    
    # Error handling
    error_callback: Optional[callable] = None  # Called when async logging errors occur
    
    def __post_init__(self):
        """Validate configuration values"""
        if self.queue_size <= 0:
            raise ValueError("queue_size must be positive")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.flush_interval <= 0:
            raise ValueError("flush_interval must be positive")


# Global default async config
_default_async_config: Optional[AsyncLoggerConfig] = None


def get_default_async_config() -> AsyncLoggerConfig:
    """Get the default async logger configuration"""
    global _default_async_config
    if _default_async_config is None:
        _default_async_config = AsyncLoggerConfig()
    return _default_async_config


def set_default_async_config(config: AsyncLoggerConfig) -> None:
    """Set the default async logger configuration"""
    global _default_async_config
    _default_async_config = config