"""
Configuration for file-based logging handlers
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FileHandlerConfig:
    """Configuration for file-based logging handlers"""

    # Basic file settings
    filename: str = "app.log"
    mode: str = "a"
    encoding: str = "utf-8"

    # Rotation settings
    max_bytes: int = 10 * 1024 * 1024  # 10MB default
    backup_count: int = 5
    rotate_on_startup: bool = False

    # Compression settings
    compress_rotated: bool = True
    compression_level: int = 6

    # Archive settings
    archive_old_logs: bool = True
    archive_after_days: int = 30
    archive_directory: Optional[str] = None

    # Performance settings
    buffer_size: int = 8192
    flush_interval: float = 5.0  # seconds
    async_compression: bool = True

    # Cleanup settings
    delete_archived_after_days: Optional[int] = 365
    cleanup_on_startup: bool = True