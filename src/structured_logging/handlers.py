"""
Advanced logging handlers for structured logging
"""

import gzip
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional


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


class RotatingFileHandler(logging.Handler):
    """
    Enhanced rotating file handler with compression and archiving
    """

    def __init__(self, config: FileHandlerConfig):
        super().__init__()
        self.config = config
        self.base_filename = config.filename
        self.mode = config.mode
        self.encoding = config.encoding

        # Create directory if it doesn't exist
        self.log_dir = Path(self.base_filename).parent
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Archive directory setup
        if config.archive_directory:
            self.archive_dir = Path(config.archive_directory)
        else:
            self.archive_dir = self.log_dir / "archive"
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        # Current state
        self.stream: Optional[Any] = None
        self.bytes_written = 0
        self.last_flush_time = time.time()

        # Thread pool for async operations
        self.executor: Optional[ThreadPoolExecutor] = None
        if config.async_compression:
            self.executor = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="log_handler"
            )

        # Initialize
        self._open_stream()

        if config.rotate_on_startup:
            self.do_rollover()

        if config.cleanup_on_startup:
            self._cleanup_old_files()

    def _open_stream(self):
        """Open the log file stream"""
        if self.stream:
            self.stream.close()

        # Get current file size if file exists
        if os.path.exists(self.base_filename):
            self.bytes_written = os.path.getsize(self.base_filename)
        else:
            self.bytes_written = 0

        self.stream = open(
            self.base_filename,
            self.mode,
            encoding=self.encoding,
            buffering=self.config.buffer_size,
        )

    def emit(self, record: logging.LogRecord):
        """Emit a log record"""
        try:
            # Check if rotation is needed
            if self._should_rollover(record):
                self.do_rollover()

            # Format and write the record
            msg = self.format(record)
            stream = self.stream

            # Write to stream
            stream.write(msg + "\n")
            self.bytes_written += len(msg.encode(self.encoding)) + 1

            # Check if we should flush
            if self._should_flush():
                self.flush()

        except Exception:
            self.handleError(record)

    def _should_rollover(self, record: logging.LogRecord) -> bool:
        """Determine if rollover should occur"""
        if self.config.max_bytes > 0:
            # Estimate the size of the formatted record
            msg = self.format(record)
            record_size = len(msg.encode(self.encoding)) + 1

            return (self.bytes_written + record_size) >= self.config.max_bytes
        return False

    def _should_flush(self) -> bool:
        """Determine if we should flush the stream"""
        return (time.time() - self.last_flush_time) >= self.config.flush_interval

    def flush(self):
        """Flush the stream"""
        if self.stream:
            self.stream.flush()
            self.last_flush_time = time.time()

    def do_rollover(self):
        """Perform log file rotation"""
        if self.stream:
            self.stream.close()
            self.stream = None

        # Generate rotated filenames
        rotated_files = []

        # Shift existing backup files
        for i in range(self.config.backup_count - 1, 0, -1):
            sfn = f"{self.base_filename}.{i}"
            dfn = f"{self.base_filename}.{i + 1}"
            if os.path.exists(sfn):
                if os.path.exists(dfn):
                    os.remove(dfn)
                os.rename(sfn, dfn)
                rotated_files.append(dfn)

        # Move current file to .1
        dfn = f"{self.base_filename}.1"
        if os.path.exists(dfn):
            os.remove(dfn)
        if os.path.exists(self.base_filename):
            os.rename(self.base_filename, dfn)
            rotated_files.append(dfn)

        # Compress rotated files if configured
        if self.config.compress_rotated and rotated_files:
            if self.config.async_compression and self.executor:
                # Compress asynchronously
                for filename in rotated_files:
                    self.executor.submit(self._compress_file, filename)
            else:
                # Compress synchronously
                for filename in rotated_files:
                    self._compress_file(filename)

        # Archive old logs if configured
        if self.config.archive_old_logs:
            if self.config.async_compression and self.executor:
                self.executor.submit(self._archive_old_logs)
            else:
                self._archive_old_logs()

        # Reopen the log file
        self._open_stream()

    def _compress_file(self, filename: str):
        """Compress a log file using gzip"""
        try:
            compressed_filename = f"{filename}.gz"

            with open(filename, "rb") as f_in:
                with gzip.open(
                    compressed_filename,
                    "wb",
                    compresslevel=self.config.compression_level,
                ) as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Remove original file after successful compression
            os.remove(filename)

        except Exception as e:
            # Log compression error (to stderr to avoid recursion)
            print(
                f"Error compressing log file {filename}: {e}",
                file=__import__("sys").stderr,
            )

    def _archive_old_logs(self):
        """Move old log files to archive directory"""
        try:
            cutoff_date = datetime.now() - timedelta(
                days=self.config.archive_after_days
            )

            # Find files to archive
            log_pattern = f"{Path(self.base_filename).name}.*"

            for file_path in self.log_dir.glob(log_pattern):
                if file_path == Path(self.base_filename):
                    continue  # Skip current log file

                # Check file age
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_mtime < cutoff_date:
                    # Move to archive directory
                    archive_path = self.archive_dir / file_path.name

                    # If archive file exists, add timestamp to make unique
                    if archive_path.exists():
                        timestamp = file_mtime.strftime("%Y%m%d_%H%M%S")
                        archive_path = (
                            self.archive_dir / f"{timestamp}_{file_path.name}"
                        )

                    shutil.move(str(file_path), str(archive_path))

        except Exception as e:
            print(f"Error archiving old logs: {e}", file=__import__("sys").stderr)

    def _cleanup_old_files(self):
        """Clean up very old archived files"""
        if not self.config.delete_archived_after_days:
            return

        try:
            cutoff_date = datetime.now() - timedelta(
                days=self.config.delete_archived_after_days
            )

            for file_path in self.archive_dir.glob("*"):
                if file_path.is_file():
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_mtime < cutoff_date:
                        file_path.unlink()

        except Exception as e:
            print(
                f"Error cleaning up old archived files: {e}",
                file=__import__("sys").stderr,
            )

    def close(self):
        """Close the handler and clean up resources"""
        if self.stream:
            self.stream.close()
            self.stream = None

        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None

        super().close()


class TimedRotatingFileHandler(RotatingFileHandler):
    """
    File handler that rotates based on time intervals
    """

    def __init__(
        self, config: FileHandlerConfig, when: str = "midnight", interval: int = 1
    ):
        """
        Initialize timed rotating file handler

        Args:
            config: File handler configuration
            when: Type of interval ('S', 'M', 'H', 'D', 'midnight', 'W0'-'W6')
            interval: Number of intervals between rotations
        """
        self.when = when.upper()
        self.interval = interval
        self.suffix = None
        self.ext_match = None

        # Set up time-based rotation parameters
        if self.when == "S":
            self.interval_seconds = 1
            self.suffix = "%Y-%m-%d_%H-%M-%S"
        elif self.when == "M":
            self.interval_seconds = 60
            self.suffix = "%Y-%m-%d_%H-%M"
        elif self.when == "H":
            self.interval_seconds = 60 * 60
            self.suffix = "%Y-%m-%d_%H"
        elif self.when == "D" or self.when == "MIDNIGHT":
            self.interval_seconds = 60 * 60 * 24
            self.suffix = "%Y-%m-%d"
        elif self.when.startswith("W"):
            self.interval_seconds = 60 * 60 * 24 * 7
            self.suffix = "%Y-%m-%d"
        else:
            raise ValueError(f"Invalid value for 'when': {when}")

        # Calculate next rollover time
        self.rollover_at = self._compute_rollover_time()

        super().__init__(config)

    def _compute_rollover_time(self) -> float:
        """Compute the next rollover time"""
        current_time = int(time.time())

        if self.when == "MIDNIGHT":
            # Roll over at midnight
            t = time.localtime(current_time)
            next_midnight = time.mktime(
                (t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1)
            )
            next_midnight += 24 * 60 * 60  # Next day midnight
            return next_midnight
        else:
            # Roll over at regular intervals
            return current_time + (self.interval * self.interval_seconds)

    def _should_rollover(self, record: logging.LogRecord) -> bool:
        """Determine if rollover should occur based on time"""
        return time.time() >= self.rollover_at

    def do_rollover(self):
        """Perform time-based log file rotation"""
        if self.stream:
            self.stream.close()
            self.stream = None

        # Generate timestamped filename
        t = time.localtime(self.rollover_at - 1)  # Use time just before rollover
        timestamped_filename = f"{self.base_filename}.{time.strftime(self.suffix, t)}"

        # Move current file to timestamped name
        if os.path.exists(self.base_filename):
            if os.path.exists(timestamped_filename):
                os.remove(timestamped_filename)
            os.rename(self.base_filename, timestamped_filename)

            # Compress if configured
            if self.config.compress_rotated:
                if self.config.async_compression and self.executor:
                    self.executor.submit(self._compress_file, timestamped_filename)
                else:
                    self._compress_file(timestamped_filename)

        # Calculate next rollover time
        self.rollover_at = self._compute_rollover_time()

        # Archive old logs if configured
        if self.config.archive_old_logs:
            if self.config.async_compression and self.executor:
                self.executor.submit(self._archive_old_logs)
            else:
                self._archive_old_logs()

        # Reopen the log file
        self._open_stream()


def create_file_logger(
    name: str,
    config: FileHandlerConfig,
    formatter: Optional[logging.Formatter] = None,
    handler_type: str = "rotating",
) -> logging.Logger:
    """
    Create a logger with file handler

    Args:
        name: Logger name
        config: File handler configuration
        formatter: Optional custom formatter
        handler_type: Type of handler ('rotating' or 'timed')

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create appropriate handler
    if handler_type == "timed":
        handler = TimedRotatingFileHandler(config)
    else:
        handler = RotatingFileHandler(config)

    # Set formatter
    if formatter:
        handler.setFormatter(formatter)
    else:
        # Use simple default formatter
        simple_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(simple_formatter)

    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    return logger
