"""
Enhanced rotating file handler with compression and archiving
"""

import gzip
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from .config import FileHandlerConfig


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