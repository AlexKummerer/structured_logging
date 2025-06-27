"""
File handler that rotates based on time intervals
"""

import os
import time

from .config import FileHandlerConfig
from .rotating_handler import RotatingFileHandler


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

    def _should_rollover(self, record) -> bool:
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