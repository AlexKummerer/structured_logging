"""
Level-based log filtering
"""

import logging
from typing import Any, Dict, Union

from .base import FilterResult, LogFilter


class LevelFilter(LogFilter):
    """Filter logs based on log level"""

    def __init__(self, min_level: Union[str, int] = logging.INFO):
        if isinstance(min_level, str):
            self.min_level = getattr(logging, min_level.upper())
        else:
            self.min_level = min_level

    def should_log(
        self, record: logging.LogRecord, context: Dict[str, Any]
    ) -> FilterResult:
        should_log = record.levelno >= self.min_level
        return FilterResult(
            should_log=should_log,
            reason=f"level_filter: {record.levelname} {'>=' if should_log else '<'} {logging.getLevelName(self.min_level)}",
        )