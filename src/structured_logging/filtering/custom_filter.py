"""
Custom function-based log filtering
"""

import logging
from typing import Any, Callable, Dict

from .base import FilterResult, LogFilter


class CustomFilter(LogFilter):
    """Filter logs using custom function"""

    def __init__(
        self,
        filter_func: Callable[[logging.LogRecord, Dict[str, Any]], bool],
        name: str = "custom",
    ):
        self.filter_func = filter_func
        self.name = name

    def should_log(
        self, record: logging.LogRecord, context: Dict[str, Any]
    ) -> FilterResult:
        try:
            should_log = self.filter_func(record, context)
            return FilterResult(
                should_log=should_log,
                reason=f"{self.name}_filter: {'passed' if should_log else 'rejected'}",
            )
        except Exception as e:
            return FilterResult(
                should_log=True,  # Default to logging on error
                reason=f"{self.name}_filter: error - {e}",
                metadata={"filter_error": str(e)},
            )