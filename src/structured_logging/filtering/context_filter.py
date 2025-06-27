"""
Context-based log filtering
"""

import logging
from typing import Any, Dict, List, Optional

from .base import FilterResult, LogFilter


class ContextFilter(LogFilter):
    """Filter logs based on context values"""

    def __init__(
        self,
        required_keys: Optional[List[str]] = None,
        excluded_keys: Optional[List[str]] = None,
        key_value_filters: Optional[Dict[str, Any]] = None,
    ):
        self.required_keys = set(required_keys or [])
        self.excluded_keys = set(excluded_keys or [])
        self.key_value_filters = key_value_filters or {}

    def should_log(
        self, record: logging.LogRecord, context: Dict[str, Any]
    ) -> FilterResult:
        # Check required keys
        if self.required_keys and not self.required_keys.issubset(context.keys()):
            missing = self.required_keys - context.keys()
            return FilterResult(
                should_log=False,
                reason=f"context_filter: missing required keys {missing}",
            )

        # Check excluded keys
        if self.excluded_keys and any(key in context for key in self.excluded_keys):
            found = self.excluded_keys.intersection(context.keys())
            return FilterResult(
                should_log=False, reason=f"context_filter: found excluded keys {found}"
            )

        # Check key-value filters
        for key, expected_value in self.key_value_filters.items():
            if key in context and context[key] != expected_value:
                return FilterResult(
                    should_log=False,
                    reason=f"context_filter: {key}={context[key]} != {expected_value}",
                )

        return FilterResult(should_log=True, reason="context_filter: passed")