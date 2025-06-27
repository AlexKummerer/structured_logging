"""
Main filtering engine that applies multiple filters
"""

import logging
from collections import defaultdict
from typing import Any, Dict

from .base import FilterResult
from .config import FilterConfig


class FilterEngine:
    """Main filtering engine that applies multiple filters"""

    def __init__(self, config: FilterConfig):
        self.config = config
        self.metrics: Dict[str, int] = defaultdict(int)
        self._filter_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def should_log(
        self, record: logging.LogRecord, context: Dict[str, Any]
    ) -> FilterResult:
        """Apply all filters and return final decision"""
        if not self.config.enabled:
            return FilterResult(should_log=True, reason="filtering_disabled")

        self.metrics["total_evaluated"] += 1

        # Apply each filter in sequence
        for i, filter_obj in enumerate(self.config.filters):
            result = filter_obj.should_log(record, context)

            # Update metrics
            if self.config.collect_metrics:
                filter_name = f"{filter_obj.__class__.__name__}_{i}"
                self._filter_stats[filter_name]["total"] += 1
                if result.should_log:
                    self._filter_stats[filter_name]["passed"] += 1
                else:
                    self._filter_stats[filter_name]["rejected"] += 1

            # If any filter rejects, stop processing
            if not result.should_log:
                self.metrics["filtered_out"] += 1
                return result

        # All filters passed
        self.metrics["passed_through"] += 1
        return FilterResult(should_log=True, reason="all_filters_passed")

    def get_metrics(self) -> Dict[str, Any]:
        """Get filtering metrics"""
        total_evaluated = self.metrics.get("total_evaluated", 0)
        passed_through = self.metrics.get("passed_through", 0)
        filtered_out = self.metrics.get("filtered_out", 0)

        return {
            "summary": {
                "total_evaluated": total_evaluated,
                "passed_through": passed_through,
                "filtered_out": filtered_out,
            },
            "filter_stats": dict(self._filter_stats),
            "pass_rate": passed_through / max(1, total_evaluated),
        }

    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics.clear()
        self._filter_stats.clear()