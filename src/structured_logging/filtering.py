"""
Log filtering and sampling system for production-grade logging control
"""

import hashlib
import logging
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union


@dataclass
class FilterResult:
    """Result of log filtering operation"""

    should_log: bool
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LogFilter(ABC):
    """Abstract base class for log filters"""

    @abstractmethod
    def should_log(
        self, record: logging.LogRecord, context: Dict[str, Any]
    ) -> FilterResult:
        """Determine if log record should be processed"""
        pass


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


class SamplingFilter(LogFilter):
    """Sample logs based on various strategies"""

    def __init__(
        self,
        sample_rate: float = 1.0,
        strategy: str = "random",
        max_per_second: Optional[int] = None,
        burst_allowance: int = 10,
    ):
        self.sample_rate = max(0.0, min(1.0, sample_rate))
        self.strategy = strategy
        self.max_per_second = max_per_second
        self.burst_allowance = burst_allowance

        # Rate limiting state
        self._rate_limiter_state: Dict[str, deque] = defaultdict(deque)
        self._hash_sampling_state: Dict[str, int] = {}

    def should_log(
        self, record: logging.LogRecord, context: Dict[str, Any]
    ) -> FilterResult:
        # Apply rate limiting first if configured
        if self.max_per_second is not None:
            rate_result = self._apply_rate_limiting(record, context)
            if not rate_result.should_log:
                return rate_result

        # Apply sampling strategy
        if self.strategy == "random":
            return self._random_sampling(record, context)
        elif self.strategy == "hash":
            return self._hash_sampling(record, context)
        elif self.strategy == "level_based":
            return self._level_based_sampling(record, context)
        else:
            return FilterResult(
                should_log=True,
                reason=f"sampling_filter: unknown strategy {self.strategy}, defaulting to pass",
            )

    def _apply_rate_limiting(
        self, record: logging.LogRecord, context: Dict[str, Any]
    ) -> FilterResult:
        """Apply rate limiting based on max_per_second"""
        now = time.time()
        key = f"{record.name}:{record.levelname}"

        # Clean old entries (older than 1 second)
        times = self._rate_limiter_state[key]
        while times and times[0] < now - 1.0:
            times.popleft()

        # Check if we're within limits
        if len(times) < self.max_per_second:
            times.append(now)
            return FilterResult(should_log=True, reason="rate_limiter: within limits")

        # Apply burst allowance for high-priority logs
        if (
            record.levelno >= logging.ERROR
            and len(times) < self.max_per_second + self.burst_allowance
        ):
            times.append(now)
            return FilterResult(
                should_log=True,
                reason="rate_limiter: burst allowance for error",
                metadata={"burst_used": True},
            )

        return FilterResult(
            should_log=False,
            reason=f"rate_limiter: exceeded {self.max_per_second}/sec limit",
            metadata={"rate_limited": True},
        )

    def _random_sampling(
        self, record: logging.LogRecord, context: Dict[str, Any]
    ) -> FilterResult:
        """Random sampling based on sample_rate"""
        should_log = random.random() < self.sample_rate
        return FilterResult(
            should_log=should_log,
            reason=f"random_sampling: {self.sample_rate * 100}% rate",
        )

    def _hash_sampling(
        self, record: logging.LogRecord, context: Dict[str, Any]
    ) -> FilterResult:
        """Deterministic hash-based sampling"""
        # Create hash key from request_id or message
        hash_key = context.get("request_id") or record.getMessage()
        hash_value = int(hashlib.md5(hash_key.encode()).hexdigest()[:8], 16)
        threshold = int(self.sample_rate * 0xFFFFFFFF)

        should_log = hash_value < threshold
        return FilterResult(
            should_log=should_log,
            reason=f"hash_sampling: {self.sample_rate * 100}% deterministic rate",
        )

    def _level_based_sampling(
        self, record: logging.LogRecord, context: Dict[str, Any]
    ) -> FilterResult:
        """Sample based on log level - higher levels get higher rates"""
        level_rates = {
            logging.DEBUG: self.sample_rate * 0.1,
            logging.INFO: self.sample_rate * 0.3,
            logging.WARNING: self.sample_rate * 0.7,
            logging.ERROR: self.sample_rate * 0.9,
            logging.CRITICAL: 1.0,  # Always log critical
        }

        effective_rate = level_rates.get(record.levelno, self.sample_rate)
        should_log = random.random() < effective_rate

        return FilterResult(
            should_log=should_log,
            reason=f"level_sampling: {effective_rate * 100}% for {record.levelname}",
        )


@dataclass
class FilterConfig:
    """Configuration for log filtering system"""

    enabled: bool = True
    filters: List[LogFilter] = field(default_factory=list)
    default_action: str = "log"  # "log" or "drop"
    collect_metrics: bool = True

    @classmethod
    def create_production_config(
        cls, sample_rate: float = 0.1, max_logs_per_second: int = 1000
    ) -> "FilterConfig":
        """Create a production-ready filter configuration"""
        return cls(
            enabled=True,
            filters=[
                LevelFilter(min_level=logging.INFO),
                SamplingFilter(
                    sample_rate=sample_rate,
                    strategy="level_based",
                    max_per_second=max_logs_per_second,
                    burst_allowance=50,
                ),
            ],
            collect_metrics=True,
        )

    @classmethod
    def create_debug_config(cls) -> "FilterConfig":
        """Create a debug-friendly filter configuration"""
        return cls(
            enabled=True,
            filters=[LevelFilter(min_level=logging.DEBUG)],
            collect_metrics=False,
        )


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
