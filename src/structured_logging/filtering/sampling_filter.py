"""
Sampling-based log filtering with various strategies
"""

import hashlib
import logging
import random
import time
from collections import defaultdict, deque
from typing import Any, Dict, Optional

from .base import FilterResult, LogFilter


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