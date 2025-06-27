"""
Configuration for log filtering system
"""

import logging
from dataclasses import dataclass, field
from typing import List

from .base import LogFilter
from .level_filter import LevelFilter
from .sampling_filter import SamplingFilter


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