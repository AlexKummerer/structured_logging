"""
Base classes for log filtering system
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


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