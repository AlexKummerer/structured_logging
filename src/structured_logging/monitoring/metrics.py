"""
Metrics aggregation for monitoring dashboard

This module provides real-time metrics collection and aggregation
for log streams.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

from ..logger import get_logger


@dataclass
class TimeSeriesMetric:
    """Time series metric data"""

    name: str
    timestamps: List[datetime] = field(default_factory=list)
    values: List[float] = field(default_factory=list)

    def add_point(self, timestamp: datetime, value: float) -> None:
        """Add a data point"""
        self.timestamps.append(timestamp)
        self.values.append(value)

    def get_range(
        self, start: datetime, end: datetime
    ) -> Tuple[List[datetime], List[float]]:
        """Get data points in time range"""
        timestamps = []
        values = []

        for i, ts in enumerate(self.timestamps):
            if start <= ts <= end:
                timestamps.append(ts)
                values.append(self.values[i])

        return timestamps, values

    def cleanup(self, cutoff: datetime) -> None:
        """Remove old data points"""
        indices_to_keep = [i for i, ts in enumerate(self.timestamps) if ts >= cutoff]

        self.timestamps = [self.timestamps[i] for i in indices_to_keep]
        self.values = [self.values[i] for i in indices_to_keep]


@dataclass
class MetricsSnapshot:
    """Snapshot of current metrics"""

    timestamp: datetime
    total_logs: int = 0
    error_count: int = 0
    warning_count: int = 0
    error_rate: float = 0.0
    avg_response_time: float = 0.0
    logs_per_second: float = 0.0
    active_alerts: int = 0
    custom_metrics: Dict[str, float] = field(default_factory=dict)


class MetricsAggregator:
    """
    Aggregates metrics from log streams

    Tracks:
    - Log volume and rates
    - Error and warning rates
    - Response time percentiles
    - Custom metrics from logs
    - Time series data
    """

    def __init__(self, retention_minutes: int = 60):
        self.retention_minutes = retention_minutes
        self.logger = get_logger("monitoring.metrics")

        # Counters
        self._total_logs = 0
        self._error_count = 0
        self._warning_count = 0

        # Time series metrics
        self._time_series: Dict[str, TimeSeriesMetric] = {}
        self._log_rate = deque(maxlen=300)  # 5 minutes at 1s intervals
        self._error_rate = deque(maxlen=300)
        self._response_times = deque(maxlen=1000)

        # Window metrics
        self._window_logs: Dict[str, int] = defaultdict(int)
        self._window_errors: Dict[str, int] = defaultdict(int)

        # Custom metrics
        self._custom_metrics: Dict[str, deque] = {}

        # Last update
        self._last_update = datetime.now()
        self._last_second_logs = 0
        self._last_second_errors = 0

    async def update(self, log: Dict[str, Any]) -> None:
        """Update metrics with new log entry"""
        self._total_logs += 1

        # Update counters
        level = log.get("level", "info").lower()
        if level == "error":
            self._error_count += 1
            self._last_second_errors += 1
        elif level == "warning":
            self._warning_count += 1

        # Extract response time
        response_time = log.get("response_time") or log.get("duration")
        if response_time is not None:
            try:
                self._response_times.append(float(response_time))
            except (ValueError, TypeError):
                pass

        # Extract custom metrics
        for key, value in log.items():
            if key.endswith("_metric") or key in [
                "latency",
                "duration",
                "size",
                "count",
            ]:
                try:
                    metric_value = float(value)
                    self._add_custom_metric(key, metric_value)
                except (ValueError, TypeError):
                    pass

        # Update time window metrics
        timestamp = log.get("timestamp", datetime.now())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        window_key = self._get_window_key(timestamp)
        self._window_logs[window_key] += 1
        if level == "error":
            self._window_errors[window_key] += 1

        # Update rates every second
        now = datetime.now()
        if (now - self._last_update).total_seconds() >= 1:
            self._update_rates(now)

    def _get_window_key(self, timestamp: datetime) -> str:
        """Get minute window key for timestamp"""
        return timestamp.strftime("%Y-%m-%d %H:%M")

    def _update_rates(self, now: datetime) -> None:
        """Update rate metrics"""
        # Log rate
        self._log_rate.append(self._last_second_logs)
        self._last_second_logs = 0

        # Error rate
        self._error_rate.append(self._last_second_errors)
        self._last_second_errors = 0

        self._last_update = now

        # Cleanup old data
        self._cleanup_old_data()

    def _cleanup_old_data(self) -> None:
        """Remove old metric data"""
        cutoff = datetime.now() - timedelta(minutes=self.retention_minutes)
        cutoff_key = self._get_window_key(cutoff)

        # Cleanup window metrics
        old_keys = [key for key in self._window_logs.keys() if key < cutoff_key]
        for key in old_keys:
            del self._window_logs[key]
            self._window_errors.pop(key, None)

        # Cleanup time series
        for metric in self._time_series.values():
            metric.cleanup(cutoff)

    def _add_custom_metric(self, name: str, value: float) -> None:
        """Add custom metric value"""
        if name not in self._custom_metrics:
            self._custom_metrics[name] = deque(maxlen=1000)

        self._custom_metrics[name].append(value)

        # Add to time series
        if name not in self._time_series:
            self._time_series[name] = TimeSeriesMetric(name)

        self._time_series[name].add_point(datetime.now(), value)

    async def get_current_snapshot(self) -> Dict[str, Any]:
        """Get current metrics snapshot"""
        # Calculate rates
        log_rate = sum(self._log_rate) / max(len(self._log_rate), 1)
        error_rate = (
            (sum(self._error_rate) / max(sum(self._log_rate), 1)) * 100
            if self._log_rate
            else 0
        )

        # Calculate response time stats
        avg_response_time = 0
        p95_response_time = 0
        p99_response_time = 0

        if self._response_times:
            sorted_times = sorted(self._response_times)
            avg_response_time = sum(sorted_times) / len(sorted_times)
            p95_index = int(len(sorted_times) * 0.95)
            p99_index = int(len(sorted_times) * 0.99)
            p95_response_time = (
                sorted_times[p95_index] if p95_index < len(sorted_times) else 0
            )
            p99_response_time = (
                sorted_times[p99_index] if p99_index < len(sorted_times) else 0
            )

        # Custom metrics averages
        custom_metrics = {}
        for name, values in self._custom_metrics.items():
            if values:
                custom_metrics[name] = {
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

        return {
            "timestamp": datetime.now().isoformat(),
            "total_logs": self._total_logs,
            "error_count": self._error_count,
            "warning_count": self._warning_count,
            "error_rate": error_rate,
            "logs_per_second": log_rate,
            "avg_response_time": avg_response_time,
            "p95_response_time": p95_response_time,
            "p99_response_time": p99_response_time,
            "custom_metrics": custom_metrics,
        }

    async def get_metrics(self, time_range: str) -> Dict[str, Any]:
        """
        Get metrics for time range

        Args:
            time_range: Time range string (e.g., "5m", "1h", "24h")

        Returns:
            Metrics data with time series
        """
        # Parse time range
        minutes = self._parse_time_range(time_range)
        start_time = datetime.now() - timedelta(minutes=minutes)
        end_time = datetime.now()

        # Get window metrics
        window_data = self._get_window_metrics(start_time, end_time)

        # Get time series data
        time_series = {}
        for name, metric in self._time_series.items():
            timestamps, values = metric.get_range(start_time, end_time)
            if timestamps:
                time_series[name] = {
                    "timestamps": [ts.isoformat() for ts in timestamps],
                    "values": values,
                }

        # Get current snapshot
        snapshot = await self.get_current_snapshot()

        return {
            "range": time_range,
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "current": snapshot,
            "windows": window_data,
            "time_series": time_series,
        }

    def _parse_time_range(self, time_range: str) -> int:
        """Parse time range string to minutes"""
        if time_range.endswith("m"):
            return int(time_range[:-1])
        elif time_range.endswith("h"):
            return int(time_range[:-1]) * 60
        elif time_range.endswith("d"):
            return int(time_range[:-1]) * 60 * 24
        else:
            return 5  # Default 5 minutes

    def _get_window_metrics(
        self, start_time: datetime, end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get metrics aggregated by time window"""
        start_key = self._get_window_key(start_time)
        end_key = self._get_window_key(end_time)

        windows = []
        for key in sorted(self._window_logs.keys()):
            if start_key <= key <= end_key:
                log_count = self._window_logs[key]
                error_count = self._window_errors.get(key, 0)

                windows.append(
                    {
                        "window": key,
                        "logs": log_count,
                        "errors": error_count,
                        "error_rate": (
                            (error_count / log_count * 100) if log_count > 0 else 0
                        ),
                    }
                )

        return windows

    def get_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile value"""
        if not values:
            return 0

        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)

        if index >= len(sorted_values):
            return sorted_values[-1]

        return sorted_values[index]

    def reset(self) -> None:
        """Reset all metrics"""
        self._total_logs = 0
        self._error_count = 0
        self._warning_count = 0
        self._time_series.clear()
        self._log_rate.clear()
        self._error_rate.clear()
        self._response_times.clear()
        self._window_logs.clear()
        self._window_errors.clear()
        self._custom_metrics.clear()
        self._last_second_logs = 0
        self._last_second_errors = 0

        self.logger.info("Metrics reset")
