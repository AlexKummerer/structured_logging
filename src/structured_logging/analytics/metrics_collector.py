"""
Performance metrics collection for structured logs

This module provides comprehensive metrics collection and analysis capabilities
for log data, including performance metrics, error rates, and throughput analysis.
"""

import statistics
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from ..logger import get_logger


@dataclass
class MetricsConfig:
    """Configuration for metrics collection"""

    # Metrics to collect
    collect_performance_metrics: bool = True  # Response times, latencies
    collect_error_metrics: bool = True  # Error rates and types
    collect_throughput_metrics: bool = True  # Request rates, volumes
    collect_user_metrics: bool = True  # User activity patterns

    # Time windows
    metrics_window_minutes: int = 60  # Rolling window for metrics
    time_bucket_minutes: int = 5  # Granularity of time buckets

    # Performance thresholds
    slow_request_threshold: float = 1.0  # Seconds
    error_rate_threshold: float = 0.05  # 5% error rate

    # Fields to track
    latency_fields: List[str] = field(
        default_factory=lambda: [
            "response_time",
            "duration",
            "elapsed_time",
            "processing_time",
            "query_time",
        ]
    )
    error_fields: List[str] = field(
        default_factory=lambda: ["error", "exception", "error_type", "status_code"]
    )
    identifier_fields: List[str] = field(
        default_factory=lambda: ["user_id", "session_id", "endpoint", "operation"]
    )

    # Percentiles to calculate
    percentiles: List[int] = field(default_factory=lambda: [50, 75, 90, 95, 99])

    # Output settings
    include_histogram: bool = True  # Include distribution histograms
    include_trends: bool = True  # Include trend analysis
    max_unique_values: int = 100  # Max unique values to track per field


@dataclass
class PerformanceMetrics:
    """Performance metrics for a time period"""

    count: int
    mean: float
    median: float
    min: float
    max: float
    std_dev: float
    percentiles: Dict[int, float]
    slow_count: int = 0
    slow_percentage: float = 0.0

    @property
    def p95(self) -> float:
        """95th percentile shortcut"""
        return self.percentiles.get(95, 0.0)

    @property
    def p99(self) -> float:
        """99th percentile shortcut"""
        return self.percentiles.get(99, 0.0)


@dataclass
class ErrorMetrics:
    """Error metrics for a time period"""

    total_count: int
    error_count: int
    error_rate: float
    error_types: Counter
    status_codes: Counter
    error_messages: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        return 1.0 - self.error_rate if self.total_count > 0 else 1.0


@dataclass
class ThroughputMetrics:
    """Throughput metrics for a time period"""

    total_requests: int
    time_span_minutes: float
    requests_per_minute: float
    requests_per_second: float
    peak_rpm: float
    min_rpm: float
    bytes_processed: int = 0
    unique_users: int = 0
    unique_sessions: int = 0


@dataclass
class UserMetrics:
    """User activity metrics"""

    active_users: int
    new_users: int
    returning_users: int
    user_sessions: Dict[str, int]  # user_id -> session count
    top_users: List[Tuple[str, int]]  # (user_id, request_count)
    user_errors: Dict[str, int]  # user_id -> error count


@dataclass
class MetricsSummary:
    """Complete metrics summary for a time period"""

    start_time: datetime
    end_time: datetime
    total_logs: int
    performance: Optional[PerformanceMetrics] = None
    errors: Optional[ErrorMetrics] = None
    throughput: Optional[ThroughputMetrics] = None
    users: Optional[UserMetrics] = None
    endpoint_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    time_series: Dict[str, List[Tuple[datetime, float]]] = field(default_factory=dict)

    @property
    def duration(self) -> timedelta:
        """Duration of metrics period"""
        return self.end_time - self.start_time

    @property
    def health_score(self) -> float:
        """Calculate overall health score (0-100)"""
        score = 100.0

        # Deduct for high error rate
        if self.errors and self.errors.error_rate > 0.01:
            score -= min(self.errors.error_rate * 100, 30)

        # Deduct for slow requests
        if self.performance and self.performance.slow_percentage > 0.1:
            score -= min(self.performance.slow_percentage * 50, 20)

        # Deduct for high latency
        if self.performance and self.performance.p95 > 2.0:
            score -= min((self.performance.p95 - 2.0) * 10, 20)

        return max(score, 0.0)


class MetricsCollector:
    """
    Collects and analyzes performance metrics from structured logs

    Provides comprehensive metrics including:
    - Performance metrics (latency, response times)
    - Error metrics (rates, types, patterns)
    - Throughput metrics (requests per minute/second)
    - User activity metrics
    """

    def __init__(self, config: Optional[MetricsConfig] = None):
        self.config = config or MetricsConfig()
        self.logger = get_logger("analytics.metrics_collector")

        # Metrics storage
        self._performance_data: Dict[str, List[float]] = defaultdict(list)
        self._error_data: List[Dict[str, Any]] = []
        self._throughput_data: deque = deque(maxlen=1000)
        self._user_data: Dict[str, Set[str]] = defaultdict(set)

        # Time series data
        self._time_series: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

    def collect_metrics(
        self, logs: List[Dict[str, Any]], time_window: Optional[timedelta] = None
    ) -> MetricsSummary:
        """
        Collect metrics from logs

        Args:
            logs: List of structured log entries
            time_window: Optional time window to analyze

        Returns:
            Comprehensive metrics summary
        """
        if not logs:
            return self._create_empty_summary()

        # Filter by time window if specified
        if time_window:
            logs = self._filter_by_time_window(logs, time_window)

        # Determine time range
        timestamps = [log.get("timestamp", datetime.now()) for log in logs]
        start_time = min(timestamps)
        end_time = max(timestamps)

        # Initialize summary
        summary = MetricsSummary(
            start_time=start_time, end_time=end_time, total_logs=len(logs)
        )

        # Collect different metrics
        if self.config.collect_performance_metrics:
            summary.performance = self._collect_performance_metrics(logs)

        if self.config.collect_error_metrics:
            summary.errors = self._collect_error_metrics(logs)

        if self.config.collect_throughput_metrics:
            summary.throughput = self._collect_throughput_metrics(logs)

        if self.config.collect_user_metrics:
            summary.users = self._collect_user_metrics(logs)

        # Collect endpoint-specific metrics
        summary.endpoint_metrics = self._collect_endpoint_metrics(logs)

        # Generate time series data
        if self.config.include_trends:
            summary.time_series = self._generate_time_series(logs)

        self.logger.info(
            f"Collected metrics for {len(logs)} logs over {summary.duration}",
            extra={
                "log_count": len(logs),
                "duration_minutes": summary.duration.total_seconds() / 60,
                "health_score": summary.health_score,
            },
        )

        return summary

    def _collect_performance_metrics(
        self, logs: List[Dict[str, Any]]
    ) -> PerformanceMetrics:
        """Collect performance-related metrics"""
        latencies = []

        for log in logs:
            # Extract latency values
            for field_name in self.config.latency_fields:
                value = log.get(field_name)
                if isinstance(value, (int, float)) and value >= 0:
                    latencies.append(float(value))
                    break  # Use first valid latency field

        if not latencies:
            return self._create_empty_performance_metrics()

        # Calculate statistics
        slow_count = sum(
            1 for lat in latencies if lat > self.config.slow_request_threshold
        )

        # Calculate percentiles
        percentiles = {}
        for p in self.config.percentiles:
            percentiles[p] = self._percentile(latencies, p)

        return PerformanceMetrics(
            count=len(latencies),
            mean=statistics.mean(latencies),
            median=statistics.median(latencies),
            min=min(latencies),
            max=max(latencies),
            std_dev=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            percentiles=percentiles,
            slow_count=slow_count,
            slow_percentage=slow_count / len(latencies) if latencies else 0.0,
        )

    def _collect_error_metrics(self, logs: List[Dict[str, Any]]) -> ErrorMetrics:
        """Collect error-related metrics"""
        error_count = 0
        error_types = Counter()
        status_codes = Counter()
        error_messages = []

        for log in logs:
            is_error = False

            # Check log level
            if log.get("level", "").lower() in ["error", "critical", "fatal"]:
                is_error = True

            # Check error fields
            for field_name in self.config.error_fields:
                if field_name in log and log[field_name]:
                    is_error = True
                    if field_name == "error_type":
                        error_types[log[field_name]] += 1
                    break

            # Check status codes
            status_code = log.get("status_code")
            if status_code:
                status_codes[status_code] += 1
                if isinstance(status_code, int) and status_code >= 400:
                    is_error = True

            if is_error:
                error_count += 1
                # Collect error message samples
                message = log.get("message", "")
                if message and len(error_messages) < 10:
                    error_messages.append(message[:200])  # Truncate long messages

        error_rate = error_count / len(logs) if logs else 0.0

        return ErrorMetrics(
            total_count=len(logs),
            error_count=error_count,
            error_rate=error_rate,
            error_types=error_types,
            status_codes=status_codes,
            error_messages=error_messages,
        )

    def _collect_throughput_metrics(
        self, logs: List[Dict[str, Any]]
    ) -> ThroughputMetrics:
        """Collect throughput-related metrics"""
        if not logs:
            return self._create_empty_throughput_metrics()

        # Time-based grouping
        time_buckets = self._group_by_time_buckets(logs)

        # Calculate rates
        bucket_counts = [len(bucket_logs) for bucket_logs in time_buckets.values()]

        # Time span
        timestamps = [log.get("timestamp", datetime.now()) for log in logs]
        time_span = (max(timestamps) - min(timestamps)).total_seconds() / 60
        time_span = max(time_span, 1.0)  # Avoid division by zero

        # Unique users and sessions
        unique_users = set()
        unique_sessions = set()
        bytes_processed = 0

        for log in logs:
            user_id = log.get("user_id")
            if user_id:
                unique_users.add(user_id)

            session_id = log.get("session_id")
            if session_id:
                unique_sessions.add(session_id)

            # Estimate bytes (simplified)
            bytes_processed += len(str(log))

        requests_per_minute = len(logs) / time_span

        return ThroughputMetrics(
            total_requests=len(logs),
            time_span_minutes=time_span,
            requests_per_minute=requests_per_minute,
            requests_per_second=requests_per_minute / 60,
            peak_rpm=(
                max(bucket_counts) * (60 / self.config.time_bucket_minutes)
                if bucket_counts
                else 0
            ),
            min_rpm=(
                min(bucket_counts) * (60 / self.config.time_bucket_minutes)
                if bucket_counts
                else 0
            ),
            bytes_processed=bytes_processed,
            unique_users=len(unique_users),
            unique_sessions=len(unique_sessions),
        )

    def _collect_user_metrics(self, logs: List[Dict[str, Any]]) -> UserMetrics:
        """Collect user activity metrics"""
        user_requests = Counter()
        user_sessions = defaultdict(set)
        user_errors = Counter()
        user_first_seen = {}

        for log in logs:
            user_id = log.get("user_id")
            if not user_id:
                continue

            user_requests[user_id] += 1

            # Track sessions
            session_id = log.get("session_id")
            if session_id:
                user_sessions[user_id].add(session_id)

            # Track errors
            if log.get("level", "").lower() in ["error", "critical"]:
                user_errors[user_id] += 1

            # Track first seen
            timestamp = log.get("timestamp", datetime.now())
            if user_id not in user_first_seen:
                user_first_seen[user_id] = timestamp
            else:
                user_first_seen[user_id] = min(user_first_seen[user_id], timestamp)

        # Determine new vs returning users (simplified)
        cutoff_time = datetime.now() - timedelta(hours=1)
        new_users = sum(1 for t in user_first_seen.values() if t > cutoff_time)

        return UserMetrics(
            active_users=len(user_requests),
            new_users=new_users,
            returning_users=len(user_requests) - new_users,
            user_sessions={u: len(sessions) for u, sessions in user_sessions.items()},
            top_users=user_requests.most_common(10),
            user_errors=dict(user_errors),
        )

    def _collect_endpoint_metrics(
        self, logs: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Collect per-endpoint metrics"""
        endpoint_data = defaultdict(
            lambda: {
                "count": 0,
                "errors": 0,
                "latencies": [],
                "status_codes": Counter(),
            }
        )

        for log in logs:
            endpoint = log.get("endpoint") or log.get("path") or log.get("operation")
            if not endpoint:
                continue

            data = endpoint_data[endpoint]
            data["count"] += 1

            # Check for errors
            if log.get("level", "").lower() in ["error", "critical"]:
                data["errors"] += 1

            # Collect latency
            for field_name in self.config.latency_fields:
                value = log.get(field_name)
                if isinstance(value, (int, float)) and value >= 0:
                    data["latencies"].append(float(value))
                    break

            # Status codes
            status_code = log.get("status_code")
            if status_code:
                data["status_codes"][status_code] += 1

        # Calculate endpoint statistics
        endpoint_metrics = {}
        for endpoint, data in endpoint_data.items():
            if len(endpoint_metrics) >= self.config.max_unique_values:
                break

            metrics = {
                "count": data["count"],
                "error_rate": (
                    data["errors"] / data["count"] if data["count"] > 0 else 0.0
                ),
                "status_codes": dict(data["status_codes"]),
            }

            if data["latencies"]:
                metrics.update(
                    {
                        "avg_latency": statistics.mean(data["latencies"]),
                        "p95_latency": self._percentile(data["latencies"], 95),
                        "max_latency": max(data["latencies"]),
                    }
                )

            endpoint_metrics[endpoint] = metrics

        return endpoint_metrics

    def _generate_time_series(
        self, logs: List[Dict[str, Any]]
    ) -> Dict[str, List[Tuple[datetime, float]]]:
        """Generate time series data for trending"""
        time_series = defaultdict(list)

        # Group logs by time buckets
        time_buckets = self._group_by_time_buckets(logs)

        for bucket_key, bucket_logs in sorted(time_buckets.items()):
            # Parse bucket time
            bucket_time = datetime.strptime(bucket_key, "%Y-%m-%d %H:%M")

            # Request rate
            rpm = len(bucket_logs) * (60 / self.config.time_bucket_minutes)
            time_series["requests_per_minute"].append((bucket_time, rpm))

            # Error rate
            error_count = sum(
                1
                for log in bucket_logs
                if log.get("level", "").lower() in ["error", "critical"]
            )
            error_rate = error_count / len(bucket_logs) if bucket_logs else 0.0
            time_series["error_rate"].append((bucket_time, error_rate))

            # Average latency
            latencies = []
            for log in bucket_logs:
                for field_name in self.config.latency_fields:
                    value = log.get(field_name)
                    if isinstance(value, (int, float)) and value >= 0:
                        latencies.append(float(value))
                        break

            if latencies:
                avg_latency = statistics.mean(latencies)
                time_series["avg_latency"].append((bucket_time, avg_latency))

        return dict(time_series)

    def _group_by_time_buckets(
        self, logs: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group logs into time buckets"""
        buckets = defaultdict(list)

        for log in logs:
            timestamp = log.get("timestamp", datetime.now())
            bucket_time = timestamp.replace(
                minute=(timestamp.minute // self.config.time_bucket_minutes)
                * self.config.time_bucket_minutes,
                second=0,
                microsecond=0,
            )
            bucket_key = bucket_time.strftime("%Y-%m-%d %H:%M")
            buckets[bucket_key].append(log)

        return dict(buckets)

    def _filter_by_time_window(
        self, logs: List[Dict[str, Any]], time_window: timedelta
    ) -> List[Dict[str, Any]]:
        """Filter logs by time window"""
        if not logs:
            return logs

        latest_time = max(log.get("timestamp", datetime.now()) for log in logs)
        cutoff_time = latest_time - time_window

        return [
            log for log in logs if log.get("timestamp", datetime.now()) >= cutoff_time
        ]

    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = (len(sorted_values) - 1) * percentile / 100
        lower = sorted_values[int(index)]
        upper = sorted_values[min(int(index) + 1, len(sorted_values) - 1)]
        return lower + (upper - lower) * (index - int(index))

    def _create_empty_summary(self) -> MetricsSummary:
        """Create empty metrics summary"""
        now = datetime.now()
        return MetricsSummary(start_time=now, end_time=now, total_logs=0)

    def _create_empty_performance_metrics(self) -> PerformanceMetrics:
        """Create empty performance metrics"""
        return PerformanceMetrics(
            count=0,
            mean=0.0,
            median=0.0,
            min=0.0,
            max=0.0,
            std_dev=0.0,
            percentiles={p: 0.0 for p in self.config.percentiles},
        )

    def _create_empty_throughput_metrics(self) -> ThroughputMetrics:
        """Create empty throughput metrics"""
        return ThroughputMetrics(
            total_requests=0,
            time_span_minutes=0.0,
            requests_per_minute=0.0,
            requests_per_second=0.0,
            peak_rpm=0.0,
            min_rpm=0.0,
        )

    def generate_report(self, summary: MetricsSummary) -> str:
        """
        Generate a human-readable metrics report

        Args:
            summary: Metrics summary to report on

        Returns:
            Formatted report string
        """
        lines = [
            "=== Metrics Report ===",
            f"Period: {summary.start_time.strftime('%Y-%m-%d %H:%M')} - "
            f"{summary.end_time.strftime('%H:%M')}",
            f"Duration: {summary.duration}",
            f"Total Logs: {summary.total_logs:,}",
            f"Health Score: {summary.health_score:.1f}/100",
            "",
        ]

        # Performance metrics
        if summary.performance:
            perf = summary.performance
            lines.extend(
                [
                    "Performance Metrics:",
                    f"  Requests: {perf.count:,}",
                    f"  Mean Latency: {perf.mean:.3f}s",
                    f"  Median Latency: {perf.median:.3f}s",
                    f"  95th Percentile: {perf.p95:.3f}s",
                    f"  99th Percentile: {perf.p99:.3f}s",
                    f"  Min/Max: {perf.min:.3f}s / {perf.max:.3f}s",
                    f"  Slow Requests: {perf.slow_count:,} "
                    f"({perf.slow_percentage:.1%})",
                    "",
                ]
            )

        # Error metrics
        if summary.errors:
            err = summary.errors
            lines.extend(
                [
                    "Error Metrics:",
                    f"  Total Errors: {err.error_count:,} / {err.total_count:,}",
                    f"  Error Rate: {err.error_rate:.2%}",
                    f"  Success Rate: {err.success_rate:.2%}",
                ]
            )

            if err.error_types:
                lines.append("  Top Error Types:")
                for error_type, count in err.error_types.most_common(5):
                    lines.append(f"    - {error_type}: {count}")
            lines.append("")

        # Throughput metrics
        if summary.throughput:
            thr = summary.throughput
            lines.extend(
                [
                    "Throughput Metrics:",
                    f"  Total Requests: {thr.total_requests:,}",
                    f"  Avg Rate: {thr.requests_per_minute:.1f} rpm "
                    f"({thr.requests_per_second:.1f} rps)",
                    f"  Peak/Min Rate: {thr.peak_rpm:.1f} / {thr.min_rpm:.1f} rpm",
                    f"  Unique Users: {thr.unique_users:,}",
                    f"  Unique Sessions: {thr.unique_sessions:,}",
                    "",
                ]
            )

        # User metrics
        if summary.users:
            usr = summary.users
            lines.extend(
                [
                    "User Metrics:",
                    f"  Active Users: {usr.active_users:,}",
                    f"  New Users: {usr.new_users:,}",
                    f"  Returning Users: {usr.returning_users:,}",
                ]
            )

            if usr.top_users:
                lines.append("  Top Users:")
                for user_id, count in usr.top_users[:5]:
                    lines.append(f"    - {user_id}: {count:,} requests")
            lines.append("")

        # Top endpoints
        if summary.endpoint_metrics:
            lines.append("Top Endpoints:")
            sorted_endpoints = sorted(
                summary.endpoint_metrics.items(),
                key=lambda x: x[1]["count"],
                reverse=True,
            )

            for endpoint, metrics in sorted_endpoints[:5]:
                avg_latency = metrics.get("avg_latency", 0)
                error_rate = metrics.get("error_rate", 0)
                lines.append(
                    f"  - {endpoint}: {metrics['count']:,} requests, "
                    f"{avg_latency:.3f}s avg, {error_rate:.1%} errors"
                )

        return "\n".join(lines)


def collect_metrics(
    logs: List[Dict[str, Any]],
    config: Optional[MetricsConfig] = None,
    time_window: Optional[timedelta] = None,
) -> MetricsSummary:
    """
    Convenience function to collect metrics from logs

    Args:
        logs: List of structured log entries
        config: Optional configuration
        time_window: Optional time window to analyze

    Returns:
        Comprehensive metrics summary
    """
    collector = MetricsCollector(config)
    return collector.collect_metrics(logs, time_window)


def create_metrics_collector(
    config: Optional[MetricsConfig] = None,
) -> MetricsCollector:
    """
    Create a metrics collector instance

    Args:
        config: Optional configuration

    Returns:
        MetricsCollector instance
    """
    return MetricsCollector(config)
