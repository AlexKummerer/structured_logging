"""
Advanced analytics for structured logging

This package provides log analysis capabilities including pattern detection,
anomaly identification, and performance metrics collection.
"""

from .anomaly_detector import (
    AnomalyDetector,
    AnomalyDetectorConfig,
    AnomalyScore,
    LogAnomaly,
    create_anomaly_detector,
    detect_anomalies,
)
from .metrics_collector import (
    ErrorMetrics,
    MetricsCollector,
    MetricsConfig,
    MetricsSummary,
    PerformanceMetrics,
    ThroughputMetrics,
    UserMetrics,
    collect_metrics,
    create_metrics_collector,
)
from .pattern_detector import (
    LogPattern,
    PatternDetector,
    PatternDetectorConfig,
    PatternMatch,
    create_pattern_detector,
    detect_patterns,
)

__all__ = [
    # Pattern detection
    "PatternDetector",
    "PatternDetectorConfig",
    "LogPattern",
    "PatternMatch",
    "detect_patterns",
    "create_pattern_detector",
    # Anomaly detection
    "AnomalyDetector",
    "AnomalyDetectorConfig",
    "LogAnomaly",
    "AnomalyScore",
    "detect_anomalies",
    "create_anomaly_detector",
    # Metrics collection
    "MetricsCollector",
    "MetricsConfig",
    "PerformanceMetrics",
    "ErrorMetrics",
    "ThroughputMetrics",
    "UserMetrics",
    "MetricsSummary",
    "collect_metrics",
    "create_metrics_collector",
]
