"""
Real-time monitoring dashboard for structured logging

This package provides a web-based dashboard for monitoring log streams
in real-time with metrics, alerts, and visualizations.
"""

from .alerts import (
    AlertLevel,
    AlertManager,
    AlertRule,
    create_alert_rule,
)
from .dashboard import (
    DashboardConfig,
    DashboardServer,
    create_dashboard,
)
from .metrics import (
    MetricsAggregator,
    MetricsSnapshot,
    TimeSeriesMetric,
)
from .visualizations import (
    ErrorHeatmap,
    LogChart,
    MetricChart,
    create_visualization,
)

__all__ = [
    # Dashboard
    "DashboardServer",
    "DashboardConfig",
    "create_dashboard",
    # Metrics
    "MetricsAggregator",
    "MetricsSnapshot",
    "TimeSeriesMetric",
    # Alerts
    "AlertManager",
    "AlertRule",
    "AlertLevel",
    "create_alert_rule",
    # Visualizations
    "LogChart",
    "MetricChart",
    "ErrorHeatmap",
    "create_visualization",
]
