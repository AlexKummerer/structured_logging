"""
Visualization components for monitoring dashboard

This module provides chart and visualization configurations for
displaying log data and metrics.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..logger import get_logger


class ChartType(Enum):
    """Available chart types"""

    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    TABLE = "table"


@dataclass
class ChartConfig:
    """Base configuration for charts"""

    title: str
    chart_type: ChartType
    width: Optional[int] = None
    height: Optional[int] = None
    refresh_interval: int = 5  # seconds
    theme: str = "light"  # light, dark

    # Data settings
    max_points: int = 1000
    aggregation: Optional[str] = None  # sum, avg, min, max

    # Display settings
    show_legend: bool = True
    show_grid: bool = True
    show_tooltip: bool = True

    # Custom styling
    colors: List[str] = field(default_factory=list)
    styles: Dict[str, Any] = field(default_factory=dict)


class Visualization(ABC):
    """Base class for visualizations"""

    def __init__(self, config: ChartConfig):
        self.config = config
        self.logger = get_logger("monitoring.visualization")

    @abstractmethod
    def prepare_data(self, raw_data: Any) -> Dict[str, Any]:
        """Prepare data for visualization"""
        pass

    @abstractmethod
    def get_chart_spec(self) -> Dict[str, Any]:
        """Get chart specification (e.g., for Chart.js, D3.js)"""
        pass


@dataclass
class LogChart(Visualization):
    """
    Time series chart for log volume

    Shows log count over time with level breakdown
    """

    def __init__(
        self,
        title: str = "Log Volume",
        time_window: int = 300,  # 5 minutes
        group_by_level: bool = True,
    ):
        config = ChartConfig(title=title, chart_type=ChartType.LINE, refresh_interval=5)
        super().__init__(config)

        self.time_window = time_window
        self.group_by_level = group_by_level

    def prepare_data(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare log data for time series chart"""
        # Group logs by time buckets
        buckets: Dict[str, Dict[str, int]] = {}

        for log in logs:
            timestamp = log.get("timestamp", datetime.now())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)

            # Round to nearest minute
            bucket_time = timestamp.replace(second=0, microsecond=0)
            bucket_key = bucket_time.isoformat()

            if bucket_key not in buckets:
                buckets[bucket_key] = {
                    "total": 0,
                    "error": 0,
                    "warning": 0,
                    "info": 0,
                    "debug": 0,
                }

            buckets[bucket_key]["total"] += 1

            if self.group_by_level:
                level = log.get("level", "info").lower()
                if level in buckets[bucket_key]:
                    buckets[bucket_key][level] += 1

        # Convert to chart data
        sorted_times = sorted(buckets.keys())

        datasets = []

        if self.group_by_level:
            for level in ["error", "warning", "info", "debug"]:
                datasets.append(
                    {
                        "label": level.capitalize(),
                        "data": [buckets[t].get(level, 0) for t in sorted_times],
                        "borderColor": self._get_level_color(level),
                        "backgroundColor": self._get_level_color(level, alpha=0.2),
                    }
                )
        else:
            datasets.append(
                {
                    "label": "Total Logs",
                    "data": [buckets[t]["total"] for t in sorted_times],
                    "borderColor": "rgb(75, 192, 192)",
                    "tension": 0.1,
                }
            )

        return {"labels": sorted_times, "datasets": datasets}

    def get_chart_spec(self) -> Dict[str, Any]:
        """Get Chart.js specification"""
        return {
            "type": self.config.chart_type.value,
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {"display": True, "text": self.config.title},
                    "legend": {"display": self.config.show_legend},
                },
                "scales": {
                    "x": {
                        "type": "time",
                        "time": {"displayFormats": {"minute": "HH:mm"}},
                        "title": {"display": True, "text": "Time"},
                    },
                    "y": {
                        "title": {"display": True, "text": "Log Count"},
                        "beginAtZero": True,
                    },
                },
            },
        }

    def _get_level_color(self, level: str, alpha: float = 1.0) -> str:
        """Get color for log level"""
        colors = {
            "error": f"rgba(255, 99, 132, {alpha})",
            "warning": f"rgba(255, 205, 86, {alpha})",
            "info": f"rgba(54, 162, 235, {alpha})",
            "debug": f"rgba(201, 203, 207, {alpha})",
        }
        return colors.get(level, f"rgba(75, 192, 192, {alpha})")


@dataclass
class MetricChart(Visualization):
    """
    Chart for displaying metrics over time

    Supports multiple metrics on the same chart
    """

    def __init__(
        self,
        title: str,
        metrics: List[str],
        chart_type: ChartType = ChartType.LINE,
        y_axis_label: str = "Value",
    ):
        config = ChartConfig(title=title, chart_type=chart_type, refresh_interval=10)
        super().__init__(config)

        self.metrics = metrics
        self.y_axis_label = y_axis_label

    def prepare_data(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare metrics data for chart"""
        datasets = []

        # Extract time series data for each metric
        time_series = metrics_data.get("time_series", {})

        for i, metric in enumerate(self.metrics):
            if metric in time_series:
                data = time_series[metric]
                datasets.append(
                    {
                        "label": metric.replace("_", " ").title(),
                        "data": [
                            {"x": ts, "y": val}
                            for ts, val in zip(data["timestamps"], data["values"])
                        ],
                        "borderColor": self._get_metric_color(i),
                        "backgroundColor": self._get_metric_color(i, alpha=0.2),
                        "tension": 0.1,
                    }
                )

        return {"datasets": datasets}

    def get_chart_spec(self) -> Dict[str, Any]:
        """Get Chart.js specification"""
        return {
            "type": self.config.chart_type.value,
            "options": {
                "responsive": True,
                "plugins": {"title": {"display": True, "text": self.config.title}},
                "scales": {
                    "x": {"type": "time", "title": {"display": True, "text": "Time"}},
                    "y": {"title": {"display": True, "text": self.y_axis_label}},
                },
            },
        }

    def _get_metric_color(self, index: int, alpha: float = 1.0) -> str:
        """Get color for metric by index"""
        colors = [
            f"rgba(75, 192, 192, {alpha})",
            f"rgba(255, 99, 132, {alpha})",
            f"rgba(54, 162, 235, {alpha})",
            f"rgba(255, 205, 86, {alpha})",
            f"rgba(153, 102, 255, {alpha})",
        ]
        return colors[index % len(colors)]


@dataclass
class ErrorHeatmap(Visualization):
    """
    Heatmap showing error patterns over time

    Useful for identifying error hotspots
    """

    def __init__(
        self,
        title: str = "Error Patterns",
        time_buckets: int = 24,  # hours
        error_categories: Optional[List[str]] = None,
    ):
        config = ChartConfig(
            title=title,
            chart_type=ChartType.HEATMAP,
            refresh_interval=60,  # Update less frequently
        )
        super().__init__(config)

        self.time_buckets = time_buckets
        self.error_categories = error_categories or [
            "database",
            "network",
            "authentication",
            "validation",
            "other",
        ]

    def prepare_data(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare error data for heatmap"""
        # Initialize grid
        grid = []
        for _ in range(len(self.error_categories)):
            grid.append([0] * self.time_buckets)

        # Categorize errors
        now = datetime.now()

        for log in logs:
            if log.get("level", "").lower() != "error":
                continue

            timestamp = log.get("timestamp", now)
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)

            # Calculate time bucket
            hours_ago = (now - timestamp).total_seconds() / 3600
            bucket = min(int(hours_ago), self.time_buckets - 1)

            if bucket < 0:
                continue

            # Categorize error
            category_index = self._categorize_error(log)
            if 0 <= category_index < len(self.error_categories):
                grid[category_index][self.time_buckets - 1 - bucket] += 1

        # Convert to heatmap data
        data_points = []
        for i, category in enumerate(self.error_categories):
            for j in range(self.time_buckets):
                if grid[i][j] > 0:
                    data_points.append({"x": j, "y": i, "value": grid[i][j]})

        # Generate time labels
        time_labels = []
        for i in range(self.time_buckets):
            hours_ago = self.time_buckets - i - 1
            label_time = now - timedelta(hours=hours_ago)
            time_labels.append(label_time.strftime("%H:00"))

        return {
            "datasets": [
                {
                    "label": "Errors",
                    "data": data_points,
                    "backgroundColor": "rgba(255, 99, 132, 0.8)",
                }
            ],
            "xLabels": time_labels,
            "yLabels": self.error_categories,
        }

    def get_chart_spec(self) -> Dict[str, Any]:
        """Get Chart.js specification for heatmap"""
        return {
            "type": "matrix",  # Requires chartjs-chart-matrix plugin
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {"display": True, "text": self.config.title},
                    "tooltip": {
                        "callbacks": {
                            "title": lambda: "",
                            "label": lambda ctx: f"Errors: {ctx.raw.value}",
                        }
                    },
                },
                "scales": {
                    "x": {
                        "type": "category",
                        "title": {"display": True, "text": "Time (Hours)"},
                    },
                    "y": {
                        "type": "category",
                        "title": {"display": True, "text": "Error Category"},
                    },
                },
            },
        }

    def _categorize_error(self, log: Dict[str, Any]) -> int:
        """Categorize error based on content"""
        message = str(log.get("message", "")).lower()
        error = str(log.get("error", "")).lower()
        combined = f"{message} {error}"

        # Simple keyword-based categorization
        if any(kw in combined for kw in ["database", "sql", "query", "connection"]):
            return 0  # database
        elif any(kw in combined for kw in ["network", "timeout", "socket", "http"]):
            return 1  # network
        elif any(kw in combined for kw in ["auth", "login", "permission", "forbidden"]):
            return 2  # authentication
        elif any(kw in combined for kw in ["validation", "invalid", "format", "parse"]):
            return 3  # validation
        else:
            return 4  # other


@dataclass
class GaugeChart(Visualization):
    """
    Gauge chart for displaying single metrics

    Good for KPIs like error rate, response time
    """

    def __init__(
        self,
        title: str,
        metric: str,
        min_value: float = 0,
        max_value: float = 100,
        thresholds: Optional[List[Tuple[float, str]]] = None,
    ):
        config = ChartConfig(
            title=title, chart_type=ChartType.GAUGE, refresh_interval=5
        )
        super().__init__(config)

        self.metric = metric
        self.min_value = min_value
        self.max_value = max_value
        self.thresholds = thresholds or [
            (0.7, "success"),
            (0.9, "warning"),
            (1.0, "danger"),
        ]

    def prepare_data(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare gauge data"""
        current = metrics_data.get("current", {})
        value = current.get(self.metric, 0)

        # Normalize to 0-1 range
        normalized = (value - self.min_value) / (self.max_value - self.min_value)
        normalized = max(0, min(1, normalized))

        return {
            "value": value,
            "normalized": normalized,
            "min": self.min_value,
            "max": self.max_value,
        }

    def get_chart_spec(self) -> Dict[str, Any]:
        """Get gauge specification"""
        return {
            "type": "doughnut",
            "options": {
                "responsive": True,
                "circumference": 180,
                "rotation": 270,
                "plugins": {"title": {"display": True, "text": self.config.title}},
            },
        }


def create_visualization(viz_type: str, **kwargs) -> Visualization:
    """
    Factory function for creating visualizations

    Args:
        viz_type: Type of visualization (log_chart, metric_chart, etc.)
        **kwargs: Visualization-specific parameters

    Returns:
        Visualization instance
    """
    if viz_type == "log_chart":
        return LogChart(**kwargs)
    elif viz_type == "metric_chart":
        return MetricChart(**kwargs)
    elif viz_type == "error_heatmap":
        return ErrorHeatmap(**kwargs)
    elif viz_type == "gauge":
        return GaugeChart(**kwargs)
    else:
        raise ValueError(f"Unknown visualization type: {viz_type}")
