"""
Tests for metrics collection functionality
"""

from collections import Counter
from datetime import datetime, timedelta

import pytest

from structured_logging.analytics.metrics_collector import (
    ErrorMetrics,
    MetricsCollector,
    MetricsConfig,
    MetricsSummary,
    PerformanceMetrics,
    collect_metrics,
    create_metrics_collector,
)


class TestMetricsCollector:
    """Test the MetricsCollector class"""

    def test_init(self):
        """Test collector initialization"""
        # Default config
        collector = MetricsCollector()
        assert collector.config.collect_performance_metrics is True
        assert collector.config.metrics_window_minutes == 60

        # Custom config
        config = MetricsConfig(collect_error_metrics=False, time_bucket_minutes=10)
        collector = MetricsCollector(config)
        assert collector.config.collect_error_metrics is False
        assert collector.config.time_bucket_minutes == 10

    def test_collect_empty_logs(self):
        """Test collecting metrics from empty logs"""
        collector = MetricsCollector()
        summary = collector.collect_metrics([])

        assert summary.total_logs == 0
        assert summary.performance is None
        assert summary.errors is None

    def test_collect_performance_metrics(self):
        """Test performance metrics collection"""
        now = datetime.now()
        logs = [
            {"timestamp": now, "response_time": 0.1},
            {"timestamp": now, "response_time": 0.2},
            {"timestamp": now, "response_time": 0.3},
            {"timestamp": now, "response_time": 1.5},  # Slow request
            {"timestamp": now, "duration": 0.15},  # Different field
        ]

        config = MetricsConfig(
            collect_performance_metrics=True,
            collect_error_metrics=False,
            collect_throughput_metrics=False,
            collect_user_metrics=False,
            slow_request_threshold=1.0,
        )
        collector = MetricsCollector(config)
        summary = collector.collect_metrics(logs)

        assert summary.performance is not None
        perf = summary.performance

        assert perf.count == 5
        assert perf.mean == pytest.approx(
            0.45, 0.01
        )  # (0.1 + 0.2 + 0.3 + 1.5 + 0.15) / 5 = 0.45
        assert perf.median == pytest.approx(0.2, 0.01)
        assert perf.min == 0.1
        assert perf.max == 1.5
        assert perf.slow_count == 1
        assert perf.slow_percentage == pytest.approx(0.2, 0.01)
        assert perf.p95 > perf.median
        assert perf.p99 >= perf.p95

    def test_collect_error_metrics(self):
        """Test error metrics collection"""
        now = datetime.now()
        logs = [
            {"timestamp": now, "level": "info"},
            {"timestamp": now, "level": "error", "error_type": "ValidationError"},
            {"timestamp": now, "level": "error", "error_type": "ValidationError"},
            {"timestamp": now, "level": "critical", "error_type": "DatabaseError"},
            {"timestamp": now, "status_code": 200},
            {"timestamp": now, "status_code": 404},
            {"timestamp": now, "status_code": 500},
        ]

        config = MetricsConfig(
            collect_performance_metrics=False,
            collect_error_metrics=True,
            collect_throughput_metrics=False,
            collect_user_metrics=False,
        )
        collector = MetricsCollector(config)
        summary = collector.collect_metrics(logs)

        assert summary.errors is not None
        err = summary.errors

        assert err.total_count == 7
        assert err.error_count == 6  # 3 error levels + 2 error status codes + 1 404
        assert err.error_rate == pytest.approx(6 / 7, 0.01)
        assert err.success_rate == pytest.approx(1 / 7, 0.01)
        assert err.error_types["ValidationError"] == 2
        assert err.error_types["DatabaseError"] == 1
        assert err.status_codes[200] == 1
        assert err.status_codes[404] == 1
        assert err.status_codes[500] == 1

    def test_collect_throughput_metrics(self):
        """Test throughput metrics collection"""
        base_time = datetime.now()
        logs = []

        # Create logs over 10 minutes
        for i in range(100):
            timestamp = base_time + timedelta(seconds=i * 6)  # 10 per minute
            logs.append(
                {
                    "timestamp": timestamp,
                    "user_id": f"user_{i % 10}",
                    "session_id": f"session_{i % 20}",
                }
            )

        config = MetricsConfig(
            collect_performance_metrics=False,
            collect_error_metrics=False,
            collect_throughput_metrics=True,
            collect_user_metrics=False,
            time_bucket_minutes=5,
        )
        collector = MetricsCollector(config)
        summary = collector.collect_metrics(logs)

        assert summary.throughput is not None
        thr = summary.throughput

        assert thr.total_requests == 100
        assert thr.requests_per_minute == pytest.approx(10.0, 0.5)
        assert thr.requests_per_second == pytest.approx(0.167, 0.01)
        assert thr.unique_users == 10
        assert thr.unique_sessions == 20
        assert thr.bytes_processed > 0

    def test_collect_user_metrics(self):
        """Test user activity metrics collection"""
        now = datetime.now()
        logs = [
            {"timestamp": now, "user_id": "user1", "session_id": "sess1"},
            {"timestamp": now, "user_id": "user1", "session_id": "sess2"},
            {"timestamp": now, "user_id": "user2", "session_id": "sess3"},
            {
                "timestamp": now,
                "user_id": "user2",
                "session_id": "sess3",
                "level": "error",
            },
            {"timestamp": now, "user_id": "user3", "session_id": "sess4"},
            {"timestamp": now - timedelta(hours=2), "user_id": "user4"},  # Old user
        ]

        config = MetricsConfig(
            collect_performance_metrics=False,
            collect_error_metrics=False,
            collect_throughput_metrics=False,
            collect_user_metrics=True,
        )
        collector = MetricsCollector(config)
        summary = collector.collect_metrics(logs)

        assert summary.users is not None
        usr = summary.users

        assert usr.active_users == 4
        assert usr.new_users == 3  # user1, user2, user3 are new
        assert usr.returning_users == 1  # user4 is old
        assert len(usr.user_sessions) == 3  # user4 has no session
        assert usr.user_sessions["user1"] == 2  # 2 sessions
        assert usr.user_errors["user2"] == 1
        assert len(usr.top_users) == 4
        assert usr.top_users[0] == ("user1", 2) or usr.top_users[0] == ("user2", 2)

    def test_collect_endpoint_metrics(self):
        """Test per-endpoint metrics collection"""
        now = datetime.now()
        logs = [
            {
                "timestamp": now,
                "endpoint": "/api/users",
                "response_time": 0.1,
                "status_code": 200,
            },
            {
                "timestamp": now,
                "endpoint": "/api/users",
                "response_time": 0.2,
                "status_code": 200,
            },
            {
                "timestamp": now,
                "endpoint": "/api/users",
                "response_time": 0.3,
                "level": "error",
                "status_code": 500,
            },
            {
                "timestamp": now,
                "endpoint": "/api/products",
                "response_time": 0.5,
                "status_code": 200,
            },
            {
                "timestamp": now,
                "endpoint": "/api/products",
                "response_time": 0.6,
                "status_code": 404,
            },
        ]

        collector = MetricsCollector()
        summary = collector.collect_metrics(logs)

        assert "/api/users" in summary.endpoint_metrics
        assert "/api/products" in summary.endpoint_metrics

        users_metrics = summary.endpoint_metrics["/api/users"]
        assert users_metrics["count"] == 3
        assert users_metrics["error_rate"] == pytest.approx(1 / 3, 0.01)
        assert users_metrics["avg_latency"] == pytest.approx(0.2, 0.01)
        assert users_metrics["p95_latency"] > 0
        assert users_metrics["status_codes"][200] == 2
        assert users_metrics["status_codes"][500] == 1

        products_metrics = summary.endpoint_metrics["/api/products"]
        assert products_metrics["count"] == 2
        assert products_metrics["avg_latency"] == pytest.approx(0.55, 0.01)

    def test_time_series_generation(self):
        """Test time series data generation"""
        base_time = datetime.now().replace(second=0, microsecond=0)
        logs = []

        # Create logs with varying patterns
        for i in range(30):
            timestamp = base_time + timedelta(minutes=i)
            # Increase errors over time
            is_error = i > 20
            # Increase latency over time
            latency = 0.1 + (i * 0.01)

            logs.append(
                {
                    "timestamp": timestamp,
                    "level": "error" if is_error else "info",
                    "response_time": latency,
                }
            )

        config = MetricsConfig(time_bucket_minutes=5, include_trends=True)
        collector = MetricsCollector(config)
        summary = collector.collect_metrics(logs)

        assert "requests_per_minute" in summary.time_series
        assert "error_rate" in summary.time_series
        assert "avg_latency" in summary.time_series

        # Check that error rate increases
        error_rates = summary.time_series["error_rate"]
        assert error_rates[0][1] < error_rates[-1][1]

        # Check that latency increases
        latencies = summary.time_series["avg_latency"]
        assert latencies[0][1] < latencies[-1][1]

    def test_time_window_filtering(self):
        """Test filtering logs by time window"""
        now = datetime.now()
        logs = [
            {"timestamp": now - timedelta(hours=2)},  # Old
            {"timestamp": now - timedelta(minutes=30)},  # Recent
            {"timestamp": now - timedelta(minutes=10)},  # Recent
            {"timestamp": now},  # Current
        ]

        collector = MetricsCollector()

        # No filter
        summary = collector.collect_metrics(logs)
        assert summary.total_logs == 4

        # 1 hour window
        summary = collector.collect_metrics(logs, time_window=timedelta(hours=1))
        assert summary.total_logs == 3

        # 15 minute window
        summary = collector.collect_metrics(logs, time_window=timedelta(minutes=15))
        assert summary.total_logs == 2

    def test_health_score_calculation(self):
        """Test health score calculation"""
        now = datetime.now()

        # Healthy logs
        healthy_logs = [
            {"timestamp": now, "response_time": 0.1, "level": "info"}
            for _ in range(100)
        ]

        collector = MetricsCollector()
        summary = collector.collect_metrics(healthy_logs)
        assert summary.health_score == 100.0

        # Unhealthy logs - high error rate
        error_logs = healthy_logs[:70] + [
            {"timestamp": now, "level": "error"} for _ in range(30)
        ]
        summary = collector.collect_metrics(error_logs)
        assert summary.health_score < 100.0
        assert summary.health_score >= 70.0  # Max 30 point deduction for errors

        # Unhealthy logs - slow requests
        slow_logs = [
            {"timestamp": now, "response_time": 2.0, "level": "info"} for _ in range(50)
        ] + healthy_logs[:50]
        summary = collector.collect_metrics(slow_logs)
        assert summary.health_score < 100.0

    def test_generate_report(self):
        """Test report generation"""
        now = datetime.now()
        logs = [
            {
                "timestamp": now,
                "response_time": 0.1,
                "user_id": "user1",
                "endpoint": "/api/test",
            },
            {
                "timestamp": now,
                "response_time": 0.2,
                "level": "error",
                "error_type": "TestError",
            },
            {
                "timestamp": now,
                "response_time": 1.5,
                "user_id": "user1",
                "endpoint": "/api/test",
            },
        ]

        collector = MetricsCollector()
        summary = collector.collect_metrics(logs)
        report = collector.generate_report(summary)

        # Check report contents
        assert "=== Metrics Report ===" in report
        assert "Total Logs: 3" in report
        assert "Health Score:" in report
        assert "Performance Metrics:" in report
        assert "Error Metrics:" in report
        assert "Throughput Metrics:" in report
        assert "User Metrics:" in report
        assert "Top Endpoints:" in report
        assert "/api/test" in report

    def test_percentile_calculation(self):
        """Test percentile calculation accuracy"""
        collector = MetricsCollector()

        # Test with known values
        values = list(range(1, 101))  # 1 to 100

        assert collector._percentile(values, 50) == 50.5  # Median
        assert collector._percentile(values, 25) == 25.75
        assert collector._percentile(values, 75) == 75.25
        assert collector._percentile(values, 95) == 95.05
        assert collector._percentile(values, 99) == 99.01

        # Test edge cases
        assert collector._percentile([], 50) == 0.0
        assert collector._percentile([5], 50) == 5.0
        assert collector._percentile([1, 2], 50) == 1.5

    def test_different_field_names(self):
        """Test handling different field names for same concepts"""
        now = datetime.now()
        logs = [
            {"timestamp": now, "response_time": 0.1},
            {"timestamp": now, "duration": 0.2},
            {"timestamp": now, "elapsed_time": 0.3},
            {"timestamp": now, "processing_time": 0.4},
            {"timestamp": now, "path": "/api/test"},  # Alternative to endpoint
            {"timestamp": now, "operation": "query"},  # Alternative to endpoint
        ]

        collector = MetricsCollector()
        summary = collector.collect_metrics(logs)

        # Should collect all latency fields
        assert summary.performance.count == 4

        # Should collect endpoint alternatives
        assert len(summary.endpoint_metrics) == 2

    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets"""
        # Generate large dataset
        now = datetime.now()
        logs = []
        for i in range(10000):
            logs.append(
                {
                    "timestamp": now + timedelta(seconds=i),
                    "response_time": 0.1 + (i % 10) * 0.01,
                    "user_id": f"user_{i % 100}",
                    "endpoint": f"/api/endpoint_{i % 20}",
                    "level": "error" if i % 50 == 0 else "info",
                }
            )

        config = MetricsConfig(max_unique_values=50)  # Limit unique values tracked
        collector = MetricsCollector(config)

        # Should complete without memory issues
        summary = collector.collect_metrics(logs)

        assert summary.total_logs == 10000
        assert len(summary.endpoint_metrics) <= 50  # Respects limit
        assert summary.performance.count == 10000
        assert summary.users.active_users == 100


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_collect_metrics_function(self):
        """Test collect_metrics convenience function"""
        now = datetime.now()
        logs = [
            {"timestamp": now, "response_time": 0.1},
            {"timestamp": now, "response_time": 0.2},
        ]

        summary = collect_metrics(logs)
        assert summary.total_logs == 2
        assert summary.performance.count == 2

        # With config
        config = MetricsConfig(collect_performance_metrics=False)
        summary = collect_metrics(logs, config=config)
        assert summary.performance is None

        # With time window
        summary = collect_metrics(logs, time_window=timedelta(hours=1))
        assert summary.total_logs == 2

    def test_create_metrics_collector_function(self):
        """Test create_metrics_collector convenience function"""
        collector = create_metrics_collector()
        assert isinstance(collector, MetricsCollector)
        assert collector.config.collect_performance_metrics is True

        # With config
        config = MetricsConfig(time_bucket_minutes=10)
        collector = create_metrics_collector(config)
        assert collector.config.time_bucket_minutes == 10


class TestMetricsDataClasses:
    """Test metrics data classes"""

    def test_performance_metrics(self):
        """Test PerformanceMetrics dataclass"""
        metrics = PerformanceMetrics(
            count=100,
            mean=0.5,
            median=0.4,
            min=0.1,
            max=2.0,
            std_dev=0.3,
            percentiles={50: 0.4, 95: 1.5, 99: 1.9},
            slow_count=10,
            slow_percentage=0.1,
        )

        assert metrics.p95 == 1.5
        assert metrics.p99 == 1.9

    def test_error_metrics(self):
        """Test ErrorMetrics dataclass"""
        metrics = ErrorMetrics(
            total_count=100,
            error_count=5,
            error_rate=0.05,
            error_types=Counter({"ValidationError": 3, "DatabaseError": 2}),
            status_codes=Counter({200: 95, 500: 5}),
        )

        assert metrics.success_rate == 0.95

    def test_metrics_summary(self):
        """Test MetricsSummary dataclass"""
        start = datetime.now()
        end = start + timedelta(hours=1)

        summary = MetricsSummary(start_time=start, end_time=end, total_logs=1000)

        assert summary.duration == timedelta(hours=1)
        assert summary.health_score == 100.0  # No errors or slow requests


def test_metrics_config_defaults():
    """Test MetricsConfig default values"""
    config = MetricsConfig()

    assert config.collect_performance_metrics is True
    assert config.metrics_window_minutes == 60
    assert config.slow_request_threshold == 1.0
    assert "response_time" in config.latency_fields
    assert "error" in config.error_fields
    assert 95 in config.percentiles
