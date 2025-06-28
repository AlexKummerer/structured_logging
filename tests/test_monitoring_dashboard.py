"""Tests for monitoring dashboard"""

import asyncio
import json
import pytest
from datetime import datetime

from structured_logging.monitoring import (
    DashboardServer,
    DashboardConfig,
    AlertLevel,
    AlertCondition,
    create_dashboard,
)
from structured_logging.monitoring.alerts import AlertManager, create_alert_rule
from structured_logging.monitoring.metrics import MetricsAggregator


@pytest.mark.asyncio
async def test_dashboard_creation():
    """Test creating dashboard"""
    dashboard = create_dashboard()
    assert dashboard is not None
    assert isinstance(dashboard.config, DashboardConfig)
    assert isinstance(dashboard.alerts, AlertManager)
    assert isinstance(dashboard.metrics, MetricsAggregator)


@pytest.mark.asyncio
async def test_dashboard_config():
    """Test dashboard configuration"""
    config = DashboardConfig(
        host="0.0.0.0",
        port=9090,
        enable_alerts=False,
        enable_metrics=False,
        buffer_size=500
    )
    
    dashboard = DashboardServer(config)
    assert dashboard.config.host == "0.0.0.0"
    assert dashboard.config.port == 9090
    assert dashboard.config.enable_alerts is False
    assert dashboard.config.buffer_size == 500


@pytest.mark.asyncio
async def test_dashboard_process_log():
    """Test processing logs"""
    dashboard = DashboardServer()
    
    # Process a log
    test_log = {
        "timestamp": datetime.now(),
        "level": "info",
        "message": "Test log"
    }
    
    await dashboard.process_log(test_log)
    
    # Check buffer
    assert len(dashboard._log_buffer) == 1
    assert dashboard._log_buffer[0] == test_log


@pytest.mark.asyncio
async def test_dashboard_buffer_limit():
    """Test log buffer limit"""
    config = DashboardConfig(buffer_size=5)
    dashboard = DashboardServer(config)
    
    # Add more logs than buffer size
    for i in range(10):
        await dashboard.process_log({
            "id": i,
            "message": f"Log {i}"
        })
    
    # Buffer should maintain size limit
    assert len(dashboard._log_buffer) == 5
    # Should keep latest logs
    assert dashboard._log_buffer[0]["id"] == 5
    assert dashboard._log_buffer[-1]["id"] == 9


@pytest.mark.asyncio
async def test_dashboard_metrics_update():
    """Test metrics update"""
    dashboard = DashboardServer()
    
    # Process logs with metrics
    await dashboard.process_log({
        "level": "info",
        "response_time": 100
    })
    
    await dashboard.process_log({
        "level": "error",
        "response_time": 500
    })
    
    # Get metrics snapshot
    metrics = await dashboard.metrics.get_current_snapshot()
    
    assert metrics["total_logs"] == 2
    assert metrics["error_count"] == 1
    assert metrics["avg_response_time"] > 0


@pytest.mark.asyncio
async def test_dashboard_alerts():
    """Test alert processing"""
    dashboard = DashboardServer()
    triggered_alerts = []
    
    # Register alert handler
    def alert_handler(alert):
        triggered_alerts.append(alert)
    
    dashboard.alerts.register_handler(lambda alert: triggered_alerts.append(alert))
    
    # Add custom alert rule
    dashboard.alerts.add_rule(
        name="Test Alert",
        conditions=[
            AlertCondition(
                field_name="error_count",
                operator="gt",
                value=5
            )
        ],
        level=AlertLevel.WARNING
    )
    
    # Process log that triggers alert
    await dashboard.process_log({
        "level": "error",
        "error_count": 10
    })
    
    # Alert should be triggered
    assert len(dashboard.alerts.get_active_alerts()) > 0


@pytest.mark.asyncio
async def test_alert_manager_default_rules():
    """Test default alert rules"""
    manager = AlertManager()
    
    # Should have default rules
    assert len(manager.rules) > 0
    
    # Test high error rate rule
    log = {"level": "error", "message": "Database connection failed"}
    alerts = await manager.check(log)
    
    assert len(alerts) > 0
    assert any(alert["name"] == "High Error Rate" for alert in alerts)


@pytest.mark.asyncio
async def test_alert_condition_evaluation():
    """Test alert condition evaluation"""
    # Equals condition
    cond = AlertCondition(field_name="status", operator="equals", value=500)
    assert cond.evaluate({"status": 500}) is True
    assert cond.evaluate({"status": 200}) is False
    
    # Greater than condition
    cond = AlertCondition(field_name="count", operator="gt", value=10)
    assert cond.evaluate({"count": 15}) is True
    assert cond.evaluate({"count": 5}) is False
    
    # Contains condition
    cond = AlertCondition(field_name="message", operator="contains", value="error")
    assert cond.evaluate({"message": "Database error occurred"}) is True
    assert cond.evaluate({"message": "Success"}) is False
    
    # Regex condition
    cond = AlertCondition(field_name="path", operator="regex", value=r"/api/.*")
    assert cond.evaluate({"path": "/api/users"}) is True
    assert cond.evaluate({"path": "/health"}) is False


@pytest.mark.asyncio
async def test_alert_rate_limiting():
    """Test alert rate limiting"""
    manager = AlertManager()
    
    # Create rule with cooldown
    rule_id = manager.add_rule(
        name="Rate Limited Alert",
        conditions=[
            AlertCondition(field_name="level", operator="equals", value="error")
        ],
        cooldown_seconds=1,
        max_alerts_per_hour=2
    )
    
    error_log = {"level": "error", "message": "Test error"}
    
    # First alert should trigger
    alerts1 = await manager.check(error_log)
    assert len(alerts1) > 0
    
    # Immediate second alert should not trigger (cooldown)
    alerts2 = await manager.check(error_log)
    assert not any(a["name"] == "Rate Limited Alert" for a in alerts2)
    
    # Wait for cooldown
    await asyncio.sleep(1.1)
    
    # Should trigger again
    alerts3 = await manager.check(error_log)
    assert any(a["name"] == "Rate Limited Alert" for a in alerts3)


@pytest.mark.asyncio
async def test_metrics_aggregator():
    """Test metrics aggregator"""
    aggregator = MetricsAggregator()
    
    # Update with logs
    await aggregator.update({
        "level": "info",
        "response_time": 100,
        "custom_metric": 42
    })
    
    await aggregator.update({
        "level": "error",
        "response_time": 200
    })
    
    # Get snapshot
    snapshot = await aggregator.get_current_snapshot()
    
    assert snapshot["total_logs"] == 2
    assert snapshot["error_count"] == 1
    assert snapshot["avg_response_time"] == 150.0
    
    # Check custom metrics
    assert "custom_metric" in snapshot["custom_metrics"]
    assert snapshot["custom_metrics"]["custom_metric"]["avg"] == 42


@pytest.mark.asyncio
async def test_metrics_time_series():
    """Test time series metrics"""
    aggregator = MetricsAggregator()
    
    # Add metrics over time
    for i in range(5):
        await aggregator.update({
            "response_time": 100 + i * 10
        })
        await asyncio.sleep(0.1)
    
    # Get metrics with time range
    metrics = await aggregator.get_metrics("5m")
    
    assert "time_series" in metrics
    assert "response_time" in metrics["time_series"]
    
    ts_data = metrics["time_series"]["response_time"]
    assert len(ts_data["values"]) == 5
    assert ts_data["values"] == [100, 110, 120, 130, 140]


@pytest.mark.asyncio
async def test_create_alert_rule():
    """Test alert rule creation helper"""
    rule = create_alert_rule(
        name="CPU Alert",
        field="cpu_usage",
        operator="gt",
        value=80,
        level=AlertLevel.WARNING
    )
    
    assert rule.name == "CPU Alert"
    assert rule.level == AlertLevel.WARNING
    assert len(rule.conditions) == 1
    assert rule.conditions[0].field_name == "cpu_usage"
    assert rule.conditions[0].operator == "gt"
    assert rule.conditions[0].value == 80


@pytest.mark.asyncio
async def test_dashboard_search():
    """Test log search functionality"""
    config = DashboardConfig(enable_search=True)
    dashboard = DashboardServer(config)
    
    # Add test logs
    logs = [
        {"level": "info", "message": "User login", "user": "alice"},
        {"level": "error", "message": "Database error", "user": "bob"},
        {"level": "info", "message": "User logout", "user": "alice"},
    ]
    
    for log in logs:
        await dashboard.process_log(log)
    
    # Search by text
    results = dashboard._search_log_buffer("error", {})
    assert len(results) == 1
    assert results[0]["level"] == "error"
    
    # Search with filters
    results = dashboard._search_log_buffer("", {"user": "alice"})
    assert len(results) == 2
    assert all(r["user"] == "alice" for r in results)