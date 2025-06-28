"""
Example: Real-time monitoring dashboard

This example demonstrates:
- Setting up the monitoring dashboard
- Stream processing with real-time updates
- Custom alerts and visualizations
- Integration with log streams
"""

import asyncio
import random
import time
from datetime import datetime

from structured_logging import get_logger
from structured_logging.monitoring import (
    AlertCondition,
    AlertLevel,
    DashboardConfig,
    DashboardServer,
)
from structured_logging.streaming import (
    LogStreamProcessor,
    StreamConfig,
    TumblingWindow,
    avg,
    count,
    create_source,
)


async def generate_sample_logs():
    """Generate sample logs for demonstration"""
    logger = get_logger("demo.app")

    levels = ["info", "info", "info", "warning", "error"]
    endpoints = ["/api/users", "/api/products", "/api/orders", "/api/auth"]

    while True:
        # Simulate API request
        endpoint = random.choice(endpoints)
        response_time = random.gauss(200, 100)  # Normal distribution

        # Occasionally generate slow requests
        if random.random() < 0.1:
            response_time = random.uniform(1000, 5000)

        # Occasionally generate errors
        if random.random() < 0.05:
            logger.error(
                f"Request failed: {endpoint}",
                endpoint=endpoint,
                error="Internal Server Error",
                status_code=500,
                response_time=response_time,
            )
        else:
            level = random.choice(levels)
            logger.log(
                level.upper(),
                f"Request processed: {endpoint}",
                endpoint=endpoint,
                status_code=200 if level != "error" else 500,
                response_time=response_time,
                user_id=f"user_{random.randint(1, 100)}",
                request_size=random.randint(100, 10000),
                response_size=random.randint(100, 50000),
            )

        await asyncio.sleep(random.uniform(0.1, 0.5))


async def setup_monitoring_pipeline():
    """Setup stream processing and monitoring"""

    # Create stream processor with configuration
    stream_config = StreamConfig(
        buffer_size=10000, batch_size=50, enable_metrics=True, watermark_delay_ms=2000
    )

    processor = LogStreamProcessor(stream_config)

    # Add memory source (in production, use file/kafka/etc.)
    source = create_source("memory", logs=[], repeat=False)

    # Add processing pipeline
    processor.source(source).filter(
        lambda log: log.get("response_time") is not None
    ).window(
        TumblingWindow(
            size_seconds=60,  # 1-minute windows
            aggregators=[count(), avg("response_time")],
        )
    )

    # Create dashboard
    dashboard_config = DashboardConfig(
        host="localhost",
        port=8080,
        enable_alerts=True,
        enable_metrics=True,
        enable_search=True,
    )

    dashboard = DashboardServer(dashboard_config)

    # Add custom alerts
    dashboard.alerts.add_rule(
        name="High Response Time",
        conditions=[
            AlertCondition(field_name="response_time", operator="gt", value=2000)
        ],
        level=AlertLevel.WARNING,
        description="Response time exceeds 2 seconds",
        cooldown_seconds=60,
        actions=[
            {"type": "log", "message": "High response time detected"},
            {"type": "webhook", "url": "http://localhost:9000/alerts"},
        ],
    )

    dashboard.alerts.add_rule(
        name="Error Spike",
        conditions=[
            AlertCondition(field_name="level", operator="equals", value="error")
        ],
        level=AlertLevel.ERROR,
        description="Multiple errors detected",
        cooldown_seconds=300,
        max_alerts_per_hour=5,
    )

    # Connect dashboard to stream
    dashboard.attach_to_stream(processor)

    return processor, dashboard


async def monitor_stream(processor: LogStreamProcessor):
    """Monitor stream processor status"""
    while True:
        await asyncio.sleep(10)

        state = processor.get_state()
        print(f"\nStream Status: {state['state']}")
        print(f"Processed: {state['processed_count']} logs")
        print(f"Errors: {state['error_count']}")
        print(f"Queue size: {state['queue_size']}")
        print(f"Active windows: {state['window_count']}")


async def main():
    """Run the monitoring dashboard demo"""
    print("Setting up monitoring dashboard...")

    # Setup pipeline
    processor, dashboard = await setup_monitoring_pipeline()

    # Start dashboard
    await dashboard.start()

    print("\nDashboard running at http://localhost:8080")
    print("Generating sample logs...")

    # Start stream processor
    await processor.start()

    # Create tasks
    tasks = [
        asyncio.create_task(generate_sample_logs()),
        asyncio.create_task(monitor_stream(processor)),
    ]

    # Feed logs to processor
    try:
        # Run for demonstration
        start_time = time.time()

        while True:
            # Generate some logs directly
            for _ in range(random.randint(5, 15)):
                log = {
                    "timestamp": datetime.now(),
                    "level": random.choice(["info", "warning", "error"]),
                    "message": "Sample log entry",
                    "response_time": random.gauss(200, 100),
                    "endpoint": random.choice(["/api/users", "/api/products"]),
                    "source": "direct",
                }

                await dashboard.process_log(log)

            await asyncio.sleep(1)

            # Stop after 5 minutes for demo
            if time.time() - start_time > 300:
                break

    except KeyboardInterrupt:
        print("\nShutting down...")

    finally:
        # Cleanup
        await processor.stop()
        await dashboard.stop()

        for task in tasks:
            task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)


# Additional example: Custom visualization
def create_custom_dashboard():
    """Create dashboard with custom visualizations"""
    from structured_logging.monitoring import create_dashboard, create_visualization

    # Create dashboard
    dashboard = create_dashboard()

    # Add custom visualizations
    visualizations = [
        # Log volume over time
        create_visualization(
            "log_chart",
            title="Log Volume (5 min)",
            time_window=300,
            group_by_level=True,
        ),
        # Response time metrics
        create_visualization(
            "metric_chart",
            title="API Performance",
            metrics=["avg_response_time", "p95_response_time"],
            y_axis_label="Response Time (ms)",
        ),
        # Error heatmap
        create_visualization(
            "error_heatmap",
            title="Error Patterns (24h)",
            time_buckets=24,
            error_categories=["api", "database", "auth", "validation", "other"],
        ),
        # KPI gauges
        create_visualization(
            "gauge",
            title="Error Rate",
            metric="error_rate",
            min_value=0,
            max_value=10,
            thresholds=[(0.5, "success"), (0.8, "warning"), (1.0, "danger")],
        ),
    ]

    return dashboard, visualizations


# Example: Alert webhook handler
async def alert_webhook_handler(request):
    """Handle alert webhooks"""
    data = await request.json()
    alert = data.get("alert", {})

    print(f"\n🚨 ALERT: {alert.get('name')}")
    print(f"Level: {alert.get('level')}")
    print(f"Description: {alert.get('description')}")
    print(f"Triggered at: {alert.get('triggered_at')}")

    # In production, send to Slack, PagerDuty, etc.

    return {"status": "acknowledged"}


if __name__ == "__main__":
    print("Structured Logging - Real-time Monitoring Dashboard")
    print("=" * 50)

    asyncio.run(main())
