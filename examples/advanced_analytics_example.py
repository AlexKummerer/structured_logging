#!/usr/bin/env python3
"""
Advanced Analytics Examples for Structured Logging

This example demonstrates the analytics capabilities including:
- Pattern detection for identifying common patterns
- Anomaly detection for finding outliers
- Performance metrics collection and analysis
"""

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List

from structured_logging import get_logger
from structured_logging.analytics import (
    # Anomaly detection
    AnomalyDetector,
    AnomalyDetectorConfig,
    # Metrics collection
    MetricsCollector,
    MetricsConfig,
    # Pattern detection
    PatternDetector,
    PatternDetectorConfig,
    collect_metrics,
    detect_anomalies,
    detect_patterns,
)


def generate_sample_logs(num_logs: int = 1000) -> List[Dict[str, Any]]:
    """Generate sample logs with various patterns and anomalies"""
    logs = []
    base_time = datetime.now() - timedelta(hours=2)

    # Common endpoints
    endpoints = [
        "/api/users",
        "/api/products",
        "/api/orders",
        "/api/auth/login",
        "/api/search",
    ]

    # Common error types
    error_types = [
        "ValidationError",
        "DatabaseError",
        "TimeoutError",
        "AuthenticationError",
    ]

    # User IDs
    user_ids = [f"user_{i}" for i in range(50)]

    for i in range(num_logs):
        timestamp = base_time + timedelta(seconds=i * 7.2)  # ~500 logs per hour

        # Normal pattern: Most requests are successful
        is_error = random.random() < 0.05  # 5% error rate normally

        # Anomaly: Error spike at certain time
        if 400 <= i <= 450:
            is_error = random.random() < 0.30  # 30% error rate spike

        # Normal response times with some outliers
        if random.random() < 0.95:
            response_time = random.gauss(0.2, 0.05)  # Normal: 200ms ± 50ms
        else:
            response_time = random.uniform(1.0, 5.0)  # Outliers: 1-5 seconds

        # Anomaly: Slow responses at certain time
        if 600 <= i <= 650:
            response_time = random.gauss(2.0, 0.5)  # Slow period

        log = {
            "timestamp": timestamp,
            "level": "error" if is_error else "info",
            "endpoint": random.choice(endpoints),
            "user_id": random.choice(user_ids),
            "session_id": f"session_{random.randint(1, 200)}",
            "response_time": max(0.01, response_time),  # Ensure positive
            "status_code": 500 if is_error else 200,
        }

        # Add error details
        if is_error:
            log["error_type"] = random.choice(error_types)
            log["message"] = f"Failed to process request: {log['error_type']}"
        else:
            log["message"] = f"Successfully processed {log['endpoint']} request"

        # Pattern: Certain users make more requests
        if log["user_id"] in ["user_1", "user_2", "user_3"]:
            # Power users - add duplicate requests
            for _ in range(random.randint(1, 3)):
                logs.append(log.copy())

        logs.append(log)

    return logs


def pattern_detection_example():
    """Demonstrate pattern detection capabilities"""
    print("=== Pattern Detection Example ===\n")

    # Generate sample logs
    logs = generate_sample_logs(1000)

    # Configure pattern detector
    config = PatternDetectorConfig(
        min_pattern_frequency=5,  # Minimum 5 occurrences
        detect_error_patterns=True,
        detect_sequence_patterns=True,
        detect_field_patterns=True,
        detect_frequency_patterns=True,
        include_samples=True,
        max_samples=3,
    )

    # Create detector and find patterns
    detector = PatternDetector(config)
    patterns = detector.detect_patterns(logs)

    print(f"Found {len(patterns)} patterns in {len(logs)} logs\n")

    # Display top patterns by type
    pattern_types = {}
    for pattern in patterns:
        pattern_type = pattern.pattern_type
        if pattern_type not in pattern_types:
            pattern_types[pattern_type] = []
        pattern_types[pattern_type].append(pattern)

    for pattern_type, type_patterns in pattern_types.items():
        print(f"\n{pattern_type.upper()} PATTERNS:")
        print("-" * 50)

        for pattern in type_patterns[:5]:  # Top 5 per type
            print(f"\nPattern: {pattern.pattern_value}")
            print(f"Frequency: {pattern.frequency}")
            print(f"Rate: {pattern.rate_per_minute:.2f} per minute")
            print(f"Duration: {pattern.duration}")

            if pattern.metadata:
                print(f"Metadata: {pattern.metadata}")

            if pattern.samples:
                print(f"Sample: {pattern.samples[0].get('message', 'N/A')[:80]}...")

    # Find patterns matching specific log
    test_log = {
        "timestamp": datetime.now(),
        "level": "error",
        "message": "Failed to process request: ValidationError",
        "error_type": "ValidationError",
        "endpoint": "/api/users",
    }

    matches = detector.find_pattern_matches(test_log, patterns)
    if matches:
        print(f"\n\nLog matches {len(matches)} patterns:")
        for match in matches:
            print(f"  - {match.pattern.pattern_type}: {match.pattern.pattern_value}")


def anomaly_detection_example():
    """Demonstrate anomaly detection capabilities"""
    print("\n\n=== Anomaly Detection Example ===\n")

    # Generate logs with anomalies
    logs = generate_sample_logs(1000)

    # Configure anomaly detector
    config = AnomalyDetectorConfig(
        enable_statistical_detection=True,
        enable_pattern_deviation=True,
        enable_frequency_anomalies=True,
        enable_value_anomalies=True,
        zscore_threshold=2.5,  # More sensitive
        min_anomaly_score=0.7,
        include_context_window=3,
        track_anomaly_chains=True,
    )

    # Split data for baseline
    baseline_logs = logs[:300]  # First 30% as baseline
    test_logs = logs[300:]  # Rest for testing

    # Detect anomalies
    detector = AnomalyDetector(config)
    anomalies = detector.detect_anomalies(test_logs, baseline_logs)

    print(f"Found {len(anomalies)} anomalies in {len(test_logs)} logs")
    print(f"Anomaly rate: {len(anomalies) / len(test_logs) * 100:.2f}%\n")

    # Group anomalies by type
    anomaly_types = {}
    for anomaly in anomalies:
        atype = anomaly.anomaly_type
        if atype not in anomaly_types:
            anomaly_types[atype] = []
        anomaly_types[atype].append(anomaly)

    # Display anomalies by type
    for atype, type_anomalies in anomaly_types.items():
        print(f"\n{atype.upper()} ANOMALIES: {len(type_anomalies)}")
        print("-" * 50)

        for anomaly in type_anomalies[:3]:  # Top 3 per type
            print(f"\nAnomaly ID: {anomaly.anomaly_id[:20]}...")
            print(f"Timestamp: {anomaly.timestamp}")
            print(f"Total Score: {anomaly.total_score:.2f}")
            print(f"Primary Reason: {anomaly.primary_reason}")

            # Show all detection scores
            for score in anomaly.scores:
                print(
                    f"  - {score.detection_method}: {score.score:.2f} - {score.reason}"
                )

            # Show anomalous values
            log = anomaly.log_entry
            print(
                f"Log: level={log.get('level')}, "
                f"endpoint={log.get('endpoint')}, "
                f"response_time={log.get('response_time', 0):.3f}s"
            )

            # Show related anomalies
            if anomaly.related_anomalies:
                print(f"Related anomalies: {len(anomaly.related_anomalies)}")

    # Find anomaly chains
    chains = []
    processed = set()

    for anomaly in anomalies:
        if anomaly.anomaly_id in processed:
            continue

        if anomaly.related_anomalies:
            chain = [anomaly.anomaly_id]
            chain.extend(anomaly.related_anomalies)
            chains.append(chain)
            processed.update(chain)

    if chains:
        print(f"\n\nFound {len(chains)} anomaly chains:")
        for i, chain in enumerate(chains[:3]):
            print(f"  Chain {i+1}: {len(chain)} related anomalies")


def metrics_collection_example():
    """Demonstrate metrics collection and analysis"""
    print("\n\n=== Metrics Collection Example ===\n")

    # Generate logs
    logs = generate_sample_logs(2000)

    # Configure metrics collector
    config = MetricsConfig(
        collect_performance_metrics=True,
        collect_error_metrics=True,
        collect_throughput_metrics=True,
        collect_user_metrics=True,
        time_bucket_minutes=10,
        slow_request_threshold=0.5,
        percentiles=[50, 75, 90, 95, 99],
        include_histogram=True,
        include_trends=True,
    )

    # Collect metrics
    collector = MetricsCollector(config)
    summary = collector.collect_metrics(logs, time_window=timedelta(hours=2))

    # Generate and print report
    report = collector.generate_report(summary)
    print(report)

    # Additional analysis
    print("\n\nDETAILED ANALYSIS:")
    print("-" * 50)

    # Performance distribution
    if summary.performance:
        print("\nPerformance Distribution:")
        lt_100 = sum(1 for log in logs if log.get('response_time', 0) < 0.1)
        r_100_500 = sum(
            1 for log in logs if 0.1 <= log.get('response_time', 0) < 0.5
        )
        r_500_1000 = sum(
            1 for log in logs if 0.5 <= log.get('response_time', 0) < 1.0
        )
        gt_1000 = sum(1 for log in logs if log.get('response_time', 0) >= 1.0)
        total = len(logs)
        
        print(f"  < 100ms: ~{lt_100 / total * 100:.1f}%")
        print(f"  100-500ms: ~{r_100_500 / total * 100:.1f}%")
        print(f"  500ms-1s: ~{r_500_1000 / total * 100:.1f}%")
        print(f"  > 1s: ~{gt_1000 / total * 100:.1f}%")

    # Time series trends
    if summary.time_series:
        print("\nTime Series Trends:")

        for metric_name, time_series in summary.time_series.items():
            if time_series:
                values = [value for _, value in time_series]
                trend = "increasing" if values[-1] > values[0] else "decreasing"
                print(
                    f"  {metric_name}: {trend} "
                    f"(from {values[0]:.3f} to {values[-1]:.3f})"
                )

    # Endpoint comparison
    if summary.endpoint_metrics:
        print("\nEndpoint Performance Comparison:")
        sorted_endpoints = sorted(
            summary.endpoint_metrics.items(),
            key=lambda x: x[1].get("avg_latency", 0),
            reverse=True,
        )

        for endpoint, metrics in sorted_endpoints[:5]:
            print(f"  {endpoint}:")
            print(f"    Avg latency: {metrics.get('avg_latency', 0):.3f}s")
            print(f"    Error rate: {metrics.get('error_rate', 0):.1%}")
            print(f"    Volume: {metrics['count']} requests")


def real_time_monitoring_example():
    """Demonstrate real-time monitoring use case"""
    print("\n\n=== Real-Time Monitoring Example ===\n")

    # Initialize analytics components
    pattern_detector = PatternDetector()
    anomaly_detector = AnomalyDetector()
    metrics_collector = MetricsCollector()

    # Simulate real-time log stream
    print("Simulating real-time monitoring for 60 seconds...")
    print("(In production, this would process actual logs)")

    all_logs = []
    baseline_established = False

    for second in range(60):
        # Generate logs for this second
        num_logs = random.randint(5, 15)
        current_logs = []

        for _ in range(num_logs):
            is_anomaly = second in [20, 21, 40, 41] and random.random() < 0.5

            log = {
                "timestamp": datetime.now(),
                "level": "error" if is_anomaly else "info",
                "endpoint": random.choice(["/api/users", "/api/products"]),
                "response_time": 5.0 if is_anomaly else random.gauss(0.2, 0.05),
                "user_id": f"user_{random.randint(1, 10)}",
                "message": "Anomalous behavior" if is_anomaly else "Normal operation",
            }

            current_logs.append(log)
            all_logs.append(log)

        # Process logs every 10 seconds
        if second % 10 == 9:
            print(f"\n[{second+1}s] Processing {len(all_logs)} logs...")

            # Establish baseline after 20 seconds
            if not baseline_established and len(all_logs) > 100:
                baseline_logs = all_logs[:100]
                anomaly_detector._build_baseline(baseline_logs)
                baseline_established = True
                print("  - Baseline established")

            # Detect patterns
            patterns = pattern_detector.detect_patterns(all_logs[-100:])
            if patterns:
                print(f"  - Found {len(patterns)} patterns")

            # Detect anomalies
            if baseline_established:
                recent_logs = all_logs[-50:]
                anomalies = anomaly_detector.detect_anomalies(recent_logs)
                if anomalies:
                    print(f"  - ALERT: {len(anomalies)} anomalies detected!")
                    for anomaly in anomalies[:2]:
                        print(f"    * {anomaly.primary_reason}")

            # Collect metrics
            metrics = metrics_collector.collect_metrics(all_logs[-100:])
            print(f"  - Health Score: {metrics.health_score:.1f}/100")

            if metrics.performance:
                print(f"  - Avg Response Time: {metrics.performance.mean:.3f}s")
            if metrics.errors:
                print(f"  - Error Rate: {metrics.errors.error_rate:.1%}")

    print("\n\nMonitoring complete!")
    print(f"Total logs processed: {len(all_logs)}")


def integrated_analysis_example():
    """Demonstrate integrated analysis combining all features"""
    print("\n\n=== Integrated Analysis Example ===\n")

    # Generate comprehensive dataset
    logs = generate_sample_logs(5000)

    print(f"Analyzing {len(logs)} logs...")

    # 1. Collect metrics for overview
    metrics = collect_metrics(logs)
    print(f"\nSystem Health Score: {metrics.health_score:.1f}/100")

    # 2. Detect patterns
    patterns = detect_patterns(logs, PatternDetectorConfig(min_pattern_frequency=10))

    # 3. Detect anomalies
    anomalies = detect_anomalies(logs)

    # 4. Correlate findings
    print("\nAnalysis Summary:")
    print(f"  - Patterns found: {len(patterns)}")
    print(f"  - Anomalies found: {len(anomalies)}")
    print(
        f"  - Error rate: {metrics.errors.error_rate:.1%}"
        if metrics.errors
        else "  - No errors"
    )
    print(
        f"  - Avg response time: {metrics.performance.mean:.3f}s"
        if metrics.performance
        else "  - No performance data"
    )

    # Find correlations
    if anomalies and patterns:
        print("\nCorrelations:")

        # Check if anomalies match any patterns
        anomaly_pattern_matches = 0
        for anomaly in anomalies:
            pattern_detector = PatternDetector()
            matches = pattern_detector.find_pattern_matches(anomaly.log_entry, patterns)
            if matches:
                anomaly_pattern_matches += 1

        matches_msg = (
            f"  - {anomaly_pattern_matches}/{len(anomalies)} "
            "anomalies match known patterns"
        )
        print(matches_msg)

        # Check if error spikes correlate with performance issues
        if (
            metrics.time_series
            and "error_rate" in metrics.time_series
            and "avg_latency" in metrics.time_series
        ):
            error_series = metrics.time_series["error_rate"]
            latency_series = metrics.time_series["avg_latency"]

            # Simple correlation check
            high_error_times = [t for t, rate in error_series if rate > 0.1]
            high_latency_times = [t for t, lat in latency_series if lat > 0.5]

            if high_error_times and high_latency_times:
                print(f"  - Found {len(high_error_times)} high error periods")
                print(f"  - Found {len(high_latency_times)} high latency periods")

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    print("Structured Logging - Advanced Analytics Examples")
    print("=" * 60)

    # Run all examples
    pattern_detection_example()
    anomaly_detection_example()
    metrics_collection_example()
    real_time_monitoring_example()
    integrated_analysis_example()

    print("\n\nAll examples completed!")
    print("\nThese analytics features enable:")
    print("  ✓ Automatic pattern discovery")
    print("  ✓ Real-time anomaly detection")
    print("  ✓ Comprehensive performance metrics")
    print("  ✓ Proactive monitoring and alerting")
    print("  ✓ Data-driven insights from logs")
