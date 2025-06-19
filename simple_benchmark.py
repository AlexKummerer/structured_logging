#!/usr/bin/env python3
"""
Simple performance analysis for structured logging
"""

import statistics
import time

from structured_logging import (
    LoggerConfig,
    get_logger,
    log_with_context,
    request_context,
)


def redirect_to_devnull():
    """Redirect stdout to devnull to suppress log output"""
    return open("/dev/null", "w")


def time_function(func, iterations=1000):
    """Time a function over multiple iterations"""
    times = []
    for _ in range(5):  # Run 5 times for average
        start = time.perf_counter()
        for _ in range(iterations):
            func()
        end = time.perf_counter()
        times.append((end - start) / iterations * 1000)  # ms per iteration

    return {
        "mean": statistics.mean(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0,
        "min": min(times),
        "max": max(times),
    }


def setup_logger(formatter_type="json"):
    """Setup logger with suppressed output"""
    config = LoggerConfig(formatter_type=formatter_type)
    logger = get_logger(f"bench_{formatter_type}", config)

    # Suppress output
    if logger.handlers:
        logger.handlers[0].stream = redirect_to_devnull()

    return logger


def main():
    print("🚀 Structured Logging Performance Analysis")
    print("=" * 50)

    # Test 1: Basic logging performance
    print("\n1. Basic Logging Performance (per log call)")
    logger = setup_logger("json")

    def basic_log():
        logger.info("Test message")

    basic_stats = time_function(basic_log, 1000)
    print(f"   Mean: {basic_stats['mean']:.3f}ms")
    print(f"   Std:  {basic_stats['stdev']:.3f}ms")

    # Test 2: Context logging performance
    print("\n2. Context Logging Performance (per log call)")

    def context_log():
        with request_context(user_id="user123"):
            log_with_context(logger, "info", "Test message")

    context_stats = time_function(context_log, 1000)
    print(f"   Mean: {context_stats['mean']:.3f}ms")
    print(f"   Std:  {context_stats['stdev']:.3f}ms")
    print(f"   Overhead: {context_stats['mean'] - basic_stats['mean']:.3f}ms")

    # Test 3: Formatter comparison
    print("\n3. Formatter Comparison (per log call)")
    formatters = ["json", "csv", "plain"]

    for fmt in formatters:
        fmt_logger = setup_logger(fmt)

        def fmt_log():
            fmt_logger.info("Test message")

        fmt_stats = time_function(fmt_log, 1000)
        print(f"   {fmt.upper():<5}: {fmt_stats['mean']:.3f}ms")

    # Test 4: Timestamp overhead
    print("\n4. Timestamp Overhead (per log call)")

    # With timestamp
    config_ts = LoggerConfig(include_timestamp=True)
    logger_ts = get_logger("bench_ts", config_ts)
    logger_ts.handlers[0].stream = redirect_to_devnull()

    def log_with_ts():
        logger_ts.info("Test message")

    ts_stats = time_function(log_with_ts, 1000)

    # Without timestamp
    config_no_ts = LoggerConfig(include_timestamp=False)
    logger_no_ts = get_logger("bench_no_ts", config_no_ts)
    logger_no_ts.handlers[0].stream = redirect_to_devnull()

    def log_without_ts():
        logger_no_ts.info("Test message")

    no_ts_stats = time_function(log_without_ts, 1000)

    print(f"   With timestamp:    {ts_stats['mean']:.3f}ms")
    print(f"   Without timestamp: {no_ts_stats['mean']:.3f}ms")
    print(f"   Timestamp overhead: {ts_stats['mean'] - no_ts_stats['mean']:.3f}ms")

    print("\n" + "=" * 50)
    print("✅ Performance analysis complete!")

    # Performance bottlenecks identified
    print("\n📊 Performance Bottlenecks Identified:")
    print(
        f"1. Context overhead: {context_stats['mean'] - basic_stats['mean']:.3f}ms per call"
    )
    print(
        f"2. Timestamp overhead: {ts_stats['mean'] - no_ts_stats['mean']:.3f}ms per call"
    )

    # Calculate throughput
    basic_throughput = 1000 / basic_stats["mean"]
    context_throughput = 1000 / context_stats["mean"]

    print("\n🚀 Throughput:")
    print(f"Basic logging: {basic_throughput:.0f} logs/second")
    print(f"Context logging: {context_throughput:.0f} logs/second")


if __name__ == "__main__":
    main()
