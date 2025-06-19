#!/usr/bin/env python3
"""
Performance benchmarks for structured logging library
"""

import cProfile
import gc
import io
import pstats
import time
from contextlib import contextmanager

from structured_logging import (
    LoggerConfig,
    get_logger,
    log_with_context,
    request_context,
)


@contextmanager
def capture_performance():
    """Context manager to capture performance metrics"""
    pr = cProfile.Profile()
    gc.collect()  # Clean up before test

    start_time = time.perf_counter()
    pr.enable()

    try:
        yield
    finally:
        pr.disable()
        end_time = time.perf_counter()

        # Get stats
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(20)  # Top 20 functions

        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.6f} seconds")
        print(f"Performance profile:\n{s.getvalue()}")


def benchmark_basic_logging(iterations: int = 10000):
    """Benchmark basic logging performance"""
    print(f"\n=== Basic Logging Benchmark ({iterations:,} iterations) ===")

    # JSON formatter - redirect output to devnull
    config = LoggerConfig(formatter_type="json")
    logger = get_logger("benchmark_json", config)

    # Redirect logger output to suppress console spam

    logger.handlers[0].stream = open("/dev/null", "w")

    with capture_performance():
        for i in range(iterations):
            logger.info(f"Test message {i}")


def benchmark_context_logging(iterations: int = 10000):
    """Benchmark context-aware logging performance"""
    print(f"\n=== Context Logging Benchmark ({iterations:,} iterations) ===")

    logger = get_logger("benchmark_context")

    with capture_performance():
        with request_context(user_id="user123", tenant_id="tenant456"):
            for i in range(iterations):
                log_with_context(
                    logger,
                    "info",
                    f"Context message {i}",
                    request_number=i,
                    batch_id=f"batch_{i//100}",
                )


def benchmark_formatter_types(iterations: int = 5000):
    """Benchmark different formatter types"""
    print(f"\n=== Formatter Types Benchmark ({iterations:,} iterations each) ===")

    formatters = ["json", "csv", "plain"]

    for formatter_type in formatters:
        print(f"\n--- {formatter_type.upper()} Formatter ---")
        config = LoggerConfig(formatter_type=formatter_type)
        logger = get_logger(f"benchmark_{formatter_type}", config)

        with capture_performance():
            with request_context(
                user_id="user123", service=f"{formatter_type}_service"
            ):
                for i in range(iterations):
                    log_with_context(
                        logger,
                        "info",
                        f"Message {i}",
                        iteration=i,
                        formatter=formatter_type,
                    )


def benchmark_timestamp_generation(iterations: int = 100000):
    """Benchmark timestamp generation overhead"""
    print(f"\n=== Timestamp Generation Benchmark ({iterations:,} iterations) ===")

    # With timestamps
    print("\n--- With Timestamps ---")
    config_with_ts = LoggerConfig(include_timestamp=True)
    logger_with_ts = get_logger("benchmark_ts", config_with_ts)

    with capture_performance():
        for i in range(iterations):
            logger_with_ts.info(f"Message {i}")

    # Without timestamps
    print("\n--- Without Timestamps ---")
    config_no_ts = LoggerConfig(include_timestamp=False)
    logger_no_ts = get_logger("benchmark_no_ts", config_no_ts)

    with capture_performance():
        for i in range(iterations):
            logger_no_ts.info(f"Message {i}")


def benchmark_context_access(iterations: int = 50000):
    """Benchmark context variable access patterns"""
    print(f"\n=== Context Access Benchmark ({iterations:,} iterations) ===")

    logger = get_logger("benchmark_ctx_access")

    print("\n--- With Context ---")
    with capture_performance():
        with request_context(user_id="user123", tenant_id="tenant456"):
            for i in range(iterations):
                log_with_context(logger, "info", f"Message {i}")

    print("\n--- Without Context ---")
    with capture_performance():
        for i in range(iterations):
            logger.info(f"Message {i}")


def memory_benchmark():
    """Benchmark memory usage"""
    import tracemalloc

    print("\n=== Memory Usage Benchmark ===")

    tracemalloc.start()

    logger = get_logger("memory_test")

    # Baseline measurement
    snapshot1 = tracemalloc.take_snapshot()

    # Logging operations
    with request_context(user_id="user123", tenant_id="tenant456"):
        for i in range(1000):
            log_with_context(
                logger,
                "info",
                f"Memory test {i}",
                data={"key": f"value_{i}", "number": i},
            )

    snapshot2 = tracemalloc.take_snapshot()

    # Calculate difference
    top_stats = snapshot2.compare_to(snapshot1, "lineno")

    print("Top 10 memory allocations:")
    for index, stat in enumerate(top_stats[:10], 1):
        print(f"{index}. {stat}")

    tracemalloc.stop()


def run_all_benchmarks():
    """Run all performance benchmarks"""
    print("🚀 Structured Logging Performance Benchmarks")
    print("=" * 50)

    # Basic performance tests
    benchmark_basic_logging(10000)
    benchmark_context_logging(10000)
    benchmark_formatter_types(5000)
    benchmark_timestamp_generation(50000)
    benchmark_context_access(25000)

    # Memory usage
    memory_benchmark()

    print("\n" + "=" * 50)
    print("✅ All benchmarks completed!")


if __name__ == "__main__":
    run_all_benchmarks()
