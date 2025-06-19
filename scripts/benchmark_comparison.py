#!/usr/bin/env python3
"""
Benchmark comparison script to compare performance across versions
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from unittest.mock import patch

# Add the source to the path
sys.path.insert(0, "src")

from structured_logging import (
    FilterConfig,
    LevelFilter,
    LoggerConfig,
    SamplingFilter,
    get_logger,
    log_with_context,
)


def benchmark_basic_logging(iterations=10000):
    """Benchmark basic logging performance"""
    import logging

    logger = logging.getLogger("benchmark_basic")
    logger.handlers.clear()

    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    start_time = time.perf_counter()
    for i in range(iterations):
        logger.info(f"Benchmark message {i}")
    duration = time.perf_counter() - start_time

    throughput = iterations / duration
    logger.handlers.clear()

    return {
        "test": "basic_logging",
        "iterations": iterations,
        "duration": duration,
        "throughput": throughput,
        "log_size": len(stream.getvalue()),
    }


def benchmark_structured_logging(iterations=5000):
    """Benchmark structured logging performance"""
    config = LoggerConfig(
        output_type="console",
        formatter_type="json",
        include_timestamp=True,
        include_request_id=True,
    )

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        logger = get_logger("benchmark_structured", config)

        start_time = time.perf_counter()
        for i in range(iterations):
            log_with_context(
                logger,
                "info",
                f"Structured message {i}",
                config,
                test_id=i,
                benchmark="structured",
            )
        duration = time.perf_counter() - start_time

        throughput = iterations / duration
        log_size = len(mock_stdout.getvalue())

    return {
        "test": "structured_logging",
        "iterations": iterations,
        "duration": duration,
        "throughput": throughput,
        "log_size": log_size,
    }


def benchmark_filtered_logging(iterations=3000):
    """Benchmark logging with filtering"""
    filter_config = FilterConfig(
        enabled=True,
        filters=[
            LevelFilter(min_level="INFO"),
            SamplingFilter(sample_rate=1.0, strategy="random"),
        ],
    )

    config = LoggerConfig(
        output_type="console", formatter_type="json", filter_config=filter_config
    )

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        with patch("random.random", return_value=0.5):  # Consistent sampling
            logger = get_logger("benchmark_filtered", config)

            start_time = time.perf_counter()
            for i in range(iterations):
                log_with_context(
                    logger,
                    "info",
                    f"Filtered message {i}",
                    config,
                    test_id=i,
                    benchmark="filtered",
                )
            duration = time.perf_counter() - start_time

            throughput = iterations / duration
            log_size = len(mock_stdout.getvalue())

    return {
        "test": "filtered_logging",
        "iterations": iterations,
        "duration": duration,
        "throughput": throughput,
        "log_size": log_size,
    }


def benchmark_concurrent_logging(workers=4, logs_per_worker=1000):
    """Benchmark concurrent logging performance"""
    config = LoggerConfig(output_type="console", formatter_type="json")

    def worker(worker_id):
        with patch("sys.stdout", new_callable=StringIO):
            logger = get_logger(f"benchmark_worker_{worker_id}", config)

            start_time = time.perf_counter()
            for i in range(logs_per_worker):
                log_with_context(
                    logger,
                    "info",
                    f"Concurrent message {i}",
                    config,
                    worker_id=worker_id,
                    log_id=i,
                )
            return time.perf_counter() - start_time

    total_logs = workers * logs_per_worker
    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        worker_durations = list(executor.map(worker, range(workers)))

    total_duration = time.perf_counter() - start_time
    throughput = total_logs / total_duration

    return {
        "test": "concurrent_logging",
        "workers": workers,
        "logs_per_worker": logs_per_worker,
        "total_logs": total_logs,
        "total_duration": total_duration,
        "throughput": throughput,
        "worker_durations": worker_durations,
    }


def run_all_benchmarks():
    """Run all benchmark tests"""
    print("🚀 Running Performance Benchmarks...\n")

    results = []

    # Basic logging
    print("1️⃣  Testing basic logging...")
    result = benchmark_basic_logging()
    results.append(result)
    print(f"   {result['throughput']:,.0f} logs/sec\n")

    # Structured logging
    print("2️⃣  Testing structured logging...")
    result = benchmark_structured_logging()
    results.append(result)
    print(f"   {result['throughput']:,.0f} logs/sec\n")

    # Filtered logging
    print("3️⃣  Testing filtered logging...")
    result = benchmark_filtered_logging()
    results.append(result)
    print(f"   {result['throughput']:,.0f} logs/sec\n")

    # Concurrent logging
    print("4️⃣  Testing concurrent logging...")
    result = benchmark_concurrent_logging()
    results.append(result)
    print(f"   {result['throughput']:,.0f} logs/sec\n")

    return results


def print_summary(results):
    """Print benchmark summary"""
    print("📊 Performance Summary")
    print("=" * 50)

    for result in results:
        test_name = result["test"].replace("_", " ").title()
        throughput = result["throughput"]

        if throughput >= 50000:
            status = "🟢 EXCELLENT"
        elif throughput >= 10000:
            status = "🟡 GOOD"
        elif throughput >= 1000:
            status = "🟠 ACCEPTABLE"
        else:
            status = "🔴 POOR"

        print(f"{test_name:.<30} {throughput:>10,.0f} logs/sec {status}")

    print("\n🎯 Performance Requirements:")
    basic_result = next(r for r in results if r["test"] == "basic_logging")
    structured_result = next(r for r in results if r["test"] == "structured_logging")
    filtered_result = next(r for r in results if r["test"] == "filtered_logging")

    checks = [
        ("Basic Logging ≥ 50,000 logs/sec", basic_result["throughput"] >= 50000),
        (
            "Structured Logging ≥ 5,000 logs/sec",
            structured_result["throughput"] >= 5000,
        ),
        ("Filtered Logging ≥ 3,000 logs/sec", filtered_result["throughput"] >= 3000),
    ]

    all_passed = True
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check_name:.<40} {status}")
        if not passed:
            all_passed = False

    print(
        f"\n{'🎉 ALL REQUIREMENTS MET!' if all_passed else '⚠️  SOME REQUIREMENTS NOT MET'}"
    )


def main():
    parser = argparse.ArgumentParser(description="Run structured logging benchmarks")
    parser.add_argument("--json", type=str, help="Save results to JSON file")
    parser.add_argument(
        "--compare", type=str, help="Compare with previous results from JSON file"
    )

    args = parser.parse_args()

    results = run_all_benchmarks()
    print_summary(results)

    # Save results
    if args.json:
        report = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results}

        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Results saved to: {args.json}")

    # Compare with previous results
    if args.compare:
        try:
            with open(args.compare, "r") as f:
                previous = json.load(f)

            print(f"\n📈 Comparison with {args.compare}:")
            print("-" * 50)

            for current in results:
                test_name = current["test"]
                previous_result = next(
                    (r for r in previous["results"] if r["test"] == test_name), None
                )

                if previous_result:
                    current_throughput = current["throughput"]
                    previous_throughput = previous_result["throughput"]
                    change = (
                        (current_throughput - previous_throughput) / previous_throughput
                    ) * 100

                    if change > 5:
                        status = f"🟢 +{change:.1f}%"
                    elif change < -5:
                        status = f"🔴 {change:.1f}%"
                    else:
                        status = f"🟡 {change:+.1f}%"

                    display_name = test_name.replace("_", " ").title()
                    print(f"{display_name:.<30} {status}")

        except FileNotFoundError:
            print(f"⚠️  Comparison file not found: {args.compare}")
        except Exception as e:
            print(f"⚠️  Error reading comparison file: {e}")


if __name__ == "__main__":
    main()
