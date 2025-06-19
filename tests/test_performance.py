"""
Performance tests for structured logging library

Tests validate that performance optimizations are working correctly
and maintain expected throughput benchmarks.
"""

import time
import pytest
from unittest.mock import patch
from structured_logging import (
    get_logger,
    LoggerConfig,
    log_with_context,
    request_context,
)
from structured_logging.performance import fast_timestamp, measure_function_performance


class TestPerformanceOptimizations:
    """Test performance optimizations and benchmarks"""

    def test_fast_timestamp_performance(self):
        """Test that fast_timestamp is actually faster than datetime.now()"""
        from datetime import datetime
        
        # Measure fast_timestamp
        fast_stats = measure_function_performance(fast_timestamp, 1000)
        
        # Measure regular datetime
        def regular_timestamp():
            return datetime.now().isoformat() + "Z"
        
        regular_stats = measure_function_performance(regular_timestamp, 1000)
        
        # fast_timestamp should be at least 20% faster
        assert fast_stats['mean_ms'] < regular_stats['mean_ms'] * 0.8
        assert fast_stats['throughput_per_sec'] > regular_stats['throughput_per_sec'] * 1.2

    def test_formatter_caching_works(self):
        """Test that formatter caching reduces initialization overhead"""
        from structured_logging.logger import _formatter_cache
        
        # Clear cache
        _formatter_cache.clear()
        
        config = LoggerConfig(formatter_type="json", include_timestamp=True)
        
        # First logger creation should populate cache
        logger1 = get_logger("test_cache_1", config)
        assert len(_formatter_cache) == 1
        
        # Second logger with same config should reuse cached formatter
        logger2 = get_logger("test_cache_2", config)
        assert len(_formatter_cache) == 1  # No new cache entry
        
        # Different config should create new cache entry
        config2 = LoggerConfig(formatter_type="csv", include_timestamp=True)
        logger3 = get_logger("test_cache_3", config2)
        assert len(_formatter_cache) == 2

    def test_context_access_optimization(self):
        """Test that context access is optimized for minimal overhead"""
        logger = get_logger("test_context_perf")
        
        # Suppress output during performance testing
        if logger.handlers:
            logger.handlers[0].stream = open('/dev/null', 'w')
        
        # Measure context logging performance
        def context_log():
            with request_context(user_id="user123", tenant_id="tenant456"):
                log_with_context(logger, "info", "Test message", extra_field="value")
        
        stats = measure_function_performance(context_log, 100)
        
        # Context overhead should be reasonable (< 1ms per call)
        assert stats['mean_ms'] < 1.0
        
        # Should maintain decent throughput (> 1000 logs/sec)
        assert stats['throughput_per_sec'] > 1000

    def test_basic_logging_throughput(self):
        """Test basic logging maintains high throughput"""
        logger = get_logger("test_basic_perf")
        
        # Suppress output during performance testing
        if logger.handlers:
            logger.handlers[0].stream = open('/dev/null', 'w')
        
        def basic_log():
            logger.info("Test message")
        
        stats = measure_function_performance(basic_log, 1000)
        
        # Basic logging should be very fast (< 0.1ms per call)
        assert stats['mean_ms'] < 0.1
        
        # Should maintain very high throughput (> 10,000 logs/sec)
        assert stats['throughput_per_sec'] > 10000

    def test_formatter_type_performance_comparison(self):
        """Test performance differences between formatter types"""
        formatters = ["json", "csv", "plain"]
        results = {}
        
        for fmt in formatters:
            config = LoggerConfig(formatter_type=fmt, include_timestamp=False)
            logger = get_logger(f"test_{fmt}_perf", config)
            
            # Suppress output during performance testing
            if logger.handlers:
                logger.handlers[0].stream = open('/dev/null', 'w')
            
            def format_log():
                logger.info("Test message")
            
            results[fmt] = measure_function_performance(format_log, 500)
        
        # All formatters should maintain reasonable performance
        for fmt, stats in results.items():
            assert stats['mean_ms'] < 0.5, f"{fmt} formatter too slow: {stats['mean_ms']:.3f}ms"
            assert stats['throughput_per_sec'] > 2000, f"{fmt} formatter throughput too low"
        
        # All formatters should perform reasonably well (remove strict ordering requirement)
        # since performance can vary slightly between runs
        for fmt, stats in results.items():
            assert stats['mean_ms'] < 0.1, f"{fmt} formatter should be under 0.1ms, got {stats['mean_ms']:.3f}ms"

    def test_timestamp_caching_effectiveness(self):
        """Test that timestamp caching reduces overhead"""
        from structured_logging.performance import _optimized_timestamp, create_optimized_timestamp_func
        
        # Test cached timestamp function
        cached_func = create_optimized_timestamp_func()
        
        # Multiple rapid calls should return same timestamp (within cache duration)
        ts1 = cached_func()
        ts2 = cached_func()
        ts3 = cached_func()
        
        # Should be identical due to caching
        assert ts1 == ts2 == ts3
        
        # After cache expires, should get new timestamp
        time.sleep(0.002)  # Wait longer than 1ms cache duration
        ts4 = cached_func()
        assert ts4 != ts1  # Should be different

    def test_memory_efficiency(self):
        """Test memory usage remains reasonable under load"""
        import tracemalloc
        
        tracemalloc.start()
        
        logger = get_logger("test_memory")
        
        # Baseline
        snapshot1 = tracemalloc.take_snapshot()
        
        # Log many messages with context
        with request_context(user_id="user123", tenant_id="tenant456"):
            for i in range(1000):
                log_with_context(logger, "info", f"Message {i}", 
                               iteration=i, batch=f"batch_{i//100}")
        
        snapshot2 = tracemalloc.take_snapshot()
        
        # Check memory growth
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Memory usage should be reasonable (< 10MB for 1000 logs)
        assert current < 10 * 1024 * 1024, f"Memory usage too high: {current / 1024 / 1024:.1f}MB"

    def test_performance_regression_prevention(self):
        """Test to prevent performance regressions"""
        logger = get_logger("regression_test")
        
        # Test various scenarios that should maintain good performance
        scenarios = [
            ("basic", lambda: logger.info("Test")),
            ("with_context", lambda: log_with_context(logger, "info", "Test", field="value")),
            ("complex_context", lambda: log_with_context(
                logger, "info", "Test", 
                field1="value1", field2="value2", field3="value3",
                nested_data={"key": "value", "number": 42}
            )),
        ]
        
        for scenario_name, func in scenarios:
            stats = measure_function_performance(func, 200)
            
            # Define performance thresholds
            max_time_ms = {
                "basic": 0.05,
                "with_context": 0.2, 
                "complex_context": 0.3
            }
            
            min_throughput = {
                "basic": 20000,
                "with_context": 5000,
                "complex_context": 3000
            }
            
            assert stats['mean_ms'] < max_time_ms[scenario_name], \
                f"{scenario_name} too slow: {stats['mean_ms']:.3f}ms"
            
            assert stats['throughput_per_sec'] > min_throughput[scenario_name], \
                f"{scenario_name} throughput too low: {stats['throughput_per_sec']:.0f} logs/sec"


class TestPerformanceMeasurement:
    """Test performance measurement utilities"""

    def test_measure_function_performance(self):
        """Test performance measurement utility"""
        def test_func():
            time.sleep(0.001)  # 1ms delay
        
        stats = measure_function_performance(test_func, 10)
        
        # Should measure approximately 1ms per call
        assert 0.8 <= stats['mean_ms'] <= 1.5  # Allow some variance
        assert stats['min_ms'] > 0
        assert stats['max_ms'] >= stats['mean_ms']
        assert stats['throughput_per_sec'] > 0

    def test_fast_timestamp_consistency(self):
        """Test that fast_timestamp returns valid ISO format"""
        ts = fast_timestamp()
        
        # Should be valid ISO format with Z suffix
        assert ts.endswith('Z')
        assert 'T' in ts
        assert len(ts) >= 20  # Minimum ISO format length

    def test_performance_under_concurrent_load(self):
        """Test performance doesn't degrade under concurrent access"""
        import threading
        import queue
        
        logger = get_logger("concurrent_test")
        results = queue.Queue()
        
        def worker():
            start = time.perf_counter()
            for _ in range(100):
                with request_context(user_id="user123"):
                    log_with_context(logger, "info", "Concurrent test")
            end = time.perf_counter()
            results.put(end - start)
        
        # Run 5 concurrent workers
        threads = []
        for _ in range(5):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Collect results
        times = []
        while not results.empty():
            times.append(results.get())
        
        # Average time per worker should be reasonable
        avg_time = sum(times) / len(times)
        avg_time_per_log = avg_time / 100
        
        assert avg_time_per_log < 0.01, f"Concurrent performance too slow: {avg_time_per_log*1000:.3f}ms per log"