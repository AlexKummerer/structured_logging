"""
Performance tests and benchmarks for structured logging
Run with: pytest tests/performance/ -m performance
"""

import asyncio
import io
import logging
import os
import tempfile
import time
import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch

from structured_logging import (
    get_logger, 
    log_with_context,
    LoggerConfig,
    FilterConfig,
    LevelFilter,
    SamplingFilter,
    FileHandlerConfig,
    RotatingFileHandler,
    AsyncLogger,
    get_async_logger
)
from structured_logging.context import request_context
from structured_logging.performance import fast_timestamp, measure_function_performance


@pytest.mark.performance
class TestSyncLoggingPerformance:
    """Performance tests for synchronous logging"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "perf_test.log")
    
    def teardown_method(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_basic_logging_throughput(self):
        """Test basic logging throughput without filtering"""
        # Create memory-only logger for speed
        logger = logging.getLogger("perf_test")
        logger.handlers.clear()
        
        # Use StringIO for in-memory logging (faster than file)
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Benchmark logging
        iterations = 10000
        message = "Performance test message"
        
        start_time = time.perf_counter()
        for i in range(iterations):
            logger.info(f"{message} {i}")
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        throughput = iterations / duration
        
        print(f"\nBasic Logging Performance:")
        print(f"  Iterations: {iterations:,}")
        print(f"  Duration: {duration:.3f}s")
        print(f"  Throughput: {throughput:,.0f} logs/sec")
        print(f"  Log size: {len(stream.getvalue())} bytes")
        
        # Should achieve at least 20,000 logs/sec (accounting for test environment overhead)
        assert throughput >= 20000, f"Throughput {throughput:.0f} < 20,000 logs/sec"
        
        logger.handlers.clear()
    
    def test_structured_logging_throughput(self):
        """Test structured logging throughput with JSON formatting"""
        config = LoggerConfig(
            output_type="console",
            formatter_type="json",
            include_timestamp=True,
            include_request_id=True
        )
        
        # Patch sys.stdout to capture output without actual I/O
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            logger = get_logger("structured_perf", config)
            
            iterations = 5000  # Lower for structured logging due to complexity
            message = "Structured performance test"
            
            start_time = time.perf_counter()
            for i in range(iterations):
                log_with_context(logger, "info", f"{message} {i}", config, 
                               test_id=i, batch="performance")
            end_time = time.perf_counter()
            
            duration = end_time - start_time
            throughput = iterations / duration
            
            print(f"\nStructured Logging Performance:")
            print(f"  Iterations: {iterations:,}")
            print(f"  Duration: {duration:.3f}s")
            print(f"  Throughput: {throughput:,.0f} logs/sec")
            print(f"  Log size: {len(mock_stdout.getvalue())} bytes")
            
            # Should achieve at least 3,000 logs/sec for structured logging
            assert throughput >= 3000, f"Structured throughput {throughput:.0f} < 3,000 logs/sec"
    
    def test_filtering_performance_impact(self):
        """Test performance impact of filtering"""
        # Test without filtering
        config_no_filter = LoggerConfig(
            output_type="console",
            formatter_type="json"
        )
        
        # Test with filtering
        filter_config = FilterConfig(
            enabled=True,
            filters=[
                LevelFilter(min_level=logging.INFO),
                SamplingFilter(sample_rate=1.0, strategy="random")
            ]
        )
        config_with_filter = LoggerConfig(
            output_type="console",
            formatter_type="json",
            filter_config=filter_config
        )
        
        iterations = 3000
        message = "Filter performance test"
        
        # Test without filtering
        with patch('sys.stdout', new_callable=io.StringIO):
            logger1 = get_logger("no_filter_perf", config_no_filter)
            
            start_time = time.perf_counter()
            for i in range(iterations):
                log_with_context(logger1, "info", f"{message} {i}", config_no_filter)
            no_filter_duration = time.perf_counter() - start_time
            no_filter_throughput = iterations / no_filter_duration
        
        # Test with filtering
        with patch('sys.stdout', new_callable=io.StringIO):
            logger2 = get_logger("with_filter_perf", config_with_filter)
            
            start_time = time.perf_counter()
            for i in range(iterations):
                log_with_context(logger2, "info", f"{message} {i}", config_with_filter)
            with_filter_duration = time.perf_counter() - start_time
            with_filter_throughput = iterations / with_filter_duration
        
        overhead_percentage = ((with_filter_duration - no_filter_duration) / no_filter_duration) * 100
        
        print(f"\nFiltering Performance Impact:")
        print(f"  Without filtering: {no_filter_throughput:,.0f} logs/sec")
        print(f"  With filtering: {with_filter_throughput:,.0f} logs/sec")
        print(f"  Overhead: {overhead_percentage:.1f}%")
        
        # Filtering should add less than 50% overhead
        assert overhead_percentage < 50, f"Filtering overhead {overhead_percentage:.1f}% > 50%"
        
        # Should still achieve reasonable throughput with filtering
        assert with_filter_throughput >= 3000, f"Filtered throughput {with_filter_throughput:.0f} < 3,000 logs/sec"
    
    def test_file_logging_performance(self):
        """Test file logging performance"""
        file_config = FileHandlerConfig(
            filename=self.log_file,
            max_bytes=100 * 1024 * 1024,  # 100MB to avoid rotation
            compress_rotated=False,
            async_compression=False
        )
        
        config = LoggerConfig(
            output_type="file",
            file_config=file_config,
            formatter_type="json"
        )
        
        logger = get_logger("file_perf", config)
        
        iterations = 2000  # Lower for file I/O
        message = "File logging performance test"
        
        start_time = time.perf_counter()
        for i in range(iterations):
            log_with_context(logger, "info", f"{message} {i}", config, test_id=i)
        
        # Flush to ensure all logs are written
        for handler in logger.handlers:
            handler.flush()
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        throughput = iterations / duration
        
        # Check file size
        file_size = os.path.getsize(self.log_file)
        
        print(f"\nFile Logging Performance:")
        print(f"  Iterations: {iterations:,}")
        print(f"  Duration: {duration:.3f}s")
        print(f"  Throughput: {throughput:,.0f} logs/sec")
        print(f"  File size: {file_size:,} bytes")
        
        # File logging should achieve at least 1,000 logs/sec
        assert throughput >= 1000, f"File throughput {throughput:.0f} < 1,000 logs/sec"
        assert file_size > 0, "Log file should contain data"
    
    def test_concurrent_logging_performance(self):
        """Test performance under concurrent load"""
        config = LoggerConfig(
            output_type="console",
            formatter_type="json"
        )
        
        def log_worker(worker_id: int, num_logs: int) -> float:
            """Worker function for concurrent logging"""
            with patch('sys.stdout', new_callable=io.StringIO):
                logger = get_logger(f"concurrent_worker_{worker_id}", config)
                
                start_time = time.perf_counter()
                for i in range(num_logs):
                    log_with_context(logger, "info", f"Worker {worker_id} log {i}", config,
                                   worker_id=worker_id, log_id=i)
                return time.perf_counter() - start_time
        
        num_workers = 4
        logs_per_worker = 1000
        total_logs = num_workers * logs_per_worker
        
        start_time = time.perf_counter()
        
        # Run concurrent workers
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(log_worker, worker_id, logs_per_worker)
                for worker_id in range(num_workers)
            ]
            
            worker_durations = [future.result() for future in as_completed(futures)]
        
        total_duration = time.perf_counter() - start_time
        total_throughput = total_logs / total_duration
        avg_worker_duration = sum(worker_durations) / len(worker_durations)
        
        print(f"\nConcurrent Logging Performance:")
        print(f"  Workers: {num_workers}")
        print(f"  Logs per worker: {logs_per_worker:,}")
        print(f"  Total logs: {total_logs:,}")
        print(f"  Total duration: {total_duration:.3f}s")
        print(f"  Total throughput: {total_throughput:,.0f} logs/sec")
        print(f"  Avg worker duration: {avg_worker_duration:.3f}s")
        
        # Concurrent logging should achieve reasonable throughput
        assert total_throughput >= 3000, f"Concurrent throughput {total_throughput:.0f} < 3,000 logs/sec"


@pytest.mark.performance
@pytest.mark.asyncio
class TestAsyncLoggingPerformance:
    """Performance tests for async logging"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "async_perf_test.log")
    
    def teardown_method(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    async def test_async_logging_throughput(self):
        """Test async logging throughput"""
        from structured_logging.async_config import AsyncLoggerConfig
        
        async_config = AsyncLoggerConfig(
            queue_size=10000,
            batch_size=100,
            flush_interval=0.1,
            max_workers=2
        )
        
        config = LoggerConfig(
            output_type="console",
            formatter_type="json"
        )
        
        logger = get_async_logger("async_perf", config, async_config)
        
        iterations = 5000
        message = "Async performance test"
        
        # Patch stdout to avoid actual I/O
        with patch('sys.stdout', new_callable=io.StringIO):
            start_time = time.perf_counter()
            
            # Log asynchronously
            tasks = []
            for i in range(iterations):
                task = logger.ainfo(f"{message} {i}", test_id=i, batch="async_perf")
                tasks.append(task)
            
            # Wait for all logs to be queued
            await asyncio.gather(*tasks)
            
            # Flush and wait for processing
            await logger.flush()
            await asyncio.sleep(0.2)  # Allow processing to complete
            
            end_time = time.perf_counter()
        
        duration = end_time - start_time
        throughput = iterations / duration
        
        print(f"\nAsync Logging Performance:")
        print(f"  Iterations: {iterations:,}")
        print(f"  Duration: {duration:.3f}s")
        print(f"  Throughput: {throughput:,.0f} logs/sec")
        
        await logger.stop()
        
        # Async logging should achieve high throughput
        assert throughput >= 10000, f"Async throughput {throughput:.0f} < 10,000 logs/sec"
    
    async def test_async_concurrent_performance(self):
        """Test async logging with concurrent tasks"""
        from structured_logging.async_config import AsyncLoggerConfig
        
        async_config = AsyncLoggerConfig(
            queue_size=20000,
            batch_size=200,
            flush_interval=0.1,
            max_workers=4
        )
        
        config = LoggerConfig(
            output_type="console",
            formatter_type="json"
        )
        
        async def log_coroutine(coro_id: int, num_logs: int) -> None:
            """Coroutine for concurrent async logging"""
            logger = get_async_logger(f"async_concurrent_{coro_id}", config, async_config)
            
            for i in range(num_logs):
                await logger.ainfo(f"Coroutine {coro_id} log {i}", 
                                 coro_id=coro_id, log_id=i)
            
            await logger.stop()
        
        num_coroutines = 8
        logs_per_coroutine = 500
        total_logs = num_coroutines * logs_per_coroutine
        
        with patch('sys.stdout', new_callable=io.StringIO):
            start_time = time.perf_counter()
            
            # Run concurrent coroutines
            tasks = [
                log_coroutine(coro_id, logs_per_coroutine)
                for coro_id in range(num_coroutines)
            ]
            
            await asyncio.gather(*tasks)
            
            end_time = time.perf_counter()
        
        duration = end_time - start_time
        throughput = total_logs / duration
        
        print(f"\nAsync Concurrent Performance:")
        print(f"  Coroutines: {num_coroutines}")
        print(f"  Logs per coroutine: {logs_per_coroutine:,}")
        print(f"  Total logs: {total_logs:,}")
        print(f"  Duration: {duration:.3f}s")
        print(f"  Throughput: {throughput:,.0f} logs/sec")
        
        # Async concurrent should achieve very high throughput
        assert throughput >= 8000, f"Async concurrent throughput {throughput:.0f} < 8,000 logs/sec"


@pytest.mark.performance
class TestComponentPerformance:
    """Performance tests for individual components"""
    
    def test_timestamp_generation_performance(self):
        """Test timestamp generation performance"""
        iterations = 100000
        
        # Test our fast timestamp
        start_time = time.perf_counter()
        for _ in range(iterations):
            fast_timestamp()
        fast_duration = time.perf_counter() - start_time
        fast_throughput = iterations / fast_duration
        
        # Test standard datetime
        from datetime import datetime
        start_time = time.perf_counter()
        for _ in range(iterations):
            datetime.now().isoformat() + "Z"
        std_duration = time.perf_counter() - start_time
        std_throughput = iterations / std_duration
        
        improvement = (std_duration - fast_duration) / std_duration * 100
        
        print(f"\nTimestamp Generation Performance:")
        print(f"  Fast timestamp: {fast_throughput:,.0f} ops/sec")
        print(f"  Standard datetime: {std_throughput:,.0f} ops/sec")
        print(f"  Improvement: {improvement:.1f}%")
        
        # Fast timestamp should be faster
        assert fast_throughput > std_throughput, "Fast timestamp should be faster than standard"
        assert improvement > 0, "Fast timestamp should show improvement"
    
    def test_context_access_performance(self):
        """Test context variable access performance"""
        from structured_logging.context import (
            get_request_id, get_user_context, get_custom_context,
            set_request_id, set_user_context, set_custom_context
        )
        
        # Set up context
        set_request_id("test-request-123")
        set_user_context({"user_id": "user123", "tenant_id": "tenant456"})
        set_custom_context({"key1": "value1", "key2": "value2"})
        
        iterations = 50000
        
        start_time = time.perf_counter()
        for _ in range(iterations):
            req_id = get_request_id()
            user_ctx = get_user_context()
            custom_ctx = get_custom_context()
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        throughput = iterations / duration
        
        print(f"\nContext Access Performance:")
        print(f"  Iterations: {iterations:,}")
        print(f"  Duration: {duration:.3f}s")
        print(f"  Throughput: {throughput:,.0f} accesses/sec")
        
        # Context access should be very fast
        assert throughput >= 100000, f"Context access {throughput:.0f} < 100,000 ops/sec"
    
    def test_filter_evaluation_performance(self):
        """Test filter evaluation performance"""
        from structured_logging.filtering import FilterEngine, FilterConfig, LevelFilter, SamplingFilter
        
        # Create filter configuration
        filter_config = FilterConfig(
            enabled=True,
            filters=[
                LevelFilter(min_level=logging.INFO),
                SamplingFilter(sample_rate=1.0, strategy="random")
            ]
        )
        
        engine = FilterEngine(filter_config)
        
        # Create test record
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="test message", args=(), exc_info=None
        )
        
        context = {"user_id": "123", "request_id": "abc"}
        
        iterations = 10000
        
        with patch('random.random', return_value=0.5):  # Consistent sampling
            start_time = time.perf_counter()
            for _ in range(iterations):
                result = engine.should_log(record, context)
            end_time = time.perf_counter()
        
        duration = end_time - start_time
        throughput = iterations / duration
        
        print(f"\nFilter Evaluation Performance:")
        print(f"  Iterations: {iterations:,}")
        print(f"  Duration: {duration:.3f}s")
        print(f"  Throughput: {throughput:,.0f} evaluations/sec")
        
        # Filter evaluation should be fast
        assert throughput >= 40000, f"Filter evaluation {throughput:.0f} < 40,000 ops/sec"


@pytest.mark.performance
class TestMemoryUsage:
    """Memory usage tests"""
    
    def test_memory_efficiency(self):
        """Test memory usage under sustained logging"""
        try:
            import psutil
            import gc
            
            process = psutil.Process()
            gc.collect()  # Clean up before measurement
            
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create logger
            config = LoggerConfig(
                output_type="console",
                formatter_type="json"
            )
            
            with patch('sys.stdout', new_callable=io.StringIO):
                logger = get_logger("memory_test", config)
                
                # Log many messages
                iterations = 10000
                for i in range(iterations):
                    log_with_context(logger, "info", f"Memory test message {i}", config,
                                   test_id=i, iteration=i)
                    
                    # Force garbage collection every 1000 iterations
                    if i % 1000 == 0:
                        gc.collect()
            
            gc.collect()  # Clean up after logging
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            memory_increase = final_memory - initial_memory
            memory_per_log = memory_increase / iterations * 1024  # KB per log
            
            print(f"\nMemory Usage:")
            print(f"  Initial memory: {initial_memory:.1f} MB")
            print(f"  Final memory: {final_memory:.1f} MB")
            print(f"  Memory increase: {memory_increase:.1f} MB")
            print(f"  Memory per log: {memory_per_log:.3f} KB")
            
            # Memory increase should be reasonable (relaxed for test environment)
            assert memory_increase < 100, f"Memory increase {memory_increase:.1f} MB > 100 MB"
            assert memory_per_log < 5, f"Memory per log {memory_per_log:.3f} KB > 5 KB"
            
        except ImportError:
            pytest.skip("psutil not available for memory testing")


@pytest.mark.performance
class TestPerformanceRegression:
    """Tests to prevent performance regression"""
    
    def test_baseline_performance_requirements(self):
        """Test that we meet baseline performance requirements"""
        
        # Test 1: Basic logging should achieve 30,000+ logs/sec
        logger = logging.getLogger("baseline")
        logger.handlers.clear()
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        iterations = 10000
        start_time = time.perf_counter()
        for i in range(iterations):
            logger.info(f"Baseline test {i}")
        duration = time.perf_counter() - start_time
        basic_throughput = iterations / duration
        
        # Test 2: Structured logging with filtering should achieve 3,000+ logs/sec
        filter_config = FilterConfig(
            enabled=True,
            filters=[LevelFilter(min_level=logging.INFO)]
        )
        config = LoggerConfig(
            output_type="console",
            formatter_type="json",
            filter_config=filter_config
        )
        
        with patch('sys.stdout', new_callable=io.StringIO):
            logger = get_logger("structured_baseline", config)
            
            iterations = 3000
            start_time = time.perf_counter()
            for i in range(iterations):
                log_with_context(logger, "info", f"Structured baseline {i}", config)
            duration = time.perf_counter() - start_time
            structured_throughput = iterations / duration
        
        print(f"\nBaseline Performance Requirements:")
        print(f"  Basic logging: {basic_throughput:,.0f} logs/sec (required: 20,000)")
        print(f"  Structured + filtering: {structured_throughput:,.0f} logs/sec (required: 3,000)")
        
        # Verify requirements
        assert basic_throughput >= 20000, f"Basic logging {basic_throughput:.0f} < 20,000 logs/sec"
        assert structured_throughput >= 3000, f"Structured logging {structured_throughput:.0f} < 3,000 logs/sec"
        
        print("✅ All performance requirements met!")


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-m", "performance"])