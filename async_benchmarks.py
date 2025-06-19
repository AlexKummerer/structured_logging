#!/usr/bin/env python3
"""
Async performance benchmarks for structured logging library
"""

import asyncio
import time
import statistics
from contextlib import contextmanager
from typing import List, Dict, Any
import gc

from structured_logging import (
    get_logger,
    get_async_logger,
    LoggerConfig,
    AsyncLoggerConfig,
    request_context,
    async_request_context,
    log_with_context,
    alog_with_context,
    shutdown_all_async_loggers,
)


@contextmanager
def suppress_output():
    """Context manager to suppress output during benchmarks"""
    import os
    with open(os.devnull, 'w') as devnull:
        yield devnull


async def time_async_function(func, iterations: int = 1000) -> Dict[str, float]:
    """Time an async function over multiple iterations"""
    times = []
    
    for _ in range(5):  # Run 5 samples
        gc.collect()  # Clean up before each run
        start = time.perf_counter()
        
        for _ in range(iterations):
            await func()
        
        end = time.perf_counter()
        times.append((end - start) / iterations * 1000)  # ms per iteration
    
    return {
        'mean_ms': statistics.mean(times),
        'stdev_ms': statistics.stdev(times) if len(times) > 1 else 0,
        'min_ms': min(times),
        'max_ms': max(times),
        'throughput_per_sec': 1000 / statistics.mean(times)
    }


async def benchmark_sync_vs_async_basic(iterations: int = 5000):
    """Benchmark sync vs async basic logging"""
    print(f"\n=== Sync vs Async Basic Logging ({iterations:,} iterations) ===")
    
    # Sync logger
    sync_logger = get_logger("bench_sync")
    with suppress_output() as devnull:
        sync_logger.handlers[0].stream = devnull
        
        def sync_log():
            sync_logger.info("Sync test message")
        
        # Time sync logging
        sync_times = []
        for _ in range(5):
            gc.collect()
            start = time.perf_counter()
            for _ in range(iterations):
                sync_log()
            end = time.perf_counter()
            sync_times.append((end - start) / iterations * 1000)
        
        sync_stats = {
            'mean_ms': statistics.mean(sync_times),
            'throughput_per_sec': 1000 / statistics.mean(sync_times)
        }
    
    # Async logger
    async_config = AsyncLoggerConfig(batch_size=50, queue_size=2000)
    async_logger = get_async_logger("bench_async", async_config=async_config)
    async_logger.processor.stream = suppress_output().__enter__()
    
    async def async_log():
        await async_logger.ainfo("Async test message")
    
    async_stats = await time_async_function(async_log, iterations)
    
    # Wait for async processing to complete
    await asyncio.sleep(0.2)
    await async_logger.flush()
    
    print(f"Sync Logging:")
    print(f"  Mean time: {sync_stats['mean_ms']:.3f}ms per log")
    print(f"  Throughput: {sync_stats['throughput_per_sec']:.0f} logs/second")
    
    print(f"Async Logging:")
    print(f"  Mean time: {async_stats['mean_ms']:.3f}ms per log") 
    print(f"  Throughput: {async_stats['throughput_per_sec']:.0f} logs/second")
    
    improvement = async_stats['throughput_per_sec'] / sync_stats['throughput_per_sec']
    print(f"Async Improvement: {improvement:.1f}x faster")
    
    await async_logger.stop()


async def benchmark_context_propagation(iterations: int = 2000):
    """Benchmark context propagation performance"""
    print(f"\n=== Context Propagation Benchmark ({iterations:,} iterations) ===")
    
    # Sync context benchmark
    sync_logger = get_logger("bench_sync_ctx")
    with suppress_output() as devnull:
        sync_logger.handlers[0].stream = devnull
        
        def sync_context_log():
            with request_context(user_id="user123", tenant_id="tenant456"):
                log_with_context(sync_logger, "info", "Sync context message", 
                               operation="test", value=42)
        
        sync_times = []
        for _ in range(5):
            gc.collect()
            start = time.perf_counter()
            for _ in range(iterations):
                sync_context_log()
            end = time.perf_counter()
            sync_times.append((end - start) / iterations * 1000)
        
        sync_stats = {
            'mean_ms': statistics.mean(sync_times),
            'throughput_per_sec': 1000 / statistics.mean(sync_times)
        }
    
    # Async context benchmark
    async_logger = get_async_logger("bench_async_ctx")
    async_logger.processor.stream = suppress_output().__enter__()
    
    async def async_context_log():
        async with async_request_context(user_id="user123", tenant_id="tenant456"):
            await alog_with_context(async_logger, "info", "Async context message",
                                  operation="test", value=42)
    
    async_stats = await time_async_function(async_context_log, iterations)
    
    # Wait for processing
    await asyncio.sleep(0.2)
    await async_logger.flush()
    
    print(f"Sync Context Logging:")
    print(f"  Mean time: {sync_stats['mean_ms']:.3f}ms per log")
    print(f"  Throughput: {sync_stats['throughput_per_sec']:.0f} logs/second")
    
    print(f"Async Context Logging:")
    print(f"  Mean time: {async_stats['mean_ms']:.3f}ms per log")
    print(f"  Throughput: {async_stats['throughput_per_sec']:.0f} logs/second")
    
    improvement = async_stats['throughput_per_sec'] / sync_stats['throughput_per_sec']
    print(f"Async Improvement: {improvement:.1f}x faster")
    
    await async_logger.stop()


async def benchmark_concurrent_logging(concurrent_tasks: int = 10, logs_per_task: int = 100):
    """Benchmark concurrent async logging"""
    print(f"\n=== Concurrent Logging Benchmark ({concurrent_tasks} tasks, {logs_per_task} logs each) ===")
    
    async_logger = get_async_logger("bench_concurrent")
    async_logger.processor.stream = suppress_output().__enter__()
    
    async def logging_task(task_id: int):
        async with async_request_context(user_id=f"user{task_id}", task_id=task_id):
            start_time = time.perf_counter()
            
            for i in range(logs_per_task):
                await async_logger.ainfo(f"Task {task_id} log {i}")
            
            end_time = time.perf_counter()
            return end_time - start_time
    
    # Run concurrent tasks
    start_total = time.perf_counter()
    task_times = await asyncio.gather(*[
        logging_task(i) for i in range(concurrent_tasks)
    ])
    end_total = time.perf_counter()
    
    # Wait for all logs to be processed
    await asyncio.sleep(0.5)
    await async_logger.flush()
    
    total_logs = concurrent_tasks * logs_per_task
    total_time = end_total - start_total
    avg_task_time = statistics.mean(task_times)
    
    print(f"Total logs: {total_logs:,}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Average task time: {avg_task_time:.3f}s")
    print(f"Overall throughput: {total_logs / total_time:.0f} logs/second")
    print(f"Per-task throughput: {logs_per_task / avg_task_time:.0f} logs/second")
    
    await async_logger.stop()


async def benchmark_batch_sizes():
    """Benchmark different batch sizes"""
    print(f"\n=== Batch Size Optimization Benchmark ===")
    
    batch_sizes = [1, 10, 50, 100, 200]
    iterations = 1000
    
    results = {}
    
    for batch_size in batch_sizes:
        async_config = AsyncLoggerConfig(
            batch_size=batch_size,
            flush_interval=5.0,  # Long interval to force batching
            queue_size=2000
        )
        
        async_logger = get_async_logger(f"bench_batch_{batch_size}", async_config=async_config)
        async_logger.processor.stream = suppress_output().__enter__()
        
        async def batch_log():
            await async_logger.ainfo(f"Batch test message")
        
        stats = await time_async_function(batch_log, iterations)
        
        # Force flush and wait
        await async_logger.flush()
        await asyncio.sleep(0.1)
        
        results[batch_size] = stats
        
        print(f"Batch size {batch_size:3d}: {stats['throughput_per_sec']:8.0f} logs/sec "
              f"({stats['mean_ms']:.3f}ms per log)")
        
        await async_logger.stop()
    
    # Find optimal batch size
    optimal_batch = max(results.keys(), key=lambda x: results[x]['throughput_per_sec'])
    print(f"Optimal batch size: {optimal_batch} ({results[optimal_batch]['throughput_per_sec']:.0f} logs/sec)")


async def benchmark_formatter_types_async():
    """Benchmark async logging with different formatters"""
    print(f"\n=== Async Formatter Types Benchmark ===")
    
    formatters = ["json", "csv", "plain"]
    iterations = 1000
    
    for formatter_type in formatters:
        config = LoggerConfig(formatter_type=formatter_type)
        async_logger = get_async_logger(f"bench_fmt_{formatter_type}", config)
        async_logger.processor.stream = suppress_output().__enter__()
        
        async def format_log():
            async with async_request_context(user_id="user123", service="test"):
                await alog_with_context(async_logger, "info", f"{formatter_type} test",
                                      field1="value1", field2=42)
        
        stats = await time_async_function(format_log, iterations)
        
        await asyncio.sleep(0.1)
        await async_logger.flush()
        
        print(f"{formatter_type.upper():<5}: {stats['throughput_per_sec']:8.0f} logs/sec "
              f"({stats['mean_ms']:.3f}ms per log)")
        
        await async_logger.stop()


async def benchmark_memory_usage():
    """Benchmark memory usage of async logging"""
    print(f"\n=== Memory Usage Benchmark ===")
    
    import tracemalloc
    
    tracemalloc.start()
    
    # Baseline memory
    snapshot1 = tracemalloc.take_snapshot()
    
    # Create logger and log many messages
    async_config = AsyncLoggerConfig(queue_size=5000, batch_size=100)
    async_logger = get_async_logger("bench_memory", async_config=async_config)
    async_logger.processor.stream = suppress_output().__enter__()
    
    # Log many messages to test memory usage
    log_count = 10000
    async with async_request_context(user_id="memory_test_user"):
        for i in range(log_count):
            await async_logger.ainfo(f"Memory test log {i}", 
                                   batch=i // 100, 
                                   data={"key": f"value_{i}"})
    
    # Wait for processing
    await asyncio.sleep(1.0)
    await async_logger.flush()
    
    # Measure memory after logging
    snapshot2 = tracemalloc.take_snapshot()
    
    # Calculate memory usage
    current, peak = tracemalloc.get_traced_memory()
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    
    print(f"Logged {log_count:,} messages")
    print(f"Current memory: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")
    print(f"Memory per log: {current / log_count:.0f} bytes")
    
    tracemalloc.stop()
    await async_logger.stop()


async def benchmark_queue_performance():
    """Benchmark queue performance under different loads"""
    print(f"\n=== Queue Performance Benchmark ===")
    
    queue_sizes = [100, 500, 1000, 2000]
    test_load = 2000  # Number of logs to queue rapidly
    
    for queue_size in queue_sizes:
        async_config = AsyncLoggerConfig(
            queue_size=queue_size,
            batch_size=25,
            flush_interval=0.1,
            drop_on_overflow=False
        )
        
        async_logger = get_async_logger(f"bench_queue_{queue_size}", async_config=async_config)
        async_logger.processor.stream = suppress_output().__enter__()
        
        # Time rapid queueing
        start_time = time.perf_counter()
        
        tasks = []
        for i in range(test_load):
            task = async_logger.ainfo(f"Queue test {i}")
            tasks.append(task)
        
        # Wait for all to be queued
        results = await asyncio.gather(*tasks)
        queue_time = time.perf_counter() - start_time
        
        # Count successful logs
        successful = sum(1 for r in results if r)
        dropped = test_load - successful
        
        # Wait for processing
        await asyncio.sleep(0.5)
        await async_logger.flush()
        
        print(f"Queue size {queue_size:4d}: {queue_time:.3f}s to queue {test_load:,} logs "
              f"({successful:,} success, {dropped:,} dropped)")
        
        await async_logger.stop()


async def run_all_async_benchmarks():
    """Run all async performance benchmarks"""
    print("🚀 Async Structured Logging Performance Benchmarks")
    print("=" * 60)
    
    try:
        # Core performance comparisons
        await benchmark_sync_vs_async_basic(5000)
        await benchmark_context_propagation(2000)
        
        # Concurrency benchmarks
        await benchmark_concurrent_logging(10, 100)
        
        # Configuration optimization
        await benchmark_batch_sizes()
        await benchmark_formatter_types_async()
        
        # Resource usage
        await benchmark_memory_usage()
        await benchmark_queue_performance()
        
        print("\n" + "=" * 60)
        print("✅ All async benchmarks completed!")
        
    finally:
        # Ensure all async loggers are properly shut down
        await shutdown_all_async_loggers(timeout=2.0)


if __name__ == "__main__":
    asyncio.run(run_all_async_benchmarks())