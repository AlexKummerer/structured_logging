# Performance Guide

This guide provides detailed information about the performance characteristics of the Structured Logging library and optimization strategies.

## Benchmarks (Version 0.6.0)

### Core Logging Performance

| Feature | Throughput | Memory Usage | Notes |
|---------|------------|--------------|-------|
| Basic Logging | 130,000+ logs/sec | <2KB/log | JSON formatting |
| Scientific Data | 25,000+ logs/sec | <5KB/log | With NumPy arrays |
| Network Logging | 15,000+ logs/sec | <3KB/log | HTTP batched |
| Lazy Serialization | 95% improvement | Minimal | For large objects |
| Schema Validation | 45,000+ logs/sec | <3KB/log | JSON Schema |

### Scientific Computing Performance

| Data Type | Small Objects | Large Objects | Compression Ratio |
|-----------|---------------|---------------|-------------------|
| NumPy Arrays | 50,000+ logs/sec | 8,000+ logs/sec | 90%+ |
| Pandas DataFrames | 30,000+ logs/sec | 5,000+ logs/sec | 85%+ |
| SciPy Sparse Matrices | 40,000+ logs/sec | 12,000+ logs/sec | 95%+ |

### Network Handler Performance

| Handler Type | Throughput | Latency | Reliability |
|--------------|------------|---------|-------------|
| Syslog (UDP) | 20,000+ logs/sec | <1ms | Best effort |
| Syslog (TCP) | 15,000+ logs/sec | <5ms | Reliable |
| HTTP API | 10,000+ logs/sec | <10ms | With retry |
| Raw Socket | 25,000+ logs/sec | <2ms | Configurable |

## Optimization Strategies

### 1. Lazy Serialization

Enable lazy serialization for large objects that may not always be logged:

```python
from structured_logging.serializers import SerializationConfig

config = SerializationConfig(
    enable_lazy_serialization=True,
    lazy_threshold_bytes=1000,      # Objects larger than 1KB
    lazy_threshold_items=10,        # Collections with >10 items
    lazy_cache_size=500            # Cache up to 500 serialized objects
)
```

**Performance Impact**: 95% improvement for large objects when logs are filtered out.

### 2. Scientific Data Configuration

Optimize for your specific use case:

```python
# High-throughput configuration (minimal metadata)
fast_config = SerializationConfig(
    numpy_include_metadata=False,
    numpy_stats_for_numeric=False,
    pandas_include_dtypes=False,
    pandas_include_memory_usage=False
)

# High-precision configuration (detailed analysis)
precise_config = SerializationConfig(
    numpy_array_precision=8,
    numpy_include_metadata=True,
    pandas_include_describe=True,
    pandas_sample_method="systematic"
)
```

### 3. Network Handler Optimization

Configure network handlers for your throughput requirements:

```python
# High-throughput UDP logging
udp_config = SocketConfig(
    protocol="udp",
    host="fast-collector.internal",
    port=8125,
    batch_size=100,          # Large batches
    flush_interval=0.1       # Fast flushing
)

# Reliable TCP logging with connection pooling
tcp_config = SocketConfig(
    protocol="tcp",
    connection_pool_size=10,  # Reuse connections
    batch_size=50,
    flush_interval=1.0
)
```

### 4. Smart Filtering

Use appropriate filtering to reduce processing overhead:

```python
from structured_logging import FilterConfig, SamplingFilter, LevelFilter

# Production filtering (sample DEBUG, keep errors)
filter_config = FilterConfig(
    enabled=True,
    filters=[
        LevelFilter(min_level="INFO"),
        SamplingFilter(
            sample_rate=0.01,           # 1% of DEBUG logs
            strategy="hash",            # Consistent sampling
            burst_levels=["ERROR", "CRITICAL"],
            burst_allowance=100         # All errors/critical logs
        )
    ]
)
```

## Memory Management

### Memory Usage Patterns

- **Base overhead**: ~1KB per logger instance
- **Context variables**: ~100 bytes per request context
- **Scientific objects**: Varies by data size and configuration
- **Network buffers**: Configurable (default 1MB per handler)

### Memory Optimization

1. **Limit array serialization**:
   ```python
   config = SerializationConfig(
       numpy_array_max_size=50,        # Compress larger arrays
       numpy_compression_threshold=100  # Aggressive compression
   )
   ```

2. **Use systematic sampling for DataFrames**:
   ```python
   config = SerializationConfig(
       pandas_max_rows=10,
       pandas_sample_method="systematic"  # Memory efficient
   )
   ```

3. **Configure network buffers**:
   ```python
   http_config = HTTPConfig(
       buffer_size=512,      # Smaller buffer
       batch_size=20,        # Frequent flushing
       max_batch_time=1.0
   )
   ```

## Profiling and Monitoring

### Built-in Performance Monitoring

Enable performance tracking:

```python
from structured_logging.serializers import get_performance_stats

# Get serialization performance statistics
stats = get_performance_stats()
print(f"Total serializations: {stats['total_count']}")
print(f"Average time: {stats['avg_time_ms']:.2f}ms")
print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
```

### Custom Profiling

Profile your specific use case:

```python
import time
from structured_logging import get_logger

logger = get_logger("performance_test")

# Measure logging performance
start_time = time.perf_counter()
for i in range(10000):
    logger.info(f"Test message {i}", data={"value": i})
end_time = time.perf_counter()

throughput = 10000 / (end_time - start_time)
print(f"Throughput: {throughput:.0f} logs/second")
```

## Benchmarking Scripts

The library includes comprehensive benchmarking tools:

```bash
# Run all benchmarks
python scripts/benchmark_comparison.py

# Performance regression testing
python scripts/run_performance_tests.py

# Scientific data benchmarks
python benchmarks.py --scientific

# Network handler benchmarks
python benchmarks.py --network
```

## Performance Tips by Use Case

### High-Throughput Applications

```python
config = SerializationConfig(
    enable_lazy_serialization=True,
    numpy_include_metadata=False,
    pandas_include_dtypes=False,
    auto_detect_types=False,        # Disable for speed
    validate_schemas=False          # Skip validation
)
```

### Scientific Computing

```python
config = SerializationConfig(
    numpy_compression_threshold=500,
    pandas_sample_method="systematic",
    enable_lazy_serialization=True,
    lazy_threshold_bytes=5000       # Large threshold for arrays
)
```

### Network Logging

```python
# Batched HTTP logging
http_config = HTTPConfig(
    batch_size=100,                 # Large batches
    max_batch_time=5.0,            # Infrequent flushing
    compress_payload=True,          # Network compression
    fallback_to_local=True         # Reliability
)
```

### Memory-Constrained Environments

```python
config = SerializationConfig(
    numpy_array_max_size=10,        # Aggressive compression
    pandas_max_rows=3,              # Minimal data
    enable_lazy_serialization=True,
    lazy_cache_size=50,             # Small cache
    truncate_strings=100            # Limit string size
)
```

## Performance Regression Testing

The library includes automated performance regression testing:

```bash
# Run performance tests (excluded from regular tests)
pytest -m performance

# Generate performance report
python scripts/performance_report.py
```

Performance thresholds are automatically validated to ensure no regressions in new versions.

## Version Performance History

| Version | Basic Logging | Scientific | Network | Notes |
|---------|---------------|------------|---------|-------|
| 0.6.0   | 130,000/sec   | 25,000/sec | 15,000/sec | Scientific integration |
| 0.5.0   | 130,000/sec   | N/A        | N/A     | Filtering optimization |
| 0.4.0   | 110,000/sec   | N/A        | N/A     | Async support |
| 0.3.0   | 85,000/sec    | N/A        | N/A     | Enhanced serialization |

## Troubleshooting Performance Issues

### Common Issues and Solutions

1. **Slow serialization of large arrays**:
   - Enable compression: `numpy_compression_threshold=100`
   - Use lazy serialization: `enable_lazy_serialization=True`

2. **High memory usage with DataFrames**:
   - Reduce sampling: `pandas_max_rows=5`
   - Use systematic sampling: `pandas_sample_method="systematic"`

3. **Network handler bottlenecks**:
   - Increase batch size: `batch_size=100`
   - Use connection pooling: `connection_pool_size=5`

4. **Context switching overhead**:
   - Reuse loggers: Cache logger instances
   - Minimize context updates: Use `request_context()` efficiently

For additional performance optimization, see the examples in `examples/performance_optimization.py`.