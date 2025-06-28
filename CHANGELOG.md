# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] - In Development

### Added - Stream Processing
- **Core Stream Processor**: Real-time log processing with async pipeline, backpressure handling, and state management
- **Window Operations**: Tumbling, sliding, and session windows with configurable aggregations
- **Stream Sources**: File tailing, WebSocket, HTTP polling, memory, and Kafka placeholder
- **Stream Sinks**: Console, metrics export, WebSocket broadcast, and storage persistence

### Added - Real-time Monitoring Dashboard
- **Web Dashboard**: Real-time log visualization with WebSocket support for live updates
- **Metrics Aggregation**: Time-series metrics with automatic aggregation and retention
- **Alert Management**: Rule-based alerting with rate limiting, cooldowns, and multiple notification channels
- **Visualizations**: Pre-built charts including log volume, metrics trends, error heatmaps, and KPI gauges

## [0.7.0] - 2025-06-28

### Added - Cloud Platform Integration
- **AWS CloudWatch**: Native integration with automatic batching, compression, and retry logic
- **Google Cloud Logging**: Stackdriver integration with resource types and structured logging
- **Azure Monitor**: Log Analytics, Application Insights, and DCE integration with managed identity
- **OpenTelemetry Integration**: Distributed tracing with automatic context propagation

### Added - Framework Integrations
- **Django Integration**: Middleware with request/response logging, database query tracking, and signal integration
- **aiohttp Integration**: Async middleware with WebSocket support and client request logging
- **Celery Integration**: Task execution logging with retry tracking, worker events, and correlation ID propagation
- **SQLAlchemy Integration**: Query execution logging with timing, slow query detection, connection pool monitoring, and ORM event tracking

### Added - Advanced Analytics
- **Pattern Detection**: Automatic discovery of common patterns including error patterns, sequential operations, field value patterns, and frequency-based patterns
- **Anomaly Detection**: Multi-method anomaly detection using statistical analysis, pattern deviation, frequency anomalies, and value anomalies
- **Performance Metrics Collection**: Comprehensive metrics including latency analysis, error rates, throughput metrics, and user activity tracking

## [0.6.1] - 2025-06-28

### 🐛 Bug Fixes
- **Logger Configuration**: Fixed logger reconfiguration when config is explicitly provided
- **Handler Logic**: Updated handler logic to properly support all output_type values ("console", "file", "both", "network", etc.)
- **NumPy Serialization**: Fixed NumPy scalar serialization to support metadata inclusion with `numpy_include_metadata` config
- **Pandas Compatibility**: Fixed deprecation warnings by replacing `is_categorical_dtype` with `isinstance(dtype, pd.CategoricalDtype)`
- **Network Handlers**: Fixed network handler tests by correcting import errors and using proper exception types
- **Type Detection**: Fixed tests to handle type-detected masked values in FastAPI integration
- **Validation**: Added missing `reset_validation_stats` method to SchemaValidator class

### ♻️ Code Improvements
- **Major Refactoring**: Completed comprehensive refactoring to comply with programming guidelines
  - Split large modules into well-organized packages: `formatter/`, `filtering/`, `handlers/`
  - Refactored 87+ functions to follow Single Responsibility Principle (SRP)
  - Fixed all critical import violations (all imports now at top of files)
  - Updated programming guidelines to allow functions >20 lines if they follow SRP
- **Test Coverage**: Improved test coverage from ~80% to 84%
- **Code Organization**: Better separation of concerns with modular package structure

### 📚 Documentation
- Updated PROGRAMMING_GUIDELINES.md with SRP exceptions and examples
- Maintained comprehensive CLAUDE.md for AI-assisted development

## [0.6.0] - 2025-06-19

### 🧬 Scientific Computing & Network Integration

#### Added - Scientific Data Support
- **NumPy Integration**: Native support for arrays and scalars with intelligent serialization
  - Enhanced scalar serialization with metadata and precision control
  - Advanced array serialization with 3 compression strategies (full/summary/compressed)
  - Multi-dimensional array support with shape, dtype, and memory information
  - Special value handling (NaN, +inf, -inf) with statistics
  - Masked array support for `numpy.ma.MaskedArray` objects
  - NumPy 2.0 compatibility fixes

- **Pandas Integration**: Advanced DataFrame and Series serialization
  - Enhanced DataFrame serialization with memory usage, dtypes, and statistical summaries
  - Column sampling for wide DataFrames with configurable limits
  - Multiple sampling methods: head_tail, random, systematic
  - Enhanced Series serialization with data type detection and analysis
  - Categorical data support with codes/categories serialization options
  - Index type handling (DatetimeIndex, MultiIndex, custom indexes)
  - Timestamp serialization with multiple format options

- **SciPy Integration**: Sparse matrix and scientific computing support
  - Sparse matrix serialization with density calculation and memory info
  - Format detection (CSR, CSC, COO) with comprehensive metadata
  - Memory-efficient sample data extraction for large sparse matrices

#### Added - Network Handlers
- **Syslog Integration**: RFC 3164/5424 compliant syslog handlers
  - RFC 3164 (traditional) and RFC 5424 (modern) format support
  - SSL/TLS encryption with certificate verification
  - Configurable facilities and severities
  - Hostname and process ID injection

- **HTTP API Logging**: Enterprise-grade HTTP log transmission
  - Multiple authentication methods (Bearer, Basic, API Key)
  - Batch processing with configurable sizes and timeouts
  - SSL/TLS support with custom headers
  - Payload compression and retry logic with exponential backoff

- **Raw Socket Logging**: High-performance TCP/UDP logging
  - TCP logging with connection pooling and keep-alive
  - UDP logging with configurable buffer sizes
  - Message delimiters and encoding options
  - Connection management and error handling

#### Added - Intelligent Type Detection
- **Automatic Detection**: Smart identification and conversion of complex data types
  - DateTime string detection and conversion
  - UUID string detection and validation
  - JSON string parsing and enhancement
  - URL string validation and metadata extraction
  - Custom pattern detection with confidence scoring

- **Type Enhancement**: Enrichment of detected types with metadata
  - Detection confidence scoring
  - Original value preservation
  - Type-specific validation and formatting
  - Caching for performance optimization

#### Added - Lazy Serialization
- **Memory Efficiency**: Serialize only when logs are actually written
  - Configurable size and item thresholds
  - LRU caching for frequently accessed objects
  - Memory usage monitoring and statistics
  - 95% performance improvement for large objects

- **Streaming Serialization**: Handle massive datasets without memory loading
  - Lazy evaluation of expensive operations
  - Configurable cache sizes and TTL
  - Memory pressure detection and management

#### Added - Schema Validation
- **Runtime Validation**: JSON Schema integration with automatic generation
  - Schema registration and management system
  - Type annotation-based schema creation
  - Flexible validation modes (strict/warn/disabled)
  - Custom validator functions and error handling

- **Schema Generation**: Automatic schema creation from Python types
  - Function signature analysis
  - Class attribute inspection
  - Nested object schema generation
  - Schema versioning and evolution

#### Added - Configuration System
- **SerializationConfig**: 20+ new configuration options for scientific data
  - NumPy: array limits, precision, compression thresholds, metadata options
  - Pandas: row/column limits, sampling methods, dtype inclusion
  - Type Detection: enable/disable specific detectors, confidence thresholds
  - Lazy Serialization: size thresholds, cache configuration
  - Schema Validation: validation modes, custom validators

#### Added - Examples and Documentation
- **Scientific Examples** (`examples/scientific_data_examples.py`): 500+ lines
  - NumPy scalar and array serialization with real-world scenarios
  - Pandas DataFrame/Series logging with performance optimizations
  - SciPy sparse matrix integration examples
  - Climate data analysis workflow demonstration

- **Network Examples** (`examples/advanced_network_examples.py`): 600+ lines
  - Complete Syslog integration examples with SSL configuration
  - HTTP API logging with all authentication methods
  - Production multi-handler setup patterns
  - Async compatibility and monitoring integration

- **Performance Documentation** (`docs/PERFORMANCE.md`)
  - Detailed benchmarks for all feature categories
  - Memory optimization strategies for scientific computing
  - Network handler performance characteristics
  - Profiling and monitoring tools documentation

#### Changed
- **Serialization Order**: Custom serializers now checked before primitive types
- **Import Compliance**: All imports moved to global scope following programming guidelines
- **Type Registry**: Enhanced with scientific type support and extensible architecture
- **README**: Comprehensive update with Version 0.6.0 features and examples

#### Performance
- **Scientific Data**: 25,000+ logs/second with NumPy arrays
- **Network Logging**: 15,000+ logs/second with HTTP batching
- **Lazy Serialization**: 95% performance improvement for large objects
- **Schema Validation**: 45,000+ logs/second with validation enabled
- **Memory Efficiency**: Intelligent compression and sampling for massive datasets

#### Testing
- **Scientific Integration**: 23 comprehensive tests with 100% pass rate
- **Network Handlers**: Production-ready reliability and error handling tests
- **Type Detection**: Extensive test coverage for all detection patterns
- **Schema Validation**: Comprehensive validation and error handling tests
- **Performance**: Automated regression testing for all new features

#### Technical Enhancements
- **NumPy 2.0 Compatibility**: Removed deprecated `np.unicode_` references
- **SciPy Density Calculation**: Fixed sparse matrix density using total vs stored elements
- **Method Signatures**: Converted static methods to instance methods for proper registry
- **Error Handling**: Graceful fallbacks for optional dependencies and serialization errors

## [0.5.0] - 2025-06-19

### 🎉 Production-Ready Performance & Framework Integration

#### Added - Advanced Log Filtering
- **LevelFilter**: Filter logs by minimum level with configurable thresholds
- **ContextFilter**: Filter based on required context fields
- **SamplingFilter**: Intelligent sampling with multiple strategies
  - Random sampling with configurable rates
  - Hash-based sampling for consistent results
  - Rate limiting with burst allowance for errors
- **CustomFilter**: User-defined filter functions with full control

#### Added - FastAPI Integration
- **One-line Middleware**: `add_structured_logging()` for instant integration
- **FastAPILoggingConfig**: Comprehensive configuration options
- **Request/Response Logging**: Automatic HTTP request and response capture
- **Sensitive Data Masking**: Automatic masking of passwords, tokens, and custom patterns
- **Performance Optimization**: Configurable minimum duration and path exclusions
- **Context Injection**: Automatic request ID and user context propagation

#### Added - File Handlers
- **RotatingFileHandler**: Automatic file rotation with size limits
- **Gzip Compression**: Transparent compression of rotated files
- **Async Compression**: Non-blocking compression for better performance
- **Archive Management**: Configurable backup count and retention policies
- **Custom Naming**: Flexible file naming patterns and timestamps

#### Added - Smart Sampling
- **Multiple Strategies**: Random, hash-based, and rate-limiting sampling
- **Configurable Rates**: Per-level sampling with burst protection
- **Consistent Sampling**: Hash-based sampling for reproducible results
- **Burst Allowance**: Allow all ERROR/CRITICAL logs during incidents
- **Performance Monitoring**: Sampling statistics and throughput metrics

#### Performance Achievements
- **130,000+ logs/second**: Maintained high throughput with filtering enabled
- **Memory Efficient**: <5KB per log entry with compression
- **Filtering Overhead**: <0.001ms additional latency per filter
- **Async File I/O**: Non-blocking file operations for better concurrency

#### Developer Experience
- **50+ New Tests**: Comprehensive test coverage for all filtering features
- **Performance Benchmarks**: Automated performance regression testing
- **Production Examples**: Real-world configuration patterns and best practices
- **Framework Documentation**: Complete integration guides for FastAPI

## [0.4.0] - 2025-06-19

### 🚀 Async Excellence

#### Added - Async Logging Infrastructure
- **AsyncLogger**: Complete async logging class with queue-based processing
- **Async Context Management**: `async_request_context()` for async-aware context propagation
- **Queue-Based Processing**: Background log processing with configurable batching
- **Async Configuration**: `AsyncLoggerConfig` for fine-tuning async performance
- **Concurrent Logging**: Support for high-concurrency async applications
- **Graceful Shutdown**: Proper async logger lifecycle management

#### New API Components
- **AsyncLogger**: Main async logging class with `ainfo()`, `aerror()`, etc. methods
- **get_async_logger()**: Factory function for creating async loggers
- **async_request_context()**: Async context manager for request-scoped logging
- **alog_with_context()**: Async version of context-aware logging
- **AsyncLoggerConfig**: Configuration for queue sizes, batching, and performance tuning
- **shutdown_all_async_loggers()**: Cleanup function for graceful shutdown

#### Performance Features
- **Background Processing**: Non-blocking log queuing with background workers
- **Configurable Batching**: Optimal batch sizes for different workloads
- **Memory Management**: Controlled memory usage with configurable queue limits
- **Error Handling**: Async error callbacks and overflow protection

#### Performance Benchmarks
- **Concurrent Throughput**: 40,153+ logs/second across 10 concurrent tasks
- **Memory Efficient**: ~923 bytes per log entry
- **Optimal Batching**: Best performance at batch size 200 (99,523 logs/sec)
- **Low Latency**: ~0.010ms per async log call

#### Backward Compatibility
- **Full Compatibility**: All existing sync APIs remain unchanged
- **Coexistence**: Sync and async loggers can be used together
- **Shared Configuration**: Same formatter and context systems
- **No Breaking Changes**: Seamless migration path

## [0.3.0] - 2025-06-19

### ⚡ Performance Optimizations

#### Added - Performance Enhancements
- **Fast Timestamp Generation**: Micro-caching with 1ms cache duration
- **Formatter Caching**: Instance caching to reduce initialization overhead
- **Optimized Context Access**: Reduced context variable lookups
- **Lazy Evaluation**: Expensive operations only executed when needed
- **Memory Optimization**: Efficient memory usage patterns

#### Performance Improvements
- **Basic Logging**: 131,920+ logs/second (significant improvement)
- **Context Logging**: 55,801+ logs/second (optimized from baseline)
- **Timestamp Overhead**: Reduced to < 0.001ms per log
- **Memory Efficiency**: < 10MB for 1,000 structured logs
- **Context Overhead**: Optimized to ~0.010ms per context access

#### Technical Enhancements
- **Performance Tests**: 11 new performance validation tests
- **Benchmark Tools**: Comprehensive performance analysis utilities
- **Regression Prevention**: Automated performance threshold testing
- **Profiling Support**: Built-in performance measurement utilities

## [0.2.0] - 2024-12-19

### 📄 Multiple Output Formats

#### Added - Formatter Variety
- **CSVFormatter**: Machine-readable CSV format for data analysis
- **PlainTextFormatter**: Human-readable format for development
- **Formatter Selection**: `formatter_type` parameter ("json", "csv", "plain")
- **Environment Configuration**: `STRUCTURED_LOG_FORMATTER` environment variable
- **Type Safety**: FormatterType literal type for formatter selection

#### Features
- **CSV Format**: Structured tabular output for log analysis tools
- **Plain Text Format**: Readable format with context in parentheses
- **Backward Compatible**: All existing JSON functionality unchanged
- **Environment Driven**: Configure formatter via environment variables

## [0.1.0] - 2024-12-19

### 🎯 Foundation Release

#### Added - Core Features
- **StructuredFormatter**: JSON output with configurable fields
- **LoggerConfig**: Flexible configuration with environment variables
- **Context Management**: Thread-safe context variables using contextvars
- **Request Context**: Context manager for request-scoped logging
- **Automatic Context Injection**: Seamless integration of context data

#### Features
- **JSON Formatter**: Structured logging with timestamp and context
- **Request ID Tracking**: Automatic UUID generation and propagation
- **User Context**: Support for user_id, tenant_id, and custom fields
- **Environment Configuration**: Configure via environment variables
- **Type Hints**: Full type safety throughout codebase

#### Developer Experience
- **Python 3.13+ Support**: Modern Python features
- **98% Test Coverage**: Comprehensive test suite
- **Development Tools**: Black, Ruff, MyPy integration
- **Documentation**: Comprehensive README and examples

## Version Support & Compatibility

### Python Requirements
- **Python**: 3.13+ (leveraging newest features)
- **Dependencies**: Pure Python standard library (no external dependencies)
- **Optional Dependencies**: `numpy`, `pandas`, `scipy` for scientific data support

### Backward Compatibility Promise
- **Minor Versions**: No breaking changes in minor versions (0.x.y)
- **Major Versions**: Clear migration path with deprecation warnings
- **API Stability**: Core APIs remain stable across versions

## Migration Guide

### From 0.5.x to 0.6.x
```python
# Scientific data now serialized automatically
import numpy as np
import pandas as pd
from structured_logging import get_logger
from structured_logging.serializers import SerializationConfig

# Configure for scientific data
config = SerializationConfig(
    numpy_include_metadata=True,
    pandas_max_rows=10
)

logger = get_logger("science_app", config=config)

# NumPy arrays and Pandas DataFrames now work automatically
logger.info("Experiment results", 
           measurements=np.random.randn(1000),
           data=pd.DataFrame({'x': [1, 2, 3]}))
```

### From 0.4.x to 0.5.x
```python
# FastAPI integration is now one-line
from fastapi import FastAPI
from structured_logging.integrations import add_structured_logging

app = FastAPI()
app = add_structured_logging(app)  # One line!
```

### From Legacy Logging
```python
# Old way
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.info("User action", extra={"user_id": "123"})

# New way with structured logging
from structured_logging import get_logger, request_context
logger = get_logger(__name__)
with request_context(user_id="123"):
    logger.info("User action")
```

## Performance Evolution

| Version | Basic Logging | Advanced Features | Notes |
|---------|---------------|-------------------|-------|
| 0.1.0   | ~50,000/sec   | N/A              | Foundation |
| 0.2.0   | ~75,000/sec   | N/A              | Multiple formats |
| 0.3.0   | 131,920/sec   | N/A              | Performance optimization |
| 0.4.0   | 131,920/sec   | 40,153/sec (async) | Async support |
| 0.5.0   | 130,000/sec   | 54,000/sec (filtering) | Production features |
| 0.6.0   | 130,000/sec   | 25,000/sec (scientific) | Scientific computing |

---

**Current Release: Version 0.6.0 - Scientific Computing & Network Integration Leader** 🧬🌐⚡

*Next Release: Version 0.7.0 - Cloud Platform Integration* ☁️🚀