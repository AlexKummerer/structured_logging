# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Nothing yet

### Changed
- Nothing yet

### Deprecated
- Nothing yet

### Removed
- Nothing yet

### Fixed
- Nothing yet

### Security
- Nothing yet

## [0.5.0] - TBD (In Planning)

### Planned Features
- **Log Filtering & Sampling**: Smart filtering and rate-limiting for high-volume logs
- **FastAPI Integration**: One-line middleware for automatic request/response logging
- **File Handler**: Rotating file handler with compression and archiving
- **Network Handlers**: Send logs to remote systems (syslog, HTTP endpoints)
- **Enhanced Data Types**: Support for complex data structures and validation

### Goals
- Maintain >50,000 logs/second performance with filtering enabled
- Framework integrations for major Python web frameworks
- Production-ready file and network output handlers
- Advanced filtering and sampling capabilities

## [0.4.0] - 2025-06-19

### Added
- **Async Logging Support**: Complete async logging infrastructure for high-performance applications
- **AsyncLogger**: Non-blocking async logger with queue-based processing
- **Async Context Management**: `async_request_context()` for async-aware context propagation
- **Queue-Based Processing**: Background log processing with configurable batching
- **Async Configuration**: `AsyncLoggerConfig` for fine-tuning async performance
- **Concurrent Logging**: Support for high-concurrency async applications
- **Graceful Shutdown**: Proper async logger lifecycle management

### New API Components
- **AsyncLogger**: Main async logging class with `ainfo()`, `aerror()`, etc. methods
- **get_async_logger()**: Factory function for creating async loggers
- **async_request_context()**: Async context manager for request-scoped logging
- **alog_with_context()**: Async version of context-aware logging
- **AsyncLoggerConfig**: Configuration for queue sizes, batching, and performance tuning
- **shutdown_all_async_loggers()**: Cleanup function for graceful shutdown

### Performance Features
- **Background Processing**: Non-blocking log queuing with background workers
- **Configurable Batching**: Optimal batch sizes for different workloads
- **Memory Management**: Controlled memory usage with configurable queue limits
- **Error Handling**: Async error callbacks and overflow protection
- **Framework Integration**: Ready for FastAPI, aiohttp, and other async frameworks

### Backward Compatibility
- **Full Compatibility**: All existing sync APIs remain unchanged
- **Coexistence**: Sync and async loggers can be used together
- **Shared Configuration**: Same formatter and context systems
- **No Breaking Changes**: Seamless migration path

### Performance Benchmarks
- **Concurrent Throughput**: 40,153+ logs/second across 10 concurrent tasks
- **Memory Efficient**: ~923 bytes per log entry
- **Optimal Batching**: Best performance at batch size 200 (99,523 logs/sec)
- **Low Latency**: ~0.010ms per async log call

### Developer Experience
- **20 New Tests**: Comprehensive async test coverage
- **Documentation**: Complete async API documentation and examples
- **Example Code**: Real-world async logging examples
- **Performance Tools**: Async benchmarking utilities

## [0.3.0] - 2025-06-19

### Added
- **Performance Optimizations**: Significant throughput improvements across all operations
- **Fast Timestamp Generation**: Micro-caching timestamp function with 1ms cache duration
- **Formatter Caching**: Instance caching to reduce initialization overhead
- **Optimized Context Access**: Reduced context variable lookups for better performance
- **Lazy Evaluation**: Expensive operations only executed when needed
- **Performance Benchmarks**: Comprehensive performance analysis tools
- **Performance Tests**: Automated regression prevention and benchmark validation
- **Memory Optimization**: Efficient memory usage patterns for high-throughput scenarios

### Changed
- **Timestamp Generation**: Now uses optimized caching for better performance
- **Context Variable Access**: Batch retrieval to minimize overhead
- **Formatter Initialization**: Cached instances for repeated configurations
- **Documentation**: Added performance section with benchmarks and optimization tips

### Performance Improvements
- **Basic Logging**: 131,920+ logs/second (significant improvement)
- **Context Logging**: 55,801+ logs/second (optimized from baseline)
- **Timestamp Overhead**: Reduced to < 0.001ms per log
- **Memory Efficiency**: < 10MB for 1,000 structured logs
- **Context Overhead**: Optimized to ~0.010ms per context access

### Developer Experience
- **Performance Tests**: 11 new performance validation tests
- **Benchmark Tools**: `simple_benchmark.py` and `benchmarks.py` for performance analysis
- **Regression Prevention**: Automated performance threshold testing
- **Profiling Support**: Built-in performance measurement utilities

### Technical Enhancements
- **src/structured_logging/performance.py**: New performance utilities module
- **Concurrent Safety**: Performance optimizations maintain thread safety
- **Cache Management**: Intelligent caching strategies for optimal performance
- **Memory Profiling**: Built-in memory usage tracking and optimization

## [0.2.0] - 2024-12-19

### Added
- **Multiple Output Formats**: CSV and Plain Text formatters alongside existing JSON
- **CSVFormatter**: Machine-readable CSV format for data analysis and processing
- **PlainTextFormatter**: Human-readable plain text format for development and debugging
- **Formatter Selection**: New `formatter_type` parameter in LoggerConfig ("json", "csv", "plain")
- **Environment Configuration**: `STRUCTURED_LOG_FORMATTER` environment variable support
- **Type Safety**: FormatterType literal type for formatter selection
- **Comprehensive Tests**: 12 additional tests for new formatters (99% coverage maintained)

### Changed
- **Enhanced LoggerConfig**: Added `formatter_type` field with validation
- **Extended API**: New formatter classes exported in public API
- **Improved Documentation**: Added examples for all formatter types

### Features
- **CSV Format**: Structured tabular output perfect for log analysis tools
- **Plain Text Format**: Readable format with context in parentheses
- **Backward Compatible**: All existing JSON functionality unchanged
- **Environment Driven**: Configure formatter via environment variables
- **Type Hints**: Full type safety for formatter selection

### Developer Experience
- **99% Test Coverage**: Comprehensive test suite for all formatters
- **Documentation**: Updated README with format examples
- **API Consistency**: All formatters share same configuration options

## [0.1.0] - 2024-12-19

### Added
- Initial release of structured logging library
- JSON formatter for structured logging
- Context management with contextvars
- Request ID tracking
- User context support (user_id, tenant_id)
- Custom context fields
- Environment-based configuration
- Type hints throughout codebase
- Comprehensive test suite (98% coverage)
- Modern packaging with pyproject.toml

### Features
- **StructuredFormatter**: JSON output with configurable fields
- **LoggerConfig**: Flexible configuration with environment variables
- **Context Management**: Thread-safe context variables
- **Request Context**: Context manager for request-scoped logging
- **Automatic Context Injection**: Seamless integration of context data

### Developer Experience
- Python 3.13+ support (modern features)
- Black code formatting
- Ruff linting and import sorting  
- MyPy type checking
- Pytest with coverage reporting
- Development dependencies with version ranges

### Documentation
- Comprehensive README with examples
- API reference documentation
- Programming guidelines compliance
- Version strategy documentation

## Version Support

- **Python**: 3.13+ (leveraging newest features)
- **Dependencies**: Only Python standard library
- **Development Tools**: Version ranges for stability

## Migration Guide

### From 0.x to 1.0 (Future)
- Will maintain backward compatibility
- Deprecated features will have migration path
- Breaking changes will be documented

### From Legacy Logging
```python
# Old way
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.info("User action", extra={"user_id": "123"})

# New way
from structured_logging import get_logger, request_context
logger = get_logger(__name__)
with request_context(user_id="123"):
    logger.info("User action")
```