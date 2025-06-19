# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Async logger support (planned)

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

## [0.3.0] - 2024-12-19

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