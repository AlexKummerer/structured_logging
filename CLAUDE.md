# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive Python structured logging library (currently v0.6.0) that provides JSON-formatted logging with advanced features including:
- Scientific data integration (NumPy, Pandas, SciPy) with intelligent compression
- Async logging with high-performance queue-based processing
- Network handlers (Syslog RFC 3164/5424, HTTP API, raw sockets)
- Schema validation and automatic type detection
- Context management for multi-tenant applications using contextvars
- Zero core dependencies (stdlib only), with optional scientific packages

## Architecture

### Core Components

The library follows a modular architecture with clear separation of concerns:

- **Core Logger (`logger.py`)**: Main logger factory with formatter caching and filter integration
- **Formatters (`formatter.py`)**: JSON, CSV, and PlainText formatters with structured output
- **Context Management (`context.py`, `async_context.py`)**: Thread-safe and async-safe context isolation using `contextvars`
- **Serializers (`serializers.py`)**: Advanced JSON serialization with scientific data support, lazy loading, and type detection
- **Network Handlers (`network_handlers.py`)**: Syslog (RFC 3164/5424), HTTP API, and raw socket logging
- **Async Logger (`async_logger.py`)**: Non-blocking async logging with queue-based processing
- **Filtering (`filtering.py`)**: Advanced log filtering with sampling, context-based, and custom filters
- **Integrations (`integrations.py`)**: FastAPI and Flask middleware for automatic request logging

### Key Architecture Patterns

- **Lazy Serialization**: Large objects are only serialized when logs are actually written
- **Formatter Caching**: Reuses formatter instances for performance optimization
- **Context Variable Prefix System**: Context fields prefixed with `ctx_` are automatically injected and cleaned
- **Scientific Data Handling**: Intelligent compression and sampling for NumPy arrays and Pandas DataFrames
- **Dual API**: Complete sync and async APIs with shared configuration

### Key Functions

- `get_logger(name, config=None)`: Main factory function for structured loggers
- `get_async_logger(name, async_config=None, logger_config=None)`: Factory for async loggers
- `log_with_context()` / `alog_with_context()`: Enhanced logging with automatic context injection
- `request_context()` / `async_request_context()`: Context managers for request-scoped logging
- `serialize_for_logging(obj, config=None)`: Serialize complex objects including scientific data

## Development Commands

```bash
# Installation and setup
pip install -e ".[dev]"                     # Install with development dependencies
pip install -e ".[dev,scientific]"          # Include NumPy, Pandas, SciPy for scientific features

# Testing (coverage must maintain >80%, currently at 98%+)
pytest --cov=structured_logging --cov-fail-under=80     # Full test suite with coverage
pytest tests/test_logger.py                              # Run specific test file
pytest tests/test_logger.py::test_basic_logging          # Run specific test
pytest -m "not performance"                              # Skip performance benchmarks (default)
pytest -m "performance"                                  # Run only performance benchmarks
pytest -m "slow"                                         # Run slow integration tests

# Code quality checks (must pass before committing)
black src/ tests/                    # Format code (line length: 88)
ruff check --fix src/ tests/         # Lint and fix issues (Python 3.13 target)
mypy src/                           # Type checking (strict mode)

# Version management
bump2version patch                  # Bug fixes: 0.6.0 -> 0.6.1
bump2version minor                  # New features: 0.6.0 -> 0.7.0
bump2version major                  # Breaking changes: 0.6.0 -> 1.0.0
```

## Testing Architecture

- **Performance Tests**: Located in `tests/performance/` with minimum throughput requirements
- **Unit Tests**: Comprehensive coverage of all modules with mocking for external dependencies
- **Integration Tests**: Real-world scenarios including FastAPI integration and network handlers
- **Scientific Data Tests**: NumPy/Pandas integration with various data types and edge cases
- **Async Tests**: Complete async API coverage with queue processing and context isolation

### Performance Benchmarks

The test suite includes automated performance validation:
- Basic logging: >50,000 logs/sec
- Structured logging: >3,000 logs/sec  
- Async logging: >10,000 logs/sec
- Memory usage: <1.0 KB per log entry

## Development Guidelines

- **File Length**: Maximum 250 lines per file (enforced by PROGRAMMING_GUIDELINES.md)
- **Function Length**: Maximum 20 lines per function
- **Test Coverage**: Must maintain >80% coverage, currently at 98%+
- **Type Annotations**: All functions must have complete type annotations
- **Import Rules**: ALL imports must be at the top of files (critical rule from PROGRAMMING_GUIDELINES.md)
- **Context Prefix System**: Context variables prefixed with `ctx_` are automatically processed
- **Scientific Data**: Use `SerializationConfig` for NumPy/Pandas serialization settings
- **Async Safety**: Always use async context managers and proper cleanup in async code
- **Python Version**: Requires Python 3.13+ for modern features (contextvars, type hints)

## Key Implementation Details

- **Context Management**: Uses Python's `contextvars` for proper async/thread isolation
- **Null Value Filtering**: All context values that are None are automatically filtered out
- **Handler Configuration**: StreamHandler with stdout by default, propagation enabled for testing
- **Formatter Caching**: Formatters are cached by configuration signature for performance
- **Scientific Data Serialization**: Automatic compression and sampling for large arrays/DataFrames
- **Network Handler Fallbacks**: HTTP handlers can fallback to local logging on network failures
- **Optional Dependencies**: Scientific packages (numpy/pandas/scipy) are detected at runtime with graceful fallback
- **Zero Core Dependencies**: Uses only Python stdlib for core functionality, ensuring lightweight deployment