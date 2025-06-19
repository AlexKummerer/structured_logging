# Structured Logging

A flexible Python library for structured JSON logging with context management, async support, and production-ready performance.

## Features

- **JSON Structured Logging**: All logs are formatted as JSON for easy parsing and analysis
- **Advanced Log Filtering**: Smart filtering and sampling for production performance
- **File Handlers**: Rotating file logging with gzip compression and archiving
- **FastAPI Integration**: One-line middleware for automatic request/response logging
- **Async Logging Support**: High-performance non-blocking async logging with queue-based processing
- **Context Management**: Automatic injection of request IDs, user context, and custom fields
- **Multiple Output Formats**: JSON, CSV, and Plain Text formatters
- **Flexible Configuration**: Environment-based configuration with sensible defaults
- **Multi-tenant Support**: Built-in support for user_id and tenant_id fields
- **Custom Context Fields**: Add any custom fields to your log entries
- **Thread-safe & Async-safe**: Uses Python's contextvars for proper context isolation
- **High Performance**: Optimized for high-throughput logging scenarios

## Installation

```bash
pip install structured-logging
```

## Quick Start

### Basic Usage

```python
from structured_logging import get_logger

logger = get_logger("my_app")
logger.info("Application started")
```

Output:
```json
{"timestamp": "2024-01-15T10:30:00.000Z", "level": "INFO", "logger": "my_app", "message": "Application started"}
```

### With Request Context

```python
from structured_logging import get_logger, request_context

logger = get_logger("my_app")

with request_context(user_id="user123", tenant_id="tenant456"):
    logger.info("User logged in successfully")
```

Output (JSON format):
```json
{"timestamp": "2024-01-15T10:30:00.000Z", "level": "INFO", "logger": "my_app", "message": "User logged in successfully", "request_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479", "user_id": "user123", "tenant_id": "tenant456"}
```

### Log Filtering & Sampling

```python
from structured_logging import get_logger, LoggerConfig, FilterConfig, LevelFilter, SamplingFilter

# Configure filtering
filter_config = FilterConfig(
    enabled=True,
    filters=[
        LevelFilter(min_level="INFO"),           # Only INFO and above
        SamplingFilter(sample_rate=0.1, strategy="random")  # Sample 10% of logs
    ]
)

config = LoggerConfig(
    formatter_type="json",
    filter_config=filter_config
)

logger = get_logger("filtered_app", config)

# Only 10% of these will be logged
for i in range(1000):
    logger.info(f"Message {i}")
```

### File Logging with Rotation

```python
from structured_logging import get_logger, LoggerConfig, FileHandlerConfig

# Configure rotating file handler
file_config = FileHandlerConfig(
    filename="app.log",
    max_bytes=10 * 1024 * 1024,    # 10MB
    backup_count=5,                 # Keep 5 backup files
    compress_rotated=True,          # Gzip compress rotated files
    async_compression=True          # Compress asynchronously
)

config = LoggerConfig(
    output_type="file",
    file_config=file_config,
    formatter_type="json"
)

logger = get_logger("file_app", config)
logger.info("This goes to a rotating log file")
```

### FastAPI Integration

```python
from fastapi import FastAPI
from structured_logging.integrations import add_structured_logging, FastAPILoggingConfig

app = FastAPI()

# Add structured logging middleware
logging_config = FastAPILoggingConfig(
    log_request_body=True,
    log_response_body=False,
    mask_sensitive_data=True,
    exclude_paths={"/health", "/metrics"}
)

app = add_structured_logging(app, logging_config)

@app.get("/users/{user_id}")
def get_user(user_id: str):
    return {"user_id": user_id, "name": "John Doe"}

# All requests are automatically logged with context
```

### Async Logging

```python
import asyncio
from structured_logging import get_async_logger, async_request_context

async def main():
    # Create async logger
    async_logger = get_async_logger("async_app")
    
    async with async_request_context(user_id="user123"):
        await async_logger.ainfo("Async operation started")
        await asyncio.sleep(0.1)  # Simulate async work
        await async_logger.ainfo("Async operation completed")
    
    # Ensure all logs are flushed
    await async_logger.flush()

asyncio.run(main())
```

## Configuration

### Environment Variables

```bash
STRUCTURED_LOG_LEVEL=DEBUG
STRUCTURED_LOG_FORMATTER=json          # json, csv, or plain
STRUCTURED_LOG_OUTPUT=console           # console, file, both
STRUCTURED_LOG_TIMESTAMP=true
STRUCTURED_LOG_REQUEST_ID=true
STRUCTURED_LOG_USER_CONTEXT=true
```

### Programmatic Configuration

```python
from structured_logging import LoggerConfig, set_default_config

config = LoggerConfig(
    log_level="DEBUG",
    formatter_type="json",              # json, csv, or plain
    output_type="both",                 # console, file, both
    include_timestamp=True,
    include_request_id=True,
    include_user_context=True
)
set_default_config(config)
```

## Advanced Features

### Smart Log Filtering

```python
from structured_logging import FilterConfig, LevelFilter, ContextFilter, CustomFilter, SamplingFilter

# Multiple filter types
filter_config = FilterConfig(
    enabled=True,
    filters=[
        LevelFilter(min_level="INFO"),
        ContextFilter(required_fields=["user_id"]),      # Only log if user_id present
        SamplingFilter(sample_rate=0.1, strategy="hash"), # Consistent sampling
        CustomFilter(lambda record, context: "error" in record.getMessage().lower())
    ]
)
```

### Rate Limiting

```python
from structured_logging import FilterConfig, SamplingFilter

# Rate limiting with burst allowance
filter_config = FilterConfig(
    enabled=True,
    filters=[
        SamplingFilter(
            sample_rate=0.01,           # 1% normal rate
            strategy="rate_limit",
            burst_allowance=10,         # Allow 10 ERROR logs immediately
            burst_levels=["ERROR", "CRITICAL"]
        )
    ]
)
```

### Custom Context

```python
from structured_logging import get_logger, log_with_context, request_context

logger = get_logger("my_app")

with request_context(service="payment-api", version="1.2.0"):
    log_with_context(
        logger, 
        "info", 
        "Payment processed", 
        payment_id="pay_123",
        amount=99.99,
        currency="EUR"
    )
```

## Performance

### Benchmarks (Version 0.5.0)

- **Basic Logging**: 130,000+ logs/second  
- **Structured Logging**: 54,000+ logs/second
- **Filtered Logging**: 29,000+ logs/second
- **Async Logging**: 10,000+ logs/second
- **Memory Efficient**: <5KB per log entry

### Performance Testing

```bash
# Run performance benchmarks
python scripts/benchmark_comparison.py

# Run comprehensive performance tests
python scripts/run_performance_tests.py

# Performance tests are excluded from regular test runs
pytest                          # Regular tests only
pytest -m performance          # Performance tests only
```

### Built-in Optimizations

- **Formatter Caching**: Reduces initialization overhead
- **Fast Timestamp Generation**: Micro-caching for sub-millisecond accuracy
- **Optimized Context Access**: Minimal context variable lookups
- **Smart Filtering**: Efficient early filtering to reduce processing overhead
- **Async Queue Processing**: Non-blocking logging with batched processing

## Framework Integrations

### FastAPI

```python
from structured_logging.integrations import add_structured_logging, FastAPILoggingConfig

app = add_structured_logging(app, FastAPILoggingConfig(
    log_request_headers=True,
    log_response_body=False,
    minimum_duration_ms=100,        # Only log slow requests
    exclude_paths={"/health"}
))
```

### Flask

```python
from flask import Flask
from structured_logging.integrations import add_flask_logging

app = Flask(__name__)
add_flask_logging(app)  # Automatic request/response logging
```

## API Reference

### Core Functions

- `get_logger(name, config=None)`: Create a structured logger
- `log_with_context(logger, level, message, **extra)`: Log with automatic context injection
- `request_context(user_id=None, tenant_id=None, **custom_fields)`: Context manager for request scoping

### Async Functions

- `get_async_logger(name, logger_config=None, async_config=None)`: Create async logger
- `async_request_context(**context)`: Async context manager

### Context Management

- `get_request_id()` / `set_request_id(req_id)`: Request ID management
- `get_user_context()` / `set_user_context(context)`: User context management  
- `get_custom_context()` / `set_custom_context(context)`: Custom context management
- `update_custom_context(**kwargs)`: Add fields to custom context

### Configuration

- `LoggerConfig`: Main configuration dataclass
- `FilterConfig`: Filtering configuration
- `FileHandlerConfig`: File handler configuration
- `AsyncLoggerConfig`: Async logging configuration

## Requirements

- **Python 3.13+** (leverages modern Python features)
- **No external dependencies** (uses only Python standard library)

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=structured_logging --cov-fail-under=80

# Run performance tests
pytest -m performance

# Format code
black src/ tests/
ruff check --fix src/ tests/

# Type checking
mypy src/

# Version management
bump2version patch  # 0.5.0 -> 0.5.1
bump2version minor  # 0.5.0 -> 0.6.0  
bump2version major  # 0.5.0 -> 1.0.0
```

## Version History

### 🎉 **Version 0.5.0** (Current) - Production-Ready Performance
- ✅ **Advanced Log Filtering**: LevelFilter, ContextFilter, SamplingFilter, CustomFilter
- ✅ **Smart Sampling**: Random, hash-based, and rate-limiting strategies
- ✅ **File Handlers**: Rotating file logging with gzip compression
- ✅ **FastAPI Integration**: Complete middleware with sensitive data masking
- ✅ **Performance Framework**: Comprehensive benchmarking and regression testing
- ✅ **130,000+ logs/sec**: Exceptional performance validated by automated benchmarks

### 🚀 **Version 0.4.0** - Async Excellence
- ✅ Complete async logging support
- ✅ High-performance queue-based processing
- ✅ 40,153+ logs/second concurrent throughput

### 📈 **Future Versions**
- **0.6.0**: Network handlers for remote logging (Syslog, HTTP, TCP/UDP)
- **0.7.0**: Enhanced data types and cloud platform integrations
- **0.8.0**: Monitoring and observability features
- **1.0.0**: Production-ready stable API

## Performance Tips

1. **Use appropriate formatter**: JSON formatter is fastest for high throughput
2. **Configure smart filtering**: Use sampling for high-volume scenarios
3. **Enable file compression**: Use async compression for better performance
4. **Batch context updates**: Use `request_context()` for request-scoped logging
5. **Cache loggers**: Reuse logger instances instead of creating new ones

```python
# Optimal configuration for high throughput
config = LoggerConfig(
    formatter_type="json",
    output_type="file",
    filter_config=FilterConfig(
        enabled=True,
        filters=[SamplingFilter(sample_rate=0.1, strategy="hash")]
    )
)
logger = get_logger("high_perf", config)  # Reuse this logger
```

## License

MIT License - see LICENSE file for details.