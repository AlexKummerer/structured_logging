# Structured Logging

A flexible Python library for structured JSON logging with context management and request tracing.

## Features

- **JSON Structured Logging**: All logs are formatted as JSON for easy parsing and analysis
- **Context Management**: Automatic injection of request IDs, user context, and custom fields
- **Flexible Configuration**: Environment-based configuration with sensible defaults
- **Multi-tenant Support**: Built-in support for user_id and tenant_id fields
- **Custom Context Fields**: Add any custom fields to your log entries
- **Thread-safe**: Uses Python's contextvars for proper context isolation

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

### Multiple Output Formats

```python
from structured_logging import get_logger, LoggerConfig

# CSV Format
config = LoggerConfig(formatter_type="csv")
logger = get_logger("my_app", config)
logger.info("CSV formatted log")
```

Output (CSV format):
```csv
timestamp,level,logger,message,request_id,user_id,tenant_id
2024-01-15T10:30:00.000Z,INFO,my_app,CSV formatted log,f47ac10b-58cc-4372-a567-0e02b2c3d479,user123,tenant456
```

```python
# Plain Text Format  
config = LoggerConfig(formatter_type="plain")
logger = get_logger("my_app", config)
logger.info("Plain text log")
```

Output (Plain text format):
```
[2024-01-15T10:30:00.000Z] INFO my_app Plain text log (request_id=f47ac10b-58cc-4372-a567-0e02b2c3d479, user_id=user123, tenant_id=tenant456)
```

### With Custom Context

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

Output:
```json
{"timestamp": "2024-01-15T10:30:00.000Z", "level": "INFO", "logger": "my_app", "message": "Payment processed", "request_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479", "service": "payment-api", "version": "1.2.0", "payment_id": "pay_123", "amount": 99.99, "currency": "EUR"}
```

## Configuration

### Environment Variables

```bash
STRUCTURED_LOG_LEVEL=DEBUG
STRUCTURED_LOG_FORMATTER=json          # json, csv, or plain
STRUCTURED_LOG_TIMESTAMP=true
STRUCTURED_LOG_REQUEST_ID=true
STRUCTURED_LOG_USER_CONTEXT=true
```

### Programmatic Configuration

```python
from structured_logging import LoggerConfig, set_default_config

config = LoggerConfig(
    log_level="DEBUG",
    formatter_type="csv",              # json, csv, or plain
    include_timestamp=True,
    include_request_id=True,
    include_user_context=True
)
set_default_config(config)
```

## Advanced Usage

### Manual Context Management

```python
from structured_logging import (
    set_request_id, 
    set_user_context, 
    update_custom_context
)

# Set request ID manually
set_request_id("custom-request-123")

# Set user context
set_user_context({"user_id": "user456", "tenant_id": "tenant789"})

# Add custom fields
update_custom_context(service="auth", environment="production")
```

### Custom Logger Configuration

```python
from structured_logging import get_logger, LoggerConfig

# Critical logger with plain text format
config = LoggerConfig(
    log_level="WARNING", 
    formatter_type="plain", 
    include_timestamp=False
)
logger = get_logger("critical_logger", config)

# CSV logger for data analysis
csv_config = LoggerConfig(formatter_type="csv")
data_logger = get_logger("data_logger", csv_config)
```

## API Reference

### Core Functions

- `get_logger(name, config=None)`: Create a structured logger
- `log_with_context(logger, level, message, **extra)`: Log with automatic context injection
- `request_context(user_id=None, tenant_id=None, **custom_fields)`: Context manager for request scoping

### Context Management

- `get_request_id()` / `set_request_id(req_id)`: Request ID management
- `get_user_context()` / `set_user_context(context)`: User context management  
- `get_custom_context()` / `set_custom_context(context)`: Custom context management
- `update_custom_context(**kwargs)`: Add fields to custom context

### Configuration

- `LoggerConfig`: Configuration dataclass
- `get_default_config()` / `set_default_config(config)`: Default configuration management

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

# Format code
black src/ tests/
ruff check --fix src/ tests/

# Type checking
mypy src/

# Version management
bump2version patch  # 0.1.0 -> 0.1.1
bump2version minor  # 0.1.1 -> 0.2.0  
bump2version major  # 0.2.0 -> 1.0.0
```

## Performance

This library is optimized for high-throughput logging scenarios:

### Performance Benchmarks (v0.3.0)

- **Basic logging**: 131,920+ logs/second  
- **Context logging**: 55,801+ logs/second
- **Memory efficient**: < 10MB for 1,000 structured logs
- **Timestamp overhead**: < 0.001ms per log

### Performance Tips

1. **Use appropriate formatter**: JSON formatter is fastest for high throughput
2. **Disable unnecessary features**: Turn off timestamps if not needed
3. **Batch context updates**: Use `request_context()` for request-scoped logging
4. **Cache loggers**: Reuse logger instances instead of creating new ones

```python
# Optimal configuration for high throughput
config = LoggerConfig(
    formatter_type="json",  # Fastest formatter
    include_timestamp=False,  # Disable if not needed
    include_request_id=True,
    include_user_context=True
)
logger = get_logger("high_perf", config)  # Reuse this logger
```

### Built-in Optimizations

- **Formatter caching**: Reduces initialization overhead
- **Fast timestamp generation**: Micro-caching for sub-millisecond accuracy
- **Optimized context access**: Minimal context variable lookups
- **Lazy evaluation**: Expensive operations only when needed

## Version Strategy

This library follows [Semantic Versioning](https://semver.org/) and supports the latest Python versions:

- **Current**: Python 3.13+
- **Future**: Python 3.14+ support planned
- **Philosophy**: Stay current with Python innovations

See [VERSION_STRATEGY.md](VERSION_STRATEGY.md) for detailed version support policy.

## License

MIT License - see LICENSE file for details.