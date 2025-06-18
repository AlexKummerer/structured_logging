# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Performance benchmarks
- Additional formatter types (CSV, Plain Text)
- Async logger support

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