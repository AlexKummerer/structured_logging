# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python structured logging library that provides JSON-formatted logging with contextual information. The library uses context variables for request tracing and supports multi-tenant applications.

## Architecture

### Core Components

- **StructuredFormatter**: Custom JSON formatter that converts log records to structured JSON format with timestamps, levels, and contextual data
- **Context Management**: Uses Python's `contextvars` module to maintain request-scoped context including:
  - `request_id`: Unique identifier for each request
  - `user_context`: Dictionary containing user_id and tenant_id for multi-tenant support
- **Context Injection**: Automatic injection of contextual information into log entries via the `ctx_` prefix system

### Key Functions

- `get_logger(name)`: Factory function that creates configured structured loggers
- `log_with_context()`: Enhanced logging function that automatically injects request and user context
- `request_context()`: Context manager for setting up request-scoped logging context

### Configuration System

The logging system uses a flexible `LoggerConfig` class that supports:
- Environment variables for configuration (e.g., `STRUCTURED_LOG_LEVEL`)
- Programmatic configuration via `LoggerConfig` instances  
- Sensible defaults for all configuration options
- No external dependencies required

## Development Commands

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run tests with coverage (must maintain >80%)
pytest --cov=structured_logging --cov-fail-under=80

# Code quality checks (must pass)
black src/ tests/                    # Format code
ruff check --fix src/ tests/         # Lint and fix issues  
mypy src/                           # Type checking

# Version management
bump2version patch                  # Bug fixes: 0.1.0 -> 0.1.1
bump2version minor                  # New features: 0.1.1 -> 0.2.0
bump2version major                  # Breaking changes: 0.2.0 -> 1.0.0
```

## Version Requirements

- **Python**: 3.13+ (required for modern features)
- **Dependencies**: None (pure Python standard library)
- **Development**: Version ranges for stability (see pyproject.toml)

## Development Notes

- Context variables are prefixed with `ctx_` in log entries and automatically stripped during formatting
- All context values that are None are filtered out from log entries
- The logger uses StreamHandler with stdout by default and enables propagation for testing
- Code follows strict programming guidelines (see PROGRAMMING_GUIDELINES.md)
- Maintains 98%+ test coverage