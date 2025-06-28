# Programming Guidelines

This document outlines the coding standards and best practices for the Structured Logging Python library project.

## Core Principles

### 1. KISS (Keep It Simple, Stupid)
- Write simple, readable code that solves the problem directly
- Avoid over-engineering and unnecessary complexity
- Prefer explicit over implicit implementations
- Use clear, descriptive names for variables, functions, and classes

### 2. Single Responsibility Principle (SRP)
- Each class should have only one reason to change
- Each function should do one thing well
- Separate business logic, data access, and presentation concerns
- Example: `CharacterService` handles business logic, `CharacterDB` handles data mapping

### 3. Clean Code Standards

#### File Length Limits
- **Maximum file length: 250 lines**
- If a file exceeds 250 lines, split it into multiple files
- Group related functionality into separate modules
- Use meaningful file names that reflect their purpose

#### Function Length Limits
- **Maximum function length: ca. 20 lines**
- Functions should be focused and do one thing
- Extract complex logic into helper functions
- Use early returns to reduce nesting

```python
# Good - under 20 lines, single responsibility
def validate_log_data(data: dict, schema: str) -> None:
    if not data.get("level"):
        raise ValidationError("Log level is required")
    
    if not data.get("message"):
        raise ValidationError("Log message is required")
    
    if data.get("timestamp") and not isinstance(data["timestamp"], datetime):
        raise ValidationError("Timestamp must be datetime object")

# Bad - too long, multiple responsibilities
def create_logger_with_validation_and_serialization(config):
    # 30+ lines of mixed validation, creation, and serialization logic
```

## Code Structure

### 1. Modularität (Modular Design)
- Organize code into logical modules and packages
- Keep related functionality together
- Separate concerns into different layers:
  - **Core** (`logger.py`, `context.py`): Main logging functionality
  - **Configuration** (`config.py`): Configuration management
  - **Formatters** (`formatter.py`): Output formatting (JSON, CSV, plain text)
  - **Serializers** (`serializers.py`): Data serialization and validation
  - **Handlers** (`handlers.py`, `network_handlers.py`): Output destinations
  - **Integrations** (`integrations.py`): Framework integrations (FastAPI, Flask)
  - **Filtering** (`filtering.py`): Log filtering and sampling

### 2. Import Organization - CRITICAL RULE
**ALL IMPORTS MUST BE GLOBAL AND AT THE TOP OF THE FILE**

```python
# Standard library imports (alphabetically ordered)
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Third-party imports (alphabetically ordered)
import numpy as np
import pandas as pd
import pytest

# Local imports (alphabetically ordered by module)
from .config import LoggerConfig, SerializationConfig
from .context import get_request_id, set_user_context
from .formatter import StructuredFormatter
from .serializers import serialize_for_logging, ValidationError
```

**NEVER use local imports inside functions:**
```python
# BAD - local imports inside functions
def serialize_numpy_array(array):
    import numpy as np  # NEVER DO THIS
    return array.tolist()

# GOOD - global imports at top
import numpy as np

def serialize_numpy_array(array):
    return array.tolist()
```

**Exception handling for optional imports:**
```python
# For optional dependencies, use try/catch at module level only
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

# Then use the flag in functions
def serialize_array(array):
    if not HAS_NUMPY:
        return str(array)
    return array.tolist()
```

### 3. Error Handling
- Use specific exception types
- Handle errors at the appropriate level
- Provide meaningful error messages
- Don't suppress exceptions without good reason

```python
# Good
try:
    serialized_data = serialize_for_logging(complex_object)
    if not serialized_data:
        raise SerializationError("Failed to serialize object")
    return serialized_data
except ValidationError as e:
    logger.error("Validation failed", error=str(e), object_type=type(complex_object).__name__)
    raise

# Bad
try:
    # some operation
    pass
except:
    pass  # Silent failure
```

## Testing Guidelines

### 1. Test Structure
- Write unit tests for all business logic
- Use integration tests for API endpoints
- Test both happy path and error scenarios
- Aim for high test coverage (>80%)

### 2. Test Organization
```python
# tests/test_serializers.py
import pytest
from structured_logging.serializers import serialize_for_logging, ValidationError

class TestSerializers:
    def test_serialize_datetime_success(self):
        # Arrange
        from datetime import datetime
        test_datetime = datetime(2024, 1, 15, 10, 30, 0)
        
        # Act
        result = serialize_for_logging(test_datetime)
        
        # Assert
        assert isinstance(result, str)
        assert "2024-01-15" in result

    def test_serialize_invalid_object_error(self):
        # Test error scenario
        class UnserializableClass:
            def __repr__(self):
                raise Exception("Cannot serialize")
        
        with pytest.raises(Exception):
            serialize_for_logging(UnserializableClass())
```

### 3. Mocking and Fixtures
- Use pytest fixtures for test data setup
- Mock external dependencies
- Keep tests isolated and independent

## Structured Logging Library Specific Guidelines

### 1. Library Usage Patterns
```python
from structured_logging import get_logger, log_with_context, request_context

# Configure logger
logger = get_logger("my_app")

# Use context management
with request_context(request_id="req_123", user_id="user_456"):
    log_with_context(logger, "info", "Processing request", 
                    operation="create_user", 
                    data={"email": "user@example.com"})
```

### 2. Logging Best Practices
- Use appropriate log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Include relevant context in log messages
- Log important business events
- Don't log sensitive information (passwords, tokens)
- Use lazy serialization for complex objects

```python
# Good - structured logging with context and lazy serialization
from structured_logging import create_lazy_serializable

complex_data = create_lazy_serializable({
    "user_profile": large_user_object,
    "transaction_data": expensive_calculation()
})

log_with_context(logger, "info", "Transaction processed", 
                transaction_id=transaction.id, 
                user_id=transaction.user_id,
                details=complex_data)  # Only serialized if log is actually written

# Bad - unstructured logging
print(f"Transaction {transaction.id} processed")
```

### 3. Log Levels Usage
- **DEBUG**: Detailed diagnostic information
- **INFO**: General information about program execution
- **WARNING**: Something unexpected happened, but the program continues
- **ERROR**: A serious problem occurred
- **CRITICAL**: The program may not continue

## Code Quality Checks

### 1. Type Hints
- Use type hints for all function parameters and return values
- Use `Optional` for nullable values
- Use `List`, `Dict` for collections

```python
from typing import Any, Dict, List, Optional
from structured_logging.serializers import ValidationError

def serialize_for_logging(obj: Any, config: Optional[SerializationConfig] = None) -> Any:
    pass

def validate_log_data(data: Dict[str, Any], schema_name: str) -> bool:
    pass
```

### 2. Documentation
- Write docstrings for all public functions and classes
- Use Google or NumPy docstring format
- Include parameter types and return types
- Document exceptions that may be raised

```python
def serialize_for_logging(obj: Any, config: Optional[SerializationConfig] = None) -> Any:
    """
    Serialize an object for structured logging with enhanced type support.

    Args:
        obj: The object to serialize (can be any Python type)
        config: Optional serialization configuration for customizing behavior

    Returns:
        JSON-serializable representation of the object

    Raises:
        SerializationError: If the object cannot be serialized
        ValidationError: If schema validation is enabled and fails
    """
```

### 3. Code Formatting Tools

#### Black (Code Formatting)
- **Purpose**: Automatic code formatting
- **Configuration**: Maximum line length 88 characters
- **Usage**: 
  ```bash
  # Format all Python files
  black .
  
  # Check formatting without changes
  black --check .
  
  # Format specific file
  black app.py
  ```
- **Rules**:
  - Consistent string quote usage
  - Automatic line breaking and indentation
  - Consistent spacing around operators

#### Ruff (Linting and Import Sorting)
- **Purpose**: Fast Python linter and code formatter (replaces flake8, isort, and more)
- **Configuration**: Create `pyproject.toml` or `ruff.toml`
- **Usage**:
  ```bash
  # Lint all files
  ruff check .
  
  # Auto-fix issues where possible
  ruff check --fix .
  
  # Format imports and code
  ruff format .
  
  # Check specific file
  ruff check app.py
  ```
- **Key Features**:
  - Import sorting (replaces isort)
  - Unused import removal
  - Code style enforcement
  - Security issue detection

#### MyPy (Type Checking)
- **Purpose**: Static type checking for Python
- **Usage**:
  ```bash
  # Type check all files
  mypy .
  
  # Type check specific file
  mypy app.py
  
  # Type check with strict mode
  mypy --strict .
  ```
- **Configuration**: Create `mypy.ini` or use `pyproject.toml`
  ```ini
  [mypy]
  python_version = 3.11
  warn_return_any = True
  warn_unused_configs = True
  disallow_untyped_defs = True
  ```

#### Recommended Configuration Files

**pyproject.toml**:
```toml
[tool.black]
line-length = 88
target-version = ['py311']

[tool.ruff]
line-length = 88
target-version = "py311"
select = [
    "E",  # pycodestyle errors
    "W",  # pycodestyle warnings
    "F",  # pyflakes
    "I",  # isort
    "B",  # flake8-bugbear
    "C4", # flake8-comprehensions
    "UP", # pyupgrade
]
ignore = []

[tool.ruff.per-file-ignores]
"__init__.py" = ["F401"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

## Naming Conventions

### 1. Variables and Functions
- Use `snake_case` for variables and functions
- Use descriptive names that explain purpose
- Avoid abbreviations unless they're well-known

```python
# Good
serialization_config = SerializationConfig()
lazy_serialization_manager = LazySerializationManager()

# Bad
ser_cfg = SerializationConfig()
lazy_mgr = LazySerializationManager()
```

### 2. Classes
- Use `PascalCase` for class names
- Use descriptive names that indicate the class purpose

```python
# Good
class StructuredFormatter:
    pass

class LazySerializationManager:
    pass

# Bad
class struct_fmt:
    pass
```

### 3. Constants
- Use `UPPER_SNAKE_CASE` for constants
- Define constants at module level

```python
# Good
DEFAULT_LOG_LEVEL = "INFO"
MAX_SERIALIZATION_DEPTH = 10
LAZY_THRESHOLD_BYTES = 1000

# Bad
default_log_level = "INFO"
max_depth = 10
```

## Performance Guidelines

### 1. Serialization Performance
- Use lazy serialization for complex objects
- Configure appropriate thresholds for lazy activation
- Cache serialization results when possible
- Avoid deep recursion in object traversal

### 2. Memory Usage
- Don't serialize all log data immediately
- Use lazy evaluation for expensive operations
- Clean up resources in finally blocks or use context managers
- Monitor lazy serialization statistics for optimization

```python
# Good - lazy serialization for performance
from structured_logging import create_lazy_serializable

def process_large_dataset(data):
    # Only serialize if log actually gets written
    lazy_data = create_lazy_serializable(data)
    logger.info("Processing dataset", data_summary=lazy_data)

# Bad - immediate serialization
def process_large_dataset(data):
    # Always serializes, even if log is filtered
    serialized_data = serialize_for_logging(data)
    logger.info("Processing dataset", data_summary=serialized_data)
```

## Security Guidelines

### 1. Data Sanitization
- Never log sensitive information (passwords, tokens, API keys)
- Use schema validation to ensure data integrity
- Sanitize user input before logging
- Implement data masking for sensitive fields

### 2. Serialization Security
- Validate objects before serialization
- Prevent serialization of unsafe object types
- Use secure defaults for serialization configuration
- Implement size limits to prevent DoS attacks

### 3. Error Messages
- Don't expose internal system details in log messages
- Sanitize error messages that might contain sensitive data
- Use structured error logging for internal debugging

```python
# Good - sanitized logging
logger.info("User authentication failed", 
           user_id=user.id,
           ip_address=request.remote_addr,
           # No password or sensitive data
           )

# Bad - exposing sensitive data
logger.info("User authentication failed", 
           username=user.username,
           password=user.password,  # NEVER LOG PASSWORDS
           api_key=user.api_key     # NEVER LOG KEYS
           )
```

## Import Guidelines - CRITICAL RULES

### ⚠️ ABSOLUTE REQUIREMENTS ⚠️

1. **ALL imports MUST be at the top of the file**
2. **NO local imports inside functions (except for optional dependency handling)**
3. **Imports MUST be grouped and alphabetically sorted**

### Import Order:
```python
# 1. Standard library imports (alphabetical)
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 2. Third-party imports (alphabetical)
import numpy as np
import pandas as pd
import pytest

# 3. Local/relative imports (alphabetical)
from .config import LoggerConfig
from .context import get_request_id
from .serializers import serialize_for_logging
```

### Optional Dependencies Pattern:
```python
# At module top level only
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

# Use in functions
def serialize_array(array):
    if not HAS_NUMPY:
        return str(array)
    return array.tolist()
```

### ❌ VIOLATIONS - NEVER DO THIS:
```python
def bad_function():
    import json  # NEVER - import inside function
    import time  # NEVER - import inside function
    from datetime import datetime  # NEVER - import inside function
```

### ✅ CORRECT PATTERN:
```python
import json
import time
from datetime import datetime

def good_function():
    # Use already imported modules
    result = json.dumps({"timestamp": datetime.now()})
```

## Summary Checklist

Before committing code, ensure:
- [ ] **ALL imports are at the top of the file (CRITICAL)**
- [ ] **NO local imports inside functions (CRITICAL)**
- [ ] **Imports are grouped and alphabetically sorted (CRITICAL)**
- [ ] Files are under 250 lines
- [ ] Functions are under 20 lines
- [ ] All functions have type hints
- [ ] Code follows SRP principle
- [ ] Error handling is implemented
- [ ] Structured logging guidelines followed
- [ ] Lazy serialization used for complex objects
- [ ] Schema validation implemented where appropriate
- [ ] Tests are written for new functionality
- [ ] Code is formatted with black: `black .`
- [ ] Code passes ruff linting: `ruff check .`
- [ ] Code passes type checking: `mypy .`
- [ ] No sensitive information is exposed in logs

## Development Workflow Commands

Run these commands before committing:

```bash
# Format code
black .

# Lint and auto-fix
ruff check --fix .

# Type check
mypy .

# Run all formatting and checks
black . && ruff check --fix . && mypy .
```