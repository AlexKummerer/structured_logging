# Programming Guidelines

This document outlines the coding standards and best practices for this Python FastAPI project.

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
- **Maximum function length: 20 lines**
- Functions should be focused and do one thing
- Extract complex logic into helper functions
- Use early returns to reduce nesting

```python
# Good - under 20 lines, single responsibility
def validate_character_data(character_data: dict) -> None:
    if not character_data.get("name"):
        raise ValueError("Character name is required")
    
    if not character_data.get("role"):
        raise ValueError("Character role is required")
    
    if character_data.get("age") and character_data["age"] < 0:
        raise ValueError("Age cannot be negative")

# Bad - too long, multiple responsibilities
def create_character_with_validation_and_logging(character_data):
    # 30+ lines of mixed validation, creation, and logging logic
```

## Code Structure

### 1. Modularität (Modular Design)
- Organize code into logical modules and packages
- Use dependency injection for testability
- Keep related functionality together
- Separate concerns into different layers:
  - **Routes** (`app.py`): HTTP request/response handling
  - **Services** (`services/`): Business logic
  - **Models** (`models/`): Data structures and ORM
  - **Auth** (`auth/`): Authentication and authorization
  - **Database** (`db/`): Database configuration

### 2. Import Organization
```python
# Standard library imports
import os
from typing import List, Optional

# Third-party imports
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

# Local imports
from models.character import Character
from services.character_service import CharacterService
```

### 3. Error Handling
- Use specific exception types
- Handle errors at the appropriate level
- Provide meaningful error messages
- Don't suppress exceptions without good reason

```python
# Good
try:
    character = character_service.get_character_by_id(character_id)
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    return character
except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))

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
# tests/test_character_service.py
import pytest
from services.character_service import CharacterService

class TestCharacterService:
    def test_get_character_by_id_success(self):
        # Arrange
        service = CharacterService([])
        
        # Act
        result = service.get_character_by_id(1)
        
        # Assert
        assert result is not None

    def test_get_character_by_id_not_found(self):
        # Test error scenario
        pass
```

### 3. Mocking and Fixtures
- Use pytest fixtures for test data setup
- Mock external dependencies
- Keep tests isolated and independent

## Structured Logging

### 1. Logging Configuration
```python
import logging
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()
```

### 2. Logging Best Practices
- Use appropriate log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Include relevant context in log messages
- Log important business events
- Don't log sensitive information (passwords, tokens)

```python
# Good - structured logging with context
logger.info("Character created", 
           character_id=character.id, 
           character_name=character.name,
           user_id=current_user.id)

# Bad - unstructured logging
print(f"Character {character.name} created")
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
from typing import List, Optional

def get_characters(limit: int = 20, skip: int = 0) -> List[Character]:
    pass

def get_character_by_id(character_id: int) -> Optional[Character]:
    pass
```

### 2. Documentation
- Write docstrings for all public functions and classes
- Use Google or NumPy docstring format
- Include parameter types and return types
- Document exceptions that may be raised

```python
def update_character(self, character_id: int, updated_data: dict) -> Optional[Character]:
    """
    Updates an existing character with new data.

    Args:
        character_id: The ID of the character to update
        updated_data: Dictionary containing the fields to update

    Returns:
        The updated character or None if not found

    Raises:
        ValueError: If the updated_data contains invalid values
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
character_service = CharacterService()
user_authentication_token = generate_token()

# Bad
cs = CharacterService()
token = gen_tok()
```

### 2. Classes
- Use `PascalCase` for class names
- Use descriptive names that indicate the class purpose

```python
# Good
class CharacterService:
    pass

class DatabaseConnection:
    pass

# Bad
class char_svc:
    pass
```

### 3. Constants
- Use `UPPER_SNAKE_CASE` for constants
- Define constants at module level

```python
# Good
SECRET_KEY = "supersecretkey"
MAX_CHARACTER_NAME_LENGTH = 100

# Bad
secret_key = "supersecretkey"
```

## Performance Guidelines

### 1. Database Queries
- Use appropriate indexes
- Avoid N+1 query problems
- Use pagination for large result sets
- Close database connections properly

### 2. Memory Usage
- Don't load all data into memory at once
- Use generators for large datasets
- Clean up resources in finally blocks or use context managers

## Security Guidelines

### 1. Input Validation
- Validate all input data
- Use Pydantic models for request validation
- Sanitize data before database operations

### 2. Authentication & Authorization
- Never store passwords in plain text
- Use secure JWT secret keys
- Implement proper role-based access control
- Validate JWT tokens on protected endpoints

### 3. Error Messages
- Don't expose internal system details in error messages
- Log detailed errors internally
- Return generic error messages to users

## Summary Checklist

Before committing code, ensure:
- [ ] Files are under 250 lines
- [ ] Functions are under 20 lines
- [ ] All functions have type hints
- [ ] Code follows SRP principle
- [ ] Error handling is implemented
- [ ] Logging is structured and meaningful
- [ ] Tests are written for new functionality
- [ ] Code is formatted with black: `black .`
- [ ] Code passes ruff linting: `ruff check .`
- [ ] Code passes type checking: `mypy .`
- [ ] No sensitive information is exposed

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