"""
Schema validation and type annotation support for structured logging
"""

import inspect
import time
from dataclasses import is_dataclass
from functools import wraps
from typing import Any, Callable, Collection, Dict, List, Optional, get_type_hints

from .config import SerializationConfig

# Default configuration instance
DEFAULT_CONFIG = None


class ValidationError(Exception):
    """Exception raised when schema validation fails"""

    pass


class SchemaValidator:
    """
    Runtime schema validation for structured logging data

    Provides comprehensive validation of log data against predefined schemas
    with support for complex nested structures and type constraints.
    """

    def __init__(self, config: Optional[SerializationConfig] = None):
        global DEFAULT_CONFIG
        if DEFAULT_CONFIG is None:
            from .base import DEFAULT_CONFIG

        self.config = config or DEFAULT_CONFIG
        self._schemas = {}
        self._validation_stats = {
            "validations_performed": 0,
            "validation_failures": 0,
            "validation_time": 0.0,
        }

    def register_schema(self, name: str, schema: Dict[str, Any]) -> None:
        """
        Register a validation schema

        Args:
            name: Schema name/identifier
            schema: Schema definition dictionary
        """
        self._schemas[name] = self._compile_schema(schema)

    def _compile_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Compile and optimize schema for faster validation"""
        compiled = {}

        for field, constraints in schema.items():
            if isinstance(constraints, str):
                # Simple type constraint: {"field": "str"}
                compiled[field] = {"type": constraints, "required": True}
            elif isinstance(constraints, dict):
                # Complex constraint: {"field": {"type": "str", "max_length": 100}}
                compiled[field] = constraints.copy()
                if "required" not in compiled[field]:
                    compiled[field]["required"] = True
            elif isinstance(constraints, type):
                # Python type: {"field": str}
                compiled[field] = {"type": constraints.__name__, "required": True}
            else:
                compiled[field] = {"type": "any", "required": True}

        return compiled

    def validate(
        self, data: Dict[str, Any], schema_name: str, strict: bool = False
    ) -> bool:
        """
        Validate data against a registered schema

        Args:
            data: Data to validate
            schema_name: Name of registered schema
            strict: If True, raise ValidationError on failure

        Returns:
            True if valid, False otherwise (unless strict=True)

        Raises:
            ValidationError: If validation fails and strict=True
        """
        start_time = time.perf_counter()
        self._validation_stats["validations_performed"] += 1

        try:
            if schema_name not in self._schemas:
                raise ValidationError(f"Schema '{schema_name}' not found")

            schema = self._schemas[schema_name]
            errors = []

            # Check required fields
            for field, constraints in schema.items():
                if constraints.get("required", True) and field not in data:
                    errors.append(f"Required field '{field}' is missing")

            # Validate fields
            for field, value in data.items():
                if field in schema:
                    field_errors = self._validate_field(field, value, schema[field])
                    errors.extend(field_errors)

            if errors:
                self._validation_stats["validation_failures"] += 1
                if strict:
                    raise ValidationError(
                        f"Validation failed: {'; '.join(errors[:5])}"
                        + (f" ... and {len(errors) - 5} more" if len(errors) > 5 else "")
                    )
                return False

            return True

        finally:
            self._validation_stats["validation_time"] += time.perf_counter() - start_time

    def _validate_field(
        self, field: str, value: Any, constraints: Dict[str, Any]
    ) -> List[str]:
        """Validate a single field against constraints"""
        errors = []

        # Type validation
        expected_type = constraints.get("type")
        if expected_type and expected_type != "any":
            if not self._check_type(value, expected_type):
                errors.append(
                    f"Field '{field}' expected type {expected_type}, got {type(value).__name__}"
                )
                return errors  # Skip other validations if type is wrong

        # String constraints
        if isinstance(value, str):
            max_length = constraints.get("max_length")
            if max_length and len(value) > max_length:
                errors.append(
                    f"Field '{field}' exceeds max length of {max_length} characters"
                )

            min_length = constraints.get("min_length")
            if min_length and len(value) < min_length:
                errors.append(
                    f"Field '{field}' below min length of {min_length} characters"
                )

            pattern = constraints.get("pattern")
            if pattern:
                import re

                if not re.match(pattern, value):
                    errors.append(f"Field '{field}' doesn't match pattern: {pattern}")

        # Numeric constraints
        if isinstance(value, (int, float)):
            min_value = constraints.get("min_value")
            if min_value is not None and value < min_value:
                errors.append(f"Field '{field}' below minimum value of {min_value}")

            max_value = constraints.get("max_value")
            if max_value is not None and value > max_value:
                errors.append(f"Field '{field}' exceeds maximum value of {max_value}")

        # Collection constraints
        if isinstance(value, (list, set, tuple)):
            max_items = constraints.get("max_items")
            if max_items and len(value) > max_items:
                errors.append(f"Field '{field}' exceeds max items of {max_items}")

            min_items = constraints.get("min_items")
            if min_items and len(value) < min_items:
                errors.append(f"Field '{field}' below min items of {min_items}")

        # Enum constraints
        allowed_values = constraints.get("enum")
        if allowed_values and value not in allowed_values:
            errors.append(
                f"Field '{field}' value not in allowed set: {allowed_values}"
            )

        return errors

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type"""
        type_map = {
            "str": str,
            "int": int,
            "float": (int, float),  # int is acceptable for float
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "none": type(None),
            "number": (int, float),
            "collection": (list, set, tuple),
            "mapping": dict,
        }

        expected = type_map.get(expected_type.lower())
        if expected:
            return isinstance(value, expected)

        # Try to resolve as actual type
        try:
            expected_class = eval(expected_type)
            return isinstance(value, expected_class)
        except:
            return True  # Can't validate unknown types

    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        total = self._validation_stats["validations_performed"]
        failures = self._validation_stats["validation_failures"]

        return {
            **self._validation_stats,
            "success_rate": (
                ((total - failures) / total * 100) if total > 0 else 100.0
            ),
            "registered_schemas": len(self._schemas),
        }


class TypeAnnotationExtractor:
    """
    Extract type information from function annotations for automatic schema generation

    This class analyzes Python type annotations to automatically create
    validation schemas for function parameters and return types.
    """

    def __init__(self):
        self._type_cache = {}

    def extract_from_function(self, func: Callable) -> Dict[str, Any]:
        """
        Extract type information from function annotations

        Args:
            func: Function to analyze

        Returns:
            Schema definition based on type hints
        """
        # Check cache
        func_id = id(func)
        if func_id in self._type_cache:
            return self._type_cache[func_id]

        try:
            hints = get_type_hints(func)
        except Exception:
            # Fallback for functions without proper annotations
            hints = {}

        schema = {}
        sig = inspect.signature(func)

        for param_name, param in sig.parameters.items():
            if param_name == "self" or param_name == "cls":
                continue

            # Get type hint
            param_type = hints.get(param_name, param.annotation)
            if param_type == inspect.Parameter.empty:
                param_type = Any

            # Convert to schema constraint
            constraint = self._type_to_constraint(param_type)

            # Check if parameter has default
            if param.default != inspect.Parameter.empty:
                constraint["required"] = False
                constraint["default"] = param.default

            schema[param_name] = constraint

        # Cache result
        self._type_cache[func_id] = schema
        return schema

    def _type_to_constraint(self, type_hint: Any) -> Dict[str, Any]:
        """Convert Python type hint to schema constraint"""
        from typing import Union, get_args, get_origin

        # Handle None type
        if type_hint is type(None):
            return {"type": "none", "required": False}

        # Handle basic types
        basic_types = {
            str: "str",
            int: "int",
            float: "float",
            bool: "bool",
            list: "list",
            dict: "dict",
            set: "set",
            tuple: "tuple",
        }

        if type_hint in basic_types:
            return {"type": basic_types[type_hint], "required": True}

        # Handle Optional types
        origin = get_origin(type_hint)
        args = get_args(type_hint)

        if origin is Union:
            # Check if it's Optional (Union with None)
            if type(None) in args:
                # It's Optional[T]
                non_none_types = [t for t in args if t is not type(None)]
                if len(non_none_types) == 1:
                    constraint = self._type_to_constraint(non_none_types[0])
                    constraint["required"] = False
                    return constraint

        # Handle generic types (List[str], Dict[str, int], etc.)
        if origin in (list, List):
            constraint = {"type": "list", "required": True}
            if args:
                constraint["item_type"] = self._type_to_constraint(args[0])
            return constraint

        if origin in (dict, Dict):
            constraint = {"type": "dict", "required": True}
            if len(args) >= 2:
                constraint["key_type"] = self._type_to_constraint(args[0])
                constraint["value_type"] = self._type_to_constraint(args[1])
            return constraint

        # Handle Any
        if type_hint is Any:
            return {"type": "any", "required": True}

        # Default for unknown types
        return {"type": str(type_hint), "required": True}

    def extract_from_dataclass(self, dataclass_type: type) -> Dict[str, Any]:
        """Extract schema from dataclass definition"""
        if not is_dataclass(dataclass_type):
            raise ValueError(f"{dataclass_type} is not a dataclass")

        schema = {}
        hints = get_type_hints(dataclass_type)

        for field in dataclass_type.__dataclass_fields__.values():
            constraint = self._type_to_constraint(hints.get(field.name, field.type))

            # Check if field has default
            if field.default != field.default_factory:
                constraint["required"] = False
                if field.default != field.default_factory:
                    constraint["default"] = field.default

            schema[field.name] = constraint

        return schema


class StructuredDataValidator:
    """
    High-level validator for structured logging data

    Combines schema validation with type extraction for comprehensive
    data validation in logging contexts.
    """

    def __init__(self):
        self.schema_validator = SchemaValidator()
        self.type_extractor = TypeAnnotationExtractor()
        self._auto_schemas = {}

    def validate_with_function(
        self, data: Dict[str, Any], func: Callable, strict: bool = False
    ) -> bool:
        """
        Validate data against function type annotations

        Args:
            data: Data to validate
            func: Function whose annotations define the schema
            strict: If True, raise ValidationError on failure

        Returns:
            True if valid, False otherwise
        """
        # Get or create schema from function
        func_name = f"{func.__module__}.{func.__name__}"
        if func_name not in self._auto_schemas:
            schema = self.type_extractor.extract_from_function(func)
            self.schema_validator.register_schema(func_name, schema)
            self._auto_schemas[func_name] = schema

        return self.schema_validator.validate(data, func_name, strict)

    def validate_with_dataclass(
        self, data: Dict[str, Any], dataclass_type: type, strict: bool = False
    ) -> bool:
        """
        Validate data against dataclass definition

        Args:
            data: Data to validate
            dataclass_type: Dataclass type defining the schema
            strict: If True, raise ValidationError on failure

        Returns:
            True if valid, False otherwise
        """
        # Get or create schema from dataclass
        class_name = f"{dataclass_type.__module__}.{dataclass_type.__name__}"
        if class_name not in self._auto_schemas:
            schema = self.type_extractor.extract_from_dataclass(dataclass_type)
            self.schema_validator.register_schema(class_name, schema)
            self._auto_schemas[class_name] = schema

        return self.schema_validator.validate(data, class_name, strict)

    def create_validated_logger(self, logger, schema_name: str, strict: bool = False):
        """
        Create a logger wrapper that validates all log data

        Args:
            logger: Base logger instance
            schema_name: Schema to validate against
            strict: If True, invalid data prevents logging

        Returns:
            Wrapped logger with validation
        """

        class ValidatedLogger:
            def __init__(self, base_logger, validator, schema, strict):
                self._logger = base_logger
                self._validator = validator
                self._schema = schema
                self._strict = strict

            def _log_with_validation(self, level: str, message: str, **kwargs):
                # Validate kwargs against schema
                try:
                    if not self._validator.validate(kwargs, self._schema, self._strict):
                        if not self._strict:
                            # Log warning about validation failure
                            self._logger.warning(
                                f"Log data validation failed for schema {self._schema}"
                            )
                except ValidationError as e:
                    if self._strict:
                        raise
                    self._logger.error(f"Log validation error: {e}")
                    return

                # Log with validated data
                getattr(self._logger, level)(message, **kwargs)

            def debug(self, message: str, **kwargs):
                self._log_with_validation("debug", message, **kwargs)

            def info(self, message: str, **kwargs):
                self._log_with_validation("info", message, **kwargs)

            def warning(self, message: str, **kwargs):
                self._log_with_validation("warning", message, **kwargs)

            def error(self, message: str, **kwargs):
                self._log_with_validation("error", message, **kwargs)

            def critical(self, message: str, **kwargs):
                self._log_with_validation("critical", message, **kwargs)

        return ValidatedLogger(logger, self.schema_validator, schema_name, strict)


# Global instances
_global_schema_validator = SchemaValidator()
_global_data_validator = StructuredDataValidator()


def register_validation_schema(name: str, schema: Dict[str, Any]) -> None:
    """
    Register a global validation schema

    Args:
        name: Schema identifier
        schema: Schema definition
    """
    _global_schema_validator.register_schema(name, schema)


def validate_log_data(data: Dict[str, Any], schema_name: str) -> bool:
    """
    Validate log data against a registered schema

    Args:
        data: Data to validate
        schema_name: Registered schema name

    Returns:
        True if valid, False otherwise
    """
    return _global_schema_validator.validate(data, schema_name)


def auto_validate_function(func: Callable) -> Callable:
    """
    Decorator that validates function parameters based on type annotations

    Args:
        func: Function to wrap with validation

    Returns:
        Wrapped function with parameter validation
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract parameter names and values
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Validate against function schema
        if not _global_data_validator.validate_with_function(
            bound_args.arguments, func, strict=True
        ):
            raise ValidationError(f"Invalid parameters for {func.__name__}")

        return func(*args, **kwargs)

    return wrapper


def get_validation_stats() -> Dict[str, Any]:
    """Get global validation statistics"""
    return _global_schema_validator.get_stats()


def reset_validation_stats() -> None:
    """Reset global validation statistics"""
    _global_schema_validator._validation_stats = {
        "validations_performed": 0,
        "validation_failures": 0,
        "validation_time": 0.0,
    }