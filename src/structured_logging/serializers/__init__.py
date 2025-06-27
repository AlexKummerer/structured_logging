"""
Serializers package for structured logging

This package provides enhanced serialization support for complex Python data types.
"""

from .base import DEFAULT_CONFIG, serialize_for_logging, serialize_for_logging_lazy_aware
from .config import SerializationConfig
from .json_encoder import EnhancedJSONEncoder, enhanced_json_dumps
from .lazy import (
    LazyDict,
    LazySerializable,
    LazySerializationManager,
    create_lazy_serializable,
    get_lazy_serialization_stats,
    reset_lazy_serialization_stats,
    should_use_lazy_serialization,
)
from .registry import TypeRegistry, register_custom_serializer
from .smart_converter import SmartConverter
from .type_detection import TypeDetector
from .validation import (
    SchemaValidator,
    StructuredDataValidator,
    TypeAnnotationExtractor,
    ValidationError,
    auto_validate_function,
    get_validation_stats,
    register_validation_schema,
    reset_validation_stats,
    validate_log_data,
)

__all__ = [
    # Core functions
    "serialize_for_logging",
    "serialize_for_logging_lazy_aware",
    # Configuration
    "SerializationConfig",
    "DEFAULT_CONFIG",
    # JSON encoding
    "EnhancedJSONEncoder",
    "enhanced_json_dumps",
    # Lazy serialization
    "LazySerializable",
    "LazyDict",
    "LazySerializationManager",
    "create_lazy_serializable",
    "should_use_lazy_serialization",
    "get_lazy_serialization_stats",
    "reset_lazy_serialization_stats",
    # Type detection and conversion
    "TypeDetector",
    "SmartConverter",
    # Type registry
    "TypeRegistry",
    "register_custom_serializer",
    # Validation
    "ValidationError",
    "SchemaValidator",
    "TypeAnnotationExtractor",
    "StructuredDataValidator",
    "register_validation_schema",
    "validate_log_data",
    "auto_validate_function",
    "get_validation_stats",
    "reset_validation_stats",
]