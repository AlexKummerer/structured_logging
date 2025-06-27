"""
Base serialization functions for structured logging
"""

from typing import Any, Dict, Optional

from .config import SerializationConfig
from .lazy import _lazy_manager
from .registry import _global_registry
from .smart_converter import SmartConverter
from .type_detection import TypeDetector

# Default configuration instance
DEFAULT_CONFIG = SerializationConfig()


def serialize_for_logging_lazy_aware(
    obj: Any, config: Optional[SerializationConfig] = None, use_lazy: bool = True
) -> Any:
    """
    Serialize an object for logging with lazy serialization support

    This function can return LazySerializable objects when beneficial,
    or perform immediate serialization when needed.

    Args:
        obj: Object to serialize
        config: Optional serialization configuration
        use_lazy: Whether to use lazy serialization when beneficial

    Returns:
        Serialized object or LazySerializable wrapper
    """
    config = config or DEFAULT_CONFIG

    # Check if we should use lazy serialization
    if use_lazy and config.enable_lazy_serialization:
        if _lazy_manager.should_use_lazy(obj, config):
            return _lazy_manager.create_lazy(obj, config)

    # Fall back to immediate serialization
    return serialize_for_logging(obj, config)


def _serialize_with_custom_serializer(obj: Any, config: SerializationConfig) -> Optional[Any]:
    """Try to serialize object with custom serializer"""
    serializer = _global_registry.get_serializer(obj)
    if serializer:
        try:
            return serializer(obj, config)
        except Exception as e:
            return {
                "__serialization_error__": str(e),
                "__type__": type(obj).__name__,
                "__repr__": repr(obj)[:200],
            }
    return None


def _serialize_primitive(obj: Any, config: SerializationConfig) -> Any:
    """Serialize primitive types with type detection"""
    if config.auto_detect_types:
        detector = TypeDetector(config)
        detected = detector.detect_and_convert(obj)
        if detected != obj:
            smart_converter = SmartConverter(config)
            return smart_converter._enhance_detected_object(detected)
    
    if isinstance(obj, str) and config.truncate_strings:
        if len(obj) > config.truncate_strings:
            return obj[: config.truncate_strings] + "..."
    return obj


def _serialize_collection(obj: Union[list, tuple], config: SerializationConfig) -> list:
    """Serialize list or tuple with size limits"""
    result = []
    for i, item in enumerate(obj):
        if i >= config.max_collection_size:
            result.append(f"... ({len(obj) - i} more items)")
            break
        result.append(serialize_for_logging(item, config))
    return result


def _serialize_dict(obj: dict, config: SerializationConfig) -> dict:
    """Serialize dictionary with size limits"""
    result = {}
    count = 0
    for key, value in obj.items():
        if count >= config.max_collection_size:
            result["..."] = f"({len(obj) - count} more items)"
            break
        str_key = str(key) if not isinstance(key, str) else key
        result[str_key] = serialize_for_logging(value, config)
        count += 1
    return result


def _serialize_fallback(obj: Any) -> dict:
    """Fallback serialization with safe repr"""
    try:
        obj_repr = repr(obj)[:200]
    except Exception as e:
        obj_repr = f"<repr failed: {str(e)[:50]}>"
    return {"__unserializable__": type(obj).__name__, "__repr__": obj_repr}


def serialize_for_logging(
    obj: Any, config: Optional[SerializationConfig] = None
) -> Any:
    """
    Serialize an object for logging with enhanced type support

    This is the main function used by the logging system to prepare
    complex objects for JSON serialization.

    Args:
        obj: Object to serialize
        config: Optional serialization configuration

    Returns:
        JSON-serializable representation of the object
    """
    config = config or DEFAULT_CONFIG

    if obj is None:
        return None

    # Try custom serializers first
    custom_result = _serialize_with_custom_serializer(obj, config)
    if custom_result is not None:
        return custom_result

    # Handle primitives
    if isinstance(obj, (str, int, float, bool)):
        return _serialize_primitive(obj, config)

    # Handle collections
    if isinstance(obj, (list, tuple)):
        return _serialize_collection(obj, config)

    if isinstance(obj, dict):
        return _serialize_dict(obj, config)

    # Final fallback
    return _serialize_fallback(obj)


# Update lazy.py references
from . import lazy

lazy.serialize_for_logging = serialize_for_logging
lazy.DEFAULT_CONFIG = DEFAULT_CONFIG
lazy._global_registry = _global_registry

# Update smart_converter.py reference
from . import smart_converter

smart_converter.serialize_for_logging = serialize_for_logging