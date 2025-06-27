"""
Lazy serialization system for deferred processing of large objects
"""

import json
import time
from typing import Any, Dict, Optional

from .config import SerializationConfig

# Forward imports to avoid circular dependencies
# These will be imported from base.py when it's created
serialize_for_logging = None
DEFAULT_CONFIG = None
_global_registry = None


class LazySerializable:
    """
    Lazy serialization wrapper that defers expensive serialization until needed

    This wrapper holds the original object and configuration, and only performs
    serialization when the result is actually accessed or converted to string.
    """

    def __init__(self, obj: Any, config: Optional[SerializationConfig] = None):
        self._obj = obj
        self._config = config or DEFAULT_CONFIG
        self._serialized = None
        self._is_serialized = False
        self._hash = None

    def __repr__(self) -> str:
        if self._is_serialized:
            return f"LazySerializable(serialized={type(self._serialized).__name__})"
        else:
            return f"LazySerializable(obj={type(self._obj).__name__}, pending=True)"

    def __str__(self) -> str:
        """Convert to string - triggers serialization if needed"""
        return str(self._get_serialized())

    def __hash__(self) -> int:
        """Support for use in sets/dicts - uses object identity"""
        if self._hash is None:
            self._hash = hash((id(self._obj), id(self._config)))
        return self._hash

    def __eq__(self, other) -> bool:
        """Equality comparison"""
        if isinstance(other, LazySerializable):
            return self._obj is other._obj and self._config == other._config
        return False

    def _get_serialized(self) -> Any:
        """Get the serialized value, performing serialization if needed"""
        if not self._is_serialized:
            # Import here to avoid circular dependency
            global serialize_for_logging
            if serialize_for_logging is None:
                from .base import serialize_for_logging
            self._serialized = serialize_for_logging(self._obj, self._config)
            self._is_serialized = True
        return self._serialized

    def is_serialized(self) -> bool:
        """Check if serialization has been performed"""
        return self._is_serialized

    def get_original(self) -> Any:
        """Get the original object without triggering serialization"""
        return self._obj

    def force_serialize(self) -> Any:
        """Force serialization and return the result"""
        return self._get_serialized()

    def to_json(self) -> str:
        """Convert to JSON string - triggers serialization"""
        return json.dumps(self._get_serialized(), separators=(",", ":"))


class LazyDict(dict):
    """
    Dictionary that supports lazy serialization of values

    Values can be LazySerializable objects that are only evaluated when accessed
    or when the dictionary is converted to a final form (JSON, etc.)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._serialization_forced = False

    def __getitem__(self, key):
        """Get item, forcing serialization if needed for final output"""
        value = super().__getitem__(key)
        if isinstance(value, LazySerializable) and self._serialization_forced:
            return value.force_serialize()
        return value

    def __setitem__(self, key, value):
        """Set item - can accept LazySerializable objects"""
        super().__setitem__(key, value)

    def force_serialize_all(self) -> dict:
        """Force serialization of all lazy values and return a regular dict"""
        result = {}
        for key, value in super().items():
            if isinstance(value, LazySerializable):
                result[key] = value.force_serialize()
            elif isinstance(value, LazyDict):
                result[key] = value.force_serialize_all()
            elif isinstance(value, list):
                result[key] = self._serialize_list(value)
            else:
                result[key] = value
        return result

    def _serialize_list(self, lst: list) -> list:
        """Helper to serialize lists that may contain lazy objects"""
        result = []
        for item in lst:
            if isinstance(item, LazySerializable):
                result.append(item.force_serialize())
            elif isinstance(item, LazyDict):
                result.append(item.force_serialize_all())
            elif isinstance(item, list):
                result.append(self._serialize_list(item))
            else:
                result.append(item)
        return result

    def mark_for_serialization(self):
        """Mark this dict to force serialization on access"""
        self._serialization_forced = True

    def to_json(self) -> str:
        """Convert to JSON string - forces all serialization"""
        return json.dumps(self.force_serialize_all(), separators=(",", ":"))

    def items(self):
        """Override items to handle lazy serialization"""
        if self._serialization_forced:
            return self.force_serialize_all().items()
        return super().items()

    def values(self):
        """Override values to handle lazy serialization"""
        if self._serialization_forced:
            return self.force_serialize_all().values()
        return super().values()


class LazySerializationManager:
    """
    Manager for lazy serialization operations with performance tracking

    Provides centralized control over when and how lazy serialization occurs,
    with statistics tracking and performance optimization.
    """

    def __init__(self):
        self._stats = {
            "objects_created": 0,
            "objects_serialized": 0,
            "objects_skipped": 0,
            "serialization_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        self._cache = {}  # Simple serialization cache
        self._cache_enabled = True
        self._max_cache_size = 1000

    def create_lazy(
        self, obj: Any, config: Optional[SerializationConfig] = None
    ) -> LazySerializable:
        """Create a lazy serializable wrapper"""
        self._stats["objects_created"] += 1
        return LazySerializable(obj, config)

    def create_lazy_dict(self, data: dict = None) -> LazyDict:
        """Create a lazy dictionary"""
        return LazyDict(data or {})

    def should_use_lazy(
        self, obj: Any, config: Optional[SerializationConfig] = None
    ) -> bool:
        """
        Determine if lazy serialization should be used for an object

        Args:
            obj: The object to potentially wrap
            config: Serialization configuration

        Returns:
            True if lazy serialization would be beneficial
        """
        # Import here to avoid circular dependency
        global DEFAULT_CONFIG, _global_registry
        if DEFAULT_CONFIG is None:
            from .base import DEFAULT_CONFIG
        if _global_registry is None:
            from .registry import _global_registry

        config = config or DEFAULT_CONFIG

        # Check if lazy serialization is enabled
        if not config.enable_lazy_serialization:
            return False

        # Don't use lazy for primitives unless they're large strings
        if isinstance(obj, (int, float, bool, type(None))):
            return False

        # For strings, check size threshold
        if isinstance(obj, str):
            return len(obj.encode("utf-8")) > config.lazy_threshold_bytes

        # For collections, check item count threshold
        if isinstance(obj, (dict, list, tuple, set)):
            return len(obj) > config.lazy_threshold_items

        # Force lazy for detection if enabled
        if config.force_lazy_for_detection and config.auto_detect_types:
            # Use lazy for objects that benefit from detection
            if isinstance(obj, str) and len(obj) > 10:
                return True

        # Use lazy for custom objects that need serialization
        serializer = _global_registry.get_serializer(obj)
        if serializer is not None:
            return True

        # Estimate object size for other objects
        try:
            obj_size = len(str(obj).encode("utf-8"))
            return obj_size > config.lazy_threshold_bytes
        except:
            # If we can't estimate size, err on side of using lazy for unknown objects
            return True

    def wrap_if_beneficial(
        self, obj: Any, config: Optional[SerializationConfig] = None
    ) -> Any:
        """
        Wrap object in lazy serialization if beneficial, otherwise return as-is

        Args:
            obj: Object to potentially wrap
            config: Serialization configuration

        Returns:
            LazySerializable wrapper or original object
        """
        if self.should_use_lazy(obj, config):
            return self.create_lazy(obj, config)
        return obj

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total_objects = self._stats["objects_created"]
        serialized = self._stats["objects_serialized"]
        skipped = self._stats["objects_skipped"]

        efficiency = (skipped / total_objects * 100) if total_objects > 0 else 0

        return {
            **self._stats,
            "efficiency_percent": round(efficiency, 2),
            "cache_hit_rate": round(
                self._stats["cache_hits"]
                / max(1, self._stats["cache_hits"] + self._stats["cache_misses"])
                * 100,
                2,
            ),
        }

    def reset_stats(self):
        """Reset performance statistics"""
        self._stats = {
            "objects_created": 0,
            "objects_serialized": 0,
            "objects_skipped": 0,
            "serialization_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        self._cache.clear()


# Global lazy serialization manager
_lazy_manager = LazySerializationManager()


def create_lazy_serializable(
    obj: Any, config: Optional[SerializationConfig] = None
) -> LazySerializable:
    """
    Create a lazy serializable wrapper for an object

    Args:
        obj: Object to wrap
        config: Optional serialization configuration

    Returns:
        LazySerializable wrapper
    """
    return _lazy_manager.create_lazy(obj, config)


def should_use_lazy_serialization(
    obj: Any, config: Optional[SerializationConfig] = None
) -> bool:
    """
    Check if lazy serialization would be beneficial for an object

    Args:
        obj: Object to check
        config: Optional serialization configuration

    Returns:
        True if lazy serialization is recommended
    """
    return _lazy_manager.should_use_lazy(obj, config)


def get_lazy_serialization_stats() -> Dict[str, Any]:
    """Get lazy serialization performance statistics"""
    return _lazy_manager.get_stats()


def reset_lazy_serialization_stats():
    """Reset lazy serialization statistics"""
    _lazy_manager.reset_stats()