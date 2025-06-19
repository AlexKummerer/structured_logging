"""
Enhanced serialization support for complex Python data types

This module provides custom serializers that extend JSON functionality
to handle Python objects that aren't natively JSON serializable.
"""

import inspect
import json
import re
import time
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path, PurePath
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
from uuid import UUID

# Optional imports for scientific libraries
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

try:
    import scipy.sparse

    HAS_SCIPY = True
except ImportError:
    scipy = None
    HAS_SCIPY = False


class SerializationConfig:
    """Configuration for enhanced serialization"""

    def __init__(
        self,
        datetime_format: str = "iso",  # iso, timestamp, custom
        custom_datetime_format: Optional[str] = None,
        decimal_as_float: bool = False,
        include_type_hints: bool = False,
        max_collection_size: int = 1000,
        truncate_strings: Optional[int] = None,
        enum_as_value: bool = True,
        dataclass_as_dict: bool = True,
        path_as_string: bool = True,
        # NumPy configuration
        numpy_array_max_size: int = 100,
        numpy_array_precision: int = 6,  # Decimal precision for floats
        numpy_include_metadata: bool = True,  # Include shape, dtype, etc.
        numpy_compression_threshold: int = 1000,  # Compress arrays larger than this
        numpy_sample_size: int = 10,  # Sample size for large arrays
        numpy_stats_for_numeric: bool = True,  # Include statistical summaries
        numpy_preserve_sparse: bool = True,  # Handle sparse arrays specially
        numpy_handle_inf_nan: bool = True,  # Handle inf/nan values
        # Pandas configuration
        pandas_max_rows: int = 5,
        pandas_max_cols: int = 10,  # Max columns to include in full serialization
        pandas_include_dtypes: bool = False,
        pandas_include_index: bool = True,  # Include index information
        pandas_include_memory_usage: bool = True,  # Include memory statistics
        pandas_categorical_as_codes: bool = False,  # Serialize categoricals as codes
        pandas_datetime_format: str = "iso",  # Format for datetime columns
        pandas_include_describe: bool = True,  # Include statistical description
        pandas_sample_method: str = "head_tail",  # head_tail, random, or systematic
        pandas_handle_multiindex: bool = True,  # Special handling for MultiIndex
        auto_detect_types: bool = True,
        auto_convert_strings: bool = True,
        auto_convert_numbers: bool = True,
        detect_datetime_strings: bool = True,
        detect_uuid_strings: bool = True,
        detect_json_strings: bool = True,
        detect_url_strings: bool = True,
        type_detection_cache_size: int = 1000,
        # Lazy serialization options
        enable_lazy_serialization: bool = True,
        lazy_threshold_bytes: int = 1000,  # Use lazy for objects larger than this
        lazy_threshold_items: int = 10,  # Use lazy for collections with more items
        lazy_cache_size: int = 500,  # Cache size for lazy serialization
        force_lazy_for_detection: bool = True,  # Always use lazy when auto-detection is on
    ):
        self.datetime_format = datetime_format
        self.custom_datetime_format = custom_datetime_format
        self.decimal_as_float = decimal_as_float
        self.include_type_hints = include_type_hints
        self.max_collection_size = max_collection_size
        self.truncate_strings = truncate_strings
        self.enum_as_value = enum_as_value
        self.dataclass_as_dict = dataclass_as_dict
        self.path_as_string = path_as_string
        # NumPy settings
        self.numpy_array_max_size = numpy_array_max_size
        self.numpy_array_precision = numpy_array_precision
        self.numpy_include_metadata = numpy_include_metadata
        self.numpy_compression_threshold = numpy_compression_threshold
        self.numpy_sample_size = numpy_sample_size
        self.numpy_stats_for_numeric = numpy_stats_for_numeric
        self.numpy_preserve_sparse = numpy_preserve_sparse
        self.numpy_handle_inf_nan = numpy_handle_inf_nan
        # Pandas settings
        self.pandas_max_rows = pandas_max_rows
        self.pandas_max_cols = pandas_max_cols
        self.pandas_include_dtypes = pandas_include_dtypes
        self.pandas_include_index = pandas_include_index
        self.pandas_include_memory_usage = pandas_include_memory_usage
        self.pandas_categorical_as_codes = pandas_categorical_as_codes
        self.pandas_datetime_format = pandas_datetime_format
        self.pandas_include_describe = pandas_include_describe
        self.pandas_sample_method = pandas_sample_method
        self.pandas_handle_multiindex = pandas_handle_multiindex
        self.auto_detect_types = auto_detect_types
        self.auto_convert_strings = auto_convert_strings
        self.auto_convert_numbers = auto_convert_numbers
        self.detect_datetime_strings = detect_datetime_strings
        self.detect_uuid_strings = detect_uuid_strings
        self.detect_json_strings = detect_json_strings
        self.detect_url_strings = detect_url_strings
        self.type_detection_cache_size = type_detection_cache_size
        # Lazy serialization settings
        self.enable_lazy_serialization = enable_lazy_serialization
        self.lazy_threshold_bytes = lazy_threshold_bytes
        self.lazy_threshold_items = lazy_threshold_items
        self.lazy_cache_size = lazy_cache_size
        self.force_lazy_for_detection = force_lazy_for_detection


class TypeDetector:
    """Automatic type detection and conversion system"""

    def __init__(self, config: SerializationConfig):
        self.config = config
        self._cache = {}  # Simple cache for detected types
        self._cache_hits = 0
        self._cache_misses = 0

        # Pre-compiled regex patterns for efficiency
        self._uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            re.IGNORECASE,
        )

        self._datetime_patterns = [
            # ISO 8601 formats
            re.compile(
                r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?$"
            ),
            # Common date formats
            re.compile(r"^\d{4}-\d{2}-\d{2}$"),
            re.compile(r"^\d{2}/\d{2}/\d{4}$"),
            re.compile(r"^\d{2}-\d{2}-\d{4}$"),
            # Common datetime formats
            re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"),
            re.compile(r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}$"),
        ]

        self._url_pattern = re.compile(
            r"^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$",
            re.IGNORECASE,
        )

        self._number_patterns = {
            "integer": re.compile(r"^[+-]?\d+$"),
            "float": re.compile(r"^[+-]?\d*\.\d+$"),
            "scientific": re.compile(r"^[+-]?\d*\.?\d+[eE][+-]?\d+$"),
            "decimal": re.compile(r"^[+-]?\d+\.\d{2,}$"),  # Likely currency/precision
        }

    def detect_and_convert(self, obj: Any) -> Any:
        """
        Detect type and auto-convert if appropriate

        Args:
            obj: Object to detect and potentially convert

        Returns:
            Original object or converted version
        """
        if not self.config.auto_detect_types:
            return obj

        # Skip if already a complex type that we handle
        if not isinstance(obj, (str, int, float)):
            return obj

        # Check cache first
        cache_key = (type(obj).__name__, str(obj)[:100])  # Limit cache key length
        if cache_key in self._cache:
            self._cache_hits += 1
            converter = self._cache[cache_key]
            if converter is None:
                return obj
            return converter(obj)

        self._cache_misses += 1

        # Detect and convert
        converter = self._detect_type_converter(obj)

        # Cache the result (with size limit)
        if len(self._cache) < self.config.type_detection_cache_size:
            self._cache[cache_key] = converter

        if converter is None:
            return obj

        try:
            return converter(obj)
        except Exception:
            # If conversion fails, return original
            return obj

    def _detect_type_converter(self, obj: Any) -> Optional[Callable]:
        """Detect appropriate type converter for an object"""

        if isinstance(obj, str):
            return self._detect_string_type(obj)
        elif isinstance(obj, (int, float)) and self.config.auto_convert_numbers:
            return self._detect_numeric_type(obj)

        return None

    def _detect_string_type(self, value: str) -> Optional[Callable]:
        """Detect type for string values"""
        if not self.config.auto_convert_strings or not value:
            return None

        # UUID detection
        if self.config.detect_uuid_strings and self._uuid_pattern.match(value):
            return lambda x: {"type": "uuid", "value": x}

        # DateTime detection
        if self.config.detect_datetime_strings:
            for pattern in self._datetime_patterns:
                if pattern.match(value):
                    return lambda x: {
                        "type": "datetime_string",
                        "value": x,
                        "detected_format": True,
                    }

        # URL detection
        if self.config.detect_url_strings and self._url_pattern.match(value):
            return lambda x: {"type": "url", "value": x}

        # JSON detection
        if self.config.detect_json_strings and self._looks_like_json(value):
            return self._json_converter

        # Numeric string detection
        for num_type, pattern in self._number_patterns.items():
            if pattern.match(value):
                return self._get_numeric_converter(num_type)

        return None

    def _detect_numeric_type(self, value: Union[int, float]) -> Optional[Callable]:
        """Detect enhanced representation for numeric values"""

        # Timestamp detection (Unix timestamp)
        if isinstance(value, (int, float)) and 1000000000 <= value <= 9999999999:
            # Likely a timestamp (between 2001 and 2286)
            return lambda x: {
                "type": "timestamp",
                "value": x,
                "human_readable": datetime.fromtimestamp(x).isoformat() + "Z",
            }

        # Large numbers with unit detection
        if isinstance(value, (int, float)) and abs(value) >= 1000:
            return lambda x: {
                "type": "large_number",
                "value": x,
                "formatted": self._format_large_number(x),
            }

        return None

    def _looks_like_json(self, value: str) -> bool:
        """Quick check if string looks like JSON"""
        if len(value) < 2:
            return False

        stripped = value.strip()
        return (stripped.startswith("{") and stripped.endswith("}")) or (
            stripped.startswith("[") and stripped.endswith("]")
        )

    def _json_converter(self, value: str) -> Dict[str, Any]:
        """Convert JSON string to structured representation"""
        try:
            parsed = json.loads(value)
            return {
                "type": "json_string",
                "parsed": parsed,
                "original_length": len(value),
            }
        except json.JSONDecodeError:
            return {
                "type": "json_like_string",
                "value": value[:100] + "..." if len(value) > 100 else value,
            }

    def _get_numeric_converter(self, num_type: str) -> Callable:
        """Get converter for numeric strings"""

        def converter(value: str) -> Dict[str, Any]:
            try:
                if num_type == "integer":
                    return {
                        "type": "numeric_string",
                        "value": int(value),
                        "original": value,
                    }
                elif num_type == "float":
                    return {
                        "type": "numeric_string",
                        "value": float(value),
                        "original": value,
                    }
                elif num_type == "decimal":
                    return {
                        "type": "decimal_string",
                        "value": str(Decimal(value)),
                        "original": value,
                    }
                elif num_type == "scientific":
                    return {
                        "type": "scientific_string",
                        "value": float(value),
                        "original": value,
                    }
            except (ValueError, TypeError):
                pass
            return {"type": "numeric_like_string", "value": value}

        return converter

    def _format_large_number(self, value: Union[int, float]) -> str:
        """Format large numbers with appropriate units"""
        abs_value = abs(value)

        if abs_value >= 1_000_000_000_000:  # Trillion
            return f"{value / 1_000_000_000_000:.2f}T"
        elif abs_value >= 1_000_000_000:  # Billion
            return f"{value / 1_000_000_000:.2f}B"
        elif abs_value >= 1_000_000:  # Million
            return f"{value / 1_000_000:.2f}M"
        elif abs_value >= 1_000:  # Thousand
            return f"{value / 1_000:.2f}K"
        else:
            return str(value)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (
            (self._cache_hits / total_requests * 100) if total_requests > 0 else 0
        )

        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests,
        }

    def clear_cache(self) -> None:
        """Clear the type detection cache"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


class SmartConverter:
    """Smart converter that combines type detection with serialization"""

    def __init__(self, config: SerializationConfig):
        self.config = config
        self.detector = TypeDetector(config)
        self.type_registry = TypeRegistry()

    def convert_intelligently(self, obj: Any) -> Any:
        """
        Intelligently convert object using detection + serialization

        Args:
            obj: Object to convert

        Returns:
            Intelligently converted object
        """
        # First, try auto-detection and conversion
        detected = self.detector.detect_and_convert(obj)

        # If detection changed the object, serialize the new version
        if detected != obj:
            return self._serialize_detected(detected)

        # Otherwise, use standard serialization without auto-detection to avoid recursion
        config_without_detection = SerializationConfig(
            datetime_format=self.config.datetime_format,
            custom_datetime_format=self.config.custom_datetime_format,
            decimal_as_float=self.config.decimal_as_float,
            include_type_hints=self.config.include_type_hints,
            max_collection_size=self.config.max_collection_size,
            truncate_strings=self.config.truncate_strings,
            enum_as_value=self.config.enum_as_value,
            dataclass_as_dict=self.config.dataclass_as_dict,
            path_as_string=self.config.path_as_string,
            numpy_array_max_size=self.config.numpy_array_max_size,
            pandas_max_rows=self.config.pandas_max_rows,
            pandas_include_dtypes=self.config.pandas_include_dtypes,
            auto_detect_types=False,  # Disable auto-detection to prevent recursion
        )
        return serialize_for_logging(obj, config_without_detection)

    def _serialize_detected(self, obj: Any) -> Any:
        """Serialize a detected/converted object"""
        # If it's a dict with type info, enhance it
        if isinstance(obj, dict) and "type" in obj:
            return self._enhance_detected_object(obj)

        # Otherwise serialize normally without auto-detection to avoid recursion
        config_without_detection = SerializationConfig(
            datetime_format=self.config.datetime_format,
            custom_datetime_format=self.config.custom_datetime_format,
            decimal_as_float=self.config.decimal_as_float,
            include_type_hints=self.config.include_type_hints,
            max_collection_size=self.config.max_collection_size,
            truncate_strings=self.config.truncate_strings,
            enum_as_value=self.config.enum_as_value,
            dataclass_as_dict=self.config.dataclass_as_dict,
            path_as_string=self.config.path_as_string,
            numpy_array_max_size=self.config.numpy_array_max_size,
            pandas_max_rows=self.config.pandas_max_rows,
            pandas_include_dtypes=self.config.pandas_include_dtypes,
            auto_detect_types=False,  # Disable auto-detection to prevent recursion
        )
        return serialize_for_logging(obj, config_without_detection)

    def _enhance_detected_object(self, obj: dict) -> dict:
        """Enhance detected objects with additional metadata"""
        result = obj.copy()

        # Add confidence and detection metadata
        result["_detection"] = {
            "detected_at": datetime.now().isoformat() + "Z",
            "confidence": self._calculate_confidence(obj),
            "auto_detected": True,
        }

        return result

    def _calculate_confidence(self, obj: dict) -> float:
        """Calculate confidence score for detection"""
        obj_type = obj.get("type", "unknown")

        # Higher confidence for stricter patterns
        confidence_map = {
            "uuid": 0.95,
            "url": 0.90,
            "timestamp": 0.85,
            "json_string": 0.80,
            "decimal_string": 0.75,
            "numeric_string": 0.70,
            "datetime_string": 0.65,
            "large_number": 0.60,
            "json_like_string": 0.30,
            "numeric_like_string": 0.25,
        }

        return confidence_map.get(obj_type, 0.50)

    def get_detection_stats(self) -> Dict[str, Any]:
        """Get type detection statistics"""
        return self.detector.get_cache_stats()


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


class TypeRegistry:
    """Registry for custom type serializers"""

    def __init__(self):
        self._serializers: Dict[Type, Callable[[Any, SerializationConfig], Any]] = {}
        self._setup_default_serializers()

    def _setup_default_serializers(self) -> None:
        """Setup built-in serializers for common types"""

        # Date and time types
        self.register(datetime, self._serialize_datetime)
        self.register(date, self._serialize_date)
        self.register(time, self._serialize_time)
        self.register(timedelta, self._serialize_timedelta)

        # Numeric types
        self.register(Decimal, self._serialize_decimal)
        self.register(complex, self._serialize_complex)

        # UUID
        self.register(UUID, self._serialize_uuid)

        # Path objects
        self.register(Path, self._serialize_path)
        self.register(PurePath, self._serialize_path)

        # Collections
        self.register(set, self._serialize_set)
        self.register(frozenset, self._serialize_set)

        # Enum
        self.register(Enum, self._serialize_enum)

        # Bytes
        self.register(bytes, self._serialize_bytes)
        self.register(bytearray, self._serialize_bytes)

        # Scientific types (if available)
        if HAS_NUMPY:
            # NumPy scalar types
            self.register(np.integer, self._serialize_numpy_scalar)
            self.register(np.floating, self._serialize_numpy_scalar)
            self.register(np.complexfloating, self._serialize_numpy_scalar)
            self.register(np.bool_, self._serialize_numpy_scalar)
            self.register(np.str_, self._serialize_numpy_scalar)
            # Note: np.unicode_ was removed in NumPy 2.0, now it's just np.str_

            # NumPy array types
            self.register(np.ndarray, self._serialize_numpy_array)
            self.register(np.matrix, self._serialize_numpy_matrix)

            # NumPy special arrays
            if hasattr(np, "ma"):
                self.register(np.ma.MaskedArray, self._serialize_numpy_masked_array)

        if HAS_PANDAS:
            # Pandas core types
            self.register(pd.DataFrame, self._serialize_dataframe)
            self.register(pd.Series, self._serialize_series)
            self.register(pd.Timestamp, self._serialize_pandas_timestamp)

            # Note: Index serialization is handled by _serialize_pandas_index helper function

            # Pandas data types
            self.register(pd.Categorical, self._serialize_pandas_categorical)

        # SciPy types (if available)
        if HAS_SCIPY:
            # Register common scipy sparse matrix types
            for sparse_type in [
                "csr_matrix",
                "csc_matrix",
                "coo_matrix",
                "bsr_matrix",
                "lil_matrix",
                "dok_matrix",
                "dia_matrix",
            ]:
                if hasattr(scipy.sparse, sparse_type):
                    self.register(
                        getattr(scipy.sparse, sparse_type),
                        self._serialize_scipy_sparse_matrix,
                    )

    def register(
        self, type_class: Type, serializer: Callable[[Any, SerializationConfig], Any]
    ) -> None:
        """Register a custom serializer for a type"""
        self._serializers[type_class] = serializer

    def get_serializer(
        self, obj: Any
    ) -> Optional[Callable[[Any, SerializationConfig], Any]]:
        """Get serializer for an object"""
        obj_type = type(obj)

        # Direct type match
        if obj_type in self._serializers:
            return self._serializers[obj_type]

        # Check inheritance (MRO - Method Resolution Order)
        for base_type in obj_type.__mro__:
            if base_type in self._serializers:
                return self._serializers[base_type]

        # Special cases
        if is_dataclass(obj):
            return self._serialize_dataclass

        if isinstance(obj, Enum):
            return self._serialize_enum

        return None

    # Built-in serializers

    @staticmethod
    def _serialize_datetime(dt: datetime, config: SerializationConfig) -> str:
        """Serialize datetime objects"""
        if config.datetime_format == "timestamp":
            return dt.timestamp()
        elif config.datetime_format == "custom" and config.custom_datetime_format:
            return dt.strftime(config.custom_datetime_format)
        else:  # iso format (default)
            return dt.isoformat() + "Z" if dt.tzinfo is None else dt.isoformat()

    @staticmethod
    def _serialize_date(d: date, config: SerializationConfig) -> str:
        """Serialize date objects"""
        return d.isoformat()

    @staticmethod
    def _serialize_time(t: time, config: SerializationConfig) -> str:
        """Serialize time objects"""
        return t.isoformat()

    @staticmethod
    def _serialize_timedelta(
        td: timedelta, config: SerializationConfig
    ) -> Dict[str, float]:
        """Serialize timedelta objects"""
        return {
            "days": td.days,
            "seconds": td.seconds,
            "microseconds": td.microseconds,
            "total_seconds": td.total_seconds(),
        }

    @staticmethod
    def _serialize_decimal(
        decimal_val: Decimal, config: SerializationConfig
    ) -> Union[float, str]:
        """Serialize Decimal objects"""
        if config.decimal_as_float:
            return float(decimal_val)
        else:
            return str(decimal_val)

    @staticmethod
    def _serialize_complex(
        complex_val: complex, config: SerializationConfig
    ) -> Dict[str, float]:
        """Serialize complex numbers"""
        return {"real": complex_val.real, "imag": complex_val.imag}

    @staticmethod
    def _serialize_uuid(uuid_val: UUID, config: SerializationConfig) -> str:
        """Serialize UUID objects"""
        return str(uuid_val)

    @staticmethod
    def _serialize_path(
        path_obj: Union[Path, PurePath], config: SerializationConfig
    ) -> Union[str, Dict[str, Any]]:
        """Serialize Path objects"""
        if config.path_as_string:
            return str(path_obj)
        else:
            return {
                "path": str(path_obj),
                "is_absolute": path_obj.is_absolute(),
                "parts": path_obj.parts,
                "suffix": path_obj.suffix,
            }

    @staticmethod
    def _serialize_set(
        set_obj: Union[set, frozenset], config: SerializationConfig
    ) -> list:
        """Serialize set objects to lists"""
        items = list(set_obj)
        if len(items) > config.max_collection_size:
            items = items[: config.max_collection_size]
            items.append(
                f"... ({len(set_obj) - config.max_collection_size} more items)"
            )
        return items

    @staticmethod
    def _serialize_enum(
        enum_obj: Enum, config: SerializationConfig
    ) -> Union[Any, Dict[str, Any]]:
        """Serialize Enum objects"""
        if config.enum_as_value:
            return enum_obj.value
        else:
            return {
                "name": enum_obj.name,
                "value": enum_obj.value,
                "type": enum_obj.__class__.__name__,
            }

    @staticmethod
    def _serialize_bytes(
        bytes_obj: Union[bytes, bytearray], config: SerializationConfig
    ) -> Dict[str, Any]:
        """Serialize bytes objects"""
        try:
            # Try to decode as UTF-8 for text data
            decoded = bytes_obj.decode("utf-8")
            if config.truncate_strings and len(decoded) > config.truncate_strings:
                decoded = decoded[: config.truncate_strings] + "..."
            return {"type": "text", "data": decoded, "size": len(bytes_obj)}
        except UnicodeDecodeError:
            # Binary data - show hex representation (truncated)
            hex_data = bytes_obj.hex()
            if len(hex_data) > 100:
                hex_data = hex_data[:100] + "..."
            return {"type": "binary", "hex": hex_data, "size": len(bytes_obj)}

    @staticmethod
    def _serialize_dataclass(obj: Any, config: SerializationConfig) -> Dict[str, Any]:
        """Serialize dataclass objects"""
        if config.dataclass_as_dict:
            result = asdict(obj)
            if config.include_type_hints:
                result["__type__"] = obj.__class__.__name__
            return result
        else:
            return {
                "__type__": obj.__class__.__name__,
                "__fields__": list(obj.__dataclass_fields__.keys()),
            }

    # NumPy serializers (if available)

    def _serialize_numpy_scalar(self, scalar: Any, config: SerializationConfig) -> Any:
        """Serialize NumPy scalar types with enhanced type handling"""
        if not HAS_NUMPY:
            return str(scalar)

        # Handle inf/nan values if configured
        if config.numpy_handle_inf_nan and isinstance(scalar, np.floating):
            if np.isnan(scalar):
                return {"__numpy_special__": "nan", "__dtype__": str(scalar.dtype)}
            elif np.isinf(scalar):
                sign = "+" if scalar > 0 else "-"
                return {
                    "__numpy_special__": f"{sign}inf",
                    "__dtype__": str(scalar.dtype),
                }

        # Enhanced type conversion with precision control
        if isinstance(scalar, np.integer):
            # Preserve integer type information if needed
            result = int(scalar)
            if config.numpy_include_metadata:
                return {
                    "value": result,
                    "__numpy_dtype__": str(scalar.dtype),
                    "__numpy_type__": "integer",
                }
            return result

        elif isinstance(scalar, np.floating):
            # Apply precision formatting for floats
            if config.numpy_array_precision is not None:
                result = round(float(scalar), config.numpy_array_precision)
            else:
                result = float(scalar)

            if config.numpy_include_metadata:
                return {
                    "value": result,
                    "__numpy_dtype__": str(scalar.dtype),
                    "__numpy_type__": "floating",
                }
            return result

        elif isinstance(scalar, np.complexfloating):
            real_part = (
                round(float(scalar.real), config.numpy_array_precision)
                if config.numpy_array_precision
                else float(scalar.real)
            )
            imag_part = (
                round(float(scalar.imag), config.numpy_array_precision)
                if config.numpy_array_precision
                else float(scalar.imag)
            )

            result = {"real": real_part, "imag": imag_part}

            if config.numpy_include_metadata:
                result.update(
                    {"__numpy_dtype__": str(scalar.dtype), "__numpy_type__": "complex"}
                )
            return result

        elif isinstance(scalar, np.bool_):
            result = bool(scalar)
            if config.numpy_include_metadata:
                return {
                    "value": result,
                    "__numpy_dtype__": str(scalar.dtype),
                    "__numpy_type__": "boolean",
                }
            return result

        elif isinstance(scalar, np.str_):
            result = str(scalar)
            if config.numpy_include_metadata:
                return {
                    "value": result,
                    "__numpy_dtype__": str(scalar.dtype),
                    "__numpy_type__": "string",
                }
            return result

        else:
            # Fallback for other NumPy scalar types
            try:
                result = scalar.item()
                if config.numpy_include_metadata:
                    return {
                        "value": result,
                        "__numpy_dtype__": str(scalar.dtype),
                        "__numpy_type__": "unknown",
                    }
                return result
            except (ValueError, TypeError):
                return {
                    "__numpy_scalar_error__": "Cannot convert to Python type",
                    "__numpy_dtype__": str(scalar.dtype),
                    "__repr__": str(scalar),
                }

    def _serialize_numpy_array(
        self, array: Any, config: SerializationConfig
    ) -> Dict[str, Any]:
        """Enhanced NumPy array serialization with comprehensive features"""
        if not HAS_NUMPY:
            return {"error": "NumPy not available"}

        result = {}

        # Basic metadata
        if config.numpy_include_metadata:
            result.update(
                {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "size": int(array.size),
                    "ndim": int(array.ndim),
                    "itemsize": int(array.itemsize),
                    "nbytes": int(array.nbytes),
                    "__numpy_type__": "ndarray",
                }
            )

            # Memory layout information
            result["flags"] = {
                "c_contiguous": bool(array.flags.c_contiguous),
                "f_contiguous": bool(array.flags.f_contiguous),
                "writeable": bool(array.flags.writeable),
                "aligned": bool(array.flags.aligned),
            }
        else:
            # Minimal metadata
            result.update(
                {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "size": int(array.size),
                }
            )

        # Handle special values (inf/nan) before serialization
        if config.numpy_handle_inf_nan and np.issubdtype(array.dtype, np.floating):
            has_nan = bool(np.any(np.isnan(array)))
            has_inf = bool(np.any(np.isinf(array)))
            if has_nan or has_inf:
                result["special_values"] = {
                    "has_nan": has_nan,
                    "has_inf": has_inf,
                    "nan_count": int(np.sum(np.isnan(array))) if has_nan else 0,
                    "inf_count": int(np.sum(np.isinf(array))) if has_inf else 0,
                }

        # Data serialization strategy based on size
        if array.size <= config.numpy_array_max_size:
            # Small arrays: include full data
            try:
                if config.numpy_array_precision and np.issubdtype(
                    array.dtype, np.floating
                ):
                    # Apply precision rounding for floating point arrays
                    rounded_array = np.round(array, config.numpy_array_precision)
                    result["data"] = rounded_array.tolist()
                else:
                    result["data"] = array.tolist()
                result["serialization_method"] = "full"
            except (ValueError, TypeError):
                # Fallback for arrays that can't be converted to list
                result["data_error"] = "Cannot convert array to list"
                result["repr_sample"] = str(array.flat[:5])

        elif array.size <= config.numpy_compression_threshold:
            # Medium arrays: include sample and statistics
            result.update(TypeRegistry._get_numpy_array_summary(array, config))
            result["serialization_method"] = "summary"

        else:
            # Large arrays: compressed representation
            result.update(TypeRegistry._get_numpy_array_compressed(array, config))
            result["serialization_method"] = "compressed"

        # Statistics for numeric arrays (if enabled)
        if (
            config.numpy_stats_for_numeric
            and np.issubdtype(array.dtype, np.number)
            and array.size > 0
        ):
            try:
                stats = TypeRegistry._compute_numpy_stats(array, config)
                result["statistics"] = stats
            except Exception as e:
                result["statistics_error"] = f"Could not compute statistics: {str(e)}"

        # Handle sparse arrays (if scipy is available and array is sparse)
        if config.numpy_preserve_sparse:
            try:
                # Check if it's a scipy sparse matrix
                if hasattr(array, "format") and hasattr(array, "nnz"):
                    result["sparse_info"] = {
                        "format": array.format,
                        "nnz": int(array.nnz),
                        "density": (
                            float(array.nnz / array.size) if array.size > 0 else 0.0
                        ),
                        "__is_sparse__": True,
                    }
            except:
                pass  # Not a sparse array or scipy not available

        return result

    @staticmethod
    def _get_numpy_array_summary(
        array: Any, config: SerializationConfig
    ) -> Dict[str, Any]:
        """Generate summary for medium-sized arrays"""
        result = {}

        # Sample data
        sample_size = min(config.numpy_sample_size, array.size)
        if array.ndim == 1:
            # 1D array: take first and last elements
            if array.size <= sample_size * 2:
                sample_data = array.tolist()
            else:
                head = array[: sample_size // 2].tolist()
                tail = array[-sample_size // 2 :].tolist()
                sample_data = {"head": head, "tail": tail}
        else:
            # Multi-dimensional: flatten and sample
            flat = array.flatten()
            if flat.size <= sample_size:
                sample_data = flat.tolist()
            else:
                sample_data = flat[:sample_size].tolist()

        result["sample"] = sample_data
        return result

    @staticmethod
    def _get_numpy_array_compressed(
        array: Any, config: SerializationConfig
    ) -> Dict[str, Any]:
        """Generate compressed representation for large arrays"""
        result = {}

        # Minimal sample for very large arrays
        sample_size = min(5, array.size)
        flat = array.flatten()
        result["minimal_sample"] = flat[:sample_size].tolist()

        # Memory and storage information
        result["memory_info"] = {
            "memory_usage_mb": float(array.nbytes / (1024 * 1024)),
            "compression_ratio": f"1:{array.size // config.numpy_array_max_size}",
            "elements_omitted": array.size - sample_size,
        }

        return result

    @staticmethod
    def _compute_numpy_stats(array: Any, config: SerializationConfig) -> Dict[str, Any]:
        """Compute statistical summary for numeric arrays"""
        stats = {}

        try:
            # Basic statistics
            stats["min"] = float(np.min(array))
            stats["max"] = float(np.max(array))
            stats["mean"] = float(np.mean(array))
            stats["std"] = float(np.std(array))

            # Additional statistics
            stats["median"] = float(np.median(array))
            stats["var"] = float(np.var(array))

            # Percentiles
            stats["percentiles"] = {
                "25th": float(np.percentile(array, 25)),
                "75th": float(np.percentile(array, 75)),
                "90th": float(np.percentile(array, 90)),
                "95th": float(np.percentile(array, 95)),
            }

            # Apply precision rounding if configured
            if config.numpy_array_precision:
                for key, value in stats.items():
                    if isinstance(value, float):
                        stats[key] = round(value, config.numpy_array_precision)
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, float):
                                stats[key][sub_key] = round(
                                    sub_value, config.numpy_array_precision
                                )

        except Exception as e:
            stats["computation_error"] = str(e)

        return stats

    # Pandas serializers (if available)

    def _serialize_dataframe(
        self, df: Any, config: SerializationConfig
    ) -> Dict[str, Any]:
        """Enhanced Pandas DataFrame serialization with comprehensive features"""
        if not HAS_PANDAS:
            return {"error": "Pandas not available"}

        result = {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "__pandas_type__": "DataFrame",
        }

        # Index information
        if config.pandas_include_index:
            result["index_info"] = self._serialize_pandas_index(df.index, config)

        # Memory usage information
        if config.pandas_include_memory_usage:
            try:
                memory_usage = df.memory_usage(deep=True)
                result["memory_usage"] = {
                    "total_bytes": int(memory_usage.sum()),
                    "total_mb": round(float(memory_usage.sum() / (1024 * 1024)), 3),
                    "per_column": {col: int(mem) for col, mem in memory_usage.items()},
                }
            except Exception as e:
                result["memory_usage_error"] = str(e)

        # Data types information
        if config.pandas_include_dtypes:
            result["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

            # Categorize column types
            result["column_types"] = TypeRegistry._categorize_dataframe_columns(df)

        # Column sampling based on configuration
        cols_to_include = TypeRegistry._select_columns_for_serialization(df, config)
        if cols_to_include != list(df.columns):
            result["columns_sampled"] = True
            result["columns_included"] = cols_to_include
            result["columns_omitted"] = len(df.columns) - len(cols_to_include)
            df_for_serialization = df[cols_to_include]
        else:
            result["columns_sampled"] = False
            df_for_serialization = df

        # Data serialization based on size
        if (
            len(df) <= config.pandas_max_rows
            and len(cols_to_include) <= config.pandas_max_cols
        ):
            # Small DataFrames: include full data
            result["data"] = self._serialize_dataframe_data(
                df_for_serialization, config
            )
            result["serialization_method"] = "full"

        else:
            # Large DataFrames: use sampling strategy
            result.update(
                self._serialize_dataframe_sample(df_for_serialization, config)
            )
            result["serialization_method"] = "sampled"

        # Statistical summary for numeric columns
        if config.pandas_include_describe:
            try:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    describe_data = df[numeric_cols].describe()
                    result["statistics"] = {
                        col: {
                            stat: float(val) for stat, val in describe_data[col].items()
                        }
                        for col in numeric_cols
                    }
            except Exception as e:
                result["statistics_error"] = str(e)

        # Handle MultiIndex columns
        if config.pandas_handle_multiindex and isinstance(df.columns, pd.MultiIndex):
            result["multiindex_columns"] = {
                "levels": [list(level) for level in df.columns.levels],
                "names": list(df.columns.names),
                "nlevels": df.columns.nlevels,
            }

        return result

    def _serialize_pandas_index(
        self, index: Any, config: SerializationConfig
    ) -> Dict[str, Any]:
        """Serialize pandas Index with comprehensive information"""
        index_info = {
            "name": index.name,
            "dtype": str(index.dtype),
            "size": len(index),
            "is_unique": bool(index.is_unique),
            "is_monotonic": bool(index.is_monotonic_increasing),
        }

        # Handle different index types
        if isinstance(index, pd.DatetimeIndex):
            index_info["index_type"] = "DatetimeIndex"
            index_info["freq"] = str(index.freq) if index.freq else None
            index_info["tz"] = str(index.tz) if index.tz else None
            if len(index) > 0:
                index_info["range"] = {
                    "start": index.min().isoformat(),
                    "end": index.max().isoformat(),
                }
        elif isinstance(index, pd.MultiIndex):
            index_info["index_type"] = "MultiIndex"
            index_info["nlevels"] = index.nlevels
            index_info["names"] = list(index.names)
            index_info["levels"] = [list(level) for level in index.levels]
        elif isinstance(index, pd.CategoricalIndex):
            index_info["index_type"] = "CategoricalIndex"
            index_info["categories"] = list(index.categories)
            index_info["ordered"] = bool(index.ordered)
        else:
            index_info["index_type"] = "Index"

        # Sample of index values
        sample_size = min(5, len(index))
        if sample_size > 0:
            try:
                index_info["sample"] = index[:sample_size].tolist()
            except:
                index_info["sample"] = [str(x) for x in index[:sample_size]]

        return index_info

    @staticmethod
    def _categorize_dataframe_columns(df: Any) -> Dict[str, List[str]]:
        """Categorize DataFrame columns by type"""
        categories = {
            "numeric": [],
            "datetime": [],
            "categorical": [],
            "string": [],
            "boolean": [],
            "other": [],
        }

        for col in df.columns:
            dtype = df[col].dtype

            if pd.api.types.is_numeric_dtype(dtype):
                categories["numeric"].append(col)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                categories["datetime"].append(col)
            elif pd.api.types.is_categorical_dtype(dtype):
                categories["categorical"].append(col)
            elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(
                dtype
            ):
                categories["string"].append(col)
            elif pd.api.types.is_bool_dtype(dtype):
                categories["boolean"].append(col)
            else:
                categories["other"].append(col)

        return {
            k: v for k, v in categories.items() if v
        }  # Only include non-empty categories

    @staticmethod
    def _select_columns_for_serialization(
        df: Any, config: SerializationConfig
    ) -> List[str]:
        """Select which columns to include in serialization"""
        if len(df.columns) <= config.pandas_max_cols:
            return list(df.columns)

        # Priority-based column selection
        priority_cols = []
        remaining_cols = list(df.columns)

        # 1. Numeric columns (for statistics)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        priority_cols.extend(numeric_cols[: config.pandas_max_cols // 2])
        remaining_cols = [col for col in remaining_cols if col not in priority_cols]

        # 2. Datetime columns
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns
        priority_cols.extend(datetime_cols[:2])
        remaining_cols = [col for col in remaining_cols if col not in priority_cols]

        # 3. Fill remaining slots with other columns
        remaining_slots = config.pandas_max_cols - len(priority_cols)
        priority_cols.extend(remaining_cols[:remaining_slots])

        return priority_cols[: config.pandas_max_cols]

    def _serialize_dataframe_data(self, df: Any, config: SerializationConfig) -> Any:
        """Serialize DataFrame data with proper type handling"""
        try:
            # Handle datetime columns
            df_copy = df.copy()
            for col in df_copy.select_dtypes(include=["datetime64"]).columns:
                if config.pandas_datetime_format == "iso":
                    df_copy[col] = df_copy[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
                elif config.pandas_datetime_format == "timestamp":
                    df_copy[col] = (
                        df_copy[col].astype("int64") // 10**9
                    )  # Unix timestamp

            # Handle categorical columns
            for col in df_copy.select_dtypes(include=["category"]).columns:
                if config.pandas_categorical_as_codes:
                    df_copy[col] = df_copy[col].cat.codes
                else:
                    df_copy[col] = df_copy[col].astype(str)

            return df_copy.to_dict("records")

        except Exception as e:
            return {"serialization_error": str(e)}

    def _serialize_dataframe_sample(
        self, df: Any, config: SerializationConfig
    ) -> Dict[str, Any]:
        """Generate sample representation for large DataFrames"""
        result = {}

        if config.pandas_sample_method == "head_tail":
            # Traditional head/tail sampling
            head_rows = min(config.pandas_max_rows // 2, len(df))
            tail_rows = min(config.pandas_max_rows - head_rows, len(df))

            result["sample"] = {
                "head": self._serialize_dataframe_data(df.head(head_rows), config),
                "tail": self._serialize_dataframe_data(df.tail(tail_rows), config),
                "rows_omitted": len(df) - head_rows - tail_rows,
            }

        elif config.pandas_sample_method == "random":
            # Random sampling
            sample_size = min(config.pandas_max_rows, len(df))
            sampled_df = df.sample(n=sample_size, random_state=42)
            result["sample"] = {
                "random": self._serialize_dataframe_data(sampled_df, config),
                "rows_omitted": len(df) - sample_size,
                "sampling_method": "random",
            }

        elif config.pandas_sample_method == "systematic":
            # Systematic sampling
            sample_size = min(config.pandas_max_rows, len(df))
            step = max(1, len(df) // sample_size)
            sampled_df = df.iloc[::step][:sample_size]
            result["sample"] = {
                "systematic": self._serialize_dataframe_data(sampled_df, config),
                "rows_omitted": len(df) - len(sampled_df),
                "sampling_method": "systematic",
                "step_size": step,
            }

        return result

    def _serialize_series(
        self, series: Any, config: SerializationConfig
    ) -> Dict[str, Any]:
        """Enhanced Pandas Series serialization with comprehensive features"""
        if not HAS_PANDAS:
            return {"error": "Pandas not available"}

        result = {
            "name": series.name,
            "dtype": str(series.dtype),
            "size": len(series),
            "__pandas_type__": "Series",
        }

        # Memory usage information
        if config.pandas_include_memory_usage:
            try:
                memory_bytes = int(series.memory_usage(deep=True))
                result["memory_usage"] = {
                    "bytes": memory_bytes,
                    "mb": round(memory_bytes / (1024 * 1024), 3),
                }
            except Exception as e:
                result["memory_usage_error"] = str(e)

        # Index information
        if config.pandas_include_index:
            result["index_info"] = self._serialize_pandas_index(series.index, config)

        # Data type specific information
        result.update(TypeRegistry._analyze_series_type(series, config))

        # Data serialization based on size
        if len(series) <= config.pandas_max_rows:
            # Small series: include full data
            result["data"] = self._serialize_series_data(series, config)
            result["serialization_method"] = "full"
        else:
            # Large series: use sampling
            result.update(self._serialize_series_sample(series, config))
            result["serialization_method"] = "sampled"

        # Statistical summary for numeric series
        if config.pandas_include_describe and pd.api.types.is_numeric_dtype(series):
            try:
                describe_data = series.describe()
                result["statistics"] = {
                    stat: float(val) for stat, val in describe_data.items()
                }
            except Exception as e:
                result["statistics_error"] = str(e)

        return result

    @staticmethod
    def _analyze_series_type(
        series: Any, config: SerializationConfig
    ) -> Dict[str, Any]:
        """Analyze Series data type and provide specific information"""
        analysis = {}

        if pd.api.types.is_numeric_dtype(series):
            analysis["data_type"] = "numeric"
            if len(series) > 0:
                analysis["value_range"] = {
                    "min": float(series.min()),
                    "max": float(series.max()),
                }

        elif pd.api.types.is_datetime64_any_dtype(series):
            analysis["data_type"] = "datetime"
            if len(series) > 0:
                analysis["datetime_range"] = {
                    "start": series.min().isoformat(),
                    "end": series.max().isoformat(),
                }

        elif pd.api.types.is_categorical_dtype(series):
            analysis["data_type"] = "categorical"
            analysis["categories"] = list(series.cat.categories)
            analysis["ordered"] = bool(series.cat.ordered)
            analysis["num_categories"] = len(series.cat.categories)

        elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(
            series
        ):
            analysis["data_type"] = "string/object"
            if len(series) > 0:
                analysis["unique_values"] = int(series.nunique())
                analysis["null_count"] = int(series.isnull().sum())

        elif pd.api.types.is_bool_dtype(series):
            analysis["data_type"] = "boolean"
            if len(series) > 0:
                analysis["value_counts"] = {
                    "true": int(series.sum()),
                    "false": int((~series).sum()),
                    "null": int(series.isnull().sum()),
                }
        else:
            analysis["data_type"] = "other"

        return analysis

    def _serialize_series_data(self, series: Any, config: SerializationConfig) -> Any:
        """Serialize Series data with proper type handling"""
        try:
            # Handle different data types appropriately
            if pd.api.types.is_datetime64_any_dtype(series):
                if config.pandas_datetime_format == "iso":
                    return series.dt.strftime("%Y-%m-%dT%H:%M:%S").tolist()
                elif config.pandas_datetime_format == "timestamp":
                    return (series.astype("int64") // 10**9).tolist()
                else:
                    return series.tolist()

            elif pd.api.types.is_categorical_dtype(series):
                if config.pandas_categorical_as_codes:
                    return series.cat.codes.tolist()
                else:
                    return series.astype(str).tolist()
            else:
                return series.tolist()

        except Exception as e:
            return {"serialization_error": str(e)}

    def _serialize_series_sample(
        self, series: Any, config: SerializationConfig
    ) -> Dict[str, Any]:
        """Generate sample representation for large Series"""
        result = {}

        if config.pandas_sample_method == "head_tail":
            # Traditional head/tail sampling
            head_size = min(config.pandas_max_rows // 2, len(series))
            tail_size = min(config.pandas_max_rows - head_size, len(series))

            result["sample"] = {
                "head": self._serialize_series_data(series.head(head_size), config),
                "tail": self._serialize_series_data(series.tail(tail_size), config),
                "values_omitted": len(series) - head_size - tail_size,
            }

        elif config.pandas_sample_method == "random":
            # Random sampling
            sample_size = min(config.pandas_max_rows, len(series))
            sampled_series = series.sample(n=sample_size, random_state=42)
            result["sample"] = {
                "random": self._serialize_series_data(sampled_series, config),
                "values_omitted": len(series) - sample_size,
                "sampling_method": "random",
            }

        elif config.pandas_sample_method == "systematic":
            # Systematic sampling
            sample_size = min(config.pandas_max_rows, len(series))
            step = max(1, len(series) // sample_size)
            sampled_series = series.iloc[::step][:sample_size]
            result["sample"] = {
                "systematic": self._serialize_series_data(sampled_series, config),
                "values_omitted": len(series) - len(sampled_series),
                "sampling_method": "systematic",
                "step_size": step,
            }

        return result

    def _serialize_pandas_timestamp(self, ts: Any, config: SerializationConfig) -> Any:
        """Enhanced Pandas Timestamp serialization"""
        if not HAS_PANDAS:
            return str(ts)

        if config.pandas_datetime_format == "iso":
            result = ts.isoformat()
        elif config.pandas_datetime_format == "timestamp":
            result = int(ts.timestamp())
        else:
            result = str(ts)

        # Include timezone info if available
        if config.pandas_include_index and ts.tz is not None:
            return {
                "datetime": result,
                "timezone": str(ts.tz),
                "__pandas_type__": "Timestamp",
            }

        return result

    # Additional scientific data types

    def _serialize_numpy_matrix(
        self, matrix: Any, config: SerializationConfig
    ) -> Dict[str, Any]:
        """Serialize NumPy matrix objects"""
        if not HAS_NUMPY:
            return {"error": "NumPy not available"}

        # Convert matrix to array and use array serialization
        array_data = TypeRegistry._serialize_numpy_array(np.asarray(matrix), config)
        array_data["__numpy_type__"] = "matrix"
        array_data["matrix_type"] = type(matrix).__name__

        return array_data

    def _serialize_numpy_masked_array(
        self, masked_array: Any, config: SerializationConfig
    ) -> Dict[str, Any]:
        """Serialize NumPy masked arrays"""
        if not HAS_NUMPY:
            return {"error": "NumPy not available"}

        result = {
            "shape": list(masked_array.shape),
            "dtype": str(masked_array.dtype),
            "size": int(masked_array.size),
            "__numpy_type__": "MaskedArray",
        }

        # Mask information
        if hasattr(masked_array, "mask") and masked_array.mask is not False:
            result["mask_info"] = {
                "has_mask": True,
                "masked_count": (
                    int(np.sum(masked_array.mask)) if np.any(masked_array.mask) else 0
                ),
                "fill_value": str(masked_array.fill_value),
            }
        else:
            result["mask_info"] = {"has_mask": False}

        # Data serialization
        if masked_array.size <= config.numpy_array_max_size:
            try:
                # Convert to regular array for serialization
                data_array = np.asarray(masked_array)
                result["data"] = data_array.tolist()
                if result["mask_info"]["has_mask"]:
                    result["mask"] = masked_array.mask.tolist()
            except Exception as e:
                result["data_error"] = str(e)
        else:
            # Large masked array: use compressed representation
            result.update(
                TypeRegistry._get_numpy_array_compressed(
                    np.asarray(masked_array), config
                )
            )

        return result

    def _serialize_scipy_sparse_matrix(
        self, sparse_matrix: Any, config: SerializationConfig
    ) -> Dict[str, Any]:
        """Serialize SciPy sparse matrices"""
        if not HAS_SCIPY:
            return {"error": "SciPy not available"}

        if not isinstance(sparse_matrix, scipy.sparse.spmatrix):
            return {"error": "Not a SciPy sparse matrix"}

        result = {
            "shape": list(sparse_matrix.shape),
            "dtype": str(sparse_matrix.dtype),
            "format": sparse_matrix.format,
            "nnz": int(sparse_matrix.nnz),
            "density": (
                float(
                    sparse_matrix.nnz
                    / (sparse_matrix.shape[0] * sparse_matrix.shape[1])
                )
                if (sparse_matrix.shape[0] * sparse_matrix.shape[1]) > 0
                else 0.0
            ),
            "__scipy_type__": "sparse_matrix",
        }

        # Memory information
        result["memory_info"] = {
            "data_bytes": (
                sparse_matrix.data.nbytes if hasattr(sparse_matrix, "data") else 0
            ),
            "total_bytes": (
                sparse_matrix.data.nbytes
                + sparse_matrix.indices.nbytes
                + sparse_matrix.indptr.nbytes
                if hasattr(sparse_matrix, "indices")
                and hasattr(sparse_matrix, "indptr")
                else 0
            ),
        }

        # For very sparse matrices, include some sample data
        if sparse_matrix.nnz <= config.numpy_array_max_size and sparse_matrix.nnz > 0:
            try:
                # Convert to COO format to get coordinates and values
                coo = sparse_matrix.tocoo()
                result["sample_data"] = {
                    "coordinates": list(
                        zip(coo.row[:10].tolist(), coo.col[:10].tolist())
                    ),
                    "values": coo.data[:10].tolist(),
                }
            except Exception as e:
                result["sample_error"] = str(e)

        return result

    def _serialize_pandas_categorical(
        self, categorical: Any, config: SerializationConfig
    ) -> Dict[str, Any]:
        """Serialize Pandas Categorical data"""
        if not HAS_PANDAS:
            return {"error": "Pandas not available"}

        result = {
            "size": len(categorical),
            "categories": list(categorical.categories),
            "ordered": bool(categorical.ordered),
            "num_categories": len(categorical.categories),
            "__pandas_type__": "Categorical",
        }

        # Value counts
        try:
            value_counts = categorical.value_counts()
            result["value_counts"] = {
                str(cat): int(count) for cat, count in value_counts.items()
            }
        except Exception as e:
            result["value_counts_error"] = str(e)

        # Data serialization
        if len(categorical) <= config.pandas_max_rows:
            if config.pandas_categorical_as_codes:
                result["data"] = categorical.codes.tolist()
                result["data_type"] = "codes"
            else:
                result["data"] = categorical.astype(str).tolist()
                result["data_type"] = "values"
        else:
            # Large categorical: sample
            sample_size = min(config.pandas_max_rows, len(categorical))
            if config.pandas_categorical_as_codes:
                result["sample"] = categorical.codes[:sample_size].tolist()
                result["data_type"] = "codes"
            else:
                result["sample"] = categorical.astype(str)[:sample_size].tolist()
                result["data_type"] = "values"
            result["values_omitted"] = len(categorical) - sample_size

        return result


class EnhancedJSONEncoder(json.JSONEncoder):
    """JSON encoder with support for complex Python types"""

    def __init__(self, config: Optional[SerializationConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or SerializationConfig()
        self.type_registry = TypeRegistry()

    def default(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable formats"""

        # Handle lazy serializable objects
        if isinstance(obj, LazySerializable):
            return obj.force_serialize()

        # Handle lazy dictionaries
        if isinstance(obj, LazyDict):
            return obj.force_serialize_all()

        # Get custom serializer
        serializer = self.type_registry.get_serializer(obj)
        if serializer:
            try:
                return serializer(obj, self.config)
            except Exception as e:
                # Fallback to safe representation
                return {
                    "__serialization_error__": str(e),
                    "__type__": type(obj).__name__,
                    "__repr__": repr(obj)[:200],  # Truncate repr
                }

        # Handle collections with size limits
        if isinstance(obj, (list, tuple)):
            if len(obj) > self.config.max_collection_size:
                truncated = list(obj[: self.config.max_collection_size])
                truncated.append(
                    f"... ({len(obj) - self.config.max_collection_size} more items)"
                )
                return truncated
            return list(obj)

        # Handle strings with truncation
        if isinstance(obj, str) and self.config.truncate_strings:
            if len(obj) > self.config.truncate_strings:
                return obj[: self.config.truncate_strings] + "..."

        # Fallback to default behavior
        try:
            return super().default(obj)
        except TypeError:
            # Last resort: safe string representation
            try:
                obj_repr = repr(obj)[:200]
            except Exception as e:
                obj_repr = f"<repr failed: {str(e)[:50]}>"

            return {"__unserializable__": type(obj).__name__, "__repr__": obj_repr}


def enhanced_json_dumps(
    obj: Any, config: Optional[SerializationConfig] = None, **kwargs
) -> str:
    """
    JSON dumps with enhanced serialization support

    Args:
        obj: Object to serialize
        config: Serialization configuration
        **kwargs: Additional arguments passed to json.dumps

    Returns:
        JSON string representation
    """
    encoder = EnhancedJSONEncoder(config=config)
    return json.dumps(obj, cls=EnhancedJSONEncoder, **kwargs)


def register_custom_serializer(
    type_class: Type, serializer: Callable[[Any, SerializationConfig], Any]
) -> None:
    """
    Register a custom serializer for a type globally

    Args:
        type_class: The type to register serializer for
        serializer: Function that takes (obj, config) and returns serializable data
    """
    # Get global registry (we'll create a default one)
    global _global_registry
    if "_global_registry" not in globals():
        _global_registry = TypeRegistry()

    _global_registry.register(type_class, serializer)


# Default global configuration
DEFAULT_CONFIG = SerializationConfig()

# Global type registry
_global_registry = TypeRegistry()


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

    # Handle None
    if obj is None:
        return None

    # Check for custom serializers first (including NumPy types)
    # This must come before primitive checks since NumPy types inherit from Python primitives
    serializer = _global_registry.get_serializer(obj)
    if serializer:
        try:
            return serializer(obj, config)
        except Exception as e:
            # Safe fallback
            return {
                "__serialization_error__": str(e),
                "__type__": type(obj).__name__,
                "__repr__": repr(obj)[:200],
            }

    # Handle primitives, apply detection if enabled
    if isinstance(obj, (str, int, float, bool)):
        if config.auto_detect_types:
            # Apply detection to primitives
            detector = TypeDetector(config)
            detected = detector.detect_and_convert(obj)
            if detected != obj:
                # Auto-detection found something, enhance it
                smart_converter = SmartConverter(config)
                return smart_converter._enhance_detected_object(detected)

        # Handle string truncation for regular strings
        if isinstance(obj, str) and config.truncate_strings:
            if len(obj) > config.truncate_strings:
                return obj[: config.truncate_strings] + "..."
        return obj

    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        result = []
        for i, item in enumerate(obj):
            if i >= config.max_collection_size:
                result.append(f"... ({len(obj) - i} more items)")
                break
            result.append(serialize_for_logging(item, config))
        return result

    # Handle dictionaries
    if isinstance(obj, dict):
        result = {}
        count = 0
        for key, value in obj.items():
            if count >= config.max_collection_size:
                result["..."] = f"({len(obj) - count} more items)"
                break
            # Ensure key is string
            str_key = str(key) if not isinstance(key, str) else key
            result[str_key] = serialize_for_logging(value, config)
            count += 1
        return result

    # Final fallback with safe repr
    try:
        obj_repr = repr(obj)[:200]
    except Exception as e:
        obj_repr = f"<repr failed: {str(e)[:50]}>"

    return {"__unserializable__": type(obj).__name__, "__repr__": obj_repr}


# Schema validation and type annotation support


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

    def validate(self, data: Dict[str, Any], schema_name: str) -> bool:
        """
        Validate data against a registered schema

        Args:
            data: Data to validate
            schema_name: Name of schema to validate against

        Returns:
            True if validation passes

        Raises:
            ValidationError: If validation fails
        """
        start_time = time.perf_counter()

        try:
            self._validation_stats["validations_performed"] += 1

            if schema_name not in self._schemas:
                raise ValidationError(f"Schema '{schema_name}' not found")

            schema = self._schemas[schema_name]
            errors = []

            # Check required fields
            for field, constraints in schema.items():
                if constraints.get("required", True) and field not in data:
                    errors.append(f"Required field '{field}' is missing")
                    continue

                if field in data:
                    field_errors = self._validate_field(field, data[field], constraints)
                    errors.extend(field_errors)

            # Check for unexpected fields if strict mode
            if schema.get("strict", False):
                for field in data:
                    if field not in schema:
                        errors.append(
                            f"Unexpected field '{field}' not allowed in strict mode"
                        )

            if errors:
                self._validation_stats["validation_failures"] += 1
                raise ValidationError(f"Validation failed: {'; '.join(errors)}")

            return True

        finally:
            self._validation_stats["validation_time"] += (
                time.perf_counter() - start_time
            )

    def _validate_field(
        self, field_name: str, value: Any, constraints: Dict[str, Any]
    ) -> List[str]:
        """Validate a single field value against constraints"""
        errors = []

        # Type validation
        expected_type = constraints.get("type")
        if expected_type and expected_type != "any":
            if not self._check_type(value, expected_type):
                errors.append(
                    f"Field '{field_name}' must be of type {expected_type}, got {type(value).__name__}"
                )
                return errors  # Skip other validations if type is wrong

        # String constraints
        if isinstance(value, str):
            if "min_length" in constraints and len(value) < constraints["min_length"]:
                errors.append(
                    f"Field '{field_name}' must be at least {constraints['min_length']} characters"
                )

            if "max_length" in constraints and len(value) > constraints["max_length"]:
                errors.append(
                    f"Field '{field_name}' must be at most {constraints['max_length']} characters"
                )

            if "pattern" in constraints:
                if not re.match(constraints["pattern"], value):
                    errors.append(
                        f"Field '{field_name}' does not match required pattern"
                    )

        # Numeric constraints
        if isinstance(value, (int, float)):
            if "min_value" in constraints and value < constraints["min_value"]:
                errors.append(
                    f"Field '{field_name}' must be at least {constraints['min_value']}"
                )

            if "max_value" in constraints and value > constraints["max_value"]:
                errors.append(
                    f"Field '{field_name}' must be at most {constraints['max_value']}"
                )

        # Collection constraints
        if isinstance(value, (list, tuple, dict)):
            if "min_items" in constraints and len(value) < constraints["min_items"]:
                errors.append(
                    f"Field '{field_name}' must have at least {constraints['min_items']} items"
                )

            if "max_items" in constraints and len(value) > constraints["max_items"]:
                errors.append(
                    f"Field '{field_name}' must have at most {constraints['max_items']} items"
                )

        # Enum/choice validation
        if "choices" in constraints:
            if value not in constraints["choices"]:
                errors.append(
                    f"Field '{field_name}' must be one of {constraints['choices']}"
                )

        # Custom validator function
        if "validator" in constraints:
            validator_func = constraints["validator"]
            try:
                if not validator_func(value):
                    errors.append(f"Field '{field_name}' failed custom validation")
            except Exception as e:
                errors.append(f"Field '{field_name}' validation error: {str(e)}")

        return errors

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type"""
        type_mapping = {
            "str": str,
            "int": int,
            "float": (int, float),  # int is also acceptable for float
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "datetime": datetime,
            "uuid": UUID,
            "decimal": Decimal,
            "path": (str, Path, PurePath),
            "any": object,
        }

        expected_types = type_mapping.get(expected_type, str)
        return isinstance(value, expected_types)

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return self._validation_stats.copy()

    def reset_validation_stats(self) -> None:
        """Reset validation statistics"""
        self._validation_stats = {
            "validations_performed": 0,
            "validation_failures": 0,
            "validation_time": 0.0,
        }


class TypeAnnotationExtractor:
    """
    Extract and validate type annotations from functions and classes

    Provides introspection capabilities for automatic schema generation
    from Python type hints.
    """

    def __init__(self):
        self._annotation_cache = {}
        self._cache_enabled = True

    def extract_function_schema(self, func: Callable) -> Dict[str, Any]:
        """
        Extract schema from function type annotations

        Args:
            func: Function to analyze

        Returns:
            Schema dictionary based on function parameters and return type
        """
        if self._cache_enabled and func in self._annotation_cache:
            return self._annotation_cache[func]

        try:
            type_hints = get_type_hints(func)
            sig = inspect.signature(func)

            schema = {
                "function_name": func.__name__,
                "parameters": {},
                "return_type": None,
            }

            # Extract parameter types
            for param_name, param in sig.parameters.items():
                if param_name in type_hints:
                    param_schema = self._annotation_to_schema(type_hints[param_name])
                    param_schema["required"] = param.default == inspect.Parameter.empty
                    schema["parameters"][param_name] = param_schema

            # Extract return type
            if "return" in type_hints:
                schema["return_type"] = self._annotation_to_schema(type_hints["return"])

            if self._cache_enabled:
                self._annotation_cache[func] = schema

            return schema

        except Exception as e:
            return {"error": f"Failed to extract schema: {str(e)}"}

    def extract_class_schema(self, cls: Type) -> Dict[str, Any]:
        """
        Extract schema from class type annotations

        Args:
            cls: Class to analyze

        Returns:
            Schema dictionary based on class attributes and methods
        """
        try:
            type_hints = get_type_hints(cls)

            schema = {"class_name": cls.__name__, "attributes": {}, "methods": {}}

            # Extract attribute types
            for attr_name, attr_type in type_hints.items():
                schema["attributes"][attr_name] = self._annotation_to_schema(attr_type)

            # Extract method schemas
            for method_name in dir(cls):
                if not method_name.startswith("_"):
                    method = getattr(cls, method_name)
                    if callable(method):
                        schema["methods"][method_name] = self.extract_function_schema(
                            method
                        )

            return schema

        except Exception as e:
            return {"error": f"Failed to extract class schema: {str(e)}"}

    def _annotation_to_schema(self, annotation: Type) -> Dict[str, Any]:
        """Convert type annotation to schema constraint"""
        # Handle basic types
        if annotation in (str, int, float, bool, list, dict, tuple, set):
            return {"type": annotation.__name__}

        # Handle special types
        if annotation == datetime:
            return {"type": "datetime"}
        elif annotation == UUID:
            return {"type": "uuid"}
        elif annotation == Decimal:
            return {"type": "decimal"}
        elif annotation in (Path, PurePath):
            return {"type": "path"}

        # Handle generic types (List[str], Dict[str, int], etc.)
        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin is list:
            schema = {"type": "list"}
            if args:
                schema["item_type"] = self._annotation_to_schema(args[0])
            return schema

        elif origin is dict:
            schema = {"type": "dict"}
            if len(args) >= 2:
                schema["key_type"] = self._annotation_to_schema(args[0])
                schema["value_type"] = self._annotation_to_schema(args[1])
            return schema

        elif origin is tuple:
            schema = {"type": "tuple"}
            if args:
                schema["item_types"] = [self._annotation_to_schema(arg) for arg in args]
            return schema

        elif origin is Union:
            # Handle Optional[T] and Union types
            if len(args) == 2 and type(None) in args:
                # Optional type
                non_none_type = args[0] if args[1] is type(None) else args[1]
                schema = self._annotation_to_schema(non_none_type)
                schema["required"] = False
                return schema
            else:
                # True union
                return {
                    "type": "union",
                    "types": [self._annotation_to_schema(arg) for arg in args],
                }

        # Fallback for unknown types
        return {
            "type": "custom",
            "type_name": getattr(annotation, "__name__", str(annotation)),
        }


class StructuredDataValidator:
    """
    High-level validator that combines schema validation with type annotations

    Provides a unified interface for validating structured logging data
    against both explicit schemas and inferred type annotations.
    """

    def __init__(self, config: Optional[SerializationConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.schema_validator = SchemaValidator(config)
        self.type_extractor = TypeAnnotationExtractor()
        self._auto_schemas = {}

    def register_schema(self, name: str, schema: Dict[str, Any]) -> None:
        """Register an explicit validation schema"""
        self.schema_validator.register_schema(name, schema)

    def register_function_schema(
        self, func: Callable, schema_name: Optional[str] = None
    ) -> str:
        """
        Register a schema automatically generated from function type annotations

        Args:
            func: Function to analyze
            schema_name: Optional custom name, defaults to function name

        Returns:
            Name of the registered schema
        """
        schema_name = schema_name or f"func_{func.__name__}"
        extracted = self.type_extractor.extract_function_schema(func)

        # Convert function schema to validation schema
        validation_schema = {}
        for param_name, param_info in extracted.get("parameters", {}).items():
            validation_schema[param_name] = param_info

        self.schema_validator.register_schema(schema_name, validation_schema)
        self._auto_schemas[schema_name] = extracted

        return schema_name

    def register_class_schema(
        self, cls: Type, schema_name: Optional[str] = None
    ) -> str:
        """
        Register a schema automatically generated from class type annotations

        Args:
            cls: Class to analyze
            schema_name: Optional custom name, defaults to class name

        Returns:
            Name of the registered schema
        """
        schema_name = schema_name or f"class_{cls.__name__}"
        extracted = self.type_extractor.extract_class_schema(cls)

        # Convert class schema to validation schema
        validation_schema = {}
        for attr_name, attr_info in extracted.get("attributes", {}).items():
            validation_schema[attr_name] = attr_info

        self.schema_validator.register_schema(schema_name, validation_schema)
        self._auto_schemas[schema_name] = extracted

        return schema_name

    def validate_against_function(self, func: Callable, data: Dict[str, Any]) -> bool:
        """
        Validate data against function signature

        Args:
            func: Function whose signature to validate against
            data: Data to validate

        Returns:
            True if validation passes
        """
        schema_name = self.register_function_schema(func)
        return self.schema_validator.validate(data, schema_name)

    def validate_against_class(self, cls: Type, data: Dict[str, Any]) -> bool:
        """
        Validate data against class attributes

        Args:
            cls: Class whose attributes to validate against
            data: Data to validate

        Returns:
            True if validation passes
        """
        schema_name = self.register_class_schema(cls)
        return self.schema_validator.validate(data, schema_name)

    def validate(self, data: Dict[str, Any], schema_name: str) -> bool:
        """Validate data against named schema"""
        return self.schema_validator.validate(data, schema_name)

    def get_schema_info(self, schema_name: str) -> Dict[str, Any]:
        """Get information about a registered schema"""
        if schema_name in self._auto_schemas:
            return self._auto_schemas[schema_name]
        elif schema_name in self.schema_validator._schemas:
            return {
                "schema_name": schema_name,
                "type": "explicit",
                "constraints": self.schema_validator._schemas[schema_name],
            }
        else:
            return {"error": f"Schema '{schema_name}' not found"}

    def list_schemas(self) -> List[str]:
        """List all registered schema names"""
        explicit = list(self.schema_validator._schemas.keys())
        auto = list(self._auto_schemas.keys())
        return sorted(set(explicit + auto))

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics"""
        return self.schema_validator.get_validation_stats()


# Global validator instance
_global_validator = StructuredDataValidator()


def register_validation_schema(name: str, schema: Dict[str, Any]) -> None:
    """
    Register a validation schema globally

    Args:
        name: Schema name
        schema: Schema definition
    """
    _global_validator.register_schema(name, schema)


def validate_log_data(data: Dict[str, Any], schema_name: str) -> bool:
    """
    Validate log data against a registered schema

    Args:
        data: Data to validate
        schema_name: Name of schema to validate against

    Returns:
        True if validation passes

    Raises:
        ValidationError: If validation fails
    """
    return _global_validator.validate(data, schema_name)


def auto_validate_function(func: Callable) -> Callable:
    """
    Decorator to automatically validate function arguments

    Args:
        func: Function to decorate

    Returns:
        Decorated function that validates arguments
    """
    schema_name = _global_validator.register_function_schema(func)

    def wrapper(*args, **kwargs):
        # Convert args to kwargs for validation
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # Validate arguments
        _global_validator.validate(dict(bound.arguments), schema_name)

        return func(*args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


def get_validation_stats() -> Dict[str, Any]:
    """Get global validation statistics"""
    return _global_validator.get_validation_stats()


def reset_validation_stats() -> None:
    """Reset global validation statistics"""
    _global_validator.schema_validator.reset_validation_stats()
