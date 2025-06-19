"""
Enhanced serialization support for complex Python data types

This module provides custom serializers that extend JSON functionality
to handle Python objects that aren't natively JSON serializable.
"""

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path, PurePath
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
from uuid import UUID
import re
import inspect

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
        numpy_array_max_size: int = 100,
        pandas_max_rows: int = 5,
        pandas_include_dtypes: bool = False,
        auto_detect_types: bool = True,
        auto_convert_strings: bool = True,
        auto_convert_numbers: bool = True,
        detect_datetime_strings: bool = True,
        detect_uuid_strings: bool = True,
        detect_json_strings: bool = True,
        detect_url_strings: bool = True,
        type_detection_cache_size: int = 1000,
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
        self.numpy_array_max_size = numpy_array_max_size
        self.pandas_max_rows = pandas_max_rows
        self.pandas_include_dtypes = pandas_include_dtypes
        self.auto_detect_types = auto_detect_types
        self.auto_convert_strings = auto_convert_strings
        self.auto_convert_numbers = auto_convert_numbers
        self.detect_datetime_strings = detect_datetime_strings
        self.detect_uuid_strings = detect_uuid_strings
        self.detect_json_strings = detect_json_strings
        self.detect_url_strings = detect_url_strings
        self.type_detection_cache_size = type_detection_cache_size


class TypeDetector:
    """Automatic type detection and conversion system"""
    
    def __init__(self, config: SerializationConfig):
        self.config = config
        self._cache = {}  # Simple cache for detected types
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Pre-compiled regex patterns for efficiency
        self._uuid_pattern = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            re.IGNORECASE
        )
        
        self._datetime_patterns = [
            # ISO 8601 formats
            re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?$'),
            # Common date formats
            re.compile(r'^\d{4}-\d{2}-\d{2}$'),
            re.compile(r'^\d{2}/\d{2}/\d{4}$'),
            re.compile(r'^\d{2}-\d{2}-\d{4}$'),
            # Common datetime formats
            re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$'),
            re.compile(r'^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}$'),
        ]
        
        self._url_pattern = re.compile(
            r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$',
            re.IGNORECASE
        )
        
        self._number_patterns = {
            'integer': re.compile(r'^[+-]?\d+$'),
            'float': re.compile(r'^[+-]?\d*\.\d+$'),
            'scientific': re.compile(r'^[+-]?\d*\.?\d+[eE][+-]?\d+$'),
            'decimal': re.compile(r'^[+-]?\d+\.\d{2,}$'),  # Likely currency/precision
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
                    return lambda x: {"type": "datetime_string", "value": x, "detected_format": True}
        
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
                "human_readable": datetime.fromtimestamp(x).isoformat() + "Z"
            }
        
        # Large numbers with unit detection
        if isinstance(value, (int, float)) and abs(value) >= 1000:
            return lambda x: {
                "type": "large_number",
                "value": x,
                "formatted": self._format_large_number(x)
            }
        
        return None
    
    def _looks_like_json(self, value: str) -> bool:
        """Quick check if string looks like JSON"""
        if len(value) < 2:
            return False
        
        stripped = value.strip()
        return ((stripped.startswith('{') and stripped.endswith('}')) or
                (stripped.startswith('[') and stripped.endswith(']')))
    
    def _json_converter(self, value: str) -> Dict[str, Any]:
        """Convert JSON string to structured representation"""
        try:
            parsed = json.loads(value)
            return {
                "type": "json_string",
                "parsed": parsed,
                "original_length": len(value)
            }
        except json.JSONDecodeError:
            return {
                "type": "json_like_string",
                "value": value[:100] + "..." if len(value) > 100 else value
            }
    
    def _get_numeric_converter(self, num_type: str) -> Callable:
        """Get converter for numeric strings"""
        def converter(value: str) -> Dict[str, Any]:
            try:
                if num_type == 'integer':
                    return {"type": "numeric_string", "value": int(value), "original": value}
                elif num_type == 'float':
                    return {"type": "numeric_string", "value": float(value), "original": value}
                elif num_type == 'decimal':
                    return {"type": "decimal_string", "value": str(Decimal(value)), "original": value}
                elif num_type == 'scientific':
                    return {"type": "scientific_string", "value": float(value), "original": value}
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
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests
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
            auto_detect_types=False  # Disable auto-detection to prevent recursion
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
            auto_detect_types=False  # Disable auto-detection to prevent recursion
        )
        return serialize_for_logging(obj, config_without_detection)
    
    def _enhance_detected_object(self, obj: dict) -> dict:
        """Enhance detected objects with additional metadata"""
        result = obj.copy()
        
        # Add confidence and detection metadata
        result["_detection"] = {
            "detected_at": datetime.now().isoformat() + "Z",
            "confidence": self._calculate_confidence(obj),
            "auto_detected": True
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
            "numeric_like_string": 0.25
        }
        
        return confidence_map.get(obj_type, 0.50)
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get type detection statistics"""
        return self.detector.get_cache_stats()


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
            self.register(np.integer, self._serialize_numpy_scalar)
            self.register(np.floating, self._serialize_numpy_scalar)
            self.register(np.complexfloating, self._serialize_numpy_scalar)
            self.register(np.ndarray, self._serialize_numpy_array)
            self.register(np.bool_, self._serialize_numpy_scalar)
            
        if HAS_PANDAS:
            self.register(pd.DataFrame, self._serialize_dataframe)
            self.register(pd.Series, self._serialize_series)
            self.register(pd.Timestamp, self._serialize_pandas_timestamp)
    
    def register(self, type_class: Type, serializer: Callable[[Any, SerializationConfig], Any]) -> None:
        """Register a custom serializer for a type"""
        self._serializers[type_class] = serializer
    
    def get_serializer(self, obj: Any) -> Optional[Callable[[Any, SerializationConfig], Any]]:
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
    def _serialize_timedelta(td: timedelta, config: SerializationConfig) -> Dict[str, float]:
        """Serialize timedelta objects"""
        return {
            "days": td.days,
            "seconds": td.seconds,
            "microseconds": td.microseconds,
            "total_seconds": td.total_seconds()
        }
    
    @staticmethod
    def _serialize_decimal(decimal_val: Decimal, config: SerializationConfig) -> Union[float, str]:
        """Serialize Decimal objects"""
        if config.decimal_as_float:
            return float(decimal_val)
        else:
            return str(decimal_val)
    
    @staticmethod
    def _serialize_complex(complex_val: complex, config: SerializationConfig) -> Dict[str, float]:
        """Serialize complex numbers"""
        return {
            "real": complex_val.real,
            "imag": complex_val.imag
        }
    
    @staticmethod
    def _serialize_uuid(uuid_val: UUID, config: SerializationConfig) -> str:
        """Serialize UUID objects"""
        return str(uuid_val)
    
    @staticmethod
    def _serialize_path(path_obj: Union[Path, PurePath], config: SerializationConfig) -> Union[str, Dict[str, Any]]:
        """Serialize Path objects"""
        if config.path_as_string:
            return str(path_obj)
        else:
            return {
                "path": str(path_obj),
                "is_absolute": path_obj.is_absolute(),
                "parts": path_obj.parts,
                "suffix": path_obj.suffix
            }
    
    @staticmethod
    def _serialize_set(set_obj: Union[set, frozenset], config: SerializationConfig) -> list:
        """Serialize set objects to lists"""
        items = list(set_obj)
        if len(items) > config.max_collection_size:
            items = items[:config.max_collection_size]
            items.append(f"... ({len(set_obj) - config.max_collection_size} more items)")
        return items
    
    @staticmethod
    def _serialize_enum(enum_obj: Enum, config: SerializationConfig) -> Union[Any, Dict[str, Any]]:
        """Serialize Enum objects"""
        if config.enum_as_value:
            return enum_obj.value
        else:
            return {
                "name": enum_obj.name,
                "value": enum_obj.value,
                "type": enum_obj.__class__.__name__
            }
    
    @staticmethod
    def _serialize_bytes(bytes_obj: Union[bytes, bytearray], config: SerializationConfig) -> Dict[str, Any]:
        """Serialize bytes objects"""
        try:
            # Try to decode as UTF-8 for text data
            decoded = bytes_obj.decode('utf-8')
            if config.truncate_strings and len(decoded) > config.truncate_strings:
                decoded = decoded[:config.truncate_strings] + "..."
            return {
                "type": "text",
                "data": decoded,
                "size": len(bytes_obj)
            }
        except UnicodeDecodeError:
            # Binary data - show hex representation (truncated)
            hex_data = bytes_obj.hex()
            if len(hex_data) > 100:
                hex_data = hex_data[:100] + "..."
            return {
                "type": "binary",
                "hex": hex_data,
                "size": len(bytes_obj)
            }
    
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
                "__fields__": list(obj.__dataclass_fields__.keys())
            }
    
    # NumPy serializers (if available)
    
    @staticmethod
    def _serialize_numpy_scalar(scalar: Any, config: SerializationConfig) -> Any:
        """Serialize NumPy scalar types"""
        if not HAS_NUMPY:
            return str(scalar)
        
        # Convert to native Python types
        if isinstance(scalar, np.integer):
            return int(scalar)
        elif isinstance(scalar, np.floating):
            return float(scalar)
        elif isinstance(scalar, np.complexfloating):
            return {
                "real": float(scalar.real),
                "imag": float(scalar.imag)
            }
        elif isinstance(scalar, np.bool_):
            return bool(scalar)
        else:
            return scalar.item()
    
    @staticmethod
    def _serialize_numpy_array(array: Any, config: SerializationConfig) -> Dict[str, Any]:
        """Serialize NumPy arrays"""
        if not HAS_NUMPY:
            return {"error": "NumPy not available"}
        
        result = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "size": int(array.size)
        }
        
        # Include actual data for small arrays
        if array.size <= config.numpy_array_max_size:
            result["data"] = array.tolist()
        else:
            # For large arrays, provide summary statistics
            result["stats"] = {
                "min": float(np.min(array)),
                "max": float(np.max(array)),
                "mean": float(np.mean(array)),
                "std": float(np.std(array))
            }
            # Show first few elements
            flat = array.flatten()
            sample_size = min(10, len(flat))
            result["sample"] = flat[:sample_size].tolist()
        
        return result
    
    # Pandas serializers (if available)
    
    @staticmethod
    def _serialize_dataframe(df: Any, config: SerializationConfig) -> Dict[str, Any]:
        """Serialize Pandas DataFrame"""
        if not HAS_PANDAS:
            return {"error": "Pandas not available"}
        
        result = {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "index_name": df.index.name,
            "memory_usage": int(df.memory_usage(deep=True).sum())
        }
        
        if config.pandas_include_dtypes:
            result["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Include sample data for small DataFrames
        if len(df) <= config.pandas_max_rows:
            result["data"] = df.to_dict('records')
        else:
            # For large DataFrames, show head and tail
            head = df.head(config.pandas_max_rows // 2).to_dict('records')
            tail = df.tail(config.pandas_max_rows // 2).to_dict('records')
            result["sample"] = {
                "head": head,
                "tail": tail
            }
            
            # Basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                result["stats"] = df[numeric_cols].describe().to_dict()
        
        return result
    
    @staticmethod
    def _serialize_series(series: Any, config: SerializationConfig) -> Dict[str, Any]:
        """Serialize Pandas Series"""
        if not HAS_PANDAS:
            return {"error": "Pandas not available"}
        
        result = {
            "name": series.name,
            "dtype": str(series.dtype),
            "size": len(series),
            "memory_usage": int(series.memory_usage(deep=True))
        }
        
        # Include actual data for small series
        if len(series) <= config.pandas_max_rows:
            result["data"] = series.tolist()
        else:
            # For large series, provide statistics
            if pd.api.types.is_numeric_dtype(series):
                result["stats"] = {
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "mean": float(series.mean()),
                    "std": float(series.std())
                }
            
            # Show sample values
            sample_size = min(config.pandas_max_rows, len(series))
            result["sample"] = series.head(sample_size).tolist()
        
        return result
    
    @staticmethod
    def _serialize_pandas_timestamp(ts: Any, config: SerializationConfig) -> str:
        """Serialize Pandas Timestamp"""
        if not HAS_PANDAS:
            return str(ts)
        
        return ts.isoformat()


class EnhancedJSONEncoder(json.JSONEncoder):
    """JSON encoder with support for complex Python types"""
    
    def __init__(self, config: Optional[SerializationConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or SerializationConfig()
        self.type_registry = TypeRegistry()
    
    def default(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable formats"""
        
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
                    "__repr__": repr(obj)[:200]  # Truncate repr
                }
        
        # Handle collections with size limits
        if isinstance(obj, (list, tuple)):
            if len(obj) > self.config.max_collection_size:
                truncated = list(obj[:self.config.max_collection_size])
                truncated.append(f"... ({len(obj) - self.config.max_collection_size} more items)")
                return truncated
            return list(obj)
        
        # Handle strings with truncation
        if isinstance(obj, str) and self.config.truncate_strings:
            if len(obj) > self.config.truncate_strings:
                return obj[:self.config.truncate_strings] + "..."
        
        # Fallback to default behavior
        try:
            return super().default(obj)
        except TypeError:
            # Last resort: safe string representation
            return {
                "__unserializable__": type(obj).__name__,
                "__repr__": repr(obj)[:200]
            }


def enhanced_json_dumps(obj: Any, config: Optional[SerializationConfig] = None, **kwargs) -> str:
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


def register_custom_serializer(type_class: Type, serializer: Callable[[Any, SerializationConfig], Any]) -> None:
    """
    Register a custom serializer for a type globally
    
    Args:
        type_class: The type to register serializer for
        serializer: Function that takes (obj, config) and returns serializable data
    """
    # Get global registry (we'll create a default one)
    global _global_registry
    if '_global_registry' not in globals():
        _global_registry = TypeRegistry()
    
    _global_registry.register(type_class, serializer)


# Default global configuration
DEFAULT_CONFIG = SerializationConfig()

# Global type registry
_global_registry = TypeRegistry()


def serialize_for_logging(obj: Any, config: Optional[SerializationConfig] = None) -> Any:
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
    
    # Handle primitives first, apply detection if enabled
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
                return obj[:config.truncate_strings] + "..."
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
    
    # Use custom serializer
    serializer = _global_registry.get_serializer(obj)
    if serializer:
        try:
            return serializer(obj, config)
        except Exception as e:
            # Safe fallback
            return {
                "__serialization_error__": str(e),
                "__type__": type(obj).__name__,
                "__repr__": repr(obj)[:200]
            }
    
    # Final fallback
    return {
        "__unserializable__": type(obj).__name__,
        "__repr__": repr(obj)[:200]
    }