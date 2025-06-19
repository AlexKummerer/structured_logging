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
from typing import Any, Callable, Dict, Optional, Set, Type, Union
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
    
    # Handle primitives (already JSON serializable)
    if isinstance(obj, (str, int, float, bool)):
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