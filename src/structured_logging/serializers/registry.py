"""
Type registry for custom serializers
"""

from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path, PurePath
from typing import Any, Callable, Dict, Optional, Type, Union
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

from .config import SerializationConfig


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
            self.register(np.integer, _serialize_numpy_scalar)
            self.register(np.floating, _serialize_numpy_scalar)
            self.register(np.complexfloating, _serialize_numpy_scalar)
            self.register(np.bool_, _serialize_numpy_scalar)
            self.register(np.str_, _serialize_numpy_scalar)

            # NumPy array types
            self.register(np.ndarray, _serialize_numpy_array)
            self.register(np.matrix, _serialize_numpy_matrix)

            # NumPy special arrays
            if hasattr(np, "ma"):
                self.register(np.ma.MaskedArray, _serialize_numpy_masked_array)

        if HAS_PANDAS:
            # Pandas core types
            self.register(pd.DataFrame, _serialize_dataframe)
            self.register(pd.Series, _serialize_series)
            self.register(pd.Timestamp, _serialize_pandas_timestamp)

            # Pandas data types
            self.register(pd.Categorical, _serialize_pandas_categorical)

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
                        _serialize_scipy_sparse_matrix,
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


# Import serializers from other modules to avoid circular imports
def _serialize_numpy_scalar(scalar: Any, config: SerializationConfig) -> Any:
    from .numpy_serializer import serialize_numpy_scalar

    return serialize_numpy_scalar(scalar)


def _serialize_numpy_array(array: Any, config: SerializationConfig) -> Dict[str, Any]:
    from .numpy_serializer import serialize_numpy_array

    return serialize_numpy_array(array, config)


def _serialize_numpy_matrix(matrix: Any, config: SerializationConfig) -> Dict[str, Any]:
    """Serialize NumPy matrix objects"""
    if not HAS_NUMPY:
        return {"error": "NumPy not available"}

    # Convert matrix to array and use array serialization
    array_data = _serialize_numpy_array(np.asarray(matrix), config)
    array_data["__numpy_type__"] = "matrix"
    array_data["matrix_type"] = type(matrix).__name__

    return array_data


def _serialize_numpy_masked_array(
    masked_array: Any, config: SerializationConfig
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
        from .numpy_serializer import _get_numpy_array_compressed

        result.update(_get_numpy_array_compressed(np.asarray(masked_array), config))

    return result


def _serialize_dataframe(df: Any, config: SerializationConfig) -> Dict[str, Any]:
    from .pandas_serializer import serialize_dataframe

    return serialize_dataframe(df, config)


def _serialize_series(series: Any, config: SerializationConfig) -> Dict[str, Any]:
    from .pandas_serializer import serialize_series

    return serialize_series(series, config)


def _serialize_pandas_timestamp(ts: Any, config: SerializationConfig) -> Any:
    from .pandas_serializer import serialize_pandas_timestamp

    return serialize_pandas_timestamp(ts, config)


def _serialize_pandas_categorical(
    categorical: Any, config: SerializationConfig
) -> Dict[str, Any]:
    from .pandas_serializer import serialize_pandas_categorical

    return serialize_pandas_categorical(categorical, config)


def _serialize_scipy_sparse_matrix(
    sparse_matrix: Any, config: SerializationConfig
) -> Dict[str, Any]:
    from .scipy_serializer import serialize_scipy_sparse_matrix

    return serialize_scipy_sparse_matrix(sparse_matrix, config)


# Global registry instance
_global_registry = TypeRegistry()


def register_custom_serializer(
    type_class: Type, serializer: Callable[[Any, SerializationConfig], Any]
) -> None:
    """
    Register a custom serializer for a specific type

    Args:
        type_class: The type to register
        serializer: Function that takes object and config, returns serialized form
    """
    _global_registry.register(type_class, serializer)