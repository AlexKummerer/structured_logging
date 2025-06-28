"""
NumPy array serialization module
"""

from typing import Any, Dict

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

from .config import SerializationConfig


def _add_array_metadata(array: Any, config: SerializationConfig, result: Dict[str, Any]) -> None:
    """Add array metadata to result dictionary"""
    if config.numpy_include_metadata:
        result.update({
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "size": int(array.size),
            "ndim": int(array.ndim),
            "itemsize": int(array.itemsize),
            "nbytes": int(array.nbytes),
            "__numpy_type__": "ndarray",
        })
        result["flags"] = {
            "c_contiguous": bool(array.flags.c_contiguous),
            "f_contiguous": bool(array.flags.f_contiguous),
            "writeable": bool(array.flags.writeable),
            "aligned": bool(array.flags.aligned),
        }
    else:
        result.update({
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "size": int(array.size),
        })


def _add_special_values_info(array: Any, config: SerializationConfig, result: Dict[str, Any]) -> None:
    """Add information about special values (NaN/Inf) to result"""
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


def _serialize_array_data(array: Any, config: SerializationConfig, result: Dict[str, Any]) -> None:
    """Serialize array data based on size strategy"""
    if array.size <= config.numpy_array_max_size:
        # Small arrays: include full data
        try:
            if config.numpy_array_precision and np.issubdtype(array.dtype, np.floating):
                rounded_array = np.round(array, config.numpy_array_precision)
                result["data"] = rounded_array.tolist()
            else:
                result["data"] = array.tolist()
            result["serialization_method"] = "full"
        except (ValueError, TypeError):
            result["data_error"] = "Cannot convert array to list"
            result["repr_sample"] = str(array.flat[:5])
    elif array.size <= config.numpy_compression_threshold:
        result.update(_get_numpy_array_summary(array, config))
        result["serialization_method"] = "summary"
    else:
        result.update(_get_numpy_array_compressed(array, config))
        result["serialization_method"] = "compressed"


def _add_array_statistics(array: Any, config: SerializationConfig, result: Dict[str, Any]) -> None:
    """Add statistical information to result"""
    if config.numpy_stats_for_numeric and np.issubdtype(array.dtype, np.number) and array.size > 0:
        try:
            stats = _compute_numpy_stats(array, config)
            result["statistics"] = stats
        except Exception as e:
            result["statistics_error"] = f"Could not compute statistics: {str(e)}"


def _add_sparse_info(array: Any, config: SerializationConfig, result: Dict[str, Any]) -> None:
    """Add sparse array information to result"""
    if config.numpy_preserve_sparse:
        try:
            if hasattr(array, "format") and hasattr(array, "nnz"):
                result["sparse_info"] = {
                    "format": array.format,
                    "nnz": int(array.nnz),
                    "density": float(array.nnz / array.size) if array.size > 0 else 0.0,
                    "__is_sparse__": True,
                }
        except:
            pass


def serialize_numpy_array(array: Any, config: SerializationConfig) -> Dict[str, Any]:
    """Enhanced NumPy array serialization with comprehensive features"""
    if not HAS_NUMPY:
        return {"error": "NumPy not available"}

    result = {}
    
    _add_array_metadata(array, config, result)
    _add_special_values_info(array, config, result)
    _serialize_array_data(array, config, result)
    _add_array_statistics(array, config, result)
    _add_sparse_info(array, config, result)
    
    return result


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


def serialize_numpy_scalar(scalar: Any, config: SerializationConfig) -> Any:
    """Convert NumPy scalar to Python type"""
    if not HAS_NUMPY:
        return str(scalar)

    # Determine the type category
    if np.issubdtype(scalar.dtype, np.integer):
        value = int(scalar)
        type_name = "integer"
    elif np.issubdtype(scalar.dtype, np.floating):
        if np.isnan(scalar):
            value = "__numpy_nan__"
        elif np.isinf(scalar):
            value = "__numpy_inf__" if scalar > 0 else "__numpy_-inf__"
        else:
            # Apply precision if configured
            if config.numpy_array_precision is not None:
                value = round(float(scalar), config.numpy_array_precision)
            else:
                value = float(scalar)
        type_name = "floating"
    elif np.issubdtype(scalar.dtype, np.complexfloating):
        value = {"real": float(scalar.real), "imag": float(scalar.imag)}
        type_name = "complex"
    elif np.issubdtype(scalar.dtype, np.bool_):
        value = bool(scalar)
        type_name = "boolean"
    elif np.issubdtype(scalar.dtype, np.datetime64):
        # Convert to ISO format string
        value = np.datetime_as_string(scalar, unit="auto")
        type_name = "datetime64"
    elif np.issubdtype(scalar.dtype, np.timedelta64):
        # Convert to seconds
        value = float(scalar / np.timedelta64(1, "s"))
        type_name = "timedelta64"
    elif np.issubdtype(scalar.dtype, np.str_):
        value = str(scalar)
        type_name = "string"
    else:
        # For other types, try direct conversion or fall back to string
        try:
            value = scalar.item()
            type_name = "unknown"
        except:
            return {
                "__numpy_scalar_error__": "Cannot convert to Python type",
                "__numpy_dtype__": str(scalar.dtype),
                "__repr__": str(scalar),
            }

    # Return with metadata if requested
    if config.numpy_include_metadata:
        return {
            "value": value,
            "__numpy_type__": type_name,
            "__numpy_dtype__": str(scalar.dtype),
        }
    else:
        return value