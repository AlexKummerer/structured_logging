"""
Configuration for enhanced serialization
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SerializationConfig:
    """Configuration for enhanced serialization"""

    # Basic serialization options
    datetime_format: str = "iso"  # iso, timestamp, custom
    custom_datetime_format: Optional[str] = None
    decimal_as_float: bool = False
    include_type_hints: bool = False
    max_collection_size: int = 1000
    truncate_strings: Optional[int] = None
    enum_as_value: bool = True
    dataclass_as_dict: bool = True
    path_as_string: bool = True

    # NumPy configuration
    numpy_array_max_size: int = 100
    numpy_array_precision: int = 6  # Decimal precision for floats
    numpy_include_metadata: bool = True  # Include shape, dtype, etc.
    numpy_compression_threshold: int = 1000  # Compress arrays larger than this
    numpy_sample_size: int = 10  # Sample size for large arrays
    numpy_stats_for_numeric: bool = True  # Include statistical summaries
    numpy_preserve_sparse: bool = True  # Handle sparse arrays specially
    numpy_handle_inf_nan: bool = True  # Handle inf/nan values

    # Pandas configuration
    pandas_max_rows: int = 5
    pandas_max_cols: int = 10  # Max columns to include in full serialization
    pandas_include_dtypes: bool = False
    pandas_include_index: bool = True  # Include index information
    pandas_include_memory_usage: bool = True  # Include memory statistics
    pandas_categorical_as_codes: bool = False  # Serialize categoricals as codes
    pandas_datetime_format: str = "iso"  # Format for datetime columns
    pandas_include_describe: bool = True  # Include statistical description
    pandas_sample_method: str = "head_tail"  # head_tail, random, or systematic
    pandas_handle_multiindex: bool = True  # Special handling for MultiIndex

    # Type detection options
    auto_detect_types: bool = True
    auto_convert_strings: bool = True
    auto_convert_numbers: bool = True
    detect_datetime_strings: bool = True
    detect_uuid_strings: bool = True
    detect_json_strings: bool = True
    detect_url_strings: bool = True
    type_detection_cache_size: int = 1000

    # Lazy serialization options
    enable_lazy_serialization: bool = True
    lazy_threshold_bytes: int = 1000  # Use lazy for objects larger than this
    lazy_threshold_items: int = 10  # Use lazy for collections with more items
    lazy_cache_size: int = 500  # Cache size for lazy serialization
    force_lazy_for_detection: bool = True  # Always use lazy when auto-detection is on