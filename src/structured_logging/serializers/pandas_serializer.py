"""
Pandas DataFrame and Series serialization module
"""

from typing import Any, Dict, List

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

from .config import SerializationConfig


def _add_memory_usage_info(df: Any, config: SerializationConfig, result: Dict[str, Any]) -> None:
    """Add memory usage information to result"""
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


def _add_dtype_info(df: Any, config: SerializationConfig, result: Dict[str, Any]) -> None:
    """Add data type information to result"""
    if config.pandas_include_dtypes:
        result["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
        result["column_types"] = _categorize_dataframe_columns(df)


def _handle_column_sampling(df: Any, config: SerializationConfig, result: Dict[str, Any]) -> Any:
    """Handle column sampling and return DataFrame for serialization"""
    cols_to_include = _select_columns_for_serialization(df, config)
    if cols_to_include != list(df.columns):
        result["columns_sampled"] = True
        result["columns_included"] = cols_to_include
        result["columns_omitted"] = len(df.columns) - len(cols_to_include)
        return df[cols_to_include]
    else:
        result["columns_sampled"] = False
        return df


def _serialize_dataframe_content(df_for_serialization: Any, cols_to_include: list, config: SerializationConfig, result: Dict[str, Any]) -> None:
    """Serialize DataFrame content based on size"""
    if len(df_for_serialization) <= config.pandas_max_rows and len(cols_to_include) <= config.pandas_max_cols:
        result["data"] = _serialize_dataframe_data(df_for_serialization, config)
        result["serialization_method"] = "full"
    else:
        result.update(_serialize_dataframe_sample(df_for_serialization, config))
        result["serialization_method"] = "sampled"


def _add_statistics(df: Any, config: SerializationConfig, result: Dict[str, Any]) -> None:
    """Add statistical summary for numeric columns"""
    if config.pandas_include_describe and HAS_NUMPY:
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                describe_data = df[numeric_cols].describe()
                result["statistics"] = {
                    col: {stat: float(val) for stat, val in describe_data[col].items()}
                    for col in numeric_cols
                }
        except Exception as e:
            result["statistics_error"] = str(e)


def _add_multiindex_info(df: Any, config: SerializationConfig, result: Dict[str, Any]) -> None:
    """Add MultiIndex column information to result"""
    if config.pandas_handle_multiindex and isinstance(df.columns, pd.MultiIndex):
        result["multiindex_columns"] = {
            "levels": [list(level) for level in df.columns.levels],
            "names": list(df.columns.names),
            "nlevels": df.columns.nlevels,
        }


def serialize_dataframe(df: Any, config: SerializationConfig) -> Dict[str, Any]:
    """Enhanced Pandas DataFrame serialization with comprehensive features"""
    if not HAS_PANDAS:
        return {"error": "Pandas not available"}

    result = {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "__pandas_type__": "DataFrame",
    }

    if config.pandas_include_index:
        result["index_info"] = _serialize_pandas_index(df.index, config)

    _add_memory_usage_info(df, config, result)
    _add_dtype_info(df, config, result)
    
    df_for_serialization = _handle_column_sampling(df, config, result)
    cols_to_include = result.get("columns_included", list(df.columns))
    
    _serialize_dataframe_content(df_for_serialization, cols_to_include, config, result)
    _add_statistics(df, config, result)
    _add_multiindex_info(df, config, result)

    return result


def _add_series_memory_usage(series: Any, config: SerializationConfig, result: Dict[str, Any]) -> None:
    """Add memory usage information to series result"""
    if config.pandas_include_memory_usage:
        try:
            memory_bytes = series.memory_usage(deep=True)
            result["memory_usage"] = {
                "bytes": int(memory_bytes),
                "mb": round(float(memory_bytes / (1024 * 1024)), 3),
            }
        except Exception as e:
            result["memory_usage_error"] = str(e)


def _add_series_value_analysis(series: Any, config: SerializationConfig, result: Dict[str, Any]) -> None:
    """Add value analysis to series result"""
    try:
        result["value_analysis"] = _analyze_series_values(series, config)
    except Exception as e:
        result["value_analysis_error"] = str(e)


def _serialize_series_content(series: Any, config: SerializationConfig, result: Dict[str, Any]) -> None:
    """Serialize series data based on size"""
    if len(series) <= config.pandas_max_rows:
        result["data"] = _serialize_series_data(series, config)
        result["serialization_method"] = "full"
    else:
        result.update(_serialize_series_sample(series, config))
        result["serialization_method"] = "sampled"


def _add_series_statistics(series: Any, config: SerializationConfig, result: Dict[str, Any]) -> None:
    """Add statistics for numeric series"""
    if config.pandas_include_describe and pd.api.types.is_numeric_dtype(series):
        try:
            describe = series.describe()
            result["statistics"] = {
                stat: float(val) for stat, val in describe.items()
            }
        except Exception as e:
            result["statistics_error"] = str(e)


def serialize_series(series: Any, config: SerializationConfig) -> Dict[str, Any]:
    """Enhanced Pandas Series serialization"""
    if not HAS_PANDAS:
        return {"error": "Pandas not available"}

    result = {
        "name": series.name,
        "dtype": str(series.dtype),
        "size": len(series),
        "__pandas_type__": "Series",
    }

    if config.pandas_include_index:
        result["index_info"] = _serialize_pandas_index(series.index, config)
    
    _add_series_memory_usage(series, config, result)
    _add_series_value_analysis(series, config, result)
    _serialize_series_content(series, config, result)
    _add_series_statistics(series, config, result)

    return result


def serialize_pandas_timestamp(ts: Any, config: SerializationConfig) -> Any:
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


def _add_categorical_value_counts(categorical: Any, result: Dict[str, Any]) -> None:
    """Add value counts to categorical result"""
    try:
        value_counts = categorical.value_counts()
        result["value_counts"] = {
            str(cat): int(count) for cat, count in value_counts.items()
        }
    except Exception as e:
        result["value_counts_error"] = str(e)


def _serialize_small_categorical(categorical: Any, config: SerializationConfig, result: Dict[str, Any]) -> None:
    """Serialize small categorical data"""
    if config.pandas_categorical_as_codes:
        result["data"] = categorical.codes.tolist()
        result["data_type"] = "codes"
    else:
        result["data"] = categorical.astype(str).tolist()
        result["data_type"] = "values"


def _serialize_large_categorical(categorical: Any, config: SerializationConfig, result: Dict[str, Any]) -> None:
    """Serialize large categorical data with sampling"""
    head_size = min(config.pandas_max_rows // 2, len(categorical))
    tail_size = min(config.pandas_max_rows - head_size, len(categorical))

    if config.pandas_categorical_as_codes:
        result["sample"] = {
            "head": categorical.codes[:head_size].tolist(),
            "tail": categorical.codes[-tail_size:].tolist(),
            "data_type": "codes",
        }
    else:
        result["sample"] = {
            "head": categorical[:head_size].astype(str).tolist(),
            "tail": categorical[-tail_size:].astype(str).tolist(),
            "data_type": "values",
        }


def serialize_pandas_categorical(categorical: Any, config: SerializationConfig) -> Dict[str, Any]:
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
    
    _add_categorical_value_counts(categorical, result)
    
    if len(categorical) <= config.pandas_max_rows:
        _serialize_small_categorical(categorical, config, result)
    else:
        _serialize_large_categorical(categorical, config, result)

    return result


# Helper functions
def _serialize_pandas_index(index: Any, config: SerializationConfig) -> Dict[str, Any]:
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


def _select_columns_for_serialization(
    df: Any, config: SerializationConfig
) -> List[str]:
    """Select which columns to include in serialization"""
    if len(df.columns) <= config.pandas_max_cols:
        return list(df.columns)

    # Priority-based column selection
    priority_cols = []
    remaining_cols = list(df.columns)

    if HAS_NUMPY:
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

    return priority_cols


def _serialize_dataframe_data(df: Any, config: SerializationConfig) -> Dict[str, Any]:
    """Serialize DataFrame data with proper type handling"""
    result = {}

    # Convert each column based on its type
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            if config.pandas_datetime_format == "iso":
                result[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist()
            elif config.pandas_datetime_format == "timestamp":
                result[col] = (df[col].astype("int64") // 10**9).tolist()
            else:
                result[col] = df[col].tolist()
        elif pd.api.types.is_categorical_dtype(df[col]):
            if config.pandas_categorical_as_codes:
                result[col] = df[col].cat.codes.tolist()
            else:
                result[col] = df[col].astype(str).tolist()
        else:
            try:
                result[col] = df[col].tolist()
            except Exception as e:
                result[col] = {"error": str(e), "sample": str(df[col].iloc[:5])}

    return result


def _serialize_dataframe_sample(
    df: Any, config: SerializationConfig
) -> Dict[str, Any]:
    """Generate sample representation for large DataFrames"""
    result = {}

    if config.pandas_sample_method == "head_tail":
        # Traditional head/tail sampling
        head_size = min(config.pandas_max_rows // 2, len(df))
        tail_size = min(config.pandas_max_rows - head_size, len(df))

        result["sample"] = {
            "head": _serialize_dataframe_data(df.head(head_size), config),
            "tail": _serialize_dataframe_data(df.tail(tail_size), config),
            "rows_omitted": len(df) - head_size - tail_size,
        }

    elif config.pandas_sample_method == "random":
        # Random sampling
        sample_size = min(config.pandas_max_rows, len(df))
        sampled_df = df.sample(n=sample_size, random_state=42)
        result["sample"] = {
            "random": _serialize_dataframe_data(sampled_df, config),
            "rows_omitted": len(df) - sample_size,
            "sampling_method": "random",
        }

    elif config.pandas_sample_method == "systematic":
        # Systematic sampling
        sample_size = min(config.pandas_max_rows, len(df))
        step = max(1, len(df) // sample_size)
        sampled_df = df.iloc[::step][:sample_size]
        result["sample"] = {
            "systematic": _serialize_dataframe_data(sampled_df, config),
            "rows_omitted": len(df) - len(sampled_df),
            "sampling_method": "systematic",
            "step_size": step,
        }

    return result


def _analyze_null_values(series: Any) -> Dict[str, Any]:
    """Analyze null values in series"""
    null_count = series.isnull().sum()
    return {
        "null_count": int(null_count),
        "null_percent": round(float(null_count / len(series) * 100), 2)
    }

def _analyze_unique_values(series: Any) -> Dict[str, Any]:
    """Analyze unique values in series"""
    unique_count = series.nunique()
    result = {
        "unique_count": int(unique_count),
        "unique_percent": round(float(unique_count / len(series) * 100), 2)
    }
    
    # Top values (if reasonable number of unique values)
    if unique_count <= 20:
        value_counts = series.value_counts()
        result["top_values"] = {
            str(val): int(count) for val, count in value_counts.head(10).items()
        }
    
    return result

def _analyze_numeric_series(series: Any) -> Dict[str, Any]:
    """Analyze numeric series"""
    result = {"data_type": "numeric"}
    if len(series) > 0:
        result["range"] = {
            "min": float(series.min()),
            "max": float(series.max()),
        }
    return result

def _analyze_datetime_series(series: Any) -> Dict[str, Any]:
    """Analyze datetime series"""
    result = {"data_type": "datetime"}
    if len(series) > 0:
        result["range"] = {
            "start": series.min().isoformat(),
            "end": series.max().isoformat(),
        }
    return result

def _analyze_categorical_series(series: Any) -> Dict[str, Any]:
    """Analyze categorical series"""
    return {
        "data_type": "categorical",
        "num_categories": len(series.cat.categories)
    }

def _analyze_boolean_series(series: Any) -> Dict[str, Any]:
    """Analyze boolean series"""
    result = {"data_type": "boolean"}
    if len(series) > 0:
        result["value_counts"] = {
            "true": int(series.sum()),
            "false": int((~series).sum()),
            "null": int(series.isnull().sum()),
        }
    return result

def _analyze_series_by_type(series: Any) -> Dict[str, Any]:
    """Analyze series based on its data type"""
    if pd.api.types.is_numeric_dtype(series):
        return _analyze_numeric_series(series)
    elif pd.api.types.is_datetime64_any_dtype(series):
        return _analyze_datetime_series(series)
    elif pd.api.types.is_categorical_dtype(series):
        return _analyze_categorical_series(series)
    elif pd.api.types.is_bool_dtype(series):
        return _analyze_boolean_series(series)
    else:
        return {"data_type": "other"}

def _analyze_series_values(series: Any, config: SerializationConfig) -> Dict[str, Any]:
    """Analyze Series values for summary information"""
    analysis = {}
    
    analysis.update(_analyze_null_values(series))
    analysis.update(_analyze_unique_values(series))
    analysis.update(_analyze_series_by_type(series))
    
    return analysis


def _serialize_series_data(series: Any, config: SerializationConfig) -> Any:
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
    series: Any, config: SerializationConfig
) -> Dict[str, Any]:
    """Generate sample representation for large Series"""
    result = {}

    if config.pandas_sample_method == "head_tail":
        # Traditional head/tail sampling
        head_size = min(config.pandas_max_rows // 2, len(series))
        tail_size = min(config.pandas_max_rows - head_size, len(series))

        result["sample"] = {
            "head": _serialize_series_data(series.head(head_size), config),
            "tail": _serialize_series_data(series.tail(tail_size), config),
            "values_omitted": len(series) - head_size - tail_size,
        }

    elif config.pandas_sample_method == "random":
        # Random sampling
        sample_size = min(config.pandas_max_rows, len(series))
        sampled_series = series.sample(n=sample_size, random_state=42)
        result["sample"] = {
            "random": _serialize_series_data(sampled_series, config),
            "values_omitted": len(series) - sample_size,
            "sampling_method": "random",
        }

    elif config.pandas_sample_method == "systematic":
        # Systematic sampling
        sample_size = min(config.pandas_max_rows, len(series))
        step = max(1, len(series) // sample_size)
        sampled_series = series.iloc[::step][:sample_size]
        result["sample"] = {
            "systematic": _serialize_series_data(sampled_series, config),
            "values_omitted": len(series) - len(sampled_series),
            "sampling_method": "systematic",
            "step_size": step,
        }

    return result