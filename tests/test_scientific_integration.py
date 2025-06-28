"""
Tests for enhanced NumPy/Pandas integration
"""

import pytest

from structured_logging.serializers import (
    SerializationConfig,
    serialize_for_logging,
)

# Optional imports
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


@pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")
class TestEnhancedNumpyIntegration:
    """Tests for enhanced NumPy serialization"""

    def test_numpy_scalar_enhanced_serialization(self):
        """Test enhanced NumPy scalar serialization with metadata"""
        config = SerializationConfig(
            numpy_include_metadata=True, numpy_array_precision=3
        )

        # Test different scalar types
        test_cases = [
            (np.int32(42), "integer"),
            (np.float64(3.14159), "floating"),
            (np.complex128(1 + 2j), "complex"),
            (np.bool_(True), "boolean"),
            (np.str_("test"), "string"),
        ]

        for scalar, expected_type in test_cases:
            result = serialize_for_logging(scalar, config)
            assert isinstance(result, dict)
            assert result["__numpy_type__"] == expected_type
            assert "__numpy_dtype__" in result

    def test_numpy_scalar_precision_control(self):
        """Test precision control for NumPy floating scalars"""
        config = SerializationConfig(
            numpy_include_metadata=False, numpy_array_precision=2
        )

        float_scalar = np.float64(3.14159265359)
        result = serialize_for_logging(float_scalar, config)
        assert result == 3.14

    def test_numpy_inf_nan_handling(self):
        """Test handling of inf and nan values"""
        config = SerializationConfig(numpy_handle_inf_nan=True)

        # Test NaN
        nan_scalar = np.float64(np.nan)
        result = serialize_for_logging(nan_scalar, config)
        assert result["__numpy_special__"] == "nan"

        # Test positive infinity
        pos_inf = np.float64(np.inf)
        result = serialize_for_logging(pos_inf, config)
        assert result["__numpy_special__"] == "+inf"

        # Test negative infinity
        neg_inf = np.float64(-np.inf)
        result = serialize_for_logging(neg_inf, config)
        assert result["__numpy_special__"] == "-inf"

    def test_numpy_array_enhanced_metadata(self):
        """Test enhanced NumPy array serialization with comprehensive metadata"""
        config = SerializationConfig(
            numpy_include_metadata=True,
            numpy_array_max_size=5,
            numpy_stats_for_numeric=True,
        )

        # Small array with full data
        small_array = np.array([1, 2, 3, 4])
        result = serialize_for_logging(small_array, config)

        assert result["__numpy_type__"] == "ndarray"
        assert result["shape"] == [4]
        assert result["ndim"] == 1
        assert "flags" in result
        assert result["serialization_method"] == "full"
        assert "data" in result
        assert "statistics" in result

    def test_numpy_array_compression_strategies(self):
        """Test different array compression strategies"""
        config = SerializationConfig(
            numpy_array_max_size=10,
            numpy_compression_threshold=50,
            numpy_sample_size=5,
            numpy_stats_for_numeric=True,
        )

        # Medium array (summary)
        medium_array = np.arange(25)
        result = serialize_for_logging(medium_array, config)
        assert result["serialization_method"] == "summary"
        assert "sample" in result
        assert "statistics" in result

        # Large array (compressed)
        large_array = np.arange(100)
        result = serialize_for_logging(large_array, config)
        assert result["serialization_method"] == "compressed"
        assert "minimal_sample" in result
        assert "memory_info" in result

    def test_numpy_multidimensional_arrays(self):
        """Test serialization of multi-dimensional arrays"""
        config = SerializationConfig(
            numpy_include_metadata=True, numpy_array_max_size=50
        )

        # 2D array
        array_2d = np.random.rand(3, 4)
        result = serialize_for_logging(array_2d, config)

        assert result["shape"] == [3, 4]
        assert result["ndim"] == 2
        assert result["size"] == 12
        assert len(result["data"]) == 3  # 3 rows
        assert len(result["data"][0]) == 4  # 4 columns per row

    def test_numpy_array_with_special_values(self):
        """Test arrays containing inf/nan values"""
        config = SerializationConfig(numpy_handle_inf_nan=True, numpy_array_max_size=20)

        array_with_special = np.array([1.0, np.nan, np.inf, -np.inf, 2.0])
        result = serialize_for_logging(array_with_special, config)

        assert "special_values" in result
        assert result["special_values"]["has_nan"] is True
        assert result["special_values"]["has_inf"] is True
        assert result["special_values"]["nan_count"] == 1
        assert result["special_values"]["inf_count"] == 2

    @pytest.mark.skipif(not hasattr(np, "ma"), reason="numpy.ma not available")
    def test_numpy_masked_array_serialization(self):
        """Test NumPy masked array serialization"""
        config = SerializationConfig(numpy_array_max_size=20)

        # Create masked array
        data = np.array([1, 2, 3, 4, 5])
        mask = np.array([False, True, False, True, False])
        masked_array = np.ma.masked_array(data, mask=mask)

        result = serialize_for_logging(masked_array, config)

        assert result["__numpy_type__"] == "MaskedArray"
        assert result["mask_info"]["has_mask"] is True
        assert result["mask_info"]["masked_count"] == 2
        assert "data" in result
        assert "mask" in result


@pytest.mark.skipif(not HAS_PANDAS, reason="Pandas not available")
class TestEnhancedPandasIntegration:
    """Tests for enhanced Pandas serialization"""

    def test_dataframe_enhanced_serialization(self):
        """Test enhanced DataFrame serialization with comprehensive features"""
        config = SerializationConfig(
            pandas_max_rows=10,
            pandas_max_cols=5,
            pandas_include_dtypes=True,
            pandas_include_index=True,
            pandas_include_memory_usage=True,
            pandas_include_describe=True,
        )

        # Create test DataFrame
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3, 4, 5],
                "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
                "str_col": ["a", "b", "c", "d", "e"],
                "bool_col": [True, False, True, False, True],
                "datetime_col": pd.date_range("2024-01-01", periods=5),
            }
        )

        result = serialize_for_logging(df, config)

        assert result["__pandas_type__"] == "DataFrame"
        assert result["shape"] == [5, 5]
        assert "memory_usage" in result
        assert "dtypes" in result
        assert "column_types" in result
        assert "index_info" in result
        assert "statistics" in result
        assert result["serialization_method"] == "full"

    def test_dataframe_column_sampling(self):
        """Test DataFrame column sampling for wide DataFrames"""
        config = SerializationConfig(pandas_max_cols=3, pandas_include_dtypes=True)

        # Create wide DataFrame
        df = pd.DataFrame(
            {
                f"col_{i}": (
                    np.random.randn(10)
                    if i % 2 == 0
                    else [f"str_{j}" for j in range(10)]
                )
                for i in range(10)
            }
        )

        result = serialize_for_logging(df, config)

        assert result["columns_sampled"] is True
        assert len(result["columns_included"]) <= 3
        assert result["columns_omitted"] > 0

    def test_dataframe_sampling_methods(self):
        """Test different DataFrame sampling methods"""
        # Create large DataFrame
        df = pd.DataFrame(
            {
                "values": np.arange(100),
                "letters": [chr(65 + (i % 26)) for i in range(100)],
            }
        )

        # Test head_tail sampling
        config_head_tail = SerializationConfig(
            pandas_max_rows=10, pandas_sample_method="head_tail"
        )
        result = serialize_for_logging(df, config_head_tail)
        assert "head" in result["sample"]
        assert "tail" in result["sample"]

        # Test random sampling
        config_random = SerializationConfig(
            pandas_max_rows=10, pandas_sample_method="random"
        )
        result = serialize_for_logging(df, config_random)
        assert "random" in result["sample"]
        assert result["sample"]["sampling_method"] == "random"

        # Test systematic sampling
        config_systematic = SerializationConfig(
            pandas_max_rows=10, pandas_sample_method="systematic"
        )
        result = serialize_for_logging(df, config_systematic)
        assert "systematic" in result["sample"]
        assert "step_size" in result["sample"]

    def test_series_enhanced_serialization(self):
        """Test enhanced Series serialization"""
        config = SerializationConfig(
            pandas_include_index=True,
            pandas_include_memory_usage=True,
            pandas_include_describe=True,
        )

        # Numeric series
        numeric_series = pd.Series(np.random.randn(20), name="numeric_data")
        result = serialize_for_logging(numeric_series, config)

        assert result["__pandas_type__"] == "Series"
        assert "value_analysis" in result
        assert result["value_analysis"]["data_type"] == "numeric"
        assert "memory_usage" in result
        assert "index_info" in result
        assert "statistics" in result
        assert "range" in result["value_analysis"]

    def test_series_categorical_serialization(self):
        """Test Series with categorical data"""
        config = SerializationConfig(
            pandas_categorical_as_codes=False, pandas_max_rows=20
        )

        # Create categorical series
        categories = ["small", "medium", "large"]
        categorical_series = pd.Series(
            pd.Categorical(
                ["small", "large", "medium", "small", "large"] * 10,
                categories=categories,
                ordered=True,
            ),
            name="size_category",
        )

        result = serialize_for_logging(categorical_series, config)

        assert result["data_type"] == "categorical"
        assert result["categories"] == categories
        assert result["ordered"] is True
        assert result["num_categories"] == 3

    def test_pandas_timestamp_serialization(self):
        """Test Pandas Timestamp serialization"""
        config_iso = SerializationConfig(pandas_datetime_format="iso")
        config_timestamp = SerializationConfig(pandas_datetime_format="timestamp")

        ts = pd.Timestamp("2024-01-15 10:30:00")

        # ISO format
        result_iso = serialize_for_logging(ts, config_iso)
        assert isinstance(result_iso, str)
        assert "2024-01-15" in result_iso

        # Timestamp format
        result_timestamp = serialize_for_logging(ts, config_timestamp)
        assert isinstance(result_timestamp, int)

    def test_pandas_index_types_serialization(self):
        """Test different pandas Index types"""
        config = SerializationConfig(pandas_include_index=True)

        # DatetimeIndex
        datetime_index = pd.date_range("2024-01-01", periods=5, freq="D")
        df_datetime = pd.DataFrame({"values": [1, 2, 3, 4, 5]}, index=datetime_index)
        result = serialize_for_logging(df_datetime, config)
        assert result["index_info"]["index_type"] == "DatetimeIndex"
        assert "range" in result["index_info"]

        # MultiIndex
        multi_index = pd.MultiIndex.from_tuples(
            [("A", 1), ("A", 2), ("B", 1), ("B", 2)], names=["letter", "number"]
        )
        df_multi = pd.DataFrame({"values": [1, 2, 3, 4]}, index=multi_index)
        result = serialize_for_logging(df_multi, config)
        assert result["index_info"]["index_type"] == "MultiIndex"
        assert result["index_info"]["nlevels"] == 2

    def test_pandas_categorical_standalone(self):
        """Test standalone Categorical serialization"""
        config = SerializationConfig(
            pandas_categorical_as_codes=True, pandas_max_rows=10
        )

        categorical = pd.Categorical(
            ["red", "blue", "red", "green", "blue"] * 5,
            categories=["red", "green", "blue"],
            ordered=False,
        )

        result = serialize_for_logging(categorical, config)

        assert result["__pandas_type__"] == "Categorical"
        assert result["num_categories"] == 3
        assert "value_counts" in result
        assert result["data_type"] == "codes"  # Because categorical_as_codes=True


@pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
class TestSciPyIntegration:
    """Tests for SciPy integration"""

    def test_scipy_sparse_matrix_serialization(self):
        """Test SciPy sparse matrix serialization"""
        config = SerializationConfig(numpy_array_max_size=100)

        # Create sparse matrix
        from scipy.sparse import csr_matrix

        data = np.array([1, 2, 3, 4])
        row = np.array([0, 0, 1, 2])
        col = np.array([0, 2, 1, 0])
        sparse_matrix = csr_matrix((data, (row, col)), shape=(3, 3))

        result = serialize_for_logging(sparse_matrix, config)

        assert result["__scipy_type__"] == "sparse_matrix"
        assert result["format"] == "csr"
        assert result["shape"] == [3, 3]
        assert result["nnz"] == 4
        assert result["density"] < 1.0
        assert "memory_info" in result
        assert "sample_data" in result


@pytest.mark.skipif(
    not HAS_NUMPY or not HAS_PANDAS, reason="NumPy or Pandas not available"
)
class TestScientificIntegrationEdgeCases:
    """Edge case tests for scientific integration"""

    def test_empty_arrays_and_dataframes(self):
        """Test serialization of empty scientific objects"""
        config = SerializationConfig()

        # Empty NumPy array
        empty_array = np.array([])
        result = serialize_for_logging(empty_array, config)
        assert result["size"] == 0
        assert result["shape"] == [0]

        # Empty DataFrame
        empty_df = pd.DataFrame()
        result = serialize_for_logging(empty_df, config)
        assert result["shape"] == [0, 0]

        # Empty Series
        empty_series = pd.Series([], dtype=float)
        result = serialize_for_logging(empty_series, config)
        assert result["size"] == 0

    def test_very_large_objects_memory_efficiency(self):
        """Test memory efficiency with very large objects"""
        config = SerializationConfig(
            numpy_array_max_size=10, numpy_compression_threshold=100, pandas_max_rows=5
        )

        # Very large array
        large_array = np.arange(10000)
        result = serialize_for_logging(large_array, config)
        assert result["serialization_method"] == "compressed"
        assert "memory_info" in result
        assert result["memory_info"]["elements_omitted"] > 0

        # Very large DataFrame
        large_df = pd.DataFrame(
            {"col1": np.arange(1000), "col2": np.random.randn(1000)}
        )
        result = serialize_for_logging(large_df, config)
        assert result["serialization_method"] == "sampled"
        assert (
            "rows_omitted" in result["sample"]["head_tail"]
            if "head_tail" in result["sample"]
            else True
        )

    def test_error_handling_in_serialization(self):
        """Test error handling in scientific serialization"""
        config = SerializationConfig()

        # Create an object that might cause serialization issues
        class ProblematicArray:
            def __init__(self):
                self.shape = (5, 5)
                self.dtype = "object"
                self.size = 25

            def tolist(self):
                raise ValueError("Cannot convert to list")

        problematic = ProblematicArray()
        # This should not crash, but handle the error gracefully
        result = serialize_for_logging(problematic, config)
        # Should get fallback representation
        assert isinstance(result, dict)

    def test_precision_and_rounding_consistency(self):
        """Test precision and rounding consistency across types"""
        config = SerializationConfig(
            numpy_array_precision=3,
            numpy_include_metadata=False,  # Disable metadata to get raw values
        )

        # Test that precision is applied consistently
        float_scalar = np.float64(3.14159265359)
        float_array = np.array([3.14159265359, 2.71828182846])

        scalar_result = serialize_for_logging(float_scalar, config)
        array_result = serialize_for_logging(float_array, config)

        # Both should be rounded to 3 decimal places
        assert scalar_result == 3.142
        assert array_result["data"] == [3.142, 2.718]


class TestScientificConfigurationOptions:
    """Test configuration options for scientific serialization"""

    def test_numpy_configuration_options(self):
        """Test various NumPy configuration combinations"""
        # Minimal configuration
        minimal_config = SerializationConfig(
            numpy_include_metadata=False,
            numpy_stats_for_numeric=False,
            numpy_handle_inf_nan=False,
        )

        # Comprehensive configuration
        comprehensive_config = SerializationConfig(
            numpy_include_metadata=True,
            numpy_stats_for_numeric=True,
            numpy_handle_inf_nan=True,
            numpy_array_precision=4,
            numpy_sample_size=15,
        )

        if HAS_NUMPY:
            test_array = np.array([1.123456789, 2.987654321, np.nan])

            minimal_result = serialize_for_logging(test_array, minimal_config)
            comprehensive_result = serialize_for_logging(
                test_array, comprehensive_config
            )

            # Minimal should have less information
            assert "statistics" not in minimal_result
            assert "special_values" not in minimal_result

            # Comprehensive should have more information
            assert "statistics" in comprehensive_result
            assert "special_values" in comprehensive_result

    def test_pandas_configuration_options(self):
        """Test various Pandas configuration combinations"""
        # Minimal configuration
        minimal_config = SerializationConfig(
            pandas_include_dtypes=False,
            pandas_include_index=False,
            pandas_include_memory_usage=False,
            pandas_include_describe=False,
        )

        # Comprehensive configuration
        comprehensive_config = SerializationConfig(
            pandas_include_dtypes=True,
            pandas_include_index=True,
            pandas_include_memory_usage=True,
            pandas_include_describe=True,
            pandas_categorical_as_codes=True,
        )

        if HAS_PANDAS:
            test_df = pd.DataFrame(
                {
                    "numbers": [1, 2, 3, 4, 5],
                    "categories": pd.Categorical(["A", "B", "A", "C", "B"]),
                }
            )

            minimal_result = serialize_for_logging(test_df, minimal_config)
            comprehensive_result = serialize_for_logging(test_df, comprehensive_config)

            # Minimal should have basic information only
            assert "dtypes" not in minimal_result
            assert "index_info" not in minimal_result
            assert "memory_usage" not in minimal_result

            # Comprehensive should have detailed information
            assert "dtypes" in comprehensive_result
            assert "index_info" in comprehensive_result
            assert "memory_usage" in comprehensive_result
