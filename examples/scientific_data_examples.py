#!/usr/bin/env python3
"""
Scientific Data Logging Examples

This file demonstrates how to use structured logging with scientific data types
including NumPy arrays, Pandas DataFrames, and SciPy sparse matrices.

Features demonstrated:
- NumPy array and scalar serialization with metadata
- Pandas DataFrame/Series logging with sampling strategies
- SciPy sparse matrix integration
- Configuration options for scientific data
- Performance considerations for large datasets
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from datetime import datetime

from structured_logging import get_logger, request_context
from structured_logging.serializers import SerializationConfig


def demo_numpy_integration():
    """Demonstrate NumPy integration features"""
    print("=== NumPy Integration Demo ===")
    
    # Configure logger with NumPy-specific settings
    config = SerializationConfig(
        numpy_include_metadata=True,
        numpy_array_precision=3,
        numpy_handle_inf_nan=True,
        numpy_stats_for_numeric=True
    )
    
    logger = get_logger("numpy_demo", config=config)
    
    # 1. NumPy Scalars with metadata
    print("\n1. NumPy Scalar Logging:")
    with request_context(user_id="scientist_001", experiment_id="exp_123"):
        temperature = np.float64(23.456789)
        pressure = np.int32(1013)
        
        logger.info("Environmental measurements", 
                   temperature=temperature,
                   pressure=pressure,
                   equipment="sensor_v2")
    
    # 2. Small arrays with full data
    print("\n2. Small Array Logging:")
    measurements = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    logger.info("Sensor readings collected", 
               readings=measurements,
               sensor_count=len(measurements))
    
    # 3. Large arrays with compression
    print("\n3. Large Array with Compression:")
    config_compressed = SerializationConfig(
        numpy_array_max_size=10,
        numpy_compression_threshold=50,
        numpy_sample_size=5
    )
    logger_compressed = get_logger("numpy_compressed", config=config_compressed)
    
    large_dataset = np.random.randn(1000)
    logger_compressed.info("Large dataset processed", 
                          dataset=large_dataset,
                          processing_time="2.3s")
    
    # 4. Multi-dimensional arrays
    print("\n4. Multi-dimensional Array:")
    image_data = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    logger.info("Image processed",
               image_shape=image_data.shape,
               image_data=image_data,
               algorithm="edge_detection")
    
    # 5. Special values (NaN, Inf)
    print("\n5. Special Values Handling:")
    results = np.array([1.0, np.nan, np.inf, -np.inf, 2.5])
    logger.info("Analysis results with special values",
               results=results,
               valid_count=np.sum(np.isfinite(results)))


def demo_pandas_integration():
    """Demonstrate Pandas integration features"""
    print("\n=== Pandas Integration Demo ===")
    
    # Configure logger with Pandas-specific settings
    config = SerializationConfig(
        pandas_max_rows=5,
        pandas_max_cols=8,
        pandas_include_dtypes=True,
        pandas_include_memory_usage=True,
        pandas_include_describe=True,
        pandas_sample_method="head_tail"
    )
    
    logger = get_logger("pandas_demo", config=config)
    
    # 1. Small DataFrame with full data
    print("\n1. Small DataFrame Logging:")
    df_small = pd.DataFrame({
        'experiment_id': ['A001', 'A002', 'A003'],
        'temperature': [23.1, 24.5, 22.8],
        'humidity': [45.2, 47.1, 43.9],
        'success': [True, False, True]
    })
    
    logger.info("Experiment batch completed", 
               results=df_small,
               batch_size=len(df_small))
    
    # 2. Large DataFrame with sampling
    print("\n2. Large DataFrame with Sampling:")
    # Create larger dataset
    dates = pd.date_range('2024-01-01', periods=100, freq='H')
    df_large = pd.DataFrame({
        'timestamp': dates,
        'sensor_1': np.random.randn(100),
        'sensor_2': np.random.randn(100),
        'sensor_3': np.random.randn(100),
        'status': np.random.choice(['active', 'idle', 'error'], 100),
        'category': pd.Categorical(np.random.choice(['A', 'B', 'C'], 100))
    })
    
    logger.info("Large sensor dataset processed",
               dataset=df_large,
               processing_duration="5.2s")
    
    # 3. Series logging
    print("\n3. Series Logging:")
    temperature_series = pd.Series(
        np.random.normal(23, 2, 50),
        index=pd.date_range('2024-01-01', periods=50, freq='H'),
        name='temperature_celsius'
    )
    
    logger.info("Temperature monitoring complete",
               temperature_data=temperature_series,
               mean_temp=temperature_series.mean())
    
    # 4. Categorical data
    print("\n4. Categorical Data:")
    config_categorical = SerializationConfig(
        pandas_categorical_as_codes=False
    )
    logger_cat = get_logger("categorical_demo", config=config_categorical)
    
    categories = pd.Categorical(
        ['low', 'medium', 'high', 'low', 'high'] * 10,
        categories=['low', 'medium', 'high'],
        ordered=True
    )
    
    logger_cat.info("Priority analysis complete",
                   priorities=categories,
                   distribution=pd.Series(categories).value_counts())


def demo_scipy_integration():
    """Demonstrate SciPy integration features"""
    print("\n=== SciPy Integration Demo ===")
    
    config = SerializationConfig(
        numpy_array_max_size=100
    )
    
    logger = get_logger("scipy_demo", config=config)
    
    # 1. Sparse matrix logging
    print("\n1. Sparse Matrix Logging:")
    # Create a sparse matrix (adjacency matrix for a graph)
    data = np.array([1, 1, 1, 1, 1])
    row = np.array([0, 1, 2, 3, 4])
    col = np.array([1, 2, 3, 4, 0])
    adjacency_matrix = csr_matrix((data, (row, col)), shape=(5, 5))
    
    logger.info("Graph analysis completed",
               adjacency_matrix=adjacency_matrix,
               num_nodes=5,
               num_edges=len(data))
    
    # 2. Large sparse matrix
    print("\n2. Large Sparse Matrix:")
    # Create larger sparse matrix
    large_data = np.random.rand(1000)
    large_row = np.random.randint(0, 10000, 1000)
    large_col = np.random.randint(0, 10000, 1000)
    large_sparse = csr_matrix((large_data, (large_row, large_col)), shape=(10000, 10000))
    
    logger.info("Large sparse computation finished",
               result_matrix=large_sparse,
               computation_time="12.5s",
               memory_saved=f"{(1 - large_sparse.nnz / large_sparse.size) * 100:.1f}%")


def demo_performance_configurations():
    """Demonstrate different performance configurations for scientific data"""
    print("\n=== Performance Configuration Demo ===")
    
    # 1. Memory-efficient configuration for large datasets
    print("\n1. Memory-Efficient Configuration:")
    memory_config = SerializationConfig(
        numpy_array_max_size=5,
        numpy_compression_threshold=20,
        pandas_max_rows=3,
        pandas_max_cols=4,
        enable_lazy_serialization=True,
        lazy_threshold_items=10
    )
    
    memory_logger = get_logger("memory_efficient", config=memory_config)
    
    # Large dataset that will be compressed
    big_array = np.random.randn(10000)
    big_df = pd.DataFrame(np.random.randn(1000, 20))
    
    memory_logger.info("Processing large scientific dataset",
                      raw_data=big_array,
                      processed_data=big_df,
                      status="completed")
    
    # 2. High-precision configuration for accurate measurements
    print("\n2. High-Precision Configuration:")
    precision_config = SerializationConfig(
        numpy_array_precision=8,
        numpy_include_metadata=True,
        numpy_stats_for_numeric=True,
        pandas_include_describe=True
    )
    
    precision_logger = get_logger("high_precision", config=precision_config)
    
    precise_measurements = np.array([
        3.141592653589793,
        2.718281828459045,
        1.4142135623730951
    ])
    
    precision_logger.info("High-precision calculations completed",
                         measurements=precise_measurements,
                         algorithm="monte_carlo",
                         iterations=1000000)
    
    # 3. Fast logging configuration (minimal metadata)
    print("\n3. Fast Logging Configuration:")
    fast_config = SerializationConfig(
        numpy_include_metadata=False,
        numpy_stats_for_numeric=False,
        pandas_include_dtypes=False,
        pandas_include_memory_usage=False,
        pandas_include_describe=False
    )
    
    fast_logger = get_logger("fast_logging", config=fast_config)
    
    # Quick data logging without metadata overhead
    quick_data = np.random.randn(100)
    fast_logger.info("Quick data snapshot", data=quick_data)


def demo_real_world_scenario():
    """Demonstrate a realistic scientific computing scenario"""
    print("\n=== Real-World Scenario: Climate Data Analysis ===")
    
    # Configuration optimized for climate data
    climate_config = SerializationConfig(
        numpy_array_precision=2,
        numpy_include_metadata=True,
        pandas_max_rows=10,
        pandas_include_describe=True,
        pandas_sample_method="systematic"
    )
    
    logger = get_logger("climate_analysis", config=climate_config)
    
    with request_context(
        user_id="climate_researcher",
        project_id="climate_change_2024",
        location="antarctica"
    ):
        # Simulate climate data collection
        timestamps = pd.date_range('2024-01-01', periods=365, freq='D')
        
        climate_data = pd.DataFrame({
            'date': timestamps,
            'temperature': np.random.normal(-15, 10, 365),
            'wind_speed': np.random.exponential(8, 365),
            'humidity': np.random.beta(2, 5, 365) * 100,
            'pressure': np.random.normal(1013, 20, 365),
            'precipitation': np.random.poisson(2, 365),
            'weather_type': pd.Categorical(
                np.random.choice(['clear', 'cloudy', 'stormy'], 365)
            )
        })
        
        # Log data collection
        logger.info("Daily climate data collected",
                   dataset=climate_data,
                   collection_method="automated_station",
                   data_quality="validated")
        
        # Perform analysis
        monthly_avg = climate_data.groupby(climate_data['date'].dt.month).mean()
        
        logger.info("Monthly climate analysis completed",
                   monthly_averages=monthly_avg,
                   analysis_type="statistical_summary")
        
        # Log anomalies
        temp_anomalies = climate_data[
            (climate_data['temperature'] > -5) | 
            (climate_data['temperature'] < -30)
        ]
        
        if len(temp_anomalies) > 0:
            logger.warning("Temperature anomalies detected",
                          anomalies=temp_anomalies,
                          anomaly_count=len(temp_anomalies),
                          severity="requires_investigation")
        
        # Final summary
        summary_stats = {
            'mean_temperature': float(climate_data['temperature'].mean()),
            'max_wind_speed': float(climate_data['wind_speed'].max()),
            'total_precipitation': int(climate_data['precipitation'].sum()),
            'data_completeness': float((~climate_data.isnull()).mean().mean())
        }
        
        logger.info("Climate analysis summary",
                   summary=summary_stats,
                   analysis_duration="45 minutes",
                   next_collection=timestamps[-1] + pd.Timedelta(days=1))


if __name__ == "__main__":
    print("🔬 Scientific Data Logging Examples")
    print("====================================")
    
    try:
        demo_numpy_integration()
        demo_pandas_integration()
        demo_scipy_integration()
        demo_performance_configurations()
        demo_real_world_scenario()
        
        print("\n✅ All scientific data logging examples completed successfully!")
        print("\nKey Features Demonstrated:")
        print("- NumPy array and scalar serialization with metadata")
        print("- Pandas DataFrame/Series logging with intelligent sampling")
        print("- SciPy sparse matrix integration")
        print("- Configurable precision and performance settings")
        print("- Real-world climate data analysis workflow")
        
    except ImportError as e:
        print(f"\n⚠️  Missing scientific library: {e}")
        print("Install with: pip install numpy pandas scipy")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")