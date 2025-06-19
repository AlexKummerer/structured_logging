"""
Performance utilities for optimized structured logging
"""

import time
from datetime import datetime
from typing import Callable


# Performance optimization: Cache timestamp format function
_timestamp_cache_time = 0.0
_timestamp_cache_value = ""
_timestamp_cache_duration = 0.001  # 1ms cache duration


def get_optimized_timestamp() -> str:
    """
    Optimized timestamp generation with micro-caching
    
    Caches timestamp for 1ms to avoid excessive datetime.now() calls
    in high-frequency logging scenarios.
    """
    global _timestamp_cache_time, _timestamp_cache_value
    
    current_time = time.perf_counter()
    
    # Check if cached timestamp is still valid (within 1ms)
    if current_time - _timestamp_cache_time < _timestamp_cache_duration:
        return _timestamp_cache_value
    
    # Generate new timestamp
    _timestamp_cache_time = current_time
    _timestamp_cache_value = datetime.now().isoformat() + "Z"
    
    return _timestamp_cache_value


def create_optimized_timestamp_func() -> Callable[[], str]:
    """
    Create a closure-based optimized timestamp function
    
    Returns a function that generates timestamps with minimal overhead
    """
    last_time = 0.0
    last_timestamp = ""
    
    def optimized_timestamp() -> str:
        nonlocal last_time, last_timestamp
        
        current_time = time.perf_counter()
        
        # Cache for 1ms to balance accuracy vs performance
        if current_time - last_time < 0.001:
            return last_timestamp
            
        last_time = current_time
        last_timestamp = datetime.now().isoformat() + "Z"
        return last_timestamp
    
    return optimized_timestamp


# Pre-create optimized timestamp function
_optimized_timestamp = create_optimized_timestamp_func()


def fast_timestamp() -> str:
    """
    Fast timestamp generation using pre-created closure
    
    This is the fastest timestamp generation method available
    """
    return _optimized_timestamp()


# Performance measurement utilities
def measure_function_performance(func: Callable, iterations: int = 1000) -> dict:
    """
    Measure function performance over multiple iterations
    
    Args:
        func: Function to measure
        iterations: Number of iterations to run
        
    Returns:
        Dictionary with performance metrics
    """
    times = []
    
    for _ in range(5):  # Run 5 samples
        start = time.perf_counter()
        for _ in range(iterations):
            func()
        end = time.perf_counter()
        times.append((end - start) / iterations * 1000)  # ms per call
    
    return {
        'mean_ms': sum(times) / len(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'throughput_per_sec': 1000 / (sum(times) / len(times))
    }