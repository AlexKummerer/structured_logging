"""
Configuration for performance tests
"""

import pytest


def pytest_configure(config):
    """Configure pytest for performance tests"""
    config.addinivalue_line(
        "markers", "performance: mark test as performance benchmark"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle performance tests"""
    for item in items:
        # Add performance marker to all tests in performance/ directory
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)


@pytest.fixture(scope="session")
def performance_config():
    """Configuration for performance tests"""
    return {
        "min_throughput_basic": 50000,  # logs/sec for basic logging
        "min_throughput_structured": 3000,  # logs/sec for structured logging
        "min_throughput_filtered": 2000,  # logs/sec for filtered logging
        "min_throughput_async": 10000,  # logs/sec for async logging
        "max_memory_per_log": 1.0,  # KB per log entry
        "max_filter_overhead": 50,  # % overhead for filtering
    }
