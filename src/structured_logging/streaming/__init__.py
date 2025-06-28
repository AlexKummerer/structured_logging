"""
Log stream processing for structured logging

This package provides real-time stream processing capabilities for logs,
including windowing, aggregation, and complex event processing.
"""

from .processor import (
    LogStreamProcessor,
    StreamConfig,
    StreamState,
    create_stream_processor,
)
from .operations import (
    # Window operations
    TumblingWindow,
    SlidingWindow,
    SessionWindow,
    
    # Aggregation operations
    count,
    sum_of,
    avg,
    min_of,
    max_of,
    
    # Stream operations
    filter_stream,
    map_stream,
    reduce_stream,
    group_by,
)
from .sinks import (
    StreamSink,
    ConsoleSink,
    MetricsSink,
    WebSocketSink,
    StorageSink,
    create_sink,
)
from .sources import (
    StreamSource,
    LogFileSource,
    MemorySource,
    WebSocketSource,
    create_source,
)

__all__ = [
    # Core processor
    "LogStreamProcessor",
    "StreamConfig",
    "StreamState",
    "create_stream_processor",
    
    # Windows
    "TumblingWindow",
    "SlidingWindow",
    "SessionWindow",
    
    # Aggregations
    "count",
    "sum_of",
    "avg",
    "min_of",
    "max_of",
    
    # Operations
    "filter_stream",
    "map_stream",
    "reduce_stream",
    "group_by",
    
    # Sinks
    "StreamSink",
    "ConsoleSink",
    "MetricsSink",
    "WebSocketSink",
    "StorageSink",
    "create_sink",
    
    # Sources
    "StreamSource",
    "LogFileSource",
    "MemorySource",
    "WebSocketSource",
    "create_source",
]