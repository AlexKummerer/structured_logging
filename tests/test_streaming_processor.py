"""Tests for stream processor"""

import asyncio
import pytest
from datetime import datetime, timedelta

from structured_logging.streaming import (
    LogStreamProcessor,
    StreamConfig,
    StreamState,
    create_stream_processor,
)


@pytest.mark.asyncio
async def test_stream_processor_creation():
    """Test creating stream processor"""
    processor = create_stream_processor()
    assert processor is not None
    assert processor._state == StreamState.INITIALIZED


@pytest.mark.asyncio
async def test_stream_processor_start_stop():
    """Test starting and stopping processor"""
    processor = LogStreamProcessor()
    
    # Start processor
    await processor.start()
    assert processor._state == StreamState.RUNNING
    
    # Stop processor
    await processor.stop()
    assert processor._state == StreamState.STOPPED


@pytest.mark.asyncio
async def test_stream_processor_pause_resume():
    """Test pausing and resuming processor"""
    processor = LogStreamProcessor()
    
    await processor.start()
    assert processor._state == StreamState.RUNNING
    
    await processor.pause()
    assert processor._state == StreamState.PAUSED
    
    await processor.resume()
    assert processor._state == StreamState.RUNNING
    
    await processor.stop()


@pytest.mark.asyncio
async def test_stream_processor_filter():
    """Test filter operation"""
    processor = LogStreamProcessor()
    results = []
    
    # Add filter
    processor.filter(lambda log: log.get("level") == "error")
    
    # Add sink to collect results
    async def collect_sink(items):
        results.extend(items)
    
    processor.sink(collect_sink)
    
    # Process items
    await processor.start()
    
    # Feed test data
    test_logs = [
        {"level": "info", "message": "Info log"},
        {"level": "error", "message": "Error log"},
        {"level": "warning", "message": "Warning log"},
        {"level": "error", "message": "Another error"},
    ]
    
    for log in test_logs:
        await processor._processing_queue.put(log)
    
    # Wait for processing
    await asyncio.sleep(0.5)
    
    await processor.stop()
    
    # Check results
    assert len(results) == 2
    assert all(log["level"] == "error" for log in results)


@pytest.mark.asyncio
async def test_stream_processor_map():
    """Test map operation"""
    processor = LogStreamProcessor()
    results = []
    
    # Add map operation
    processor.map(lambda log: {**log, "processed": True})
    
    # Add sink
    async def collect_sink(items):
        results.extend(items)
    
    processor.sink(collect_sink)
    
    await processor.start()
    
    # Feed test data
    test_log = {"level": "info", "message": "Test"}
    await processor._processing_queue.put(test_log)
    
    await asyncio.sleep(0.2)
    await processor.stop()
    
    # Check results
    assert len(results) == 1
    assert results[0]["processed"] is True


@pytest.mark.asyncio
async def test_stream_processor_multiple_sinks():
    """Test multiple sinks"""
    processor = LogStreamProcessor()
    results1 = []
    results2 = []
    
    # Add multiple sinks
    async def sink1(items):
        results1.extend(items)
    
    async def sink2(items):
        results2.extend(items)
    
    processor.sink(sink1, sink2)
    
    await processor.start()
    
    # Feed test data
    test_log = {"message": "Test"}
    await processor._processing_queue.put(test_log)
    
    await asyncio.sleep(0.2)
    await processor.stop()
    
    # Both sinks should receive the data
    assert len(results1) == 1
    assert len(results2) == 1


@pytest.mark.asyncio
async def test_stream_processor_backpressure():
    """Test backpressure handling"""
    config = StreamConfig(
        buffer_size=10,
        enable_backpressure=True,
        backpressure_threshold=0.8
    )
    
    processor = LogStreamProcessor(config)
    
    # Fill queue to trigger backpressure
    for i in range(8):  # 80% of buffer
        await processor._processing_queue.put({"id": i})
    
    # Queue should be at threshold
    assert processor._processing_queue.qsize() >= 8


@pytest.mark.asyncio
async def test_stream_processor_batch_processing():
    """Test batch processing"""
    config = StreamConfig(
        batch_size=5,
        buffer_timeout_ms=100
    )
    
    processor = LogStreamProcessor(config)
    results = []
    
    async def batch_sink(items):
        results.append(len(items))
    
    processor.sink(batch_sink)
    
    await processor.start()
    
    # Send exactly one batch worth
    for i in range(5):
        await processor._processing_queue.put({"id": i})
    
    await asyncio.sleep(0.3)
    
    # Send partial batch
    for i in range(3):
        await processor._processing_queue.put({"id": i})
    
    await asyncio.sleep(0.3)
    
    await processor.stop()
    
    # Should have processed 2 batches
    assert len(results) >= 2


@pytest.mark.asyncio
async def test_stream_processor_error_handling():
    """Test error handling"""
    config = StreamConfig(
        max_retries=2,
        retry_delay_ms=10
    )
    
    processor = LogStreamProcessor(config)
    error_count = 0
    
    # Sink that fails first time
    async def failing_sink(items):
        nonlocal error_count
        error_count += 1
        if error_count == 1:
            raise Exception("First attempt fails")
    
    processor.sink(failing_sink)
    
    await processor.start()
    
    await processor._processing_queue.put({"test": "data"})
    
    await asyncio.sleep(0.1)
    await processor.stop()
    
    # Should have retried
    assert error_count > 1


@pytest.mark.asyncio
async def test_stream_processor_state():
    """Test getting processor state"""
    processor = LogStreamProcessor()
    
    # Add a sink
    async def dummy_sink(items):
        pass
    
    processor.sink(dummy_sink)
    
    # Get initial state
    state = processor.get_state()
    assert state["state"] == "initialized"
    assert state["processed_count"] == 0
    assert state["error_count"] == 0
    assert state["sink_count"] == 1
    
    # Start and process some items
    await processor.start()
    
    for i in range(3):
        await processor._processing_queue.put({"id": i})
    
    await asyncio.sleep(0.3)
    
    state = processor.get_state()
    assert state["state"] == "running"
    assert state["processed_count"] > 0
    
    await processor.stop()


@pytest.mark.asyncio
async def test_stream_processor_watermark():
    """Test watermark updates"""
    processor = LogStreamProcessor()
    
    # Initial watermark
    assert processor._watermark == datetime.min
    
    # Update watermark
    now = datetime.now()
    processor._update_watermark(now)
    
    # Watermark should be updated with delay
    expected = now - timedelta(milliseconds=processor.config.watermark_delay_ms)
    assert processor._watermark >= expected