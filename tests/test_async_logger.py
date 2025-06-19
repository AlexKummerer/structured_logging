"""
Tests for async logging functionality
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from io import StringIO

from structured_logging import (
    AsyncLogger,
    AsyncLoggerConfig,
    get_async_logger,
    alog_with_context,
    async_request_context,
    shutdown_all_async_loggers,
    LoggerConfig,
)


class TestAsyncLoggerConfig:
    """Test async logger configuration"""
    
    def test_default_config(self):
        """Test default async configuration"""
        config = AsyncLoggerConfig()
        
        assert config.queue_size == 1000
        assert config.max_workers == 2
        assert config.batch_size == 50
        assert config.flush_interval == 1.0
        assert config.queue_timeout == 0.1
        assert config.shutdown_timeout == 5.0
        assert config.max_memory_mb is None
        assert config.drop_on_overflow is False
        assert config.error_callback is None
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config
        config = AsyncLoggerConfig(queue_size=100, max_workers=1, batch_size=10)
        assert config.queue_size == 100
        
        # Invalid queue_size
        with pytest.raises(ValueError, match="queue_size must be positive"):
            AsyncLoggerConfig(queue_size=0)
        
        # Invalid max_workers
        with pytest.raises(ValueError, match="max_workers must be positive"):
            AsyncLoggerConfig(max_workers=0)
        
        # Invalid batch_size
        with pytest.raises(ValueError, match="batch_size must be positive"):
            AsyncLoggerConfig(batch_size=0)
        
        # Invalid flush_interval
        with pytest.raises(ValueError, match="flush_interval must be positive"):
            AsyncLoggerConfig(flush_interval=0)
    
    def test_custom_error_callback(self):
        """Test custom error callback"""
        callback = MagicMock()
        config = AsyncLoggerConfig(error_callback=callback)
        assert config.error_callback == callback


class TestAsyncContext:
    """Test async context management"""
    
    @pytest.mark.asyncio
    async def test_async_request_context(self):
        """Test async request context manager"""
        async with async_request_context(user_id="user123", tenant_id="tenant456"):
            from structured_logging.async_context import aget_user_context, aget_request_id
            
            # Check context is set
            user_context = await aget_user_context()
            assert user_context["user_id"] == "user123" 
            assert user_context["tenant_id"] == "tenant456"
            
            # Check request ID is generated
            request_id = await aget_request_id()
            assert request_id is not None
            assert len(request_id) > 0
    
    @pytest.mark.asyncio
    async def test_async_request_context_with_custom_fields(self):
        """Test async context with custom fields"""
        async with async_request_context(
            user_id="user123", 
            service="payment-api",
            version="1.2.0"
        ):
            from structured_logging.async_context import aget_custom_context
            
            custom_context = await aget_custom_context()
            assert custom_context["service"] == "payment-api"
            assert custom_context["version"] == "1.2.0"
    
    @pytest.mark.asyncio
    async def test_async_context_isolation(self):
        """Test that async contexts are properly isolated"""
        async def task_with_context(user_id: str, results: list):
            async with async_request_context(user_id=user_id):
                from structured_logging.async_context import aget_user_context
                await asyncio.sleep(0.01)  # Simulate async work
                user_context = await aget_user_context()
                results.append(user_context["user_id"])
        
        # Run multiple concurrent tasks
        results = []
        tasks = [
            task_with_context("user1", results),
            task_with_context("user2", results),
            task_with_context("user3", results),
        ]
        
        await asyncio.gather(*tasks)
        
        # Each task should have seen its own user_id
        assert len(results) == 3
        assert "user1" in results
        assert "user2" in results
        assert "user3" in results


class TestAsyncLogger:
    """Test async logger functionality"""
    
    @pytest.mark.asyncio
    async def test_async_logger_creation(self):
        """Test async logger creation"""
        logger = get_async_logger("test_async")
        assert isinstance(logger, AsyncLogger)
        assert logger.name == "test_async"
        
        # Cleanup
        await logger.stop()
    
    @pytest.mark.asyncio
    async def test_async_logger_with_config(self):
        """Test async logger with custom configuration"""
        async_config = AsyncLoggerConfig(batch_size=10, queue_size=100)
        logger_config = LoggerConfig(formatter_type="json")
        
        logger = get_async_logger("test_config", logger_config, async_config)
        assert logger.async_config.batch_size == 10
        assert logger.async_config.queue_size == 100
        
        # Cleanup
        await logger.stop()
    
    @pytest.mark.asyncio
    async def test_async_logging_methods(self):
        """Test all async logging methods"""
        # Capture output
        output = StringIO()
        
        # Use DEBUG level to capture all messages
        logger_config = LoggerConfig(log_level="DEBUG")
        logger = get_async_logger("test_methods", logger_config)
        logger.processor.stream = output
        
        # Test all log levels
        await logger.adebug("Debug message")
        await logger.ainfo("Info message")
        await logger.awarning("Warning message")
        await logger.aerror("Error message")
        await logger.acritical("Critical message")
        
        # Wait for processing and flush
        await asyncio.sleep(0.1)
        await logger.flush()
        
        # Check output
        output_text = output.getvalue()
        assert "Debug message" in output_text
        assert "Info message" in output_text
        assert "Warning message" in output_text
        assert "Error message" in output_text
        assert "Critical message" in output_text
        
        # Cleanup
        await logger.stop()
    
    @pytest.mark.asyncio
    async def test_async_logging_with_context(self):
        """Test async logging with context"""
        output = StringIO()
        
        logger = get_async_logger("test_context")
        logger.processor.stream = output
        
        async with async_request_context(user_id="user123", tenant_id="tenant456"):
            await logger.ainfo("User action")
            await logger.aerror("User error", error_code=500)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        await logger.flush()
        
        # Check output contains context
        output_text = output.getvalue()
        assert "user123" in output_text
        assert "tenant456" in output_text
        assert "error_code" in output_text
        assert "500" in output_text
        
        # Cleanup
        await logger.stop()
    
    @pytest.mark.asyncio
    async def test_async_log_with_context_function(self):
        """Test alog_with_context function"""
        output = StringIO()
        
        logger = get_async_logger("test_func")
        logger.processor.stream = output
        
        async with async_request_context(user_id="user789"):
            success = await alog_with_context(
                logger, "info", "Context function test", 
                operation="payment", amount=99.99
            )
            assert success is True
        
        # Wait for processing
        await asyncio.sleep(0.1)
        await logger.flush()
        
        # Check output
        output_text = output.getvalue()
        assert "Context function test" in output_text
        assert "user789" in output_text
        assert "operation" in output_text
        assert "99.99" in output_text
        
        # Cleanup
        await logger.stop()
    
    @pytest.mark.asyncio
    async def test_async_logger_performance(self):
        """Test async logger performance characteristics"""
        logger = get_async_logger("test_perf")
        logger.processor.stream = StringIO()  # Suppress output
        
        # Measure time for many async log calls
        start_time = time.time()
        
        tasks = []
        for i in range(100):
            task = logger.ainfo(f"Performance test {i}")
            tasks.append(task)
        
        # Wait for all log calls to be queued
        results = await asyncio.gather(*tasks)
        queue_time = time.time() - start_time
        
        # All logs should be successfully queued
        assert all(results)
        
        # Queuing should be very fast (< 0.1 seconds for 100 logs)
        assert queue_time < 0.1
        
        # Wait for processing to complete
        await asyncio.sleep(0.2)
        await logger.flush()
        
        # Cleanup
        await logger.stop()
    
    @pytest.mark.asyncio
    async def test_async_logger_concurrent_access(self):
        """Test concurrent access to async logger"""
        logger = get_async_logger("test_concurrent")
        logger.processor.stream = StringIO()  # Suppress output
        
        async def log_worker(worker_id: int, log_count: int):
            results = []
            async with async_request_context(user_id=f"user{worker_id}"):
                for i in range(log_count):
                    success = await logger.ainfo(f"Worker {worker_id} log {i}")
                    results.append(success)
            return results
        
        # Run multiple workers concurrently
        workers = [
            log_worker(1, 20),
            log_worker(2, 20),
            log_worker(3, 20),
        ]
        
        worker_results = await asyncio.gather(*workers)
        
        # All workers should succeed
        for results in worker_results:
            assert all(results)
            assert len(results) == 20
        
        # Wait for processing
        await asyncio.sleep(0.2)
        await logger.flush()
        
        # Cleanup
        await logger.stop()
    
    @pytest.mark.asyncio
    async def test_async_logger_queue_overflow(self):
        """Test async logger behavior with queue overflow"""
        # Small queue that will overflow quickly
        async_config = AsyncLoggerConfig(
            queue_size=5,
            drop_on_overflow=True,
            batch_size=1,
            flush_interval=10.0  # Very long interval
        )
        
        errors = []
        def error_callback(error):
            errors.append(error)
        
        async_config.error_callback = error_callback
        
        logger = get_async_logger("test_overflow", async_config=async_config)
        logger.processor.stream = StringIO()  # Suppress output
        
        # Flood the logger to trigger overflow
        tasks = []
        for i in range(20):  # More than queue_size
            task = logger.ainfo(f"Overflow test {i}")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Some logs should be dropped due to overflow
        successful_logs = sum(1 for r in results if r)
        dropped_logs = sum(1 for r in results if not r)
        
        assert successful_logs > 0  # Some logs should succeed
        assert dropped_logs > 0     # Some logs should be dropped
        assert len(errors) > 0      # Error callback should be called
        
        # Cleanup
        await logger.stop()
    
    @pytest.mark.asyncio
    async def test_async_logger_graceful_shutdown(self):
        """Test graceful shutdown of async logger"""
        logger = get_async_logger("test_shutdown")
        logger.processor.stream = StringIO()
        
        # Start logging
        await logger.ainfo("Before shutdown")
        
        # Shutdown should complete without errors
        await logger.stop(timeout=1.0)
        
        # Logger should be stopped
        assert not logger.processor.running
    
    @pytest.mark.asyncio
    async def test_shutdown_all_async_loggers(self):
        """Test shutdown of all async loggers"""
        # Create multiple loggers
        logger1 = get_async_logger("shutdown_test_1")
        logger2 = get_async_logger("shutdown_test_2")
        
        # Both should be running after first log
        await logger1.ainfo("Test 1")
        await logger2.ainfo("Test 2")
        
        # Shutdown all
        await shutdown_all_async_loggers(timeout=1.0)
        
        # All should be stopped
        assert not logger1.processor.running
        assert not logger2.processor.running


class TestAsyncLoggerFormats:
    """Test async logger with different formatters"""
    
    @pytest.mark.asyncio
    async def test_async_logger_json_format(self):
        """Test async logger with JSON formatter"""
        config = LoggerConfig(formatter_type="json")
        logger = get_async_logger("test_json", config)
        
        output = StringIO()
        logger.processor.stream = output
        
        await logger.ainfo("JSON test message")
        await asyncio.sleep(0.1)
        await logger.flush()
        
        output_text = output.getvalue()
        assert '"level":"INFO"' in output_text
        assert '"message":"JSON test message"' in output_text
        
        await logger.stop()
    
    @pytest.mark.asyncio
    async def test_async_logger_csv_format(self):
        """Test async logger with CSV formatter"""
        config = LoggerConfig(formatter_type="csv")
        logger = get_async_logger("test_csv", config)
        
        output = StringIO()
        logger.processor.stream = output
        
        await logger.ainfo("CSV test message")
        await asyncio.sleep(0.1)
        await logger.flush()
        
        output_text = output.getvalue()
        # CSV should contain the message
        assert "CSV test message" in output_text
        assert "INFO" in output_text
        
        await logger.stop()
    
    @pytest.mark.asyncio
    async def test_async_logger_plain_format(self):
        """Test async logger with plain text formatter"""
        config = LoggerConfig(formatter_type="plain")
        logger = get_async_logger("test_plain", config)
        
        output = StringIO()
        logger.processor.stream = output
        
        await logger.ainfo("Plain test message")
        await asyncio.sleep(0.1)
        await logger.flush()
        
        output_text = output.getvalue()
        assert "Plain test message" in output_text
        assert "INFO" in output_text
        
        await logger.stop()


@pytest.mark.asyncio
async def test_async_logger_integration():
    """Integration test for complete async logging workflow"""
    # Setup
    async_config = AsyncLoggerConfig(batch_size=5, flush_interval=0.1)
    logger_config = LoggerConfig(formatter_type="json", log_level="DEBUG")  # Include DEBUG messages
    
    logger = get_async_logger("integration_test", logger_config, async_config)
    output = StringIO()
    logger.processor.stream = output
    
    try:
        # Simulate real application workflow
        async with async_request_context(
            user_id="integration_user",
            tenant_id="integration_tenant",
            service="integration_service"
        ):
            # Start operation
            await logger.ainfo("Starting integration test")
            
            # Simulate some async work with multiple log entries
            for i in range(3):
                await logger.adebug(f"Processing step {i}")
                await asyncio.sleep(0.01)  # Simulate async work
            
            # Log completion with extra data
            await alog_with_context(
                logger, "info", "Integration test completed",
                steps_completed=3,
                duration_ms=30
            )
        
        # Wait for all logs to be processed
        await asyncio.sleep(0.2)
        await logger.flush()
        
        # Verify output
        output_text = output.getvalue()
        lines = [line for line in output_text.split('\n') if line.strip()]
        
        # Should have 5 log entries (1 info + 3 debug + 1 info)
        assert len(lines) == 5
        
        # Check context propagation
        for line in lines:
            assert "integration_user" in line
            assert "integration_tenant" in line
            assert "integration_service" in line
        
        # Check specific messages
        assert any("Starting integration test" in line for line in lines)
        assert any("Integration test completed" in line for line in lines)
        assert any("steps_completed" in line for line in lines)
        
    finally:
        await logger.stop()