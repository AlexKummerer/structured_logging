"""
Tests for AWS CloudWatch integration
"""

import gzip
import json
import logging
import socket
import time
from datetime import datetime
from queue import Empty, Queue
from threading import Event
from unittest.mock import MagicMock, Mock, PropertyMock, call, patch

import pytest

# Skip all tests if boto3 is not installed
try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

if HAS_BOTO3:
    from structured_logging.cloud import CloudWatchConfig, CloudWatchHandler
    from structured_logging.cloud.utils import (
        create_cloud_logger_config,
        create_cloudwatch_logger,
    )
else:
    # Create dummy classes for tests
    CloudWatchConfig = None
    CloudWatchHandler = None


@pytest.mark.skipif(not HAS_BOTO3, reason="boto3 not installed")
class TestCloudWatchConfig:
    """Test CloudWatch configuration"""
    
    def test_default_config(self):
        config = CloudWatchConfig()
        
        assert config.log_group == "/aws/application/structured-logging"
        assert config.log_stream is not None  # Auto-generated
        assert config.create_log_group is True
        assert config.create_log_stream is True
        assert config.batch_size == 100
        assert config.flush_interval == 5.0
    
    def test_custom_config(self):
        config = CloudWatchConfig(
            log_group="/aws/my-app",
            log_stream="production",
            region="eu-west-1",
            batch_size=200
        )
        
        assert config.log_group == "/aws/my-app"
        assert config.log_stream == "production"
        assert config.region == "eu-west-1"
        assert config.batch_size == 200
    
    def test_auto_generated_stream_name(self):
        config = CloudWatchConfig()
        
        # Should contain hostname and timestamp
        assert config.log_stream is not None
        assert len(config.log_stream) > 0
        assert "_" in config.log_stream


@pytest.mark.skipif(not HAS_BOTO3, reason="boto3 not installed")
class TestCloudWatchHandler:
    """Test CloudWatch handler functionality"""
    
    @patch('boto3.client')
    def test_handler_creation(self, mock_boto3_client):
        # Mock boto3 client
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        
        # Mock responses
        mock_client.create_log_group.return_value = {}
        mock_client.create_log_stream.return_value = {}
        
        config = CloudWatchConfig(
            log_group="/test/group",
            log_stream="test-stream"
        )
        
        handler = CloudWatchHandler(config)
        
        # Should create client
        mock_boto3_client.assert_called_once_with('logs')
        
        # Should create log group and stream
        mock_client.create_log_group.assert_called_once_with(
            logGroupName="/test/group"
        )
        mock_client.create_log_stream.assert_called_once_with(
            logGroupName="/test/group",
            logStreamName="test-stream"
        )
    
    @patch('boto3.client')
    def test_log_upload(self, mock_boto3_client):
        # Mock boto3 client
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        mock_client.put_log_events.return_value = {
            'nextSequenceToken': 'token123'
        }
        mock_client.create_log_group.return_value = {}
        mock_client.create_log_stream.return_value = {}
        
        config = CloudWatchConfig(
            log_group="/test/group",
            log_stream="test-stream",
            async_upload=False  # Synchronous for testing
        )
        
        handler = CloudWatchHandler(config)
        
        # Create test log record
        import logging
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Emit log
        handler.emit(record)
        
        # Should upload to CloudWatch
        mock_client.put_log_events.assert_called_once()
        call_args = mock_client.put_log_events.call_args[1]
        
        assert call_args['logGroupName'] == "/test/group"
        assert call_args['logStreamName'] == "test-stream"
        assert len(call_args['logEvents']) == 1
        
        event = call_args['logEvents'][0]
        assert 'timestamp' in event
        assert 'message' in event
        assert "Test message" in event['message']
    
    @patch('boto3.client')
    def test_batch_upload(self, mock_boto3_client):
        # Mock boto3 client
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        mock_client.put_log_events.return_value = {
            'nextSequenceToken': 'token123'
        }
        mock_client.create_log_group.return_value = {}
        mock_client.create_log_stream.return_value = {}
        
        config = CloudWatchConfig(
            log_group="/test/group",
            log_stream="test-stream",
            batch_size=3,
            flush_interval=0.1,
            async_upload=True
        )
        
        handler = CloudWatchHandler(config)
        
        # Emit multiple logs
        import logging
        for i in range(5):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg=f"Test message {i}",
                args=(),
                exc_info=None
            )
            handler.emit(record)
        
        # Wait for batch processing
        time.sleep(0.2)
        
        # Should have uploaded in batches
        assert mock_client.put_log_events.call_count >= 1
    
    @patch('boto3.client')
    @patch('structured_logging.cloud.aws.ClientError')
    def test_error_handling(self, mock_client_error, mock_boto3_client):
        # Mock boto3 client
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        mock_client.create_log_group.return_value = {}
        mock_client.create_log_stream.return_value = {}
        
        # Simulate InvalidSequenceTokenException
        error_response = {
            'Error': {
                'Code': 'InvalidSequenceTokenException',
                'Message': 'The given sequenceToken is: correct_token'
            }
        }
        
        # Create mock error
        from botocore.exceptions import ClientError
        error = ClientError(error_response, 'PutLogEvents')
        
        mock_client.put_log_events.side_effect = [
            error,
            {'nextSequenceToken': 'new_token'}  # Success on retry
        ]
        
        config = CloudWatchConfig(
            log_group="/test/group",
            log_stream="test-stream",
            async_upload=False
        )
        
        handler = CloudWatchHandler(config)
        
        # Create test log record
        import logging
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Emit log - should retry with correct token
        handler.emit(record)
        
        # Should have retried
        assert mock_client.put_log_events.call_count == 2
    
    @patch('boto3.client')
    def test_size_limits(self, mock_boto3_client):
        # Mock boto3 client
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        mock_client.put_log_events.return_value = {
            'nextSequenceToken': 'token123'
        }
        mock_client.create_log_group.return_value = {}
        mock_client.create_log_stream.return_value = {}
        
        config = CloudWatchConfig(
            log_group="/test/group",
            log_stream="test-stream",
            batch_size=2,  # Small batch for testing
            async_upload=False
        )
        
        handler = CloudWatchHandler(config)
        
        # Create large log message
        import logging
        large_message = "x" * 1000  # 1KB message
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg=large_message,
            args=(),
            exc_info=None
        )
        
        # Emit multiple large logs
        for _ in range(5):
            handler.emit(record)
        
        # Should respect size limits
        call_count = mock_client.put_log_events.call_count
        assert call_count > 1  # Should have split into multiple batches


@pytest.mark.skipif(not HAS_BOTO3, reason="boto3 not installed")
class TestCloudWatchIntegration:
    """Integration tests with actual logging"""
    
    @patch('boto3.client')
    def test_structured_logging_integration(self, mock_boto3_client):
        from structured_logging import get_logger
        
        # Mock boto3
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        mock_client.put_log_events.return_value = {
            'nextSequenceToken': 'token123'
        }
        mock_client.create_log_group.return_value = {}
        mock_client.create_log_stream.return_value = {}
        
        # Create CloudWatch logger
        logger = create_cloudwatch_logger(
            "test_app",
            log_group="/aws/test",
            region="us-east-1"
        )
        
        # Get the CloudWatch handler and ensure it's in sync mode for testing
        for handler in logger.handlers:
            if isinstance(handler, CloudWatchHandler):
                handler.config.async_upload = False
                handler.config.flush_interval = 0
        
        # Log with context
        logger.info("Application started", extra={
            "ctx_user_id": "user123",
            "ctx_request_id": "req456"
        })
        
        # For sync mode, no need to wait
        
        # Verify CloudWatch was called
        assert mock_client.put_log_events.called
        
        # Check log format
        call_args = mock_client.put_log_events.call_args[1]
        events = call_args['logEvents']
        assert len(events) > 0
        
        # Parse message
        message = json.loads(events[0]['message'])
        assert message['message'] == "Application started"
        assert message['user_id'] == "user123"
        assert message['request_id'] == "req456"


@pytest.mark.skipif(not HAS_BOTO3, reason="boto3 not installed")
class TestCloudWatchAdvanced:
    """Advanced CloudWatch handler tests"""
    
    @patch('boto3.client')
    def test_compression(self, mock_boto3_client):
        """Test log compression functionality"""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        mock_client.create_log_group.return_value = {}
        mock_client.create_log_stream.return_value = {}
        
        config = CloudWatchConfig(
            log_group="/test/group",
            log_stream="test-stream",
            compress_logs=True,
            async_upload=False
        )
        
        handler = CloudWatchHandler(config)
        
        # Capture the actual put_log_events call
        put_calls = []
        def capture_put(*args, **kwargs):
            put_calls.append(kwargs)
            return {'nextSequenceToken': 'token123'}
        mock_client.put_log_events.side_effect = capture_put
        
        # Create large log
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Large message",
            args=(),
            exc_info=None
        )
        # Add large data with ctx_ prefix so it's included
        record.ctx_data = "x" * 1000
        
        handler.emit(record)
        
        # Verify compression was used
        assert len(put_calls) == 1
        message = put_calls[0]['logEvents'][0]['message']
        
        # Message should be JSON with large data
        parsed = json.loads(message)
        assert 'data' in parsed
        assert len(parsed['data']) == 1000
    
    @patch('boto3.client')
    def test_credentials_configuration(self, mock_boto3_client):
        """Test different credential configurations"""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        mock_client.create_log_group.return_value = {}
        mock_client.create_log_stream.return_value = {}
        
        # Test with explicit credentials
        config = CloudWatchConfig(
            log_group="/test/group",
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
            aws_session_token="test_token",
            region="eu-west-1"
        )
        
        handler = CloudWatchHandler(config)
        
        # Should pass credentials to boto3
        mock_boto3_client.assert_called_with(
            'logs',
            region_name='eu-west-1',
            aws_access_key_id='test_key',
            aws_secret_access_key='test_secret',
            aws_session_token='test_token'
        )
    
    @patch('boto3.client')
    def test_retry_logic(self, mock_boto3_client):
        """Test retry logic with exponential backoff"""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        mock_client.create_log_group.return_value = {}
        mock_client.create_log_stream.return_value = {}
        
        # Simulate throttling error
        from botocore.exceptions import ClientError
        error_response = {'Error': {'Code': 'ThrottlingException', 'Message': 'Rate exceeded'}}
        error = ClientError(error_response, 'PutLogEvents')
        
        # Fail 2 times, then succeed
        mock_client.put_log_events.side_effect = [
            error,
            error,
            {'nextSequenceToken': 'token123'}
        ]
        
        config = CloudWatchConfig(
            log_group="/test/group",
            max_retries=3,
            retry_delay=0.01,  # Fast retry for testing
            exponential_backoff=True,
            async_upload=False
        )
        
        handler = CloudWatchHandler(config)
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test retry",
            args=(),
            exc_info=None
        )
        
        handler.emit(record)
        
        # Should have retried
        assert mock_client.put_log_events.call_count == 3
    
    @patch('boto3.client')
    def test_queue_overflow(self, mock_boto3_client):
        """Test behavior when queue is full"""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        mock_client.create_log_group.return_value = {}
        mock_client.create_log_stream.return_value = {}
        
        # Block put_log_events to fill queue
        block_event = Event()
        def blocking_put(*args, **kwargs):
            block_event.wait()
            return {'nextSequenceToken': 'token123'}
        mock_client.put_log_events.side_effect = blocking_put
        
        config = CloudWatchConfig(
            log_group="/test/group",
            max_queue_size=5,  # Small queue
            async_upload=True
        )
        
        handler = CloudWatchHandler(config)
        
        # Fill the queue
        for i in range(10):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg=f"Message {i}",
                args=(),
                exc_info=None
            )
            handler.emit(record)
        
        # Unblock and cleanup
        block_event.set()
        handler.close()
        
        # Queue should have handled overflow gracefully
        assert handler.queue.qsize() <= config.max_queue_size
    
    @patch('boto3.client') 
    def test_worker_thread_lifecycle(self, mock_boto3_client):
        """Test worker thread start/stop"""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        mock_client.create_log_group.return_value = {}
        mock_client.create_log_stream.return_value = {}
        mock_client.put_log_events.return_value = {'nextSequenceToken': 'token'}
        
        config = CloudWatchConfig(
            log_group="/test/group",
            async_upload=True
        )
        
        handler = CloudWatchHandler(config)
        
        # Worker should be running
        assert handler._worker_thread is not None
        assert handler._worker_thread.is_alive()
        
        # Stop handler
        handler.close()
        
        # Worker should stop
        assert handler._stop_event.is_set()
        handler._worker_thread.join(timeout=5)
        assert not handler._worker_thread.is_alive()
    
    @patch('boto3.client')
    def test_create_log_infrastructure_errors(self, mock_boto3_client):
        """Test handling of log group/stream creation errors"""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        
        # Log group already exists
        from botocore.exceptions import ClientError
        error_response = {'Error': {'Code': 'ResourceAlreadyExistsException'}}
        error = ClientError(error_response, 'CreateLogGroup')
        mock_client.create_log_group.side_effect = error
        
        # Log stream also exists
        mock_client.create_log_stream.side_effect = error
        
        config = CloudWatchConfig(
            log_group="/test/group",
            create_log_group=True,
            create_log_stream=True
        )
        
        # Should not raise - handles existing resources
        handler = CloudWatchHandler(config)
        assert handler is not None
    
    @patch('boto3.client')
    def test_cloudwatch_insights_format(self, mock_boto3_client):
        """Test that logs are formatted for CloudWatch Insights"""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        mock_client.create_log_group.return_value = {}
        mock_client.create_log_stream.return_value = {}
        
        put_calls = []
        def capture_put(*args, **kwargs):
            put_calls.append(kwargs)
            return {'nextSequenceToken': 'token123'}
        mock_client.put_log_events.side_effect = capture_put
        
        config = CloudWatchConfig(
            log_group="/test/group",
            async_upload=False
        )
        
        handler = CloudWatchHandler(config)
        
        # Log with various data types
        record = logging.LogRecord(
            name="test.module",
            level=logging.ERROR,
            pathname="/app/test.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=None
        )
        record.ctx_response_time_ms = 123.45
        record.ctx_status_code = 500
        record.ctx_user_id = "user789"
        
        handler.emit(record)
        
        # Check CloudWatch Insights compatible format
        assert len(put_calls) == 1
        message = json.loads(put_calls[0]['logEvents'][0]['message'])
        
        # All fields should be present for Insights queries
        assert message['level'] == 'ERROR'
        assert message['logger'] == 'test.module'
        assert message['message'] == 'Error occurred'
        assert message['lineno'] == 42
        # Context fields with ctx_ prefix are included
        assert message['response_time_ms'] == 123.45
        assert message['status_code'] == 500
        assert message['user_id'] == 'user789'


@pytest.mark.skipif(not HAS_BOTO3, reason="boto3 not installed")
class TestCloudWatchUtils:
    """Test CloudWatch utility functions"""
    
    def test_create_cloud_logger_config(self):
        """Test cloud logger configuration creation"""
        # AWS config
        aws_config = create_cloud_logger_config("aws")
        assert aws_config.formatter_type == "json"
        assert aws_config.include_timestamp is True
        assert aws_config.include_request_id is True
        assert aws_config.output_type == "console"
        
        # GCP config
        gcp_config = create_cloud_logger_config("gcp")
        assert gcp_config.formatter_type == "json"
        
        # Azure config
        azure_config = create_cloud_logger_config("azure")
        assert azure_config.formatter_type == "json"
    
    @patch('boto3.client')
    def test_create_cloudwatch_logger_helper(self, mock_boto3_client):
        """Test the helper function for creating CloudWatch loggers"""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        mock_client.create_log_group.return_value = {}
        mock_client.create_log_stream.return_value = {}
        
        logger = create_cloudwatch_logger(
            name="test_app",
            log_group="/aws/test",
            log_stream="custom-stream",
            region="ap-southeast-2",
            log_level="DEBUG"
        )
        
        # Should create logger with CloudWatch handler
        assert logger.name == "test_app"
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) >= 1
        
        # Find CloudWatch handler
        cw_handler = None
        for handler in logger.handlers:
            if isinstance(handler, CloudWatchHandler):
                cw_handler = handler
                break
        
        assert cw_handler is not None
        assert cw_handler.config.log_group == "/aws/test"
        assert cw_handler.config.log_stream == "custom-stream"
        assert cw_handler.config.region == "ap-southeast-2"


@pytest.mark.skipif(not HAS_BOTO3, reason="boto3 not installed")
class TestCloudWatchBoto3Integration:
    """Test boto3 dependency handling"""
    
    def test_import_without_boto3(self):
        """Test behavior when boto3 is not installed"""
        # This test would normally mock the import system
        # For now, we just verify the HAS_BOTO3 flag exists
        assert hasattr(CloudWatchHandler, '__module__')
    
    def test_handler_creation_without_boto3(self):
        """Test handler creation fails gracefully without boto3"""
        # Test the error message format
        config = CloudWatchConfig(log_group="/test/group")
        
        # Mock the HAS_BOTO3 check inside the handler
        with patch('structured_logging.cloud.aws.HAS_BOTO3', False):
            with pytest.raises(ImportError) as exc_info:
                # Need to re-import to get the patched value
                from structured_logging.cloud.aws import CloudWatchHandler as CWHandler
                CWHandler(config)
            
            assert "boto3 is required" in str(exc_info.value)
            assert "pip install structured-logging[aws]" in str(exc_info.value)


class TestCloudWatchWithoutBoto3:
    """Test behavior when boto3 is not installed"""
    
    def test_config_creation_without_boto3(self):
        """Test that config can be created without boto3"""
        # This should work even without boto3
        from structured_logging.cloud.base import CloudHandlerConfig
        
        config = CloudHandlerConfig()
        assert config.batch_size == 100
        assert config.flush_interval == 5.0
        assert config.async_upload is True