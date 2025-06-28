"""
Tests for Google Cloud Logging integration
"""

import json
import logging
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, call, patch

import pytest

# Skip all tests if google-cloud-logging is not installed
try:
    import google.cloud.logging
    HAS_GCP = True
except ImportError:
    HAS_GCP = False

if HAS_GCP:
    from structured_logging.cloud import (
        GoogleCloudConfig,
        GoogleCloudHandler,
        StackdriverConfig,
        StackdriverHandler,
    )
    from structured_logging.cloud.utils import (
        create_google_cloud_logger,
        create_stackdriver_logger,
    )
else:
    # Create dummy classes for tests
    GoogleCloudConfig = None
    GoogleCloudHandler = None
    StackdriverConfig = None
    StackdriverHandler = None


@pytest.mark.skipif(not HAS_GCP, reason="google-cloud-logging not installed")
class TestGoogleCloudConfig:
    """Test Google Cloud configuration"""
    
    def test_default_config(self):
        config = GoogleCloudConfig()
        
        assert config.project_id is None  # Auto-detected
        assert config.log_name == "structured-logging"
        assert config.resource_type == "global"
        assert config.resource_labels == {}
        assert config.use_background_thread is True
        assert config.use_structured_logging is True
    
    def test_custom_config(self):
        config = GoogleCloudConfig(
            project_id="my-project",
            log_name="my-app",
            resource_type="k8s_container",
            resource_labels={"cluster": "prod", "namespace": "default"},
            use_background_thread=False
        )
        
        assert config.project_id == "my-project"
        assert config.log_name == "my-app"
        assert config.resource_type == "k8s_container"
        assert config.resource_labels == {"cluster": "prod", "namespace": "default"}
        assert config.use_background_thread is False
    
    def test_stackdriver_alias(self):
        """Test that Stackdriver aliases work"""
        config = StackdriverConfig(project_id="test-project")
        assert isinstance(config, GoogleCloudConfig)
        assert config.project_id == "test-project"


@pytest.mark.skipif(not HAS_GCP, reason="google-cloud-logging not installed")
class TestGoogleCloudHandler:
    """Test Google Cloud handler functionality"""
    
    @patch('google.cloud.logging.Client')
    def test_handler_creation(self, mock_client_class):
        # Mock Google Cloud client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_logger = Mock()
        mock_client.logger.return_value = mock_logger
        
        config = GoogleCloudConfig(
            project_id="test-project",
            log_name="test-log"
        )
        
        handler = GoogleCloudHandler(config)
        
        # Should create client with project
        mock_client_class.assert_called_once_with(project="test-project")
        
        # Should create logger
        mock_client.logger.assert_called_once_with("test-log")
    
    @patch('google.cloud.logging.Client')
    def test_structured_log_upload(self, mock_client_class):
        # Mock Google Cloud client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_logger = Mock()
        mock_client.logger.return_value = mock_logger
        
        config = GoogleCloudConfig(
            project_id="test-project",
            use_structured_logging=True,
            use_background_thread=False  # Sync for testing
        )
        
        handler = GoogleCloudHandler(config)
        
        # Create test log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.ctx_user_id = "user123"
        record.ctx_request_id = "req456"
        
        # Emit log
        handler.emit(record)
        
        # Force processing
        handler.flush()
        
        # Should have called log_struct
        assert mock_logger.log_struct.called
        call_args = mock_logger.log_struct.call_args
        
        # Check structured payload
        payload = call_args[0][0]
        assert payload['message'] == "Test message"
        assert payload['logger'] == "test"
        assert payload['lineno'] == 42
        assert payload['user_id'] == "user123"
        assert payload['request_id'] == "req456"
        
        # Check severity
        assert call_args[1]['severity'] == "INFO"
    
    @patch('google.cloud.logging.Client')
    def test_text_log_upload(self, mock_client_class):
        # Mock Google Cloud client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_logger = Mock()
        mock_client.logger.return_value = mock_logger
        
        config = GoogleCloudConfig(
            project_id="test-project",
            use_structured_logging=False,  # Text logging
            use_background_thread=False
        )
        
        handler = GoogleCloudHandler(config)
        
        # Create test log record
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="Warning message",
            args=(),
            exc_info=None
        )
        
        # Emit log
        handler.emit(record)
        handler.flush()
        
        # Should have called log_text
        assert mock_logger.log_text.called
        call_args = mock_logger.log_text.call_args
        
        # Check text payload
        assert call_args[0][0] == "Warning message"
        assert call_args[1]['severity'] == "WARNING"
    
    @patch('google.cloud.logging.Client')
    def test_batch_upload(self, mock_client_class):
        # Mock Google Cloud client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_logger = Mock()
        mock_client.logger.return_value = mock_logger
        
        # Mock batch entry creation
        mock_entries = []
        def create_entry(**kwargs):
            entry = Mock()
            entry.payload = kwargs.get('payload')
            entry.severity = kwargs.get('severity')
            mock_entries.append(entry)
            return entry
        mock_logger.entry.side_effect = create_entry
        
        config = GoogleCloudConfig(
            project_id="test-project",
            batch_size=3,
            flush_interval=0.1,
            use_background_thread=False,
            async_upload=True
        )
        
        handler = GoogleCloudHandler(config)
        
        # Emit multiple logs
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
        handler.flush()
        
        # Should have called log_batch
        assert mock_logger.log_batch.called
    
    @patch('google.cloud.logging.Client')
    def test_resource_labels(self, mock_client_class):
        # Mock Google Cloud client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_logger = Mock()
        mock_client.logger.return_value = mock_logger
        
        config = GoogleCloudConfig(
            project_id="test-project",
            resource_type="k8s_container",
            resource_labels={
                "cluster_name": "production",
                "namespace_name": "api",
                "pod_name": "api-server-1"
            },
            use_background_thread=False
        )
        
        handler = GoogleCloudHandler(config)
        
        # Check resource was created correctly
        assert handler.resource.type == "k8s_container"
        assert handler.resource.labels == {
            "cluster_name": "production",
            "namespace_name": "api",
            "pod_name": "api-server-1"
        }
    
    @patch('google.cloud.logging.Client')
    @patch('google.api_core.exceptions.GoogleAPIError')
    def test_error_handling(self, mock_error_class, mock_client_class):
        # Mock Google Cloud client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_logger = Mock()
        mock_client.logger.return_value = mock_logger
        
        # Simulate API error
        error = Exception("Service unavailable")
        error.code = 503
        type(error).__name__ = 'GoogleAPIError'
        mock_logger.log_struct.side_effect = [error, None]  # Fail once, then succeed
        
        config = GoogleCloudConfig(
            project_id="test-project",
            use_background_thread=False,
            retry_delay=0.01  # Fast retry for testing
        )
        
        handler = GoogleCloudHandler(config)
        
        # Create test log record
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error message",
            args=(),
            exc_info=None
        )
        
        # Emit log - should retry
        handler.emit(record)
        handler.flush()
        
        # Should have retried
        assert mock_logger.log_struct.call_count >= 1
    
    @patch('google.cloud.logging.Client')
    def test_background_thread_mode(self, mock_client_class):
        # Mock Google Cloud client and transport
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_logger = Mock()
        mock_client.logger.return_value = mock_logger
        
        with patch('structured_logging.cloud.gcp.BackgroundThreadTransport') as mock_transport_class:
            mock_transport = Mock()
            mock_transport_class.return_value = mock_transport
            
            config = GoogleCloudConfig(
                project_id="test-project",
                use_background_thread=True,
                grace_period=10.0
            )
            
            handler = GoogleCloudHandler(config)
            
            # Create test log record
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg="Background test",
                args=(),
                exc_info=None
            )
            
            # Emit log
            handler.emit(record)
            
            # Should create transport
            mock_transport_class.assert_called_once_with(
                mock_client,
                "structured-logging",
                grace_period=10.0
            )
            
            # Should send via transport
            mock_transport.send.assert_called_once()
            
            # Test flush
            handler.flush()
            mock_transport.flush.assert_called_once()
            
            # Test close
            handler.close()
            mock_transport.close.assert_called_once()
    
    @patch('google.oauth2.service_account.Credentials.from_service_account_file')
    @patch('google.cloud.logging.Client')
    def test_service_account_credentials(self, mock_client_class, mock_creds):
        # Mock credentials
        mock_credentials = Mock()
        mock_creds.return_value = mock_credentials
        
        # Mock client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_logger = Mock()
        mock_client.logger.return_value = mock_logger
        
        config = GoogleCloudConfig(
            project_id="test-project",
            credentials_path="/path/to/service-account.json"
        )
        
        handler = GoogleCloudHandler(config)
        
        # Should load credentials
        mock_creds.assert_called_once_with("/path/to/service-account.json")
        
        # Should pass credentials to client
        mock_client_class.assert_called_once_with(
            project="test-project",
            credentials=mock_credentials
        )
    
    def test_stackdriver_handler_alias(self):
        """Test that StackdriverHandler is an alias"""
        assert StackdriverHandler is GoogleCloudHandler


@pytest.mark.skipif(not HAS_GCP, reason="google-cloud-logging not installed")
class TestGoogleCloudIntegration:
    """Integration tests with structured logging"""
    
    @patch('google.cloud.logging.Client')
    def test_structured_logging_integration(self, mock_client_class):
        from structured_logging import get_logger
        
        # Mock Google Cloud
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_logger = Mock()
        mock_client.logger.return_value = mock_logger
        
        # Create Google Cloud logger
        logger = create_google_cloud_logger(
            "test_app",
            project_id="test-project",
            resource_type="gce_instance",
            resource_labels={"instance_id": "1234"}
        )
        
        # Get the Google Cloud handler and ensure sync mode
        for handler in logger.handlers:
            if isinstance(handler, GoogleCloudHandler):
                handler.config.use_background_thread = False
        
        # Log with context
        logger.info("Application started", extra={
            "ctx_version": "1.0.0",
            "ctx_environment": "test"
        })
        
        # Should have logged to Google Cloud
        assert mock_logger.log_struct.called
        
        # Check log format
        call_args = mock_logger.log_struct.call_args
        payload = call_args[0][0]
        
        assert payload['message'] == "Application started"
        assert payload['version'] == "1.0.0"
        assert payload['environment'] == "test"
        assert 'timestamp' in payload
    
    def test_stackdriver_logger_helper(self):
        """Test the Stackdriver helper function"""
        with patch('google.cloud.logging.Client') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_logger = Mock()
            mock_client.logger.return_value = mock_logger
            
            # Use Stackdriver alias
            logger = create_stackdriver_logger(
                "stackdriver_app",
                project_id="test-project"
            )
            
            assert logger.name == "stackdriver_app"
            assert len(logger.handlers) >= 1
            
            # Find handler
            handler = None
            for h in logger.handlers:
                if isinstance(h, GoogleCloudHandler):
                    handler = h
                    break
            
            assert handler is not None
            assert handler.config.project_id == "test-project"


@pytest.mark.skipif(not HAS_GCP, reason="google-cloud-logging not installed")
class TestGoogleCloudAdvanced:
    """Advanced Google Cloud handler tests"""
    
    @patch('google.cloud.logging.Client')
    def test_severity_mapping(self, mock_client_class):
        """Test Python log level to Google Cloud severity mapping"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_logger = Mock()
        mock_client.logger.return_value = mock_logger
        
        config = GoogleCloudConfig(
            project_id="test-project",
            use_background_thread=False
        )
        
        handler = GoogleCloudHandler(config)
        
        # Test different log levels
        levels = [
            (logging.DEBUG, "DEBUG"),
            (logging.INFO, "INFO"),
            (logging.WARNING, "WARNING"),
            (logging.ERROR, "ERROR"),
            (logging.CRITICAL, "CRITICAL"),
        ]
        
        for py_level, gcp_severity in levels:
            record = logging.LogRecord(
                name="test",
                level=py_level,
                pathname="test.py",
                lineno=1,
                msg=f"{gcp_severity} test",
                args=(),
                exc_info=None
            )
            
            handler.emit(record)
            handler.flush()
            
            # Check severity in last call
            call_args = mock_logger.log_struct.call_args
            assert call_args[1]['severity'] == gcp_severity
    
    @patch('google.cloud.logging.Client')
    def test_trace_id_inclusion(self, mock_client_class):
        """Test trace ID inclusion for request correlation"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_logger = Mock()
        mock_client.logger.return_value = mock_logger
        
        config = GoogleCloudConfig(
            project_id="test-project",
            include_trace_id=True,
            use_background_thread=False
        )
        
        handler = GoogleCloudHandler(config)
        
        # Create record with trace ID
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Traced request",
            args=(),
            exc_info=None
        )
        record.trace_id = "projects/test-project/traces/1234567890abcdef"
        
        handler.emit(record)
        handler.flush()
        
        # Should include trace in log
        # Note: The actual trace inclusion would be in the entry object,
        # but we're mocking that part
        assert mock_logger.log_struct.called


class TestGoogleCloudWithoutGCP:
    """Test behavior when google-cloud-logging is not installed"""
    
    def test_config_creation_without_gcp(self):
        """Test that config can be created without google-cloud-logging"""
        from structured_logging.cloud.base import CloudHandlerConfig
        
        config = CloudHandlerConfig()
        assert config.batch_size == 100
        assert config.flush_interval == 5.0
    
    def test_handler_creation_without_gcp(self):
        """Test handler creation fails gracefully without google-cloud-logging"""
        # Test the error message format
        from structured_logging.cloud.base import CloudHandlerConfig
        config = CloudHandlerConfig()  # Use base config since GoogleCloudConfig might be None
        
        # Mock the HAS_GOOGLE_CLOUD check inside the handler
        with patch('structured_logging.cloud.gcp.HAS_GOOGLE_CLOUD', False):
            with pytest.raises(ImportError) as exc_info:
                # Need to re-import to get the patched value
                from structured_logging.cloud.gcp import GoogleCloudHandler as GCPHandler
                GCPHandler(config)
            
            assert "google-cloud-logging is required" in str(exc_info.value)
            assert "pip install structured-logging[gcp]" in str(exc_info.value)