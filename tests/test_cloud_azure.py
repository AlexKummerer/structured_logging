"""
Tests for Azure Monitor integration
"""

import base64
import gzip
import json
import logging
import time
from unittest.mock import Mock, patch

import pytest

# Skip all tests if azure dependencies are not installed
try:
    import requests  # noqa: F401

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import azure.monitor.ingestion  # noqa: F401

    HAS_AZURE_MONITOR = True
except ImportError:
    HAS_AZURE_MONITOR = False

# Import Azure classes conditionally
if HAS_REQUESTS or HAS_AZURE_MONITOR:
    from structured_logging.cloud import (
        ApplicationInsightsConfig,
        ApplicationInsightsHandler,
        AzureMonitorConfig,
        AzureMonitorHandler,
    )
    from structured_logging.cloud.utils import (
        create_application_insights_logger,
        create_azure_monitor_logger,
    )
else:
    # Create dummy classes for tests
    AzureMonitorConfig = None
    AzureMonitorHandler = None
    ApplicationInsightsConfig = None
    ApplicationInsightsHandler = None


@pytest.mark.skipif(not HAS_REQUESTS, reason="requests not installed")
class TestAzureMonitorConfig:
    """Test Azure Monitor configuration"""

    def test_workspace_config(self):
        """Test configuration with workspace ID and key"""
        config = AzureMonitorConfig(
            workspace_id="12345678-1234-1234-1234-123456789012",
            workspace_key="test_key_123",
        )

        assert config.workspace_id == "12345678-1234-1234-1234-123456789012"
        assert config.workspace_key == "test_key_123"
        assert config.log_type == "StructuredLogs"
        assert config.api_version == "2016-04-01"
        assert config.use_compression is True

    def test_dce_config(self):
        """Test configuration with DCE endpoint"""
        config = AzureMonitorConfig(
            dce_endpoint="https://my-dce.eastus.ingest.monitor.azure.com",
            dcr_immutable_id="dcr-1234567890abcdef",
            stream_name="Custom-MyLogs",
        )

        assert config.dce_endpoint == "https://my-dce.eastus.ingest.monitor.azure.com"
        assert config.dcr_immutable_id == "dcr-1234567890abcdef"
        assert config.stream_name == "Custom-MyLogs"

    def test_appinsights_config(self):
        """Test configuration with Application Insights"""
        config = AzureMonitorConfig(
            instrumentation_key="12345678-1234-1234-1234-123456789012",
            cloud_role_name="MyWebApp",
        )

        assert config.instrumentation_key == "12345678-1234-1234-1234-123456789012"
        assert config.cloud_role_name == "MyWebApp"
        assert config.include_cloud_role is True

    def test_invalid_config(self):
        """Test configuration validation"""
        with pytest.raises(ValueError) as exc_info:
            AzureMonitorConfig()  # No authentication method

        assert "Azure Monitor requires either" in str(exc_info.value)

    def test_appinsights_alias(self):
        """Test Application Insights config alias"""
        config = ApplicationInsightsConfig(instrumentation_key="test-key")

        assert isinstance(config, AzureMonitorConfig)
        assert config.log_type == "AppTraces"
        assert config.time_field == "time"


@pytest.mark.skipif(not HAS_REQUESTS, reason="requests not installed")
class TestAzureMonitorHandler:
    """Test Azure Monitor handler functionality"""

    @patch.dict("os.environ", {"WEBSITE_SITE_NAME": "test-app-service"})
    def test_handler_creation_workspace(self):
        """Test handler creation with workspace configuration"""
        config = AzureMonitorConfig(
            workspace_id="test-workspace", workspace_key="test-key"
        )

        handler = AzureMonitorHandler(config)

        assert handler.ingestion_type == "workspace"
        assert handler.workspace_url == (
            "https://test-workspace.ods.opinsights.azure.com"
            "/api/logs?api-version=2016-04-01"
        )
        assert handler.cloud_role_name == "test-app-service"

    @patch("structured_logging.cloud.azure.DefaultAzureCredential")
    @patch("structured_logging.cloud.azure.LogsIngestionClient")
    def test_handler_creation_dce(self, mock_client_class, mock_cred_class):
        """Test handler creation with DCE configuration"""
        if not HAS_AZURE_MONITOR:
            pytest.skip("azure-monitor-ingestion not installed")

        mock_credential = Mock()
        mock_cred_class.return_value = mock_credential

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        config = AzureMonitorConfig(
            dce_endpoint="https://test-dce.eastus.ingest.monitor.azure.com",
            dcr_immutable_id="dcr-test",
        )

        handler = AzureMonitorHandler(config)

        assert handler.ingestion_type == "dce"
        assert handler.client is not None

        # Should create credential and client
        mock_cred_class.assert_called_once()
        mock_client_class.assert_called_once_with(
            endpoint="https://test-dce.eastus.ingest.monitor.azure.com",
            credential=mock_credential,
            logging_enable=False,
        )

    def test_handler_creation_appinsights(self):
        """Test handler creation with Application Insights"""
        config = AzureMonitorConfig(instrumentation_key="test-ikey")

        handler = AzureMonitorHandler(config)

        assert handler.ingestion_type == "appinsights"
        assert (
            handler.appinsights_url == "https://dc.services.visualstudio.com/v2/track"
        )

    @patch("requests.post")
    def test_workspace_log_upload(self, mock_post):
        """Test log upload via workspace API"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        config = AzureMonitorConfig(
            workspace_id="test-workspace",
            workspace_key=base64.b64encode(b"test-key").decode("utf-8"),
            log_type="TestLogs",
            async_upload=False,
        )

        handler = AzureMonitorHandler(config)

        # Create test log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.ctx_user_id = "user123"
        record.ctx_session_id = "session456"

        # Emit log
        handler.emit(record)
        handler.flush()

        # Should have called requests.post
        assert mock_post.called
        call_args = mock_post.call_args

        # Check URL
        assert call_args[0][0] == handler.workspace_url

        # Check headers
        headers = call_args[1]["headers"]
        assert headers["Log-Type"] == "TestLogs"
        assert headers["x-ms-date"] is not None
        assert headers["Authorization"].startswith("SharedKey")

        # Check body
        body = call_args[1]["data"]
        if isinstance(body, bytes) and headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)

        logs = json.loads(body.decode("utf-8"))
        assert len(logs) == 1
        assert logs[0]["Message"] == "Test message"
        assert logs[0]["SeverityLevel"] == 1  # INFO
        assert logs[0]["Properties"]["user_id"] == "user123"
        assert logs[0]["Properties"]["session_id"] == "session456"

    @patch("requests.post")
    def test_appinsights_log_upload(self, mock_post):
        """Test log upload via Application Insights"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        config = ApplicationInsightsConfig(
            instrumentation_key="test-ikey",
            cloud_role_name="TestApp",
            async_upload=False,
        )

        handler = ApplicationInsightsHandler(config)

        # Create test log record
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=100,
            msg="Error occurred",
            args=(),
            exc_info=None,
        )
        record.ctx_error_code = "E001"

        # Emit log
        handler.emit(record)
        handler.flush()

        # Should have called requests.post
        assert mock_post.called
        call_args = mock_post.call_args

        # Check URL
        assert call_args[0][0] == "https://dc.services.visualstudio.com/v2/track"

        # Check body
        telemetry = call_args[1]["json"]
        assert len(telemetry) == 1

        item = telemetry[0]
        assert item["name"] == "Microsoft.ApplicationInsights.Trace"
        assert item["iKey"] == "test-ikey"
        assert item["tags"]["ai.cloud.role"] == "TestApp"

        data = item["data"]["baseData"]
        assert data["message"] == "Error occurred"
        assert data["severityLevel"] == 3  # ERROR
        assert data["properties"]["error_code"] == "E001"

    @patch("structured_logging.cloud.azure.LogsIngestionClient")
    @patch("structured_logging.cloud.azure.DefaultAzureCredential")
    def test_dce_log_upload(self, mock_cred_class, mock_client_class):
        """Test log upload via DCE"""
        if not HAS_AZURE_MONITOR:
            pytest.skip("azure-monitor-ingestion not installed")

        mock_credential = Mock()
        mock_cred_class.return_value = mock_credential

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        config = AzureMonitorConfig(
            dce_endpoint="https://test-dce.eastus.ingest.monitor.azure.com",
            dcr_immutable_id="dcr-test",
            stream_name="Custom-TestLogs",
            async_upload=False,
        )

        handler = AzureMonitorHandler(config)

        # Create test log record
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=50,
            msg="Warning message",
            args=(),
            exc_info=None,
        )

        # Emit log
        handler.emit(record)
        handler.flush()

        # Should have called client.upload
        mock_client.upload.assert_called_once()
        call_args = mock_client.upload.call_args

        assert call_args[1]["rule_id"] == "dcr-test"
        assert call_args[1]["stream_name"] == "Custom-TestLogs"

        logs = call_args[1]["logs"]
        assert len(logs) == 1
        assert logs[0]["Message"] == "Warning message"
        assert logs[0]["SeverityLevel"] == 2  # WARNING

    @patch("requests.post")
    def test_batch_upload(self, mock_post):
        """Test batch log upload"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        config = AzureMonitorConfig(
            workspace_id="test-workspace",
            workspace_key=base64.b64encode(b"test-key").decode("utf-8"),
            batch_size=3,
            flush_interval=0.1,
            async_upload=True,
        )

        handler = AzureMonitorHandler(config)

        # Emit multiple logs
        for i in range(5):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=i,
                msg=f"Test message {i}",
                args=(),
                exc_info=None,
            )
            handler.emit(record)

        # Wait for batch processing
        time.sleep(0.2)
        handler.flush()

        # Should have batched uploads
        assert mock_post.called

    @patch("requests.post")
    def test_retry_on_error(self, mock_post):
        """Test retry logic on server errors"""
        # First call fails, second succeeds
        mock_response_fail = Mock()
        mock_response_fail.status_code = 503
        mock_response_fail.text = "Service Unavailable"

        mock_response_success = Mock()
        mock_response_success.status_code = 200

        mock_post.side_effect = [mock_response_fail, mock_response_success]

        config = AzureMonitorConfig(
            workspace_id="test-workspace",
            workspace_key=base64.b64encode(b"test-key").decode("utf-8"),
            retry_delay=0.01,
            async_upload=False,
        )

        handler = AzureMonitorHandler(config)

        # Create test log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test retry",
            args=(),
            exc_info=None,
        )

        # Emit log
        handler.emit(record)
        handler.flush()

        # Should have retried
        assert mock_post.call_count == 2

    def test_cloud_role_detection(self):
        """Test cloud role auto-detection"""

        # Test App Service
        with patch.dict("os.environ", {"WEBSITE_SITE_NAME": "my-app-service"}):
            config = AzureMonitorConfig(workspace_id="test", workspace_key="test")
            handler = AzureMonitorHandler(config)
            assert handler.cloud_role_name == "my-app-service"

        # Test Functions
        with patch.dict("os.environ", {"FUNCTIONS_WORKER_RUNTIME": "python"}):
            config = AzureMonitorConfig(workspace_id="test", workspace_key="test")
            handler = AzureMonitorHandler(config)
            assert handler.cloud_role_name == "python"

        # Test Container Apps
        with patch.dict("os.environ", {"CONTAINER_APP_NAME": "my-container-app"}):
            config = AzureMonitorConfig(workspace_id="test", workspace_key="test")
            handler = AzureMonitorHandler(config)
            assert handler.cloud_role_name == "my-container-app"

        # Test custom override
        config = AzureMonitorConfig(
            workspace_id="test", workspace_key="test", cloud_role_name="CustomRole"
        )
        handler = AzureMonitorHandler(config)
        assert handler.cloud_role_name == "CustomRole"


@pytest.mark.skipif(not HAS_REQUESTS, reason="requests not installed")
class TestAzureMonitorIntegration:
    """Integration tests with structured logging"""

    @patch("requests.post")
    def test_structured_logging_integration(self, mock_post):

        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Create Azure Monitor logger
        logger = create_azure_monitor_logger(
            "test_app",
            workspace_id="test-workspace",
            workspace_key=base64.b64encode(b"test-key").decode("utf-8"),
            log_type="ApplicationLogs",
        )

        # Get the Azure handler and ensure sync mode
        for handler in logger.handlers:
            if isinstance(handler, AzureMonitorHandler):
                handler.config.async_upload = False

        # Log with context
        logger.info(
            "Application started",
            extra={"ctx_version": "1.0.0", "ctx_environment": "test"},
        )

        # Should have logged to Azure Monitor
        assert mock_post.called

        # Check log format
        body = mock_post.call_args[1]["data"]
        if mock_post.call_args[1]["headers"].get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)

        logs = json.loads(body.decode("utf-8"))
        assert len(logs) == 1

        log = logs[0]
        assert log["Message"] == "Application started"
        assert log["Properties"]["version"] == "1.0.0"
        assert log["Properties"]["environment"] == "test"

    @patch("requests.post")
    def test_application_insights_helper(self, mock_post):
        """Test Application Insights helper function"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Create Application Insights logger
        logger = create_application_insights_logger(
            "insights_app",
            instrumentation_key="test-ikey",
            cloud_role_name="TestService",
        )

        assert logger.name == "insights_app"
        assert len(logger.handlers) >= 1

        # Find handler
        handler = None
        for h in logger.handlers:
            if isinstance(h, ApplicationInsightsHandler):
                handler = h
                break

        assert handler is not None
        assert handler.config.instrumentation_key == "test-ikey"
        assert handler.config.cloud_role_name == "TestService"


@pytest.mark.skipif(not HAS_REQUESTS, reason="requests not installed")
class TestAzureMonitorAdvanced:
    """Advanced Azure Monitor handler tests"""

    @patch("requests.post")
    def test_severity_mapping(self, mock_post):
        """Test Python log level to Azure severity mapping"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        config = AzureMonitorConfig(
            workspace_id="test",
            workspace_key=base64.b64encode(b"test-key").decode("utf-8"),
            async_upload=False,
        )

        handler = AzureMonitorHandler(config)

        # Test different log levels
        levels = [
            (logging.DEBUG, 0),  # Verbose
            (logging.INFO, 1),  # Information
            (logging.WARNING, 2),  # Warning
            (logging.ERROR, 3),  # Error
            (logging.CRITICAL, 4),  # Critical
        ]

        for py_level, azure_severity in levels:
            record = logging.LogRecord(
                name="test",
                level=py_level,
                pathname="test.py",
                lineno=1,
                msg=f"Severity {azure_severity} test",
                args=(),
                exc_info=None,
            )

            handler.emit(record)
            handler.flush()

            # Check severity in last call
            body = mock_post.call_args[1]["data"]
            if mock_post.call_args[1]["headers"].get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)

            logs = json.loads(body.decode("utf-8"))
            assert logs[0]["SeverityLevel"] == azure_severity

    @patch("requests.post")
    def test_compression(self, mock_post):
        """Test log compression"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        config = AzureMonitorConfig(
            workspace_id="test",
            workspace_key=base64.b64encode(b"test-key").decode("utf-8"),
            use_compression=True,
            async_upload=False,
        )

        handler = AzureMonitorHandler(config)

        # Create large log
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.ctx_large_data = "x" * 1000

        handler.emit(record)
        handler.flush()

        # Should have compressed
        headers = mock_post.call_args[1]["headers"]
        assert headers.get("Content-Encoding") == "gzip"
        assert headers["Content-Type"] == "application/octet-stream"

        # Body should be compressed
        body = mock_post.call_args[1]["data"]
        assert isinstance(body, bytes)

        # Decompress and verify
        decompressed = gzip.decompress(body)
        logs = json.loads(decompressed.decode("utf-8"))
        assert logs[0]["Properties"]["large_data"] == "x" * 1000

    @patch("structured_logging.cloud.azure.LogsIngestionClient")
    def test_service_principal_auth(self, mock_client_class):
        """Test service principal authentication"""
        if not HAS_AZURE_MONITOR:
            pytest.skip("azure-monitor-ingestion not installed")

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock the ClientSecretCredential import
        with patch("azure.identity.ClientSecretCredential") as mock_cred_class:
            mock_credential = Mock()
            mock_cred_class.return_value = mock_credential

            config = AzureMonitorConfig(
                dce_endpoint="https://test-dce.eastus.ingest.monitor.azure.com",
                dcr_immutable_id="dcr-test",
                tenant_id="test-tenant",
                client_id="test-client",
                client_secret="test-secret",
            )

            AzureMonitorHandler(config)

            # Should use ClientSecretCredential
            mock_cred_class.assert_called_once_with(
                tenant_id="test-tenant",
                client_id="test-client",
                client_secret="test-secret",
            )

            # Should pass credential to client
            mock_client_class.assert_called_once_with(
                endpoint="https://test-dce.eastus.ingest.monitor.azure.com",
                credential=mock_credential,
                logging_enable=False,
            )


class TestAzureMonitorWithoutDeps:
    """Test behavior when Azure dependencies are not installed"""

    def test_config_creation_without_deps(self):
        """Test that config can be created without dependencies"""
        from structured_logging.cloud.base import CloudHandlerConfig

        config = CloudHandlerConfig()
        assert config.batch_size == 100
        assert config.flush_interval == 5.0

    def test_handler_creation_without_requests(self):
        """Test handler creation fails gracefully without requests"""
        if HAS_REQUESTS:
            pytest.skip("requests is installed")

        from structured_logging.cloud.base import CloudHandlerConfig

        config = CloudHandlerConfig()
        config.workspace_id = "test"
        config.workspace_key = "test"

        with pytest.raises(ImportError) as exc_info:
            from structured_logging.cloud.azure import AzureMonitorHandler as AzHandler

            AzHandler(config)

        assert "requests is required" in str(exc_info.value)
        assert "pip install structured-logging[azure]" in str(exc_info.value)

