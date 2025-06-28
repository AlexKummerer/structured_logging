"""
Tests for OpenTelemetry integration
"""

import logging
import time
from unittest.mock import Mock, patch

import pytest

# Skip all tests if opentelemetry dependencies are not installed
try:
    import opentelemetry  # noqa: F401

    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False

# Import OpenTelemetry classes conditionally
if HAS_OPENTELEMETRY:
    from structured_logging.integrations import (
        OpenTelemetryConfig,
        OpenTelemetryHandler,
        OpenTelemetryLogger,
        add_otel_handler_to_logger,
        configure_otel_logging,
        create_otel_logger,
        logged_span,
    )
else:
    # Create dummy classes for tests
    OpenTelemetryConfig = None
    OpenTelemetryHandler = None
    OpenTelemetryLogger = None


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="opentelemetry not installed")
class TestOpenTelemetryConfig:
    """Test OpenTelemetry configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = OpenTelemetryConfig()

        assert config.enable_trace_correlation is True
        assert config.trace_id_field == "trace_id"
        assert config.span_id_field == "span_id"
        assert config.include_resource_attributes is True
        assert config.include_span_attributes is True
        assert config.resource_attribute_prefix == "resource."
        assert config.span_attribute_prefix == "span."
        assert config.max_span_attributes == 20
        assert config.enable_instrumentation is True
        assert config.max_attribute_length == 1000

    def test_custom_config(self):
        """Test custom configuration"""
        config = OpenTelemetryConfig(
            enable_trace_correlation=False,
            trace_id_field="custom_trace_id",
            span_id_field="custom_span_id",
            resource_attribute_prefix="res_",
            span_attribute_prefix="span_",
            max_span_attributes=10,
            max_attribute_length=500,
            only_when_tracing=True,
            minimum_log_level="ERROR",
        )

        assert config.enable_trace_correlation is False
        assert config.trace_id_field == "custom_trace_id"
        assert config.span_id_field == "custom_span_id"
        assert config.resource_attribute_prefix == "res_"
        assert config.span_attribute_prefix == "span_"
        assert config.max_span_attributes == 10
        assert config.max_attribute_length == 500
        assert config.only_when_tracing is True
        assert config.minimum_log_level == "ERROR"

    def test_attribute_mapping(self):
        """Test attribute mapping configuration"""
        custom_mapping = {"old_field": "new_field", "legacy": "modern"}
        exclude_list = ["sensitive_data", "internal_field"]

        config = OpenTelemetryConfig(
            attribute_mapping=custom_mapping, exclude_attributes=exclude_list
        )

        assert config.attribute_mapping == custom_mapping
        assert config.exclude_attributes == exclude_list


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="opentelemetry not installed")
class TestOpenTelemetryHandler:
    """Test OpenTelemetry handler functionality"""

    @patch("structured_logging.integrations.opentelemetry.trace")
    def test_handler_creation(self, mock_trace):
        """Test handler creation and initialization"""
        # Mock tracer provider and resource
        mock_tracer_provider = Mock()
        mock_resource = Mock()
        mock_resource.attributes = {
            "service.name": "test-service",
            "service.version": "1.0.0",
            "deployment.environment": "test",
        }
        mock_tracer_provider.resource = mock_resource
        mock_trace.get_tracer_provider.return_value = mock_tracer_provider

        config = OpenTelemetryConfig()
        handler = OpenTelemetryHandler(config)

        assert handler.config == config
        assert len(handler._resource_attributes) > 0
        assert "resource.service.name" in handler._resource_attributes
        assert handler._resource_attributes["resource.service.name"] == "test-service"

    @patch("structured_logging.integrations.opentelemetry.trace")
    def test_trace_correlation(self, mock_trace):
        """Test trace and span ID injection"""
        # Mock span context
        mock_span_context = Mock()
        mock_span_context.trace_id = 0x12345678901234567890123456789012
        mock_span_context.span_id = 0x1234567890123456
        mock_span_context.trace_flags = 1
        mock_span_context.is_valid = True

        mock_span = Mock()
        mock_span.get_span_context.return_value = mock_span_context
        mock_trace.get_current_span.return_value = mock_span

        config = OpenTelemetryConfig()
        handler = OpenTelemetryHandler(config)

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

        # Process the record
        handler.emit(record)

        # Check trace correlation fields
        assert hasattr(record, "trace_id")
        assert hasattr(record, "span_id")
        assert hasattr(record, "trace_flags")
        assert record.trace_id == "12345678901234567890123456789012"
        assert record.span_id == "1234567890123456"
        assert record.trace_flags == 1

    @patch("structured_logging.integrations.opentelemetry.trace")
    def test_resource_attributes(self, mock_trace):
        """Test resource attribute injection"""
        # Mock tracer provider with resource
        mock_tracer_provider = Mock()
        mock_resource = Mock()
        mock_resource.attributes = {
            "service.name": "test-service",
            "service.version": "1.0.0",
            "deployment.environment": "production",
            "host.name": "test-host",
        }
        mock_tracer_provider.resource = mock_resource
        mock_trace.get_tracer_provider.return_value = mock_tracer_provider

        # Mock current span
        mock_span = Mock()
        mock_span.get_span_context.return_value = Mock()
        mock_trace.get_current_span.return_value = mock_span

        config = OpenTelemetryConfig(include_resource_attributes=True)
        handler = OpenTelemetryHandler(config)

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

        # Process the record
        handler.emit(record)

        # Check resource attributes
        assert hasattr(record, "resource.service.name")
        assert hasattr(record, "resource.service.version")
        assert hasattr(record, "resource.deployment.environment")
        assert hasattr(record, "resource.host.name")
        assert getattr(record, "resource.service.name") == "test-service"
        assert getattr(record, "resource.service.version") == "1.0.0"

    @patch("structured_logging.integrations.opentelemetry.trace")
    def test_span_attributes(self, mock_trace):
        """Test span attribute injection"""
        # Mock span with attributes
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_span.attributes = {
            "http.method": "GET",
            "http.url": "https://api.example.com/users",
            "http.status_code": 200,
            "custom.attribute": "test-value",
        }
        mock_span.get_span_context.return_value = Mock()
        mock_trace.get_current_span.return_value = mock_span

        config = OpenTelemetryConfig(include_span_attributes=True)
        handler = OpenTelemetryHandler(config)

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

        # Process the record
        handler.emit(record)

        # Check span attributes
        assert hasattr(record, "span.http.method")
        assert hasattr(record, "span.http.url")
        assert hasattr(record, "span.http.status_code")
        assert hasattr(record, "span.custom.attribute")
        assert getattr(record, "span.http.method") == "GET"
        assert getattr(record, "span.http.status_code") == 200

    @patch("structured_logging.integrations.opentelemetry.trace")
    def test_attribute_mapping(self, mock_trace):
        """Test custom attribute mapping"""
        mock_trace.get_current_span.return_value = Mock()

        config = OpenTelemetryConfig(
            map_log_attributes=True,
            attribute_mapping={"levelname": "log_level", "pathname": "file_path"},
            exclude_attributes=["args", "created"],
        )
        handler = OpenTelemetryHandler(config)

        # Create test log record with attributes to map
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="/path/to/test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.args = ("should", "be", "excluded")
        record.created = time.time()

        # Process the record
        handler.emit(record)

        # Check mapped attributes
        assert hasattr(record, "log_level")
        assert hasattr(record, "file_path")
        assert getattr(record, "log_level") == "WARNING"
        assert getattr(record, "file_path") == "/path/to/test.py"

        # Check excluded attributes are removed
        assert not hasattr(record, "args")
        assert not hasattr(record, "created")

    @patch("structured_logging.integrations.opentelemetry.trace")
    def test_only_when_tracing(self, mock_trace):
        """Test only_when_tracing configuration"""
        # Mock no active span
        mock_trace.get_current_span.return_value = Mock()
        mock_trace.get_current_span.return_value.get_span_context.return_value = None

        config = OpenTelemetryConfig(only_when_tracing=True)
        handler = OpenTelemetryHandler(config)

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

        # Process the record
        handler.emit(record)

        # Should not add any OTel fields when not tracing
        assert not hasattr(record, "trace_id")
        assert not hasattr(record, "span_id")

    @patch("structured_logging.integrations.opentelemetry.trace")
    def test_minimum_log_level(self, mock_trace):
        """Test minimum log level filtering"""
        mock_trace.get_current_span.return_value = Mock()

        config = OpenTelemetryConfig(minimum_log_level="ERROR")
        handler = OpenTelemetryHandler(config)

        # Create DEBUG log record (below minimum)
        debug_record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="test.py",
            lineno=42,
            msg="Debug message",
            args=(),
            exc_info=None,
        )

        # Create ERROR log record (at minimum)
        error_record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=42,
            msg="Error message",
            args=(),
            exc_info=None,
        )

        # Process both records
        handler.emit(debug_record)
        handler.emit(error_record)

        # DEBUG record should not be processed
        assert not hasattr(debug_record, "trace_id")

        # ERROR record should be processed
        # (assuming mock span context is valid)

    def test_value_serialization(self):
        """Test value serialization with length limits"""
        config = OpenTelemetryConfig(max_attribute_length=10)
        handler = OpenTelemetryHandler(config)

        # Test string truncation
        long_string = "a" * 50
        result = handler._serialize_value(long_string)
        assert len(result) == 13  # 10 chars + "..."
        assert result.endswith("...")

        # Test list limitation
        long_list = list(range(20))
        result = handler._serialize_value(long_list)
        assert len(result) == 10

        # Test dict limitation
        large_dict = {f"key_{i}": f"value_{i}" for i in range(20)}
        result = handler._serialize_value(large_dict)
        assert len(result) == 10


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="opentelemetry not installed")
class TestOpenTelemetryLogger:
    """Test OpenTelemetryLogger functionality"""

    @patch("structured_logging.integrations.opentelemetry.trace")
    def test_logger_creation(self, mock_trace):
        """Test logger creation and configuration"""
        config = OpenTelemetryConfig()
        logger = OpenTelemetryLogger("test_logger", config)

        assert logger.name == "test_logger"
        assert logger.otel_config == config
        assert logger.logger is not None
        assert logger.otel_handler is not None

    @patch("structured_logging.integrations.opentelemetry.trace")
    def test_logging_methods(self, mock_trace):
        """Test all logging level methods"""
        # Mock span
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_span.name = "test-span"
        mock_span.kind = "INTERNAL"
        mock_trace.get_current_span.return_value = mock_span

        logger = OpenTelemetryLogger("test_logger")

        # Test all logging methods
        logger.debug("Debug message", extra_field="debug_value")
        logger.info("Info message", extra_field="info_value")
        logger.warning("Warning message", extra_field="warning_value")
        logger.error("Error message", extra_field="error_value")
        logger.critical("Critical message", extra_field="critical_value")

        # Test exception logging
        try:
            raise ValueError("Test exception")
        except ValueError:
            logger.exception("Exception occurred", extra_field="exception_value")

    @patch("structured_logging.integrations.opentelemetry.trace")
    def test_span_context_injection(self, mock_trace):
        """Test automatic span context injection"""
        # Mock active span
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_span.name = "http-request"
        mock_span.kind = "SERVER"
        mock_trace.get_current_span.return_value = mock_span

        logger = OpenTelemetryLogger("test_logger")

        # Capture the log call to verify span context
        with patch.object(logger.logger, "info") as mock_log:
            logger.info("Request processed", user_id="12345")

            # Verify span information was added
            mock_log.assert_called_once()
            call_kwargs = mock_log.call_args[1]["extra"]
            assert "span_name" in call_kwargs
            assert "span_kind" in call_kwargs
            assert call_kwargs["span_name"] == "http-request"
            assert call_kwargs["user_id"] == "12345"


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="opentelemetry not installed")
class TestOpenTelemetryHelpers:
    """Test helper functions"""

    def test_create_otel_logger(self):
        """Test create_otel_logger helper function"""
        logger = create_otel_logger(
            "helper_test",
            enable_trace_correlation=True,
            include_resource_attributes=False,
            resource_attribute_prefix="custom_",
        )

        assert isinstance(logger, OpenTelemetryLogger)
        assert logger.name == "helper_test"
        assert logger.otel_config.enable_trace_correlation is True
        assert logger.otel_config.include_resource_attributes is False
        assert logger.otel_config.resource_attribute_prefix == "custom_"

    def test_add_otel_handler_to_logger(self):
        """Test adding OTel handler to existing logger"""
        existing_logger = logging.getLogger("existing_logger")
        config = OpenTelemetryConfig(enable_trace_correlation=False)

        handler = add_otel_handler_to_logger(existing_logger, config)

        assert isinstance(handler, OpenTelemetryHandler)
        assert handler in existing_logger.handlers
        assert handler.config.enable_trace_correlation is False

    def test_configure_otel_logging(self):
        """Test service configuration helper"""
        config = configure_otel_logging(
            service_name="my-service",
            service_version="2.1.0",
            environment="production",
            enable_trace_correlation=False,
        )

        assert isinstance(config, OpenTelemetryConfig)
        assert config.attribute_mapping["service"] == "my-service"
        assert config.attribute_mapping["version"] == "2.1.0"
        assert config.attribute_mapping["environment"] == "production"
        assert config.enable_trace_correlation is False


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="opentelemetry not installed")
class TestLoggingSpan:
    """Test LoggingSpan context manager"""

    @patch("structured_logging.integrations.opentelemetry.trace")
    def test_logging_span_success(self, mock_trace):
        """Test successful span with logging"""
        # Mock tracer and span
        mock_tracer = Mock()
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_tracer.start_span.return_value = mock_span
        mock_trace.get_tracer.return_value = mock_tracer

        logger = OpenTelemetryLogger("test_logger")

        # Mock logger methods to capture calls
        with patch.object(logger, "info") as mock_info:
            with logged_span(logger, "test-operation"):
                pass

            # Should log start and completion
            assert mock_info.call_count == 2
            start_call = mock_info.call_args_list[0]
            end_call = mock_info.call_args_list[1]

            assert "Started span: test-operation" in start_call[0][0]
            assert "Completed span: test-operation" in end_call[0][0]

        # Verify span lifecycle
        mock_tracer.start_span.assert_called_once_with("test-operation")
        mock_span.set_status.assert_called_once()
        mock_span.end.assert_called_once()

    @patch("structured_logging.integrations.opentelemetry.trace")
    def test_logging_span_error(self, mock_trace):
        """Test span with error and logging"""
        # Mock tracer and span
        mock_tracer = Mock()
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_tracer.start_span.return_value = mock_span
        mock_trace.get_tracer.return_value = mock_tracer

        logger = OpenTelemetryLogger("test_logger")

        # Mock logger methods
        with (
            patch.object(logger, "info") as mock_info,
            patch.object(logger, "error") as mock_error,
        ):
            try:
                with logged_span(logger, "failing-operation"):
                    raise ValueError("Something went wrong")
            except ValueError:
                pass

            # Should log start and error
            mock_info.assert_called_once()  # Start
            mock_error.assert_called_once()  # Error

            error_call = mock_error.call_args
            assert "Span failed: failing-operation" in error_call[0][0]

        # Verify error status was set
        mock_span.set_status.assert_called_once()
        status_call = mock_span.set_status.call_args[0][0]
        assert hasattr(status_call, "status_code")


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="opentelemetry not installed")
class TestOpenTelemetryIntegration:
    """Integration tests with structured logging"""

    @patch("structured_logging.integrations.opentelemetry.trace")
    def test_structured_logging_integration(self, mock_trace):
        """Test integration with structured logging system"""
        # Mock OTel components
        mock_span_context = Mock()
        mock_span_context.trace_id = 0x12345678901234567890123456789012
        mock_span_context.span_id = 0x1234567890123456
        mock_span_context.trace_flags = 1
        mock_span_context.is_valid = True

        mock_span = Mock()
        mock_span.get_span_context.return_value = mock_span_context
        mock_span.is_recording.return_value = True
        mock_span.attributes = {"http.method": "POST", "http.status_code": 201}
        mock_trace.get_current_span.return_value = mock_span

        # Create logger with structured logging
        from structured_logging import get_logger

        base_logger = get_logger("integration_test")
        config = OpenTelemetryConfig()
        otel_handler = OpenTelemetryHandler(config)
        base_logger.addHandler(otel_handler)

        # Log with context
        base_logger.info(
            "User created",
            extra={"ctx_user_id": "user_123", "ctx_email": "test@example.com"},
        )

        # The log should be enriched with OTel data
        # (In real test, we'd capture the log output and verify)

    @patch("structured_logging.integrations.opentelemetry.trace")
    def test_performance_with_high_volume(self, mock_trace):
        """Test performance with high volume logging"""
        # Mock minimal OTel setup
        mock_trace.get_current_span.return_value = Mock()

        config = OpenTelemetryConfig()
        handler = OpenTelemetryHandler(config)

        # Log many records
        start_time = time.time()
        for i in range(1000):
            record = logging.LogRecord(
                name="perf_test",
                level=logging.INFO,
                pathname="test.py",
                lineno=i,
                msg=f"Performance test {i}",
                args=(),
                exc_info=None,
            )
            handler.emit(record)

        duration = time.time() - start_time
        # Should handle 1000 logs in reasonable time (< 1 second)
        assert duration < 1.0


class TestOpenTelemetryWithoutDeps:
    """Test behavior when OpenTelemetry dependencies are not installed"""

    def test_import_without_opentelemetry(self):
        """Test that modules can be imported without OpenTelemetry"""
        # This test runs when OpenTelemetry is not installed
        if HAS_OPENTELEMETRY:
            pytest.skip("OpenTelemetry is installed")

        # Should be able to import the module without errors
        from structured_logging.integrations import opentelemetry

        assert not opentelemetry.HAS_OPENTELEMETRY

    def test_handler_creation_without_opentelemetry(self):
        """Test handler creation fails gracefully without OpenTelemetry"""
        if HAS_OPENTELEMETRY:
            pytest.skip("OpenTelemetry is installed")

        from structured_logging.integrations.opentelemetry import (
            OpenTelemetryConfig,
            OpenTelemetryHandler,
        )

        config = OpenTelemetryConfig()

        with pytest.raises(ImportError) as exc_info:
            OpenTelemetryHandler(config)

        assert "opentelemetry is required" in str(exc_info.value)
        assert "pip install structured-logging[otel]" in str(exc_info.value)

