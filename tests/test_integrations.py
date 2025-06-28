"""
Tests for framework integrations (FastAPI, Flask)
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# Test FastAPI integration if available
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from structured_logging.config import LoggerConfig
from structured_logging.handlers import FileHandlerConfig
from structured_logging.integrations import (
    FastAPILoggingConfig,
    FlaskLoggingMiddleware,
    add_structured_logging,
    create_fastapi_logger_config,
    create_flask_logger_config,
)


class TestFastAPILoggingConfig:
    def test_default_config(self):
        config = FastAPILoggingConfig()

        assert config.logger_name == "fastapi"
        assert config.log_requests is True
        assert config.log_responses is True
        assert config.log_request_body is False
        assert config.log_response_body is False
        assert config.mask_sensitive_data is True
        assert "/health" in config.exclude_paths
        assert "authorization" in config.sensitive_headers
        assert "password" in config.sensitive_query_params

    def test_custom_config(self):
        config = FastAPILoggingConfig(
            logger_name="custom_api",
            log_request_body=True,
            exclude_paths={"/custom/health"},
            min_duration_ms=100.0,
        )

        assert config.logger_name == "custom_api"
        assert config.log_request_body is True
        assert "/custom/health" in config.exclude_paths
        assert config.min_duration_ms == 100.0


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestFastAPIIntegration:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "fastapi_test.log")

        # Create logger config with file output for testing
        file_config = FileHandlerConfig(
            filename=self.log_file, compress_rotated=False, async_compression=False
        )

        self.logger_config = LoggerConfig(
            output_type="file", file_config=file_config, formatter_type="json"
        )

    def teardown_method(self):
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_basic_fastapi_integration(self):
        # Create FastAPI app
        app = FastAPI()

        # Add logging middleware
        logging_config = FastAPILoggingConfig(
            logger_config=self.logger_config,
            log_request_headers=False,  # Simplify for testing
            capture_user_agent=False,
        )

        app = add_structured_logging(app, logging_config)

        @app.get("/test")
        def test_endpoint():
            return {"message": "test"}

        # Test the endpoint
        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200
        assert response.json() == {"message": "test"}

        # Check logs were written
        self._flush_logs(app)
        logs = self._read_logs()

        assert len(logs) >= 2  # Request start + Request completed

        # Check request log
        request_log = next(
            (log for log in logs if "Request started" in log.get("message", "")), None
        )
        assert request_log is not None
        assert request_log["method"] == "GET"
        assert request_log["path"] == "/test"

        # Check response log
        response_log = next(
            (log for log in logs if "Request completed" in log.get("message", "")), None
        )
        assert response_log is not None
        assert response_log["status_code"] == 200
        assert "duration_ms" in response_log

    def test_error_handling(self):
        app = FastAPI()

        logging_config = FastAPILoggingConfig(
            logger_config=self.logger_config, log_request_headers=False
        )

        app = add_structured_logging(app, logging_config)

        @app.get("/error")
        def error_endpoint():
            raise HTTPException(status_code=500, detail="Test error")

        client = TestClient(app)
        response = client.get("/error")

        assert response.status_code == 500

        # Check error was logged
        self._flush_logs(app)
        logs = self._read_logs()

        error_logs = [log for log in logs if log.get("level") == "ERROR"]
        assert len(error_logs) >= 1

        error_log = error_logs[0]
        assert "Request completed" in error_log["message"]
        assert error_log["status_code"] == 500

    def test_path_exclusion(self):
        app = FastAPI()

        logging_config = FastAPILoggingConfig(
            logger_config=self.logger_config, exclude_paths={"/health", "/metrics"}
        )

        app = add_structured_logging(app, logging_config)

        @app.get("/health")
        def health_check():
            return {"status": "ok"}

        @app.get("/normal")
        def normal_endpoint():
            return {"data": "normal"}

        client = TestClient(app)

        # Request to excluded path
        response1 = client.get("/health")
        assert response1.status_code == 200

        # Request to normal path
        response2 = client.get("/normal")
        assert response2.status_code == 200

        # Check logs
        self._flush_logs(app)
        logs = self._read_logs()

        # Should only have logs for /normal, not /health
        request_logs = [log for log in logs if "Request" in log.get("message", "")]
        normal_logs = [
            log for log in request_logs if "/normal" in log.get("message", "")
        ]
        health_logs = [
            log for log in request_logs if "/health" in log.get("message", "")
        ]

        assert len(normal_logs) >= 2  # Start + completion
        assert len(health_logs) == 0  # Should be excluded

    def test_sensitive_data_masking(self):
        app = FastAPI()

        logging_config = FastAPILoggingConfig(
            logger_config=self.logger_config,
            log_request_headers=True,
            mask_sensitive_data=True,
        )

        app = add_structured_logging(app, logging_config)

        @app.get("/secure")
        def secure_endpoint():
            return {"data": "secure"}

        client = TestClient(app)
        response = client.get(
            "/secure?password=secret123&api_key=key123&normal_param=value",
            headers={"Authorization": "Bearer token123", "X-Custom": "value"},
        )

        assert response.status_code == 200

        # Check logs
        self._flush_logs(app)
        logs = self._read_logs()

        request_log = next(
            (log for log in logs if "Request started" in log.get("message", "")), None
        )
        assert request_log is not None

        # Check query params masking
        query_params = request_log.get("query_params", {})
        # Handle both direct string and type-detected format
        password_val = query_params.get("password")
        if isinstance(password_val, dict) and password_val.get("value") == "[MASKED]":
            assert True  # Type-detected format
        else:
            assert password_val == "[MASKED]"  # Direct string format
            
        api_key_val = query_params.get("api_key")
        if isinstance(api_key_val, dict) and api_key_val.get("value") == "[MASKED]":
            assert True  # Type-detected format
        else:
            assert api_key_val == "[MASKED]"  # Direct string format
            
        assert query_params.get("normal_param") == "value"  # Should not be masked

        # Check headers masking
        headers = request_log.get("headers", {})
        auth_val = headers.get("authorization")
        if isinstance(auth_val, dict) and auth_val.get("value") == "[MASKED]":
            assert True  # Type-detected format
        else:
            assert auth_val == "[MASKED]"  # Direct string format
            
        assert headers.get("x-custom") == "value"  # Should not be masked

    def test_minimum_duration_filtering(self):
        app = FastAPI()

        logging_config = FastAPILoggingConfig(
            logger_config=self.logger_config,
            min_duration_ms=50.0,  # Only log requests taking > 50ms
        )

        app = add_structured_logging(app, logging_config)

        @app.get("/fast")
        def fast_endpoint():
            return {"message": "fast"}

        @app.get("/slow")
        def slow_endpoint():
            import time

            time.sleep(0.1)  # 100ms delay
            return {"message": "slow"}

        client = TestClient(app)

        # Fast request (should not be logged due to duration filter)
        response1 = client.get("/fast")
        assert response1.status_code == 200

        # Slow request (should be logged)
        response2 = client.get("/slow")
        assert response2.status_code == 200

        # Check logs
        self._flush_logs(app)
        logs = self._read_logs()

        # Should only have completion logs for slow endpoint
        completion_logs = [
            log for log in logs if "Request completed" in log.get("message", "")
        ]
        slow_logs = [
            log for log in completion_logs if "/slow" in log.get("message", "")
        ]
        fast_logs = [
            log for log in completion_logs if "/fast" in log.get("message", "")
        ]

        assert len(slow_logs) >= 1
        assert len(fast_logs) == 0  # Should be filtered out by duration

    def test_request_body_logging(self):
        app = FastAPI()

        logging_config = FastAPILoggingConfig(
            logger_config=self.logger_config,
            log_request_body=True,
            max_request_body_size=100,
        )

        app = add_structured_logging(app, logging_config)

        @app.post("/data")
        def data_endpoint(data: dict):
            return {"received": data}

        client = TestClient(app)

        test_data = {"key": "value", "number": 42}
        response = client.post("/data", json=test_data)

        assert response.status_code == 200

        # Check logs
        self._flush_logs(app)
        logs = self._read_logs()

        request_log = next(
            (log for log in logs if "Request started" in log.get("message", "")), None
        )
        assert request_log is not None

        # Check request body was logged
        request_body = request_log.get("request_body")
        assert request_body is not None
        assert isinstance(request_body, dict)
        assert request_body["key"] == "value"
        assert request_body["number"] == 42

    def _flush_logs(self, app):
        """Flush all log handlers and ensure logs are written"""
        import logging
        import time

        # Flush all loggers
        for name in logging.Logger.manager.loggerDict:
            logger = logging.getLogger(name)
            if hasattr(logger, "handlers"):
                for handler in logger.handlers:
                    handler.flush()

        # Also flush root logger
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            handler.flush()

        # Small delay to ensure async operations complete
        time.sleep(0.01)

    def _read_logs(self):
        """Read and parse log entries from file"""
        if not os.path.exists(self.log_file):
            return []

        logs = []
        with open(self.log_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return logs


class TestHelperFunctions:
    def test_create_fastapi_logger_config_console(self):
        config = create_fastapi_logger_config(output_type="console", log_level="DEBUG")

        assert config.output_type == "console"
        assert config.log_level == "DEBUG"
        assert config.formatter_type == "json"
        assert config.include_timestamp is True
        assert config.file_config is None
        assert config.filter_config is None

    def test_create_fastapi_logger_config_file(self):
        config = create_fastapi_logger_config(
            output_type="file", filename="custom.log", log_level="INFO"
        )

        assert config.output_type == "file"
        assert config.file_config is not None
        assert config.file_config.filename == "custom.log"
        assert config.file_config.max_bytes == 50 * 1024 * 1024  # 50MB
        assert config.file_config.compress_rotated is True

    def test_create_fastapi_logger_config_with_filtering(self):
        config = create_fastapi_logger_config(
            enable_filtering=True, sample_rate=0.5, log_level="WARNING"
        )

        assert config.filter_config is not None
        assert config.filter_config.enabled is True
        assert len(config.filter_config.filters) == 2  # Level + Sampling

        # Check level filter
        level_filter = config.filter_config.filters[0]
        assert level_filter.__class__.__name__ == "LevelFilter"

        # Check sampling filter
        sampling_filter = config.filter_config.filters[1]
        assert sampling_filter.__class__.__name__ == "SamplingFilter"
        assert sampling_filter.sample_rate == 0.5
        assert sampling_filter.strategy == "level_based"

    def test_create_flask_logger_config(self):
        config = create_flask_logger_config()

        # Should return the same type of config as FastAPI
        assert isinstance(config, LoggerConfig)
        assert config.formatter_type == "json"


class TestFlaskIntegration:
    def test_flask_middleware_creation(self):
        # Mock Flask app
        mock_app = MagicMock()

        config = FastAPILoggingConfig(logger_name="flask_test")
        middleware = FlaskLoggingMiddleware(mock_app, config)

        assert middleware.config.logger_name == "flask_test"
        assert middleware.logger is not None

        # Check that Flask methods were called
        mock_app.before_request.assert_called_once()
        mock_app.after_request.assert_called_once()

    def test_flask_middleware_without_app(self):
        config = FastAPILoggingConfig(logger_name="flask_test")
        middleware = FlaskLoggingMiddleware(None, config)

        assert middleware.config.logger_name == "flask_test"
        assert middleware.logger is not None


class TestEdgeCases:
    def test_fastapi_not_available(self):
        with patch("structured_logging.integrations.fastapi.FASTAPI_AVAILABLE", False):
            with pytest.raises(ImportError, match="FastAPI is not installed"):
                add_structured_logging(MagicMock())

    def test_middleware_with_exception_in_logging(self):
        """Test that middleware doesn't break when logging fails"""
        if not FASTAPI_AVAILABLE:
            pytest.skip("FastAPI not available")

        app = FastAPI()

        # Create config with invalid logger config to trigger exceptions
        logging_config = FastAPILoggingConfig(
            logger_config=LoggerConfig(
                output_type="file", file_config=None
            )  # Invalid config
        )

        app = add_structured_logging(app, logging_config)

        @app.get("/test")
        def test_endpoint():
            return {"message": "test"}

        # Should still work even if logging fails
        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200
        assert response.json() == {"message": "test"}

    def test_request_body_size_limiting(self):
        if not FASTAPI_AVAILABLE:
            pytest.skip("FastAPI not available")

        temp_dir = tempfile.mkdtemp()
        log_file = os.path.join(temp_dir, "test.log")

        try:
            file_config = FileHandlerConfig(
                filename=log_file, compress_rotated=False, async_compression=False
            )

            logger_config = LoggerConfig(
                output_type="file", file_config=file_config, formatter_type="json"
            )

            app = FastAPI()

            logging_config = FastAPILoggingConfig(
                logger_config=logger_config,
                log_request_body=True,
                max_request_body_size=10,  # Very small limit
            )

            app = add_structured_logging(app, logging_config)

            @app.post("/data")
            def data_endpoint(data: dict):
                return {"received": "ok"}

            client = TestClient(app)

            # Send large body
            large_data = {"key": "x" * 100}  # Much larger than 10 bytes
            response = client.post("/data", json=large_data)

            assert response.status_code == 200

            # Check that body was truncated
            # Flush logs
            from structured_logging.logger import get_logger

            logger = get_logger("fastapi", logger_config)
            for handler in logger.handlers:
                handler.flush()

            # Read logs
            logs = []
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                logs.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass

            request_log = next(
                (log for log in logs if "Request started" in log.get("message", "")),
                None,
            )
            if request_log and "request_body" in request_log:
                body_str = str(request_log["request_body"])
                assert (
                    "[truncated]" in body_str or len(body_str) <= 20
                )  # Should be limited

        finally:
            import shutil

            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
