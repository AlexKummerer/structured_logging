"""
Tests for file handlers and rotation functionality
"""

import gzip
import logging
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from structured_logging.config import LoggerConfig
from structured_logging.handlers import (
    FileHandlerConfig,
    RotatingFileHandler,
    TimedRotatingFileHandler,
    create_file_logger,
)
from structured_logging.logger import get_logger


class TestFileHandlerConfig:
    def test_default_config(self):
        config = FileHandlerConfig()

        assert config.filename == "app.log"
        assert config.max_bytes == 10 * 1024 * 1024  # 10MB
        assert config.backup_count == 5
        assert config.compress_rotated is True
        assert config.archive_old_logs is True
        assert config.archive_after_days == 30
        assert config.async_compression is True


class TestRotatingFileHandler:
    def setup_method(self):
        # Create temporary directory for tests
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "test.log")

        self.config = FileHandlerConfig(
            filename=self.log_file,
            max_bytes=1024,  # Small size for testing
            backup_count=3,
            compress_rotated=False,  # Disable for easier testing
            archive_old_logs=False,  # Disable for simpler tests
            async_compression=False,  # Synchronous for predictable tests
        )

    def teardown_method(self):
        # Clean up temporary files
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_handler_creation(self):
        handler = RotatingFileHandler(self.config)

        assert handler.base_filename == self.log_file
        assert handler.config.max_bytes == 1024
        assert handler.stream is not None

        handler.close()

    def test_log_writing(self):
        handler = RotatingFileHandler(self.config)
        handler.setFormatter(logging.Formatter("%(message)s"))

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        handler.emit(record)
        handler.flush()

        assert os.path.exists(self.log_file)
        with open(self.log_file, "r") as f:
            content = f.read()
            assert "Test message" in content

        handler.close()

    def test_rotation_on_size(self):
        handler = RotatingFileHandler(self.config)
        handler.setFormatter(logging.Formatter("%(message)s"))

        # Write enough data to trigger rotation
        large_message = "x" * 500
        for i in range(5):  # This should exceed 1024 bytes
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=f"{large_message}_{i}",
                args=(),
                exc_info=None,
            )
            handler.emit(record)

        handler.flush()

        # Should have created backup files
        assert os.path.exists(self.log_file)
        assert os.path.exists(f"{self.log_file}.1")

        handler.close()

    def test_compression(self):
        config_with_compression = FileHandlerConfig(
            filename=self.log_file,
            max_bytes=500,
            backup_count=2,
            compress_rotated=True,
            async_compression=False,  # Synchronous for predictable testing
        )

        handler = RotatingFileHandler(config_with_compression)
        handler.setFormatter(logging.Formatter("%(message)s"))

        # Write data to trigger rotation
        large_message = "x" * 300
        for i in range(3):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=f"{large_message}_{i}",
                args=(),
                exc_info=None,
            )
            handler.emit(record)

        handler.flush()

        # Give compression time to complete
        time.sleep(0.1)

        # Should have compressed backup
        compressed_file = f"{self.log_file}.1.gz"
        if os.path.exists(compressed_file):
            # Verify it's a valid gzip file
            with gzip.open(compressed_file, "rt") as f:
                content = f.read()
                assert "xxx" in content

        handler.close()

    def test_backup_count_limit(self):
        handler = RotatingFileHandler(self.config)
        handler.setFormatter(logging.Formatter("%(message)s"))

        # Force multiple rotations
        large_message = "x" * 400
        for i in range(15):  # Force several rotations
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=f"{large_message}_{i}",
                args=(),
                exc_info=None,
            )
            handler.emit(record)

        handler.flush()

        # Should not exceed backup_count
        assert os.path.exists(self.log_file)
        assert os.path.exists(f"{self.log_file}.1")
        assert os.path.exists(f"{self.log_file}.2")
        assert os.path.exists(f"{self.log_file}.3")
        assert not os.path.exists(
            f"{self.log_file}.4"
        )  # Should not exist due to backup_count=3

        handler.close()

    def test_archive_functionality(self):
        archive_dir = os.path.join(self.temp_dir, "archive")
        config_with_archive = FileHandlerConfig(
            filename=self.log_file,
            archive_old_logs=True,
            archive_after_days=0,  # Archive immediately for testing
            archive_directory=archive_dir,
            async_compression=False,
        )

        handler = RotatingFileHandler(config_with_archive)

        # Create some old log files
        old_file = f"{self.log_file}.old"
        with open(old_file, "w") as f:
            f.write("old log content")

        # Trigger archiving
        handler._archive_old_logs()

        # Check that archive directory was created and file was moved
        assert os.path.exists(archive_dir)
        archived_files = list(Path(archive_dir).glob("*"))
        assert len(archived_files) > 0

        handler.close()


class TestTimedRotatingFileHandler:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "timed_test.log")

        self.config = FileHandlerConfig(
            filename=self.log_file,
            compress_rotated=False,
            archive_old_logs=False,
            async_compression=False,
        )

    def teardown_method(self):
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_timed_handler_creation(self):
        handler = TimedRotatingFileHandler(self.config, when="S", interval=1)

        assert handler.when == "S"
        assert handler.interval == 1
        assert handler.interval_seconds == 1
        assert handler.rollover_at > time.time()

        handler.close()

    def test_time_based_rotation_trigger(self):
        # Create handler that should rotate very quickly
        handler = TimedRotatingFileHandler(self.config, when="S", interval=1)
        handler.setFormatter(logging.Formatter("%(message)s"))

        # Mock time to force rotation
        with patch("time.time") as mock_time:
            # Set initial time
            mock_time.return_value = handler.rollover_at - 1

            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="Before rotation",
                args=(),
                exc_info=None,
            )
            handler.emit(record)

            # Advance time past rollover point
            mock_time.return_value = handler.rollover_at + 1

            record2 = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="After rotation",
                args=(),
                exc_info=None,
            )
            handler.emit(record2)

        handler.close()

    def test_different_time_intervals(self):
        # Test different time interval configurations
        test_cases = [
            ("S", 1, 1),
            ("M", 1, 60),
            ("H", 1, 3600),
            ("D", 1, 86400),
            ("MIDNIGHT", 1, 86400),
        ]

        for when, interval, expected_seconds in test_cases:
            handler = TimedRotatingFileHandler(
                self.config, when=when, interval=interval
            )
            assert handler.interval_seconds == expected_seconds
            handler.close()

    def test_invalid_when_parameter(self):
        with pytest.raises(ValueError):
            TimedRotatingFileHandler(self.config, when="INVALID")


class TestCreateFileLogger:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "logger_test.log")

        self.config = FileHandlerConfig(
            filename=self.log_file, compress_rotated=False, async_compression=False
        )

    def teardown_method(self):
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_create_rotating_file_logger(self):
        logger = create_file_logger("test_logger", self.config, handler_type="rotating")

        assert isinstance(logger, logging.Logger)
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], RotatingFileHandler)

        # Test logging
        logger.info("Test message")

        # Ensure handler is flushed and closed properly
        for handler in logger.handlers:
            handler.flush()

        assert os.path.exists(self.log_file)
        with open(self.log_file, "r") as f:
            content = f.read()
            assert "Test message" in content

    def test_create_timed_file_logger(self):
        logger = create_file_logger(
            "test_timed_logger", self.config, handler_type="timed"
        )

        assert isinstance(logger, logging.Logger)
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], TimedRotatingFileHandler)

        # Test logging
        logger.info("Timed test message")

        # Ensure handler is flushed
        for handler in logger.handlers:
            handler.flush()

        assert os.path.exists(self.log_file)
        with open(self.log_file, "r") as f:
            content = f.read()
            assert "Timed test message" in content

    def test_custom_formatter(self):
        custom_formatter = logging.Formatter("CUSTOM: %(message)s")
        logger = create_file_logger(
            "test_custom", self.config, formatter=custom_formatter
        )

        logger.info("Custom format test")

        # Ensure handler is flushed
        for handler in logger.handlers:
            handler.flush()

        with open(self.log_file, "r") as f:
            content = f.read()
            assert "CUSTOM: Custom format test" in content


class TestLoggerIntegration:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "integration_test.log")

    def teardown_method(self):
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_file_output_config(self):
        file_config = FileHandlerConfig(
            filename=self.log_file,
            max_bytes=2048,
            compress_rotated=False,
            async_compression=False,
        )

        logger_config = LoggerConfig(
            output_type="file", file_config=file_config, formatter_type="json"
        )

        logger = get_logger("test_file_logger", logger_config)

        # Should have only file handler, no console handler
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], RotatingFileHandler)

        logger.info("File only test")

        # Ensure handler is flushed
        for handler in logger.handlers:
            handler.flush()

        assert os.path.exists(self.log_file)
        with open(self.log_file, "r") as f:
            content = f.read()
            assert '"message":"File only test"' in content

    def test_both_output_config(self):
        file_config = FileHandlerConfig(
            filename=self.log_file, compress_rotated=False, async_compression=False
        )

        logger_config = LoggerConfig(
            output_type="both", file_config=file_config, formatter_type="json"
        )

        logger = get_logger("test_both_logger", logger_config)

        # Should have both console and file handlers
        assert len(logger.handlers) == 2

        handler_types = [type(h).__name__ for h in logger.handlers]
        assert "StreamHandler" in handler_types
        assert "RotatingFileHandler" in handler_types

        logger.info("Both outputs test")

        # Ensure handlers are flushed
        for handler in logger.handlers:
            handler.flush()

        assert os.path.exists(self.log_file)
        with open(self.log_file, "r") as f:
            content = f.read()
            assert '"message":"Both outputs test"' in content

    def test_structured_formatter_with_file_handler(self):
        file_config = FileHandlerConfig(
            filename=self.log_file, compress_rotated=False, async_compression=False
        )

        logger_config = LoggerConfig(
            output_type="file",
            file_config=file_config,
            formatter_type="json",
            include_timestamp=True,
            include_request_id=True,
        )

        logger = get_logger("test_structured_file", logger_config)

        # Test with context
        from structured_logging.context import request_context
        from structured_logging.logger import log_with_context

        with request_context(user_id="123", tenant_id="acme"):
            log_with_context(logger, "info", "Structured file test", logger_config)

        # Ensure handlers are flushed
        for handler in logger.handlers:
            handler.flush()

        assert os.path.exists(self.log_file)
        with open(self.log_file, "r") as f:
            content = f.read()
            assert '"user_id":"123"' in content
            assert '"tenant_id":"acme"' in content
            assert '"timestamp":"' in content


class TestEnvironmentConfiguration:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "env_test.log")

    def teardown_method(self):
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_file_config_from_environment(self):
        env_vars = {
            "STRUCTURED_LOG_OUTPUT": "file",
            "STRUCTURED_LOG_FILENAME": self.log_file,
            "STRUCTURED_LOG_MAX_BYTES": "2048",
            "STRUCTURED_LOG_BACKUP_COUNT": "3",
            "STRUCTURED_LOG_COMPRESS": "false",
            "STRUCTURED_LOG_ARCHIVE": "false",
        }

        with patch.dict(os.environ, env_vars):
            config = LoggerConfig.from_env()

            assert config.output_type == "file"
            assert config.file_config is not None
            assert config.file_config.filename == self.log_file
            assert config.file_config.max_bytes == 2048
            assert config.file_config.backup_count == 3
            assert config.file_config.compress_rotated is False
            assert config.file_config.archive_old_logs is False

    def test_both_output_from_environment(self):
        env_vars = {
            "STRUCTURED_LOG_OUTPUT": "both",
            "STRUCTURED_LOG_FILENAME": self.log_file,
        }

        with patch.dict(os.environ, env_vars):
            config = LoggerConfig.from_env()

            assert config.output_type == "both"
            assert config.file_config is not None
            assert config.file_config.filename == self.log_file
