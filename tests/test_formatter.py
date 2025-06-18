import json
import logging

from structured_logging.config import LoggerConfig
from structured_logging.formatter import StructuredFormatter


def test_structured_formatter_basic():
    formatter = StructuredFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Test message",
        args=(),
        exc_info=None,
    )

    result = formatter.format(record)
    log_data = json.loads(result)

    assert log_data["level"] == "INFO"
    assert log_data["logger"] == "test_logger"
    assert log_data["message"] == "Test message"
    assert "timestamp" in log_data


def test_structured_formatter_with_context():
    formatter = StructuredFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    record.ctx_request_id = "req-123"
    record.ctx_user_id = "user-456"

    result = formatter.format(record)
    log_data = json.loads(result)

    assert log_data["request_id"] == "req-123"
    assert log_data["user_id"] == "user-456"


def test_structured_formatter_without_timestamp():
    config = LoggerConfig(include_timestamp=False)
    formatter = StructuredFormatter(config)
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Test message",
        args=(),
        exc_info=None,
    )

    result = formatter.format(record)
    log_data = json.loads(result)

    assert "timestamp" not in log_data
    assert log_data["level"] == "INFO"
    assert log_data["message"] == "Test message"
