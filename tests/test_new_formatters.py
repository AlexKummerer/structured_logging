import logging

from structured_logging import (
    CSVFormatter,
    LoggerConfig,
    PlainTextFormatter,
    get_logger,
    log_with_context,
    request_context,
    set_default_config,
)


def test_csv_formatter_basic():
    formatter = CSVFormatter()
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
    
    # Should contain CSV headers and values
    assert "INFO" in result
    assert "test_logger" in result
    assert "Test message" in result
    # CSV format should have commas
    assert "," in result


def test_csv_formatter_with_context():
    formatter = CSVFormatter()
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
    
    # Should contain context values
    assert "req-123" in result
    assert "user-456" in result


def test_csv_formatter_without_timestamp():
    config = LoggerConfig(include_timestamp=False)
    formatter = CSVFormatter(config)
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
    
    # Should not start with timestamp format
    assert not result.startswith("2")  # ISO timestamp starts with year


def test_plain_text_formatter_basic():
    formatter = PlainTextFormatter()
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
    
    # Should contain basic log elements
    assert "INFO" in result
    assert "test_logger" in result
    assert "Test message" in result
    # Plain text format should have spaces
    assert " " in result


def test_plain_text_formatter_with_context():
    formatter = PlainTextFormatter()
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
    
    # Should contain context in parentheses
    assert "request_id=req-123" in result
    assert "user_id=user-456" in result
    assert "(" in result and ")" in result


def test_plain_text_formatter_without_timestamp():
    config = LoggerConfig(include_timestamp=False)
    formatter = PlainTextFormatter(config)
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
    
    # Should not contain timestamp brackets
    assert "[" not in result or "]" not in result


def test_get_logger_with_csv_formatter():
    config = LoggerConfig(formatter_type="csv")
    set_default_config(config)
    logger = get_logger("test_csv_logger")
    
    # Logger should be created successfully
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_csv_logger"


def test_get_logger_with_plain_formatter():
    config = LoggerConfig(formatter_type="plain")
    set_default_config(config)
    logger = get_logger("test_plain_logger")
    
    # Logger should be created successfully
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_plain_logger"


def test_csv_formatter_with_request_context(caplog):
    config = LoggerConfig(formatter_type="csv")
    set_default_config(config)
    logger = get_logger("test_csv_context_logger")

    with request_context(user_id="user123", tenant_id="tenant456") as req_id:
        with caplog.at_level(logging.INFO, logger="test_csv_context_logger"):
            log_with_context(logger, "info", "Test message")

        # Check that log was created
        assert len(caplog.records) == 1


def test_plain_formatter_with_request_context(caplog):
    config = LoggerConfig(formatter_type="plain")
    set_default_config(config)
    logger = get_logger("test_plain_context_logger")

    with request_context(user_id="user123", tenant_id="tenant456") as req_id:
        with caplog.at_level(logging.INFO, logger="test_plain_context_logger"):
            log_with_context(logger, "info", "Test message")

        # Check that log was created
        assert len(caplog.records) == 1


def test_formatter_type_validation():
    # Test that invalid formatter types default to json
    config = LoggerConfig(formatter_type="json")
    assert config.formatter_type == "json"
    
    config = LoggerConfig(formatter_type="csv") 
    assert config.formatter_type == "csv"
    
    config = LoggerConfig(formatter_type="plain")
    assert config.formatter_type == "plain"


def test_environment_formatter_selection(monkeypatch):
    monkeypatch.setenv("STRUCTURED_LOG_FORMATTER", "csv")
    config = LoggerConfig.from_env()
    assert config.formatter_type == "csv"
    
    monkeypatch.setenv("STRUCTURED_LOG_FORMATTER", "plain")
    config = LoggerConfig.from_env()
    assert config.formatter_type == "plain"
    
    monkeypatch.setenv("STRUCTURED_LOG_FORMATTER", "invalid")
    config = LoggerConfig.from_env()
    assert config.formatter_type == "json"  # Should default to json