import logging

from structured_logging import (
    LoggerConfig,
    get_logger,
    log_with_context,
    request_context,
    set_default_config,
)


def test_get_logger():
    # Reset default config
    set_default_config(LoggerConfig())
    logger = get_logger("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"
    assert logger.level == logging.INFO


def test_get_logger_with_config():
    config = LoggerConfig(log_level="DEBUG")
    logger = get_logger("test_debug_logger", config)
    assert logger.level == logging.DEBUG


def test_log_with_context(caplog):
    # Reset config and clear any existing loggers
    set_default_config(LoggerConfig())
    # Use a unique logger name to avoid handler conflicts
    logger = get_logger("test_context_logger_unique")

    with caplog.at_level(logging.INFO, logger="test_context_logger_unique"):
        log_with_context(logger, "info", "Test message", extra_field="extra_value")

    # Check that log was created
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert hasattr(record, "ctx_extra_field")
    assert record.ctx_extra_field == "extra_value"


def test_log_with_request_context(caplog):
    set_default_config(LoggerConfig())
    logger = get_logger("test_request_logger_unique")

    with request_context(user_id="user123", tenant_id="tenant456") as req_id:
        with caplog.at_level(logging.INFO, logger="test_request_logger_unique"):
            log_with_context(logger, "info", "Test message")

        record = caplog.records[0]
        assert hasattr(record, "ctx_request_id")
        assert record.ctx_request_id == req_id
        assert hasattr(record, "ctx_user_id")
        assert record.ctx_user_id == "user123"
        assert hasattr(record, "ctx_tenant_id")
        assert record.ctx_tenant_id == "tenant456"


def test_log_filtering_none_values(caplog):
    set_default_config(LoggerConfig())
    logger = get_logger("test_filter_logger_unique")

    with caplog.at_level(logging.INFO, logger="test_filter_logger_unique"):
        log_with_context(
            logger, "info", "Test message", valid_field="value", none_field=None
        )

    record = caplog.records[0]
    assert hasattr(record, "ctx_valid_field")
    assert record.ctx_valid_field == "value"
    assert not hasattr(record, "ctx_none_field")
