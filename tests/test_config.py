from structured_logging.config import (
    LoggerConfig,
    get_default_config,
    set_default_config,
)


def test_logger_config_defaults():
    config = LoggerConfig()
    assert config.log_level == "INFO"
    assert config.include_timestamp is True
    assert config.include_request_id is True
    assert config.include_user_context is True


def test_logger_config_from_env(monkeypatch):
    monkeypatch.setenv("STRUCTURED_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("STRUCTURED_LOG_TIMESTAMP", "false")
    monkeypatch.setenv("STRUCTURED_LOG_REQUEST_ID", "false")
    monkeypatch.setenv("STRUCTURED_LOG_USER_CONTEXT", "false")

    config = LoggerConfig.from_env()
    assert config.log_level == "DEBUG"
    assert config.include_timestamp is False
    assert config.include_request_id is False
    assert config.include_user_context is False


def test_get_default_config():
    config = get_default_config()
    assert isinstance(config, LoggerConfig)


def test_set_default_config():
    custom_config = LoggerConfig(log_level="ERROR")
    set_default_config(custom_config)

    config = get_default_config()
    assert config.log_level == "ERROR"
