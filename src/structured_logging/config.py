import os
from dataclasses import dataclass
from typing import Literal, Optional


FormatterType = Literal["json", "csv", "plain"]


@dataclass
class LoggerConfig:
    """Configuration for structured logger"""

    log_level: str = "INFO"
    include_timestamp: bool = True
    include_request_id: bool = True
    include_user_context: bool = True
    formatter_type: FormatterType = "json"

    @classmethod
    def from_env(cls) -> "LoggerConfig":
        """Create configuration from environment variables"""
        formatter_type = os.getenv("STRUCTURED_LOG_FORMATTER", "json").lower()
        if formatter_type not in ["json", "csv", "plain"]:
            formatter_type = "json"

        return cls(
            log_level=os.getenv("STRUCTURED_LOG_LEVEL", "INFO"),
            include_timestamp=os.getenv("STRUCTURED_LOG_TIMESTAMP", "true").lower()
            == "true",
            include_request_id=os.getenv("STRUCTURED_LOG_REQUEST_ID", "true").lower()
            == "true",
            include_user_context=os.getenv(
                "STRUCTURED_LOG_USER_CONTEXT", "true"
            ).lower()
            == "true",
            formatter_type=formatter_type,
        )


_default_config: Optional[LoggerConfig] = None


def get_default_config() -> LoggerConfig:
    """Get the default configuration instance"""
    global _default_config
    if _default_config is None:
        _default_config = LoggerConfig.from_env()
    return _default_config


def set_default_config(config: LoggerConfig) -> None:
    """Set the default configuration instance"""
    global _default_config
    _default_config = config
