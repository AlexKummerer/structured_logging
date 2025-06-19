import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from .filtering import FilterConfig
    from .handlers import FileHandlerConfig

FormatterType = Literal["json", "csv", "plain"]
OutputType = Literal["console", "file", "both"]


@dataclass
class LoggerConfig:
    """Configuration for structured logger"""

    log_level: str = "INFO"
    include_timestamp: bool = True
    include_request_id: bool = True
    include_user_context: bool = True
    formatter_type: FormatterType = "json"
    filter_config: Optional["FilterConfig"] = None
    output_type: OutputType = "console"
    file_config: Optional["FileHandlerConfig"] = None

    @classmethod
    def from_env(cls) -> "LoggerConfig":
        """Create configuration from environment variables"""
        formatter_type = os.getenv("STRUCTURED_LOG_FORMATTER", "json").lower()
        if formatter_type not in ["json", "csv", "plain"]:
            formatter_type = "json"

        # Create filter config if filtering is enabled
        filter_config = None
        if os.getenv("STRUCTURED_LOG_FILTERING", "false").lower() == "true":
            from .filtering import FilterConfig, LevelFilter, SamplingFilter

            sample_rate = float(os.getenv("STRUCTURED_LOG_SAMPLE_RATE", "1.0"))
            max_per_second = os.getenv("STRUCTURED_LOG_MAX_PER_SECOND")

            filters = []

            # Add level filter
            filters.append(
                LevelFilter(min_level=os.getenv("STRUCTURED_LOG_LEVEL", "INFO"))
            )

            # Add sampling filter if configured
            if sample_rate < 1.0 or max_per_second:
                filters.append(
                    SamplingFilter(
                        sample_rate=sample_rate,
                        strategy=os.getenv(
                            "STRUCTURED_LOG_SAMPLING_STRATEGY", "level_based"
                        ),
                        max_per_second=int(max_per_second) if max_per_second else None,
                    )
                )

            filter_config = FilterConfig(
                enabled=True,
                filters=filters,
                collect_metrics=os.getenv(
                    "STRUCTURED_LOG_COLLECT_METRICS", "true"
                ).lower()
                == "true",
            )

        # Create file config if file output is enabled
        file_config = None
        output_type = os.getenv("STRUCTURED_LOG_OUTPUT", "console").lower()

        if output_type in ["file", "both"]:
            from .handlers import FileHandlerConfig

            file_config = FileHandlerConfig(
                filename=os.getenv("STRUCTURED_LOG_FILENAME", "app.log"),
                max_bytes=int(
                    os.getenv("STRUCTURED_LOG_MAX_BYTES", "10485760")
                ),  # 10MB
                backup_count=int(os.getenv("STRUCTURED_LOG_BACKUP_COUNT", "5")),
                compress_rotated=os.getenv("STRUCTURED_LOG_COMPRESS", "true").lower()
                == "true",
                archive_old_logs=os.getenv("STRUCTURED_LOG_ARCHIVE", "true").lower()
                == "true",
                archive_after_days=int(os.getenv("STRUCTURED_LOG_ARCHIVE_DAYS", "30")),
                archive_directory=os.getenv("STRUCTURED_LOG_ARCHIVE_DIR"),
                async_compression=os.getenv(
                    "STRUCTURED_LOG_ASYNC_COMPRESS", "true"
                ).lower()
                == "true",
            )

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
            filter_config=filter_config,
            output_type=output_type,
            file_config=file_config,
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
