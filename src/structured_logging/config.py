import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

# Import all required modules at the top
from .filtering import FilterConfig, LevelFilter, SamplingFilter
from .handlers import FileHandlerConfig
from .network import (
    HTTPConfig,
    NetworkHandlerConfig,
    SocketConfig,
    SyslogConfig,
)

if TYPE_CHECKING:
    pass

FormatterType = Literal["json", "csv", "plain"]
OutputType = Literal[
    "console", "file", "both", "network", "console+network", "file+network", "all"
]


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
    network_config: Optional["NetworkHandlerConfig"] = None

    @classmethod
    def _parse_bool_env(cls, key: str, default: str = "false") -> bool:
        """Parse boolean from environment variable"""
        return os.getenv(key, default).lower() == "true"

    @classmethod
    def _create_filter_config_from_env(cls) -> Optional[FilterConfig]:
        """Create filter configuration from environment variables"""
        if not cls._parse_bool_env("STRUCTURED_LOG_FILTERING"):
            return None

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
                    strategy=os.getenv("STRUCTURED_LOG_SAMPLING_STRATEGY", "level_based"),
                    max_per_second=int(max_per_second) if max_per_second else None,
                )
            )

        return FilterConfig(
            enabled=True,
            filters=filters,
            collect_metrics=cls._parse_bool_env("STRUCTURED_LOG_COLLECT_METRICS", "true"),
        )

    @classmethod
    def _create_file_config_from_env(cls) -> Optional[FileHandlerConfig]:
        """Create file handler configuration from environment variables"""
        output_type = os.getenv("STRUCTURED_LOG_OUTPUT", "console").lower()
        if "file" not in output_type:
            return None

        return FileHandlerConfig(
            filename=os.getenv("STRUCTURED_LOG_FILENAME", "app.log"),
            max_bytes=int(os.getenv("STRUCTURED_LOG_MAX_BYTES", "10485760")),
            backup_count=int(os.getenv("STRUCTURED_LOG_BACKUP_COUNT", "5")),
            compress_rotated=cls._parse_bool_env("STRUCTURED_LOG_COMPRESS", "true"),
            archive_old_logs=cls._parse_bool_env("STRUCTURED_LOG_ARCHIVE", "true"),
            archive_after_days=int(os.getenv("STRUCTURED_LOG_ARCHIVE_DAYS", "30")),
            archive_directory=os.getenv("STRUCTURED_LOG_ARCHIVE_DIR"),
            async_compression=cls._parse_bool_env("STRUCTURED_LOG_ASYNC_COMPRESS", "true"),
        )

    @classmethod
    def _create_syslog_config_from_env(cls) -> SyslogConfig:
        """Create syslog configuration from environment variables"""
        return SyslogConfig(
            host=os.getenv("STRUCTURED_LOG_SYSLOG_HOST", "localhost"),
            port=int(os.getenv("STRUCTURED_LOG_SYSLOG_PORT", "514")),
            facility=int(os.getenv("STRUCTURED_LOG_SYSLOG_FACILITY", "16")),
            rfc_format=os.getenv("STRUCTURED_LOG_SYSLOG_RFC", "3164"),
            app_name=os.getenv("STRUCTURED_LOG_APP_NAME", "python-app"),
            use_ssl=cls._parse_bool_env("STRUCTURED_LOG_SYSLOG_SSL"),
        )

    @classmethod
    def _create_http_config_from_env(cls) -> HTTPConfig:
        """Create HTTP configuration from environment variables"""
        return HTTPConfig(
            url=os.getenv("STRUCTURED_LOG_HTTP_URL", "http://localhost:8080/logs"),
            method=os.getenv("STRUCTURED_LOG_HTTP_METHOD", "POST"),
            auth_type=os.getenv("STRUCTURED_LOG_HTTP_AUTH", "none"),
            token=os.getenv("STRUCTURED_LOG_HTTP_TOKEN"),
            api_key=os.getenv("STRUCTURED_LOG_HTTP_API_KEY"),
            batch_size=int(os.getenv("STRUCTURED_LOG_HTTP_BATCH_SIZE", "10")),
        )

    @classmethod
    def _create_socket_config_from_env(cls) -> SocketConfig:
        """Create socket configuration from environment variables"""
        return SocketConfig(
            host=os.getenv("STRUCTURED_LOG_SOCKET_HOST", "localhost"),
            port=int(os.getenv("STRUCTURED_LOG_SOCKET_PORT", "5140")),
            protocol=os.getenv("STRUCTURED_LOG_SOCKET_PROTOCOL", "tcp"),
            keep_alive=cls._parse_bool_env("STRUCTURED_LOG_SOCKET_KEEPALIVE", "true"),
        )

    @classmethod
    def _create_network_config_from_env(cls) -> Optional[NetworkHandlerConfig]:
        """Create network configuration from environment variables"""
        output_type = os.getenv("STRUCTURED_LOG_OUTPUT", "console").lower()
        if "network" not in output_type:
            return None

        network_type = os.getenv("STRUCTURED_LOG_NETWORK_TYPE", "syslog").lower()
        
        if network_type == "syslog":
            return cls._create_syslog_config_from_env()
        elif network_type == "http":
            return cls._create_http_config_from_env()
        elif network_type == "socket":
            return cls._create_socket_config_from_env()
        
        return None

    @classmethod
    def from_env(cls) -> "LoggerConfig":
        """Create configuration from environment variables"""
        formatter_type = os.getenv("STRUCTURED_LOG_FORMATTER", "json").lower()
        if formatter_type not in ["json", "csv", "plain"]:
            formatter_type = "json"

        return cls(
            log_level=os.getenv("STRUCTURED_LOG_LEVEL", "INFO"),
            include_timestamp=cls._parse_bool_env("STRUCTURED_LOG_TIMESTAMP", "true"),
            include_request_id=cls._parse_bool_env("STRUCTURED_LOG_REQUEST_ID", "true"),
            include_user_context=cls._parse_bool_env("STRUCTURED_LOG_USER_CONTEXT", "true"),
            formatter_type=formatter_type,
            filter_config=cls._create_filter_config_from_env(),
            output_type=os.getenv("STRUCTURED_LOG_OUTPUT", "console").lower(),
            file_config=cls._create_file_config_from_env(),
            network_config=cls._create_network_config_from_env(),
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
