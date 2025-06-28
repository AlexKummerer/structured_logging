"""
Structured Logging Library

A flexible Python library for structured JSON logging with context management and async support.
"""

__version__ = "0.7.0"

# Async logging support
from .async_config import (
    AsyncLoggerConfig,
    get_default_async_config,
    set_default_async_config,
)
from .async_context import (
    aget_custom_context,
    aget_request_id,
    aget_user_context,
    aset_custom_context,
    aset_request_id,
    aset_user_context,
    async_request_context,
)
from .async_logger import (
    AsyncLogger,
    alog_with_context,
    get_async_logger,
    shutdown_all_async_loggers,
)
from .config import (
    FormatterType,
    LoggerConfig,
    OutputType,
    get_default_config,
    set_default_config,
)
from .context import (
    get_custom_context,
    get_request_id,
    get_user_context,
    request_context,
    set_custom_context,
    set_request_id,
    set_user_context,
    update_custom_context,
)
from .filtering import (
    ContextFilter,
    CustomFilter,
    FilterConfig,
    FilterEngine,
    FilterResult,
    LevelFilter,
    SamplingFilter,
)
from .formatter import CSVFormatter, PlainTextFormatter, StructuredFormatter
from .handlers import (
    FileHandlerConfig,
    RotatingFileHandler,
    TimedRotatingFileHandler,
    create_file_logger,
)
from .integrations import (
    FastAPILoggingConfig,
    FastAPILoggingMiddleware,
    FlaskLoggingMiddleware,
    add_structured_logging,
    create_fastapi_logger_config,
    create_flask_logger_config,
)
from .logger import (
    get_filter_metrics,
    get_logger,
    log_with_context,
    reset_filter_metrics,
)
from .network import (
    HTTPConfig,
    HTTPHandler,
    NetworkHandlerConfig,
    SocketConfig,
    SocketHandler,
    SyslogConfig,
    SyslogHandler,
)

# Import cloud handlers if available
try:
    from .cloud import CloudHandlerConfig, CloudLogHandler
    _CLOUD_AVAILABLE = True
except ImportError:
    _CLOUD_AVAILABLE = False

# Import AWS handlers if available
try:
    from .cloud import CloudWatchConfig, CloudWatchHandler
    _AWS_AVAILABLE = True
except ImportError:
    _AWS_AVAILABLE = False

# Import GCP handlers if available  
try:
    from .cloud import (
        GoogleCloudConfig,
        GoogleCloudHandler,
        StackdriverConfig,
        StackdriverHandler,
    )
    _GCP_AVAILABLE = True
except ImportError:
    _GCP_AVAILABLE = False

# Import Azure handlers if available
try:
    from .cloud import (
        AzureMonitorConfig,
        AzureMonitorHandler,
        ApplicationInsightsConfig,
        ApplicationInsightsHandler,
    )
    _AZURE_AVAILABLE = True
except ImportError:
    _AZURE_AVAILABLE = False

# Import analytics if available
try:
    from .analytics import (
        # Pattern detection
        PatternDetector,
        PatternDetectorConfig,
        LogPattern,
        PatternMatch,
        detect_patterns,
        create_pattern_detector,
        # Anomaly detection
        AnomalyDetector,
        AnomalyDetectorConfig,
        LogAnomaly,
        AnomalyScore,
        detect_anomalies,
        create_anomaly_detector,
        # Metrics collection
        MetricsCollector,
        MetricsConfig,
        PerformanceMetrics,
        ErrorMetrics,
        ThroughputMetrics,
        UserMetrics,
        MetricsSummary,
        collect_metrics,
        create_metrics_collector,
    )
    _ANALYTICS_AVAILABLE = True
except ImportError:
    _ANALYTICS_AVAILABLE = False
from .serializers import (
    EnhancedJSONEncoder,
    LazyDict,
    LazySerializable,
    LazySerializationManager,
    SchemaValidator,
    SerializationConfig,
    SmartConverter,
    StructuredDataValidator,
    TypeAnnotationExtractor,
    TypeDetector,
    TypeRegistry,
    ValidationError,
    auto_validate_function,
    create_lazy_serializable,
    enhanced_json_dumps,
    get_lazy_serialization_stats,
    get_validation_stats,
    register_custom_serializer,
    register_validation_schema,
    reset_lazy_serialization_stats,
    reset_validation_stats,
    serialize_for_logging,
    serialize_for_logging_lazy_aware,
    should_use_lazy_serialization,
    validate_log_data,
)

__all__ = [
    # Sync API
    "LoggerConfig",
    "FormatterType",
    "OutputType",
    "get_default_config",
    "set_default_config",
    "request_context",
    "get_request_id",
    "set_request_id",
    "get_user_context",
    "set_user_context",
    "get_custom_context",
    "set_custom_context",
    "update_custom_context",
    "StructuredFormatter",
    "CSVFormatter",
    "PlainTextFormatter",
    "get_logger",
    "log_with_context",
    "get_filter_metrics",
    "reset_filter_metrics",
    "FilterConfig",
    "FilterEngine",
    "LevelFilter",
    "ContextFilter",
    "CustomFilter",
    "SamplingFilter",
    "FilterResult",
    "FileHandlerConfig",
    "RotatingFileHandler",
    "TimedRotatingFileHandler",
    "create_file_logger",
    "FastAPILoggingConfig",
    "FastAPILoggingMiddleware",
    "add_structured_logging",
    "create_fastapi_logger_config",
    "create_flask_logger_config",
    "FlaskLoggingMiddleware",
    # Network handlers
    "NetworkHandlerConfig",
    "SyslogConfig",
    "SyslogHandler",
    "HTTPConfig",
    "HTTPHandler",
    "SocketConfig",
    "SocketHandler",
    # Enhanced serialization
    "SerializationConfig",
    "TypeRegistry",
    "TypeDetector",
    "SmartConverter",
    "EnhancedJSONEncoder",
    "enhanced_json_dumps",
    "register_custom_serializer",
    "serialize_for_logging",
    "serialize_for_logging_lazy_aware",
    # Lazy serialization
    "LazySerializable",
    "LazyDict",
    "LazySerializationManager",
    "create_lazy_serializable",
    "should_use_lazy_serialization",
    "get_lazy_serialization_stats",
    "reset_lazy_serialization_stats",
    # Schema validation
    "ValidationError",
    "SchemaValidator",
    "TypeAnnotationExtractor",
    "StructuredDataValidator",
    "register_validation_schema",
    "validate_log_data",
    "auto_validate_function",
    "get_validation_stats",
    "reset_validation_stats",
    # Async API
    "AsyncLoggerConfig",
    "get_default_async_config",
    "set_default_async_config",
    "async_request_context",
    "aget_request_id",
    "aget_user_context",
    "aget_custom_context",
    "aset_request_id",
    "aset_user_context",
    "aset_custom_context",
    "AsyncLogger",
    "get_async_logger",
    "alog_with_context",
    "shutdown_all_async_loggers",
]

# Add cloud exports if available
if _CLOUD_AVAILABLE:
    __all__.extend([
        "CloudHandlerConfig",
        "CloudLogHandler",
    ])

if _AWS_AVAILABLE:
    __all__.extend([
        "CloudWatchConfig",
        "CloudWatchHandler",
    ])

if _GCP_AVAILABLE:
    __all__.extend([
        "GoogleCloudConfig",
        "GoogleCloudHandler", 
        "StackdriverConfig",
        "StackdriverHandler",
    ])

if _AZURE_AVAILABLE:
    __all__.extend([
        "AzureMonitorConfig",
        "AzureMonitorHandler",
        "ApplicationInsightsConfig",
        "ApplicationInsightsHandler",
    ])

if _ANALYTICS_AVAILABLE:
    __all__.extend([
        # Pattern detection
        "PatternDetector",
        "PatternDetectorConfig",
        "LogPattern",
        "PatternMatch",
        "detect_patterns",
        "create_pattern_detector",
        # Anomaly detection
        "AnomalyDetector",
        "AnomalyDetectorConfig",
        "LogAnomaly",
        "AnomalyScore",
        "detect_anomalies",
        "create_anomaly_detector",
        # Metrics collection
        "MetricsCollector",
        "MetricsConfig",
        "PerformanceMetrics",
        "ErrorMetrics",
        "ThroughputMetrics",
        "UserMetrics",
        "MetricsSummary",
        "collect_metrics",
        "create_metrics_collector",
    ])
