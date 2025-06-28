# Structured Logging

A flexible Python library for structured JSON logging with context management, async support, scientific data integration, and production-ready performance.

## Features

### 🚀 **Version 0.7.0 (In Development) - Cloud Platform Integration**
- **☁️ AWS CloudWatch**: Native integration with automatic batching, compression, and retry logic
- **🌩️ Google Cloud Logging**: Stackdriver integration with resource types and structured logging
- **🔜 Azure Monitor**: (Coming soon) Application Insights integration

### 🧬 **Version 0.6.0 - Scientific & Network Integration**
- **🧬 Scientific Data Support**: Native NumPy, Pandas, and SciPy integration with intelligent serialization
- **🌐 Network Handlers**: Syslog (RFC 3164/5424), HTTP API, and raw socket logging with SSL/TLS support
- **🔍 Intelligent Type Detection**: Automatic detection and conversion of complex data types
- **⚡ Lazy Serialization**: Memory-efficient serialization for large datasets and complex objects
- **🛡️ Schema Validation**: Runtime validation with automatic schema generation from type annotations

### 🏗️ **Core Features**
- **JSON Structured Logging**: All logs are formatted as JSON for easy parsing and analysis
- **Advanced Log Filtering**: Smart filtering and sampling for production performance
- **File Handlers**: Rotating file logging with gzip compression and archiving
- **FastAPI Integration**: One-line middleware for automatic request/response logging
- **Async Logging Support**: High-performance non-blocking async logging with queue-based processing
- **Context Management**: Automatic injection of request IDs, user context, and custom fields
- **Multiple Output Formats**: JSON, CSV, and Plain Text formatters
- **Flexible Configuration**: Environment-based configuration with sensible defaults
- **Multi-tenant Support**: Built-in support for user_id and tenant_id fields
- **Thread-safe & Async-safe**: Uses Python's contextvars for proper context isolation

## Installation

```bash
pip install structured-logging

# Optional extras
pip install structured-logging[scientific]  # NumPy, Pandas, SciPy support
pip install structured-logging[aws]         # AWS CloudWatch integration
pip install structured-logging[gcp]         # Google Cloud Logging integration
pip install structured-logging[all]         # All optional dependencies
```

## Quick Start

### Basic Usage

```python
from structured_logging import get_logger

logger = get_logger("my_app")
logger.info("Application started")
```

Output:
```json
{"timestamp": "2024-01-15T10:30:00.000Z", "level": "INFO", "logger": "my_app", "message": "Application started"}
```

### Scientific Data Logging

```python
import numpy as np
import pandas as pd
from structured_logging import get_logger, request_context
from structured_logging.serializers import SerializationConfig

# Configure for scientific data
config = SerializationConfig(
    numpy_include_metadata=True,
    numpy_array_precision=3,
    pandas_max_rows=10,
    pandas_include_describe=True
)

logger = get_logger("science_app", config=config)

with request_context(experiment_id="exp_001", researcher="dr_smith"):
    # NumPy arrays with intelligent compression
    measurements = np.random.randn(1000)
    logger.info("Experiment completed", 
               measurements=measurements,
               mean_value=float(np.mean(measurements)))
    
    # Pandas DataFrames with sampling
    results_df = pd.DataFrame({
        'temperature': np.random.normal(23, 2, 100),
        'pressure': np.random.normal(1013, 50, 100),
        'humidity': np.random.beta(2, 5, 100) * 100
    })
    logger.info("Environmental data collected", dataset=results_df)
```

### Cloud Platform Logging

```python
# AWS CloudWatch Integration
from structured_logging.cloud.utils import create_cloudwatch_logger

logger = create_cloudwatch_logger(
    name="my_application",
    log_group="/aws/my-app/production", 
    region="us-east-1"
)

logger.info("Application started in AWS", 
           instance_id="i-1234567890",
           availability_zone="us-east-1a")

# Google Cloud Logging Integration  
from structured_logging.cloud.utils import create_google_cloud_logger

logger = create_google_cloud_logger(
    name="my_application",
    project_id="my-project",
    resource_type="k8s_container",
    resource_labels={
        "cluster_name": "production",
        "namespace_name": "default"
    }
)

logger.info("Application started in GKE",
           pod_name="api-7f8b9c-xyz",
           container_name="api")
```

### Network Logging

```python
from structured_logging import get_logger, LoggerConfig
from structured_logging.network_handlers import HTTPHandler, HTTPConfig

# Configure HTTP logging to centralized service
http_config = HTTPConfig(
    url="https://logs.company.com/api/v1/logs",
    auth_type="bearer",
    token="your_auth_token",
    batch_size=10,
    use_ssl=True
)

logger = get_logger("network_app", LoggerConfig(output_type="custom"))
logger.addHandler(HTTPHandler(http_config))

logger.info("Event logged to remote service", event_type="user_action")
```

### Intelligent Type Detection

```python
from structured_logging import get_logger
from structured_logging.serializers import SerializationConfig

# Enable automatic type detection
config = SerializationConfig(
    auto_detect_types=True,
    detect_datetime_strings=True,
    detect_uuid_strings=True,
    detect_json_strings=True
)

logger = get_logger("smart_app", config=config)

# These strings are automatically detected and enhanced
logger.info("Processing data", 
           timestamp="2024-01-15T10:30:00Z",  # Detected as datetime
           user_id="550e8400-e29b-41d4-a716-446655440000",  # Detected as UUID
           config_data='{"debug": true, "retries": 3}')  # Detected as JSON
```

### Lazy Serialization for Performance

```python
from structured_logging import get_logger
from structured_logging.serializers import SerializationConfig, create_lazy_serializable

config = SerializationConfig(
    enable_lazy_serialization=True,
    lazy_threshold_bytes=1000
)

logger = get_logger("performance_app", config=config)

# Large object serialized only if log actually gets written
large_dataset = create_lazy_serializable({
    "data": list(range(10000)),
    "metadata": {"size": "10K items", "type": "performance_test"}
})

logger.debug("Debug info with large data", dataset=large_dataset)  # Not serialized if DEBUG disabled
logger.info("Processing complete", summary=large_dataset)  # Serialized when written
```

### Schema Validation

```python
from structured_logging import get_logger
from structured_logging.serializers import SerializationConfig, register_validation_schema

# Define validation schema
user_schema = {
    "type": "object",
    "properties": {
        "user_id": {"type": "string", "pattern": "^user_[0-9]+$"},
        "action": {"type": "string", "enum": ["login", "logout", "purchase"]},
        "timestamp": {"type": "string", "format": "date-time"}
    },
    "required": ["user_id", "action"]
}

# Register schema globally
register_validation_schema("user_event", user_schema)

# Enable validation
config = SerializationConfig(validate_schemas=True)
logger = get_logger("validated_app", config=config)

# This will be validated against the schema
logger.info("User event", 
           schema="user_event",
           user_id="user_123", 
           action="login",
           timestamp="2024-01-15T10:30:00Z")
```

## Configuration

### Environment Variables

```bash
STRUCTURED_LOG_LEVEL=DEBUG
STRUCTURED_LOG_FORMATTER=json          # json, csv, or plain
STRUCTURED_LOG_OUTPUT=console           # console, file, both
STRUCTURED_LOG_TIMESTAMP=true
STRUCTURED_LOG_REQUEST_ID=true
STRUCTURED_LOG_USER_CONTEXT=true
```

### Scientific Data Configuration

```python
from structured_logging.serializers import SerializationConfig

config = SerializationConfig(
    # NumPy configuration
    numpy_array_max_size=100,           # Full serialization threshold
    numpy_array_precision=6,            # Floating point precision
    numpy_include_metadata=True,        # Include shape, dtype info
    numpy_compression_threshold=1000,   # Large array compression
    numpy_handle_inf_nan=True,          # Special value handling
    
    # Pandas configuration
    pandas_max_rows=5,                  # DataFrame row limit
    pandas_max_cols=10,                 # DataFrame column limit
    pandas_include_dtypes=True,         # Include data type info
    pandas_include_memory_usage=True,   # Memory statistics
    pandas_sample_method="head_tail",   # Sampling strategy
    
    # Type detection
    auto_detect_types=True,
    detect_datetime_strings=True,
    detect_uuid_strings=True,
    detect_json_strings=True,
    
    # Lazy serialization
    enable_lazy_serialization=True,
    lazy_threshold_bytes=1000,
    lazy_threshold_items=10,
    
    # Schema validation
    validate_schemas=True,
    schema_validation_mode="warn"       # strict, warn, or disabled
)
```

## Advanced Features

### Network Handlers

#### Syslog Integration (RFC 3164/5424)

```python
from structured_logging.network_handlers import SyslogHandler, SyslogConfig

# RFC 5424 Syslog with SSL
syslog_config = SyslogConfig(
    host="syslog.company.com",
    port=6514,
    facility=16,  # local0
    rfc_format="5424",
    use_ssl=True,
    verify_ssl=True,
    app_name="my_application"
)

handler = SyslogHandler(syslog_config)
logger.addHandler(handler)
```

#### HTTP API Logging

```python
from structured_logging.network_handlers import HTTPHandler, HTTPConfig

# Multiple authentication methods supported
configs = [
    HTTPConfig(auth_type="bearer", token="jwt_token"),
    HTTPConfig(auth_type="basic", username="user", password="pass"),
    HTTPConfig(auth_type="api_key", api_key="key", api_key_header="X-API-Key")
]
```

#### Raw Socket Logging

```python
from structured_logging.network_handlers import SocketHandler, SocketConfig

# TCP with connection pooling
tcp_config = SocketConfig(
    protocol="tcp",
    host="log-collector.internal",
    port=5140,
    connection_pool_size=5,
    keep_alive=True
)

# UDP for high-throughput metrics
udp_config = SocketConfig(
    protocol="udp",
    host="metrics.company.com",
    port=8125
)
```

### Custom Serializers

```python
from structured_logging.serializers import register_custom_serializer

# Register custom serializer for your types
def serialize_custom_object(obj, config):
    return {
        "custom_type": type(obj).__name__,
        "data": str(obj),
        "metadata": {"serialized_at": datetime.now().isoformat()}
    }

register_custom_serializer(MyCustomClass, serialize_custom_object)
```

## Performance

### Benchmarks (Version 0.6.0)

- **Basic Logging**: 130,000+ logs/second  
- **Scientific Data**: 25,000+ logs/second (with NumPy arrays)
- **Network Logging**: 15,000+ logs/second (HTTP batched)
- **Lazy Serialization**: 95% performance improvement for large objects
- **Schema Validation**: 45,000+ logs/second (validated)

### Memory Efficiency

- **Lazy Serialization**: Only serialize when logs are actually written
- **Intelligent Compression**: Automatic compression for large arrays/DataFrames
- **Smart Sampling**: Memory-efficient sampling for large datasets
- **Streaming Serialization**: Process large objects without loading into memory

## Framework Integrations

### FastAPI with Scientific Data

```python
from fastapi import FastAPI
from structured_logging.integrations import add_structured_logging, FastAPILoggingConfig

app = FastAPI()

# Enhanced logging with scientific data support
logging_config = FastAPILoggingConfig(
    log_request_body=True,
    log_response_body=False,
    mask_sensitive_data=True,
    include_scientific_data=True,  # Serialize NumPy/Pandas in requests
    max_body_size=10000           # Limit for large scientific payloads
)

app = add_structured_logging(app, logging_config)
```

## Examples

### Real-World Climate Data Analysis

```python
import numpy as np
import pandas as pd
from structured_logging import get_logger, request_context
from structured_logging.serializers import SerializationConfig

# Configure for climate research
config = SerializationConfig(
    numpy_array_precision=2,
    pandas_include_describe=True,
    pandas_sample_method="systematic"
)

logger = get_logger("climate_research", config=config)

with request_context(project="climate_change_2024", location="antarctica"):
    # Collect climate data
    climate_data = pd.DataFrame({
        'temperature': np.random.normal(-15, 10, 365),
        'wind_speed': np.random.exponential(8, 365),
        'humidity': np.random.beta(2, 5, 365) * 100
    })
    
    logger.info("Daily climate data collected", dataset=climate_data)
    
    # Analyze monthly trends
    monthly_avg = climate_data.groupby(pd.Grouper(freq='M')).mean()
    logger.info("Monthly analysis completed", trends=monthly_avg)
```

### Production Multi-Handler Setup

```python
from structured_logging import get_logger, LoggerConfig
from structured_logging.network_handlers import *

# Local debugging
local_logger = get_logger("app_local", LoggerConfig(
    output_type="file",
    file_config={"filename": "debug.log", "max_size_mb": 100}
))

# Centralized structured logging
http_handler = HTTPHandler(HTTPConfig(
    url="https://logs.company.com/structured",
    auth_type="bearer",
    token="prod_token",
    batch_size=50,
    fallback_to_local=True
))

# Security audit logging
audit_handler = SyslogHandler(SyslogConfig(
    host="audit.company.com",
    facility=13,  # Security messages
    use_ssl=True
))

# Add all handlers
prod_logger = get_logger("production_app", local_logger.config)
prod_logger.addHandler(http_handler)
prod_logger.addHandler(audit_handler)
```

## API Reference

### Core Functions

- `get_logger(name, config=None)`: Create a structured logger
- `log_with_context(logger, level, message, **extra)`: Log with automatic context injection
- `request_context(user_id=None, tenant_id=None, **custom_fields)`: Context manager for request scoping

### Scientific Functions

- `serialize_for_logging(obj, config=None)`: Serialize complex objects for logging
- `create_lazy_serializable(obj)`: Create lazy-serialized wrapper
- `register_custom_serializer(type_class, serializer_func)`: Register custom serializer
- `register_validation_schema(name, schema)`: Register JSON schema for validation

### Network Functions

- `SyslogHandler(config)`: RFC 3164/5424 compliant syslog handler
- `HTTPHandler(config)`: HTTP API logging with authentication
- `SocketHandler(config)`: Raw TCP/UDP socket logging

### Configuration Classes

- `SerializationConfig`: Scientific data and serialization configuration
- `SyslogConfig`, `HTTPConfig`, `SocketConfig`: Network handler configurations
- `LoggerConfig`: Main logger configuration
- `FilterConfig`: Advanced filtering configuration

## Requirements

- **Python 3.13+** (leverages modern Python features)
- **No required dependencies** (uses only Python standard library)
- **Optional dependencies**: `numpy`, `pandas`, `scipy` for scientific data support

## Development

```bash
# Install with all dependencies
pip install -e ".[dev,scientific]"

# Run tests with coverage (must maintain >80%)
pytest --cov=structured_logging --cov-fail-under=80

# Code quality checks
black src/ tests/                    # Format code
ruff check --fix src/ tests/         # Lint and fix issues  
mypy src/                           # Type checking

# Version management
bump2version patch                  # Bug fixes: 0.6.0 -> 0.6.1
bump2version minor                  # New features: 0.6.0 -> 0.7.0
bump2version major                  # Breaking changes: 0.6.0 -> 1.0.0
```

## Version History

### 🧬 **Version 0.6.0** (Latest) - Scientific & Network Integration
- ✅ **NumPy/Pandas/SciPy Integration**: Native support with intelligent serialization
- ✅ **Network Handlers**: Syslog, HTTP API, TCP/UDP with SSL/TLS support
- ✅ **Type Detection**: Automatic detection and conversion of complex data types
- ✅ **Lazy Serialization**: Memory-efficient serialization for large datasets
- ✅ **Schema Validation**: Runtime validation with auto-generated schemas
- ✅ **25,000+ scientific logs/sec**: Optimized performance for scientific computing

### 🎉 **Version 0.5.0** - Production-Ready Performance
- ✅ **Advanced Log Filtering**: LevelFilter, ContextFilter, SamplingFilter, CustomFilter
- ✅ **Smart Sampling**: Random, hash-based, and rate-limiting strategies
- ✅ **File Handlers**: Rotating file logging with gzip compression
- ✅ **FastAPI Integration**: Complete middleware with sensitive data masking
- ✅ **130,000+ logs/sec**: Exceptional performance validated by automated benchmarks

### 🚀 **Version 0.4.0** - Async Excellence
- ✅ Complete async logging support
- ✅ High-performance queue-based processing
- ✅ 40,153+ logs/second concurrent throughput

### 📈 **Future Versions**
- **0.7.0**: Enhanced cloud platform integrations (AWS CloudWatch, GCP Logging, Azure Monitor)
- **0.8.0**: Advanced monitoring and observability features
- **0.9.0**: Machine learning integration for log analysis
- **1.0.0**: Production-ready stable API with enterprise features

## Performance Tips

1. **Use lazy serialization**: Enable for large scientific datasets
2. **Configure intelligent sampling**: Use systematic sampling for Pandas data
3. **Enable network batching**: Use appropriate batch sizes for network handlers
4. **Optimize precision**: Set appropriate precision for numerical data
5. **Use schema validation selectively**: Enable only for critical data paths

```python
# Optimal configuration for scientific computing
config = SerializationConfig(
    numpy_array_max_size=50,
    numpy_compression_threshold=1000,
    pandas_sample_method="systematic",
    enable_lazy_serialization=True,
    lazy_threshold_bytes=5000,
    validate_schemas=False  # Disable for high-throughput scenarios
)
```

## Examples Repository

Comprehensive examples are available in the `examples/` directory:

- `scientific_data_examples.py` - NumPy, Pandas, and SciPy integration
- `advanced_network_examples.py` - Network logging with all handler types
- `schema_validation_examples.py` - Schema validation and type detection
- `lazy_serialization_examples.py` - Performance optimization techniques
- `type_detection_examples.py` - Intelligent type detection features

## License

MIT License - see LICENSE file for details.

---

**🔬 Perfect for Scientific Computing** | **🌐 Enterprise Network Logging** | **⚡ High-Performance Applications**