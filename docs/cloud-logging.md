# Cloud Logging Integration

The structured logging library provides native integration with major cloud logging services, allowing you to send structured logs directly to cloud platforms with automatic batching, compression, and retry logic.

## AWS CloudWatch Logs

### Installation

To use AWS CloudWatch integration, install with the AWS extras:

```bash
pip install structured-logging[aws]
```

This will install the required `boto3` dependency.

### Quick Start

The simplest way to start logging to CloudWatch:

```python
from structured_logging.cloud.utils import create_cloudwatch_logger

# Create a CloudWatch logger
logger = create_cloudwatch_logger(
    name="my_application",
    log_group="/aws/my-app/production",
    region="us-east-1"
)

# Start logging
logger.info("Application started")
logger.error("An error occurred", extra={"error_code": "E001"})
```

### Configuration

#### Basic Configuration

```python
from structured_logging import get_logger
from structured_logging.cloud import CloudWatchConfig, CloudWatchHandler

# Configure CloudWatch
config = CloudWatchConfig(
    log_group="/aws/my-app/production",
    log_stream="api-server-1",  # Optional - auto-generated if not provided
    region="us-east-1",
    create_log_group=True,  # Auto-create log group if needed
    create_log_stream=True  # Auto-create log stream if needed
)

# Create logger and add CloudWatch handler
logger = get_logger("my_app")
cloudwatch_handler = CloudWatchHandler(config)
logger.addHandler(cloudwatch_handler)
```

#### Advanced Configuration

```python
config = CloudWatchConfig(
    # CloudWatch settings
    log_group="/aws/my-app/production",
    log_stream="api-server-1",
    region="us-east-1",
    
    # Batching configuration
    batch_size=200,  # Number of logs per batch (default: 100)
    flush_interval=10.0,  # Seconds between flushes (default: 5.0)
    max_queue_size=50000,  # Max logs in queue (default: 10000)
    
    # Performance settings
    compress_logs=True,  # Compress log batches (default: True)
    async_upload=True,  # Upload in background (default: True)
    
    # Retry configuration
    max_retries=5,  # Max retry attempts (default: 3)
    retry_delay=1.0,  # Initial retry delay (default: 1.0)
    exponential_backoff=True,  # Use exponential backoff (default: True)
    
    # AWS credentials (optional - uses boto3 credential chain if not provided)
    aws_access_key_id="YOUR_ACCESS_KEY",
    aws_secret_access_key="YOUR_SECRET_KEY",
    aws_session_token="SESSION_TOKEN"  # For temporary credentials
)
```

### Authentication

The CloudWatch handler uses the standard boto3 credential chain:

1. **Explicit credentials** (if provided in config)
2. **Environment variables** (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
3. **AWS credentials file** (`~/.aws/credentials`)
4. **IAM role** (when running on EC2, Lambda, ECS, etc.)

### Integration with Context

CloudWatch handler works seamlessly with the context system:

```python
from structured_logging.context import request_context

with request_context(user_id="user123", request_id="req456"):
    logger.info("Processing request")
    # Logs will include user_id and request_id fields
```

### Structured Output

Logs are sent to CloudWatch as JSON with all context fields included:

```json
{
  "message": "Processing request",
  "level": "INFO",
  "logger": "my_app",
  "module": "main",
  "funcName": "process_request",
  "lineno": 42,
  "user_id": "user123",
  "request_id": "req456",
  "timestamp": "2025-06-28T12:34:56.789Z"
}
```

### Best Practices

#### 1. Log Group Organization

Use a hierarchical structure for log groups:

```python
# Environment-based
log_group = f"/aws/my-app/{environment}"  # /aws/my-app/production

# Service-based
log_group = f"/aws/{service}/{component}"  # /aws/api/auth-service

# Combined
log_group = f"/aws/{service}/{environment}/{component}"
```

#### 2. Log Stream Naming

Let the handler auto-generate stream names for unique identification:

```python
# Auto-generated format: hostname_YYYYMMDD_HHMMSS
# Example: api-server-1_20250628_123456
```

Or use meaningful names:

```python
config = CloudWatchConfig(
    log_stream=f"{service_name}-{instance_id}"
)
```

#### 3. Batching Strategy

For high-volume applications:

```python
config = CloudWatchConfig(
    batch_size=500,  # Larger batches
    flush_interval=30.0,  # Less frequent flushes
    max_queue_size=100000,  # Large queue
    compress_logs=True  # Enable compression
)
```

For real-time monitoring:

```python
config = CloudWatchConfig(
    batch_size=10,  # Small batches
    flush_interval=1.0,  # Frequent flushes
    async_upload=True  # Keep async for performance
)
```

#### 4. Error Handling

The handler automatically retries on failures:

```python
config = CloudWatchConfig(
    max_retries=5,
    retry_delay=2.0,
    exponential_backoff=True  # 2s, 4s, 8s, 16s, 32s
)
```

#### 5. Cost Optimization

- Use batching to reduce API calls
- Enable compression for large logs
- Set appropriate retention policies in AWS
- Use sampling for high-volume debug logs

### Monitoring

Monitor CloudWatch handler performance:

```python
# Check handler statistics
handler = cloudwatch_handler
print(f"Queue size: {handler.queue.qsize()}")
print(f"Worker running: {handler._worker_thread.is_alive()}")
```

### CloudWatch Insights

The JSON format works perfectly with CloudWatch Insights:

```sql
-- Find errors by user
fields @timestamp, message, user_id, error_code
| filter level = "ERROR"
| filter user_id = "user123"
| sort @timestamp desc

-- Response time analysis
fields @timestamp, response_time_ms
| filter funcName = "handle_request"
| stats avg(response_time_ms), max(response_time_ms), min(response_time_ms) by bin(5m)

-- Error rate by endpoint
fields @timestamp, path, level
| filter level = "ERROR"
| stats count() by path
```

### Troubleshooting

#### No logs appearing in CloudWatch

1. Check AWS credentials:
```python
import boto3
client = boto3.client('logs', region_name='us-east-1')
client.describe_log_groups()  # Should not raise an error
```

2. Check permissions - IAM policy needs:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    }
  ]
}
```

3. Enable debug logging:
```python
import logging
logging.getLogger('boto3').setLevel(logging.DEBUG)
logging.getLogger('botocore').setLevel(logging.DEBUG)
```

#### Sequence token errors

These are handled automatically, but if persistent:

```python
# Force new log stream
config = CloudWatchConfig(
    log_stream=f"api-server-{int(time.time())}"
)
```

#### High memory usage

Reduce queue size and batch size:

```python
config = CloudWatchConfig(
    max_queue_size=1000,  # Smaller queue
    batch_size=50,  # Smaller batches
    flush_interval=2.0  # More frequent flushes
)
```

## Coming Soon

### Google Cloud Logging

```python
from structured_logging.cloud import StackdriverConfig, StackdriverHandler

config = StackdriverConfig(
    project_id="my-project",
    log_name="my-app"
)
```

### Azure Monitor

```python
from structured_logging.cloud import AzureMonitorConfig, AzureMonitorHandler

config = AzureMonitorConfig(
    instrumentation_key="your-key",
    log_type="ApplicationLogs"
)
```