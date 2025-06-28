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

## Google Cloud Logging (Stackdriver)

### Installation

To use Google Cloud Logging integration, install with the GCP extras:

```bash
pip install structured-logging[gcp]
```

This will install the required `google-cloud-logging` dependency.

### Quick Start

The simplest way to start logging to Google Cloud:

```python
from structured_logging.cloud.utils import create_google_cloud_logger

# Create a Google Cloud logger
logger = create_google_cloud_logger(
    name="my_application",
    project_id="my-project",  # Optional - auto-detected from environment
    resource_type="gce_instance"
)

# Start logging
logger.info("Application started")
logger.error("An error occurred", extra={"error_code": "E001"})
```

### Configuration

#### Basic Configuration

```python
from structured_logging import get_logger
from structured_logging.cloud import GoogleCloudConfig, GoogleCloudHandler

# Configure Google Cloud Logging
config = GoogleCloudConfig(
    project_id="my-project",  # Optional - uses GOOGLE_CLOUD_PROJECT env var
    log_name="my-application",
    resource_type="global",  # Or "gce_instance", "k8s_container", etc.
    resource_labels={
        "environment": "production",
        "version": "1.0.0"
    }
)

# Create logger and add Google Cloud handler
logger = get_logger("my_app")
gcp_handler = GoogleCloudHandler(config)
logger.addHandler(gcp_handler)
```

#### Advanced Configuration

```python
config = GoogleCloudConfig(
    # Google Cloud settings
    project_id="my-project",
    log_name="my-application",
    resource_type="k8s_container",
    resource_labels={
        "cluster_name": "production-cluster",
        "namespace_name": "default",
        "pod_name": "api-server-1",
        "container_name": "api"
    },
    
    # Authentication
    credentials_path="/path/to/service-account.json",  # Optional
    
    # Performance settings
    use_background_thread=True,  # Background uploads (default: True)
    grace_period=10.0,  # Seconds to wait on shutdown (default: 5.0)
    
    # Structured logging
    use_structured_logging=True,  # Send as JSON (default: True)
    include_trace_id=True,  # Include trace for correlation (default: True)
    
    # Batching (from base config)
    batch_size=100,
    flush_interval=5.0,
    async_upload=True
)
```

### Authentication

The Google Cloud handler uses Application Default Credentials (ADC):

1. **Service Account** (if `credentials_path` provided)
2. **Environment variable** (`GOOGLE_APPLICATION_CREDENTIALS`)
3. **gcloud auth** (local development)
4. **GCE/GKE/Cloud Run metadata** (when running on Google Cloud)

### Resource Types

Common resource types for different Google Cloud environments:

```python
# Global (generic)
config = GoogleCloudConfig(resource_type="global")

# Compute Engine
config = GoogleCloudConfig(
    resource_type="gce_instance",
    resource_labels={
        "instance_id": "1234567890",
        "zone": "us-central1-a"
    }
)

# Kubernetes Engine
config = GoogleCloudConfig(
    resource_type="k8s_container",
    resource_labels={
        "cluster_name": "my-cluster",
        "namespace_name": "production",
        "pod_name": "api-7f8b9c-xyz",
        "container_name": "api-server"
    }
)

# Cloud Run
config = GoogleCloudConfig(
    resource_type="cloud_run_revision",
    resource_labels={
        "service_name": "my-service",
        "revision_name": "my-service-00001-abc",
        "location": "us-central1"
    }
)

# App Engine
config = GoogleCloudConfig(
    resource_type="gae_app",
    resource_labels={
        "module_id": "default",
        "version_id": "20240101t120000"
    }
)
```

### Integration with Context

Google Cloud handler works seamlessly with the context system:

```python
from structured_logging.context import request_context

with request_context(user_id="user123", request_id="req456"):
    logger.info("Processing request")
    # Logs will include user_id and request_id fields
```

### Structured Output

Logs are sent to Google Cloud as structured JSON:

```json
{
  "message": "Processing request",
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

#### 1. Resource Labels

Use meaningful resource labels for better organization:

```python
# Development
resource_labels = {
    "environment": "development",
    "developer": "john.doe"
}

# Production
resource_labels = {
    "environment": "production",
    "region": "us-central1",
    "service": "api",
    "version": "v1.2.3"
}
```

#### 2. Log Names

Use hierarchical log names:

```python
# By service
log_name = "api-service"
log_name = "worker-service"

# By environment
log_name = "production/api"
log_name = "staging/api"

# By component
log_name = "api/auth"
log_name = "api/payments"
```

#### 3. Trace Correlation

Enable trace correlation for distributed systems:

```python
config = GoogleCloudConfig(
    include_trace_id=True
)

# In your code
record.trace_id = f"projects/{project_id}/traces/{trace_id}"
```

#### 4. Performance Optimization

For high-volume applications:

```python
config = GoogleCloudConfig(
    use_background_thread=True,  # Non-blocking writes
    batch_size=200,  # Larger batches
    flush_interval=10.0,  # Less frequent flushes
)
```

For real-time monitoring:

```python
config = GoogleCloudConfig(
    use_background_thread=True,
    batch_size=10,  # Small batches
    flush_interval=1.0,  # Frequent flushes
)
```

### Monitoring

Monitor Google Cloud handler performance:

```python
# Check if background thread is running
if handler._transport:
    print("Background transport is active")
```

### Cloud Logging Query Examples

Use the Logs Explorer in Google Cloud Console:

```sql
-- Find errors by user
resource.type="k8s_container"
severity>=ERROR
jsonPayload.user_id="user123"

-- Response time analysis
resource.type="k8s_container"
jsonPayload.response_time_ms>1000

-- Group by endpoint
resource.type="k8s_container"
jsonPayload.endpoint=~"/api/.*"
| group by jsonPayload.endpoint
```

### Troubleshooting

#### No logs appearing in Google Cloud

1. Check authentication:
```python
from google.cloud import logging
client = logging.Client()
client.list_entries()  # Should not raise an error
```

2. Check project ID:
```bash
gcloud config get-value project
```

3. Enable API:
```bash
gcloud services enable logging.googleapis.com
```

4. Check permissions - service account needs:
- `logging.logEntries.create`
- `logging.logs.write`

#### Authentication errors

1. Set credentials explicitly:
```python
config = GoogleCloudConfig(
    credentials_path="/path/to/service-account.json"
)
```

2. Or use environment variable:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

#### High memory usage

Use background thread mode (default) and tune grace period:

```python
config = GoogleCloudConfig(
    use_background_thread=True,
    grace_period=2.0  # Shorter grace period
)
```

### Legacy Stackdriver Support

For backward compatibility, Stackdriver aliases are provided:

```python
from structured_logging.cloud import StackdriverConfig, StackdriverHandler
from structured_logging.cloud.utils import create_stackdriver_logger

# These are aliases for Google Cloud classes
config = StackdriverConfig(project_id="my-project")
handler = StackdriverHandler(config)
logger = create_stackdriver_logger("my_app")
```

## Azure Monitor

### Installation

To use Azure Monitor integration, install with the Azure extras:

```bash
pip install structured-logging[azure]
```

This will install the required dependencies:
- `requests` for Log Analytics and Application Insights APIs
- `azure-monitor-ingestion` for Data Collection Endpoint (DCE)
- `azure-identity` for managed identity authentication

### Quick Start

Azure Monitor supports three ingestion methods:

#### 1. Log Analytics Workspace (Direct API)

```python
from structured_logging.cloud.utils import create_azure_monitor_logger

# Create an Azure Monitor logger
logger = create_azure_monitor_logger(
    name="my_application",
    workspace_id="12345678-1234-1234-1234-123456789012",
    workspace_key="your_workspace_key_here",
    log_type="ApplicationLogs"  # Table name in Log Analytics
)

# Start logging
logger.info("Application started")
logger.error("An error occurred", extra={"error_code": "E001"})
```

#### 2. Data Collection Endpoint (DCE) with Managed Identity

```python
from structured_logging.cloud.utils import create_azure_monitor_logger

# Using DCE with managed identity (recommended for production)
logger = create_azure_monitor_logger(
    name="my_application",
    dce_endpoint="https://my-dce.eastus.ingest.monitor.azure.com",
    dcr_immutable_id="dcr-1234567890abcdef"
)

logger.info("Running on Azure with managed identity")
```

#### 3. Application Insights

```python
from structured_logging.cloud.utils import create_application_insights_logger

# Create an Application Insights logger
logger = create_application_insights_logger(
    name="my_application",
    instrumentation_key="12345678-1234-1234-1234-123456789012",
    cloud_role_name="MyWebApp"
)

logger.info("Application Insights tracking enabled")
```

### Configuration

#### Basic Configuration

```python
from structured_logging import get_logger
from structured_logging.cloud import AzureMonitorConfig, AzureMonitorHandler

# Configure Azure Monitor
config = AzureMonitorConfig(
    # Choose one authentication method:
    
    # Option 1: Log Analytics workspace
    workspace_id="your-workspace-id",
    workspace_key="your-workspace-key",
    
    # Option 2: Data Collection Endpoint
    # dce_endpoint="https://my-dce.region.ingest.monitor.azure.com",
    # dcr_immutable_id="dcr-immutable-id",
    
    # Option 3: Application Insights
    # instrumentation_key="your-instrumentation-key",
    
    # Common settings
    log_type="StructuredLogs",  # Table name
    include_cloud_role=True
)

# Create logger and add handler
logger = get_logger("my_app")
azure_handler = AzureMonitorHandler(config)
logger.addHandler(azure_handler)
```

#### Advanced Configuration

```python
config = AzureMonitorConfig(
    # Authentication
    workspace_id="your-workspace-id",
    workspace_key="your-workspace-key",
    
    # Service principal auth (for DCE)
    # tenant_id="your-tenant-id",
    # client_id="your-client-id",
    # client_secret="your-client-secret",
    
    # Log settings
    log_type="ApplicationLogs",
    time_field="TimeGenerated",  # Timestamp field name
    
    # Performance
    use_compression=True,  # Compress large payloads
    batch_size=200,
    flush_interval=10.0,
    async_upload=True,
    
    # Cloud role information
    include_cloud_role=True,
    cloud_role_name="MyAPIService",  # Override auto-detection
    cloud_role_instance="api-server-1",
    
    # Retry configuration
    max_retries=5,
    retry_delay=2.0,
    exponential_backoff=True
)
```

### Authentication Methods

#### 1. Workspace Key (Direct API)
```python
config = AzureMonitorConfig(
    workspace_id="12345678-1234-1234-1234-123456789012",
    workspace_key="your_primary_or_secondary_key"
)
```

#### 2. Managed Identity (DCE)
```python
config = AzureMonitorConfig(
    dce_endpoint="https://my-dce.eastus.ingest.monitor.azure.com",
    dcr_immutable_id="dcr-1234567890abcdef",
    use_managed_identity=True  # Default
)
```

#### 3. Service Principal (DCE)
```python
config = AzureMonitorConfig(
    dce_endpoint="https://my-dce.eastus.ingest.monitor.azure.com",
    dcr_immutable_id="dcr-1234567890abcdef",
    tenant_id="your-tenant-id",
    client_id="your-client-id",
    client_secret="your-client-secret"
)
```

#### 4. Application Insights Key
```python
config = ApplicationInsightsConfig(
    instrumentation_key="12345678-1234-1234-1234-123456789012"
)
```

### Azure Service Integration

The handler automatically detects the Azure service environment:

```python
# Automatic detection for:
# - Azure App Service (WEBSITE_SITE_NAME)
# - Azure Functions (FUNCTIONS_WORKER_RUNTIME)
# - Azure Container Apps (CONTAINER_APP_NAME)
# - Azure VMs (hostname)

# Override if needed:
config = AzureMonitorConfig(
    workspace_id="...",
    workspace_key="...",
    cloud_role_name="CustomServiceName",
    cloud_role_instance="custom-instance-1"
)
```

### Integration with Context

Azure Monitor handler works seamlessly with the context system:

```python
from structured_logging.context import request_context

with request_context(user_id="user123", subscription_tier="premium"):
    logger.info("Processing premium user request")
    # Logs will include user_id and subscription_tier in Properties
```

### Structured Output

Logs are sent to Azure Monitor with structured fields:

```json
{
  "TimeGenerated": "2025-06-28T12:34:56.789Z",
  "Message": "Processing premium user request",
  "SeverityLevel": 1,
  "Logger": "my_app",
  "Module": "main",
  "Function": "process_request",
  "Line": 42,
  "CloudRoleName": "MyAPIService",
  "CloudRoleInstance": "api-server-1",
  "Properties": {
    "user_id": "user123",
    "subscription_tier": "premium"
  }
}
```

### Best Practices

#### 1. Choose the Right Ingestion Method

- **DCE with Managed Identity**: Best for production Azure workloads
- **Log Analytics Direct API**: Good for external services or testing
- **Application Insights**: Best for application telemetry and tracing

#### 2. Log Type Naming

Use meaningful log type names (table names):

```python
# By service
log_type = "APILogs"
log_type = "WorkerLogs"

# By environment
log_type = "ProductionLogs"
log_type = "StagingLogs"

# By component
log_type = "AuthenticationLogs"
log_type = "PaymentLogs"
```

#### 3. Custom Properties

Add structured data as properties:

```python
logger.info("Order processed", extra={
    "ctx_order_id": "ORD-12345",
    "ctx_amount": 99.99,
    "ctx_currency": "USD",
    "ctx_items_count": 3
})
```

#### 4. Performance Optimization

For high-volume applications:

```python
config = AzureMonitorConfig(
    workspace_id="...",
    workspace_key="...",
    use_compression=True,  # Compress payloads
    batch_size=500,  # Large batches
    flush_interval=30.0,  # Less frequent flushes
    async_upload=True  # Non-blocking
)
```

For real-time monitoring:

```python
config = AzureMonitorConfig(
    workspace_id="...",
    workspace_key="...",
    batch_size=10,  # Small batches
    flush_interval=1.0,  # Frequent flushes
)
```

### Monitoring with KQL

Query logs using Kusto Query Language (KQL) in Azure Monitor:

```kql
// Find errors by user
StructuredLogs_CL
| where SeverityLevel >= 3
| where Properties.user_id == "user123"
| project TimeGenerated, Message, Properties

// Response time analysis
ApplicationLogs_CL
| where Properties.response_time_ms > 1000
| summarize avg(Properties.response_time_ms) by bin(TimeGenerated, 5m)

// Error rate by cloud role
StructuredLogs_CL
| where SeverityLevel >= 3
| summarize ErrorCount = count() by CloudRoleName, bin(TimeGenerated, 1h)
| render timechart
```

### Troubleshooting

#### No logs appearing in Azure Monitor

1. Check authentication:
   - Workspace: Verify workspace ID and key
   - DCE: Check managed identity or service principal permissions
   - App Insights: Verify instrumentation key

2. Check permissions:
   - For DCE: `Microsoft.Insights/dataCollectionRules/data/write`
   - For workspace: Key must be valid and not expired

3. Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Authentication errors

1. For workspace API:
```python
# Verify key is base64 encoded
import base64
key = base64.b64encode(b"your_key").decode('utf-8')
```

2. For managed identity:
```bash
# Check identity is assigned
az identity show --name <identity-name> --resource-group <rg>
```

#### High costs

1. Use batching and compression:
```python
config = AzureMonitorConfig(
    use_compression=True,
    batch_size=1000,
    flush_interval=60.0
)
```

2. Set appropriate retention in Azure Monitor
3. Use sampling for high-volume debug logs

### Application Insights Integration

For full application telemetry:

```python
from structured_logging.cloud import ApplicationInsightsConfig

config = ApplicationInsightsConfig(
    instrumentation_key="your-key",
    cloud_role_name="MyWebApp",
    cloud_role_instance="web-1"
)

# Logs appear in Application Insights traces
# Can correlate with requests, dependencies, and exceptions
```