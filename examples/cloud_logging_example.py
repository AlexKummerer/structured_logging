"""
Example: Cloud Platform Logging with AWS CloudWatch

This example demonstrates how to use the structured logging library with AWS CloudWatch.
"""

import os
import time
from unittest.mock import patch
from structured_logging import get_logger, request_context
from structured_logging.cloud import CloudWatchConfig, CloudWatchHandler
from structured_logging.cloud.utils import create_cloudwatch_logger


def basic_cloudwatch_example():
    """Basic CloudWatch logging example"""
    print("\n=== Basic CloudWatch Example ===")
    
    # Create a CloudWatch logger with minimal configuration
    logger = create_cloudwatch_logger(
        name="my_application",
        log_group="/aws/my-app/demo",
        region="us-east-1"  # Change to your region
    )
    
    # Log some messages
    logger.info("Application started")
    logger.warning("This is a warning message")
    logger.error("An error occurred", extra={"error_code": "E001"})
    
    print("✓ Logs sent to CloudWatch")


def advanced_cloudwatch_example():
    """Advanced CloudWatch configuration example"""
    print("\n=== Advanced CloudWatch Example ===")
    
    # Create custom CloudWatch configuration
    config = CloudWatchConfig(
        log_group="/aws/my-app/production",
        log_stream="api-server-1",
        region="us-east-1",
        
        # Batching configuration
        batch_size=50,  # Send logs in batches of 50
        flush_interval=2.0,  # Flush every 2 seconds
        
        # Performance settings
        compress_logs=True,  # Compress large batches
        async_upload=True,  # Non-blocking uploads
        
        # Retry configuration
        max_retries=5,
        retry_delay=1.0,
        exponential_backoff=True
    )
    
    # Create logger and add CloudWatch handler
    logger = get_logger("production_app")
    cloudwatch_handler = CloudWatchHandler(config)
    logger.addHandler(cloudwatch_handler)
    
    # Log with structured data
    logger.info("API request received", extra={
        "path": "/api/v1/users",
        "method": "GET",
        "response_time_ms": 145.2
    })
    
    print("✓ Advanced CloudWatch logger configured")


def context_aware_cloudwatch_example():
    """Example using context management with CloudWatch"""
    print("\n=== Context-Aware CloudWatch Example ===")
    
    logger = create_cloudwatch_logger(
        name="context_app",
        log_group="/aws/my-app/context-demo"
    )
    
    # Log with request context
    with request_context(
        user_id="user_123",
        tenant_id="acme_corp",
        request_id="req_456"
    ):
        logger.info("Processing user request")
        
        # Simulate some processing
        time.sleep(0.1)
        
        logger.info("Request completed", extra={
            "ctx_duration_ms": 100,
            "ctx_status": "success"
        })
    
    print("✓ Context-aware logs sent to CloudWatch")


def batch_logging_example():
    """Example demonstrating batch logging performance"""
    print("\n=== Batch Logging Example ===")
    
    # Configure for high-volume logging
    config = CloudWatchConfig(
        log_group="/aws/my-app/batch-demo",
        batch_size=500,  # Large batch size
        flush_interval=5.0,  # Less frequent flushes
        compress_logs=True,
        async_upload=True
    )
    
    logger = get_logger("batch_logger")
    handler = CloudWatchHandler(config)
    logger.addHandler(handler)
    
    # Log many messages quickly
    print("Sending 100 log messages...")
    for i in range(100):
        logger.info(f"Batch message {i}", extra={
            "message_id": i,
            "batch_test": True
        })
    
    # Force flush to send remaining logs
    handler.flush()
    
    print("✓ Batch logs sent to CloudWatch")


def error_tracking_example():
    """Example for error tracking and monitoring"""
    print("\n=== Error Tracking Example ===")
    
    logger = create_cloudwatch_logger(
        name="error_tracker",
        log_group="/aws/my-app/errors",
        log_level="ERROR"  # Only log errors and above
    )
    
    try:
        # Simulate an error
        result = 1 / 0
    except Exception as e:
        logger.error("Division by zero error", exc_info=True, extra={
            "ctx_operation": "calculate_result",
            "ctx_input_values": {"numerator": 1, "denominator": 0}
        })
    
    print("✓ Error logs sent to CloudWatch")


def cloudwatch_insights_example():
    """Example optimized for CloudWatch Insights queries"""
    print("\n=== CloudWatch Insights Example ===")
    
    logger = create_cloudwatch_logger(
        name="insights_app",
        log_group="/aws/my-app/insights"
    )
    
    # Log structured data optimized for Insights
    for i in range(10):
        logger.info("API call", extra={
            "ctx_endpoint": f"/api/v1/resource/{i}",
            "ctx_method": "GET" if i % 2 == 0 else "POST",
            "ctx_response_time_ms": 100 + i * 10,
            "ctx_status_code": 200 if i < 8 else 500,
            "ctx_user_type": "premium" if i % 3 == 0 else "standard"
        })
    
    print("✓ Insights-optimized logs sent to CloudWatch")
    print("\nExample CloudWatch Insights queries:")
    print("""
    1. Average response time by endpoint:
       fields @timestamp, endpoint, response_time_ms
       | filter endpoint like /api/
       | stats avg(response_time_ms) by endpoint
    
    2. Error rate over time:
       fields @timestamp, status_code
       | filter status_code >= 400
       | stats count() by bin(5m)
    
    3. Top users by request count:
       fields @timestamp, user_type
       | stats count() by user_type
    """)


def multi_region_example():
    """Example with multi-region logging"""
    print("\n=== Multi-Region Example ===")
    
    regions = ["us-east-1", "eu-west-1", "ap-southeast-1"]
    
    for region in regions:
        logger = create_cloudwatch_logger(
            name=f"global_app_{region}",
            log_group=f"/aws/global-app/{region}",
            region=region
        )
        
        logger.info(f"Service running in {region}", extra={
            "ctx_region": region,
            "ctx_availability_zone": f"{region}a"
        })
    
    print("✓ Logs sent to multiple regions")


def google_cloud_basic_example():
    """Basic Google Cloud Logging example"""
    print("\n=== Basic Google Cloud Example ===")
    
    from structured_logging.cloud.utils import create_google_cloud_logger
    
    # Create a Google Cloud logger
    logger = create_google_cloud_logger(
        name="my_application",
        project_id="my-project",  # Optional - uses default project
        log_name="demo-logs"
    )
    
    # Log some messages
    logger.info("Application started on Google Cloud")
    logger.warning("This is a warning message")
    logger.error("An error occurred", extra={"error_code": "E001"})
    
    print("✓ Logs sent to Google Cloud Logging")


def google_cloud_kubernetes_example():
    """Google Cloud Logging with Kubernetes resource"""
    print("\n=== Google Cloud Kubernetes Example ===")
    
    from structured_logging.cloud import GoogleCloudConfig, GoogleCloudHandler
    
    # Configure for Kubernetes
    config = GoogleCloudConfig(
        project_id="my-project",
        log_name="k8s-app",
        resource_type="k8s_container",
        resource_labels={
            "cluster_name": "production-cluster",
            "namespace_name": "default",
            "pod_name": "api-server-7f8b9c-xyz",
            "container_name": "api"
        }
    )
    
    logger = get_logger("k8s_app")
    handler = GoogleCloudHandler(config)
    logger.addHandler(handler)
    
    # Log with Kubernetes context
    logger.info("Pod started", extra={
        "ctx_version": "v1.2.3",
        "ctx_replicas": 3,
        "ctx_node": "gke-node-1"
    })
    
    print("✓ Kubernetes logs sent to Google Cloud")


def google_cloud_trace_example():
    """Google Cloud Logging with trace correlation"""
    print("\n=== Google Cloud Trace Example ===")
    
    from structured_logging.cloud.utils import create_google_cloud_logger
    
    logger = create_google_cloud_logger(
        name="traced_app",
        project_id="my-project"
    )
    
    # Simulate distributed tracing
    trace_id = "1234567890abcdef"
    
    # Log with trace ID for correlation
    for i in range(3):
        # In real app, get trace from context
        logger.info(f"Processing step {i}", extra={
            "trace_id": f"projects/my-project/traces/{trace_id}",
            "span_id": f"span_{i}",
            "ctx_step": i
        })
    
    print("✓ Traced logs sent to Google Cloud")


def stackdriver_legacy_example():
    """Example using legacy Stackdriver naming"""
    print("\n=== Stackdriver Legacy Example ===")
    
    from structured_logging.cloud.utils import create_stackdriver_logger
    
    # Use legacy Stackdriver function (alias for Google Cloud)
    logger = create_stackdriver_logger(
        name="legacy_app",
        project_id="my-project"
    )
    
    logger.info("Using Stackdriver legacy naming")
    
    print("✓ Logs sent via Stackdriver alias")


def azure_monitor_basic_example():
    """Basic Azure Monitor example"""
    print("\n=== Basic Azure Monitor Example ===")
    
    from structured_logging.cloud.utils import create_azure_monitor_logger
    
    # Create Azure Monitor logger with workspace
    logger = create_azure_monitor_logger(
        name="my_application",
        workspace_id="12345678-1234-1234-1234-123456789012",
        workspace_key="your_workspace_key_here",  # Primary or secondary key
        log_type="ApplicationLogs"  # Table name in Log Analytics
    )
    
    # Log some messages
    logger.info("Application started on Azure")
    logger.warning("This is a warning message")
    logger.error("An error occurred", extra={"error_code": "E001"})
    
    print("✓ Logs sent to Azure Monitor")


def azure_dce_example():
    """Azure Monitor with Data Collection Endpoint"""
    print("\n=== Azure DCE Example ===")
    
    from structured_logging.cloud import AzureMonitorConfig, AzureMonitorHandler
    
    # Configure for DCE with managed identity
    config = AzureMonitorConfig(
        dce_endpoint="https://my-dce.eastus.ingest.monitor.azure.com",
        dcr_immutable_id="dcr-1234567890abcdef",
        stream_name="Custom-StructuredLogs",
        include_cloud_role=True
    )
    
    logger = get_logger("dce_app")
    handler = AzureMonitorHandler(config)
    logger.addHandler(handler)
    
    # Log with Azure context
    logger.info("Using Data Collection Endpoint", extra={
        "ctx_subscription": "Production",
        "ctx_region": "East US",
        "ctx_managed_identity": True
    })
    
    print("✓ Logs sent via DCE")


def application_insights_example():
    """Application Insights example"""
    print("\n=== Application Insights Example ===")
    
    from structured_logging.cloud.utils import create_application_insights_logger
    
    # Create Application Insights logger
    logger = create_application_insights_logger(
        name="insights_app",
        instrumentation_key="12345678-1234-1234-1234-123456789012",
        cloud_role_name="MyWebApp"
    )
    
    # Log application telemetry
    logger.info("Page loaded", extra={
        "ctx_page": "/products",
        "ctx_duration_ms": 234,
        "ctx_user_agent": "Mozilla/5.0"
    })
    
    logger.error("Database connection failed", extra={
        "ctx_database": "products-db",
        "ctx_retry_count": 3
    })
    
    print("✓ Telemetry sent to Application Insights")


def azure_service_detection_example():
    """Azure service auto-detection example"""
    print("\n=== Azure Service Detection Example ===")
    
    import os
    from structured_logging.cloud import AzureMonitorConfig, AzureMonitorHandler
    
    # Simulate Azure App Service environment
    with patch.dict('os.environ', {
        'WEBSITE_SITE_NAME': 'my-web-app',
        'WEBSITE_INSTANCE_ID': 'instance-001'
    }):
        config = AzureMonitorConfig(
            workspace_id="test-workspace",
            workspace_key="test-key"
        )
        
        handler = AzureMonitorHandler(config)
        
        # Cloud role is auto-detected
        print(f"✓ Detected cloud role: {handler.cloud_role_name}")
        print(f"✓ Detected instance: {handler.cloud_role_instance}")


if __name__ == "__main__":
    print("=== Cloud Platform Logging Examples ===")
    print("\nSelect platform:")
    print("1. AWS CloudWatch")
    print("2. Google Cloud Logging")
    print("3. Azure Monitor")
    print("4. All platforms")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice in ["1", "4"]:
        print("\n=== AWS CloudWatch Examples ===")
        print("\nNOTE: These examples require:")
        print("1. AWS credentials configured (via env vars, ~/.aws/credentials, or IAM role)")
        print("2. Appropriate permissions to create log groups/streams and put log events")
        print("3. The 'boto3' package installed: pip install structured-logging[aws]")
        
        # Check if boto3 is available
        try:
            import boto3
            print("\n✓ boto3 is installed")
        except ImportError:
            print("\n✗ boto3 not found. Install with: pip install structured-logging[aws]")
            if choice == "1":
                exit(1)
        else:
            # Run AWS examples
            try:
                basic_cloudwatch_example()
                advanced_cloudwatch_example()
                context_aware_cloudwatch_example()
                batch_logging_example()
                error_tracking_example()
                cloudwatch_insights_example()
                multi_region_example()
                
                print("\n=== AWS examples completed ===")
                print("Check your AWS CloudWatch console to see the logs!")
                
            except Exception as e:
                print(f"\n✗ Error running AWS examples: {e}")
                print("\nMake sure you have:")
                print("1. Valid AWS credentials configured")
                print("2. Permissions to write to CloudWatch Logs")
                print("3. Network connectivity to AWS")
    
    if choice in ["2", "4"]:
        print("\n=== Google Cloud Logging Examples ===")
        print("\nNOTE: These examples require:")
        print("1. Google Cloud credentials (service account, gcloud auth, or metadata)")
        print("2. Appropriate permissions (logging.logEntries.create, logging.logs.write)")
        print("3. The 'google-cloud-logging' package: pip install structured-logging[gcp]")
        
        # Check if google-cloud-logging is available
        try:
            import google.cloud.logging
            print("\n✓ google-cloud-logging is installed")
        except ImportError:
            print("\n✗ google-cloud-logging not found. Install with: pip install structured-logging[gcp]")
            exit(1)
        
        # Run Google Cloud examples
        try:
            google_cloud_basic_example()
            google_cloud_kubernetes_example()
            google_cloud_trace_example()
            stackdriver_legacy_example()
            
            print("\n=== Google Cloud examples completed ===")
            print("Check your Google Cloud Console Logs Explorer to see the logs!")
            
        except Exception as e:
            print(f"\n✗ Error running Google Cloud examples: {e}")
            print("\nMake sure you have:")
            print("1. Valid Google Cloud credentials configured")
            print("2. A Google Cloud project set")
            print("3. Permissions to write logs")
            print("4. Network connectivity to Google Cloud")
    
    if choice in ["3", "4"]:
        print("\n=== Azure Monitor Examples ===")
        print("\nNOTE: These examples require:")
        print("1. Azure credentials (workspace key, managed identity, or service principal)")
        print("2. Log Analytics workspace or Application Insights resource")
        print("3. The 'requests' package: pip install structured-logging[azure]")
        
        # Check if requests is available
        try:
            import requests
            print("\n✓ requests is installed")
        except ImportError:
            print("\n✗ requests not found. Install with: pip install structured-logging[azure]")
            if choice == "3":
                exit(1)
        else:
            # Run Azure examples
            try:
                azure_monitor_basic_example()
                azure_dce_example()
                application_insights_example()
                azure_service_detection_example()
                
                print("\n=== Azure examples completed ===")
                print("Check your Azure Monitor Log Analytics workspace or Application Insights to see the logs!")
                
            except Exception as e:
                print(f"\n✗ Error running Azure examples: {e}")
                print("\nMake sure you have:")
                print("1. Valid Azure credentials configured")
                print("2. A Log Analytics workspace with valid key")
                print("3. Or Application Insights with instrumentation key")
                print("4. Network connectivity to Azure")