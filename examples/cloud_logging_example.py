"""
Example: Cloud Platform Logging with AWS CloudWatch

This example demonstrates how to use the structured logging library with AWS CloudWatch.
"""

import os
import time
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


if __name__ == "__main__":
    print("=== CloudWatch Logging Examples ===")
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
        exit(1)
    
    # Run examples
    try:
        basic_cloudwatch_example()
        advanced_cloudwatch_example()
        context_aware_cloudwatch_example()
        batch_logging_example()
        error_tracking_example()
        cloudwatch_insights_example()
        multi_region_example()
        
        print("\n=== All examples completed ===")
        print("\nCheck your AWS CloudWatch console to see the logs!")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        print("\nMake sure you have:")
        print("1. Valid AWS credentials configured")
        print("2. Permissions to write to CloudWatch Logs")
        print("3. Network connectivity to AWS")