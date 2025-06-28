"""
Example of AWS CloudWatch integration with structured logging
"""

import time
from structured_logging import get_logger, LoggerConfig
from structured_logging.context import request_context
from structured_logging.cloud import CloudWatchConfig, CloudWatchHandler
from structured_logging.cloud.utils import create_cloudwatch_logger


def basic_cloudwatch_example():
    """Basic CloudWatch logging example"""
    print("=== Basic CloudWatch Example ===")
    
    # Create CloudWatch logger using helper function
    logger = create_cloudwatch_logger(
        name="my_application",
        log_group="/aws/my-app/production",
        log_stream=None,  # Auto-generated
        region="us-east-1"
    )
    
    # Log some messages
    logger.info("Application started")
    logger.warning("This is a warning", extra={"ctx_module": "auth"})
    logger.error("An error occurred", extra={
        "ctx_error_code": "AUTH_001",
        "ctx_user_id": "user123"
    })
    
    print("Logs sent to CloudWatch!")


def advanced_cloudwatch_example():
    """Advanced CloudWatch configuration example"""
    print("\n=== Advanced CloudWatch Example ===")
    
    # Create custom CloudWatch configuration
    cloudwatch_config = CloudWatchConfig(
        log_group="/aws/my-app/staging",
        log_stream="api-server",
        region="eu-west-1",
        # Performance tuning
        batch_size=200,  # Larger batches
        flush_interval=10.0,  # Flush every 10 seconds
        max_queue_size=50000,  # Large queue for high volume
        # Retry settings
        max_retries=5,
        exponential_backoff=True,
        # Features
        compress_logs=True,
        create_log_group=True
    )
    
    # Create logger with custom config
    logger_config = LoggerConfig(
        log_level="DEBUG",
        formatter_type="json",
        include_timestamp=True,
        include_request_id=True,
        include_user_context=True
    )
    
    logger = get_logger("api_server", logger_config)
    
    # Add CloudWatch handler
    cloudwatch_handler = CloudWatchHandler(cloudwatch_config)
    logger.addHandler(cloudwatch_handler)
    
    # Log with context
    with request_context(user_id="user456", tenant_id="acme-corp"):
        logger.info("Processing API request", extra={
            "ctx_endpoint": "/api/v1/users",
            "ctx_method": "GET"
        })
        
        # Simulate some processing
        time.sleep(0.1)
        
        logger.info("API request completed", extra={
            "ctx_status_code": 200,
            "ctx_response_time_ms": 105
        })
    
    # Ensure logs are flushed
    for handler in logger.handlers:
        if hasattr(handler, 'flush'):
            handler.flush()
    
    print("Advanced logs sent to CloudWatch!")


def batch_logging_example():
    """Example of batch logging for high volume"""
    print("\n=== Batch Logging Example ===")
    
    # Configure for high-volume batch logging
    config = CloudWatchConfig(
        log_group="/aws/my-app/batch-jobs",
        batch_size=500,  # Large batches
        flush_interval=30.0,  # Flush every 30 seconds
        compress_logs=True,  # Compress for efficiency
        async_upload=True  # Async for performance
    )
    
    # Create logger
    logger = get_logger("batch_processor")
    handler = CloudWatchHandler(config)
    logger.addHandler(handler)
    
    # Simulate high-volume logging
    print("Sending 1000 log entries...")
    for i in range(1000):
        logger.info(f"Processing item {i}", extra={
            "ctx_item_id": f"item_{i}",
            "ctx_batch_id": "batch_001",
            "ctx_progress": i / 10  # Percentage
        })
        
        if i % 100 == 0:
            print(f"  Processed {i} items...")
    
    # Flush remaining logs
    handler.flush()
    print("Batch logs sent to CloudWatch!")


def error_tracking_example():
    """Example of error tracking with CloudWatch"""
    print("\n=== Error Tracking Example ===")
    
    logger = create_cloudwatch_logger(
        "error_tracker",
        log_group="/aws/my-app/errors",
        region="us-west-2"
    )
    
    # Simulate various error scenarios
    try:
        # Simulate database error
        raise ConnectionError("Database connection failed")
    except ConnectionError as e:
        logger.error("Database error occurred", extra={
            "ctx_error_type": "database",
            "ctx_error_message": str(e),
            "ctx_retry_count": 3,
            "ctx_database": "users_db"
        }, exc_info=True)
    
    try:
        # Simulate API error
        raise ValueError("Invalid API response")
    except ValueError as e:
        logger.error("External API error", extra={
            "ctx_error_type": "api",
            "ctx_api_endpoint": "https://api.example.com/users",
            "ctx_status_code": 500,
            "ctx_response_time_ms": 2500
        }, exc_info=True)
    
    print("Error logs sent to CloudWatch!")


def main():
    """Run all examples"""
    print("AWS CloudWatch Logging Examples")
    print("=" * 50)
    
    # Note: These examples require AWS credentials to be configured
    # You can set them via:
    # - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    # - AWS credentials file (~/.aws/credentials)
    # - IAM role (when running on EC2/Lambda)
    
    try:
        basic_cloudwatch_example()
        advanced_cloudwatch_example()
        batch_logging_example()
        error_tracking_example()
    except ImportError:
        print("\nError: boto3 is not installed.")
        print("Install with: pip install boto3")
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure AWS credentials are configured.")


if __name__ == "__main__":
    main()