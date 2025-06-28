"""
Utility functions for cloud logging setup
"""

from typing import Optional, Any

from ..config import LoggerConfig
from ..logger import get_logger


def create_cloudwatch_logger(
    name: str,
    log_group: str,
    log_stream: Optional[str] = None,
    region: Optional[str] = None,
    log_level: str = "INFO"
) -> Any:
    """
    Create a logger configured for AWS CloudWatch
    
    Args:
        name: Logger name
        log_group: CloudWatch log group name
        log_stream: CloudWatch log stream name (auto-generated if not provided)
        region: AWS region (uses default if not provided)
        log_level: Logging level
        
    Returns:
        Configured logger instance
        
    Example:
        logger = create_cloudwatch_logger(
            "my_app",
            log_group="/aws/my-application",
            region="us-east-1"
        )
        logger.info("Application started")
    """
    from .aws import CloudWatchConfig, CloudWatchHandler
    
    # Create CloudWatch configuration
    cloudwatch_config = CloudWatchConfig(
        log_group=log_group,
        log_stream=log_stream,
        region=region,
        batch_size=50,  # Smaller batches for more real-time logging
        flush_interval=2.0  # Flush every 2 seconds
    )
    
    # Create logger config
    logger_config = LoggerConfig(
        log_level=log_level,
        formatter_type="json",  # JSON format works well with CloudWatch
        include_timestamp=True,
        include_request_id=True
    )
    
    # Get logger
    logger = get_logger(name, logger_config)
    
    # Add CloudWatch handler
    cloudwatch_handler = CloudWatchHandler(cloudwatch_config)
    logger.addHandler(cloudwatch_handler)
    
    return logger


def create_cloud_logger_config(
    cloud_provider: str,
    **kwargs
) -> LoggerConfig:
    """
    Create a logger configuration for cloud logging
    
    Args:
        cloud_provider: Cloud provider name ("aws", "gcp", "azure")
        **kwargs: Provider-specific configuration
        
    Returns:
        LoggerConfig instance configured for cloud logging
    """
    # Common cloud logging settings
    config = LoggerConfig(
        formatter_type="json",  # JSON is best for cloud logging
        include_timestamp=True,
        include_request_id=True,
        include_user_context=True
    )
    
    # Provider-specific adjustments
    if cloud_provider == "aws":
        # AWS CloudWatch works well with JSON
        config.output_type = "console"  # CloudWatch handler will be added separately
    elif cloud_provider == "gcp":
        # Google Cloud Logging prefers structured JSON
        config.output_type = "console"
    elif cloud_provider == "azure":
        # Azure Monitor also uses JSON
        config.output_type = "console"
    
    return config