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


def create_google_cloud_logger(
    name: str,
    project_id: Optional[str] = None,
    log_name: Optional[str] = None,
    resource_type: str = "global",
    resource_labels: Optional[dict] = None,
    log_level: str = "INFO"
) -> Any:
    """
    Create a logger configured for Google Cloud Logging
    
    Args:
        name: Logger name
        project_id: Google Cloud project ID (auto-detected if not provided)
        log_name: Google Cloud log name (defaults to logger name)
        resource_type: Resource type (e.g., "global", "gce_instance", "k8s_container")
        resource_labels: Additional resource labels
        log_level: Logging level
        
    Returns:
        Configured logger instance
        
    Example:
        logger = create_google_cloud_logger(
            "my_app",
            project_id="my-project",
            resource_type="k8s_container",
            resource_labels={
                "cluster_name": "production",
                "namespace_name": "default"
            }
        )
        logger.info("Application started")
    """
    from .gcp import GoogleCloudConfig, GoogleCloudHandler
    
    # Create Google Cloud configuration
    gcp_config = GoogleCloudConfig(
        project_id=project_id,
        log_name=log_name or name,
        resource_type=resource_type,
        resource_labels=resource_labels or {},
        use_structured_logging=True,
        use_background_thread=True
    )
    
    # Create logger config
    logger_config = LoggerConfig(
        log_level=log_level,
        formatter_type="json",
        include_timestamp=True,
        include_request_id=True
    )
    
    # Get logger
    logger = get_logger(name, logger_config)
    
    # Add Google Cloud handler
    gcp_handler = GoogleCloudHandler(gcp_config)
    logger.addHandler(gcp_handler)
    
    return logger


def create_stackdriver_logger(
    name: str,
    project_id: Optional[str] = None,
    log_name: Optional[str] = None,
    log_level: str = "INFO"
) -> Any:
    """
    Create a logger configured for Stackdriver (legacy name for Google Cloud Logging)
    
    This is an alias for create_google_cloud_logger() for backward compatibility.
    """
    return create_google_cloud_logger(
        name=name,
        project_id=project_id,
        log_name=log_name,
        log_level=log_level
    )


def create_azure_monitor_logger(
    name: str,
    workspace_id: Optional[str] = None,
    workspace_key: Optional[str] = None,
    dce_endpoint: Optional[str] = None,
    dcr_immutable_id: Optional[str] = None,
    log_type: str = "StructuredLogs",
    log_level: str = "INFO"
) -> Any:
    """
    Create a logger configured for Azure Monitor
    
    Args:
        name: Logger name
        workspace_id: Log Analytics workspace ID (for direct API)
        workspace_key: Log Analytics workspace key (for direct API)
        dce_endpoint: Data Collection Endpoint URL (for DCE ingestion)
        dcr_immutable_id: Data Collection Rule immutable ID (for DCE ingestion)
        log_type: Table name in Log Analytics
        log_level: Logging level
        
    Returns:
        Configured logger instance
        
    Example:
        # Using workspace ID and key
        logger = create_azure_monitor_logger(
            "my_app",
            workspace_id="12345678-1234-1234-1234-123456789012",
            workspace_key="your_key_here",
            log_type="ApplicationLogs"
        )
        
        # Using DCE with managed identity
        logger = create_azure_monitor_logger(
            "my_app",
            dce_endpoint="https://my-dce.eastus.ingest.monitor.azure.com",
            dcr_immutable_id="dcr-1234567890abcdef"
        )
    """
    from .azure import AzureMonitorConfig, AzureMonitorHandler
    
    # Create Azure Monitor configuration
    azure_config = AzureMonitorConfig(
        workspace_id=workspace_id,
        workspace_key=workspace_key,
        dce_endpoint=dce_endpoint,
        dcr_immutable_id=dcr_immutable_id,
        log_type=log_type,
        include_cloud_role=True
    )
    
    # Create logger config
    logger_config = LoggerConfig(
        log_level=log_level,
        formatter_type="json",
        include_timestamp=True,
        include_request_id=True
    )
    
    # Get logger
    logger = get_logger(name, logger_config)
    
    # Add Azure Monitor handler
    azure_handler = AzureMonitorHandler(azure_config)
    logger.addHandler(azure_handler)
    
    return logger


def create_application_insights_logger(
    name: str,
    instrumentation_key: str,
    cloud_role_name: Optional[str] = None,
    log_level: str = "INFO"
) -> Any:
    """
    Create a logger configured for Application Insights
    
    Args:
        name: Logger name
        instrumentation_key: Application Insights instrumentation key
        cloud_role_name: Override cloud role name
        log_level: Logging level
        
    Returns:
        Configured logger instance
        
    Example:
        logger = create_application_insights_logger(
            "my_app",
            instrumentation_key="12345678-1234-1234-1234-123456789012",
            cloud_role_name="MyWebApp"
        )
    """
    from .azure import ApplicationInsightsConfig, ApplicationInsightsHandler
    
    # Create Application Insights configuration
    appinsights_config = ApplicationInsightsConfig(
        instrumentation_key=instrumentation_key,
        cloud_role_name=cloud_role_name,
        include_cloud_role=True
    )
    
    # Create logger config
    logger_config = LoggerConfig(
        log_level=log_level,
        formatter_type="json",
        include_timestamp=True,
        include_request_id=True
    )
    
    # Get logger
    logger = get_logger(name, logger_config)
    
    # Add Application Insights handler
    appinsights_handler = ApplicationInsightsHandler(appinsights_config)
    logger.addHandler(appinsights_handler)
    
    return logger