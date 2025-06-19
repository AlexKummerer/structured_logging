#!/usr/bin/env python3
"""
Network Logging Examples

Demonstrates how to use the new network handlers in Version 0.6.0:
- Syslog Handler (RFC 3164/5424)
- HTTP Handler (REST API logging)  
- Socket Handler (TCP/UDP raw logging)
"""

import time
from structured_logging import (
    get_logger,
    LoggerConfig,
    SyslogConfig,
    HTTPConfig,
    SocketConfig,
    request_context
)


def example_syslog_logging():
    """Example: Syslog logging to remote server"""
    print("🌐 Syslog Logging Example")
    print("=" * 50)
    
    # Configure Syslog handler
    syslog_config = SyslogConfig(
        host="localhost",  # Change to your syslog server
        port=514,
        facility=16,  # local0
        rfc_format="3164",  # or "5424" for newer format
        app_name="demo-app",
        include_hostname=True,
        include_timestamp=True
    )
    
    config = LoggerConfig(
        output_type="console+network",  # Both console and network
        network_config=syslog_config,
        formatter_type="json"
    )
    
    logger = get_logger("syslog_demo", config)
    
    with request_context(user_id="user123", service="payment"):
        logger.info("Payment processing started")
        logger.warning("High transaction volume detected")
        logger.error("Payment gateway timeout")
        logger.info("Payment processing completed")
    
    print("✅ Syslog messages sent (check your syslog server)")
    print()


def example_http_logging():
    """Example: HTTP API logging"""
    print("🌐 HTTP API Logging Example")
    print("=" * 50)
    
    # Configure HTTP handler with authentication
    http_config = HTTPConfig(
        url="https://httpbin.org/post",  # Test endpoint
        method="POST",
        auth_type="bearer",
        token="secret-api-token-123",
        batch_size=3,  # Send logs in batches of 3
        max_batch_time=2.0,  # Or every 2 seconds
        headers={
            "X-Service": "demo-app",
            "X-Environment": "production"
        }
    )
    
    config = LoggerConfig(
        output_type="console+network",
        network_config=http_config,
        formatter_type="json"
    )
    
    logger = get_logger("http_demo", config)
    
    # Send multiple logs to trigger batching
    with request_context(user_id="user456", tenant_id="company-abc"):
        logger.info("User login attempt")
        logger.info("Authentication successful")
        logger.info("Dashboard loaded")
        logger.warning("Slow database query detected")
        logger.info("User logout")
    
    # Give time for batching
    time.sleep(3)
    
    print("✅ HTTP logs sent in batches")
    print()


def example_socket_logging():
    """Example: Raw socket logging (TCP/UDP)"""
    print("🌐 Socket Logging Example")
    print("=" * 50)
    
    # TCP Socket configuration
    tcp_config = SocketConfig(
        host="localhost",
        port=5140,
        protocol="tcp",
        keep_alive=True,
        message_delimiter="\n",
        encoding="utf-8"
    )
    
    config = LoggerConfig(
        output_type="console+network",
        network_config=tcp_config,
        formatter_type="json"
    )
    
    logger = get_logger("socket_demo", config)
    
    with request_context(service="analytics", version="2.1.0"):
        logger.info("Analytics service started")
        logger.info("Processing batch job", batch_id=12345, records=50000)
        logger.warning("Memory usage high", memory_percent=85.2)
        logger.info("Batch job completed", duration_seconds=143.5)
    
    print("✅ Socket messages sent via TCP")
    print()


def example_udp_logging():
    """Example: UDP logging for high-volume metrics"""
    print("🌐 UDP Logging Example (High Volume)")
    print("=" * 50)
    
    # UDP Socket configuration - good for metrics
    udp_config = SocketConfig(
        host="localhost",
        port=5141,
        protocol="udp",
        udp_buffer_size=8192,
        message_delimiter="",  # No delimiter for UDP
        batch_size=1  # Send immediately
    )
    
    config = LoggerConfig(
        output_type="network",  # Only network, no console spam
        network_config=udp_config,
        formatter_type="json"
    )
    
    logger = get_logger("udp_metrics", config)
    
    # Simulate high-volume metrics
    print("Sending 100 metric logs via UDP...")
    
    with request_context(service="metrics-collector"):
        for i in range(100):
            logger.info("metric_update", 
                       metric="cpu_usage",
                       value=45.2 + i * 0.1,
                       timestamp=time.time(),
                       host=f"server-{i % 10}")
    
    print("✅ 100 UDP metrics sent")
    print()


def example_environment_config():
    """Example: Configuration via environment variables"""
    print("🌐 Environment Configuration Example")
    print("=" * 50)
    
    import os
    
    # Set environment variables
    os.environ.update({
        'STRUCTURED_LOG_OUTPUT': 'console+network',
        'STRUCTURED_LOG_NETWORK_TYPE': 'syslog',
        'STRUCTURED_LOG_SYSLOG_HOST': 'logs.company.com',
        'STRUCTURED_LOG_SYSLOG_PORT': '514',
        'STRUCTURED_LOG_APP_NAME': 'production-service',
        'STRUCTURED_LOG_SYSLOG_RFC': '5424'
    })
    
    # Auto-configure from environment
    config = LoggerConfig.from_env()
    logger = get_logger("env_demo", config)
    
    logger.info("Logger configured from environment variables")
    logger.info(f"Output type: {config.output_type}")
    logger.info(f"Network config: {type(config.network_config).__name__}")
    
    print("✅ Logger configured from environment")
    print()


def example_fallback_behavior():
    """Example: Fallback behavior when network fails"""
    print("🌐 Network Fallback Example")
    print("=" * 50)
    
    # Configure with fallback to local file
    http_config = HTTPConfig(
        url="http://nonexistent-server.invalid/logs",  # This will fail
        fallback_to_local=True,
        local_fallback_file="network_fallback.log",
        max_retries=1,
        retry_delay=0.5
    )
    
    config = LoggerConfig(
        output_type="console+network",
        network_config=http_config,
        formatter_type="json"
    )
    
    logger = get_logger("fallback_demo", config)
    
    logger.error("This should fallback to local file")
    logger.warning("Network is unreachable")
    logger.info("Application continuing with local logging")
    
    # Give time for fallback to trigger
    time.sleep(2)
    
    print("✅ Network failed, logs saved to fallback file")
    
    # Check if fallback file was created
    import os
    if os.path.exists("network_fallback.log"):
        print("📁 Fallback file created - check network_fallback.log")
    
    print()


def example_mixed_outputs():
    """Example: Multiple output types for different log levels"""
    print("🌐 Mixed Output Example")
    print("=" * 50)
    
    # Info logs to console + file
    info_config = LoggerConfig(
        output_type="console+file",
        file_config=None,  # Will use defaults
        formatter_type="json"
    )
    
    # Error logs to console + network (syslog)
    error_syslog_config = SyslogConfig(
        host="localhost",
        port=514,
        facility=16
    )
    
    error_config = LoggerConfig(
        output_type="console+network",
        network_config=error_syslog_config,
        formatter_type="json"
    )
    
    # Create different loggers for different purposes
    info_logger = get_logger("app.info", info_config)
    error_logger = get_logger("app.errors", error_config)
    
    # Log at different levels
    info_logger.info("Application started successfully")
    info_logger.info("Processing user requests")
    
    error_logger.error("Database connection failed")
    error_logger.critical("Service unavailable")
    
    print("✅ Info logs -> console+file, Error logs -> console+syslog")
    print()


def main():
    """Run all network logging examples"""
    print("🚀 Structured Logging v0.6.0 - Network Handlers Examples")
    print("=" * 60)
    print()
    
    print("📋 Available Examples:")
    print("1. Syslog logging (RFC 3164/5424)")
    print("2. HTTP API logging with batching")
    print("3. TCP socket logging")
    print("4. UDP socket logging (high volume)")
    print("5. Environment variable configuration")
    print("6. Network fallback behavior") 
    print("7. Mixed output types")
    print()
    
    # Note: Most examples won't actually send to real servers
    # They demonstrate configuration and would work with real endpoints
    
    try:
        example_syslog_logging()
        example_http_logging()
        example_socket_logging()
        example_udp_logging()
        example_environment_config()
        example_fallback_behavior()
        example_mixed_outputs()
        
        print("🎉 All examples completed!")
        print()
        print("💡 Tips:")
        print("  - Replace localhost with your actual log servers")
        print("  - Configure authentication for production HTTP endpoints")
        print("  - Use UDP for high-volume metrics, TCP for critical logs")
        print("  - Always configure fallback for production systems")
        print("  - Monitor network handler performance in production")
        
    except KeyboardInterrupt:
        print("\n⏹️  Examples interrupted")
    except Exception as e:
        print(f"❌ Example failed: {e}")


if __name__ == "__main__":
    main()