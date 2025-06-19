#!/usr/bin/env python3
"""
Advanced Network Logging Examples

This file demonstrates the advanced network logging capabilities including:
- Syslog integration (RFC 3164/5424)
- HTTP API logging with authentication
- Raw socket logging (TCP/UDP)
- Failover and reliability features
- Production deployment patterns

Features demonstrated:
- Multiple network handler types
- Authentication methods (Basic, Bearer, API Key)
- SSL/TLS configuration
- Batch processing and buffering
- Error handling and fallback strategies
"""

import asyncio
import time
from datetime import datetime

from structured_logging import get_logger, request_context
from structured_logging.network_handlers import (
    SyslogHandler, SyslogConfig,
    HTTPHandler, HTTPConfig, 
    SocketHandler, SocketConfig
)
from structured_logging.config import LoggerConfig


def demo_syslog_integration():
    """Demonstrate Syslog integration with RFC compliance"""
    print("=== Syslog Integration Demo ===")
    
    # 1. RFC 3164 Syslog (Traditional)
    print("\n1. RFC 3164 Syslog Configuration:")
    syslog_config_3164 = SyslogConfig(
        host="localhost",
        port=514,
        facility=16,  # local0
        rfc_format="3164",
        app_name="structured_app",
        include_timestamp=True,
        include_hostname=True,
        include_process_id=True,
        process_id=12345
    )
    
    syslog_handler_3164 = SyslogHandler(syslog_config_3164)
    
    # Create logger with syslog handler
    logger_config = LoggerConfig(
        name="syslog_3164_demo",
        level="INFO",
        output_type="custom"
    )
    
    logger_3164 = get_logger("syslog_3164", logger_config)
    logger_3164.addHandler(syslog_handler_3164)
    
    # Log some events
    with request_context(user_id="admin", session_id="sess_001"):
        logger_3164.info("Application started successfully", 
                        version="0.6.0", 
                        startup_time="2.3s")
        
        logger_3164.warning("High memory usage detected",
                           memory_usage="85%",
                           threshold="80%")
        
        logger_3164.error("Database connection failed",
                         database="production_db",
                         retry_count=3,
                         error_code="CONN_TIMEOUT")
    
    # 2. RFC 5424 Syslog (Modern)
    print("\n2. RFC 5424 Syslog Configuration:")
    syslog_config_5424 = SyslogConfig(
        host="syslog.company.com",
        port=6514,
        facility=16,
        rfc_format="5424",
        app_name="structured_app",
        use_ssl=True,
        verify_ssl=True
    )
    
    # Note: In production, this would connect to a real syslog server
    print(f"Configured RFC 5424 syslog to {syslog_config_5424.host}:{syslog_config_5424.port}")


def demo_http_api_logging():
    """Demonstrate HTTP API logging with different authentication methods"""
    print("\n=== HTTP API Logging Demo ===")
    
    # 1. Basic Authentication
    print("\n1. HTTP with Basic Authentication:")
    http_config_basic = HTTPConfig(
        url="https://logs.company.com/api/v1/logs",
        method="POST",
        auth_type="basic",
        username="log_user",
        password="secure_password",
        batch_size=5,
        max_batch_time=2.0,
        headers={
            "X-Application": "structured_logging",
            "X-Environment": "production"
        }
    )
    
    http_handler_basic = HTTPHandler(http_config_basic)
    
    logger_http = get_logger("http_basic", LoggerConfig(
        name="http_basic_demo",
        output_type="custom"
    ))
    logger_http.addHandler(http_handler_basic)
    
    # Log events that will be batched
    with request_context(user_id="user_123", tenant_id="tenant_456"):
        for i in range(3):
            logger_http.info(f"API request processed",
                           request_id=f"req_{i:03d}",
                           method="GET",
                           endpoint="/api/users",
                           response_time=f"{0.1 + i * 0.05:.3f}s",
                           status_code=200)
    
    # 2. Bearer Token Authentication
    print("\n2. HTTP with Bearer Token:")
    http_config_bearer = HTTPConfig(
        url="https://api.logservice.com/v2/events",
        auth_type="bearer",
        token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
        batch_size=10,
        compress_payload=True,
        content_type="application/json"
    )
    
    print(f"Configured HTTP logging to {http_config_bearer.url}")
    
    # 3. API Key Authentication
    print("\n3. HTTP with API Key:")
    http_config_apikey = HTTPConfig(
        url="https://logging.saas.com/ingest",
        auth_type="api_key",
        api_key="ak_1234567890abcdef",
        api_key_header="X-API-Key",
        batch_size=1,  # Real-time logging
        user_agent="StructuredLogging/0.6.0 (Company Internal)"
    )
    
    print(f"Configured API key logging to {http_config_apikey.url}")


def demo_socket_logging():
    """Demonstrate raw socket logging (TCP/UDP)"""
    print("\n=== Socket Logging Demo ===")
    
    # 1. TCP Socket Logging
    print("\n1. TCP Socket Configuration:")
    tcp_config = SocketConfig(
        host="log-collector.internal.com",
        port=5140,
        protocol="tcp",
        keep_alive=True,
        tcp_nodelay=True,
        connection_pool_size=3,
        message_delimiter="\n",
        encoding="utf-8"
    )
    
    tcp_handler = SocketHandler(tcp_config)
    
    logger_tcp = get_logger("tcp_socket", LoggerConfig(
        name="tcp_demo",
        output_type="custom"
    ))
    logger_tcp.addHandler(tcp_handler)
    
    # Send structured logs over TCP
    with request_context(service="payment_processor", datacenter="us-east-1"):
        logger_tcp.info("Payment processed successfully",
                       transaction_id="txn_789012",
                       amount="$149.99",
                       currency="USD",
                       payment_method="credit_card",
                       processing_time="0.234s")
        
        logger_tcp.info("Fraud check completed",
                       transaction_id="txn_789012",
                       risk_score=0.15,
                       decision="approved",
                       check_duration="0.089s")
    
    # 2. UDP Socket Logging
    print("\n2. UDP Socket Configuration:")
    udp_config = SocketConfig(
        host="metrics.company.com",
        port=8125,  # StatsD port
        protocol="udp",
        udp_buffer_size=1024,
        message_delimiter=""  # No delimiter for UDP metrics
    )
    
    print(f"Configured UDP logging to {udp_config.host}:{udp_config.port}")


def demo_reliability_features():
    """Demonstrate reliability and failover features"""
    print("\n=== Reliability Features Demo ===")
    
    # 1. Fallback Configuration
    print("\n1. Fallback to Local File:")
    reliable_config = HTTPConfig(
        url="https://unreliable-endpoint.com/logs",
        max_retries=3,
        retry_delay=1.0,
        fallback_to_local=True,
        local_fallback_file="network_fallback.log",
        timeout=5.0
    )
    
    reliable_handler = HTTPHandler(reliable_config)
    
    logger_reliable = get_logger("reliable", LoggerConfig(
        name="reliable_demo",
        output_type="custom"
    ))
    logger_reliable.addHandler(reliable_handler)
    
    # These logs will likely fall back to local file due to unreliable endpoint
    logger_reliable.info("Critical system event",
                        event_type="system_failure",
                        severity="high",
                        requires_immediate_attention=True)
    
    # 2. Buffering and Batching
    print("\n2. Buffering Configuration:")
    buffered_config = SyslogConfig(
        host="syslog.company.com",
        port=514,
        buffer_size=2048,
        batch_size=20,
        flush_interval=5.0,  # Flush every 5 seconds
        max_retries=5,
        retry_delay=2.0
    )
    
    print(f"Configured buffering: {buffered_config.batch_size} messages, "
          f"{buffered_config.flush_interval}s interval")


def demo_production_patterns():
    """Demonstrate production deployment patterns"""
    print("\n=== Production Deployment Patterns ===")
    
    # 1. Multi-Handler Setup
    print("\n1. Multi-Handler Production Setup:")
    
    # Local high-speed logging for debugging
    local_config = LoggerConfig(
        name="app_local",
        level="DEBUG",
        output_type="file",
        file_config={
            "filename": "app_debug.log",
            "max_size_mb": 100,
            "backup_count": 5
        }
    )
    
    # Centralized structured logging
    centralized_config = HTTPConfig(
        url="https://logs.company.com/api/v1/structured",
        auth_type="bearer",
        token="prod_token_here",
        batch_size=50,
        max_batch_time=10.0,
        fallback_to_local=True,
        local_fallback_file="centralized_fallback.log"
    )
    
    # Security/audit logging
    audit_config = SyslogConfig(
        host="audit.company.com",
        port=6514,
        facility=13,  # Security/audit messages
        rfc_format="5424",
        use_ssl=True,
        verify_ssl=True
    )
    
    # Create production logger with multiple handlers
    prod_logger = get_logger("production_app", local_config)
    prod_logger.addHandler(HTTPHandler(centralized_config))
    prod_logger.addHandler(SyslogHandler(audit_config))
    
    # Log with production context
    with request_context(
        user_id="prod_user_001",
        session_id="sess_abc123",
        request_id="req_xyz789",
        service_version="0.6.0",
        environment="production"
    ):
        # Business logic logging
        prod_logger.info("User authentication successful",
                        auth_method="oauth2",
                        provider="company_sso",
                        login_duration="0.456s")
        
        # Performance metrics
        prod_logger.info("Database query completed",
                        query_type="user_profile",
                        execution_time="0.023s",
                        rows_returned=1,
                        cache_hit=False)
        
        # Security event
        prod_logger.warning("Suspicious activity detected",
                           event_type="multiple_failed_logins",
                           source_ip="192.168.1.100",
                           attempt_count=5,
                           time_window="60s")
        
        # Error handling
        try:
            # Simulate an error
            raise ConnectionError("External service unavailable")
        except Exception as e:
            prod_logger.error("External service error",
                            service="payment_gateway",
                            error_type=type(e).__name__,
                            error_message=str(e),
                            recovery_action="retry_with_backoff")


async def demo_async_network_logging():
    """Demonstrate async compatibility with network handlers"""
    print("\n=== Async Network Logging Demo ===")
    
    # Configure async-compatible network logging
    async_config = HTTPConfig(
        url="https://async-logs.company.com/events",
        auth_type="api_key",
        api_key="async_key_123",
        batch_size=5,
        max_batch_time=1.0
    )
    
    logger_async = get_logger("async_demo", LoggerConfig(
        name="async_network",
        output_type="custom"
    ))
    logger_async.addHandler(HTTPHandler(async_config))
    
    # Simulate async operations with logging
    async def async_operation(operation_id):
        await asyncio.sleep(0.1)  # Simulate async work
        
        logger_async.info("Async operation completed",
                         operation_id=operation_id,
                         duration="0.1s",
                         thread_id=f"async_{operation_id}")
    
    # Run multiple async operations
    tasks = [async_operation(f"op_{i:03d}") for i in range(10)]
    await asyncio.gather(*tasks)
    
    print("Completed 10 async operations with network logging")


def demo_monitoring_integration():
    """Demonstrate integration with monitoring systems"""
    print("\n=== Monitoring Integration Demo ===")
    
    # 1. Metrics Collection Integration
    print("\n1. Metrics Integration:")
    
    # Configure for metrics collection (e.g., Prometheus)
    metrics_config = SocketConfig(
        host="prometheus-gateway.company.com",
        port=9091,
        protocol="tcp",
        message_delimiter="\n"
    )
    
    logger_metrics = get_logger("metrics", LoggerConfig(
        name="metrics_demo",
        output_type="custom"
    ))
    logger_metrics.addHandler(SocketHandler(metrics_config))
    
    # Log metrics-style events
    with request_context(service="api_server", instance="server_01"):
        logger_metrics.info("Request metrics",
                           metric_type="request_duration",
                           value=0.234,
                           method="GET",
                           endpoint="/api/users",
                           status_code=200)
        
        logger_metrics.info("Resource metrics",
                           metric_type="memory_usage",
                           value=0.78,
                           instance="server_01",
                           unit="percentage")
    
    # 2. Alert Integration
    print("\n2. Alert Integration:")
    
    # Configure for alerting system integration
    alert_config = HTTPConfig(
        url="https://alerts.company.com/webhook",
        auth_type="basic",
        username="alert_user",
        password="alert_pass",
        batch_size=1,  # Immediate alerting
        headers={
            "X-Alert-Source": "structured_logging",
            "Content-Type": "application/json"
        }
    )
    
    logger_alerts = get_logger("alerts", LoggerConfig(
        name="alert_demo",
        output_type="custom"
    ))
    logger_alerts.addHandler(HTTPHandler(alert_config))
    
    # Log alert-worthy events
    logger_alerts.critical("System critical error",
                          alert_type="system_down",
                          severity="critical",
                          affected_services=["user_api", "payment_api"],
                          estimated_impact="high",
                          requires_immediate_response=True)


if __name__ == "__main__":
    print("🌐 Advanced Network Logging Examples")
    print("===================================")
    
    try:
        demo_syslog_integration()
        demo_http_api_logging()
        demo_socket_logging()
        demo_reliability_features()
        demo_production_patterns()
        
        # Run async demo
        print("\nRunning async network logging demo...")
        asyncio.run(demo_async_network_logging())
        
        demo_monitoring_integration()
        
        print("\n✅ All network logging examples completed successfully!")
        print("\nKey Features Demonstrated:")
        print("- Syslog integration (RFC 3164/5424) with SSL support")
        print("- HTTP API logging with multiple authentication methods")
        print("- Raw socket logging (TCP/UDP) with connection pooling")
        print("- Reliability features with fallback and retry logic")
        print("- Production deployment patterns with multi-handler setup")
        print("- Async compatibility and monitoring system integration")
        
        print("\n📋 Production Checklist:")
        print("- ✅ Configure appropriate log levels for each handler")
        print("- ✅ Set up SSL/TLS for sensitive log data")
        print("- ✅ Configure fallback handlers for reliability")
        print("- ✅ Monitor network handler performance and errors")
        print("- ✅ Implement log retention and rotation policies")
        
    except Exception as e:
        print(f"\n❌ Error running network examples: {e}")
        print("Note: Most network examples require actual endpoints to be fully functional.")
        print("In production, replace placeholder URLs with real logging endpoints.")