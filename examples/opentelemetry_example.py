"""
Example: OpenTelemetry Integration with Structured Logging

This example demonstrates how to use structured logging with OpenTelemetry
for automatic trace correlation, span context injection, and observability.
"""

import time

from structured_logging import get_logger
from structured_logging.integrations import (
    HAS_OPENTELEMETRY,
    add_otel_handler_to_logger,
    configure_otel_logging,
    create_otel_logger,
    logged_span,
)

# Check if OpenTelemetry is available
if not HAS_OPENTELEMETRY:
    print(
        "❌ OpenTelemetry not available. "
        "Install with: pip install structured-logging[otel]"
    )
    exit(1)

# OpenTelemetry setup imports
from opentelemetry import baggage, trace
from opentelemetry.exporter.console import ConsoleSpanExporter
from opentelemetry.sdk.resources import (
    DEPLOYMENT_ENVIRONMENT,
    SERVICE_NAME,
    SERVICE_VERSION,
    Resource,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def setup_opentelemetry():
    """Set up OpenTelemetry tracing for the examples"""
    print("🔧 Setting up OpenTelemetry...")

    # Create resource with service information
    resource = Resource.create(
        {
            SERVICE_NAME: "structured-logging-demo",
            SERVICE_VERSION: "1.0.0",
            DEPLOYMENT_ENVIRONMENT: "development",
            "service.namespace": "examples",
            "service.instance.id": "demo-001",
        }
    )

    # Set up tracer provider
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)

    # Add console exporter for demo purposes
    console_exporter = ConsoleSpanExporter()
    span_processor = BatchSpanProcessor(console_exporter)
    tracer_provider.add_span_processor(span_processor)

    print("✅ OpenTelemetry configured")
    return trace.get_tracer(__name__)


def basic_otel_logging_example():
    """Basic OpenTelemetry logging with trace correlation"""
    print("\n=== Basic OTel Logging Example ===")

    # Create logger with OpenTelemetry integration
    logger = create_otel_logger(
        "basic_demo",
        enable_trace_correlation=True,
        include_resource_attributes=True,
        include_span_attributes=True,
    )

    tracer = trace.get_tracer(__name__)

    # Create a span and log within it
    with tracer.start_as_current_span("user-registration") as span:
        # Add span attributes
        span.set_attribute("user.id", "user_12345")
        span.set_attribute("user.email", "demo@example.com")
        span.set_attribute("operation.type", "registration")

        # Log messages - will automatically include trace/span IDs
        logger.info("Starting user registration process")

        # Simulate some work
        time.sleep(0.1)

        logger.info(
            "Validating user data",
            extra={
                "ctx_user_id": "user_12345",
                "ctx_validation_rules": ["email", "age", "terms"],
            },
        )

        # Simulate validation success
        logger.info(
            "User registration completed successfully",
            extra={"ctx_user_id": "user_12345", "ctx_duration_ms": 100},
        )

    print("✅ Trace correlation logs generated")


def span_attributes_example():
    """Example showing span attribute injection into logs"""
    print("\n=== Span Attributes Example ===")

    logger = create_otel_logger(
        "span_attrs_demo",
        include_span_attributes=True,
        span_attribute_prefix="span.",
        max_span_attributes=10,
    )

    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("http-request") as span:
        # Add HTTP-related span attributes
        span.set_attribute("http.method", "POST")
        span.set_attribute("http.url", "https://api.example.com/users")
        span.set_attribute("http.status_code", 201)
        span.set_attribute("http.user_agent", "structured-logging-demo/1.0")
        span.set_attribute("custom.request_id", "req_789012")

        # These span attributes will automatically appear in logs
        logger.info("Processing HTTP request")

        # Nested span with different attributes
        with tracer.start_as_current_span("database-query") as db_span:
            db_span.set_attribute("db.system", "postgresql")
            db_span.set_attribute("db.name", "users_db")
            db_span.set_attribute("db.operation", "INSERT")
            db_span.set_attribute("db.table", "users")

            logger.info("Executing database query")

        logger.info("HTTP request completed")

    print("✅ Span attributes automatically injected into logs")


def resource_attributes_example():
    """Example showing resource attribute injection"""
    print("\n=== Resource Attributes Example ===")

    logger = create_otel_logger(
        "resource_demo",
        include_resource_attributes=True,
        resource_attribute_prefix="resource.",
        # Only include specific resource attributes
        allowed_resource_attributes=[
            "service.name",
            "service.version",
            "deployment.environment",
        ],
    )

    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("application-startup"):
        # Resource attributes are automatically added to all logs
        logger.info("Application starting up")
        logger.info(
            "Loading configuration",
            extra={"ctx_config_source": "environment", "ctx_config_count": 15},
        )
        logger.info("Application ready to serve requests")

    print("✅ Resource attributes included in all logs")


def error_handling_example():
    """Example showing error handling and span status setting"""
    print("\n=== Error Handling Example ===")

    logger = create_otel_logger(
        "error_demo",
        set_span_status_on_error=True,  # Automatically set span status on errors
        enable_trace_correlation=True,
    )

    tracer = trace.get_tracer(__name__)

    # Successful operation
    with tracer.start_as_current_span("payment-processing") as span:
        span.set_attribute("payment.amount", 99.99)
        span.set_attribute("payment.currency", "USD")
        span.set_attribute("payment.method", "credit_card")

        logger.info(
            "Processing payment",
            extra={"ctx_payment_id": "pay_456789", "ctx_amount": 99.99},
        )

        # Simulate error condition
        try:
            if True:  # Simulate error condition
                raise ValueError("Insufficient funds")

            logger.info("Payment processed successfully")

        except ValueError as e:
            # Error logs automatically set span status to ERROR
            logger.error(
                "Payment processing failed",
                extra={
                    "ctx_payment_id": "pay_456789",
                    "ctx_error_code": "INSUFFICIENT_FUNDS",
                    "ctx_error_message": str(e),
                },
                exc_info=True,
            )

    print("✅ Error automatically set span status and logged")


def logging_span_example():
    """Example using LoggingSpan context manager"""
    print("\n=== LoggingSpan Context Manager Example ===")

    logger = create_otel_logger("span_context_demo")

    # LoggingSpan automatically logs span start/end with duration
    with logged_span(logger, "data-processing"):
        logger.info("Starting data processing")

        # Simulate some processing time
        time.sleep(0.1)

        logger.info(
            "Processing chunk 1/3", extra={"ctx_chunk_id": 1, "ctx_records_count": 1000}
        )

        time.sleep(0.05)

        logger.info(
            "Processing chunk 2/3", extra={"ctx_chunk_id": 2, "ctx_records_count": 850}
        )

        time.sleep(0.03)

        logger.info(
            "Processing chunk 3/3", extra={"ctx_chunk_id": 3, "ctx_records_count": 650}
        )

        logger.info("Data processing completed")

    print("✅ Span lifecycle automatically logged with duration")


def custom_attributes_example():
    """Example with custom attribute mapping and filtering"""
    print("\n=== Custom Attributes Example ===")

    # Configure custom attribute mappings
    config = configure_otel_logging(
        service_name="custom-demo",
        service_version="2.0.0",
        environment="staging",
        attribute_mapping={
            "levelname": "log_level",
            "pathname": "source_file",
            "funcName": "function_name",
        },
        exclude_attributes=["thread", "threadName", "processName"],
        max_attribute_length=100,
    )

    logger = create_otel_logger("custom_demo", **config.__dict__)

    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("custom-operation"):
        # Logs will have custom attribute mappings applied
        logger.warning(
            "Custom attribute mapping in action",
            extra={
                "ctx_operation": "custom_demo",
                "ctx_long_field": "x" * 200,  # Will be truncated
                "ctx_metadata": {
                    "version": "2.0.0",
                    "feature_flags": ["new_ui", "analytics"],
                    "config": {"debug": True, "timeout": 30},
                },
            },
        )

    print("✅ Custom attribute mappings and filtering applied")


def baggage_example():
    """Example showing baggage propagation with logging"""
    print("\n=== Baggage Propagation Example ===")

    logger = create_otel_logger("baggage_demo")
    tracer = trace.get_tracer(__name__)

    # Set baggage (cross-cutting concerns)
    ctx = baggage.set_baggage("user.id", "user_98765")
    ctx = baggage.set_baggage("request.source", "mobile_app", ctx)
    ctx = baggage.set_baggage("feature.flag", "new_checkout", ctx)

    with tracer.start_as_current_span("checkout-process", context=ctx) as span:
        # Get baggage values for logging
        user_id = baggage.get_baggage("user.id")
        source = baggage.get_baggage("request.source")
        feature_flag = baggage.get_baggage("feature.flag")

        span.set_attribute("checkout.user_id", user_id)
        span.set_attribute("checkout.source", source)

        logger.info(
            "Starting checkout process",
            extra={
                "ctx_user_id": user_id,
                "ctx_source": source,
                "ctx_feature_flag": feature_flag,
            },
        )

        # Nested operations inherit baggage
        with tracer.start_as_current_span("inventory-check"):
            logger.info(
                "Checking inventory",
                extra={
                    "ctx_user_id": baggage.get_baggage(
                        "user.id"
                    ),  # Inherited from parent
                    "ctx_items": ["item_1", "item_2"],
                },
            )

        with tracer.start_as_current_span("payment-authorization"):
            logger.info(
                "Authorizing payment",
                extra={
                    "ctx_user_id": baggage.get_baggage(
                        "user.id"
                    ),  # Inherited from parent
                    "ctx_payment_method": "credit_card",
                },
            )

        logger.info("Checkout completed successfully")

    print("✅ Baggage values propagated across spans and logged")


def existing_logger_integration():
    """Example adding OTel to existing structured logger"""
    print("\n=== Existing Logger Integration Example ===")

    # Create existing structured logger
    existing_logger = get_logger("existing_app")

    # Add OpenTelemetry handler to existing logger
    add_otel_handler_to_logger(existing_logger)

    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("legacy-integration"):
        # Existing logger now has OTel capabilities
        existing_logger.info("Legacy application with OTel integration")
        existing_logger.warning(
            "This log now includes trace correlation",
            extra={
                "ctx_legacy_field": "legacy_value",
                "ctx_migration_status": "in_progress",
            },
        )

    print("✅ Existing logger enhanced with OpenTelemetry")


def performance_example():
    """Example showing performance considerations"""
    print("\n=== Performance Example ===")

    # Configure for high-performance scenarios
    logger = create_otel_logger(
        "performance_demo",
        only_when_tracing=True,  # Only add OTel data when actively tracing
        minimum_log_level="INFO",  # Skip debug logs
        max_span_attributes=5,  # Limit span attributes
        max_attribute_length=200,  # Shorter attribute truncation
    )

    tracer = trace.get_tracer(__name__)

    # High-volume logging scenario
    with tracer.start_as_current_span("high-volume-operation"):
        start_time = time.time()

        for i in range(100):
            logger.info(
                f"Processing item {i}",
                extra={"ctx_item_id": f"item_{i}", "ctx_batch_id": "batch_001"},
            )

        duration = time.time() - start_time
        logger.info(
            "High-volume processing completed",
            extra={
                "ctx_items_processed": 100,
                "ctx_duration_seconds": duration,
                "ctx_items_per_second": 100 / duration,
            },
        )

    print(f"✅ Processed 100 logs with OTel integration in {duration:.3f}s")


if __name__ == "__main__":
    print("=== OpenTelemetry Integration Examples ===")
    print("This example demonstrates structured logging with OpenTelemetry integration")

    # Set up OpenTelemetry for the examples
    tracer = setup_opentelemetry()

    # Run all examples
    try:
        basic_otel_logging_example()
        span_attributes_example()
        resource_attributes_example()
        error_handling_example()
        logging_span_example()
        custom_attributes_example()
        baggage_example()
        existing_logger_integration()
        performance_example()

        print("\n=== All examples completed successfully! ===")
        print("\nKey OpenTelemetry Integration Features:")
        print("✅ Automatic trace and span ID injection")
        print("✅ Resource attribute propagation")
        print("✅ Span attribute injection")
        print("✅ Error status correlation")
        print("✅ Baggage propagation")
        print("✅ Custom attribute mapping")
        print("✅ Performance optimizations")
        print("✅ Existing logger enhancement")

        print("\nNext Steps:")
        print("1. Set up distributed tracing with Jaeger or Zipkin")
        print("2. Configure OpenTelemetry Collector")
        print("3. Add instrumentation for your frameworks (HTTP, databases, etc.)")
        print("4. Set up alerting based on trace correlation")
        print("5. Use structured logs for observability dashboards")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("\nTroubleshooting:")
        print(
            "1. Ensure OpenTelemetry is installed: pip install structured-logging[otel]"
        )
        print("2. Check OpenTelemetry SDK version compatibility")
        print("3. Verify resource configuration")
        print("4. Enable debug logging for OpenTelemetry")

