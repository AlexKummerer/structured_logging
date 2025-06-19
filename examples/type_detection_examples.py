#!/usr/bin/env python3
"""
Automatic Type Detection Examples

Demonstrates the new automatic type detection and conversion features in Version 0.6.0:
- Intelligent string analysis and conversion
- UUID, datetime, URL, and JSON detection
- Numeric pattern recognition
- Configurable detection rules
- Performance optimization with caching
"""

import time
from datetime import datetime
from decimal import Decimal

from structured_logging import (
    LoggerConfig,
    SerializationConfig,
    SmartConverter,
    TypeDetector,
    get_logger,
    log_with_context,
    request_context,
    serialize_for_logging,
)


def example_basic_string_detection():
    """Example: Basic string type detection"""
    print("🔍 Basic String Type Detection Example")
    print("=" * 50)

    # Create logger with auto-detection enabled
    config = SerializationConfig(auto_detect_types=True)
    logger_config = LoggerConfig(formatter_type="json")
    logger = get_logger("type_detection_demo", logger_config)

    # Override the formatter to use our detection config
    from structured_logging.formatter import StructuredFormatter

    for handler in logger.handlers:
        if isinstance(handler.formatter, StructuredFormatter):
            handler.formatter.serialization_config = config

    with request_context(service="type_detection", operation="string_analysis"):
        # These strings will be automatically detected and enhanced!
        log_with_context(
            logger,
            "info",
            "Processing user request",
            user_id="f47ac10b-58cc-4372-a567-0e02b2c3d479",  # → UUID detection
            created_at="2024-01-15T10:30:00Z",  # → DateTime detection
            api_endpoint="https://api.example.com/users/123",  # → URL detection
            config_json='{"theme": "dark", "notifications": true}',  # → JSON detection
            invoice_number="INV-2024-001234",  # → Regular string
            amount_str="199.99",  # → Numeric detection
            count_str="42",  # → Integer detection
        )

    print("✅ Strings automatically analyzed and enhanced!")
    print("   - UUIDs detected and validated")
    print("   - DateTime strings parsed and marked")
    print("   - URLs identified and categorized")
    print("   - JSON strings parsed and structured")
    print("   - Numeric strings converted with metadata")
    print()


def example_numeric_analysis():
    """Example: Numeric value analysis and enhancement"""
    print("🔢 Numeric Analysis Example")
    print("=" * 50)

    config = SerializationConfig(auto_detect_types=True, auto_convert_numbers=True)
    logger = get_logger("numeric_demo")

    with request_context(analysis_type="financial"):
        # These numbers will be analyzed and enhanced
        log_with_context(
            logger,
            "info",
            "Transaction processed",
            timestamp_unix=1705316400,  # → Timestamp detection
            amount_cents=150000,  # → Large number formatting
            user_count=1500000,  # → "1.50M" formatting
            file_size=2048576,  # → "2.05M" formatting
            nano_precision=0.000001234,  # → Scientific notation
            percentage=0.1567,  # → Regular float
        )

        log_with_context(
            logger,
            "warning",
            "Large dataset detected",
            records=5000000,  # → "5.00M"
            memory_bytes=1073741824,  # → "1.07B"
            processing_time=1234567890,  # → Timestamp conversion
        )

    print("✅ Numeric values analyzed and enhanced!")
    print("   - Unix timestamps converted to human-readable")
    print("   - Large numbers formatted with units (K, M, B, T)")
    print("   - Scientific notation preserved")
    print("   - Context-aware formatting")
    print()


def example_selective_detection():
    """Example: Configuring selective type detection"""
    print("⚙️ Selective Detection Configuration Example")
    print("=" * 50)

    # Example 1: Only UUID and URL detection
    security_config = SerializationConfig(
        auto_detect_types=True,
        detect_uuid_strings=True,
        detect_url_strings=True,
        detect_datetime_strings=False,  # Disabled for security logs
        detect_json_strings=False,  # Disabled to prevent injection
        auto_convert_numbers=False,
    )

    # Example 2: Only JSON and datetime detection for API logs
    api_config = SerializationConfig(
        auto_detect_types=True,
        detect_uuid_strings=False,
        detect_url_strings=False,
        detect_datetime_strings=True,
        detect_json_strings=True,
        auto_convert_strings=True,
    )

    logger = get_logger("selective_demo")

    test_data = {
        "user_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
        "timestamp": "2024-01-15T10:30:00Z",
        "endpoint": "https://api.example.com/users",
        "request_body": '{"action": "login", "remember": true}',
        "session_duration": "3600",
    }

    print("🔒 Security Config (UUIDs + URLs only):")
    security_result = serialize_for_logging(test_data, security_config)
    print(f"   - UUIDs detected: {isinstance(security_result['user_id'], dict)}")
    print(f"   - URLs detected: {isinstance(security_result['endpoint'], dict)}")
    print(f"   - DateTime ignored: {isinstance(security_result['timestamp'], str)}")
    print(f"   - JSON ignored: {isinstance(security_result['request_body'], str)}")

    print("\n🌐 API Config (DateTime + JSON only):")
    api_result = serialize_for_logging(test_data, api_config)
    print(f"   - UUIDs ignored: {isinstance(api_result['user_id'], str)}")
    print(f"   - URLs ignored: {isinstance(api_result['endpoint'], str)}")
    print(f"   - DateTime detected: {isinstance(api_result['timestamp'], dict)}")
    print(f"   - JSON detected: {isinstance(api_result['request_body'], dict)}")
    print()


def example_type_detector_direct_usage():
    """Example: Using TypeDetector directly for analysis"""
    print("🎯 Direct TypeDetector Usage Example")
    print("=" * 50)

    config = SerializationConfig(auto_detect_types=True, type_detection_cache_size=100)
    detector = TypeDetector(config)

    test_strings = [
        "f47ac10b-58cc-4372-a567-0e02b2c3d479",  # UUID
        "2024-01-15T10:30:00Z",  # ISO DateTime
        "https://api.example.com/v1/users/123?active=true",  # URL
        '{"name": "John", "age": 30, "active": true}',  # JSON
        "123456789",  # Integer string
        "99.99",  # Float string
        "1.23e-4",  # Scientific notation
        "regular string without special pattern",  # Normal string
        "",  # Empty string
    ]

    print("🔬 Analyzing various string patterns:")
    for i, test_string in enumerate(test_strings, 1):
        result = detector.detect_and_convert(test_string)

        if isinstance(result, dict) and "type" in result:
            detected_type = result["type"]
            confidence = "High" if detected_type in ["uuid", "url"] else "Medium"
            print(f"   {i}. '{test_string[:30]}...' → {detected_type} ({confidence})")
        else:
            print(f"   {i}. '{test_string[:30]}...' → No pattern detected")

    # Show cache performance
    stats = detector.get_cache_stats()
    print("\n📊 Cache Performance:")
    print(f"   - Cache size: {stats['cache_size']}")
    print(f"   - Hit rate: {stats['hit_rate_percent']:.1f}%")
    print(f"   - Total requests: {stats['total_requests']}")
    print()


def example_smart_converter_intelligence():
    """Example: SmartConverter with enhanced intelligence"""
    print("🧠 Smart Converter Intelligence Example")
    print("=" * 50)

    config = SerializationConfig(auto_detect_types=True)
    converter = SmartConverter(config)

    # Test different types of data
    test_cases = [
        ("UUID string", "f47ac10b-58cc-4372-a567-0e02b2c3d479"),
        ("Timestamp number", 1705316400),
        ("Large number", 1500000),
        ("JSON string", '{"users": [{"id": 1, "name": "John"}]}'),
        ("DateTime object", datetime(2024, 1, 15, 10, 30, 0)),
        ("Decimal object", Decimal("99.99")),
        ("URL string", "https://api.example.com/v2/users"),
        ("Regular string", "just a normal string"),
    ]

    print("🎯 Converting various data types intelligently:")
    for name, data in test_cases:
        result = converter.convert_intelligently(data)

        if isinstance(result, dict) and "_detection" in result:
            # Auto-detected with enhancement
            confidence = result["_detection"]["confidence"]
            detected_type = result.get("type", "unknown")
            print(
                f"   ✨ {name}: Auto-detected as {detected_type} (confidence: {confidence:.2f})"
            )
        elif isinstance(result, dict) and "type" not in result:
            # Standard serialization result
            print(f"   🔧 {name}: Standard serialization applied")
        else:
            # Primitive or unchanged
            print(f"   📝 {name}: Kept as {type(result).__name__}")

    # Show detection statistics
    stats = converter.get_detection_stats()
    print("\n📈 Detection Statistics:")
    print(f"   - Total analyses: {stats['total_requests']}")
    print(f"   - Cache efficiency: {stats['hit_rate_percent']:.1f}%")
    print()


def example_performance_comparison():
    """Example: Performance comparison with/without detection"""
    print("⚡ Performance Impact Analysis")
    print("=" * 50)

    # Test data with various detectable patterns
    test_data = {
        "user_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
        "session_start": "2024-01-15T10:30:00Z",
        "api_endpoint": "https://api.example.com/users/profile",
        "request_payload": '{"action": "update", "fields": ["name", "email"]}',
        "response_time": "0.125",
        "status_code": "200",
        "user_agent": "MyApp/1.0 (iOS 15.0)",
        "timestamp": 1705316400,
        "file_size": 2048576,
    }

    iterations = 1000

    # Test without detection
    config_no_detection = SerializationConfig(auto_detect_types=False)
    start_time = time.perf_counter()
    for _ in range(iterations):
        serialize_for_logging(test_data, config_no_detection)
    time_no_detection = time.perf_counter() - start_time

    # Test with detection
    config_with_detection = SerializationConfig(auto_detect_types=True)
    start_time = time.perf_counter()
    for _ in range(iterations):
        serialize_for_logging(test_data, config_with_detection)
    time_with_detection = time.perf_counter() - start_time

    # Analysis
    overhead_ratio = time_with_detection / time_no_detection
    overhead_percent = (overhead_ratio - 1) * 100

    print(f"📊 Performance Analysis ({iterations:,} iterations):")
    print(
        f"   Without detection: {time_no_detection:.3f}s ({time_no_detection/iterations*1000:.2f}ms per op)"
    )
    print(
        f"   With detection:    {time_with_detection:.3f}s ({time_with_detection/iterations*1000:.2f}ms per op)"
    )
    print(f"   Overhead:          {overhead_percent:.1f}% ({overhead_ratio:.2f}x)")

    if overhead_percent < 50:
        print("   🚀 Excellent! Detection adds minimal overhead")
    elif overhead_percent < 100:
        print("   ✅ Good! Detection overhead is reasonable")
    elif overhead_percent < 200:
        print("   ⚠️  Moderate overhead - consider selective detection")
    else:
        print("   ❌ High overhead - optimize detection rules")

    print()


def example_real_world_scenarios():
    """Example: Real-world scenarios where detection helps"""
    print("🌍 Real-World Detection Scenarios")
    print("=" * 50)

    config = SerializationConfig(auto_detect_types=True)
    logger = get_logger("real_world_demo")

    # Override formatter config
    from structured_logging.formatter import StructuredFormatter

    for handler in logger.handlers:
        if isinstance(handler.formatter, StructuredFormatter):
            handler.formatter.serialization_config = config

    print("📧 Scenario 1: Email Processing Service")
    with request_context(service="email_processor"):
        log_with_context(
            logger,
            "info",
            "Email processed",
            message_id="550e8400-e29b-41d4-a716-446655440000",  # Auto-detected UUID
            sent_at="2024-01-15T14:30:00Z",  # Auto-detected datetime
            webhook_url="https://webhook.example.com/email/sent",  # Auto-detected URL
            recipient_count="1250",  # Auto-detected number
            template_data='{"name": "John", "discount": "20%"}',  # Auto-detected JSON
        )

    print("🛒 Scenario 2: E-commerce Analytics")
    with request_context(service="analytics"):
        log_with_context(
            logger,
            "info",
            "Purchase analyzed",
            order_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            purchase_time="2024-01-15T16:45:30+01:00",
            total_amount="299.99",
            items_json='[{"id": 123, "qty": 2}, {"id": 456, "qty": 1}]',
            customer_segment="premium",
        )

    print("🔐 Scenario 3: Security Event Logging")
    with request_context(service="security"):
        log_with_context(
            logger,
            "warning",
            "Suspicious activity detected",
            session_id="f1e2d3c4-b5a6-9876-5432-1098765432ab",
            login_attempt_time="2024-01-15T18:20:15Z",
            source_ip="192.168.1.100",
            user_agent_data='{"browser": "Chrome", "version": "120.0", "os": "macOS"}',
            failed_attempts="5",
        )

    print("✅ All scenarios processed with intelligent type detection!")
    print("   - UUIDs automatically validated and formatted")
    print("   - Timestamps parsed and enhanced with metadata")
    print("   - URLs categorized and validated")
    print("   - JSON strings parsed for better searchability")
    print("   - Numeric strings converted with type preservation")
    print()


def main():
    """Run all type detection examples"""
    print("🔍 Structured Logging v0.6.0 - Automatic Type Detection Examples")
    print("=" * 75)
    print()

    print("📋 Available Examples:")
    print("1. Basic string pattern detection and enhancement")
    print("2. Numeric value analysis and formatting")
    print("3. Selective detection configuration")
    print("4. Direct TypeDetector API usage")
    print("5. SmartConverter intelligence demonstration")
    print("6. Performance impact analysis")
    print("7. Real-world usage scenarios")
    print()

    try:
        example_basic_string_detection()
        example_numeric_analysis()
        example_selective_detection()
        example_type_detector_direct_usage()
        example_smart_converter_intelligence()
        example_performance_comparison()
        example_real_world_scenarios()

        print("🎉 All type detection examples completed!")
        print()
        print("💡 Key Benefits of Automatic Type Detection:")
        print("  🔍 Intelligent string analysis and categorization")
        print("  🏷️  Automatic type labeling with confidence scores")
        print("  📊 Enhanced searchability with structured metadata")
        print("  ⚙️  Configurable detection rules for different use cases")
        print("  🚀 Optimized performance with intelligent caching")
        print("  🛡️  Safe fallbacks for edge cases and errors")
        print("  🎯 Real-time data enrichment without code changes")
        print()
        print("🚀 Your logs are now smarter and more searchable!")

    except KeyboardInterrupt:
        print("\n⏹️  Examples interrupted")
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
