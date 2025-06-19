#!/usr/bin/env python3
"""
Enhanced Serialization Examples

Demonstrates the new custom serialization features in Version 0.6.0:
- Automatic handling of complex Python types
- Custom serialization configuration
- Type-safe logging with dataclasses
- Performance optimizations
"""

import time
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import List
from uuid import uuid4, UUID

from structured_logging import (
    get_logger,
    LoggerConfig,
    SerializationConfig,
    register_custom_serializer,
    serialize_for_logging,
    request_context
)


# Define some example types
class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class Status(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class User:
    id: int
    name: str
    email: str
    created_at: datetime
    last_login: datetime = None


@dataclass 
class Order:
    id: UUID
    user: User
    amount: Decimal
    items: List[str]
    priority: Priority
    status: Status
    created_at: datetime
    metadata: dict = None


def example_basic_types():
    """Example: Basic complex types that now work automatically"""
    print("🔄 Basic Complex Types Example")
    print("=" * 50)
    
    logger = get_logger("basic_types_demo")
    
    with request_context(service="ecommerce"):
        # All of these types are now automatically handled!
        logger.info("Order created",
                   order_id=uuid4(),                    # UUID -> string
                   created_at=datetime.now(),           # datetime -> ISO string  
                   amount=Decimal("99.99"),             # Decimal -> string (preserves precision)
                   priority=Priority.HIGH,              # Enum -> value
                   file_path=Path("/tmp/order.json"),   # Path -> string
                   tags={"urgent", "premium", "vip"},   # set -> list
                   config_data=bytes("config", "utf-8") # bytes -> decoded text
                   )
        
        logger.warning("Large order detected",
                      customer_data={
                          "signup_date": date(2024, 1, 15),     # date -> ISO string
                          "session_duration": timedelta(hours=2, minutes=30),  # timedelta -> dict
                          "preferences": frozenset(["email", "sms"]),  # frozenset -> list
                          "balance": complex(100, 0)             # complex -> real/imag dict
                      })
    
    print("✅ All complex types logged automatically!")
    print()


def example_dataclass_logging():
    """Example: Dataclass logging with automatic serialization"""
    print("🏗️ Dataclass Logging Example")
    print("=" * 50)
    
    logger = get_logger("dataclass_demo")
    
    # Create complex dataclass instances
    user = User(
        id=12345,
        name="John Doe",
        email="john.doe@example.com",
        created_at=datetime(2024, 1, 15, 10, 30, 0),
        last_login=datetime.now()
    )
    
    order = Order(
        id=uuid4(),
        user=user,
        amount=Decimal("149.99"),
        items=["Laptop", "Mouse", "Keyboard"],
        priority=Priority.HIGH,
        status=Status.PROCESSING,
        created_at=datetime.now(),
        metadata={
            "source": "web",
            "campaign": "summer_sale",
            "discount_applied": True
        }
    )
    
    with request_context(user_id=str(user.id), tenant_id="company-123"):
        # Dataclasses are automatically converted to dictionaries!
        logger.info("Order processing started", order=order)
        
        logger.info("User activity", 
                   user=user,
                   action="purchase",
                   timestamp=datetime.now())
        
        # Nested dataclasses work too
        logger.warning("High-value order", 
                      order_summary={
                          "order": order,
                          "payment_method": "credit_card",
                          "shipping_address": {
                              "country": "US",
                              "expedited": True
                          }
                      })
    
    print("✅ Dataclasses logged with full structure!")
    print()


def example_custom_serialization_config():
    """Example: Custom serialization configuration"""
    print("⚙️ Custom Serialization Config Example") 
    print("=" * 50)
    
    # Create custom serialization config
    custom_config = SerializationConfig(
        datetime_format="timestamp",        # Use timestamps instead of ISO
        decimal_as_float=True,             # Convert Decimal to float
        enum_as_value=False,               # Show enum name and value
        include_type_hints=True,           # Include type information
        max_collection_size=5,             # Limit collection sizes
        truncate_strings=50,               # Truncate long strings
        path_as_string=False               # Show detailed path info
    )
    
    # Create logger with custom config  
    logger_config = LoggerConfig(formatter_type="json")
    logger = get_logger("custom_config_demo", logger_config)
    
    # Override the formatter to use our custom serialization
    from structured_logging.formatter import StructuredFormatter
    for handler in logger.handlers:
        if isinstance(handler.formatter, StructuredFormatter):
            handler.formatter.serialization_config = custom_config
    
    # Log with different serialization behavior
    test_data = {
        "timestamp": datetime.now(),
        "amount": Decimal("199.99"),
        "priority": Priority.CRITICAL,
        "file_path": Path("/very/long/path/to/important/file.log"),
        "large_list": list(range(20)),  # Will be truncated
        "long_description": "This is a very long description that should be truncated because it exceeds the configured limit for string length in our custom serialization configuration."
    }
    
    logger.info("Custom serialization demo", data=test_data)
    
    print("✅ Custom serialization applied!")
    print("   - Timestamps as numbers")
    print("   - Decimals as floats") 
    print("   - Enums with full info")
    print("   - Collections truncated")
    print("   - Strings truncated")
    print()


def example_custom_type_registration():
    """Example: Registering custom serializers for your own types"""
    print("🎯 Custom Type Registration Example")
    print("=" * 50)
    
    # Define a custom class
    class Money:
        def __init__(self, amount: Decimal, currency: str):
            self.amount = amount
            self.currency = currency
        
        def __repr__(self):
            return f"Money({self.amount}, {self.currency})"
    
    class BankAccount:
        def __init__(self, account_number: str, balance: Money, owner: str):
            self.account_number = account_number
            self.balance = balance
            self.owner = owner
    
    # Register custom serializers
    def serialize_money(money_obj, config):
        """Custom serializer for Money objects"""
        return {
            "amount": str(money_obj.amount),  # Keep precision
            "currency": money_obj.currency,
            "formatted": f"{money_obj.currency} {money_obj.amount}"
        }
    
    def serialize_bank_account(account_obj, config):
        """Custom serializer for BankAccount objects"""
        return {
            "account_number": account_obj.account_number[-4:],  # Only last 4 digits
            "balance": serialize_for_logging(account_obj.balance, config),  # Use Money serializer
            "owner": account_obj.owner,
            "account_type": "checking"  # Add additional info
        }
    
    # Register the serializers globally
    register_custom_serializer(Money, serialize_money)
    register_custom_serializer(BankAccount, serialize_bank_account)
    
    logger = get_logger("custom_types_demo")
    
    # Now these custom types work automatically!
    account = BankAccount(
        account_number="1234567890",
        balance=Money(Decimal("1500.50"), "USD"),
        owner="Jane Smith"
    )
    
    with request_context(service="banking", operation="balance_check"):
        logger.info("Account accessed", account=account)
        
        logger.warning("Large withdrawal", 
                      account=account,
                      withdrawal_amount=Money(Decimal("500.00"), "USD"),
                      remaining_balance=Money(Decimal("1000.50"), "USD"))
    
    print("✅ Custom types serialized with business logic!")
    print("   - Account numbers masked for security")
    print("   - Money objects with currency formatting")
    print("   - Additional computed fields")
    print()


def example_scientific_data():
    """Example: Scientific data logging (if numpy/pandas available)"""
    print("🔬 Scientific Data Example")
    print("=" * 50)
    
    try:
        import numpy as np
        import pandas as pd
        
        logger = get_logger("scientific_demo")
        
        # Create sample scientific data
        measurements = np.array([1.2, 2.3, 3.4, 4.5, 5.6])
        large_dataset = np.random.normal(0, 1, (100, 50))  # 5000 elements
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'temperature': np.random.normal(20, 5, 100),
            'pressure': np.random.normal(1013, 50, 100),
            'humidity': np.random.uniform(30, 90, 100)
        })
        
        with request_context(experiment="climate_monitoring"):
            # Small arrays are logged completely
            logger.info("Calibration measurements", 
                       measurements=measurements,
                       sensor_id="TEMP_001")
            
            # Large arrays get summary statistics
            logger.info("Dataset analysis complete",
                       dataset_shape=large_dataset.shape,
                       raw_data=large_dataset,  # Auto-summarized
                       summary_stats={
                           "mean": np.mean(large_dataset),
                           "std": np.std(large_dataset)
                       })
            
            # DataFrames get intelligent serialization
            logger.info("Time series analysis",
                       dataframe=df,  # Shows shape, columns, sample data
                       analysis_window="24h")
        
        print("✅ Scientific data logged intelligently!")
        print("   - Small arrays: full data")
        print("   - Large arrays: summary + sample")
        print("   - DataFrames: structure + preview")
        
    except ImportError:
        print("📊 NumPy/Pandas not available - skipping scientific example")
    
    print()


def example_performance_comparison():
    """Example: Performance comparison with/without enhanced serialization"""
    print("⚡ Performance Comparison Example")
    print("=" * 50)
    
    # Create test data with complex types
    test_data = {
        "uuid": uuid4(),
        "timestamp": datetime.now(),
        "amount": Decimal("99.99"),
        "user": User(
            id=123,
            name="Performance Test",
            email="perf@test.com",
            created_at=datetime.now()
        ),
        "priorities": [Priority.LOW, Priority.MEDIUM, Priority.HIGH] * 100,
        "metadata": {
            "path": Path("/tmp/performance_test.log"),
            "data": set(range(50)),
            "config": {"setting1": "value1", "setting2": "value2"}
        }
    }
    
    logger = get_logger("performance_demo")
    
    # Measure serialization performance
    start_time = time.perf_counter()
    
    iterations = 1000
    for i in range(iterations):
        logger.info(f"Performance test {i}", data=test_data)
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    logs_per_second = iterations / duration
    
    print(f"✅ Performance Results:")
    print(f"   - {iterations:,} complex log entries")
    print(f"   - {duration:.3f} seconds total")
    print(f"   - {logs_per_second:,.0f} logs/second")
    print(f"   - {duration/iterations*1000:.2f} ms per log")
    
    if logs_per_second > 1000:
        print("   🚀 Excellent performance!")
    elif logs_per_second > 500:
        print("   ✅ Good performance!")
    else:
        print("   ⚠️  Consider optimization")
    
    print()


def example_error_handling():
    """Example: Error handling for problematic objects"""
    print("🛡️ Error Handling Example")
    print("=" * 50)
    
    logger = get_logger("error_handling_demo")
    
    # Create some problematic objects
    class ProblematicClass:
        def __init__(self):
            pass
        
        def __repr__(self):
            raise Exception("Cannot represent this object!")
    
    class RecursiveClass:
        def __init__(self):
            self.self_ref = self
    
    problematic_data = {
        "normal_data": "This is fine",
        "problematic": ProblematicClass(),
        "recursive": RecursiveClass(),
        "very_large_string": "x" * 10000,  # Very long string
        "large_collection": list(range(5000))  # Large collection
    }
    
    with request_context(test_type="error_handling"):
        # This should not crash the application
        logger.error("Testing error handling", data=problematic_data)
        
        logger.warning("Problematic object detected", 
                      obj=ProblematicClass(),
                      fallback_info="Object serialization failed gracefully")
    
    print("✅ Error handling worked!")
    print("   - Problematic objects handled gracefully")
    print("   - Fallback representations provided")
    print("   - Application continued without crashing")
    print()


def main():
    """Run all serialization examples"""
    print("🔄 Structured Logging v0.6.0 - Enhanced Serialization Examples")
    print("=" * 70)
    print()
    
    print("📋 Available Examples:")
    print("1. Basic complex types (datetime, UUID, Decimal, etc.)")
    print("2. Dataclass automatic serialization") 
    print("3. Custom serialization configuration")
    print("4. Custom type registration")
    print("5. Scientific data handling")
    print("6. Performance benchmarking")
    print("7. Error handling and fallbacks")
    print()
    
    try:
        example_basic_types()
        example_dataclass_logging()
        example_custom_serialization_config()
        example_custom_type_registration()
        example_scientific_data()
        example_performance_comparison()
        example_error_handling()
        
        print("🎉 All serialization examples completed!")
        print()
        print("💡 Key Benefits:")
        print("  ✅ No more manual type conversion")
        print("  ✅ Automatic dataclass support")
        print("  ✅ Configurable serialization behavior")
        print("  ✅ Extensible with custom types")
        print("  ✅ High performance with large data")
        print("  ✅ Graceful error handling")
        print("  ✅ Scientific data intelligence")
        print()
        print("🚀 Ready for production use with complex data types!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Examples interrupted")
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()