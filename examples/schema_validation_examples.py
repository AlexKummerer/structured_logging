#!/usr/bin/env python3
"""
Schema Validation Examples

Demonstrates the new schema validation and type annotation features in Version 0.6.0:
- Runtime schema validation for structured logging data
- Type annotation extraction for automatic schema generation
- Custom validation constraints and error handling
- Integration with existing serialization and logging systems
- Performance monitoring and statistics
"""

import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Optional, Union
from uuid import uuid4

from structured_logging import (
    get_logger,
    LoggerConfig,
    SerializationConfig,
    ValidationError,
    SchemaValidator,
    TypeAnnotationExtractor,
    StructuredDataValidator,
    register_validation_schema,
    validate_log_data,
    auto_validate_function,
    get_validation_stats,
    reset_validation_stats,
    log_with_context,
    request_context
)


def example_basic_schema_validation():
    """Example: Basic schema validation concepts"""
    print("📋 Basic Schema Validation Example")
    print("=" * 50)
    
    # Create a validator
    validator = SchemaValidator()
    
    # Define a simple user schema
    user_schema = {
        "user_id": "str",
        "username": {
            "type": "str",
            "min_length": 3,
            "max_length": 20,
            "pattern": r"^[a-zA-Z0-9_]+$"
        },
        "email": {
            "type": "str",
            "pattern": r"^[^@]+@[^@]+\.[^@]+$"
        },
        "age": {
            "type": "int",
            "min_value": 13,
            "max_value": 120
        },
        "active": "bool",
        "tags": {
            "type": "list",
            "min_items": 0,
            "max_items": 10
        }
    }
    
    # Register the schema
    validator.register_schema("user_registration", user_schema)
    print("✅ Registered user registration schema")
    
    # Test valid data
    valid_user = {
        "user_id": "usr_123456",
        "username": "john_doe",
        "email": "john.doe@example.com",
        "age": 25,
        "active": True,
        "tags": ["developer", "python", "logging"]
    }
    
    try:
        validator.validate(valid_user, "user_registration")
        print("✅ Valid user data passed validation")
    except ValidationError as e:
        print(f"❌ Validation failed: {e}")
    
    # Test invalid data
    invalid_users = [
        {
            **valid_user,
            "username": "jo"  # Too short
        },
        {
            **valid_user,
            "email": "invalid-email"  # Invalid format
        },
        {
            **valid_user,
            "age": 200  # Too old
        },
        {
            **valid_user,
            "tags": ["tag"] * 15  # Too many tags
        }
    ]
    
    print("\n🔍 Testing invalid data:")
    for i, invalid_user in enumerate(invalid_users, 1):
        try:
            validator.validate(invalid_user, "user_registration")
            print(f"   Case {i}: ❌ Should have failed validation")
        except ValidationError as e:
            print(f"   Case {i}: ✅ Correctly caught error: {str(e)[:60]}...")
    
    print()


def example_type_annotation_extraction():
    """Example: Automatic schema generation from type annotations"""
    print("🔍 Type Annotation Extraction Example")
    print("=" * 50)
    
    # Define a function with comprehensive type annotations
    def process_payment(
        transaction_id: str,
        amount: Decimal,
        currency: str,
        user_id: str,
        metadata: Dict[str, str],
        items: List[Dict[str, Union[str, int]]],
        timestamp: Optional[datetime] = None,
        async_processing: bool = False
    ) -> Dict[str, Union[str, bool]]:
        """Process a payment transaction with comprehensive logging"""
        return {
            "status": "success",
            "processed": True,
            "transaction_ref": f"ref_{transaction_id}"
        }
    
    # Extract schema from function
    extractor = TypeAnnotationExtractor()
    schema_info = extractor.extract_function_schema(process_payment)
    
    print("📊 Extracted Function Schema:")
    print(f"   Function: {schema_info['function_name']}")
    print(f"   Parameters: {len(schema_info['parameters'])}")
    
    for param_name, param_info in schema_info['parameters'].items():
        required = "required" if param_info.get("required", True) else "optional"
        param_type = param_info.get("type", "unknown")
        print(f"   - {param_name}: {param_type} ({required})")
    
    print(f"   Return Type: {schema_info['return_type']}")
    print()
    
    # Define a class with type annotations
    @dataclass
    class OrderItem:
        product_id: str
        name: str
        price: Decimal
        quantity: int
        discount: Optional[Decimal] = None
        metadata: Dict[str, str] = None
        
        def calculate_total(self) -> Decimal:
            base_total = self.price * self.quantity
            if self.discount:
                return base_total - self.discount
            return base_total
    
    # Extract schema from class
    class_schema = extractor.extract_class_schema(OrderItem)
    
    print("📊 Extracted Class Schema:")
    print(f"   Class: {class_schema['class_name']}")
    print(f"   Attributes: {len(class_schema['attributes'])}")
    
    for attr_name, attr_info in class_schema['attributes'].items():
        required = "required" if attr_info.get("required", True) else "optional"
        attr_type = attr_info.get("type", "unknown")
        print(f"   - {attr_name}: {attr_type} ({required})")
    
    print(f"   Methods: {len(class_schema['methods'])}")
    print()


def example_structured_data_validator():
    """Example: High-level structured data validation"""
    print("🏗️ Structured Data Validator Example")
    print("=" * 50)
    
    # Create a high-level validator
    validator = StructuredDataValidator()
    
    # Define a payment processing function
    def log_payment_event(
        event_type: str,
        transaction_id: str,
        amount: float,
        user_id: str,
        timestamp: datetime,
        details: dict
    ) -> None:
        """Log a payment processing event"""
        pass
    
    # Register function schema automatically
    schema_name = validator.register_function_schema(log_payment_event)
    print(f"✅ Auto-registered schema: {schema_name}")
    
    # Test validation against function signature
    valid_payment_data = {
        "event_type": "payment_processed",
        "transaction_id": "txn_abc123",
        "amount": 99.99,
        "user_id": "user_456",
        "timestamp": datetime.now(),
        "details": {
            "method": "credit_card",
            "card_last_four": "1234",
            "processor": "stripe"
        }
    }
    
    try:
        validator.validate_against_function(log_payment_event, valid_payment_data)
        print("✅ Payment data validated against function signature")
    except ValidationError as e:
        print(f"❌ Validation failed: {e}")
    
    # Test with invalid data
    invalid_payment_data = {
        **valid_payment_data,
        "amount": "99.99"  # Should be float, not string
    }
    
    try:
        validator.validate_against_function(log_payment_event, invalid_payment_data)
        print("❌ Should have failed validation")
    except ValidationError as e:
        print(f"✅ Correctly caught type error: {str(e)[:60]}...")
    
    # Show schema information
    schemas = validator.list_schemas()
    print(f"\n📋 Registered schemas: {schemas}")
    
    schema_info = validator.get_schema_info(schema_name)
    print(f"📊 Schema info: {schema_info['function_name']}")
    print()


def example_validation_decorator():
    """Example: Function validation decorator"""
    print("🎭 Validation Decorator Example")
    print("=" * 50)
    
    # Define functions with automatic validation
    @auto_validate_function
    def create_user_account(
        username: str,
        email: str,
        age: int,
        preferences: dict
    ) -> dict:
        """Create a new user account with validation"""
        return {
            "user_id": str(uuid4()),
            "username": username,
            "email": email,
            "age": age,
            "preferences": preferences,
            "created_at": datetime.now()
        }
    
    @auto_validate_function
    def update_user_profile(
        user_id: str,
        updates: Dict[str, Union[str, int, bool]]
    ) -> bool:
        """Update user profile with validation"""
        return True
    
    print("✅ Defined functions with automatic validation decorators")
    
    # Test valid calls
    try:
        user = create_user_account(
            username="alice_wonder",
            email="alice@example.com",
            age=28,
            preferences={"theme": "dark", "notifications": True}
        )
        print(f"✅ User created: {user['user_id']}")
        
        updated = update_user_profile(
            user_id=user["user_id"],
            updates={"age": 29, "theme": "light"}
        )
        print(f"✅ Profile updated: {updated}")
        
    except ValidationError as e:
        print(f"❌ Validation failed: {e}")
    
    # Test invalid calls
    invalid_test_cases = [
        # Invalid age type
        lambda: create_user_account("bob", "bob@example.com", "thirty", {}),
        # Invalid email type  
        lambda: create_user_account("charlie", 123, 25, {}),
        # Invalid user_id type
        lambda: update_user_profile(12345, {"name": "new_name"})
    ]
    
    print("\n🔍 Testing invalid function calls:")
    for i, test_case in enumerate(invalid_test_cases, 1):
        try:
            test_case()
            print(f"   Case {i}: ❌ Should have failed validation")
        except ValidationError as e:
            print(f"   Case {i}: ✅ Correctly caught error: {str(e)[:50]}...")
        except Exception as e:
            print(f"   Case {i}: ✅ Caught other error: {str(e)[:50]}...")
    
    print()


def example_custom_validators():
    """Example: Custom validation functions"""
    print("⚙️ Custom Validators Example")
    print("=" * 50)
    
    # Define custom validator functions
    def validate_credit_card(card_number: str) -> bool:
        """Simple credit card validation (Luhn algorithm simulation)"""
        # Remove spaces and dashes
        card_number = card_number.replace(" ", "").replace("-", "")
        
        # Check if all digits and reasonable length
        if not card_number.isdigit() or len(card_number) < 13 or len(card_number) > 19:
            return False
        
        # Simple checksum simulation (not real Luhn)
        return sum(int(d) for d in card_number) % 10 == 0
    
    def validate_phone_number(phone: str) -> bool:
        """Validate phone number format"""
        import re
        # Allow formats: +1-555-123-4567, (555) 123-4567, 555.123.4567, etc.
        pattern = r'^(\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$'
        return re.match(pattern, phone) is not None
    
    def validate_password_strength(password: str) -> bool:
        """Validate password meets security requirements"""
        if len(password) < 8:
            return False
        
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password) 
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        return has_upper and has_lower and has_digit and has_special
    
    # Create validator with custom constraints
    validator = SchemaValidator()
    
    payment_schema = {
        "card_number": {
            "type": "str",
            "validator": validate_credit_card
        },
        "phone": {
            "type": "str",
            "validator": validate_phone_number
        },
        "password": {
            "type": "str",
            "validator": validate_password_strength
        },
        "amount": {
            "type": "float",
            "min_value": 0.01,
            "max_value": 10000.00
        }
    }
    
    validator.register_schema("payment_info", payment_schema)
    print("✅ Registered schema with custom validators")
    
    # Test valid data
    valid_payment = {
        "card_number": "4532015112830366",  # Ends in 0, sum divisible by 10
        "phone": "+1-555-123-4567",
        "password": "SecurePass123!",
        "amount": 150.00
    }
    
    try:
        validator.validate(valid_payment, "payment_info")
        print("✅ Valid payment info passed all custom validations")
    except ValidationError as e:
        print(f"❌ Custom validation failed: {e}")
    
    # Test invalid data
    invalid_cases = [
        {**valid_payment, "card_number": "1234567890123456"},  # Bad checksum
        {**valid_payment, "phone": "invalid-phone"},           # Bad format
        {**valid_payment, "password": "weak"},                 # Too weak
        {**valid_payment, "amount": -50.00}                    # Negative amount
    ]
    
    print("\n🔍 Testing custom validator failures:")
    for i, invalid_case in enumerate(invalid_cases, 1):
        try:
            validator.validate(invalid_case, "payment_info")
            print(f"   Case {i}: ❌ Should have failed custom validation")
        except ValidationError as e:
            print(f"   Case {i}: ✅ Custom validator caught error")
    
    print()


def example_validation_with_logging():
    """Example: Integration with structured logging"""
    print("📝 Validation with Logging Integration Example")
    print("=" * 50)
    
    # Define a schema for API request logs
    api_request_schema = {
        "request_id": "str",
        "method": {
            "type": "str",
            "choices": ["GET", "POST", "PUT", "DELETE", "PATCH"]
        },
        "endpoint": "str",
        "user_id": {"type": "str", "required": False},
        "response_status": {
            "type": "int",
            "min_value": 100,
            "max_value": 599
        },
        "response_time_ms": {
            "type": "float",
            "min_value": 0
        },
        "user_agent": {"type": "str", "required": False}
    }
    
    # Register globally
    register_validation_schema("api_request_log", api_request_schema)
    print("✅ Registered global API request log schema")
    
    # Create logger
    logger = get_logger("api_validator")
    
    # Simulate API request logging with validation
    def log_api_request(request_data: dict) -> bool:
        """Log API request with validation"""
        try:
            # Validate the log data before logging
            validate_log_data(request_data, "api_request_log")
            
            # If validation passes, log it
            with request_context(request_id=request_data["request_id"]):
                log_with_context(
                    logger, "info", "API request processed",
                    **request_data
                )
            return True
            
        except ValidationError as e:
            # Log validation failure
            logger.error(f"Invalid API request data: {e}")
            return False
    
    # Test with valid requests
    valid_requests = [
        {
            "request_id": "req_001",
            "method": "GET",
            "endpoint": "/api/users",
            "user_id": "user_123",
            "response_status": 200,
            "response_time_ms": 45.2,
            "user_agent": "Mozilla/5.0 (compatible)"
        },
        {
            "request_id": "req_002", 
            "method": "POST",
            "endpoint": "/api/orders",
            "response_status": 201,
            "response_time_ms": 123.8
        }
    ]
    
    print("\n📊 Logging valid API requests:")
    for request in valid_requests:
        success = log_api_request(request)
        status = "✅ logged" if success else "❌ failed"
        print(f"   Request {request['request_id']}: {status}")
    
    # Test with invalid requests
    invalid_requests = [
        {
            "request_id": "req_003",
            "method": "INVALID",  # Not in choices
            "endpoint": "/api/test",
            "response_status": 200,
            "response_time_ms": 25.0
        },
        {
            "request_id": "req_004",
            "method": "GET",
            "endpoint": "/api/test",
            "response_status": 999,  # Out of range
            "response_time_ms": 15.0
        }
    ]
    
    print("\n🔍 Testing invalid API requests:")
    for request in invalid_requests:
        success = log_api_request(request)
        status = "✅ logged" if success else "❌ rejected"
        print(f"   Request {request['request_id']}: {status}")
    
    print()


def example_validation_performance():
    """Example: Validation performance and statistics"""
    print("⚡ Validation Performance Example")
    print("=" * 50)
    
    # Reset statistics
    reset_validation_stats()
    
    # Create a moderately complex schema
    user_activity_schema = {
        "user_id": "str",
        "session_id": "str", 
        "activity_type": {
            "type": "str",
            "choices": ["login", "logout", "page_view", "click", "purchase", "search"]
        },
        "timestamp": "datetime",
        "page_url": {"type": "str", "required": False},
        "metadata": {"type": "dict", "required": False},
        "user_agent": {"type": "str", "required": False},
        "ip_address": {
            "type": "str",
            "pattern": r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$",
            "required": False
        }
    }
    
    register_validation_schema("user_activity", user_activity_schema)
    
    # Generate test data
    test_activities = []
    for i in range(1000):
        activity = {
            "user_id": f"user_{i % 100}",
            "session_id": f"session_{i}",
            "activity_type": ["login", "page_view", "click", "purchase"][i % 4],
            "timestamp": datetime.now(),
            "page_url": f"/page/{i % 20}",
            "metadata": {"sequence": i, "experiment": "A" if i % 2 == 0 else "B"},
            "user_agent": "Mozilla/5.0 (Test Browser)",
            "ip_address": f"192.168.{i % 256}.{(i * 7) % 256}"
        }
        test_activities.append(activity)
    
    print(f"📊 Generated {len(test_activities)} test activity records")
    
    # Performance test
    start_time = time.perf_counter()
    valid_count = 0
    invalid_count = 0
    
    for activity in test_activities:
        try:
            validate_log_data(activity, "user_activity")
            valid_count += 1
        except ValidationError:
            invalid_count += 1
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Get validation statistics
    stats = get_validation_stats()
    
    print(f"\n📈 Performance Results:")
    print(f"   Total validation time: {total_time:.3f}s")
    print(f"   Average per validation: {(total_time / len(test_activities)) * 1000:.2f}ms")
    print(f"   Validations per second: {len(test_activities) / total_time:.0f}")
    print(f"   Valid records: {valid_count}")
    print(f"   Invalid records: {invalid_count}")
    
    print(f"\n📊 Validation Statistics:")
    print(f"   Total validations performed: {stats['validations_performed']:,}")
    print(f"   Validation failures: {stats['validation_failures']:,}")
    print(f"   Success rate: {((stats['validations_performed'] - stats['validation_failures']) / stats['validations_performed'] * 100):.1f}%")
    print(f"   Total validation time: {stats['validation_time']:.3f}s")
    
    if total_time < 1.0:
        print("   🎉 Excellent performance! Under 1 second for 1000 validations")
    elif total_time < 2.0:
        print("   ✅ Good performance! Under 2 seconds for 1000 validations")
    else:
        print("   ℹ️  Performance acceptable, consider optimization for high-volume use")
    
    print()


def example_validation_error_handling():
    """Example: Comprehensive error handling"""
    print("🛠️ Validation Error Handling Example")
    print("=" * 50)
    
    # Create a schema with multiple potential failure points
    complex_schema = {
        "id": {
            "type": "str",
            "pattern": r"^[A-Z]{3}_\d{6}$"  # Format: ABC_123456
        },
        "email": {
            "type": "str",
            "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        },
        "age": {
            "type": "int",
            "min_value": 0,
            "max_value": 150
        },
        "tags": {
            "type": "list",
            "min_items": 1,
            "max_items": 5
        },
        "score": {
            "type": "float",
            "min_value": 0.0,
            "max_value": 100.0
        }
    }
    
    validator = SchemaValidator()
    validator.register_schema("complex_validation", complex_schema)
    
    # Test cases with different types of errors
    error_test_cases = [
        # Missing required field
        {
            "name": "missing_field",
            "data": {
                "email": "test@example.com",
                "age": 25,
                "tags": ["test"],
                "score": 85.5
                # Missing 'id' field
            }
        },
        # Wrong type
        {
            "name": "wrong_type",
            "data": {
                "id": "ABC_123456",
                "email": "test@example.com", 
                "age": "twenty-five",  # Should be int
                "tags": ["test"],
                "score": 85.5
            }
        },
        # Pattern mismatch
        {
            "name": "pattern_mismatch",
            "data": {
                "id": "invalid_id_format",  # Doesn't match pattern
                "email": "invalid-email",   # Doesn't match email pattern
                "age": 25,
                "tags": ["test"],
                "score": 85.5
            }
        },
        # Range violations
        {
            "name": "range_violation",
            "data": {
                "id": "ABC_123456",
                "email": "test@example.com",
                "age": 200,  # Too old
                "tags": [],  # Too few items
                "score": 150.0  # Too high
            }
        },
        # Multiple errors
        {
            "name": "multiple_errors",
            "data": {
                "id": "bad_format",
                "email": "bad-email",
                "age": -5,
                "tags": ["a", "b", "c", "d", "e", "f"],  # Too many
                "score": "not_a_number"
            }
        }
    ]
    
    print("🔍 Testing various validation error scenarios:")
    
    for test_case in error_test_cases:
        print(f"\n   Test: {test_case['name']}")
        try:
            validator.validate(test_case['data'], "complex_validation")
            print(f"      ❌ Should have failed validation")
        except ValidationError as e:
            error_msg = str(e)
            print(f"      ✅ Caught validation error:")
            # Show first part of error message
            if len(error_msg) > 100:
                print(f"         {error_msg[:100]}...")
            else:
                print(f"         {error_msg}")
    
    # Test graceful handling of schema not found
    print(f"\n🔍 Testing schema not found error:")
    try:
        validator.validate({"test": "data"}, "nonexistent_schema")
        print("      ❌ Should have failed with schema not found")
    except ValidationError as e:
        print(f"      ✅ Correctly handled missing schema: {str(e)[:60]}...")
    
    print()


def main():
    """Run all schema validation examples"""
    print("🔬 Structured Logging v0.6.0 - Schema Validation Examples")
    print("=" * 70)
    print()
    
    print("📋 Available Examples:")
    print("1. Basic schema validation concepts")
    print("2. Type annotation extraction for auto-schemas")
    print("3. Structured data validator usage")
    print("4. Function validation decorators")
    print("5. Custom validation functions")
    print("6. Integration with structured logging")
    print("7. Performance testing and statistics")
    print("8. Comprehensive error handling")
    print()
    
    try:
        example_basic_schema_validation()
        example_type_annotation_extraction()
        example_structured_data_validator()
        example_validation_decorator()
        example_custom_validators()
        example_validation_with_logging()
        example_validation_performance()
        example_validation_error_handling()
        
        print("🎉 All schema validation examples completed!")
        print()
        print("💡 Key Benefits of Schema Validation:")
        print("  ✅ Runtime validation of log data structure and types")
        print("  🔍 Automatic schema generation from type annotations")
        print("  ⚙️ Custom validation functions for business rules")
        print("  🎭 Function decorators for automatic argument validation")
        print("  📊 Performance monitoring and validation statistics")
        print("  🛠️ Comprehensive error handling and reporting")
        print("  📝 Seamless integration with structured logging")
        print("  🚀 High-performance validation suitable for production")
        print()
        print("🔬 Your log data is now validated, structured, and reliable!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Examples interrupted")
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()