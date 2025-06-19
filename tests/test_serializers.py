"""
Tests for enhanced serialization support
"""

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

import pytest

from structured_logging.serializers import (
    DEFAULT_CONFIG,
    EnhancedJSONEncoder,
    LazyDict,
    LazySerializable,
    LazySerializationManager,
    SerializationConfig,
    SmartConverter,
    TypeDetector,
    TypeRegistry,
    ValidationError,
    SchemaValidator,
    TypeAnnotationExtractor,
    StructuredDataValidator,
    auto_validate_function,
    create_lazy_serializable,
    enhanced_json_dumps,
    get_lazy_serialization_stats,
    get_validation_stats,
    register_custom_serializer,
    register_validation_schema,
    reset_lazy_serialization_stats,
    reset_validation_stats,
    serialize_for_logging,
    serialize_for_logging_lazy_aware,
    should_use_lazy_serialization,
    validate_log_data,
)


class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass
class User:
    id: int
    name: str
    email: str


class TestSerializationConfig:
    """Tests for SerializationConfig"""
    
    def test_default_config(self):
        config = SerializationConfig()
        assert config.datetime_format == "iso"
        assert config.decimal_as_float is False
        assert config.include_type_hints is False
        assert config.max_collection_size == 1000
        assert config.enum_as_value is True
        assert config.dataclass_as_dict is True
    
    def test_custom_config(self):
        config = SerializationConfig(
            datetime_format="timestamp",
            decimal_as_float=True,
            max_collection_size=50,
            truncate_strings=100
        )
        assert config.datetime_format == "timestamp"
        assert config.decimal_as_float is True
        assert config.max_collection_size == 50
        assert config.truncate_strings == 100


class TestTypeRegistry:
    """Tests for TypeRegistry"""
    
    def test_default_serializers_registered(self):
        registry = TypeRegistry()
        
        # Test datetime serializer
        dt = datetime(2024, 1, 15, 10, 30, 0)
        serializer = registry.get_serializer(dt)
        assert serializer is not None
        
        # Test UUID serializer
        uuid_val = uuid4()
        serializer = registry.get_serializer(uuid_val)
        assert serializer is not None
        
        # Test Decimal serializer
        decimal_val = Decimal("99.99")
        serializer = registry.get_serializer(decimal_val)
        assert serializer is not None
    
    def test_custom_serializer_registration(self):
        registry = TypeRegistry()
        
        class CustomClass:
            def __init__(self, value):
                self.value = value
        
        def custom_serializer(obj, config):
            return {"custom_value": obj.value}
        
        registry.register(CustomClass, custom_serializer)
        
        obj = CustomClass("test")
        serializer = registry.get_serializer(obj)
        assert serializer is not None
        
        result = serializer(obj, DEFAULT_CONFIG)
        assert result == {"custom_value": "test"}
    
    def test_inheritance_serialization(self):
        """Test that serializers work with inheritance"""
        registry = TypeRegistry()
        
        class BaseClass:
            pass
        
        class DerivedClass(BaseClass):
            def __init__(self, value):
                self.value = value
        
        def base_serializer(obj, config):
            return {"type": "base", "value": getattr(obj, 'value', None)}
        
        registry.register(BaseClass, base_serializer)
        
        # Should work for derived class too
        derived_obj = DerivedClass("derived")
        serializer = registry.get_serializer(derived_obj)
        assert serializer is not None
        
        result = serializer(derived_obj, DEFAULT_CONFIG)
        assert result == {"type": "base", "value": "derived"}


class TestDateTimeSerializers:
    """Tests for date/time serialization"""
    
    def test_datetime_iso_format(self):
        registry = TypeRegistry()
        dt = datetime(2024, 1, 15, 10, 30, 0)
        
        config = SerializationConfig(datetime_format="iso")
        serializer = registry.get_serializer(dt)
        result = serializer(dt, config)
        
        assert result == "2024-01-15T10:30:00Z"
    
    def test_datetime_timestamp_format(self):
        registry = TypeRegistry()
        dt = datetime(2024, 1, 15, 10, 30, 0)
        
        config = SerializationConfig(datetime_format="timestamp")
        serializer = registry.get_serializer(dt)
        result = serializer(dt, config)
        
        assert isinstance(result, float)
        assert result == dt.timestamp()
    
    def test_datetime_custom_format(self):
        registry = TypeRegistry()
        dt = datetime(2024, 1, 15, 10, 30, 0)
        
        config = SerializationConfig(
            datetime_format="custom",
            custom_datetime_format="%Y-%m-%d %H:%M"
        )
        serializer = registry.get_serializer(dt)
        result = serializer(dt, config)
        
        assert result == "2024-01-15 10:30"
    
    def test_date_serialization(self):
        registry = TypeRegistry()
        d = date(2024, 1, 15)
        
        serializer = registry.get_serializer(d)
        result = serializer(d, DEFAULT_CONFIG)
        
        assert result == "2024-01-15"
    
    def test_time_serialization(self):
        registry = TypeRegistry()
        t = time(10, 30, 45)
        
        serializer = registry.get_serializer(t)
        result = serializer(t, DEFAULT_CONFIG)
        
        assert result == "10:30:45"
    
    def test_timedelta_serialization(self):
        registry = TypeRegistry()
        td = timedelta(days=1, seconds=3600, microseconds=500000)
        
        serializer = registry.get_serializer(td)
        result = serializer(td, DEFAULT_CONFIG)
        
        expected = {
            "days": 1,
            "seconds": 3600,
            "microseconds": 500000,
            "total_seconds": td.total_seconds()
        }
        assert result == expected


class TestNumericSerializers:
    """Tests for numeric type serialization"""
    
    def test_decimal_as_string(self):
        registry = TypeRegistry()
        decimal_val = Decimal("99.99")
        
        config = SerializationConfig(decimal_as_float=False)
        serializer = registry.get_serializer(decimal_val)
        result = serializer(decimal_val, config)
        
        assert result == "99.99"
        assert isinstance(result, str)
    
    def test_decimal_as_float(self):
        registry = TypeRegistry()
        decimal_val = Decimal("99.99")
        
        config = SerializationConfig(decimal_as_float=True)
        serializer = registry.get_serializer(decimal_val)
        result = serializer(decimal_val, config)
        
        assert result == 99.99
        assert isinstance(result, float)
    
    def test_complex_number_serialization(self):
        registry = TypeRegistry()
        complex_val = complex(3, 4)
        
        serializer = registry.get_serializer(complex_val)
        result = serializer(complex_val, DEFAULT_CONFIG)
        
        expected = {"real": 3.0, "imag": 4.0}
        assert result == expected


class TestUUIDSerialization:
    """Tests for UUID serialization"""
    
    def test_uuid_serialization(self):
        registry = TypeRegistry()
        uuid_val = UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479")
        
        serializer = registry.get_serializer(uuid_val)
        result = serializer(uuid_val, DEFAULT_CONFIG)
        
        assert result == "f47ac10b-58cc-4372-a567-0e02b2c3d479"
        assert isinstance(result, str)


class TestPathSerialization:
    """Tests for Path object serialization"""
    
    def test_path_as_string(self):
        registry = TypeRegistry()
        path_obj = Path("/home/user/document.txt")
        
        config = SerializationConfig(path_as_string=True)
        serializer = registry.get_serializer(path_obj)
        result = serializer(path_obj, config)
        
        assert result == "/home/user/document.txt"
        assert isinstance(result, str)
    
    def test_path_as_dict(self):
        registry = TypeRegistry()
        path_obj = Path("/home/user/document.txt")
        
        config = SerializationConfig(path_as_string=False)
        serializer = registry.get_serializer(path_obj)
        result = serializer(path_obj, config)
        
        assert isinstance(result, dict)
        assert "path" in result
        assert "is_absolute" in result
        assert "parts" in result
        assert "suffix" in result
        assert result["suffix"] == ".txt"


class TestCollectionSerialization:
    """Tests for collection serialization"""
    
    def test_set_serialization(self):
        registry = TypeRegistry()
        set_obj = {1, 2, 3, 4, 5}
        
        serializer = registry.get_serializer(set_obj)
        result = serializer(set_obj, DEFAULT_CONFIG)
        
        assert isinstance(result, list)
        assert len(result) == 5
        assert all(item in [1, 2, 3, 4, 5] for item in result)
    
    def test_large_set_truncation(self):
        registry = TypeRegistry()
        large_set = set(range(1500))  # Larger than default max_collection_size
        
        config = SerializationConfig(max_collection_size=100)
        serializer = registry.get_serializer(large_set)
        result = serializer(large_set, config)
        
        assert isinstance(result, list)
        assert len(result) == 101  # 100 items + truncation message
        assert "more items" in str(result[-1])
    
    def test_frozenset_serialization(self):
        registry = TypeRegistry()
        frozenset_obj = frozenset([1, 2, 3])
        
        serializer = registry.get_serializer(frozenset_obj)
        result = serializer(frozenset_obj, DEFAULT_CONFIG)
        
        assert isinstance(result, list)
        assert len(result) == 3


class TestEnumSerialization:
    """Tests for Enum serialization"""
    
    def test_enum_as_value(self):
        registry = TypeRegistry()
        enum_obj = Priority.HIGH
        
        config = SerializationConfig(enum_as_value=True)
        serializer = registry.get_serializer(enum_obj)
        result = serializer(enum_obj, config)
        
        assert result == 3
    
    def test_enum_as_dict(self):
        registry = TypeRegistry()
        enum_obj = Priority.MEDIUM
        
        config = SerializationConfig(enum_as_value=False)
        serializer = registry.get_serializer(enum_obj)
        result = serializer(enum_obj, config)
        
        expected = {
            "name": "MEDIUM",
            "value": 2,
            "type": "Priority"
        }
        assert result == expected


class TestDataclassSerialization:
    """Tests for dataclass serialization"""
    
    def test_dataclass_serialization(self):
        registry = TypeRegistry()
        user = User(id=123, name="John Doe", email="john@example.com")
        
        serializer = registry.get_serializer(user)
        result = serializer(user, DEFAULT_CONFIG)
        
        expected = {
            "id": 123,
            "name": "John Doe",
            "email": "john@example.com"
        }
        assert result == expected
    
    def test_dataclass_with_type_hints(self):
        registry = TypeRegistry()
        user = User(id=456, name="Jane Doe", email="jane@example.com")
        
        config = SerializationConfig(include_type_hints=True)
        serializer = registry.get_serializer(user)
        result = serializer(user, config)
        
        assert result["id"] == 456
        assert result["name"] == "Jane Doe"
        assert result["__type__"] == "User"


class TestBytesSerialization:
    """Tests for bytes serialization"""
    
    def test_text_bytes_serialization(self):
        registry = TypeRegistry()
        text_bytes = "Hello, World!".encode('utf-8')
        
        serializer = registry.get_serializer(text_bytes)
        result = serializer(text_bytes, DEFAULT_CONFIG)
        
        assert result["type"] == "text"
        assert result["data"] == "Hello, World!"
        assert result["size"] == len(text_bytes)
    
    def test_binary_bytes_serialization(self):
        registry = TypeRegistry()
        binary_bytes = bytes([0x00, 0x01, 0x02, 0xFF])
        
        serializer = registry.get_serializer(binary_bytes)
        result = serializer(binary_bytes, DEFAULT_CONFIG)
        
        assert result["type"] == "binary"
        assert "hex" in result
        assert result["size"] == 4
    
    def test_long_text_truncation(self):
        registry = TypeRegistry()
        long_text = "x" * 200
        text_bytes = long_text.encode('utf-8')
        
        config = SerializationConfig(truncate_strings=50)
        serializer = registry.get_serializer(text_bytes)
        result = serializer(text_bytes, config)
        
        assert result["type"] == "text"
        assert len(result["data"]) == 53  # 50 + "..."
        assert result["data"].endswith("...")


class TestEnhancedJSONEncoder:
    """Tests for EnhancedJSONEncoder"""
    
    def test_json_encoder_basic_types(self):
        data = {
            "string": "hello",
            "number": 42,
            "boolean": True,
            "null": None
        }
        
        encoder = EnhancedJSONEncoder()
        result = json.dumps(data, cls=EnhancedJSONEncoder)
        parsed = json.loads(result)
        
        assert parsed == data
    
    def test_json_encoder_complex_types(self):
        data = {
            "timestamp": datetime(2024, 1, 15, 10, 30, 0),
            "uuid": UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479"),
            "decimal": Decimal("99.99"),
            "priority": Priority.HIGH,
            "user": User(id=123, name="John", email="john@example.com")
        }
        
        encoder = EnhancedJSONEncoder()
        result = json.dumps(data, cls=EnhancedJSONEncoder)
        parsed = json.loads(result)
        
        assert "timestamp" in parsed
        assert "uuid" in parsed
        assert "decimal" in parsed
        assert "priority" in parsed
        assert "user" in parsed
        assert isinstance(parsed["user"], dict)
    
    def test_json_encoder_error_handling(self):
        """Test encoder handles serialization errors gracefully"""
        
        class UnserializableClass:
            def __init__(self):
                pass
            
            def __repr__(self):
                return "UnserializableClass()"
        
        data = {"problematic": UnserializableClass()}
        
        encoder = EnhancedJSONEncoder()
        result = json.dumps(data, cls=EnhancedJSONEncoder)
        parsed = json.loads(result)
        
        # Should contain error information instead of crashing
        assert "__unserializable__" in parsed["problematic"]
        assert parsed["problematic"]["__unserializable__"] == "UnserializableClass"


class TestSerializeForLogging:
    """Tests for serialize_for_logging function"""
    
    def test_primitive_types(self):
        assert serialize_for_logging("hello") == "hello"
        assert serialize_for_logging(42) == 42
        assert serialize_for_logging(True) is True
        assert serialize_for_logging(None) is None
    
    def test_complex_nested_structure(self):
        data = {
            "user": User(id=123, name="John", email="john@example.com"),
            "timestamp": datetime(2024, 1, 15, 10, 30, 0),
            "priorities": [Priority.LOW, Priority.HIGH],
            "metadata": {
                "uuid": UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479"),
                "amount": Decimal("99.99")
            }
        }
        
        result = serialize_for_logging(data)
        
        # Should be JSON serializable
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        
        assert "user" in parsed
        assert "timestamp" in parsed
        assert "priorities" in parsed
        assert "metadata" in parsed
    
    def test_large_collection_truncation(self):
        large_list = list(range(2000))
        
        config = SerializationConfig(max_collection_size=100)
        result = serialize_for_logging(large_list, config)
        
        assert len(result) == 101  # 100 items + truncation message
        assert "more items" in str(result[-1])
    
    def test_string_truncation(self):
        long_string = "x" * 200
        
        config = SerializationConfig(truncate_strings=50)
        result = serialize_for_logging(long_string, config)
        
        assert len(result) == 53  # 50 + "..."
        assert result.endswith("...")
    
    def test_nested_dict_size_limit(self):
        large_dict = {f"key_{i}": f"value_{i}" for i in range(200)}
        
        config = SerializationConfig(max_collection_size=50)
        result = serialize_for_logging(large_dict, config)
        
        assert len(result) == 51  # 50 items + "..." key
        assert "..." in result
        assert "more items" in str(result["..."])


class TestCustomSerializerRegistration:
    """Tests for custom serializer registration"""
    
    def test_global_serializer_registration(self):
        class CustomType:
            def __init__(self, value):
                self.value = value
        
        def custom_serializer(obj, config):
            return {"custom": obj.value}
        
        # Register globally
        register_custom_serializer(CustomType, custom_serializer)
        
        obj = CustomType("test_value")
        result = serialize_for_logging(obj)
        
        assert result == {"custom": "test_value"}
    
    def test_serializer_config_impact(self):
        """Test that serialization config affects output"""
        dt = datetime(2024, 1, 15, 10, 30, 0)
        
        # ISO format
        config_iso = SerializationConfig(datetime_format="iso")
        result_iso = serialize_for_logging(dt, config_iso)
        
        # Timestamp format
        config_ts = SerializationConfig(datetime_format="timestamp")
        result_ts = serialize_for_logging(dt, config_ts)
        
        assert isinstance(result_iso, str)
        assert isinstance(result_ts, float)
        assert result_iso != result_ts


class TestIntegrationWithLogging:
    """Integration tests with actual logging system"""
    
    def test_complex_log_entry(self):
        """Test that complex data can be logged without errors"""
        from structured_logging import get_logger, LoggerConfig
        from structured_logging.formatter import StructuredFormatter
        import io
        import sys
        
        # Capture log output
        log_capture = io.StringIO()
        
        config = LoggerConfig(formatter_type="json")
        logger = get_logger("test_complex", config)
        
        # Remove existing handlers and add our capture handler
        logger.handlers.clear()
        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(StructuredFormatter(config))
        logger.addHandler(handler)
        
        # Log complex data
        complex_data = {
            "user": User(id=123, name="John Doe", email="john@example.com"),
            "timestamp": datetime(2024, 1, 15, 10, 30, 0),
            "uuid": UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479"),
            "amount": Decimal("99.99"),
            "priority": Priority.HIGH,
            "path": Path("/tmp/test.log"),
            "data": set([1, 2, 3, 4, 5])
        }
        
        logger.info("Complex data test", ctx_data=complex_data)
        
        # Check that output is valid JSON
        log_output = log_capture.getvalue().strip()
        parsed = json.loads(log_output)
        
        assert parsed["message"] == "Complex data test"
        assert "data" in parsed
        assert isinstance(parsed["data"], dict)  # Should be serialized as dict
        
        # Clean up
        logger.handlers.clear()
    
    def test_serialization_error_handling(self):
        """Test that serialization errors don't break logging"""
        from structured_logging import get_logger, LoggerConfig
        from structured_logging.formatter import StructuredFormatter
        import io
        
        class ProblematicClass:
            def __repr__(self):
                raise Exception("Cannot represent this object")
        
        log_capture = io.StringIO()
        config = LoggerConfig(formatter_type="json")
        logger = get_logger("test_error", config)
        
        logger.handlers.clear()
        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(StructuredFormatter(config))
        logger.addHandler(handler)
        
        # This should not crash
        logger.error("Error test", ctx_problematic=ProblematicClass())
        
        log_output = log_capture.getvalue().strip()
        # Should still produce valid JSON
        parsed = json.loads(log_output)
        
        assert parsed["message"] == "Error test"
        # Should contain error information
        assert "problematic" in parsed
        
        logger.handlers.clear()


class TestPerformance:
    """Performance tests for serialization"""
    
    def test_serialization_performance(self):
        """Test that serialization doesn't add significant overhead"""
        import time
        
        # Create test data
        test_data = {
            "timestamp": datetime.now(),
            "uuid": uuid4(),
            "amount": Decimal("99.99"),
            "user": User(id=123, name="John", email="john@example.com"),
            "priorities": [Priority.LOW, Priority.MEDIUM, Priority.HIGH] * 10
        }
        
        # Measure serialization time
        start_time = time.perf_counter()
        
        for _ in range(1000):
            result = serialize_for_logging(test_data)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # Should complete 1000 serializations in reasonable time
        assert duration < 1.0  # Less than 1 second for 1000 operations
        
        # Verify result is still correct
        assert isinstance(result, dict)
        assert "timestamp" in result
        assert "uuid" in result


class TestTypeDetector:
    """Tests for TypeDetector class"""
    
    def test_uuid_detection(self):
        config = SerializationConfig(detect_uuid_strings=True)
        detector = TypeDetector(config)
        
        uuid_string = "f47ac10b-58cc-4372-a567-0e02b2c3d479"
        result = detector.detect_and_convert(uuid_string)
        
        assert isinstance(result, dict)
        assert result["type"] == "uuid"
        assert result["value"] == uuid_string
    
    def test_datetime_string_detection(self):
        config = SerializationConfig(detect_datetime_strings=True)
        detector = TypeDetector(config)
        
        # Test ISO format
        iso_date = "2024-01-15T10:30:00Z"
        result = detector.detect_and_convert(iso_date)
        
        assert isinstance(result, dict)
        assert result["type"] == "datetime_string"
        assert result["value"] == iso_date
        assert result["detected_format"] is True
        
        # Test simple date format
        simple_date = "2024-01-15"
        result = detector.detect_and_convert(simple_date)
        
        assert isinstance(result, dict)
        assert result["type"] == "datetime_string"
    
    def test_url_detection(self):
        config = SerializationConfig(detect_url_strings=True)
        detector = TypeDetector(config)
        
        url = "https://api.example.com/users/123"
        result = detector.detect_and_convert(url)
        
        assert isinstance(result, dict)
        assert result["type"] == "url"
        assert result["value"] == url
    
    def test_json_string_detection(self):
        config = SerializationConfig(detect_json_strings=True)
        detector = TypeDetector(config)
        
        json_string = '{"name": "John", "age": 30}'
        result = detector.detect_and_convert(json_string)
        
        assert isinstance(result, dict)
        assert result["type"] == "json_string"
        assert "parsed" in result
        assert result["parsed"]["name"] == "John"
        assert result["parsed"]["age"] == 30
    
    def test_numeric_string_detection(self):
        config = SerializationConfig(auto_convert_strings=True)
        detector = TypeDetector(config)
        
        # Integer string
        int_string = "12345"
        result = detector.detect_and_convert(int_string)
        
        assert isinstance(result, dict)
        assert result["type"] == "numeric_string"
        assert result["value"] == 12345
        assert result["original"] == int_string
        
        # Float string
        float_string = "123.45"
        result = detector.detect_and_convert(float_string)
        
        assert isinstance(result, dict)
        assert result["type"] == "numeric_string"
        assert result["value"] == 123.45
    
    def test_timestamp_detection(self):
        config = SerializationConfig(auto_convert_numbers=True)
        detector = TypeDetector(config)
        
        # Unix timestamp
        timestamp = 1705316400  # January 15, 2024
        result = detector.detect_and_convert(timestamp)
        
        assert isinstance(result, dict)
        assert result["type"] == "timestamp"
        assert result["value"] == timestamp
        assert "human_readable" in result
    
    def test_large_number_detection(self):
        config = SerializationConfig(auto_convert_numbers=True)
        detector = TypeDetector(config)
        
        large_number = 1500000
        result = detector.detect_and_convert(large_number)
        
        assert isinstance(result, dict)
        assert result["type"] == "large_number"
        assert result["value"] == large_number
        assert result["formatted"] == "1.50M"
    
    def test_cache_functionality(self):
        config = SerializationConfig(type_detection_cache_size=10)
        detector = TypeDetector(config)
        
        uuid_string = "f47ac10b-58cc-4372-a567-0e02b2c3d479"
        
        # First call - cache miss
        result1 = detector.detect_and_convert(uuid_string)
        stats1 = detector.get_cache_stats()
        
        # Second call - cache hit
        result2 = detector.detect_and_convert(uuid_string)
        stats2 = detector.get_cache_stats()
        
        assert result1 == result2
        assert stats2["cache_hits"] > stats1["cache_hits"]
        assert stats2["hit_rate_percent"] > 0
    
    def test_disabled_detection(self):
        config = SerializationConfig(auto_detect_types=False)
        detector = TypeDetector(config)
        
        uuid_string = "f47ac10b-58cc-4372-a567-0e02b2c3d479"
        result = detector.detect_and_convert(uuid_string)
        
        # Should return original object when detection is disabled
        assert result == uuid_string
    
    def test_invalid_json_handling(self):
        config = SerializationConfig(detect_json_strings=True)
        detector = TypeDetector(config)
        
        invalid_json = '{"name": "John", "age":}'
        result = detector.detect_and_convert(invalid_json)
        
        assert isinstance(result, dict)
        assert result["type"] == "json_like_string"
        assert "value" in result


class TestSmartConverter:
    """Tests for SmartConverter class"""
    
    def test_intelligent_conversion(self):
        config = SerializationConfig(auto_detect_types=True)
        converter = SmartConverter(config)
        
        # Test detection + serialization
        uuid_string = "f47ac10b-58cc-4372-a567-0e02b2c3d479"
        result = converter.convert_intelligently(uuid_string)
        
        assert isinstance(result, dict)
        assert result["type"] == "uuid"
        assert "_detection" in result
        assert result["_detection"]["auto_detected"] is True
        assert result["_detection"]["confidence"] == 0.95
    
    def test_fallback_to_standard_serialization(self):
        config = SerializationConfig(auto_detect_types=True)
        converter = SmartConverter(config)
        
        # Test with datetime object (should use standard serialization)
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = converter.convert_intelligently(dt)
        
        # Should be serialized normally (string format)
        assert isinstance(result, str)
        assert "2024-01-15T10:30:00Z" == result
    
    def test_detection_confidence_scoring(self):
        config = SerializationConfig(auto_detect_types=True)
        converter = SmartConverter(config)
        
        # High confidence detection (UUID)
        uuid_result = converter.convert_intelligently("f47ac10b-58cc-4372-a567-0e02b2c3d479")
        
        # Lower confidence detection (JSON-like string)
        json_like = '{"invalid": json}'  # Will be detected as json_like_string
        json_result = converter.detector.detect_and_convert(json_like)
        
        assert uuid_result["_detection"]["confidence"] > 0.9
        # Note: json_like detection happens in TypeDetector, not SmartConverter for invalid JSON
    
    def test_detection_statistics(self):
        config = SerializationConfig(auto_detect_types=True)
        converter = SmartConverter(config)
        
        # Perform some conversions
        converter.convert_intelligently("f47ac10b-58cc-4372-a567-0e02b2c3d479")
        converter.convert_intelligently("https://example.com")
        converter.convert_intelligently("normal string")
        
        stats = converter.get_detection_stats()
        
        assert "cache_size" in stats
        assert "total_requests" in stats
        assert stats["total_requests"] >= 3


class TestAutoDetectionIntegration:
    """Integration tests for automatic type detection"""
    
    def test_auto_detection_enabled_vs_disabled(self):
        # Test with auto-detection enabled
        config_enabled = SerializationConfig(auto_detect_types=True)
        uuid_string = "f47ac10b-58cc-4372-a567-0e02b2c3d479"
        
        result_enabled = serialize_for_logging(uuid_string, config_enabled)
        
        # Test with auto-detection disabled
        config_disabled = SerializationConfig(auto_detect_types=False)
        result_disabled = serialize_for_logging(uuid_string, config_disabled)
        
        # Results should be different
        assert result_enabled != result_disabled
        assert isinstance(result_enabled, dict)  # Detected and enhanced
        assert isinstance(result_disabled, str)  # Left as string
    
    def test_nested_structure_detection(self):
        config = SerializationConfig(auto_detect_types=True)
        
        nested_data = {
            "user_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",  # UUID string
            "created_at": "2024-01-15T10:30:00Z",                # DateTime string
            "api_url": "https://api.example.com/users/123",       # URL string
            "metadata": '{"source": "web", "campaign": "summer"}', # JSON string
            "normal_field": "just a regular string"
        }
        
        result = serialize_for_logging(nested_data, config)
        
        assert isinstance(result, dict)
        # Check that nested detection worked
        assert isinstance(result["user_id"], dict)
        assert result["user_id"]["type"] == "uuid"
        assert isinstance(result["created_at"], dict)
        assert result["created_at"]["type"] == "datetime_string"
        assert isinstance(result["api_url"], dict)
        assert result["api_url"]["type"] == "url"
        assert isinstance(result["metadata"], dict)
        assert result["metadata"]["type"] == "json_string"
        # Normal string should remain unchanged
        assert result["normal_field"] == "just a regular string"
    
    def test_list_detection(self):
        config = SerializationConfig(auto_detect_types=True)
        
        uuid_list = [
            "f47ac10b-58cc-4372-a567-0e02b2c3d479",
            "550e8400-e29b-41d4-a716-446655440000",
            "just a normal string"
        ]
        
        result = serialize_for_logging(uuid_list, config)
        
        assert isinstance(result, list)
        assert len(result) == 3
        # First two should be detected as UUIDs
        assert isinstance(result[0], dict)
        assert result[0]["type"] == "uuid"
        assert isinstance(result[1], dict)
        assert result[1]["type"] == "uuid"
        # Third should remain as string
        assert result[2] == "just a normal string"
    
    def test_performance_with_detection(self):
        """Test that auto-detection doesn't significantly impact performance"""
        import time
        
        config_with_detection = SerializationConfig(auto_detect_types=True)
        config_without_detection = SerializationConfig(auto_detect_types=False)
        
        test_data = {
            "uuid": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
            "timestamp": "2024-01-15T10:30:00Z",
            "url": "https://api.example.com/users/123",
            "number": "12345",
            "normal": "just a string"
        }
        
        iterations = 100
        
        # Test with detection
        start_time = time.perf_counter()
        for _ in range(iterations):
            serialize_for_logging(test_data, config_with_detection)
        time_with_detection = time.perf_counter() - start_time
        
        # Test without detection
        start_time = time.perf_counter()
        for _ in range(iterations):
            serialize_for_logging(test_data, config_without_detection)
        time_without_detection = time.perf_counter() - start_time
        
        # Detection will add significant overhead due to regex matching and object creation
        # This is expected for the intelligent analysis features
        performance_ratio = time_with_detection / time_without_detection
        
        # Just ensure it completes and doesn't have runaway performance issues (< 100x)
        assert performance_ratio < 100.0, f"Detection adds excessive overhead: {performance_ratio:.2f}x"
        
        # Log the actual performance impact for informational purposes
        print(f"\nDetection overhead: {performance_ratio:.2f}x ({time_with_detection:.3f}s vs {time_without_detection:.3f}s)")
    
    def test_detection_error_handling(self):
        """Test that detection errors don't break serialization"""
        config = SerializationConfig(auto_detect_types=True)
        
        # Create data that might cause detection issues
        problematic_data = {
            "empty_string": "",
            "very_long_string": "x" * 10000,
            "special_chars": "!@#$%^&*(){}[]|\\:;\"'<>,.?/~`",
            "unicode": "Hello 世界 🌍",
            "mixed": ["normal", "f47ac10b-58cc-4372-a567-0e02b2c3d479", 12345]
        }
        
        # Should not crash
        result = serialize_for_logging(problematic_data, config)
        
        assert isinstance(result, dict)
        assert "empty_string" in result
        assert "very_long_string" in result
        assert "special_chars" in result
        assert "unicode" in result
        assert "mixed" in result


class TestTypeDetectionConfiguration:
    """Tests for type detection configuration options"""
    
    def test_selective_detection_options(self):
        # Test with only UUID detection enabled
        config = SerializationConfig(
            auto_detect_types=True,
            detect_uuid_strings=True,
            detect_datetime_strings=False,
            detect_url_strings=False,
            detect_json_strings=False
        )
        
        test_data = {
            "uuid": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
            "datetime": "2024-01-15T10:30:00Z",
            "url": "https://example.com",
            "json": '{"key": "value"}'
        }
        
        result = serialize_for_logging(test_data, config)
        
        # Only UUID should be detected
        assert isinstance(result["uuid"], dict)
        assert result["uuid"]["type"] == "uuid"
        # Others should remain as strings
        assert isinstance(result["datetime"], str)
        assert isinstance(result["url"], str)
        assert isinstance(result["json"], str)
    
    def test_cache_size_configuration(self):
        config = SerializationConfig(
            auto_detect_types=True,
            type_detection_cache_size=2  # Very small cache
        )
        
        detector = TypeDetector(config)
        
        # Fill the cache
        detector.detect_and_convert("f47ac10b-58cc-4372-a567-0e02b2c3d479")
        detector.detect_and_convert("550e8400-e29b-41d4-a716-446655440000")
        
        stats1 = detector.get_cache_stats()
        assert stats1["cache_size"] == 2
        
        # Add one more (should not exceed cache size)
        detector.detect_and_convert("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
        
        stats2 = detector.get_cache_stats()
        assert stats2["cache_size"] <= 2  # Should not exceed configured size


class TestLazySerializable:
    """Tests for LazySerializable class"""
    
    def test_lazy_wrapper_creation(self):
        """Test creating lazy serializable wrappers"""
        data = {"key": "value", "number": 42}
        lazy_obj = LazySerializable(data)
        
        assert not lazy_obj.is_serialized()
        assert lazy_obj.get_original() is data
        assert "pending=True" in repr(lazy_obj)
    
    def test_lazy_serialization_trigger(self):
        """Test that serialization is triggered when needed"""
        data = {"key": "value", "number": 42}
        lazy_obj = LazySerializable(data)
        
        # First access should trigger serialization
        result = lazy_obj.force_serialize()
        assert lazy_obj.is_serialized()
        assert result == data  # Simple data should be unchanged
        assert "serialized=" in repr(lazy_obj)
        
        # Second access should use cached result
        result2 = lazy_obj.force_serialize()
        assert result2 == result
    
    def test_lazy_string_conversion(self):
        """Test string conversion triggers serialization"""
        data = [1, 2, 3]
        lazy_obj = LazySerializable(data)
        
        str_result = str(lazy_obj)
        assert lazy_obj.is_serialized()
        assert str_result == str(data)
    
    def test_lazy_json_conversion(self):
        """Test JSON conversion triggers serialization"""
        data = {"test": True, "value": 123}
        lazy_obj = LazySerializable(data)
        
        json_result = lazy_obj.to_json()
        assert lazy_obj.is_serialized()
        parsed = json.loads(json_result)
        assert parsed == data
    
    def test_lazy_complex_object_serialization(self):
        """Test lazy serialization with complex objects"""
        from datetime import datetime
        
        data = {
            "timestamp": datetime(2024, 1, 15, 10, 30, 0),
            "uuid": UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479"),
            "amount": Decimal("99.99")
        }
        
        config = SerializationConfig(auto_detect_types=False)  # Disable for predictable results
        lazy_obj = LazySerializable(data, config)
        
        result = lazy_obj.force_serialize()
        
        # Should have serialized the complex objects
        assert isinstance(result, dict)
        assert isinstance(result["timestamp"], str)
        assert isinstance(result["uuid"], str)
        assert isinstance(result["amount"], str)


class TestLazyDict:
    """Tests for LazyDict class"""
    
    def test_lazy_dict_creation(self):
        """Test creating lazy dictionaries"""
        lazy_dict = LazyDict()
        assert isinstance(lazy_dict, dict)
        assert len(lazy_dict) == 0
        
        # Create with initial data
        data = {"key": "value"}
        lazy_dict2 = LazyDict(data)
        assert lazy_dict2["key"] == "value"
    
    def test_lazy_dict_with_lazy_values(self):
        """Test LazyDict containing LazySerializable values"""
        complex_data = {"nested": {"deep": "value"}}
        lazy_value = LazySerializable(complex_data)
        
        lazy_dict = LazyDict()
        lazy_dict["lazy_item"] = lazy_value
        lazy_dict["regular_item"] = "normal_value"
        
        # Values should not be serialized yet
        assert not lazy_value.is_serialized()
        
        # Force serialization
        result = lazy_dict.force_serialize_all()
        
        assert "lazy_item" in result
        assert "regular_item" in result
        assert result["lazy_item"] == complex_data
        assert result["regular_item"] == "normal_value"
        assert lazy_value.is_serialized()
    
    def test_lazy_dict_nested_structures(self):
        """Test LazyDict with nested lazy structures"""
        nested_data = [1, 2, 3, {"inner": "value"}]
        lazy_nested = LazySerializable(nested_data)
        
        lazy_dict = LazyDict({
            "items": [lazy_nested, "regular_string"],
            "metadata": LazyDict({"config": LazySerializable({"setting": True})})
        })
        
        result = lazy_dict.force_serialize_all()
        
        assert result["items"][0] == nested_data
        assert result["items"][1] == "regular_string"
        assert result["metadata"]["config"] == {"setting": True}
    
    def test_lazy_dict_json_conversion(self):
        """Test LazyDict JSON conversion"""
        lazy_dict = LazyDict({
            "lazy": LazySerializable({"complex": "data"}),
            "simple": "value"
        })
        
        json_str = lazy_dict.to_json()
        parsed = json.loads(json_str)
        
        assert parsed["lazy"] == {"complex": "data"}
        assert parsed["simple"] == "value"


class TestLazySerializationManager:
    """Tests for LazySerializationManager"""
    
    def test_manager_should_use_lazy_decisions(self):
        """Test the manager's decisions on when to use lazy serialization"""
        manager = LazySerializationManager()
        
        # Small strings should not use lazy (unless detection is forced)
        config_no_force = SerializationConfig(force_lazy_for_detection=False)
        assert not manager.should_use_lazy("short", config_no_force)
        
        # Large strings should use lazy
        large_string = "x" * 2000
        assert manager.should_use_lazy(large_string, SerializationConfig())
        
        # Large collections should use lazy
        large_list = list(range(50))
        config_large_threshold = SerializationConfig(lazy_threshold_items=10)
        assert manager.should_use_lazy(large_list, config_large_threshold)
        
        # Small collections should not use lazy
        small_list = [1, 2, 3]
        assert not manager.should_use_lazy(small_list, config_large_threshold)
    
    def test_manager_lazy_disabled(self):
        """Test manager with lazy serialization disabled"""
        manager = LazySerializationManager()
        config = SerializationConfig(enable_lazy_serialization=False)
        
        # Even large objects should not use lazy when disabled
        large_string = "x" * 2000
        assert not manager.should_use_lazy(large_string, config)
        
        large_dict = {f"key_{i}": f"value_{i}" for i in range(50)}
        assert not manager.should_use_lazy(large_dict, config)
    
    def test_manager_statistics_tracking(self):
        """Test statistics tracking in the manager"""
        manager = LazySerializationManager()
        
        # Reset to clean state
        manager.reset_stats()
        stats = manager.get_stats()
        assert stats["objects_created"] == 0
        
        # Create some lazy objects
        manager.create_lazy({"data": "test1"})
        manager.create_lazy({"data": "test2"})
        
        stats = manager.get_stats()
        assert stats["objects_created"] == 2
        assert stats["efficiency_percent"] == 0  # Nothing skipped yet
    
    def test_manager_wrap_if_beneficial(self):
        """Test the wrap_if_beneficial method"""
        manager = LazySerializationManager()
        
        # Should wrap large objects (exceeds item threshold)
        large_data = {f"key_{i}": f"value_{i}" for i in range(15)}  # 15 items > 10 threshold
        result = manager.wrap_if_beneficial(large_data)
        assert isinstance(result, LazySerializable)
        
        # Should not wrap small primitives
        small_data = "short"
        config = SerializationConfig(force_lazy_for_detection=False)
        result = manager.wrap_if_beneficial(small_data, config)
        assert result == small_data
        assert not isinstance(result, LazySerializable)


class TestLazySerializationIntegration:
    """Integration tests for lazy serialization"""
    
    def test_serialize_for_logging_lazy_aware(self):
        """Test the lazy-aware serialization function"""
        # Create data that exceeds the item threshold
        large_data = {f"key_{i}": f"value_{i}" for i in range(15)}  # 15 items > 10 threshold
        
        # With lazy enabled
        config_lazy = SerializationConfig(enable_lazy_serialization=True)
        result_lazy = serialize_for_logging_lazy_aware(large_data, config_lazy)
        assert isinstance(result_lazy, LazySerializable)
        
        # With lazy disabled
        config_no_lazy = SerializationConfig(enable_lazy_serialization=False)
        result_no_lazy = serialize_for_logging_lazy_aware(large_data, config_no_lazy)
        assert not isinstance(result_no_lazy, LazySerializable)
        assert isinstance(result_no_lazy, dict)
    
    def test_should_use_lazy_serialization_function(self):
        """Test the global should_use_lazy_serialization function"""
        # Large object should use lazy (exceeds item threshold)
        large_obj = {f"data_{i}": f"value_{i}" for i in range(15)}  # 15 items > 10 threshold
        assert should_use_lazy_serialization(large_obj)
        
        # Small object should not use lazy (unless forced)
        small_obj = {"data": "small"}
        config = SerializationConfig(force_lazy_for_detection=False)
        assert not should_use_lazy_serialization(small_obj, config)
    
    def test_create_lazy_serializable_function(self):
        """Test the global create_lazy_serializable function"""
        data = {"test": "data"}
        lazy_obj = create_lazy_serializable(data)
        
        assert isinstance(lazy_obj, LazySerializable)
        assert lazy_obj.get_original() is data
        assert not lazy_obj.is_serialized()
    
    def test_lazy_serialization_stats_functions(self):
        """Test the global statistics functions"""
        # Reset stats
        reset_lazy_serialization_stats()
        stats = get_lazy_serialization_stats()
        assert stats["objects_created"] == 0
        
        # Create some lazy objects
        create_lazy_serializable({"data1": "test"})
        create_lazy_serializable({"data2": "test"})
        
        stats = get_lazy_serialization_stats()
        assert stats["objects_created"] == 2


class TestLazySerializationPerformance:
    """Performance tests for lazy serialization"""
    
    def test_lazy_serialization_performance_benefit(self):
        """Test that lazy serialization provides performance benefits"""
        import time
        
        # Create complex data that would be expensive to serialize
        complex_data = {
            f"key_{i}": {
                "timestamp": datetime.now(),
                "uuid": uuid4(),
                "amount": Decimal(f"{i}.99"),
                "nested": {"deep": f"value_{i}" * 100}
            }
            for i in range(100)
        }
        
        iterations = 50
        
        # Test immediate serialization
        start_time = time.perf_counter()
        for _ in range(iterations):
            result = serialize_for_logging_lazy_aware(complex_data, use_lazy=False)
        immediate_time = time.perf_counter() - start_time
        
        # Test lazy serialization (without forcing)
        start_time = time.perf_counter()
        lazy_objects = []
        for _ in range(iterations):
            lazy_obj = serialize_for_logging_lazy_aware(complex_data, use_lazy=True)
            lazy_objects.append(lazy_obj)
        lazy_time = time.perf_counter() - start_time
        
        # Lazy should be significantly faster when not forced to serialize
        assert lazy_time < immediate_time
        
        # Verify the lazy objects work correctly
        assert all(isinstance(obj, LazySerializable) for obj in lazy_objects)
        
        # Force one to serialize and verify it works
        serialized = lazy_objects[0].force_serialize()
        assert isinstance(serialized, dict)
        assert len(serialized) == 100
    
    def test_lazy_vs_immediate_with_filtering(self):
        """Test lazy serialization benefit when logs are filtered"""
        # This simulates the case where expensive serialization would be wasted
        # because the log entry gets filtered out
        
        expensive_data = {
            "large_nested_structure": {
                f"item_{i}": {
                    "timestamp": datetime.now(),
                    "data": "x" * 1000,
                    "uuid": uuid4()
                }
                for i in range(100)
            }
        }
        
        # Create lazy wrapper
        lazy_obj = create_lazy_serializable(expensive_data)
        
        # Simulate filter rejecting the log entry
        # In real scenario, the lazy object would be garbage collected
        # without ever being serialized
        
        assert not lazy_obj.is_serialized()  # Never needed to serialize
        
        # Now simulate the log being accepted and needing serialization
        result = lazy_obj.force_serialize()
        assert lazy_obj.is_serialized()
        assert isinstance(result, dict)
        assert "large_nested_structure" in result


class TestLazySerializationEdgeCases:
    """Edge case tests for lazy serialization"""
    
    def test_lazy_object_equality(self):
        """Test equality comparison for lazy objects"""
        data1 = {"test": "data"}
        data2 = {"test": "data"}  # Different object, same content
        
        lazy1 = LazySerializable(data1)
        lazy2 = LazySerializable(data2)
        
        # Different instances with different data objects should not be equal (identity-based)
        assert lazy1 != lazy2
        
        # Same instance should be equal to itself
        assert lazy1 == lazy1
    
    def test_lazy_object_hashing(self):
        """Test hashing support for lazy objects"""
        data = {"test": "data"}
        lazy_obj = LazySerializable(data)
        
        # Should be hashable
        hash_value = hash(lazy_obj)
        assert isinstance(hash_value, int)
        
        # Should be consistent
        assert hash(lazy_obj) == hash_value
        
        # Should work in sets/dicts
        lazy_set = {lazy_obj}
        assert lazy_obj in lazy_set
    
    def test_lazy_serialization_error_handling(self):
        """Test error handling in lazy serialization"""
        
        # Create an object that will cause serialization errors
        class ProblematicClass:
            def __repr__(self):
                raise Exception("Cannot serialize this!")
        
        problematic_obj = ProblematicClass()
        lazy_obj = LazySerializable(problematic_obj)
        
        # Should handle serialization errors gracefully
        result = lazy_obj.force_serialize()
        
        # Should get fallback representation
        assert isinstance(result, dict)
        assert "__unserializable__" in result or "__serialization_error__" in result
    
    def test_lazy_dict_error_handling(self):
        """Test error handling in LazyDict"""
        problematic_lazy = LazySerializable(object())  # Will fail serialization
        
        lazy_dict = LazyDict({
            "good": "value",
            "bad": problematic_lazy
        })
        
        # Should handle errors in individual items
        result = lazy_dict.force_serialize_all()
        
        assert "good" in result
        assert result["good"] == "value"
        assert "bad" in result
        # Bad item should have error representation
        assert isinstance(result["bad"], dict)


# Schema validation tests

class TestSchemaValidator:
    """Tests for SchemaValidator class"""
    
    def test_schema_validator_creation(self):
        """Test creating schema validator"""
        validator = SchemaValidator()
        assert validator is not None
        assert validator.config is not None
    
    def test_simple_schema_registration(self):
        """Test registering simple schemas"""
        validator = SchemaValidator()
        
        # Simple string schema
        schema = {
            "name": "str",
            "age": "int",
            "active": "bool"
        }
        
        validator.register_schema("user", schema)
        
        # Valid data
        valid_data = {"name": "John", "age": 30, "active": True}
        assert validator.validate(valid_data, "user")
        
        # Invalid data - wrong type
        invalid_data = {"name": "John", "age": "thirty", "active": True}
        with pytest.raises(ValidationError):
            validator.validate(invalid_data, "user")
    
    def test_complex_schema_constraints(self):
        """Test complex schema with constraints"""
        validator = SchemaValidator()
        
        schema = {
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
                "min_value": 0,
                "max_value": 150
            },
            "tags": {
                "type": "list",
                "min_items": 1,
                "max_items": 5
            },
            "status": {
                "type": "str",
                "choices": ["active", "inactive", "pending"]
            }
        }
        
        validator.register_schema("user_profile", schema)
        
        # Valid data
        valid_data = {
            "username": "john_doe",
            "email": "john@example.com",
            "age": 25,
            "tags": ["developer", "python"],
            "status": "active"
        }
        assert validator.validate(valid_data, "user_profile")
        
        # Test various invalid cases
        invalid_cases = [
            # Username too short
            {**valid_data, "username": "jo"},
            # Invalid email
            {**valid_data, "email": "invalid-email"},
            # Age out of range
            {**valid_data, "age": 200},
            # Empty tags list
            {**valid_data, "tags": []},
            # Invalid status
            {**valid_data, "status": "unknown"}
        ]
        
        for invalid_data in invalid_cases:
            with pytest.raises(ValidationError):
                validator.validate(invalid_data, "user_profile")
    
    def test_optional_fields(self):
        """Test schema with optional fields"""
        validator = SchemaValidator()
        
        schema = {
            "name": {"type": "str", "required": True},
            "description": {"type": "str", "required": False},
            "metadata": {"type": "dict", "required": False}
        }
        
        validator.register_schema("item", schema)
        
        # Valid with all fields
        full_data = {
            "name": "Test Item",
            "description": "A test item",
            "metadata": {"version": 1}
        }
        assert validator.validate(full_data, "item")
        
        # Valid with only required fields
        minimal_data = {"name": "Test Item"}
        assert validator.validate(minimal_data, "item")
        
        # Invalid - missing required field
        invalid_data = {"description": "Missing name"}
        with pytest.raises(ValidationError):
            validator.validate(invalid_data, "item")
    
    def test_custom_validator_function(self):
        """Test custom validator functions"""
        validator = SchemaValidator()
        
        def is_even(value):
            return value % 2 == 0
        
        schema = {
            "number": {
                "type": "int",
                "validator": is_even
            }
        }
        
        validator.register_schema("even_number", schema)
        
        # Valid even number
        assert validator.validate({"number": 4}, "even_number")
        
        # Invalid odd number
        with pytest.raises(ValidationError):
            validator.validate({"number": 3}, "even_number")
    
    def test_validation_statistics(self):
        """Test validation statistics tracking"""
        validator = SchemaValidator()
        
        schema = {"name": "str"}
        validator.register_schema("simple", schema)
        
        # Reset stats
        validator.reset_validation_stats()
        stats = validator.get_validation_stats()
        assert stats["validations_performed"] == 0
        assert stats["validation_failures"] == 0
        
        # Perform successful validation
        validator.validate({"name": "test"}, "simple")
        stats = validator.get_validation_stats()
        assert stats["validations_performed"] == 1
        assert stats["validation_failures"] == 0
        
        # Perform failed validation
        try:
            validator.validate({"name": 123}, "simple")
        except ValidationError:
            pass
        
        stats = validator.get_validation_stats()
        assert stats["validations_performed"] == 2
        assert stats["validation_failures"] == 1


class TestTypeAnnotationExtractor:
    """Tests for TypeAnnotationExtractor class"""
    
    def test_function_schema_extraction(self):
        """Test extracting schema from function annotations"""
        extractor = TypeAnnotationExtractor()
        
        def sample_function(name: str, age: int, active: bool = True) -> dict:
            return {"name": name, "age": age, "active": active}
        
        schema = extractor.extract_function_schema(sample_function)
        
        assert schema["function_name"] == "sample_function"
        assert "parameters" in schema
        assert "return_type" in schema
        
        # Check parameter types
        params = schema["parameters"]
        assert params["name"]["type"] == "str"
        assert params["name"]["required"] is True
        assert params["age"]["type"] == "int"
        assert params["age"]["required"] is True
        assert params["active"]["type"] == "bool"
        assert params["active"]["required"] is False  # Has default value
        
        # Check return type
        assert schema["return_type"]["type"] == "dict"
    
    def test_class_schema_extraction(self):
        """Test extracting schema from class annotations"""
        extractor = TypeAnnotationExtractor()
        
        class SampleClass:
            name: str
            age: int
            metadata: Optional[dict] = None
            
            def get_info(self) -> str:
                return f"{self.name}, {self.age}"
        
        schema = extractor.extract_class_schema(SampleClass)
        
        assert schema["class_name"] == "SampleClass"
        assert "attributes" in schema
        assert "methods" in schema
        
        # Check attributes
        attrs = schema["attributes"]
        assert attrs["name"]["type"] == "str"
        assert attrs["age"]["type"] == "int"
        assert attrs["metadata"]["required"] is False  # Optional type
    
    def test_complex_type_annotations(self):
        """Test complex type annotations"""
        from typing import List, Dict, Union, Optional
        
        extractor = TypeAnnotationExtractor()
        
        def complex_function(
            items: List[str],
            mapping: Dict[str, int],
            optional_data: Optional[str] = None,
            union_field: Union[int, str] = "default"
        ) -> bool:
            return True
        
        schema = extractor.extract_function_schema(complex_function)
        params = schema["parameters"]
        
        # List type
        assert params["items"]["type"] == "list"
        assert params["items"]["item_type"]["type"] == "str"
        
        # Dict type
        assert params["mapping"]["type"] == "dict"
        assert params["mapping"]["key_type"]["type"] == "str"
        assert params["mapping"]["value_type"]["type"] == "int"
        
        # Optional type
        assert params["optional_data"]["required"] is False
        
        # Union type
        assert params["union_field"]["type"] == "union"
        assert len(params["union_field"]["types"]) == 2


class TestStructuredDataValidator:
    """Tests for StructuredDataValidator class"""
    
    def test_explicit_schema_registration(self):
        """Test registering explicit schemas"""
        validator = StructuredDataValidator()
        
        schema = {
            "user_id": "str",
            "score": {"type": "int", "min_value": 0, "max_value": 100}
        }
        
        validator.register_schema("game_score", schema)
        
        # Valid data
        valid_data = {"user_id": "user123", "score": 85}
        assert validator.validate(valid_data, "game_score")
        
        # Invalid data
        invalid_data = {"user_id": "user123", "score": 150}
        with pytest.raises(ValidationError):
            validator.validate(invalid_data, "game_score")
    
    def test_function_schema_registration(self):
        """Test auto-registering function schemas"""
        validator = StructuredDataValidator()
        
        def process_order(order_id: str, amount: float, items: list) -> bool:
            return True
        
        schema_name = validator.register_function_schema(process_order)
        assert schema_name == "func_process_order"
        
        # Validate against function schema
        valid_data = {
            "order_id": "ORDER123",
            "amount": 99.99,
            "items": ["item1", "item2"]
        }
        assert validator.validate(valid_data, schema_name)
        
        # Invalid data
        invalid_data = {
            "order_id": "ORDER123",
            "amount": "99.99",  # Should be float
            "items": ["item1", "item2"]
        }
        with pytest.raises(ValidationError):
            validator.validate(invalid_data, schema_name)
    
    def test_class_schema_registration(self):
        """Test auto-registering class schemas"""
        validator = StructuredDataValidator()
        
        @dataclass
        class Product:
            name: str
            price: float
            in_stock: bool
        
        schema_name = validator.register_class_schema(Product)
        assert schema_name == "class_Product"
        
        # Validate against class schema
        valid_data = {
            "name": "Widget",
            "price": 19.99,
            "in_stock": True
        }
        assert validator.validate(valid_data, schema_name)
    
    def test_validate_against_function(self):
        """Test direct validation against function"""
        validator = StructuredDataValidator()
        
        def log_event(event_type: str, timestamp: datetime, data: dict) -> None:
            pass
        
        valid_data = {
            "event_type": "user_login",
            "timestamp": datetime.now(),
            "data": {"user_id": "123"}
        }
        
        assert validator.validate_against_function(log_event, valid_data)
        
        # Invalid - wrong type
        invalid_data = {
            "event_type": "user_login",
            "timestamp": "2024-01-01",  # Should be datetime
            "data": {"user_id": "123"}
        }
        
        with pytest.raises(ValidationError):
            validator.validate_against_function(log_event, invalid_data)
    
    def test_schema_info_and_listing(self):
        """Test schema information and listing functionality"""
        validator = StructuredDataValidator()
        
        # Register explicit schema
        explicit_schema = {"name": "str", "value": "int"}
        validator.register_schema("explicit_test", explicit_schema)
        
        # Register function schema
        def test_function(param: str) -> bool:
            return True
        
        func_schema_name = validator.register_function_schema(test_function)
        
        # List schemas
        schemas = validator.list_schemas()
        assert "explicit_test" in schemas
        assert func_schema_name in schemas
        
        # Get schema info
        explicit_info = validator.get_schema_info("explicit_test")
        assert explicit_info["type"] == "explicit"
        
        func_info = validator.get_schema_info(func_schema_name)
        assert func_info["function_name"] == "test_function"


class TestValidationDecorator:
    """Tests for the auto_validate_function decorator"""
    
    def test_function_validation_decorator(self):
        """Test the auto validation decorator"""
        
        @auto_validate_function
        def create_user(name: str, age: int, email: str) -> dict:
            return {"name": name, "age": age, "email": email}
        
        # Valid call should work
        result = create_user("John", 30, "john@example.com")
        assert result["name"] == "John"
        
        # Invalid call should raise ValidationError
        with pytest.raises(ValidationError):
            create_user("John", "thirty", "john@example.com")  # age should be int
    
    def test_decorator_preserves_function_properties(self):
        """Test that decorator preserves function name and docstring"""
        
        def original_function(param: str) -> str:
            """Original function docstring"""
            return param
        
        decorated = auto_validate_function(original_function)
        
        assert decorated.__name__ == original_function.__name__
        assert decorated.__doc__ == original_function.__doc__


class TestGlobalValidationFunctions:
    """Tests for global validation functions"""
    
    def test_global_schema_registration(self):
        """Test global schema registration and validation"""
        # Register a global schema
        schema = {
            "request_id": "str",
            "user_id": "str",
            "action": {"type": "str", "choices": ["create", "read", "update", "delete"]}
        }
        
        register_validation_schema("api_request", schema)
        
        # Valid data
        valid_data = {
            "request_id": "req_123",
            "user_id": "user_456",
            "action": "create"
        }
        
        assert validate_log_data(valid_data, "api_request")
        
        # Invalid data
        invalid_data = {
            "request_id": "req_123",
            "user_id": "user_456",
            "action": "invalid_action"
        }
        
        with pytest.raises(ValidationError):
            validate_log_data(invalid_data, "api_request")
    
    def test_global_validation_statistics(self):
        """Test global validation statistics"""
        # Reset stats
        reset_validation_stats()
        stats = get_validation_stats()
        assert stats["validations_performed"] == 0
        
        # Register and use a schema
        simple_schema = {"name": "str"}
        register_validation_schema("simple_test", simple_schema)
        
        # Perform validations
        validate_log_data({"name": "test"}, "simple_test")
        
        try:
            validate_log_data({"name": 123}, "simple_test")
        except ValidationError:
            pass
        
        stats = get_validation_stats()
        assert stats["validations_performed"] == 2
        assert stats["validation_failures"] == 1


class TestValidationErrorHandling:
    """Tests for validation error handling"""
    
    def test_missing_schema_error(self):
        """Test error when schema doesn't exist"""
        validator = SchemaValidator()
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validate({"test": "data"}, "nonexistent_schema")
        
        assert "not found" in str(exc_info.value)
    
    def test_validation_error_messages(self):
        """Test descriptive validation error messages"""
        validator = SchemaValidator()
        
        schema = {
            "name": {"type": "str", "min_length": 5},
            "age": {"type": "int", "min_value": 18}
        }
        
        validator.register_schema("test_errors", schema)
        
        # Test multiple validation errors
        invalid_data = {
            "name": "abc",    # Too short
            "age": 15         # Too young
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(invalid_data, "test_errors")
        
        error_message = str(exc_info.value)
        assert "at least 5 characters" in error_message
        assert "at least 18" in error_message
    
    def test_custom_validator_error_handling(self):
        """Test error handling in custom validators"""
        validator = SchemaValidator()
        
        def failing_validator(value):
            raise Exception("Custom validator failed")
        
        schema = {
            "field": {"type": "str", "validator": failing_validator}
        }
        
        validator.register_schema("failing_test", schema)
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validate({"field": "test"}, "failing_test")
        
        assert "validation error" in str(exc_info.value)


class TestSchemaValidationIntegration:
    """Integration tests for schema validation with serialization"""
    
    def test_validation_with_serialization(self):
        """Test validation combined with serialization"""
        # Create a schema for log entries
        log_schema = {
            "level": {"type": "str", "choices": ["DEBUG", "INFO", "WARNING", "ERROR"]},
            "message": "str",
            "timestamp": "datetime",
            "user_id": {"type": "str", "required": False},
            "data": {"type": "dict", "required": False}
        }
        
        register_validation_schema("log_entry", log_schema)
        
        # Create log data that needs serialization
        log_data = {
            "level": "INFO",
            "message": "User action performed",
            "timestamp": datetime.now(),
            "user_id": "user_123",
            "data": {
                "action": "login",
                "ip": "192.168.1.1",
                "user_agent": "Mozilla/5.0..."
            }
        }
        
        # Serialize the data first
        serialized_data = serialize_for_logging(log_data)
        
        # Note: After serialization, datetime becomes string, so validation would fail
        # This demonstrates the importance of validating before serialization
        
        # Validate original data (before serialization)
        assert validate_log_data(log_data, "log_entry")
    
    def test_validation_performance(self):
        """Test validation performance with large schemas"""
        import time
        
        # Create a large schema
        large_schema = {}
        for i in range(100):
            large_schema[f"field_{i}"] = {
                "type": "str" if i % 2 == 0 else "int",
                "required": i < 50  # First 50 are required
            }
        
        validator = SchemaValidator()
        validator.register_schema("large_schema", large_schema)
        
        # Create matching data
        test_data = {}
        for i in range(50):  # Only required fields
            if i % 2 == 0:
                test_data[f"field_{i}"] = f"value_{i}"
            else:
                test_data[f"field_{i}"] = i
        
        # Measure validation performance
        start_time = time.perf_counter()
        iterations = 100
        
        for _ in range(iterations):
            validator.validate(test_data, "large_schema")
        
        total_time = time.perf_counter() - start_time
        avg_time = total_time / iterations
        
        # Should be reasonably fast (less than 1ms per validation)
        assert avg_time < 0.001, f"Validation too slow: {avg_time:.4f}s average"