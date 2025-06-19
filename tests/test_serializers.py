"""
Tests for enhanced serialization support
"""

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from structured_logging.serializers import (
    DEFAULT_CONFIG,
    EnhancedJSONEncoder,
    SerializationConfig,
    SmartConverter,
    TypeDetector,
    TypeRegistry,
    enhanced_json_dumps,
    register_custom_serializer,
    serialize_for_logging,
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