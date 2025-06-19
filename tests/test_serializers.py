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