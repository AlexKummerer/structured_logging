"""
Smart converter that combines type detection with serialization
"""

from datetime import datetime
from typing import Any, Dict

from .config import SerializationConfig
from .registry import TypeRegistry
from .type_detection import TypeDetector

# Forward import to avoid circular dependency
serialize_for_logging = None


class SmartConverter:
    """Smart converter that combines type detection with serialization"""

    def __init__(self, config: SerializationConfig):
        self.config = config
        self.detector = TypeDetector(config)
        self.type_registry = TypeRegistry()

    def convert_intelligently(self, obj: Any) -> Any:
        """
        Intelligently convert object using detection + serialization

        Args:
            obj: Object to convert

        Returns:
            Intelligently converted object
        """
        # First, try auto-detection and conversion
        detected = self.detector.detect_and_convert(obj)

        # If detection changed the object, serialize the new version
        if detected != obj:
            return self._serialize_detected(detected)

        # Otherwise, use standard serialization without auto-detection to avoid recursion
        config_without_detection = SerializationConfig(
            datetime_format=self.config.datetime_format,
            custom_datetime_format=self.config.custom_datetime_format,
            decimal_as_float=self.config.decimal_as_float,
            include_type_hints=self.config.include_type_hints,
            max_collection_size=self.config.max_collection_size,
            truncate_strings=self.config.truncate_strings,
            enum_as_value=self.config.enum_as_value,
            dataclass_as_dict=self.config.dataclass_as_dict,
            path_as_string=self.config.path_as_string,
            numpy_array_max_size=self.config.numpy_array_max_size,
            pandas_max_rows=self.config.pandas_max_rows,
            pandas_include_dtypes=self.config.pandas_include_dtypes,
            auto_detect_types=False,  # Disable auto-detection to prevent recursion
        )

        # Import here to avoid circular dependency
        global serialize_for_logging
        if serialize_for_logging is None:
            from .base import serialize_for_logging

        return serialize_for_logging(obj, config_without_detection)

    def _serialize_detected(self, obj: Any) -> Any:
        """Serialize a detected/converted object"""
        # If it's a dict with type info, enhance it
        if isinstance(obj, dict) and "type" in obj:
            return self._enhance_detected_object(obj)

        # Otherwise serialize normally without auto-detection to avoid recursion
        config_without_detection = SerializationConfig(
            datetime_format=self.config.datetime_format,
            custom_datetime_format=self.config.custom_datetime_format,
            decimal_as_float=self.config.decimal_as_float,
            include_type_hints=self.config.include_type_hints,
            max_collection_size=self.config.max_collection_size,
            truncate_strings=self.config.truncate_strings,
            enum_as_value=self.config.enum_as_value,
            dataclass_as_dict=self.config.dataclass_as_dict,
            path_as_string=self.config.path_as_string,
            numpy_array_max_size=self.config.numpy_array_max_size,
            pandas_max_rows=self.config.pandas_max_rows,
            pandas_include_dtypes=self.config.pandas_include_dtypes,
            auto_detect_types=False,  # Disable auto-detection to prevent recursion
        )

        # Import here to avoid circular dependency
        global serialize_for_logging
        if serialize_for_logging is None:
            from .base import serialize_for_logging

        return serialize_for_logging(obj, config_without_detection)

    def _enhance_detected_object(self, obj: dict) -> dict:
        """Enhance detected objects with additional metadata"""
        result = obj.copy()

        # Add confidence and detection metadata
        result["_detection"] = {
            "detected_at": datetime.now().isoformat() + "Z",
            "confidence": self._calculate_confidence(obj),
            "auto_detected": True,
        }

        return result

    def _calculate_confidence(self, obj: dict) -> float:
        """Calculate confidence score for detection"""
        obj_type = obj.get("type", "unknown")

        # Higher confidence for stricter patterns
        confidence_map = {
            "uuid": 0.95,
            "url": 0.90,
            "timestamp": 0.85,
            "json_string": 0.80,
            "decimal_string": 0.75,
            "numeric_string": 0.70,
            "datetime_string": 0.65,
            "large_number": 0.60,
            "json_like_string": 0.30,
            "numeric_like_string": 0.25,
        }

        return confidence_map.get(obj_type, 0.50)

    def get_detection_stats(self) -> Dict[str, Any]:
        """Get type detection statistics"""
        return self.detector.get_cache_stats()