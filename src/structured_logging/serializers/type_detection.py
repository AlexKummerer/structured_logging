"""
Automatic type detection and conversion system
"""

import json
import re
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Dict, Optional, Union

from .config import SerializationConfig


class TypeDetector:
    """Automatic type detection and conversion system"""

    def __init__(self, config: SerializationConfig):
        self.config = config
        self._cache = {}  # Simple cache for detected types
        self._cache_hits = 0
        self._cache_misses = 0

        # Pre-compiled regex patterns for efficiency
        self._uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            re.IGNORECASE,
        )

        self._datetime_patterns = [
            # ISO 8601 formats
            re.compile(
                r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?$"
            ),
            # Common date formats
            re.compile(r"^\d{4}-\d{2}-\d{2}$"),
            re.compile(r"^\d{2}/\d{2}/\d{4}$"),
            re.compile(r"^\d{2}-\d{2}-\d{4}$"),
            # Common datetime formats
            re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"),
            re.compile(r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}$"),
        ]

        self._url_pattern = re.compile(
            r"^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$",
            re.IGNORECASE,
        )

        self._number_patterns = {
            "integer": re.compile(r"^[+-]?\d+$"),
            "float": re.compile(r"^[+-]?\d*\.\d+$"),
            "scientific": re.compile(r"^[+-]?\d*\.?\d+[eE][+-]?\d+$"),
            "decimal": re.compile(r"^[+-]?\d+\.\d{2,}$"),  # Likely currency/precision
        }

    def _should_skip_detection(self, obj: Any) -> bool:
        """Check if type detection should be skipped for this object"""
        if not self.config.auto_detect_types:
            return True
        # Skip if already a complex type that we handle
        return not isinstance(obj, (str, int, float))

    def _get_cache_key(self, obj: Any) -> tuple:
        """Generate cache key for object"""
        return (type(obj).__name__, str(obj)[:100])  # Limit cache key length

    def _check_cache(self, cache_key: tuple) -> Optional[Callable]:
        """Check cache for converter, update stats"""
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        self._cache_misses += 1
        return None

    def _update_cache(self, cache_key: tuple, converter: Optional[Callable]) -> None:
        """Update cache with converter if within size limit"""
        if len(self._cache) < self.config.type_detection_cache_size:
            self._cache[cache_key] = converter

    def _apply_converter(self, obj: Any, converter: Optional[Callable]) -> Any:
        """Apply converter to object, handle exceptions"""
        if converter is None:
            return obj
        try:
            return converter(obj)
        except Exception:
            return obj  # If conversion fails, return original

    def detect_and_convert(self, obj: Any) -> Any:
        """
        Detect type and auto-convert if appropriate

        Args:
            obj: Object to detect and potentially convert

        Returns:
            Original object or converted version
        """
        if self._should_skip_detection(obj):
            return obj

        cache_key = self._get_cache_key(obj)
        converter = self._check_cache(cache_key)
        
        if converter is None:
            converter = self._detect_type_converter(obj)
            self._update_cache(cache_key, converter)
        
        return self._apply_converter(obj, converter)

    def _detect_type_converter(self, obj: Any) -> Optional[Callable]:
        """Detect appropriate type converter for an object"""

        if isinstance(obj, str):
            return self._detect_string_type(obj)
        elif isinstance(obj, (int, float)) and self.config.auto_convert_numbers:
            return self._detect_numeric_type(obj)

        return None

    def _detect_string_type(self, value: str) -> Optional[Callable]:
        """Detect type for string values"""
        if not self.config.auto_convert_strings or not value:
            return None

        # UUID detection
        if self.config.detect_uuid_strings and self._uuid_pattern.match(value):
            return lambda x: {"type": "uuid", "value": x}

        # DateTime detection
        if self.config.detect_datetime_strings:
            for pattern in self._datetime_patterns:
                if pattern.match(value):
                    return lambda x: {
                        "type": "datetime_string",
                        "value": x,
                        "detected_format": True,
                    }

        # URL detection
        if self.config.detect_url_strings and self._url_pattern.match(value):
            return lambda x: {"type": "url", "value": x}

        # JSON detection
        if self.config.detect_json_strings and self._looks_like_json(value):
            return self._json_converter

        # Numeric string detection
        for num_type, pattern in self._number_patterns.items():
            if pattern.match(value):
                return self._get_numeric_converter(num_type)

        return None

    def _detect_numeric_type(self, value: Union[int, float]) -> Optional[Callable]:
        """Detect enhanced representation for numeric values"""

        # Timestamp detection (Unix timestamp)
        if isinstance(value, (int, float)) and 1000000000 <= value <= 9999999999:
            # Likely a timestamp (between 2001 and 2286)
            return lambda x: {
                "type": "timestamp",
                "value": x,
                "human_readable": datetime.fromtimestamp(x).isoformat() + "Z",
            }

        # Large numbers with unit detection
        if isinstance(value, (int, float)) and abs(value) >= 1000:
            return lambda x: {
                "type": "large_number",
                "value": x,
                "formatted": self._format_large_number(x),
            }

        return None

    def _looks_like_json(self, value: str) -> bool:
        """Quick check if string looks like JSON"""
        if len(value) < 2:
            return False

        stripped = value.strip()
        return (stripped.startswith("{") and stripped.endswith("}")) or (
            stripped.startswith("[") and stripped.endswith("]")
        )

    def _json_converter(self, value: str) -> Dict[str, Any]:
        """Convert JSON string to structured representation"""
        try:
            parsed = json.loads(value)
            return {
                "type": "json_string",
                "parsed": parsed,
                "original_length": len(value),
            }
        except json.JSONDecodeError:
            return {
                "type": "json_like_string",
                "value": value[:100] + "..." if len(value) > 100 else value,
            }

    def _get_numeric_converter(self, num_type: str) -> Callable:
        """Get converter for numeric strings"""

        def converter(value: str) -> Dict[str, Any]:
            try:
                if num_type == "integer":
                    return {
                        "type": "numeric_string",
                        "value": int(value),
                        "original": value,
                    }
                elif num_type == "float":
                    return {
                        "type": "numeric_string",
                        "value": float(value),
                        "original": value,
                    }
                elif num_type == "decimal":
                    return {
                        "type": "decimal_string",
                        "value": str(Decimal(value)),
                        "original": value,
                    }
                elif num_type == "scientific":
                    return {
                        "type": "scientific_string",
                        "value": float(value),
                        "original": value,
                    }
            except (ValueError, TypeError):
                pass
            return {"type": "numeric_like_string", "value": value}

        return converter

    def _format_large_number(self, value: Union[int, float]) -> str:
        """Format large numbers with appropriate units"""
        abs_value = abs(value)

        if abs_value >= 1_000_000_000_000:  # Trillion
            return f"{value / 1_000_000_000_000:.2f}T"
        elif abs_value >= 1_000_000_000:  # Billion
            return f"{value / 1_000_000_000:.2f}B"
        elif abs_value >= 1_000_000:  # Million
            return f"{value / 1_000_000:.2f}M"
        elif abs_value >= 1_000:  # Thousand
            return f"{value / 1_000:.2f}K"
        else:
            return str(value)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (
            (self._cache_hits / total_requests * 100) if total_requests > 0 else 0
        )

        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests,
        }

    def clear_cache(self) -> None:
        """Clear the type detection cache"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0