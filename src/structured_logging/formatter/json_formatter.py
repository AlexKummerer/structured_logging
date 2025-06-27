"""
JSON formatter for structured logging with enhanced serialization
"""

import json
import logging
from typing import Any, Dict, Optional

from ..config import LoggerConfig, get_default_config
from ..performance import fast_timestamp
from ..serializers import (
    EnhancedJSONEncoder,
    LazyDict,
    LazySerializable,
    SerializationConfig,
    serialize_for_logging_lazy_aware,
)


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging with enhanced serialization"""

    def __init__(
        self,
        config: Optional[LoggerConfig] = None,
        serialization_config: Optional[SerializationConfig] = None,
    ):
        super().__init__()
        self.config = config or get_default_config()
        self.serialization_config = serialization_config or SerializationConfig()

    def format(self, record: logging.LogRecord) -> str:
        # Performance optimization: Pre-build log entry with required fields
        log_entry: Dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Optimized timestamp generation: Use fast cached timestamp
        if self.config.include_timestamp:
            log_entry["timestamp"] = fast_timestamp()

        # Enhanced context extraction with lazy serialization support
        for key, value in record.__dict__.items():
            if key.startswith("ctx_"):
                context_key = key[4:]  # Remove ctx_ prefix
                # Use lazy-aware serialization for complex types
                log_entry[context_key] = serialize_for_logging_lazy_aware(
                    value, self.serialization_config
                )

        # Handle lazy objects in JSON encoding
        return self._serialize_to_json(log_entry)

    def _serialize_to_json(self, log_entry: Dict[str, Any]) -> str:
        """
        Serialize log entry to JSON, handling lazy objects appropriately

        Args:
            log_entry: Dictionary containing log data with potential lazy objects

        Returns:
            JSON string representation
        """
        # Check if we have any lazy objects that need special handling
        has_lazy_objects = any(
            isinstance(value, (LazySerializable, LazyDict))
            for value in log_entry.values()
        )

        if has_lazy_objects:
            # Use custom encoder that knows how to handle lazy objects
            return json.dumps(log_entry, cls=EnhancedJSONEncoder, separators=(",", ":"))
        else:
            # Fast path for non-lazy objects
            return json.dumps(log_entry, separators=(",", ":"))