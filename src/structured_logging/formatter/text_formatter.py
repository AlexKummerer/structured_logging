"""
Plain text formatter for structured logging with enhanced serialization
"""

import json
import logging
from typing import Any, Dict, Optional

from ..config import LoggerConfig, get_default_config
from ..performance import fast_timestamp
from ..serializers import SerializationConfig, serialize_for_logging


class PlainTextFormatter(logging.Formatter):
    """Plain text formatter for structured logging with enhanced serialization"""

    def __init__(
        self,
        config: Optional[LoggerConfig] = None,
        serialization_config: Optional[SerializationConfig] = None,
    ):
        super().__init__()
        self.config = config or get_default_config()
        self.serialization_config = serialization_config or SerializationConfig()

    def format(self, record: logging.LogRecord) -> str:
        # Performance optimization: Pre-allocate list with expected size
        parts = []

        # Optimized timestamp generation
        if self.config.include_timestamp:
            parts.append(f"[{fast_timestamp()}]")

        # Core message parts
        parts.extend([record.levelname, record.name, record.getMessage()])

        # Enhanced context extraction with serialization
        context_items = []
        for key, value in record.__dict__.items():
            if key.startswith("ctx_"):
                context_key = key[4:]
                # Serialize complex types for readable plain text
                serialized = serialize_for_logging(value, self.serialization_config)
                # Convert to compact string representation
                if isinstance(serialized, (dict, list)):
                    value_str = json.dumps(serialized, separators=(",", ":"))
                    # Truncate very long values for readability
                    if len(value_str) > 100:
                        value_str = value_str[:97] + "..."
                else:
                    value_str = str(serialized)
                context_items.append(f"{context_key}={value_str}")

        if context_items:
            context_str = ", ".join(context_items)
            parts.append(f"({context_str})")

        return " ".join(parts)