import csv
import io
import json
import logging
from typing import Any, Dict, Optional

from .config import LoggerConfig, get_default_config
from .performance import fast_timestamp
from .serializers import SerializationConfig, serialize_for_logging


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging with enhanced serialization"""

    def __init__(self, config: Optional[LoggerConfig] = None, serialization_config: Optional[SerializationConfig] = None):
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

        # Enhanced context extraction with serialization support
        for key, value in record.__dict__.items():
            if key.startswith("ctx_"):
                context_key = key[4:]  # Remove ctx_ prefix
                # Use enhanced serialization for complex types
                log_entry[context_key] = serialize_for_logging(value, self.serialization_config)

        return json.dumps(log_entry, separators=(",", ":"))


class CSVFormatter(logging.Formatter):
    """CSV formatter for structured logging with enhanced serialization"""

    def __init__(self, config: Optional[LoggerConfig] = None, serialization_config: Optional[SerializationConfig] = None):
        super().__init__()
        self.config = config or get_default_config()
        self.serialization_config = serialization_config or SerializationConfig()
        self.fieldnames = ["level", "logger", "message"]
        if self.config.include_timestamp:
            self.fieldnames.insert(0, "timestamp")

    def format(self, record: logging.LogRecord) -> str:
        # Performance optimization: Build log entry efficiently
        log_entry: Dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Optimized timestamp generation
        if self.config.include_timestamp:
            log_entry["timestamp"] = fast_timestamp()

        # Enhanced context field extraction with serialization
        context_items = {}
        for key, value in record.__dict__.items():
            if key.startswith("ctx_"):
                context_key = key[4:]
                # Serialize complex types and convert to string for CSV
                serialized = serialize_for_logging(value, self.serialization_config)
                # Convert to string representation for CSV compatibility
                if isinstance(serialized, (dict, list)):
                    context_items[context_key] = json.dumps(serialized, separators=(',', ':'))
                else:
                    context_items[context_key] = str(serialized)

        log_entry.update(context_items)

        # Performance optimization: Calculate fieldnames once
        all_fieldnames = self.fieldnames + sorted(context_items.keys())

        # Efficient CSV generation
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=all_fieldnames,
            extrasaction="ignore",
            lineterminator="",  # No extra newline
        )
        writer.writerow(log_entry)
        csv_string = output.getvalue()
        output.close()

        return csv_string


class PlainTextFormatter(logging.Formatter):
    """Plain text formatter for structured logging with enhanced serialization"""

    def __init__(self, config: Optional[LoggerConfig] = None, serialization_config: Optional[SerializationConfig] = None):
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
                    value_str = json.dumps(serialized, separators=(',', ':'))
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
