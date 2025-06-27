"""
CSV formatter for structured logging with enhanced serialization
"""

import csv
import io
import json
import logging
from typing import Any, Dict, Optional

from ..config import LoggerConfig, get_default_config
from ..performance import fast_timestamp
from ..serializers import SerializationConfig, serialize_for_logging


class CSVFormatter(logging.Formatter):
    """CSV formatter for structured logging with enhanced serialization"""

    def __init__(
        self,
        config: Optional[LoggerConfig] = None,
        serialization_config: Optional[SerializationConfig] = None,
    ):
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
                    context_items[context_key] = json.dumps(
                        serialized, separators=(",", ":")
                    )
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