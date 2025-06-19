import csv
import io
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from .config import LoggerConfig, get_default_config


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def __init__(self, config: Optional[LoggerConfig] = None):
        super().__init__()
        self.config = config or get_default_config()

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if self.config.include_timestamp:
            log_entry["timestamp"] = datetime.now().isoformat() + "Z"

        for key, value in record.__dict__.items():
            if key.startswith("ctx_"):
                log_entry[key[4:]] = value

        return json.dumps(log_entry, default=str)


class CSVFormatter(logging.Formatter):
    """CSV formatter for structured logging"""

    def __init__(self, config: Optional[LoggerConfig] = None):
        super().__init__()
        self.config = config or get_default_config()
        self.fieldnames = ["level", "logger", "message"]
        if self.config.include_timestamp:
            self.fieldnames.insert(0, "timestamp")

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if self.config.include_timestamp:
            log_entry["timestamp"] = datetime.now().isoformat() + "Z"

        # Add context fields
        context_fields = []
        for key, value in record.__dict__.items():
            if key.startswith("ctx_"):
                field_name = key[4:]
                log_entry[field_name] = value
                if field_name not in context_fields:
                    context_fields.append(field_name)

        # Update fieldnames to include context fields
        all_fieldnames = self.fieldnames + sorted(context_fields)

        # Create CSV string
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=all_fieldnames, extrasaction="ignore")
        writer.writerow(log_entry)
        csv_string = output.getvalue().strip()
        output.close()

        return csv_string


class PlainTextFormatter(logging.Formatter):
    """Plain text formatter for structured logging"""

    def __init__(self, config: Optional[LoggerConfig] = None):
        super().__init__()
        self.config = config or get_default_config()

    def format(self, record: logging.LogRecord) -> str:
        parts = []

        if self.config.include_timestamp:
            timestamp = datetime.now().isoformat() + "Z"
            parts.append(f"[{timestamp}]")

        parts.append(f"{record.levelname}")
        parts.append(f"{record.name}")
        parts.append(f"{record.getMessage()}")

        # Add context fields
        context_parts = []
        for key, value in record.__dict__.items():
            if key.startswith("ctx_"):
                field_name = key[4:]
                context_parts.append(f"{field_name}={value}")

        if context_parts:
            parts.append(f"({', '.join(context_parts)})")

        return " ".join(parts)
