import csv
import io
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from .config import LoggerConfig, get_default_config
from .performance import fast_timestamp


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def __init__(self, config: Optional[LoggerConfig] = None):
        super().__init__()
        self.config = config or get_default_config()

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

        # Optimized context extraction: Use generator for memory efficiency
        context_items = (
            (key[4:], value) for key, value in record.__dict__.items() 
            if key.startswith("ctx_")
        )
        log_entry.update(context_items)

        return json.dumps(log_entry, default=str, separators=(',', ':'))


class CSVFormatter(logging.Formatter):
    """CSV formatter for structured logging"""

    def __init__(self, config: Optional[LoggerConfig] = None):
        super().__init__()
        self.config = config or get_default_config()
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

        # Optimized context field extraction
        context_items = {
            key[4:]: value for key, value in record.__dict__.items() 
            if key.startswith("ctx_")
        }
        log_entry.update(context_items)

        # Performance optimization: Calculate fieldnames once
        all_fieldnames = self.fieldnames + sorted(context_items.keys())

        # Efficient CSV generation
        output = io.StringIO()
        writer = csv.DictWriter(
            output, 
            fieldnames=all_fieldnames, 
            extrasaction="ignore",
            lineterminator=""  # No extra newline
        )
        writer.writerow(log_entry)
        csv_string = output.getvalue()
        output.close()

        return csv_string


class PlainTextFormatter(logging.Formatter):
    """Plain text formatter for structured logging"""

    def __init__(self, config: Optional[LoggerConfig] = None):
        super().__init__()
        self.config = config or get_default_config()

    def format(self, record: logging.LogRecord) -> str:
        # Performance optimization: Pre-allocate list with expected size
        parts = []

        # Optimized timestamp generation
        if self.config.include_timestamp:
            parts.append(f"[{fast_timestamp()}]")

        # Core message parts
        parts.extend([
            record.levelname,
            record.name,
            record.getMessage()
        ])

        # Optimized context extraction: Use generator and join directly
        context_items = (
            f"{key[4:]}={value}" for key, value in record.__dict__.items() 
            if key.startswith("ctx_")
        )
        context_str = ", ".join(context_items)
        
        if context_str:
            parts.append(f"({context_str})")

        return " ".join(parts)
