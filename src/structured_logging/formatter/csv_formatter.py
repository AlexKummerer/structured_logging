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

    def _build_base_log_entry(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Build base log entry from record"""
        log_entry: Dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        if self.config.include_timestamp:
            log_entry["timestamp"] = fast_timestamp()
            
        return log_entry

    def _extract_context_items(self, record: logging.LogRecord) -> Dict[str, str]:
        """Extract and serialize context fields from record"""
        context_items = {}
        for key, value in record.__dict__.items():
            if key.startswith("ctx_"):
                context_key = key[4:]
                serialized = serialize_for_logging(value, self.serialization_config)
                
                # Convert to string representation for CSV compatibility
                if isinstance(serialized, (dict, list)):
                    context_items[context_key] = json.dumps(
                        serialized, separators=(",", ":")
                    )
                else:
                    context_items[context_key] = str(serialized)
        
        return context_items

    def _generate_csv_string(self, log_entry: Dict[str, Any], 
                           fieldnames: List[str]) -> str:
        """Generate CSV string from log entry"""
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="",  # No extra newline
        )
        writer.writerow(log_entry)
        csv_string = output.getvalue()
        output.close()
        return csv_string

    def format(self, record: logging.LogRecord) -> str:
        log_entry = self._build_base_log_entry(record)
        context_items = self._extract_context_items(record)
        log_entry.update(context_items)
        
        all_fieldnames = self.fieldnames + sorted(context_items.keys())
        return self._generate_csv_string(log_entry, all_fieldnames)