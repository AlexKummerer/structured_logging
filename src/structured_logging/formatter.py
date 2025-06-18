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
