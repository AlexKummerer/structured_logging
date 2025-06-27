"""
Utility functions for logging handlers
"""

import logging
from typing import Optional

from .config import FileHandlerConfig
from .rotating_handler import RotatingFileHandler
from .timed_handler import TimedRotatingFileHandler


def create_file_logger(
    name: str,
    config: FileHandlerConfig,
    formatter: Optional[logging.Formatter] = None,
    handler_type: str = "rotating",
) -> logging.Logger:
    """
    Create a logger with file handler

    Args:
        name: Logger name
        config: File handler configuration
        formatter: Optional custom formatter
        handler_type: Type of handler ('rotating' or 'timed')

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create appropriate handler
    if handler_type == "timed":
        handler = TimedRotatingFileHandler(config)
    else:
        handler = RotatingFileHandler(config)

    # Set formatter
    if formatter:
        handler.setFormatter(formatter)
    else:
        # Use simple default formatter
        simple_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(simple_formatter)

    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    return logger