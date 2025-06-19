"""
Async logging implementation for high-performance non-blocking logging
"""

import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from .async_config import AsyncLoggerConfig, get_default_async_config
from .async_context import aget_custom_context, aget_request_id, aget_user_context
from .config import LoggerConfig, get_default_config
from .formatter import CSVFormatter, PlainTextFormatter, StructuredFormatter


@dataclass
class AsyncLogEntry:
    """Represents a log entry in the async queue"""

    level: str
    message: str
    logger_name: str
    timestamp: float
    context: Dict[str, Any]
    extra: Dict[str, Any]


class AsyncLogProcessor:
    """Background processor for async log entries"""

    def __init__(self, async_config: AsyncLoggerConfig, logger_config: LoggerConfig):
        self.async_config = async_config
        self.logger_config = logger_config
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=async_config.queue_size)
        self.batch: List[AsyncLogEntry] = []
        self.last_flush_time = time.time()
        self.running = False
        self.workers: List[asyncio.Task] = []

        # Create formatter
        if logger_config.formatter_type == "csv":
            self.formatter = CSVFormatter(logger_config)
        elif logger_config.formatter_type == "plain":
            self.formatter = PlainTextFormatter(logger_config)
        else:
            self.formatter = StructuredFormatter(logger_config)

        # Output stream
        self.stream = sys.stdout

    async def start(self):
        """Start the async log processor"""
        if self.running:
            return

        self.running = True

        # Start worker tasks
        for i in range(self.async_config.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)

        # Start flush timer
        flush_task = asyncio.create_task(self._flush_timer())
        self.workers.append(flush_task)

    async def stop(self, timeout: Optional[float] = None):
        """Stop the async log processor"""
        if not self.running:
            return

        self.running = False

        # Flush remaining logs
        await self._flush_batch()

        # Cancel workers
        for worker in self.workers:
            worker.cancel()

        # Wait for workers to finish
        if timeout is None:
            timeout = self.async_config.shutdown_timeout

        try:
            await asyncio.wait_for(
                asyncio.gather(*self.workers, return_exceptions=True), timeout=timeout
            )
        except asyncio.TimeoutError:
            # Force cancellation if timeout exceeded
            pass

        self.workers.clear()

    async def enqueue_log(self, entry: AsyncLogEntry) -> bool:
        """Enqueue a log entry for processing"""
        if not self.running:
            await self.start()

        try:
            # Try to put entry in queue
            if self.async_config.drop_on_overflow:
                # Non-blocking put, drop if queue is full
                try:
                    self.queue.put_nowait(entry)
                    return True
                except asyncio.QueueFull:
                    if self.async_config.error_callback:
                        self.async_config.error_callback(
                            Exception("Log queue overflow, dropping log entry")
                        )
                    return False
            else:
                # Blocking put with timeout
                await asyncio.wait_for(
                    self.queue.put(entry), timeout=self.async_config.queue_timeout
                )
                return True

        except asyncio.TimeoutError:
            if self.async_config.error_callback:
                self.async_config.error_callback(Exception("Log queue timeout"))
            return False
        except Exception as e:
            if self.async_config.error_callback:
                self.async_config.error_callback(e)
            return False

    async def _worker(self, worker_name: str):
        """Worker task that processes log entries"""
        try:
            while self.running:
                try:
                    # Get log entry from queue
                    entry = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=0.1,  # Short timeout to check running status
                    )

                    # Add to batch
                    self.batch.append(entry)

                    # Check if we should flush
                    should_flush = (
                        len(self.batch) >= self.async_config.batch_size
                        or time.time() - self.last_flush_time
                        >= self.async_config.flush_interval
                    )

                    if should_flush:
                        await self._flush_batch()

                    # Mark task as done
                    self.queue.task_done()

                except asyncio.TimeoutError:
                    # Check if we should flush based on time
                    if (
                        self.batch
                        and time.time() - self.last_flush_time
                        >= self.async_config.flush_interval
                    ):
                        await self._flush_batch()

                except Exception as e:
                    if self.async_config.error_callback:
                        self.async_config.error_callback(e)

        except asyncio.CancelledError:
            # Worker was cancelled, flush remaining logs
            if self.batch:
                await self._flush_batch()
            raise

    async def _flush_timer(self):
        """Timer task that ensures regular flushing"""
        try:
            while self.running:
                await asyncio.sleep(self.async_config.flush_interval)
                if self.batch:
                    await self._flush_batch()
        except asyncio.CancelledError:
            raise

    async def _flush_batch(self):
        """Flush the current batch of log entries"""
        if not self.batch:
            return

        try:
            # Process all entries in batch
            for entry in self.batch:
                await self._process_log_entry(entry)

            # Clear batch and update flush time
            self.batch.clear()
            self.last_flush_time = time.time()

            # Flush output stream
            if hasattr(self.stream, "flush"):
                self.stream.flush()

        except Exception as e:
            if self.async_config.error_callback:
                self.async_config.error_callback(e)

    async def _process_log_entry(self, entry: AsyncLogEntry):
        """Process a single log entry"""
        # Create LogRecord for formatter
        record = logging.LogRecord(
            name=entry.logger_name,
            level=getattr(logging, entry.level.upper()),
            pathname="",
            lineno=0,
            msg=entry.message,
            args=(),
            exc_info=None,
        )

        # Add context to record
        for key, value in entry.context.items():
            setattr(record, key, value)

        for key, value in entry.extra.items():
            setattr(record, f"ctx_{key}", value)

        # Format and write
        formatted_message = self.formatter.format(record)
        self.stream.write(formatted_message + "\n")


class AsyncLogger:
    """Async logger that provides non-blocking logging operations"""

    def __init__(
        self,
        name: str,
        logger_config: Optional[LoggerConfig] = None,
        async_config: Optional[AsyncLoggerConfig] = None,
    ):
        self.name = name
        self.logger_config = logger_config or get_default_config()
        self.async_config = async_config or get_default_async_config()

        # Create processor
        self.processor = AsyncLogProcessor(self.async_config, self.logger_config)

        # Minimum log level
        self.level = getattr(logging, self.logger_config.log_level.upper())

    async def _alog(self, level: str, message: str, **extra: Any) -> bool:
        """Internal async logging method"""
        # Check log level
        numeric_level = getattr(logging, level.upper())
        if numeric_level < self.level:
            return True  # Log was "successful" but filtered

        # Gather context
        context = {}

        if self.logger_config.include_request_id:
            request_id = await aget_request_id()
            if request_id:
                context["ctx_request_id"] = request_id

        if self.logger_config.include_user_context:
            user_context = await aget_user_context()
            if user_context:
                for key, value in user_context.items():
                    if value is not None:
                        context[f"ctx_{key}"] = value

        # Custom context
        custom_context = await aget_custom_context()
        if custom_context:
            for key, value in custom_context.items():
                if value is not None:
                    context[f"ctx_{key}"] = value

        # Create log entry
        entry = AsyncLogEntry(
            level=level,
            message=message,
            logger_name=self.name,
            timestamp=time.time(),
            context=context,
            extra={k: v for k, v in extra.items() if v is not None},
        )

        # Enqueue for processing
        return await self.processor.enqueue_log(entry)

    async def adebug(self, message: str, **extra: Any) -> bool:
        """Async debug logging"""
        return await self._alog("DEBUG", message, **extra)

    async def ainfo(self, message: str, **extra: Any) -> bool:
        """Async info logging"""
        return await self._alog("INFO", message, **extra)

    async def awarning(self, message: str, **extra: Any) -> bool:
        """Async warning logging"""
        return await self._alog("WARNING", message, **extra)

    async def aerror(self, message: str, **extra: Any) -> bool:
        """Async error logging"""
        return await self._alog("ERROR", message, **extra)

    async def acritical(self, message: str, **extra: Any) -> bool:
        """Async critical logging"""
        return await self._alog("CRITICAL", message, **extra)

    async def alog(self, level: Union[str, int], message: str, **extra: Any) -> bool:
        """Async log with specific level"""
        if isinstance(level, int):
            level_name = logging.getLevelName(level)
        else:
            level_name = level.upper()

        return await self._alog(level_name, message, **extra)

    async def start(self):
        """Start the async logger"""
        await self.processor.start()

    async def stop(self, timeout: Optional[float] = None):
        """Stop the async logger"""
        await self.processor.stop(timeout)

    async def flush(self):
        """Flush all pending logs"""
        await self.processor._flush_batch()


# Global async logger registry
_async_loggers: Dict[str, AsyncLogger] = {}


def get_async_logger(
    name: str,
    logger_config: Optional[LoggerConfig] = None,
    async_config: Optional[AsyncLoggerConfig] = None,
) -> AsyncLogger:
    """Get or create an async logger"""
    # Use config-based key for caching
    config_key = f"{name}_{id(logger_config)}_{id(async_config)}"

    if config_key not in _async_loggers:
        _async_loggers[config_key] = AsyncLogger(name, logger_config, async_config)

    return _async_loggers[config_key]


async def alog_with_context(
    logger: AsyncLogger, level: str, message: str, **extra: Any
) -> bool:
    """Async version of log_with_context"""
    return await logger._alog(level, message, **extra)


async def shutdown_all_async_loggers(timeout: Optional[float] = None):
    """Shutdown all async loggers"""
    shutdown_tasks = []
    for logger in _async_loggers.values():
        shutdown_tasks.append(logger.stop(timeout))

    if shutdown_tasks:
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)

    _async_loggers.clear()
