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
from .serializers import SerializationConfig, serialize_for_logging


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

    def __init__(
        self,
        async_config: AsyncLoggerConfig,
        logger_config: LoggerConfig,
        serialization_config: Optional[SerializationConfig] = None,
    ):
        self.async_config = async_config
        self.logger_config = logger_config
        self.serialization_config = serialization_config or SerializationConfig()
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=async_config.queue_size)
        self.batch: List[AsyncLogEntry] = []
        self.last_flush_time = time.time()
        self.running = False
        self.workers: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

        # Create formatter with serialization config
        if logger_config.formatter_type == "csv":
            self.formatter: Union[
                CSVFormatter, PlainTextFormatter, StructuredFormatter
            ] = CSVFormatter(logger_config, self.serialization_config)
        elif logger_config.formatter_type == "plain":
            self.formatter = PlainTextFormatter(
                logger_config, self.serialization_config
            )
        else:
            self.formatter = StructuredFormatter(
                logger_config, self.serialization_config
            )

        # Output stream (could be enhanced to support file/network handlers)
        self.stream = sys.stdout
        self._stats = {
            "processed_entries": 0,
            "dropped_entries": 0,
            "errors": 0,
            "batch_flushes": 0,
        }

    async def start(self) -> None:
        """Start the async log processor"""
        if self.running:
            return

        self.running = True
        self._shutdown_event.clear()

        # Start worker tasks
        for i in range(self.async_config.max_workers):
            worker = asyncio.create_task(
                self._worker(f"worker-{i}"), name=f"async-log-worker-{i}"
            )
            self.workers.append(worker)

        # Start flush timer
        flush_task = asyncio.create_task(
            self._flush_timer(), name="async-log-flush-timer"
        )
        self.workers.append(flush_task)

    def _cancel_active_workers(self) -> None:
        """Cancel all active worker tasks"""
        for worker in self.workers:
            if not worker.done():
                worker.cancel()

    async def _wait_for_workers(self, timeout: float) -> None:
        """Wait for all workers to finish with timeout"""
        try:
            await asyncio.wait_for(
                asyncio.gather(*self.workers, return_exceptions=True), 
                timeout=timeout
            )
        except asyncio.TimeoutError:
            pass
        finally:
            self.workers.clear()

    def _final_stream_flush(self) -> None:
        """Perform final flush of output stream"""
        if hasattr(self.stream, "flush"):
            self.stream.flush()

    async def stop(self, timeout: Optional[float] = None) -> None:
        """Stop the async log processor"""
        if not self.running:
            return

        self.running = False
        self._shutdown_event.set()
        
        await self._drain_queue()
        await self._flush_batch()
        
        self._cancel_active_workers()
        
        if timeout is None:
            timeout = self.async_config.shutdown_timeout
            
        await self._wait_for_workers(timeout)
        self._final_stream_flush()

    def _handle_queue_overflow(self) -> bool:
        """Handle queue overflow when dropping is enabled"""
        self._stats["dropped_entries"] += 1
        if self.async_config.error_callback:
            self.async_config.error_callback(
                Exception("Log queue overflow, dropping log entry")
            )
        return False

    async def _enqueue_with_drop(self, entry: AsyncLogEntry) -> bool:
        """Try to enqueue with drop-on-overflow policy"""
        try:
            self.queue.put_nowait(entry)
            return True
        except asyncio.QueueFull:
            return self._handle_queue_overflow()

    async def _enqueue_with_timeout(self, entry: AsyncLogEntry) -> bool:
        """Try to enqueue with timeout"""
        await asyncio.wait_for(
            self.queue.put(entry), timeout=self.async_config.queue_timeout
        )
        return True

    async def enqueue_log(self, entry: AsyncLogEntry) -> bool:
        """Enqueue a log entry for processing"""
        if not self.running:
            await self.start()

        try:
            if self.async_config.drop_on_overflow:
                return await self._enqueue_with_drop(entry)
            else:
                return await self._enqueue_with_timeout(entry)
        except asyncio.TimeoutError:
            if self.async_config.error_callback:
                self.async_config.error_callback(Exception("Log queue timeout"))
            return False
        except Exception as e:
            if self.async_config.error_callback:
                self.async_config.error_callback(e)
            return False

    def _should_flush_batch(self) -> bool:
        """Check if batch should be flushed"""
        return (
            len(self.batch) >= self.async_config.batch_size
            or time.time() - self.last_flush_time >= self.async_config.flush_interval
        )

    async def _process_queue_entry(self) -> bool:
        """Process a single queue entry, return True if entry was processed"""
        try:
            entry = await asyncio.wait_for(
                self.queue.get(), timeout=0.1
            )
            self.batch.append(entry)
            
            if self._should_flush_batch():
                await self._flush_batch()
            
            self.queue.task_done()
            return True
        except asyncio.TimeoutError:
            return False

    async def _handle_timeout_flush(self) -> None:
        """Handle flush on timeout"""
        if self.batch and self._should_flush_batch():
            await self._flush_batch()

    async def _handle_worker_error(self, e: Exception) -> None:
        """Handle worker errors"""
        self._stats["errors"] += 1
        if self.async_config.error_callback:
            self.async_config.error_callback(e)

    async def _handle_cancellation(self) -> None:
        """Handle worker cancellation"""
        if self.batch:
            try:
                await self._flush_batch()
            except Exception as e:
                if self.async_config.error_callback:
                    self.async_config.error_callback(e)

    async def _worker(self, worker_name: str) -> None:
        """Worker task that processes log entries"""
        try:
            while self.running:
                try:
                    if not await self._process_queue_entry():
                        await self._handle_timeout_flush()
                except Exception as e:
                    await self._handle_worker_error(e)
        except asyncio.CancelledError:
            await self._handle_cancellation()
            raise

    async def _flush_timer(self) -> None:
        """Timer task that ensures regular flushing"""
        try:
            while self.running:
                await asyncio.sleep(self.async_config.flush_interval)
                if self.batch:
                    await self._flush_batch()
        except asyncio.CancelledError:
            raise

    async def _drain_queue(self) -> None:
        """Drain remaining items from queue during shutdown"""
        try:
            while not self.queue.empty():
                try:
                    entry = self.queue.get_nowait()
                    self.batch.append(entry)
                    self.queue.task_done()
                except asyncio.QueueEmpty:
                    break
        except Exception as e:
            if self.async_config.error_callback:
                self.async_config.error_callback(e)

    async def _process_batch_entries(self, batch: List[AsyncLogEntry]) -> None:
        """Process all entries in a batch"""
        for entry in batch:
            await self._process_log_entry(entry)
            self._stats["processed_entries"] += 1

    def _update_flush_stats(self) -> None:
        """Update statistics after flush"""
        self.last_flush_time = time.time()
        self._stats["batch_flushes"] += 1
        if hasattr(self.stream, "flush"):
            self.stream.flush()

    def _handle_batch_error(self, e: Exception, batch: List[AsyncLogEntry]) -> None:
        """Handle errors during batch processing"""
        self._stats["errors"] += 1
        self.batch.extend(batch)  # Put entries back
        if self.async_config.error_callback:
            self.async_config.error_callback(e)

    async def _flush_batch(self) -> None:
        """Flush the current batch of log entries"""
        if not self.batch:
            return

        batch_to_process = self.batch.copy()
        self.batch.clear()

        try:
            await self._process_batch_entries(batch_to_process)
            self._update_flush_stats()
        except Exception as e:
            self._handle_batch_error(e, batch_to_process)

    def _create_log_record(self, entry: AsyncLogEntry) -> logging.LogRecord:
        """Create LogRecord from AsyncLogEntry"""
        record = logging.LogRecord(
            name=entry.logger_name,
            level=getattr(logging, entry.level.upper()),
            pathname="",
            lineno=0,
            msg=entry.message,
            args=(),
            exc_info=None,
        )
        record.created = entry.timestamp
        return record

    def _add_context_to_record(self, record: logging.LogRecord, entry: AsyncLogEntry) -> None:
        """Add context data to log record"""
        for key, value in entry.context.items():
            setattr(record, key, value)

    def _serialize_extra_data(self, record: logging.LogRecord, entry: AsyncLogEntry) -> None:
        """Serialize and add extra data to record"""
        for key, value in entry.extra.items():
            if value is not None:
                try:
                    serialized_value = serialize_for_logging(
                        value, self.serialization_config
                    )
                    setattr(record, f"ctx_{key}", serialized_value)
                except Exception as e:
                    setattr(record, f"ctx_{key}", str(value))
                    if self.async_config.error_callback:
                        self.async_config.error_callback(
                            Exception(f"Failed to serialize {key}: {e}")
                        )

    async def _process_log_entry(self, entry: AsyncLogEntry) -> None:
        """Process a single log entry with enhanced serialization"""
        record = self._create_log_record(entry)
        self._add_context_to_record(record, entry)
        self._serialize_extra_data(record, entry)
        
        formatted_message = self.formatter.format(record)
        self.stream.write(formatted_message + "\n")

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        return {
            **self._stats,
            "queue_size": self.queue.qsize(),
            "batch_size": len(self.batch),
            "running": self.running,
            "worker_count": len(self.workers),
        }


class AsyncLogger:
    """Async logger with non-blocking operations and enhanced serialization"""

    def __init__(
        self,
        name: str,
        logger_config: Optional[LoggerConfig] = None,
        async_config: Optional[AsyncLoggerConfig] = None,
        serialization_config: Optional[SerializationConfig] = None,
    ):
        self.name = name
        self.logger_config = logger_config or get_default_config()
        self.async_config = async_config or get_default_async_config()
        self.serialization_config = serialization_config or SerializationConfig()

        # Create processor with serialization config
        self.processor = AsyncLogProcessor(
            self.async_config, self.logger_config, self.serialization_config
        )

        # Minimum log level
        self.level = getattr(logging, self.logger_config.log_level.upper())

        # Track if started
        self._started = False

    def _should_log(self, level: str) -> bool:
        """Check if message should be logged based on level"""
        numeric_level = getattr(logging, level.upper())
        return numeric_level >= self.level

    async def _add_request_id_context(self, context: Dict[str, Any]) -> None:
        """Add request ID to context if configured"""
        if self.logger_config.include_request_id:
            request_id = await aget_request_id()
            if request_id:
                context["ctx_request_id"] = request_id

    async def _add_user_context(self, context: Dict[str, Any]) -> None:
        """Add user context fields if configured"""
        if self.logger_config.include_user_context:
            user_context = await aget_user_context()
            if user_context:
                for key, value in user_context.items():
                    if value is not None:
                        context[f"ctx_{key}"] = value

    async def _add_custom_context(self, context: Dict[str, Any]) -> None:
        """Add custom context fields"""
        custom_context = await aget_custom_context()
        if custom_context:
            for key, value in custom_context.items():
                if value is not None:
                    context[f"ctx_{key}"] = value

    async def _gather_context(self) -> Dict[str, Any]:
        """Gather all context fields"""
        context = {}
        await self._add_request_id_context(context)
        await self._add_user_context(context)
        await self._add_custom_context(context)
        return context

    async def _alog(self, level: str, message: str, **extra: Any) -> bool:
        """Internal async logging method"""
        if not self._should_log(level):
            return True

        context = await self._gather_context()
        
        entry = AsyncLogEntry(
            level=level,
            message=message,
            logger_name=self.name,
            timestamp=time.time(),
            context=context,
            extra={k: v for k, v in extra.items() if v is not None},
        )
        
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

    async def start(self) -> None:
        """Start the async logger"""
        await self.processor.start()

    async def stop(self, timeout: Optional[float] = None) -> None:
        """Stop the async logger"""
        await self.processor.stop(timeout)

    async def flush(self) -> None:
        """Flush all pending logs"""
        await self.processor._flush_batch()

    def get_stats(self) -> Dict[str, Any]:
        """Get logger statistics"""
        return self.processor.get_stats()

    async def __aenter__(self) -> "AsyncLogger":
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit"""
        await self.stop()


# Global async logger registry
_async_loggers: Dict[str, AsyncLogger] = {}


def get_async_logger(
    name: str,
    logger_config: Optional[LoggerConfig] = None,
    async_config: Optional[AsyncLoggerConfig] = None,
    serialization_config: Optional[SerializationConfig] = None,
) -> AsyncLogger:
    """Get or create an async logger with enhanced serialization support"""
    # Use config-based key for caching
    config_key = (
        f"{name}_{id(logger_config)}_{id(async_config)}_{id(serialization_config)}"
    )

    if config_key not in _async_loggers:
        _async_loggers[config_key] = AsyncLogger(
            name, logger_config, async_config, serialization_config
        )

    return _async_loggers[config_key]


async def alog_with_context(
    logger: AsyncLogger, level: str, message: str, **extra: Any
) -> bool:
    """Async version of log_with_context"""
    return await logger._alog(level, message, **extra)


async def shutdown_all_async_loggers(timeout: Optional[float] = None) -> None:
    """Shutdown all async loggers"""
    shutdown_tasks = []
    for logger in _async_loggers.values():
        shutdown_tasks.append(logger.stop(timeout))

    if shutdown_tasks:
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)

    _async_loggers.clear()
