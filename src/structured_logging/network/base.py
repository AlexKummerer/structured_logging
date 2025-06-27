"""
Base class for network logging handlers
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from typing import Any, Dict, List

from .config import NetworkHandlerConfig


class BaseNetworkHandler(logging.Handler):
    """Base class for all network handlers"""

    def __init__(self, config: NetworkHandlerConfig):
        super().__init__()
        self.config = config
        self.buffer = Queue(maxsize=1000)
        self.batch = []
        self.last_flush = time.time()
        self.executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="network-handler"
        )
        self.running = True
        self.lock = threading.Lock()

        # Start background flush thread
        self.flush_thread = threading.Thread(
            target=self._flush_worker, daemon=True, name="network-flush"
        )
        self.flush_thread.start()

        # Local fallback handler
        self.fallback_handler = None
        if config.fallback_to_local and config.local_fallback_file:
            self.fallback_handler = logging.FileHandler(config.local_fallback_file)
            self.fallback_handler.setFormatter(self.formatter)

    def emit(self, record: logging.LogRecord) -> None:
        """Add log record to buffer for network transmission"""
        if not self.running:
            return

        try:
            # Format the record
            message = self.format(record)

            # Add to buffer (non-blocking)
            try:
                self.buffer.put_nowait(
                    {"message": message, "record": record, "timestamp": time.time()}
                )
            except:
                # Buffer full - use fallback or drop
                if self.fallback_handler:
                    self.fallback_handler.emit(record)
                # Otherwise drop the message

        except Exception:
            self.handleError(record)

    def _flush_worker(self) -> None:
        """Background worker that flushes messages"""
        while self.running:
            try:
                current_time = time.time()
                should_flush = False

                # Collect messages from buffer
                messages = []
                try:
                    while len(messages) < self.config.batch_size:
                        item = self.buffer.get(timeout=0.1)
                        messages.append(item)
                except Empty:
                    pass

                # Check if we should flush
                if messages:
                    if (
                        len(messages) >= self.config.batch_size
                        or current_time - self.last_flush >= self.config.flush_interval
                    ):
                        should_flush = True

                if should_flush and messages:
                    self.executor.submit(self._send_batch, messages)
                    self.last_flush = current_time
                elif messages:
                    # Put messages back in buffer
                    for msg in messages:
                        try:
                            self.buffer.put_nowait(msg)
                        except:
                            pass  # Buffer full, drop message

                time.sleep(0.1)  # Small delay to prevent busy waiting

            except Exception:
                pass  # Continue running even if flush fails

    def _send_batch(self, messages: List[Dict[str, Any]]) -> None:
        """Send a batch of messages - implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _send_batch")

    def _send_with_retry(self, send_func, *args, **kwargs) -> bool:
        """Execute send function with retry logic"""
        for attempt in range(self.config.max_retries + 1):
            try:
                send_func(*args, **kwargs)
                return True
            except Exception:
                if attempt < self.config.max_retries:
                    time.sleep(
                        self.config.retry_delay * (2**attempt)
                    )  # Exponential backoff
                else:
                    # All retries failed - use fallback if available
                    if self.fallback_handler and "record" in kwargs:
                        self.fallback_handler.emit(kwargs["record"])
                    return False
        return False

    def flush(self) -> None:
        """Flush any buffered messages"""
        # Flush remaining messages in buffer
        messages = []
        try:
            while True:
                item = self.buffer.get_nowait()
                messages.append(item)
        except Empty:
            pass

        if messages:
            self._send_batch(messages)

    def close(self) -> None:
        """Close the handler and clean up resources"""
        self.running = False

        # Flush remaining messages
        self.flush()

        # Wait for flush thread to finish
        if self.flush_thread.is_alive():
            self.flush_thread.join(timeout=5.0)

        # Shutdown executor
        self.executor.shutdown(wait=True)

        # Close fallback handler
        if self.fallback_handler:
            self.fallback_handler.close()

        super().close()