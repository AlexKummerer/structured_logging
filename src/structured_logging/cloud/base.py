"""
Base classes for cloud logging handlers
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from queue import Queue, Empty
from threading import Thread, Event
from typing import Any, Dict, List, Optional


@dataclass
class CloudHandlerConfig:
    """Base configuration for cloud handlers"""
    
    # Batch settings
    batch_size: int = 100
    flush_interval: float = 5.0  # seconds
    max_queue_size: int = 10000
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    
    # Performance settings
    compress_logs: bool = True
    async_upload: bool = True
    
    # Credentials (cloud-specific subclasses will add their own)
    region: Optional[str] = None


class CloudLogHandler(logging.Handler, ABC):
    """
    Abstract base class for cloud logging handlers
    
    Provides common functionality for batching, queuing, and uploading logs
    to cloud services with retry logic and error handling.
    """
    
    def __init__(self, config: CloudHandlerConfig):
        super().__init__()
        self.config = config
        self.queue = Queue(maxsize=config.max_queue_size)
        self._stop_event = Event()
        self._worker_thread = None
        
        if config.async_upload:
            self._start_worker()
    
    def _start_worker(self) -> None:
        """Start background worker thread for async uploads"""
        self._worker_thread = Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
    
    def _worker(self) -> None:
        """Background worker that processes log batches"""
        batch = []
        last_flush = time.time()
        
        while not self._stop_event.is_set():
            try:
                # Try to get a log entry with timeout
                timeout = max(0.1, self.config.flush_interval - (time.time() - last_flush))
                
                try:
                    entry = self.queue.get(timeout=timeout)
                    batch.append(entry)
                except Empty:
                    pass
                
                # Check if we should flush
                should_flush = (
                    len(batch) >= self.config.batch_size or
                    time.time() - last_flush >= self.config.flush_interval
                )
                
                if should_flush and batch:
                    self._upload_batch(batch)
                    batch = []
                    last_flush = time.time()
                    
            except Exception as e:
                self.handleError(None)  # Log the error through logging system
    
    def emit(self, record: logging.LogRecord) -> None:
        """Handle a log record"""
        try:
            formatted = self.format(record)
            
            if self.config.async_upload:
                # Add to queue for async processing
                try:
                    self.queue.put_nowait({
                        'timestamp': time.time(),
                        'level': record.levelname,
                        'message': formatted,
                        'record': record
                    })
                except:
                    # Queue full, drop message
                    pass
            else:
                # Synchronous upload
                self._upload_batch([{
                    'timestamp': time.time(),
                    'level': record.levelname,
                    'message': formatted,
                    'record': record
                }])
                
        except Exception:
            self.handleError(record)
    
    def _upload_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Upload a batch of logs with retry logic"""
        if not batch:
            return
            
        for attempt in range(self.config.max_retries):
            try:
                self._upload_logs(batch)
                return  # Success
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    # Calculate delay with exponential backoff
                    delay = self.config.retry_delay
                    if self.config.exponential_backoff:
                        delay *= (2 ** attempt)
                    time.sleep(delay)
                else:
                    # Final attempt failed, log error
                    self.handleError(None)
    
    @abstractmethod
    def _upload_logs(self, batch: List[Dict[str, Any]]) -> None:
        """
        Upload logs to cloud service - must be implemented by subclasses
        
        Args:
            batch: List of log entries to upload
            
        Raises:
            Exception: If upload fails
        """
        pass
    
    def flush(self) -> None:
        """Flush any pending logs"""
        if self.config.async_upload and self._worker_thread:
            # Force flush by signaling worker
            remaining = []
            while not self.queue.empty():
                try:
                    remaining.append(self.queue.get_nowait())
                except Empty:
                    break
            
            if remaining:
                self._upload_batch(remaining)
    
    def close(self) -> None:
        """Close handler and cleanup resources"""
        if self.config.async_upload:
            self._stop_event.set()
            if self._worker_thread:
                self._worker_thread.join(timeout=5)
        
        self.flush()
        super().close()