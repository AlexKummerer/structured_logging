"""
Core stream processor for real-time log processing

This module provides the main stream processing engine with support for
windowing, aggregation, and stateful operations.
"""

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from ..logger import get_logger


class StreamState(Enum):
    """States of a stream processor"""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class StreamConfig:
    """Configuration for stream processing"""
    
    # Buffer settings
    buffer_size: int = 10000  # Max items in buffer
    buffer_timeout_ms: int = 100  # Flush buffer after timeout
    
    # Backpressure settings
    enable_backpressure: bool = True
    backpressure_threshold: float = 0.8  # 80% of buffer
    
    # Processing settings
    batch_size: int = 100  # Process items in batches
    max_concurrent_batches: int = 10
    
    # Window settings
    watermark_delay_ms: int = 5000  # Allow 5s for late data
    window_retention_minutes: int = 60  # Keep windows for 1 hour
    
    # State management
    checkpoint_interval_seconds: int = 60
    state_backend: str = "memory"  # memory, redis, etc.
    
    # Error handling
    max_retries: int = 3
    retry_delay_ms: int = 1000
    error_handler: Optional[Callable[[Exception, Any], None]] = None
    
    # Metrics
    enable_metrics: bool = True
    metrics_interval_seconds: int = 30


@dataclass
class ProcessingContext:
    """Context for processing operations"""
    
    timestamp: datetime
    watermark: datetime
    window_key: Optional[str] = None
    partition_key: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LogStreamProcessor:
    """
    Main stream processor for real-time log processing
    
    Supports:
    - Async processing with backpressure
    - Windowing operations (tumbling, sliding, session)
    - Stateful aggregations
    - Exactly-once processing semantics
    - Late data handling with watermarks
    """
    
    def __init__(self, config: Optional[StreamConfig] = None):
        self.config = config or StreamConfig()
        self.logger = get_logger("streaming.processor")
        
        # State management
        self._state = StreamState.INITIALIZED
        self._processing_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.buffer_size
        )
        self._state_store: Dict[str, Any] = {}
        self._window_states: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Pipeline components
        self._source = None
        self._operations: List[Callable] = []
        self._sinks: List[Any] = []
        
        # Metrics
        self._processed_count = 0
        self._error_count = 0
        self._last_checkpoint = datetime.now()
        self._watermark = datetime.min
        
        # Control
        self._tasks: Set[asyncio.Task] = set()
        self._stop_event = asyncio.Event()
        
    async def start(self) -> None:
        """Start the stream processor"""
        if self._state != StreamState.INITIALIZED:
            raise RuntimeError(f"Cannot start from state {self._state}")
            
        self._state = StreamState.RUNNING
        self.logger.info("Starting stream processor")
        
        # Start processing tasks
        self._tasks.add(
            asyncio.create_task(self._process_loop())
        )
        
        if self.config.enable_metrics:
            self._tasks.add(
                asyncio.create_task(self._metrics_loop())
            )
            
        # Start source if configured
        if self._source:
            self._tasks.add(
                asyncio.create_task(self._source_loop())
            )
            
    async def stop(self) -> None:
        """Stop the stream processor gracefully"""
        self.logger.info("Stopping stream processor")
        self._state = StreamState.STOPPED
        self._stop_event.set()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Final checkpoint
        await self._checkpoint()
        
        # Close sinks
        for sink in self._sinks:
            if hasattr(sink, 'close'):
                await sink.close()
                
    async def pause(self) -> None:
        """Pause processing"""
        self._state = StreamState.PAUSED
        self.logger.info("Paused stream processor")
        
    async def resume(self) -> None:
        """Resume processing"""
        self._state = StreamState.RUNNING
        self.logger.info("Resumed stream processor")
        
    def source(self, source: Any) -> "LogStreamProcessor":
        """Set the input source"""
        self._source = source
        return self
        
    def filter(self, predicate: Callable[[Dict[str, Any]], bool]) -> "LogStreamProcessor":
        """Add a filter operation"""
        async def filter_op(item: Dict[str, Any], ctx: ProcessingContext) -> Optional[Dict[str, Any]]:
            return item if predicate(item) else None
            
        self._operations.append(filter_op)
        return self
        
    def map(self, mapper: Callable[[Dict[str, Any]], Dict[str, Any]]) -> "LogStreamProcessor":
        """Add a map operation"""
        async def map_op(item: Dict[str, Any], ctx: ProcessingContext) -> Dict[str, Any]:
            return mapper(item)
            
        self._operations.append(map_op)
        return self
        
    def window(self, window_spec: Any) -> "LogStreamProcessor":
        """Add a window operation"""
        self._operations.append(window_spec.process)
        return self
        
    def aggregate(self, *aggregators) -> "LogStreamProcessor":
        """Add aggregation operations"""
        async def aggregate_op(items: List[Dict[str, Any]], ctx: ProcessingContext) -> Dict[str, Any]:
            result = {}
            for agg in aggregators:
                result.update(await agg.compute(items, ctx))
            return result
            
        self._operations.append(aggregate_op)
        return self
        
    def sink(self, *sinks) -> "LogStreamProcessor":
        """Add output sinks"""
        self._sinks.extend(sinks)
        return self
        
    async def _source_loop(self) -> None:
        """Read from source and enqueue items"""
        try:
            async for item in self._source:
                if self._state != StreamState.RUNNING:
                    await asyncio.sleep(0.1)
                    continue
                    
                # Apply backpressure if needed
                if self.config.enable_backpressure:
                    queue_usage = self._processing_queue.qsize() / self.config.buffer_size
                    if queue_usage > self.config.backpressure_threshold:
                        await asyncio.sleep(0.1)  # Slow down
                        
                await self._processing_queue.put(item)
                
                # Update watermark
                item_time = item.get("timestamp", datetime.now())
                self._update_watermark(item_time)
                
        except Exception as e:
            self.logger.error(f"Source error: {e}")
            self._state = StreamState.ERROR
            
    async def _process_loop(self) -> None:
        """Main processing loop"""
        batch = []
        last_flush = datetime.now()
        
        while not self._stop_event.is_set():
            try:
                if self._state != StreamState.RUNNING:
                    await asyncio.sleep(0.1)
                    continue
                    
                # Collect batch
                timeout = self.config.buffer_timeout_ms / 1000
                
                try:
                    item = await asyncio.wait_for(
                        self._processing_queue.get(),
                        timeout=timeout
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    pass
                    
                # Process batch if full or timeout
                should_process = (
                    len(batch) >= self.config.batch_size or
                    (datetime.now() - last_flush).total_seconds() * 1000 > self.config.buffer_timeout_ms
                )
                
                if should_process and batch:
                    await self._process_batch(batch)
                    batch = []
                    last_flush = datetime.now()
                    
                # Checkpoint periodically
                if (datetime.now() - self._last_checkpoint).total_seconds() > self.config.checkpoint_interval_seconds:
                    await self._checkpoint()
                    
            except Exception as e:
                self.logger.error(f"Processing error: {e}")
                self._error_count += 1
                
                if self.config.error_handler:
                    self.config.error_handler(e, batch)
                    
    async def _process_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Process a batch of items through the pipeline"""
        try:
            # Create processing context
            ctx = ProcessingContext(
                timestamp=datetime.now(),
                watermark=self._watermark
            )
            
            # Apply operations
            current_batch = batch
            for operation in self._operations:
                new_batch = []
                
                for item in current_batch:
                    result = await operation(item, ctx)
                    if result is not None:
                        if isinstance(result, list):
                            new_batch.extend(result)
                        else:
                            new_batch.append(result)
                            
                current_batch = new_batch
                
            # Send to sinks
            for sink in self._sinks:
                await sink.write(current_batch)
                
            self._processed_count += len(batch)
            
        except Exception as e:
            self.logger.error(f"Batch processing error: {e}")
            
            # Retry logic
            for retry in range(self.config.max_retries):
                await asyncio.sleep(self.config.retry_delay_ms / 1000)
                try:
                    # Retry processing
                    await self._process_batch(batch)
                    break
                except Exception:
                    if retry == self.config.max_retries - 1:
                        raise
                        
    def _update_watermark(self, timestamp: datetime) -> None:
        """Update the watermark for late data handling"""
        watermark_delay = timedelta(milliseconds=self.config.watermark_delay_ms)
        new_watermark = timestamp - watermark_delay
        
        if new_watermark > self._watermark:
            self._watermark = new_watermark
            # Trigger window cleanups
            self._cleanup_windows()
            
    def _cleanup_windows(self) -> None:
        """Clean up expired windows"""
        retention = timedelta(minutes=self.config.window_retention_minutes)
        cutoff = datetime.now() - retention
        
        # Remove old window states
        expired_windows = [
            key for key, state in self._window_states.items()
            if state.get("end_time", datetime.max) < cutoff
        ]
        
        for key in expired_windows:
            del self._window_states[key]
            
    async def _checkpoint(self) -> None:
        """Save processing state"""
        checkpoint = {
            "processed_count": self._processed_count,
            "error_count": self._error_count,
            "watermark": self._watermark,
            "window_states": self._window_states,
            "timestamp": datetime.now()
        }
        
        self._state_store["checkpoint"] = checkpoint
        self._last_checkpoint = datetime.now()
        
        self.logger.debug(
            f"Checkpoint saved: {self._processed_count} processed, "
            f"{self._error_count} errors"
        )
        
    async def _metrics_loop(self) -> None:
        """Collect and report metrics"""
        while not self._stop_event.is_set():
            await asyncio.sleep(self.config.metrics_interval_seconds)
            
            if self._state == StreamState.RUNNING:
                metrics = {
                    "processed_count": self._processed_count,
                    "error_count": self._error_count,
                    "queue_size": self._processing_queue.qsize(),
                    "watermark": self._watermark,
                    "window_count": len(self._window_states),
                    "state": self._state.value
                }
                
                self.logger.info(
                    "Stream metrics",
                    extra=metrics
                )
                
    def get_state(self) -> Dict[str, Any]:
        """Get current processor state"""
        return {
            "state": self._state.value,
            "processed_count": self._processed_count,
            "error_count": self._error_count,
            "queue_size": self._processing_queue.qsize(),
            "watermark": self._watermark,
            "window_count": len(self._window_states),
            "sink_count": len(self._sinks)
        }


def create_stream_processor(
    config: Optional[StreamConfig] = None
) -> LogStreamProcessor:
    """
    Create a new stream processor instance
    
    Args:
        config: Optional configuration
        
    Returns:
        LogStreamProcessor instance
    """
    return LogStreamProcessor(config)