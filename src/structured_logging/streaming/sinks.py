"""
Stream sinks for outputting processed logs

This module provides various output destinations for processed log streams.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import aiohttp

from ..logger import get_logger


class StreamSink(ABC):
    """Base class for stream sinks"""
    
    @abstractmethod
    async def write(self, items: List[Dict[str, Any]]) -> None:
        """Write items to the sink"""
        pass
        
    async def close(self) -> None:
        """Close the sink and cleanup resources"""
        pass


@dataclass
class ConsoleSink(StreamSink):
    """Write stream output to console/stdout"""
    
    format: str = "json"  # json, pretty, compact
    include_timestamp: bool = True
    
    def __post_init__(self):
        self.logger = get_logger("streaming.console_sink")
        
    async def write(self, items: List[Dict[str, Any]]) -> None:
        """Write items to console"""
        for item in items:
            if self.include_timestamp and "timestamp" not in item:
                item["timestamp"] = datetime.now().isoformat()
                
            if self.format == "json":
                print(json.dumps(item, default=str))
            elif self.format == "pretty":
                print(json.dumps(item, indent=2, default=str))
            elif self.format == "compact":
                # Single line summary
                level = item.get("level", "INFO")
                message = item.get("message", "")
                timestamp = item.get("timestamp", "")
                print(f"[{timestamp}] {level}: {message}")
            else:
                print(str(item))


@dataclass
class MetricsSink(StreamSink):
    """
    Export metrics to monitoring systems
    
    Supports Prometheus, StatsD, etc.
    """
    
    backend: str = "prometheus"  # prometheus, statsd, custom
    endpoint: Optional[str] = None
    labels: Dict[str, str] = None
    metric_extractors: Dict[str, Callable[[Dict[str, Any]], float]] = None
    
    def __post_init__(self):
        self.logger = get_logger("streaming.metrics_sink")
        self.labels = self.labels or {}
        self.metric_extractors = self.metric_extractors or {}
        
        # Default metric extractors
        if not self.metric_extractors:
            self.metric_extractors = {
                "log_count": lambda x: x.get("count", 1),
                "error_count": lambda x: 1 if x.get("level") == "error" else 0,
                "response_time": lambda x: x.get("response_time", 0),
            }
            
    async def write(self, items: List[Dict[str, Any]]) -> None:
        """Export metrics from items"""
        metrics = {}
        
        # Extract metrics from items
        for item in items:
            for metric_name, extractor in self.metric_extractors.items():
                try:
                    value = extractor(item)
                    if value is not None:
                        if metric_name not in metrics:
                            metrics[metric_name] = []
                        metrics[metric_name].append(value)
                except Exception as e:
                    self.logger.error(f"Failed to extract metric {metric_name}: {e}")
                    
        # Aggregate and export
        for metric_name, values in metrics.items():
            if values:
                # Simple aggregation (sum for counts, avg for gauges)
                if "count" in metric_name:
                    value = sum(values)
                else:
                    value = sum(values) / len(values)
                    
                await self._export_metric(metric_name, value)
                
    async def _export_metric(self, name: str, value: float) -> None:
        """Export a single metric"""
        if self.backend == "prometheus":
            # In real implementation, would use prometheus_client
            self.logger.debug(f"Prometheus metric: {name}={value} labels={self.labels}")
        elif self.backend == "statsd":
            # In real implementation, would use statsd client
            self.logger.debug(f"StatsD metric: {name}={value}")
        elif self.backend == "custom" and self.endpoint:
            # Send to custom endpoint
            async with aiohttp.ClientSession() as session:
                data = {
                    "metric": name,
                    "value": value,
                    "labels": self.labels,
                    "timestamp": datetime.now().isoformat()
                }
                async with session.post(self.endpoint, json=data) as resp:
                    if resp.status != 200:
                        self.logger.error(f"Failed to export metric: {resp.status}")


@dataclass
class WebSocketSink(StreamSink):
    """
    Stream output to WebSocket clients
    
    Used for real-time dashboards
    """
    
    host: str = "localhost"
    port: int = 8765
    path: str = "/logs"
    
    def __post_init__(self):
        self.logger = get_logger("streaming.websocket_sink")
        self.clients: List[Any] = []
        self.server = None
        self._running = False
        
    async def start_server(self) -> None:
        """Start WebSocket server"""
        try:
            import websockets
        except ImportError:
            self.logger.error("websockets package not installed")
            return
            
        async def handler(websocket, path):
            """Handle WebSocket connections"""
            if path == self.path:
                self.clients.append(websocket)
                try:
                    await websocket.wait_closed()
                finally:
                    self.clients.remove(websocket)
                    
        self.server = await websockets.serve(handler, self.host, self.port)
        self._running = True
        self.logger.info(f"WebSocket server started on ws://{self.host}:{self.port}{self.path}")
        
    async def write(self, items: List[Dict[str, Any]]) -> None:
        """Broadcast items to connected clients"""
        if not self._running:
            await self.start_server()
            
        if not self.clients:
            return
            
        # Broadcast to all connected clients
        message = json.dumps(items, default=str)
        
        # Send to all clients, remove disconnected ones
        disconnected = []
        for client in self.clients:
            try:
                await client.send(message)
            except Exception:
                disconnected.append(client)
                
        for client in disconnected:
            self.clients.remove(client)
            
    async def close(self) -> None:
        """Close WebSocket server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self._running = False


@dataclass
class StorageSink(StreamSink):
    """
    Persist processed logs to storage
    
    Supports files, databases, object storage
    """
    
    storage_type: str = "file"  # file, s3, database
    path: str = "processed_logs"
    format: str = "jsonl"  # jsonl, parquet, csv
    batch_size: int = 1000
    compression: Optional[str] = None  # gzip, zstd
    
    def __post_init__(self):
        self.logger = get_logger("streaming.storage_sink")
        self.buffer: List[Dict[str, Any]] = []
        self.file_counter = 0
        
    async def write(self, items: List[Dict[str, Any]]) -> None:
        """Write items to storage"""
        self.buffer.extend(items)
        
        # Flush if buffer is full
        if len(self.buffer) >= self.batch_size:
            await self._flush()
            
    async def _flush(self) -> None:
        """Flush buffer to storage"""
        if not self.buffer:
            return
            
        if self.storage_type == "file":
            await self._write_to_file()
        elif self.storage_type == "s3":
            await self._write_to_s3()
        elif self.storage_type == "database":
            await self._write_to_database()
            
        self.buffer.clear()
        
    async def _write_to_file(self) -> None:
        """Write to local file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.path}/logs_{timestamp}_{self.file_counter}.{self.format}"
        
        if self.compression:
            filename += f".{self.compression}"
            
        self.file_counter += 1
        
        # Ensure directory exists
        import os
        os.makedirs(self.path, exist_ok=True)
        
        # Write based on format
        if self.format == "jsonl":
            content = "\n".join(json.dumps(item, default=str) for item in self.buffer)
            
            if self.compression == "gzip":
                import gzip
                with gzip.open(filename, "wt") as f:
                    f.write(content)
            else:
                with open(filename, "w") as f:
                    f.write(content)
                    
        self.logger.info(f"Wrote {len(self.buffer)} items to {filename}")
        
    async def _write_to_s3(self) -> None:
        """Write to S3 (placeholder)"""
        self.logger.info(f"Would write {len(self.buffer)} items to S3")
        
    async def _write_to_database(self) -> None:
        """Write to database (placeholder)"""
        self.logger.info(f"Would write {len(self.buffer)} items to database")
        
    async def close(self) -> None:
        """Flush remaining items"""
        await self._flush()


@dataclass
class CompositeSink(StreamSink):
    """Sink that writes to multiple sinks"""
    
    sinks: List[StreamSink]
    
    async def write(self, items: List[Dict[str, Any]]) -> None:
        """Write to all sinks"""
        await asyncio.gather(
            *[sink.write(items) for sink in self.sinks],
            return_exceptions=True
        )
        
    async def close(self) -> None:
        """Close all sinks"""
        await asyncio.gather(
            *[sink.close() for sink in self.sinks],
            return_exceptions=True
        )


# Factory function

def create_sink(sink_type: str, **kwargs) -> StreamSink:
    """
    Create a sink instance
    
    Args:
        sink_type: Type of sink (console, metrics, websocket, storage)
        **kwargs: Sink-specific configuration
        
    Returns:
        StreamSink instance
    """
    if sink_type == "console":
        return ConsoleSink(**kwargs)
    elif sink_type == "metrics":
        return MetricsSink(**kwargs)
    elif sink_type == "websocket":
        return WebSocketSink(**kwargs)
    elif sink_type == "storage":
        return StorageSink(**kwargs)
    else:
        raise ValueError(f"Unknown sink type: {sink_type}")