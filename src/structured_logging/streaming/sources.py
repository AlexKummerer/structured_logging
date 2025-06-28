"""
Stream sources for log input

This module provides various input sources for log streaming.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import aiofiles

from ..logger import get_logger


class StreamSource(ABC):
    """Base class for stream sources"""
    
    @abstractmethod
    def __aiter__(self) -> AsyncIterator[Dict[str, Any]]:
        """Async iterator for reading items"""
        pass


@dataclass
class LogFileSource(StreamSource):
    """
    Read logs from files
    
    Supports tailing, rotation detection, and multiple files
    """
    
    paths: Union[str, List[str]]
    format: str = "json"  # json, jsonl, plain
    tail: bool = True  # Follow file like tail -f
    from_beginning: bool = False  # Start from beginning
    poll_interval: float = 0.1  # Seconds between polls
    
    def __post_init__(self):
        self.logger = get_logger("streaming.file_source")
        if isinstance(self.paths, str):
            self.paths = [self.paths]
        self._file_positions = {}
        self._running = True
        
    async def __aiter__(self) -> AsyncIterator[Dict[str, Any]]:
        """Read logs from files"""
        # Start tasks for each file
        tasks = []
        queues = []
        
        for path in self.paths:
            queue = asyncio.Queue()
            queues.append(queue)
            task = asyncio.create_task(self._read_file(path, queue))
            tasks.append(task)
            
        try:
            # Merge all queues
            while self._running:
                # Check all queues for items
                for queue in queues:
                    try:
                        item = queue.get_nowait()
                        yield item
                    except asyncio.QueueEmpty:
                        pass
                        
                await asyncio.sleep(0.01)  # Small delay to prevent busy loop
                
        finally:
            # Cancel all tasks
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            
    async def _read_file(self, path: str, queue: asyncio.Queue) -> None:
        """Read a single file and push to queue"""
        file_path = Path(path)
        
        if not file_path.exists():
            self.logger.error(f"File not found: {path}")
            return
            
        # Get initial position
        position = 0
        if not self.from_beginning and path in self._file_positions:
            position = self._file_positions[path]
        elif not self.from_beginning:
            # Start at end of file
            position = file_path.stat().st_size
            
        last_inode = file_path.stat().st_ino
        
        while self._running:
            try:
                # Check for file rotation
                current_inode = file_path.stat().st_ino
                if current_inode != last_inode:
                    self.logger.info(f"File rotated: {path}")
                    position = 0
                    last_inode = current_inode
                    
                # Read new lines
                async with aiofiles.open(path, 'r') as f:
                    await f.seek(position)
                    
                    while True:
                        line = await f.readline()
                        if not line:
                            break
                            
                        # Parse line based on format
                        item = self._parse_line(line.strip())
                        if item:
                            await queue.put(item)
                            
                    # Update position
                    position = await f.tell()
                    self._file_positions[path] = position
                    
                if not self.tail:
                    break
                    
                await asyncio.sleep(self.poll_interval)
                
            except Exception as e:
                self.logger.error(f"Error reading file {path}: {e}")
                await asyncio.sleep(1)  # Back off on error
                
    def _parse_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a log line based on format"""
        if not line:
            return None
            
        if self.format == "json" or self.format == "jsonl":
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse JSON: {line}")
                return {"message": line, "parse_error": True}
                
        else:  # plain text
            return {
                "message": line,
                "timestamp": datetime.now().isoformat(),
                "source": "file"
            }
            
    def stop(self) -> None:
        """Stop reading"""
        self._running = False


@dataclass
class MemorySource(StreamSource):
    """
    Read logs from memory (for testing/development)
    """
    
    logs: List[Dict[str, Any]]
    delay: float = 0  # Delay between items (to simulate streaming)
    repeat: bool = False  # Repeat logs indefinitely
    
    def __post_init__(self):
        self.logger = get_logger("streaming.memory_source")
        
    async def __aiter__(self) -> AsyncIterator[Dict[str, Any]]:
        """Yield logs from memory"""
        while True:
            for log in self.logs:
                # Add timestamp if missing
                if "timestamp" not in log:
                    log = {**log, "timestamp": datetime.now()}
                    
                yield log
                
                if self.delay > 0:
                    await asyncio.sleep(self.delay)
                    
            if not self.repeat:
                break


@dataclass
class WebSocketSource(StreamSource):
    """
    Read logs from WebSocket connection
    """
    
    url: str
    reconnect: bool = True
    reconnect_delay: float = 5.0
    
    def __post_init__(self):
        self.logger = get_logger("streaming.websocket_source")
        self._running = True
        
    async def __aiter__(self) -> AsyncIterator[Dict[str, Any]]:
        """Read from WebSocket"""
        try:
            import websockets
        except ImportError:
            self.logger.error("websockets package not installed")
            return
            
        while self._running:
            try:
                async with websockets.connect(self.url) as websocket:
                    self.logger.info(f"Connected to {self.url}")
                    
                    async for message in websocket:
                        try:
                            # Try to parse as JSON
                            data = json.loads(message)
                            
                            # Ensure it's a dict
                            if isinstance(data, dict):
                                yield data
                            elif isinstance(data, list):
                                # Yield each item if it's a list
                                for item in data:
                                    if isinstance(item, dict):
                                        yield item
                            else:
                                yield {"message": str(data)}
                                
                        except json.JSONDecodeError:
                            # Treat as plain text
                            yield {
                                "message": message,
                                "timestamp": datetime.now().isoformat(),
                                "source": "websocket"
                            }
                            
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
                
                if not self.reconnect:
                    break
                    
                await asyncio.sleep(self.reconnect_delay)
                
    def stop(self) -> None:
        """Stop reading"""
        self._running = False


@dataclass
class KafkaSource(StreamSource):
    """
    Read logs from Kafka topics (placeholder)
    """
    
    topics: List[str]
    bootstrap_servers: str = "localhost:9092"
    group_id: str = "structured-logging"
    
    def __post_init__(self):
        self.logger = get_logger("streaming.kafka_source")
        
    async def __aiter__(self) -> AsyncIterator[Dict[str, Any]]:
        """Read from Kafka"""
        # In real implementation, would use aiokafka
        self.logger.info(f"Would read from Kafka topics: {self.topics}")
        
        # Placeholder implementation
        while False:
            yield {}


@dataclass
class HTTPSource(StreamSource):
    """
    Poll HTTP endpoint for logs
    """
    
    url: str
    method: str = "GET"
    headers: Optional[Dict[str, str]] = None
    params: Optional[Dict[str, Any]] = None
    poll_interval: float = 5.0
    
    def __post_init__(self):
        self.logger = get_logger("streaming.http_source")
        self._running = True
        self._last_timestamp = None
        
    async def __aiter__(self) -> AsyncIterator[Dict[str, Any]]:
        """Poll HTTP endpoint"""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            while self._running:
                try:
                    # Add timestamp parameter for incremental fetching
                    params = self.params or {}
                    if self._last_timestamp:
                        params["since"] = self._last_timestamp
                        
                    async with session.request(
                        self.method,
                        self.url,
                        headers=self.headers,
                        params=params
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Handle different response formats
                            if isinstance(data, list):
                                for item in data:
                                    if isinstance(item, dict):
                                        yield item
                                        # Update timestamp
                                        if "timestamp" in item:
                                            self._last_timestamp = item["timestamp"]
                                            
                            elif isinstance(data, dict):
                                if "logs" in data and isinstance(data["logs"], list):
                                    # Nested format
                                    for item in data["logs"]:
                                        if isinstance(item, dict):
                                            yield item
                                else:
                                    # Single item
                                    yield data
                                    
                        else:
                            self.logger.error(f"HTTP error: {response.status}")
                            
                except Exception as e:
                    self.logger.error(f"Request error: {e}")
                    
                await asyncio.sleep(self.poll_interval)
                
    def stop(self) -> None:
        """Stop polling"""
        self._running = False


# Factory function

def create_source(source_type: str, **kwargs) -> StreamSource:
    """
    Create a source instance
    
    Args:
        source_type: Type of source (file, memory, websocket, kafka, http)
        **kwargs: Source-specific configuration
        
    Returns:
        StreamSource instance
    """
    if source_type == "file":
        return LogFileSource(**kwargs)
    elif source_type == "memory":
        return MemorySource(**kwargs)
    elif source_type == "websocket":
        return WebSocketSource(**kwargs)
    elif source_type == "kafka":
        return KafkaSource(**kwargs)
    elif source_type == "http":
        return HTTPSource(**kwargs)
    else:
        raise ValueError(f"Unknown source type: {source_type}")