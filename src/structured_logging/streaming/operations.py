"""
Stream operations for log processing

This module provides windowing, aggregation, and transformation operations
for stream processing.
"""

import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..logger import get_logger


# Window Operations

class Window(ABC):
    """Base class for window operations"""
    
    @abstractmethod
    async def process(
        self,
        item: Dict[str, Any],
        ctx: Any
    ) -> Optional[List[Dict[str, Any]]]:
        """Process an item and potentially emit window results"""
        pass
        
    @abstractmethod
    def get_window_key(self, timestamp: datetime) -> str:
        """Get the window key for a timestamp"""
        pass


@dataclass
class TumblingWindow(Window):
    """
    Fixed-size, non-overlapping windows
    
    Example: 5-minute windows that don't overlap
    """
    
    size_seconds: int
    aggregators: List[Any] = None
    
    def __post_init__(self):
        self.logger = get_logger("streaming.tumbling_window")
        self.windows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.window_metadata: Dict[str, Dict[str, Any]] = {}
        
    async def process(
        self,
        item: Dict[str, Any],
        ctx: Any
    ) -> Optional[List[Dict[str, Any]]]:
        """Process item into tumbling window"""
        timestamp = item.get("timestamp", datetime.now())
        window_key = self.get_window_key(timestamp)
        
        # Add to window
        self.windows[window_key].append(item)
        
        # Initialize window metadata
        if window_key not in self.window_metadata:
            window_start = self._get_window_start(timestamp)
            window_end = window_start + timedelta(seconds=self.size_seconds)
            self.window_metadata[window_key] = {
                "start": window_start,
                "end": window_end,
                "item_count": 0
            }
            
        self.window_metadata[window_key]["item_count"] += 1
        
        # Check if we should emit completed windows
        results = []
        current_watermark = ctx.watermark
        
        for key, metadata in list(self.window_metadata.items()):
            if metadata["end"] <= current_watermark:
                # Window is complete
                window_items = self.windows.pop(key, [])
                window_meta = self.window_metadata.pop(key)
                
                result = {
                    "window_start": window_meta["start"],
                    "window_end": window_meta["end"],
                    "item_count": len(window_items),
                    "items": window_items
                }
                
                # Apply aggregators if configured
                if self.aggregators:
                    for agg in self.aggregators:
                        result.update(await agg.compute(window_items, ctx))
                        
                results.append(result)
                
        return results if results else None
        
    def get_window_key(self, timestamp: datetime) -> str:
        """Get window key for timestamp"""
        window_start = self._get_window_start(timestamp)
        return window_start.isoformat()
        
    def _get_window_start(self, timestamp: datetime) -> datetime:
        """Calculate window start time"""
        epoch = datetime(1970, 1, 1)
        total_seconds = (timestamp - epoch).total_seconds()
        window_number = int(total_seconds // self.size_seconds)
        window_start_seconds = window_number * self.size_seconds
        return epoch + timedelta(seconds=window_start_seconds)


@dataclass
class SlidingWindow(Window):
    """
    Fixed-size, overlapping windows
    
    Example: 10-minute windows that slide every 1 minute
    """
    
    size_seconds: int
    slide_seconds: int
    aggregators: List[Any] = None
    
    def __post_init__(self):
        self.logger = get_logger("streaming.sliding_window")
        self.windows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.window_metadata: Dict[str, Dict[str, Any]] = {}
        
    async def process(
        self,
        item: Dict[str, Any],
        ctx: Any
    ) -> Optional[List[Dict[str, Any]]]:
        """Process item into sliding windows"""
        timestamp = item.get("timestamp", datetime.now())
        
        # Add to all applicable windows
        window_keys = self._get_window_keys(timestamp)
        for window_key in window_keys:
            self.windows[window_key].append(item)
            
            # Initialize window metadata
            if window_key not in self.window_metadata:
                window_start = datetime.fromisoformat(window_key)
                window_end = window_start + timedelta(seconds=self.size_seconds)
                self.window_metadata[window_key] = {
                    "start": window_start,
                    "end": window_end,
                    "item_count": 0
                }
                
            self.window_metadata[window_key]["item_count"] += 1
            
        # Emit completed windows
        results = []
        current_watermark = ctx.watermark
        
        for key, metadata in list(self.window_metadata.items()):
            if metadata["end"] <= current_watermark:
                window_items = self.windows.pop(key, [])
                window_meta = self.window_metadata.pop(key)
                
                result = {
                    "window_start": window_meta["start"],
                    "window_end": window_meta["end"],
                    "item_count": len(window_items),
                    "items": window_items
                }
                
                if self.aggregators:
                    for agg in self.aggregators:
                        result.update(await agg.compute(window_items, ctx))
                        
                results.append(result)
                
        return results if results else None
        
    def get_window_key(self, timestamp: datetime) -> str:
        """Get primary window key for timestamp"""
        keys = self._get_window_keys(timestamp)
        return keys[0] if keys else ""
        
    def _get_window_keys(self, timestamp: datetime) -> List[str]:
        """Get all window keys that include this timestamp"""
        keys = []
        
        # Calculate the earliest window that could contain this timestamp
        earliest_start = timestamp - timedelta(seconds=self.size_seconds - self.slide_seconds)
        
        # Generate all window starts
        current = earliest_start
        while current <= timestamp:
            if current + timedelta(seconds=self.size_seconds) > timestamp:
                keys.append(current.isoformat())
            current += timedelta(seconds=self.slide_seconds)
            
        return keys


@dataclass
class SessionWindow(Window):
    """
    Dynamic windows based on activity gaps
    
    Example: Windows that close after 30 seconds of inactivity
    """
    
    gap_seconds: int
    max_duration_seconds: Optional[int] = None
    aggregators: List[Any] = None
    
    def __post_init__(self):
        self.logger = get_logger("streaming.session_window")
        self.sessions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.session_metadata: Dict[str, Dict[str, Any]] = {}
        
    async def process(
        self,
        item: Dict[str, Any],
        ctx: Any
    ) -> Optional[List[Dict[str, Any]]]:
        """Process item into session windows"""
        timestamp = item.get("timestamp", datetime.now())
        session_key = self._get_session_key(item)
        
        # Check if we need to start a new session
        if session_key in self.session_metadata:
            last_activity = self.session_metadata[session_key]["last_activity"]
            if (timestamp - last_activity).total_seconds() > self.gap_seconds:
                # Gap exceeded, close old session
                results = await self._close_session(session_key, ctx)
                # Start new session
                self._start_session(session_key, timestamp)
                self.sessions[session_key].append(item)
                return results
                
        else:
            # New session
            self._start_session(session_key, timestamp)
            
        # Add to current session
        self.sessions[session_key].append(item)
        self.session_metadata[session_key]["last_activity"] = timestamp
        self.session_metadata[session_key]["item_count"] += 1
        
        # Check max duration
        if self.max_duration_seconds:
            session_start = self.session_metadata[session_key]["start"]
            if (timestamp - session_start).total_seconds() > self.max_duration_seconds:
                return await self._close_session(session_key, ctx)
                
        return None
        
    def get_window_key(self, timestamp: datetime) -> str:
        """Get window key (not applicable for sessions)"""
        return f"session_{timestamp.isoformat()}"
        
    def _get_session_key(self, item: Dict[str, Any]) -> str:
        """Get session key from item (e.g., user_id, session_id)"""
        return (
            item.get("session_id") or
            item.get("user_id") or
            item.get("request_id") or
            "default"
        )
        
    def _start_session(self, session_key: str, timestamp: datetime) -> None:
        """Start a new session"""
        self.session_metadata[session_key] = {
            "start": timestamp,
            "last_activity": timestamp,
            "item_count": 0
        }
        
    async def _close_session(
        self,
        session_key: str,
        ctx: Any
    ) -> List[Dict[str, Any]]:
        """Close a session and emit results"""
        session_items = self.sessions.pop(session_key, [])
        session_meta = self.session_metadata.pop(session_key, {})
        
        if not session_items:
            return []
            
        result = {
            "session_key": session_key,
            "session_start": session_meta.get("start"),
            "session_end": session_meta.get("last_activity"),
            "duration_seconds": (
                session_meta.get("last_activity", datetime.now()) -
                session_meta.get("start", datetime.now())
            ).total_seconds(),
            "item_count": len(session_items),
            "items": session_items
        }
        
        if self.aggregators:
            for agg in self.aggregators:
                result.update(await agg.compute(session_items, ctx))
                
        return [result]


# Aggregation Operations

class Aggregator(ABC):
    """Base class for aggregation operations"""
    
    @abstractmethod
    async def compute(
        self,
        items: List[Dict[str, Any]],
        ctx: Any
    ) -> Dict[str, Any]:
        """Compute aggregation on items"""
        pass


@dataclass
class CountAggregator(Aggregator):
    """Count the number of items"""
    
    field: Optional[str] = None
    
    async def compute(
        self,
        items: List[Dict[str, Any]],
        ctx: Any
    ) -> Dict[str, Any]:
        if self.field:
            # Count non-null values of field
            count = sum(1 for item in items if self.field in item and item[self.field] is not None)
            return {f"count_{self.field}": count}
        else:
            return {"count": len(items)}


@dataclass
class SumAggregator(Aggregator):
    """Sum numeric values"""
    
    field: str
    
    async def compute(
        self,
        items: List[Dict[str, Any]],
        ctx: Any
    ) -> Dict[str, Any]:
        total = sum(
            item.get(self.field, 0)
            for item in items
            if isinstance(item.get(self.field), (int, float))
        )
        return {f"sum_{self.field}": total}


@dataclass
class AvgAggregator(Aggregator):
    """Calculate average of numeric values"""
    
    field: str
    
    async def compute(
        self,
        items: List[Dict[str, Any]],
        ctx: Any
    ) -> Dict[str, Any]:
        values = [
            item.get(self.field)
            for item in items
            if isinstance(item.get(self.field), (int, float))
        ]
        
        if values:
            avg = sum(values) / len(values)
            return {f"avg_{self.field}": avg}
        else:
            return {f"avg_{self.field}": None}


@dataclass
class MinAggregator(Aggregator):
    """Find minimum value"""
    
    field: str
    
    async def compute(
        self,
        items: List[Dict[str, Any]],
        ctx: Any
    ) -> Dict[str, Any]:
        values = [
            item.get(self.field)
            for item in items
            if item.get(self.field) is not None
        ]
        
        if values:
            return {f"min_{self.field}": min(values)}
        else:
            return {f"min_{self.field}": None}


@dataclass
class MaxAggregator(Aggregator):
    """Find maximum value"""
    
    field: str
    
    async def compute(
        self,
        items: List[Dict[str, Any]],
        ctx: Any
    ) -> Dict[str, Any]:
        values = [
            item.get(self.field)
            for item in items
            if item.get(self.field) is not None
        ]
        
        if values:
            return {f"max_{self.field}": max(values)}
        else:
            return {f"max_{self.field}": None}


# Factory functions

def count(field: Optional[str] = None) -> CountAggregator:
    """Create a count aggregator"""
    return CountAggregator(field)


def sum_of(field: str) -> SumAggregator:
    """Create a sum aggregator"""
    return SumAggregator(field)


def avg(field: str) -> AvgAggregator:
    """Create an average aggregator"""
    return AvgAggregator(field)


def min_of(field: str) -> MinAggregator:
    """Create a minimum aggregator"""
    return MinAggregator(field)


def max_of(field: str) -> MaxAggregator:
    """Create a maximum aggregator"""
    return MaxAggregator(field)


# Stream transformation operations

async def filter_stream(
    items: List[Dict[str, Any]],
    predicate: Callable[[Dict[str, Any]], bool]
) -> List[Dict[str, Any]]:
    """Filter items in a stream"""
    return [item for item in items if predicate(item)]


async def map_stream(
    items: List[Dict[str, Any]],
    mapper: Callable[[Dict[str, Any]], Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Map items in a stream"""
    return [mapper(item) for item in items]


async def reduce_stream(
    items: List[Dict[str, Any]],
    reducer: Callable[[Any, Dict[str, Any]], Any],
    initial: Any = None
) -> Any:
    """Reduce items in a stream"""
    result = initial
    for item in items:
        result = reducer(result, item)
    return result


def group_by(
    field: str
) -> Callable[[List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    """Group items by field value"""
    def grouper(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        groups = defaultdict(list)
        for item in items:
            key = item.get(field, "unknown")
            groups[str(key)].append(item)
        return dict(groups)
    return grouper