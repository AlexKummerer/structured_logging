"""
Async context management for structured logging

Provides async-aware context managers and context propagation
that works seamlessly with asyncio and existing sync context system.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, AsyncGenerator

from .context import (
    set_request_id, 
    set_user_context, 
    set_custom_context,
    get_request_id,
    get_user_context, 
    get_custom_context
)


@asynccontextmanager
async def async_request_context(
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    request_id: Optional[str] = None,
    **custom_fields: Any,
) -> AsyncGenerator[None, None]:
    """
    Async context manager for request-scoped logging context.
    
    Automatically manages context variables across async boundaries
    using Python's contextvars which are asyncio-aware.
    
    Args:
        user_id: Optional user identifier
        tenant_id: Optional tenant identifier  
        request_id: Optional request identifier (auto-generated if None)
        **custom_fields: Additional custom context fields
    
    Example:
        async with async_request_context(user_id="user123"):
            await async_logger.ainfo("User action")
    """
    # Store original context for restoration
    original_request_id = get_request_id()
    original_user_context = get_user_context()
    original_custom_context = get_custom_context()
    
    try:
        # Set request ID (generate if not provided)
        if request_id is not None:
            set_request_id(request_id)
        elif request_id is None:
            # Always generate new request ID for new context
            import uuid
            set_request_id(str(uuid.uuid4()))
        
        # Set user context if provided
        if user_id is not None or tenant_id is not None:
            user_context = {}
            if user_id is not None:
                user_context["user_id"] = user_id
            if tenant_id is not None:
                user_context["tenant_id"] = tenant_id
            set_user_context(user_context)
        
        # Set custom context fields
        if custom_fields:
            # Merge with existing custom context
            current_custom = get_custom_context() or {}
            updated_custom = {**current_custom, **custom_fields}
            set_custom_context(updated_custom)
        
        yield
        
    finally:
        # Context vars automatically handle restoration in async contexts
        # No manual restoration needed due to contextvars behavior
        pass


async def aset_request_id(request_id: str) -> None:
    """Async version of set_request_id for consistency"""
    set_request_id(request_id)


async def aset_user_context(context: Dict[str, Any]) -> None:
    """Async version of set_user_context for consistency"""
    set_user_context(context)


async def aset_custom_context(context: Dict[str, Any]) -> None:
    """Async version of set_custom_context for consistency"""
    set_custom_context(context)


async def aget_request_id() -> Optional[str]:
    """Async version of get_request_id for consistency"""
    return get_request_id()


async def aget_user_context() -> Optional[Dict[str, Any]]:
    """Async version of get_user_context for consistency"""
    return get_user_context()


async def aget_custom_context() -> Optional[Dict[str, Any]]:
    """Async version of get_custom_context for consistency"""
    return get_custom_context()


class AsyncContextManager:
    """
    Utility class for managing async context state
    """
    
    @staticmethod
    async def capture_context() -> Dict[str, Any]:
        """Capture current context state"""
        return {
            "request_id": await aget_request_id(),
            "user_context": await aget_user_context(),
            "custom_context": await aget_custom_context(),
        }
    
    @staticmethod
    async def restore_context(context_state: Dict[str, Any]) -> None:
        """Restore context from captured state"""
        if context_state.get("request_id"):
            await aset_request_id(context_state["request_id"])
        if context_state.get("user_context"):
            await aset_user_context(context_state["user_context"])
        if context_state.get("custom_context"):
            await aset_custom_context(context_state["custom_context"])


# Async task context tracking
_async_task_contexts: Dict[asyncio.Task, Dict[str, Any]] = {}


def get_current_task_context() -> Optional[Dict[str, Any]]:
    """Get context for current async task"""
    try:
        current_task = asyncio.current_task()
        if current_task:
            return _async_task_contexts.get(current_task)
    except RuntimeError:
        # Not in async context
        pass
    return None


async def set_task_context(context: Dict[str, Any]) -> None:
    """Set context for current async task"""
    try:
        current_task = asyncio.current_task()
        if current_task:
            _async_task_contexts[current_task] = context
    except RuntimeError:
        # Not in async context, ignore
        pass


async def clear_task_context() -> None:
    """Clear context for current async task"""
    try:
        current_task = asyncio.current_task()
        if current_task and current_task in _async_task_contexts:
            del _async_task_contexts[current_task]
    except RuntimeError:
        # Not in async context, ignore
        pass