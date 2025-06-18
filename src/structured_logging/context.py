import uuid
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Optional

request_id: ContextVar[str] = ContextVar("request_id", default="")
user_context: ContextVar[Dict[str, Any]] = ContextVar("user_context", default={})
custom_context: ContextVar[Dict[str, Any]] = ContextVar("custom_context", default={})


def get_request_id() -> str:
    """Get current request ID"""
    return request_id.get("")


def set_request_id(req_id: str) -> None:
    """Set request ID for current context"""
    request_id.set(req_id)


def get_user_context() -> Dict[str, Any]:
    """Get current user context"""
    return user_context.get({})


def set_user_context(context: Dict[str, Any]) -> None:
    """Set user context for current context"""
    user_context.set(context or {})


def get_custom_context() -> Dict[str, Any]:
    """Get current custom context"""
    return custom_context.get({})


def set_custom_context(context: Dict[str, Any]) -> None:
    """Set custom context for current context"""
    custom_context.set(context or {})


def update_custom_context(**kwargs: Any) -> None:
    """Update custom context with additional fields"""
    current = get_custom_context()
    current.update(kwargs)
    set_custom_context(current)


@contextmanager
def request_context(
    user_id: Optional[str] = None, tenant_id: Optional[str] = None, **custom_fields: Any
) -> Generator[str, None, None]:
    """Context manager for setting up request-scoped logging context"""
    req_id = str(uuid.uuid4())

    old_request_id = get_request_id()
    old_user_context = get_user_context()
    old_custom_context = get_custom_context()

    try:
        set_request_id(req_id)

        user_ctx = {}
        if user_id is not None:
            user_ctx["user_id"] = user_id
        if tenant_id is not None:
            user_ctx["tenant_id"] = tenant_id
        set_user_context(user_ctx)

        set_custom_context(custom_fields)

        yield req_id
    finally:
        set_request_id(old_request_id)
        set_user_context(old_user_context)
        set_custom_context(old_custom_context)
