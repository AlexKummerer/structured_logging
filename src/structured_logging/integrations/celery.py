"""
Celery integration for structured logging

This module provides Celery task and worker integration for automatic structured
logging of task execution, retries, failures, and worker events.

Features:
- Automatic task execution logging
- Task retry and failure tracking
- Worker event logging
- Task result logging
- Performance metrics collection
- Distributed tracing support
- Task chain and group logging
"""

import functools
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Set

# Celery imports with availability checking
try:
    import celery
    from celery import Celery, Task, current_task
    from celery.exceptions import Retry
    from celery.result import AsyncResult
    from celery.signals import (
        after_task_publish,
        beat_init,
        before_task_publish,
        task_failure,
        task_postrun,
        task_prerun,
        task_retry,
        task_revoked,
        task_success,
        worker_process_init,
        worker_ready,
        worker_shutdown,
    )

    HAS_CELERY = True
    CELERY_VERSION = celery.__version__
except ImportError:
    Celery = None
    Task = object
    current_task = None
    AsyncResult = None
    Retry = Exception
    HAS_CELERY = False
    CELERY_VERSION = None

from ..context import request_context
from ..logger import get_logger


@dataclass
class CeleryLoggingConfig:
    """Configuration for Celery logging integration"""

    # Task logging
    log_task_arguments: bool = False  # Log task arguments (careful with sensitive data)
    log_task_kwargs: bool = False  # Log task keyword arguments
    log_task_result: bool = False  # Log task return value
    task_arg_max_length: int = 1000  # Max length for argument logging
    task_result_max_length: int = 1000  # Max length for result logging

    # Performance metrics
    include_task_runtime: bool = True  # Include task execution time
    include_queue_time: bool = True  # Include time spent in queue
    include_retry_count: bool = True  # Include retry attempt number

    # Worker events
    log_worker_events: bool = True  # Log worker lifecycle events
    log_beat_events: bool = True  # Log Celery beat scheduler events

    # Error handling
    log_exceptions: bool = True  # Log task exceptions
    include_traceback: bool = True  # Include exception traceback
    log_retries: bool = True  # Log task retry attempts

    # Task metadata
    include_task_id: bool = True  # Include task ID in logs
    include_task_name: bool = True  # Include task name
    include_queue_name: bool = True  # Include queue/routing info
    include_hostname: bool = True  # Include worker hostname
    include_correlation_id: bool = True  # Include correlation ID if present

    # Filtering
    excluded_tasks: Set[str] = field(
        default_factory=lambda: {
            "celery.chord_unlock",
            "celery.group",
            "celery.chain",
        }
    )
    excluded_queues: Set[str] = field(default_factory=set)
    only_errors: bool = False  # Only log failed tasks

    # Field naming
    task_id_field: str = "task_id"
    correlation_id_field: str = "correlation_id"
    queue_field: str = "queue"
    retry_field: str = "retry_count"

    # Distributed tracing
    propagate_correlation_id: bool = (
        True  # Propagate correlation ID through task chains
    )
    correlation_id_header: str = "correlation_id"  # Header for correlation ID


class StructuredLoggingTask(Task):
    """
    Custom Celery Task class with structured logging

    This task class automatically logs task execution with structured
    context including timing, arguments, results, and errors.
    """

    def __init__(self):
        super().__init__()
        self.logger = get_logger("celery.task")
        self.config = CeleryLoggingConfig()

    def __call__(self, *args, **kwargs):
        """Execute task with logging context"""
        # Extract correlation ID if present
        correlation_id = None
        if self.config.propagate_correlation_id:
            correlation_id = (
                self.request.headers.get(self.config.correlation_id_header)
                or self.request.correlation_id
                if hasattr(self.request, "correlation_id")
                else None
            )

        # Setup logging context
        context_data = {
            self.config.task_id_field: self.request.id,
            "task_name": self.name,
        }

        if correlation_id:
            context_data[self.config.correlation_id_field] = correlation_id

        if self.config.include_queue_name:
            context_data[self.config.queue_field] = self.request.routing_key

        # Execute with context
        with request_context(**context_data):
            return super().__call__(*args, **kwargs)


def setup_celery_logging(
    app: Optional[Celery] = None,
    config: Optional[CeleryLoggingConfig] = None,
    logger_name: Optional[str] = None,
) -> None:
    """
    Setup Celery application with structured logging

    Args:
        app: Celery application instance
        config: Logging configuration
        logger_name: Logger name (defaults to 'celery')

    Example:
        ```python
        from celery import Celery
        from structured_logging.integrations.celery import setup_celery_logging

        app = Celery('myapp')
        setup_celery_logging(app)
        ```
    """
    if not HAS_CELERY:
        raise ImportError(
            "celery is required for Celery integration. "
            "Install with: pip install celery"
        )

    config = config or CeleryLoggingConfig()
    logger_name = logger_name or "celery"

    # Store config globally for signal handlers
    _celery_logging_config.config = config
    _celery_logging_config.logger = get_logger(logger_name)

    # Connect signal handlers
    _connect_signals(config)

    # Set custom task class if app provided
    if app:
        app.Task = StructuredLoggingTask


# Global storage for configuration (used by signal handlers)
class _CeleryLoggingConfig:
    config: Optional[CeleryLoggingConfig] = None
    logger: Optional[Any] = None


_celery_logging_config = _CeleryLoggingConfig()


def _connect_signals(config: CeleryLoggingConfig) -> None:
    """Connect Celery signals for logging"""
    # Task execution signals
    task_prerun.connect(_task_prerun_handler)
    task_postrun.connect(_task_postrun_handler)

    if config.log_retries:
        task_retry.connect(_task_retry_handler)

    if config.log_exceptions:
        task_failure.connect(_task_failure_handler)

    task_success.connect(_task_success_handler)
    task_revoked.connect(_task_revoked_handler)

    # Task publishing signals
    before_task_publish.connect(_before_task_publish_handler)
    after_task_publish.connect(_after_task_publish_handler)

    # Worker signals
    if config.log_worker_events:
        worker_ready.connect(_worker_ready_handler)
        worker_shutdown.connect(_worker_shutdown_handler)
        worker_process_init.connect(_worker_process_init_handler)

    # Beat signals
    if config.log_beat_events:
        beat_init.connect(_beat_init_handler)


# Signal handlers


def _task_prerun_handler(
    sender=None, task_id=None, task=None, args=None, kwargs=None, **kw
):
    """Handle task pre-run signal"""
    config = _celery_logging_config.config
    logger = _celery_logging_config.logger

    if not config or not logger:
        return

    # Check if task should be excluded
    if task.name in config.excluded_tasks:
        return

    # Store start time
    if current_task:
        current_task.request._start_time = time.time()

    # Extract task data
    task_data = {
        "event": "task_started",
        "task_id": task_id,
        "task_name": task.name,
    }

    if config.include_hostname:
        task_data["hostname"] = sender

    if config.log_task_arguments and args:
        task_data["args"] = _truncate_data(args, config.task_arg_max_length)

    if config.log_task_kwargs and kwargs:
        task_data["kwargs"] = _truncate_data(kwargs, config.task_arg_max_length)

    logger.info(f"Task started: {task.name}", extra=task_data)


def _task_postrun_handler(
    sender=None,
    task_id=None,
    task=None,
    args=None,
    kwargs=None,
    retval=None,
    state=None,
    **kw,
):
    """Handle task post-run signal"""
    config = _celery_logging_config.config
    logger = _celery_logging_config.logger

    if not config or not logger:
        return

    if task.name in config.excluded_tasks:
        return

    # Calculate runtime
    runtime = None
    if current_task and hasattr(current_task.request, "_start_time"):
        runtime = time.time() - current_task.request._start_time

    task_data = {
        "event": "task_completed",
        "task_id": task_id,
        "task_name": task.name,
        "state": state,
    }

    if config.include_hostname:
        task_data["hostname"] = sender

    if runtime is not None and config.include_task_runtime:
        task_data["runtime_seconds"] = runtime

    if config.log_task_result and retval is not None:
        task_data["result"] = _truncate_data(retval, config.task_result_max_length)

    # Skip if only logging errors and task succeeded
    if config.only_errors and state == "SUCCESS":
        return

    logger.info(f"Task completed: {task.name} - {state}", extra=task_data)


def _task_retry_handler(sender=None, task_id=None, reason=None, einfo=None, **kw):
    """Handle task retry signal"""
    config = _celery_logging_config.config
    logger = _celery_logging_config.logger

    if not config or not logger:
        return

    retry_data = {
        "event": "task_retry",
        "task_id": task_id,
        "task_name": sender.name if sender else None,
        "reason": str(reason),
    }

    if config.include_retry_count and current_task:
        retry_data[config.retry_field] = current_task.request.retries

    if config.include_traceback and einfo:
        retry_data["traceback"] = str(einfo)

    logger.warning(
        f"Task retry: {sender.name if sender else task_id}", extra=retry_data
    )


def _task_failure_handler(
    sender=None,
    task_id=None,
    exception=None,
    args=None,
    kwargs=None,
    traceback=None,
    einfo=None,
    **kw,
):
    """Handle task failure signal"""
    config = _celery_logging_config.config
    logger = _celery_logging_config.logger

    if not config or not logger:
        return

    failure_data = {
        "event": "task_failed",
        "task_id": task_id,
        "task_name": sender.name if sender else None,
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
    }

    if config.include_traceback and einfo:
        failure_data["traceback"] = str(einfo)

    logger.error(
        (f"Task failed: {sender.name if sender else task_id} - "
         f"{type(exception).__name__}"),
        extra=failure_data,
        exc_info=config.include_traceback,
    )


def _task_success_handler(sender=None, result=None, **kw):
    """Handle task success signal"""
    config = _celery_logging_config.config
    logger = _celery_logging_config.logger

    if not config or not logger:
        return

    # Skip if only logging errors
    if config.only_errors:
        return

    success_data = {
        "event": "task_success",
        "task_name": sender.name if sender else None,
    }

    if config.log_task_result and result is not None:
        success_data["result"] = _truncate_data(
            result, config.task_result_max_length
        )

    logger.debug(
        f"Task succeeded: {sender.name if sender else 'unknown'}", extra=success_data
    )


def _task_revoked_handler(
    sender=None, request=None, terminated=None, signum=None, expired=None, **kw
):
    """Handle task revoked signal"""
    logger = _celery_logging_config.logger

    if not logger:
        return

    revoke_data = {
        "event": "task_revoked",
        "task_id": request.id,
        "task_name": request.name,
        "terminated": terminated,
        "expired": expired,
    }

    if signum:
        revoke_data["signal"] = signum

    logger.warning(f"Task revoked: {request.name}", extra=revoke_data)


def _before_task_publish_handler(
    sender=None, headers=None, body=None, routing_key=None, **kw
):
    """Handle before task publish signal"""
    config = _celery_logging_config.config
    logger = _celery_logging_config.logger

    if not config or not logger:
        return

    task_name = headers.get("task") if headers else None
    if task_name in config.excluded_tasks:
        return

    publish_data = {
        "event": "task_publishing",
        "task_name": task_name,
        "routing_key": routing_key,
    }

    logger.debug(f"Publishing task: {task_name}", extra=publish_data)


def _after_task_publish_handler(
    sender=None, headers=None, body=None, exchange=None, routing_key=None, **kw
):
    """Handle after task publish signal"""
    config = _celery_logging_config.config
    logger = _celery_logging_config.logger

    if not config or not logger:
        return

    task_name = headers.get("task") if headers else None
    if task_name in config.excluded_tasks:
        return

    publish_data = {
        "event": "task_published",
        "task_name": task_name,
        "exchange": exchange,
        "routing_key": routing_key,
    }

    logger.debug(f"Published task: {task_name}", extra=publish_data)


def _worker_ready_handler(sender=None, **kw):
    """Handle worker ready signal"""
    logger = _celery_logging_config.logger

    if not logger:
        return

    logger.info(
        "Celery worker ready",
        extra={
            "event": "worker_ready",
            "hostname": sender.hostname if sender else None,
        },
    )


def _worker_shutdown_handler(sender=None, **kw):
    """Handle worker shutdown signal"""
    logger = _celery_logging_config.logger

    if not logger:
        return

    logger.info(
        "Celery worker shutting down",
        extra={
            "event": "worker_shutdown",
            "hostname": sender.hostname if sender else None,
        },
    )


def _worker_process_init_handler(sender=None, **kw):
    """Handle worker process init signal"""
    logger = _celery_logging_config.logger

    if not logger:
        return

    logger.info(
        "Celery worker process initialized",
        extra={"event": "worker_process_init"},
    )


def _beat_init_handler(sender=None, **kw):
    """Handle beat init signal"""
    logger = _celery_logging_config.logger

    if not logger:
        return

    logger.info(
        "Celery beat scheduler initialized",
        extra={"event": "beat_init"},
    )


# Decorators


def log_task(
    name: Optional[str] = None,
    bind: bool = False,
    **log_kwargs,
):
    """
    Decorator for adding structured logging to individual Celery tasks

    Args:
        name: Custom task name
        bind: Whether to bind task instance as first argument
        **log_kwargs: Additional fields to include in logs

    Example:
        ```python
        @log_task(operation="process_payment")
        def process_payment(order_id: str) -> bool:
            # Task implementation
            return True
        ```
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger("celery.task")
            start_time = time.time()

            # Extract task info
            task = current_task
            task_name = name or func.__name__

            # Log task start
            logger.info(
                f"Task function started: {task_name}",
                extra={
                    "task_function": func.__name__,
                    "task_id": task.request.id if task else None,
                    **log_kwargs,
                },
            )

            try:
                # Execute task
                result = func(*args, **kwargs)

                # Log completion
                logger.info(
                    f"Task function completed: {task_name}",
                    extra={
                        "task_function": func.__name__,
                        "runtime_seconds": time.time() - start_time,
                        **log_kwargs,
                    },
                )

                return result

            except Exception as exc:
                logger.error(
                    f"Task function failed: {task_name}",
                    extra={
                        "task_function": func.__name__,
                        "exception_type": type(exc).__name__,
                        "runtime_seconds": time.time() - start_time,
                        **log_kwargs,
                    },
                    exc_info=True,
                )
                raise

        # Handle Celery task decorator
        if bind:
            wrapper = functools.partial(wrapper, bind=True)

        return wrapper

    return decorator


# Helper functions


def _truncate_data(data: Any, max_length: int) -> Any:
    """Truncate data for logging"""
    if isinstance(data, str) and len(data) > max_length:
        return data[:max_length] + "..."
    elif isinstance(data, (list, tuple)) and len(str(data)) > max_length:
        return f"[{type(data).__name__} with {len(data)} items]"
    elif isinstance(data, dict) and len(str(data)) > max_length:
        return f"[Dict with {len(data)} keys]"
    return data


def get_task_logger(task_name: Optional[str] = None) -> Any:
    """
    Get a structured logger for a specific task

    Args:
        task_name: Task name (defaults to current task name)

    Returns:
        Configured structured logger
    """
    if not task_name and current_task:
        task_name = current_task.name

    logger_name = f"celery.task.{task_name}" if task_name else "celery.task"
    return get_logger(logger_name)


def log_task_chain(chain_id: str, tasks: List[str]) -> None:
    """
    Log the start of a task chain

    Args:
        chain_id: Unique identifier for the chain
        tasks: List of task names in the chain
    """
    logger = get_logger("celery.chain")
    logger.info(
        f"Task chain started: {len(tasks)} tasks",
        extra={
            "chain_id": chain_id,
            "task_count": len(tasks),
            "tasks": tasks,
        },
    )


def log_task_group(group_id: str, tasks: List[str]) -> None:
    """
    Log the start of a task group

    Args:
        group_id: Unique identifier for the group
        tasks: List of task names in the group
    """
    logger = get_logger("celery.group")
    logger.info(
        f"Task group started: {len(tasks)} tasks",
        extra={
            "group_id": group_id,
            "task_count": len(tasks),
            "tasks": tasks,
        },
    )
