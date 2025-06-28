"""
SQLAlchemy integration for structured logging

This module provides SQLAlchemy event listeners and utilities for automatic
structured logging of database operations, query execution, and ORM events.

Features:
- Automatic query execution logging with timing
- Slow query detection and logging
- Connection pool event logging
- Transaction lifecycle logging
- ORM operation logging (flush, bulk operations)
- Query parameter sanitization
- Performance metrics collection
"""

import functools
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

# SQLAlchemy imports with availability checking
try:
    import sqlalchemy
    from sqlalchemy import Engine, event
    from sqlalchemy.engine import Connection
    from sqlalchemy.engine import Engine as EngineType
    from sqlalchemy.engine.cursor import CursorResult
    from sqlalchemy.orm import Session, SessionTransaction
    from sqlalchemy.pool import Pool
    from sqlalchemy.sql import ClauseElement

    HAS_SQLALCHEMY = True
    SQLALCHEMY_VERSION = sqlalchemy.__version__
except ImportError:
    event = None
    Engine = None
    Connection = None
    EngineType = None
    Session = None
    SessionTransaction = None
    Pool = None
    CursorResult = None
    ClauseElement = None
    HAS_SQLALCHEMY = False
    SQLALCHEMY_VERSION = None

from ..logger import get_logger


@dataclass
class SQLAlchemyLoggingConfig:
    """Configuration for SQLAlchemy logging integration"""

    # Query logging
    log_queries: bool = True  # Log all SQL queries
    log_query_parameters: bool = (
        False  # Log query parameters (careful with sensitive data)
    )
    query_param_length_limit: int = 1000  # Max length for parameter logging
    log_query_results: bool = False  # Log query result counts
    slow_query_threshold: float = 1.0  # Threshold for slow query warnings (seconds)

    # Connection logging
    log_connections: bool = True  # Log connection events
    log_connection_pool: bool = True  # Log connection pool events
    log_transactions: bool = True  # Log transaction begin/commit/rollback

    # ORM logging
    log_orm_events: bool = True  # Log ORM-specific events
    log_flush_events: bool = True  # Log session flush operations
    log_bulk_operations: bool = True  # Log bulk insert/update/delete

    # Performance metrics
    include_query_time: bool = True  # Include query execution time
    include_row_count: bool = True  # Include affected/returned row counts
    include_connection_info: bool = True  # Include connection details

    # Filtering
    excluded_tables: Set[str] = field(
        default_factory=lambda: {
            "alembic_version",  # Migration tracking
            "sqlalchemy_logs",  # Avoid recursive logging
        }
    )
    excluded_operations: Set[str] = field(default_factory=set)
    only_errors: bool = False  # Only log failed operations

    # Field naming
    query_field: str = "sql_query"
    duration_field: str = "query_duration_seconds"
    row_count_field: str = "row_count"

    # Sensitive data handling
    sensitive_param_names: Set[str] = field(
        default_factory=lambda: {
            "password",
            "pwd",
            "secret",
            "token",
            "api_key",
            "auth",
            "credential",
        }
    )
    param_sanitization: bool = True  # Sanitize sensitive parameters


class SQLAlchemyQueryLogger:
    """
    Query logger for SQLAlchemy with structured logging

    Tracks query execution with timing, parameters, and results.
    """

    def __init__(self, config: Optional[SQLAlchemyLoggingConfig] = None):
        self.config = config or SQLAlchemyLoggingConfig()
        self.logger = get_logger("sqlalchemy.query")
        self._query_contexts = {}

    def before_execute(
        self,
        conn: Connection,
        clauseelement: ClauseElement,
        multiparams: tuple,
        params: dict,
        execution_options: dict,
    ) -> None:
        """Called before query execution"""
        if not self.config.log_queries:
            return

        # Generate context ID for this query
        context_id = id(clauseelement)

        # Store start time
        self._query_contexts[context_id] = {
            "start_time": time.time(),
            "query": str(clauseelement),
            "params": self._sanitize_params(params) if params else None,
        }

    def after_execute(
        self,
        conn: Connection,
        clauseelement: ClauseElement,
        multiparams: tuple,
        params: dict,
        execution_options: dict,
        result: CursorResult,
    ) -> None:
        """Called after successful query execution"""
        if not self.config.log_queries:
            return

        context_id = id(clauseelement)
        query_context = self._query_contexts.pop(context_id, None)

        if not query_context:
            return

        # Calculate duration
        duration = time.time() - query_context["start_time"]

        # Check if this is a slow query
        is_slow = duration > self.config.slow_query_threshold

        # Extract query info
        query_data = {
            self.config.query_field: query_context["query"],
            self.config.duration_field: duration,
            "slow_query": is_slow,
        }

        # Add parameters if configured
        if self.config.log_query_parameters and query_context["params"]:
            query_data["parameters"] = query_context["params"]

        # Add row count if available
        if self.config.include_row_count and hasattr(result, "rowcount"):
            query_data[self.config.row_count_field] = result.rowcount

        # Add connection info if configured
        if self.config.include_connection_info:
            query_data["database"] = self._get_database_info(conn)

        # Log with appropriate level
        if is_slow:
            self.logger.warning(
                f"Slow query detected ({duration:.3f}s)", extra=query_data
            )
        else:
            self.logger.debug(f"Query executed in {duration:.3f}s", extra=query_data)

    def handle_error(
        self,
        exception_context: Dict[str, Any],
    ) -> None:
        """Called when query execution fails"""
        clauseelement = exception_context.get("clauseelement")
        exception = exception_context.get("exception")

        if not clauseelement:
            return

        context_id = id(clauseelement)
        query_context = self._query_contexts.pop(context_id, None)

        if not query_context:
            return

        # Calculate duration
        duration = time.time() - query_context["start_time"]

        error_data = {
            self.config.query_field: query_context["query"],
            self.config.duration_field: duration,
            "error_type": type(exception).__name__,
            "error_message": str(exception),
        }

        if self.config.log_query_parameters and query_context["params"]:
            error_data["parameters"] = query_context["params"]

        self.logger.error("Query execution failed", extra=error_data, exc_info=True)

    def _sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize query parameters"""
        if not self.config.param_sanitization:
            return params

        sanitized = {}
        for key, value in params.items():
            # Check if parameter name suggests sensitive data
            if any(
                sensitive in key.lower()
                for sensitive in self.config.sensitive_param_names
            ):
                sanitized[key] = "[REDACTED]"
            else:
                # Truncate if too long
                value_str = str(value)
                if len(value_str) > self.config.query_param_length_limit:
                    sanitized[key] = (
                        value_str[: self.config.query_param_length_limit] + "..."
                    )
                else:
                    sanitized[key] = value

        return sanitized

    def _get_database_info(self, conn: Connection) -> str:
        """Extract database information from connection"""
        try:
            # Try to get database URL without credentials
            url = conn.engine.url
            # Return safe representation without password
            return f"{url.drivername}://{url.host}/{url.database}"
        except Exception:
            return "unknown"


class SQLAlchemyConnectionLogger:
    """
    Connection and pool event logger for SQLAlchemy
    """

    def __init__(self, config: Optional[SQLAlchemyLoggingConfig] = None):
        self.config = config or SQLAlchemyLoggingConfig()
        self.logger = get_logger("sqlalchemy.connection")

    def on_connect(self, dbapi_conn, connection_record) -> None:
        """Called when a new connection is established"""
        if not self.config.log_connections:
            return

        self.logger.info(
            "Database connection established",
            extra={
                "event": "connection_established",
                "connection_id": id(dbapi_conn),
            },
        )

    def on_checkout(self, dbapi_conn, connection_record, connection_proxy) -> None:
        """Called when connection is checked out from pool"""
        if not self.config.log_connection_pool:
            return

        self.logger.debug(
            "Connection checked out from pool",
            extra={
                "event": "pool_checkout",
                "connection_id": id(dbapi_conn),
            },
        )

    def on_checkin(self, dbapi_conn, connection_record) -> None:
        """Called when connection is returned to pool"""
        if not self.config.log_connection_pool:
            return

        self.logger.debug(
            "Connection returned to pool",
            extra={
                "event": "pool_checkin",
                "connection_id": id(dbapi_conn),
            },
        )

    def on_close(self, dbapi_conn, connection_record) -> None:
        """Called when connection is closed"""
        if not self.config.log_connections:
            return

        self.logger.info(
            "Database connection closed",
            extra={
                "event": "connection_closed",
                "connection_id": id(dbapi_conn),
            },
        )


class SQLAlchemyTransactionLogger:
    """
    Transaction event logger for SQLAlchemy
    """

    def __init__(self, config: Optional[SQLAlchemyLoggingConfig] = None):
        self.config = config or SQLAlchemyLoggingConfig()
        self.logger = get_logger("sqlalchemy.transaction")
        self._transaction_contexts = {}

    def on_begin(self, session: Session) -> None:
        """Called when transaction begins"""
        if not self.config.log_transactions:
            return

        transaction_id = id(session.transaction)
        self._transaction_contexts[transaction_id] = {
            "start_time": time.time(),
        }

        self.logger.debug(
            "Transaction started",
            extra={
                "event": "transaction_begin",
                "transaction_id": transaction_id,
            },
        )

    def on_commit(self, session: Session) -> None:
        """Called when transaction commits"""
        if not self.config.log_transactions:
            return

        transaction_id = id(session.transaction)
        context = self._transaction_contexts.pop(transaction_id, None)

        if context:
            duration = time.time() - context["start_time"]
            self.logger.info(
                "Transaction committed",
                extra={
                    "event": "transaction_commit",
                    "transaction_id": transaction_id,
                    "duration_seconds": duration,
                },
            )

    def on_rollback(self, session: Session) -> None:
        """Called when transaction rolls back"""
        if not self.config.log_transactions:
            return

        transaction_id = id(session.transaction)
        context = self._transaction_contexts.pop(transaction_id, None)

        if context:
            duration = time.time() - context["start_time"]
            self.logger.warning(
                "Transaction rolled back",
                extra={
                    "event": "transaction_rollback",
                    "transaction_id": transaction_id,
                    "duration_seconds": duration,
                },
            )


class SQLAlchemyORMLogger:
    """
    ORM event logger for SQLAlchemy
    """

    def __init__(self, config: Optional[SQLAlchemyLoggingConfig] = None):
        self.config = config or SQLAlchemyLoggingConfig()
        self.logger = get_logger("sqlalchemy.orm")

    def before_flush(self, session: Session, flush_context, instances) -> None:
        """Called before session flush"""
        if not self.config.log_flush_events:
            return

        # Store flush start time
        session._flush_start_time = time.time()

        stats = {
            "new": len(session.new),
            "dirty": len(session.dirty),
            "deleted": len(session.deleted),
        }

        self.logger.debug(
            "Session flush starting",
            extra={
                "event": "flush_start",
                **stats,
            },
        )

    def after_flush(self, session: Session, flush_context) -> None:
        """Called after session flush"""
        if not self.config.log_flush_events:
            return

        duration = time.time() - getattr(session, "_flush_start_time", time.time())

        self.logger.info(
            f"Session flush completed in {duration:.3f}s",
            extra={
                "event": "flush_complete",
                "duration_seconds": duration,
            },
        )

    def after_bulk_insert(self, mapper, connection, target) -> None:
        """Called after bulk insert"""
        if not self.config.log_bulk_operations:
            return

        self.logger.info(
            "Bulk insert completed",
            extra={
                "event": "bulk_insert",
                "table": mapper.class_.__tablename__,
                "row_count": len(target) if hasattr(target, "__len__") else None,
            },
        )

    def after_bulk_update(self, mapper, connection, target) -> None:
        """Called after bulk update"""
        if not self.config.log_bulk_operations:
            return

        self.logger.info(
            "Bulk update completed",
            extra={
                "event": "bulk_update",
                "table": mapper.class_.__tablename__,
            },
        )

    def after_bulk_delete(self, mapper, connection, target) -> None:
        """Called after bulk delete"""
        if not self.config.log_bulk_operations:
            return

        self.logger.info(
            "Bulk delete completed",
            extra={
                "event": "bulk_delete",
                "table": mapper.class_.__tablename__,
            },
        )


def setup_sqlalchemy_logging(
    engine: Engine,
    config: Optional[SQLAlchemyLoggingConfig] = None,
    session_factory: Optional[Any] = None,
) -> None:
    """
    Setup SQLAlchemy engine with structured logging

    Args:
        engine: SQLAlchemy engine instance
        config: Logging configuration
        session_factory: Optional session factory for ORM event logging

    Example:
        ```python
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from structured_logging.integrations.sqlalchemy import setup_sqlalchemy_logging

        engine = create_engine('postgresql://localhost/mydb')
        Session = sessionmaker(bind=engine)

        setup_sqlalchemy_logging(engine, session_factory=Session)
        ```
    """
    if not HAS_SQLALCHEMY:
        raise ImportError(
            "sqlalchemy is required for SQLAlchemy integration. "
            "Install with: pip install sqlalchemy"
        )

    config = config or SQLAlchemyLoggingConfig()

    # Create loggers
    query_logger = SQLAlchemyQueryLogger(config)
    connection_logger = SQLAlchemyConnectionLogger(config)
    transaction_logger = SQLAlchemyTransactionLogger(config)
    orm_logger = SQLAlchemyORMLogger(config)

    # Register query event listeners
    event.listen(engine, "before_execute", query_logger.before_execute)
    event.listen(engine, "after_execute", query_logger.after_execute)
    event.listen(engine, "handle_error", query_logger.handle_error)

    # Register connection pool event listeners
    event.listen(engine.pool, "connect", connection_logger.on_connect)
    event.listen(engine.pool, "checkout", connection_logger.on_checkout)
    event.listen(engine.pool, "checkin", connection_logger.on_checkin)
    event.listen(engine.pool, "close", connection_logger.on_close)

    # Register session event listeners if session factory provided
    if session_factory:
        event.listen(session_factory, "after_begin", transaction_logger.on_begin)
        event.listen(session_factory, "after_commit", transaction_logger.on_commit)
        event.listen(session_factory, "after_rollback", transaction_logger.on_rollback)

        event.listen(session_factory, "before_flush", orm_logger.before_flush)
        event.listen(session_factory, "after_flush", orm_logger.after_flush)
        event.listen(session_factory, "after_bulk_insert", orm_logger.after_bulk_insert)
        event.listen(session_factory, "after_bulk_update", orm_logger.after_bulk_update)
        event.listen(session_factory, "after_bulk_delete", orm_logger.after_bulk_delete)


def log_query(
    name: Optional[str] = None,
    **extra_fields,
):
    """
    Decorator for adding structured logging to individual database functions

    Args:
        name: Custom name for the operation
        **extra_fields: Additional fields to include in logs

    Example:
        ```python
        @log_query(operation="user_lookup", table="users")
        def get_user_by_email(session, email):
            return session.query(User).filter_by(email=email).first()
        ```
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger("sqlalchemy.function")
            start_time = time.time()
            operation_name = name or func.__name__

            # Log function start
            logger.debug(
                f"Database operation started: {operation_name}",
                extra={
                    "operation": operation_name,
                    **extra_fields,
                },
            )

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Log completion
                duration = time.time() - start_time
                logger.info(
                    f"Database operation completed: {operation_name}",
                    extra={
                        "operation": operation_name,
                        "duration_seconds": duration,
                        **extra_fields,
                    },
                )

                return result

            except Exception as exc:
                duration = time.time() - start_time
                logger.error(
                    f"Database operation failed: {operation_name}",
                    extra={
                        "operation": operation_name,
                        "duration_seconds": duration,
                        "error_type": type(exc).__name__,
                        **extra_fields,
                    },
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator


def get_query_logger(name: Optional[str] = None) -> Any:
    """
    Get a structured logger for database operations

    Args:
        name: Logger name (defaults to 'sqlalchemy')

    Returns:
        Configured structured logger
    """
    return get_logger(name or "sqlalchemy")


# Context manager for database operations


class DatabaseOperation:
    """
    Context manager for logging database operations

    Example:
        ```python
        with DatabaseOperation("user_import", table="users") as op:
            # Import users
            for user_data in users:
                session.add(User(**user_data))
            session.commit()
            op.set_result({"imported": len(users)})
        ```
    """

    def __init__(self, operation_name: str, **extra_fields):
        self.operation_name = operation_name
        self.extra_fields = extra_fields
        self.logger = get_logger("sqlalchemy.operation")
        self.start_time = None
        self.result = None

    def __enter__(self):
        self.start_time = time.time()

        self.logger.info(
            f"Database operation started: {self.operation_name}",
            extra={
                "operation": self.operation_name,
                "status": "started",
                **self.extra_fields,
            },
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time

        if exc_type:
            self.logger.error(
                f"Database operation failed: {self.operation_name}",
                extra={
                    "operation": self.operation_name,
                    "status": "failed",
                    "duration_seconds": duration,
                    "error_type": exc_type.__name__,
                    "error_message": str(exc_val),
                    **self.extra_fields,
                },
                exc_info=True,
            )
        else:
            extra = {
                "operation": self.operation_name,
                "status": "completed",
                "duration_seconds": duration,
                **self.extra_fields,
            }

            if self.result:
                extra["result"] = self.result

            self.logger.info(
                f"Database operation completed: {self.operation_name}",
                extra=extra,
            )

        return False  # Don't suppress exceptions

    def set_result(self, result: Any) -> None:
        """Set operation result for logging"""
        self.result = result
