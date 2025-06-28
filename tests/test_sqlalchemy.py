"""
Tests for SQLAlchemy integration
"""

import time
from unittest.mock import Mock, patch

import pytest

# Skip all tests if SQLAlchemy is not installed
try:
    from sqlalchemy import Column, Integer, String, create_engine, event
    from sqlalchemy.engine import Connection, Engine
    from sqlalchemy.engine.cursor import CursorResult
    from sqlalchemy.orm import Session, declarative_base, sessionmaker
    from sqlalchemy.pool import Pool

    HAS_SQLALCHEMY = True
except ImportError:
    create_engine = None
    event = None
    Engine = None
    Connection = None
    Session = None
    sessionmaker = None
    declarative_base = None
    Pool = None
    CursorResult = None
    HAS_SQLALCHEMY = False

# Import SQLAlchemy integration conditionally
if HAS_SQLALCHEMY:
    from structured_logging.integrations.sqlalchemy import (
        DatabaseOperation,
        SQLAlchemyConnectionLogger,
        SQLAlchemyLoggingConfig,
        SQLAlchemyORMLogger,
        SQLAlchemyQueryLogger,
        SQLAlchemyTransactionLogger,
        get_query_logger,
        log_query,
        setup_sqlalchemy_logging,
    )

    # Create test model
    Base = declarative_base()

    class User(Base):
        __tablename__ = "users"
        id = Column(Integer, primary_key=True)
        name = Column(String(50))
        email = Column(String(100))

else:
    SQLAlchemyLoggingConfig = None
    SQLAlchemyQueryLogger = None
    Base = None
    User = None


@pytest.mark.skipif(not HAS_SQLALCHEMY, reason="SQLAlchemy not installed")
class TestSQLAlchemyLoggingConfig:
    """Test SQLAlchemy logging configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = SQLAlchemyLoggingConfig()

        assert config.log_queries is True
        assert config.log_query_parameters is False
        assert config.query_param_length_limit == 1000
        assert config.slow_query_threshold == 1.0

        assert config.log_connections is True
        assert config.log_connection_pool is True
        assert config.log_transactions is True

        assert config.log_orm_events is True
        assert config.log_flush_events is True
        assert config.log_bulk_operations is True

        assert "alembic_version" in config.excluded_tables
        assert config.only_errors is False

        assert config.query_field == "sql_query"
        assert config.duration_field == "query_duration_seconds"

        assert "password" in config.sensitive_param_names
        assert config.param_sanitization is True

    def test_custom_config(self):
        """Test custom configuration"""
        config = SQLAlchemyLoggingConfig(
            log_query_parameters=True,
            slow_query_threshold=0.5,
            excluded_tables={"custom_log_table"},
            only_errors=True,
            query_field="query",
            sensitive_param_names={"api_secret"},
        )

        assert config.log_query_parameters is True
        assert config.slow_query_threshold == 0.5
        assert config.excluded_tables == {"custom_log_table"}
        assert config.only_errors is True
        assert config.query_field == "query"
        assert config.sensitive_param_names == {"api_secret"}


@pytest.mark.skipif(not HAS_SQLALCHEMY, reason="SQLAlchemy not installed")
class TestSQLAlchemyQueryLogger:
    """Test query logging functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.config = SQLAlchemyLoggingConfig(
            log_query_parameters=True,
            include_row_count=True,
        )
        self.logger = SQLAlchemyQueryLogger(self.config)
        self.logger.logger = Mock()

    def test_query_execution_logging(self):
        """Test basic query execution logging"""
        # Mock connection and query
        mock_conn = Mock(spec=Connection)
        mock_conn.engine.url.drivername = "postgresql"
        mock_conn.engine.url.host = "localhost"
        mock_conn.engine.url.database = "testdb"

        mock_clause = Mock()
        mock_clause.__str__.return_value = "SELECT * FROM users WHERE id = :id"

        mock_result = Mock(spec=CursorResult)
        mock_result.rowcount = 1

        params = {"id": 123}

        # Execute before hook
        self.logger.before_execute(mock_conn, mock_clause, (), params, {})

        # Simulate some execution time
        time.sleep(0.1)

        # Execute after hook
        self.logger.after_execute(mock_conn, mock_clause, (), params, {}, mock_result)

        # Check logging
        self.logger.logger.debug.assert_called_once()
        call_args = self.logger.logger.debug.call_args
        assert "Query executed" in call_args[0][0]

        extra = call_args[1]["extra"]
        assert extra["sql_query"] == "SELECT * FROM users WHERE id = :id"
        assert extra["query_duration_seconds"] >= 0.1
        assert extra["row_count"] == 1
        assert extra["parameters"] == {"id": 123}
        assert extra["database"] == "postgresql://localhost/testdb"

    def test_slow_query_detection(self):
        """Test slow query warning"""
        self.config.slow_query_threshold = 0.05  # 50ms threshold

        mock_conn = Mock(spec=Connection)
        mock_clause = Mock()
        mock_clause.__str__.return_value = "SELECT COUNT(*) FROM large_table"

        mock_result = Mock(spec=CursorResult)
        mock_result.rowcount = 1000000

        # Execute hooks
        self.logger.before_execute(mock_conn, mock_clause, (), {}, {})
        time.sleep(0.1)  # Simulate slow query
        self.logger.after_execute(mock_conn, mock_clause, (), {}, {}, mock_result)

        # Should log as warning
        self.logger.logger.warning.assert_called_once()
        call_args = self.logger.logger.warning.call_args
        assert "Slow query detected" in call_args[0][0]
        extra = call_args[1]["extra"]
        assert extra["slow_query"] is True

    def test_query_error_handling(self):
        """Test query error logging"""
        mock_conn = Mock(spec=Connection)
        mock_clause = Mock()
        mock_clause.__str__.return_value = "INSERT INTO users (name) VALUES (:name)"

        params = {"name": "test"}
        exception = Exception("Constraint violation")

        # Execute before hook
        self.logger.before_execute(mock_conn, mock_clause, (), params, {})

        # Handle error
        exception_context = {
            "connection": mock_conn,
            "clauseelement": mock_clause,
            "exception": exception,
        }
        self.logger.handle_error(exception_context)

        # Check error logging
        self.logger.logger.error.assert_called_once()
        call_args = self.logger.logger.error.call_args
        assert "Query execution failed" in call_args[0][0]

        extra = call_args[1]["extra"]
        assert extra["sql_query"] == "INSERT INTO users (name) VALUES (:name)"
        assert extra["error_type"] == "Exception"
        assert extra["error_message"] == "Constraint violation"

    def test_parameter_sanitization(self):
        """Test sensitive parameter sanitization"""
        mock_conn = Mock(spec=Connection)
        mock_clause = Mock()

        params = {
            "username": "john",
            "password": "secret123",
            "api_key": "key123",
            "data": "normal_data",
        }

        # Execute hooks
        self.logger.before_execute(mock_conn, mock_clause, (), params, {})
        self.logger.after_execute(mock_conn, mock_clause, (), params, {}, Mock())

        # Check parameters were sanitized
        call_args = self.logger.logger.debug.call_args
        extra = call_args[1]["extra"]
        assert extra["parameters"]["username"] == "john"
        assert extra["parameters"]["password"] == "[REDACTED]"
        assert extra["parameters"]["api_key"] == "[REDACTED]"
        assert extra["parameters"]["data"] == "normal_data"

    def test_parameter_length_limit(self):
        """Test parameter truncation for long values"""
        self.config.query_param_length_limit = 50

        mock_conn = Mock(spec=Connection)
        mock_clause = Mock()

        long_value = "x" * 100
        params = {"data": long_value}

        # Execute hooks
        self.logger.before_execute(mock_conn, mock_clause, (), params, {})
        self.logger.after_execute(mock_conn, mock_clause, (), params, {}, Mock())

        # Check parameter was truncated
        call_args = self.logger.logger.debug.call_args
        extra = call_args[1]["extra"]
        assert len(extra["parameters"]["data"]) == 53  # 50 + "..."
        assert extra["parameters"]["data"].endswith("...")


@pytest.mark.skipif(not HAS_SQLALCHEMY, reason="SQLAlchemy not installed")
class TestSQLAlchemyConnectionLogger:
    """Test connection logging functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.config = SQLAlchemyLoggingConfig()
        self.logger = SQLAlchemyConnectionLogger(self.config)
        self.logger.logger = Mock()

    def test_connection_established(self):
        """Test connection establishment logging"""
        mock_dbapi_conn = Mock()
        mock_record = Mock()

        self.logger.on_connect(mock_dbapi_conn, mock_record)

        self.logger.logger.info.assert_called_once()
        call_args = self.logger.logger.info.call_args
        assert "Database connection established" in call_args[0][0]
        extra = call_args[1]["extra"]
        assert extra["event"] == "connection_established"
        assert "connection_id" in extra

    def test_pool_checkout(self):
        """Test connection pool checkout logging"""
        mock_dbapi_conn = Mock()
        mock_record = Mock()
        mock_proxy = Mock()

        self.logger.on_checkout(mock_dbapi_conn, mock_record, mock_proxy)

        self.logger.logger.debug.assert_called_once()
        call_args = self.logger.logger.debug.call_args
        assert "Connection checked out from pool" in call_args[0][0]
        extra = call_args[1]["extra"]
        assert extra["event"] == "pool_checkout"

    def test_pool_checkin(self):
        """Test connection pool checkin logging"""
        mock_dbapi_conn = Mock()
        mock_record = Mock()

        self.logger.on_checkin(mock_dbapi_conn, mock_record)

        self.logger.logger.debug.assert_called_once()
        call_args = self.logger.logger.debug.call_args
        assert "Connection returned to pool" in call_args[0][0]
        extra = call_args[1]["extra"]
        assert extra["event"] == "pool_checkin"

    def test_connection_closed(self):
        """Test connection close logging"""
        mock_dbapi_conn = Mock()
        mock_record = Mock()

        self.logger.on_close(mock_dbapi_conn, mock_record)

        self.logger.logger.info.assert_called_once()
        call_args = self.logger.logger.info.call_args
        assert "Database connection closed" in call_args[0][0]
        extra = call_args[1]["extra"]
        assert extra["event"] == "connection_closed"


@pytest.mark.skipif(not HAS_SQLALCHEMY, reason="SQLAlchemy not installed")
class TestSQLAlchemyTransactionLogger:
    """Test transaction logging functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.config = SQLAlchemyLoggingConfig()
        self.logger = SQLAlchemyTransactionLogger(self.config)
        self.logger.logger = Mock()

    def test_transaction_lifecycle(self):
        """Test transaction begin, commit, rollback logging"""
        # Mock session
        mock_session = Mock(spec=Session)
        mock_session.transaction = Mock()

        # Test begin
        self.logger.on_begin(mock_session)

        self.logger.logger.debug.assert_called_once()
        call_args = self.logger.logger.debug.call_args
        assert "Transaction started" in call_args[0][0]

        # Simulate some transaction time
        time.sleep(0.1)

        # Test commit
        self.logger.on_commit(mock_session)

        self.logger.logger.info.assert_called_once()
        call_args = self.logger.logger.info.call_args
        assert "Transaction committed" in call_args[0][0]
        extra = call_args[1]["extra"]
        assert extra["duration_seconds"] >= 0.1

        # Reset mocks for rollback test
        self.logger.logger.reset_mock()

        # Test rollback
        self.logger.on_begin(mock_session)
        time.sleep(0.05)
        self.logger.on_rollback(mock_session)

        self.logger.logger.warning.assert_called_once()
        call_args = self.logger.logger.warning.call_args
        assert "Transaction rolled back" in call_args[0][0]


@pytest.mark.skipif(not HAS_SQLALCHEMY, reason="SQLAlchemy not installed")
class TestSQLAlchemyORMLogger:
    """Test ORM event logging functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.config = SQLAlchemyLoggingConfig()
        self.logger = SQLAlchemyORMLogger(self.config)
        self.logger.logger = Mock()

    def test_flush_events(self):
        """Test session flush logging"""
        # Mock session
        mock_session = Mock(spec=Session)
        mock_session.new = [Mock(), Mock()]  # 2 new objects
        mock_session.dirty = [Mock()]  # 1 dirty object
        mock_session.deleted = []  # 0 deleted objects

        # Test before flush
        self.logger.before_flush(mock_session, None, None)

        self.logger.logger.debug.assert_called_once()
        call_args = self.logger.logger.debug.call_args
        assert "Session flush starting" in call_args[0][0]
        extra = call_args[1]["extra"]
        assert extra["new"] == 2
        assert extra["dirty"] == 1
        assert extra["deleted"] == 0

        # Simulate flush time
        time.sleep(0.05)

        # Test after flush
        self.logger.after_flush(mock_session, None)

        self.logger.logger.info.assert_called_once()
        call_args = self.logger.logger.info.call_args
        assert "Session flush completed" in call_args[0][0]
        extra = call_args[1]["extra"]
        assert extra["duration_seconds"] >= 0.05

    def test_bulk_operations(self):
        """Test bulk operation logging"""
        # Mock mapper
        mock_mapper = Mock()
        mock_mapper.class_.__tablename__ = "users"

        mock_connection = Mock()

        # Test bulk insert
        target = [Mock(), Mock(), Mock()]  # 3 objects
        self.logger.after_bulk_insert(mock_mapper, mock_connection, target)

        self.logger.logger.info.assert_called_once()
        call_args = self.logger.logger.info.call_args
        assert "Bulk insert completed" in call_args[0][0]
        extra = call_args[1]["extra"]
        assert extra["table"] == "users"
        assert extra["row_count"] == 3

        # Reset mock
        self.logger.logger.reset_mock()

        # Test bulk update
        self.logger.after_bulk_update(mock_mapper, mock_connection, None)

        call_args = self.logger.logger.info.call_args
        assert "Bulk update completed" in call_args[0][0]

        # Reset mock
        self.logger.logger.reset_mock()

        # Test bulk delete
        self.logger.after_bulk_delete(mock_mapper, mock_connection, None)

        call_args = self.logger.logger.info.call_args
        assert "Bulk delete completed" in call_args[0][0]


@pytest.mark.skipif(not HAS_SQLALCHEMY, reason="SQLAlchemy not installed")
class TestSQLAlchemyHelperFunctions:
    """Test helper functions and decorators"""

    def test_setup_sqlalchemy_logging(self):
        """Test engine setup with logging"""
        # Create in-memory SQLite engine
        engine = create_engine("sqlite:///:memory:")
        session_factory = sessionmaker(bind=engine)

        # Mock event.listen
        with patch(
            "structured_logging.integrations.sqlalchemy.event.listen"
        ) as mock_listen:
            setup_sqlalchemy_logging(engine, session_factory=session_factory)

        # Check that event listeners were registered
        assert mock_listen.call_count >= 10  # Multiple event types

        # Check event types registered
        registered_events = [call[0][1] for call in mock_listen.call_args_list]
        assert "before_execute" in registered_events
        assert "after_execute" in registered_events
        assert "handle_error" in registered_events
        assert "connect" in registered_events
        assert "after_begin" in registered_events
        assert "after_commit" in registered_events

    def test_log_query_decorator(self):
        """Test query logging decorator"""
        mock_logger = Mock()

        with patch(
            "structured_logging.integrations.sqlalchemy.get_logger",
            return_value=mock_logger,
        ):

            @log_query(operation="user_lookup", table="users")
            def get_user(user_id):
                return {"id": user_id, "name": "Test User"}

            # Execute decorated function
            result = get_user(123)

        assert result == {"id": 123, "name": "Test User"}
        assert mock_logger.debug.call_count == 1
        assert mock_logger.info.call_count == 1

        # Check start log
        start_call = mock_logger.debug.call_args
        assert "Database operation started" in start_call[0][0]
        extra = start_call[1]["extra"]
        assert extra["operation"] == "user_lookup"
        assert extra["table"] == "users"

        # Check completion log
        complete_call = mock_logger.info.call_args
        assert "Database operation completed" in complete_call[0][0]

    def test_log_query_decorator_with_error(self):
        """Test query logging decorator with exception"""
        mock_logger = Mock()

        with patch(
            "structured_logging.integrations.sqlalchemy.get_logger",
            return_value=mock_logger,
        ):

            @log_query(name="failing_query")
            def failing_query():
                raise ValueError("Query failed")

            with pytest.raises(ValueError):
                failing_query()

        # Check error was logged
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert "Database operation failed" in call_args[0][0]
        assert call_args[1]["exc_info"] is True

    def test_get_query_logger(self):
        """Test getting query logger"""
        logger = get_query_logger()
        assert logger is not None

        custom_logger = get_query_logger("myapp.db")
        assert custom_logger is not None


@pytest.mark.skipif(not HAS_SQLALCHEMY, reason="SQLAlchemy not installed")
class TestDatabaseOperation:
    """Test DatabaseOperation context manager"""

    def test_successful_operation(self):
        """Test successful database operation logging"""
        with patch(
            "structured_logging.integrations.sqlalchemy.get_logger"
        ) as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with DatabaseOperation("user_import", table="users") as op:
                # Simulate operation
                time.sleep(0.05)
                op.set_result({"imported": 10})

            # Check logging
            assert mock_logger.info.call_count == 2

            # Check start log
            start_call = mock_logger.info.call_args_list[0]
            assert "Database operation started" in start_call[0][0]
            extra = start_call[1]["extra"]
            assert extra["operation"] == "user_import"
            assert extra["status"] == "started"
            assert extra["table"] == "users"

            # Check completion log
            complete_call = mock_logger.info.call_args_list[1]
            assert "Database operation completed" in complete_call[0][0]
            extra = complete_call[1]["extra"]
            assert extra["status"] == "completed"
            assert extra["duration_seconds"] >= 0.05
            assert extra["result"] == {"imported": 10}

    def test_failed_operation(self):
        """Test failed database operation logging"""
        with patch(
            "structured_logging.integrations.sqlalchemy.get_logger"
        ) as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            try:
                with DatabaseOperation("data_migration", source="old_db"):
                    raise RuntimeError("Migration failed")
            except RuntimeError:
                pass

            # Check error logging
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args
            assert "Database operation failed" in call_args[0][0]
            extra = call_args[1]["extra"]
            assert extra["operation"] == "data_migration"
            assert extra["status"] == "failed"
            assert extra["error_type"] == "RuntimeError"
            assert extra["error_message"] == "Migration failed"


@pytest.mark.skipif(not HAS_SQLALCHEMY, reason="SQLAlchemy not installed")
class TestSQLAlchemyIntegration:
    """Integration tests with real SQLAlchemy engine"""

    def test_full_integration(self):
        """Test full integration with real engine and session"""
        # Create in-memory database
        engine = create_engine("sqlite:///:memory:")
        session_factory = sessionmaker(bind=engine)

        # Create tables
        Base.metadata.create_all(engine)

        # Setup logging
        config = SQLAlchemyLoggingConfig(
            log_query_parameters=True,
            log_flush_events=True,
        )

        # Mock the loggers
        with patch(
            "structured_logging.integrations.sqlalchemy.get_logger"
        ) as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            setup_sqlalchemy_logging(engine, config, session_factory)

            # Perform database operations
            session = session_factory()

            # Add user
            user = User(name="John Doe", email="john@example.com")
            session.add(user)
            session.commit()

            # Query user
            result = session.query(User).filter_by(name="John Doe").first()
            assert result is not None
            assert result.email == "john@example.com"

            session.close()

            # Basic assertions that logging occurred
            assert mock_get_logger.call_count >= 1


class TestSQLAlchemyWithoutDeps:
    """Test behavior when SQLAlchemy is not installed"""

    def test_import_without_sqlalchemy(self):
        """Test that modules can be imported without SQLAlchemy"""
        if HAS_SQLALCHEMY:
            pytest.skip("SQLAlchemy is installed")

        # Should be able to import the module
        from structured_logging.integrations import sqlalchemy

        assert not sqlalchemy.HAS_SQLALCHEMY

    def test_setup_without_sqlalchemy(self):
        """Test setup fails gracefully without SQLAlchemy"""
        if HAS_SQLALCHEMY:
            pytest.skip("SQLAlchemy is installed")

        from structured_logging.integrations.sqlalchemy import setup_sqlalchemy_logging

        with pytest.raises(ImportError) as exc_info:
            setup_sqlalchemy_logging(None)

        assert "sqlalchemy is required" in str(exc_info.value)
