"""
Tests for Celery integration
"""

import time
from unittest.mock import Mock, patch

import pytest

# Skip all tests if Celery is not installed
try:
    from celery import Celery, Task, current_task
    from celery.exceptions import Retry
    from celery.result import AsyncResult
    from celery.signals import (  # noqa: F401
        task_failure,
        task_postrun,
        task_prerun,
        task_retry,
        task_success,
    )

    HAS_CELERY = True
except ImportError:
    Celery = None
    Task = None
    current_task = None
    AsyncResult = None
    Retry = Exception
    HAS_CELERY = False

# Import Celery integration conditionally
if HAS_CELERY:
    from structured_logging.integrations.celery import (
        CeleryLoggingConfig,
        StructuredLoggingTask,
        _celery_logging_config,
        _task_failure_handler,
        _task_postrun_handler,
        _task_prerun_handler,
        _task_retry_handler,
        _task_revoked_handler,
        _task_success_handler,
        _worker_ready_handler,
        get_task_logger,
        log_task,
        log_task_chain,
        log_task_group,
        setup_celery_logging,
    )
else:
    CeleryLoggingConfig = None
    StructuredLoggingTask = None


@pytest.mark.skipif(not HAS_CELERY, reason="Celery not installed")
class TestCeleryLoggingConfig:
    """Test Celery logging configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = CeleryLoggingConfig()

        assert config.log_task_arguments is False
        assert config.log_task_kwargs is False
        assert config.log_task_result is False
        assert config.task_arg_max_length == 1000
        assert config.task_result_max_length == 1000

        assert config.include_task_runtime is True
        assert config.include_queue_time is True
        assert config.include_retry_count is True

        assert config.log_worker_events is True
        assert config.log_beat_events is True
        assert config.log_exceptions is True
        assert config.include_traceback is True

        assert "celery.chord_unlock" in config.excluded_tasks
        assert config.only_errors is False

        assert config.task_id_field == "task_id"
        assert config.correlation_id_field == "correlation_id"
        assert config.propagate_correlation_id is True

    def test_custom_config(self):
        """Test custom configuration"""
        config = CeleryLoggingConfig(
            log_task_arguments=True,
            log_task_result=True,
            task_arg_max_length=500,
            excluded_tasks={"custom.task"},
            only_errors=True,
            task_id_field="celery_task_id",
        )

        assert config.log_task_arguments is True
        assert config.log_task_result is True
        assert config.task_arg_max_length == 500
        assert config.excluded_tasks == {"custom.task"}
        assert config.only_errors is True
        assert config.task_id_field == "celery_task_id"


@pytest.mark.skipif(not HAS_CELERY, reason="Celery not installed")
class TestStructuredLoggingTask:
    """Test custom task class"""

    def test_task_class_creation(self):
        """Test task class initialization"""
        task = StructuredLoggingTask()

        assert task.logger is not None
        assert isinstance(task.config, CeleryLoggingConfig)

    @patch("structured_logging.integrations.celery.request_context")
    def test_task_execution_with_context(self, mock_context):
        """Test task execution with logging context"""
        # Create task instance
        task = StructuredLoggingTask()
        task.name = "test_task"

        # Mock request
        task.request = Mock()
        task.request.id = "task-123"
        task.request.routing_key = "default"
        task.request.headers = {"correlation_id": "corr-456"}
        task.request.correlation_id = None

        # Mock the parent __call__ method
        def mock_call(self, *args, **kwargs):
            return "task_result"

        with patch.object(Task, "__call__", mock_call):
            # Execute task
            task()

        # Check context was set up
        mock_context.assert_called_once_with(
            task_id="task-123",
            task_name="test_task",
            correlation_id="corr-456",
            queue="default",
        )

        # Check context manager was used
        mock_context.return_value.__enter__.assert_called_once()
        mock_context.return_value.__exit__.assert_called_once()


@pytest.mark.skipif(not HAS_CELERY, reason="Celery not installed")
class TestCelerySignalHandlers:
    """Test Celery signal handlers"""

    def setup_method(self):
        """Setup test fixtures"""
        # Setup mock logger and config
        _celery_logging_config.logger = Mock()
        _celery_logging_config.config = CeleryLoggingConfig()

    def test_task_prerun_handler(self):
        """Test task pre-run signal handler"""
        # Mock task and current_task
        mock_task = Mock()
        mock_task.name = "app.tasks.process_data"

        mock_current = Mock()
        mock_current.request = Mock()

        with patch("structured_logging.integrations.celery.current_task", mock_current):
            _task_prerun_handler(
                sender="worker-1",
                task_id="task-123",
                task=mock_task,
                args=["arg1", "arg2"],
                kwargs={"key": "value"},
            )

        # Check start time was stored
        assert hasattr(mock_current.request, "_start_time")

        # Check logging
        _celery_logging_config.logger.info.assert_called_once()
        call_args = _celery_logging_config.logger.info.call_args
        assert "Task started" in call_args[0][0]
        extra = call_args[1]["extra"]
        assert extra["event"] == "task_started"
        assert extra["task_id"] == "task-123"
        assert extra["task_name"] == "app.tasks.process_data"

    def test_task_prerun_with_arguments(self):
        """Test task pre-run with argument logging enabled"""
        _celery_logging_config.config.log_task_arguments = True
        _celery_logging_config.config.log_task_kwargs = True

        mock_task = Mock()
        mock_task.name = "test_task"

        _task_prerun_handler(
            task_id="task-123",
            task=mock_task,
            args=["arg1", "arg2"],
            kwargs={"key": "value"},
        )

        call_args = _celery_logging_config.logger.info.call_args
        extra = call_args[1]["extra"]
        assert "args" in extra
        assert extra["args"] == ["arg1", "arg2"]
        assert "kwargs" in extra
        assert extra["kwargs"] == {"key": "value"}

    def test_task_postrun_handler(self):
        """Test task post-run signal handler"""
        mock_task = Mock()
        mock_task.name = "app.tasks.process_data"

        mock_current = Mock()
        mock_current.request._start_time = time.time() - 2.5  # 2.5 seconds ago

        with patch("structured_logging.integrations.celery.current_task", mock_current):
            _task_postrun_handler(
                sender="worker-1",
                task_id="task-123",
                task=mock_task,
                retval="result",
                state="SUCCESS",
            )

        call_args = _celery_logging_config.logger.info.call_args
        assert "Task completed" in call_args[0][0]
        extra = call_args[1]["extra"]
        assert extra["event"] == "task_completed"
        assert extra["state"] == "SUCCESS"
        assert "runtime_seconds" in extra
        assert extra["runtime_seconds"] >= 2.5

    def test_task_postrun_with_result(self):
        """Test task post-run with result logging enabled"""
        _celery_logging_config.config.log_task_result = True

        mock_task = Mock()
        mock_task.name = "test_task"

        _task_postrun_handler(
            task_id="task-123",
            task=mock_task,
            retval={"status": "processed", "count": 42},
            state="SUCCESS",
        )

        call_args = _celery_logging_config.logger.info.call_args
        extra = call_args[1]["extra"]
        assert "result" in extra
        assert extra["result"] == {"status": "processed", "count": 42}

    def test_task_retry_handler(self):
        """Test task retry signal handler"""
        mock_sender = Mock()
        mock_sender.name = "retry_task"

        mock_current = Mock()
        mock_current.request.retries = 2

        with patch("structured_logging.integrations.celery.current_task", mock_current):
            _task_retry_handler(
                sender=mock_sender,
                task_id="task-123",
                reason="Connection timeout",
                einfo="Traceback...",
            )

        _celery_logging_config.logger.warning.assert_called_once()
        call_args = _celery_logging_config.logger.warning.call_args
        assert "Task retry" in call_args[0][0]
        extra = call_args[1]["extra"]
        assert extra["event"] == "task_retry"
        assert extra["reason"] == "Connection timeout"
        assert extra["retry_count"] == 2

    def test_task_failure_handler(self):
        """Test task failure signal handler"""
        mock_sender = Mock()
        mock_sender.name = "failing_task"

        exception = ValueError("Invalid input")

        _task_failure_handler(
            sender=mock_sender,
            task_id="task-123",
            exception=exception,
            einfo="Traceback...",
        )

        _celery_logging_config.logger.error.assert_called_once()
        call_args = _celery_logging_config.logger.error.call_args
        assert "Task failed" in call_args[0][0]
        extra = call_args[1]["extra"]
        assert extra["event"] == "task_failed"
        assert extra["exception_type"] == "ValueError"
        assert extra["exception_message"] == "Invalid input"

    def test_task_success_handler(self):
        """Test task success signal handler"""
        mock_sender = Mock()
        mock_sender.name = "successful_task"

        _task_success_handler(
            sender=mock_sender,
            result={"status": "completed"},
        )

        _celery_logging_config.logger.debug.assert_called_once()
        call_args = _celery_logging_config.logger.debug.call_args
        assert "Task succeeded" in call_args[0][0]

    def test_task_revoked_handler(self):
        """Test task revoked signal handler"""
        mock_request = Mock()
        mock_request.id = "task-123"
        mock_request.name = "revoked_task"

        _task_revoked_handler(
            request=mock_request,
            terminated=True,
            signum=15,
            expired=False,
        )

        _celery_logging_config.logger.warning.assert_called_once()
        call_args = _celery_logging_config.logger.warning.call_args
        assert "Task revoked" in call_args[0][0]
        extra = call_args[1]["extra"]
        assert extra["event"] == "task_revoked"
        assert extra["terminated"] is True
        assert extra["signal"] == 15

    def test_excluded_tasks(self):
        """Test excluded tasks are not logged"""
        _celery_logging_config.config.excluded_tasks = {"excluded_task"}

        mock_task = Mock()
        mock_task.name = "excluded_task"

        _task_prerun_handler(
            task_id="task-123",
            task=mock_task,
        )

        # Should not log excluded tasks
        _celery_logging_config.logger.info.assert_not_called()

    def test_only_errors_mode(self):
        """Test only_errors configuration"""
        _celery_logging_config.config.only_errors = True

        mock_task = Mock()
        mock_task.name = "test_task"

        # Successful task - should not log
        _task_postrun_handler(
            task_id="task-123",
            task=mock_task,
            state="SUCCESS",
        )

        _celery_logging_config.logger.info.assert_not_called()

        # Failed task - should log
        _task_postrun_handler(
            task_id="task-456",
            task=mock_task,
            state="FAILURE",
        )

        _celery_logging_config.logger.info.assert_called_once()


@pytest.mark.skipif(not HAS_CELERY, reason="Celery not installed")
class TestWorkerSignalHandlers:
    """Test worker signal handlers"""

    def setup_method(self):
        """Setup test fixtures"""
        _celery_logging_config.logger = Mock()
        _celery_logging_config.config = CeleryLoggingConfig()

    def test_worker_ready_handler(self):
        """Test worker ready signal handler"""
        mock_sender = Mock()
        mock_sender.hostname = "worker-1@host"

        _worker_ready_handler(sender=mock_sender)

        _celery_logging_config.logger.info.assert_called_once()
        call_args = _celery_logging_config.logger.info.call_args
        assert "Celery worker ready" in call_args[0][0]
        extra = call_args[1]["extra"]
        assert extra["event"] == "worker_ready"
        assert extra["hostname"] == "worker-1@host"


@pytest.mark.skipif(not HAS_CELERY, reason="Celery not installed")
class TestCeleryHelperFunctions:
    """Test Celery helper functions"""

    def test_setup_celery_logging(self):
        """Test Celery logging setup"""
        mock_app = Mock(spec=Celery)
        config = CeleryLoggingConfig(log_task_arguments=True)

        with patch(
            "structured_logging.integrations.celery._connect_signals"
        ) as mock_connect:
            setup_celery_logging(mock_app, config, logger_name="test_celery")

        # Check config was stored
        assert _celery_logging_config.config == config
        assert _celery_logging_config.logger is not None

        # Check signals were connected
        mock_connect.assert_called_once_with(config)

        # Check task class was set
        assert mock_app.Task == StructuredLoggingTask

    def test_get_task_logger(self):
        """Test getting task-specific logger"""
        # Without current task
        logger = get_task_logger("my_task")
        assert logger is not None

        # With current task
        mock_task = Mock()
        mock_task.name = "app.tasks.process"

        with patch("structured_logging.integrations.celery.current_task", mock_task):
            logger = get_task_logger()
            assert logger is not None

    def test_log_task_chain(self):
        """Test task chain logging"""
        with patch(
            "structured_logging.integrations.celery.get_logger"
        ) as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            log_task_chain("chain-123", ["task1", "task2", "task3"])

            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            assert "Task chain started" in call_args[0][0]
            extra = call_args[1]["extra"]
            assert extra["chain_id"] == "chain-123"
            assert extra["task_count"] == 3
            assert extra["tasks"] == ["task1", "task2", "task3"]

    def test_log_task_group(self):
        """Test task group logging"""
        with patch(
            "structured_logging.integrations.celery.get_logger"
        ) as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            log_task_group("group-456", ["task1", "task2"])

            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            assert "Task group started" in call_args[0][0]
            extra = call_args[1]["extra"]
            assert extra["group_id"] == "group-456"
            assert extra["task_count"] == 2


@pytest.mark.skipif(not HAS_CELERY, reason="Celery not installed")
class TestLogTaskDecorator:
    """Test log_task decorator"""

    def test_log_task_decorator_success(self):
        """Test task decorator with successful execution"""
        mock_logger = Mock()

        with patch(
            "structured_logging.integrations.celery.get_logger",
            return_value=mock_logger,
        ):
            with patch(
                "structured_logging.integrations.celery.current_task"
            ) as mock_task:
                mock_task.request.id = "task-123"

                @log_task(operation="test_op")
                def test_function(x, y):
                    return x + y

                result = test_function(2, 3)

        assert result == 5
        assert mock_logger.info.call_count == 2

        # Check start log
        start_call = mock_logger.info.call_args_list[0]
        assert "Task function started" in start_call[0][0]
        extra = start_call[1]["extra"]
        assert extra["task_function"] == "test_function"
        assert extra["operation"] == "test_op"

        # Check completion log
        complete_call = mock_logger.info.call_args_list[1]
        assert "Task function completed" in complete_call[0][0]

    def test_log_task_decorator_failure(self):
        """Test task decorator with exception"""
        mock_logger = Mock()

        with patch(
            "structured_logging.integrations.celery.get_logger",
            return_value=mock_logger,
        ):

            @log_task(name="failing_task")
            def failing_function():
                raise ValueError("Test error")

            with pytest.raises(ValueError):
                failing_function()

        # Check error was logged
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert "Task function failed" in call_args[0][0]
        assert call_args[1]["exc_info"] is True


@pytest.mark.skipif(not HAS_CELERY, reason="Celery not installed")
class TestDataTruncation:
    """Test data truncation for logging"""

    def setup_method(self):
        """Setup test fixtures"""
        from structured_logging.integrations.celery import _truncate_data

        self._truncate_data = _truncate_data

    def test_truncate_string(self):
        """Test string truncation"""
        # Short string - not truncated
        result = self._truncate_data("short", 100)
        assert result == "short"

        # Long string - truncated
        long_string = "x" * 200
        result = self._truncate_data(long_string, 100)
        assert len(result) == 103  # 100 + "..."
        assert result.endswith("...")

    def test_truncate_list(self):
        """Test list truncation"""
        # Small list - not truncated
        small_list = [1, 2, 3]
        result = self._truncate_data(small_list, 100)
        assert result == small_list

        # Large list - truncated
        large_list = list(range(1000))
        result = self._truncate_data(large_list, 100)
        assert result == "[list with 1000 items]"

    def test_truncate_dict(self):
        """Test dict truncation"""
        # Small dict - not truncated
        small_dict = {"a": 1, "b": 2}
        result = self._truncate_data(small_dict, 100)
        assert result == small_dict

        # Large dict - truncated
        large_dict = {f"key{i}": f"value{i}" for i in range(100)}
        result = self._truncate_data(large_dict, 100)
        assert result == "[Dict with 100 keys]"


class TestCeleryWithoutDeps:
    """Test behavior when Celery is not installed"""

    def test_import_without_celery(self):
        """Test that modules can be imported without Celery"""
        if HAS_CELERY:
            pytest.skip("Celery is installed")

        # Should be able to import the module
        from structured_logging.integrations import celery

        assert not celery.HAS_CELERY

    def test_setup_without_celery(self):
        """Test setup fails gracefully without Celery"""
        if HAS_CELERY:
            pytest.skip("Celery is installed")

        from structured_logging.integrations.celery import setup_celery_logging

        with pytest.raises(ImportError) as exc_info:
            setup_celery_logging()

        assert "celery is required" in str(exc_info.value)
