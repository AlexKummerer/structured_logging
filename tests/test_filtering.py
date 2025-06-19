"""
Tests for log filtering and sampling functionality
"""

import logging
import os
from unittest.mock import MagicMock, patch

from structured_logging.config import LoggerConfig
from structured_logging.filtering import (
    ContextFilter,
    CustomFilter,
    FilterConfig,
    FilterEngine,
    FilterResult,
    LevelFilter,
    SamplingFilter,
)
from structured_logging.logger import (
    get_filter_metrics,
    get_logger,
    log_with_context,
    reset_filter_metrics,
)


class TestFilterResult:
    def test_filter_result_creation(self):
        result = FilterResult(should_log=True, reason="test")
        assert result.should_log is True
        assert result.reason == "test"
        assert result.metadata == {}

        result_with_metadata = FilterResult(
            should_log=False, reason="filtered", metadata={"filter_type": "level"}
        )
        assert result_with_metadata.should_log is False
        assert result_with_metadata.metadata["filter_type"] == "level"


class TestLevelFilter:
    def setup_method(self):
        self.filter = LevelFilter(min_level=logging.WARNING)

    def test_level_filter_accepts_higher_level(self):
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="error message",
            args=(),
            exc_info=None,
        )
        result = self.filter.should_log(record, {})
        assert result.should_log is True

    def test_level_filter_rejects_lower_level(self):
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="info message",
            args=(),
            exc_info=None,
        )
        result = self.filter.should_log(record, {})
        assert result.should_log is False

    def test_level_filter_with_string_level(self):
        filter_str = LevelFilter(min_level="ERROR")
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="warning message",
            args=(),
            exc_info=None,
        )
        result = filter_str.should_log(record, {})
        assert result.should_log is False


class TestContextFilter:
    def test_required_keys_filter(self):
        filter_obj = ContextFilter(required_keys=["user_id", "request_id"])

        # Should pass with all required keys
        result = filter_obj.should_log(
            MagicMock(), {"user_id": "123", "request_id": "abc", "extra": "value"}
        )
        assert result.should_log is True

        # Should fail with missing required key
        result = filter_obj.should_log(
            MagicMock(), {"user_id": "123", "extra": "value"}
        )
        assert result.should_log is False
        assert "missing required keys" in result.reason

    def test_excluded_keys_filter(self):
        filter_obj = ContextFilter(excluded_keys=["debug_info", "internal"])

        # Should pass without excluded keys
        result = filter_obj.should_log(
            MagicMock(), {"user_id": "123", "request_id": "abc"}
        )
        assert result.should_log is True

        # Should fail with excluded key
        result = filter_obj.should_log(
            MagicMock(), {"user_id": "123", "debug_info": "secret"}
        )
        assert result.should_log is False
        assert "found excluded keys" in result.reason

    def test_key_value_filters(self):
        filter_obj = ContextFilter(
            key_value_filters={"env": "production", "service": "api"}
        )

        # Should pass with matching values
        result = filter_obj.should_log(
            MagicMock(), {"env": "production", "service": "api", "user_id": "123"}
        )
        assert result.should_log is True

        # Should fail with non-matching value
        result = filter_obj.should_log(
            MagicMock(), {"env": "development", "service": "api"}
        )
        assert result.should_log is False
        assert "env=development != production" in result.reason


class TestCustomFilter:
    def test_custom_filter_function(self):
        def only_errors(record, context):
            return record.levelno >= logging.ERROR

        filter_obj = CustomFilter(only_errors, name="error_only")

        # Should pass for error level
        error_record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="error",
            args=(),
            exc_info=None,
        )
        result = filter_obj.should_log(error_record, {})
        assert result.should_log is True

        # Should fail for info level
        info_record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="info",
            args=(),
            exc_info=None,
        )
        result = filter_obj.should_log(info_record, {})
        assert result.should_log is False

    def test_custom_filter_exception_handling(self):
        def failing_filter(record, context):
            raise ValueError("Test error")

        filter_obj = CustomFilter(failing_filter, name="failing")
        result = filter_obj.should_log(MagicMock(), {})

        # Should default to logging on error
        assert result.should_log is True
        assert "error" in result.reason
        assert "filter_error" in result.metadata


class TestSamplingFilter:
    def test_random_sampling(self):
        # Test with 0% sampling rate
        filter_zero = SamplingFilter(sample_rate=0.0, strategy="random")
        with patch("random.random", return_value=0.5):
            result = filter_zero.should_log(MagicMock(), {})
            assert result.should_log is False

        # Test with 100% sampling rate
        filter_full = SamplingFilter(sample_rate=1.0, strategy="random")
        with patch("random.random", return_value=0.5):
            result = filter_full.should_log(MagicMock(), {})
            assert result.should_log is True

    def test_hash_sampling(self):
        filter_obj = SamplingFilter(sample_rate=0.5, strategy="hash")

        # Test deterministic behavior
        context = {"request_id": "test-123"}
        record = MagicMock()

        result1 = filter_obj.should_log(record, context)
        result2 = filter_obj.should_log(record, context)

        # Same input should give same result
        assert result1.should_log == result2.should_log

    def test_level_based_sampling(self):
        filter_obj = SamplingFilter(sample_rate=0.5, strategy="level_based")

        # Critical should always pass
        critical_record = logging.LogRecord(
            name="test",
            level=logging.CRITICAL,
            pathname="",
            lineno=0,
            msg="critical",
            args=(),
            exc_info=None,
        )

        # Test multiple times to ensure critical always passes
        for _ in range(10):
            result = filter_obj.should_log(critical_record, {})
            assert result.should_log is True

    def test_rate_limiting_isolation(self):
        """Test rate limiting in isolation"""
        filter_obj = SamplingFilter(
            sample_rate=1.0, max_per_second=2, burst_allowance=0, strategy="random"
        )

        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )

        # Test rate limiting function directly
        result1 = filter_obj._apply_rate_limiting(record, {})
        assert result1.should_log is True
        assert "within limits" in result1.reason

        result2 = filter_obj._apply_rate_limiting(record, {})
        assert result2.should_log is True
        assert "within limits" in result2.reason

        # Third should be rate limited
        result3 = filter_obj._apply_rate_limiting(record, {})
        assert result3.should_log is False
        assert "exceeded" in result3.reason

    def test_burst_allowance_isolation(self):
        """Test burst allowance for error-level logs in isolation"""
        filter_obj = SamplingFilter(
            sample_rate=1.0,
            max_per_second=1,  # Very low limit
            burst_allowance=2,  # Allow burst for errors
            strategy="random",
        )

        error_record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="error message",
            args=(),
            exc_info=None,
        )

        # Test rate limiting function directly
        result1 = filter_obj._apply_rate_limiting(error_record, {})
        assert result1.should_log is True
        assert "within limits" in result1.reason

        # Second should be rate limited but pass due to burst allowance
        result2 = filter_obj._apply_rate_limiting(error_record, {})
        assert result2.should_log is True
        assert "burst allowance" in result2.reason
        assert result2.metadata.get("burst_used") is True

        # Third should also pass due to burst allowance
        result3 = filter_obj._apply_rate_limiting(error_record, {})
        assert result3.should_log is True
        assert "burst allowance" in result3.reason

        # Fourth should be blocked (exceeded normal + burst limit)
        result4 = filter_obj._apply_rate_limiting(error_record, {})
        assert result4.should_log is False
        assert "exceeded" in result4.reason


class TestFilterConfig:
    def test_production_config(self):
        config = FilterConfig.create_production_config(
            sample_rate=0.1, max_logs_per_second=500
        )

        assert config.enabled is True
        assert config.collect_metrics is True
        assert len(config.filters) == 2

        # Should have level filter and sampling filter
        filter_types = [type(f).__name__ for f in config.filters]
        assert "LevelFilter" in filter_types
        assert "SamplingFilter" in filter_types

    def test_debug_config(self):
        config = FilterConfig.create_debug_config()

        assert config.enabled is True
        assert config.collect_metrics is False
        assert len(config.filters) == 1
        assert isinstance(config.filters[0], LevelFilter)


class TestFilterEngine:
    def setup_method(self):
        self.config = FilterConfig(
            enabled=True,
            filters=[
                LevelFilter(min_level=logging.INFO),
                SamplingFilter(sample_rate=1.0, strategy="random"),
            ],
            collect_metrics=True,
        )
        self.engine = FilterEngine(self.config)

    def test_filter_engine_with_passing_filters(self):
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="info message",
            args=(),
            exc_info=None,
        )

        with patch("random.random", return_value=0.5):
            result = self.engine.should_log(record, {})
            assert result.should_log is True

    def test_filter_engine_with_failing_filter(self):
        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="",
            lineno=0,
            msg="debug message",
            args=(),
            exc_info=None,
        )

        result = self.engine.should_log(record, {})
        assert result.should_log is False

    def test_filter_engine_metrics(self):
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="info message",
            args=(),
            exc_info=None,
        )

        # Process some logs
        with patch("random.random", return_value=0.5):
            for _ in range(5):
                self.engine.should_log(record, {})

        metrics = self.engine.get_metrics()
        assert "summary" in metrics
        assert "filter_stats" in metrics
        assert "pass_rate" in metrics

        assert metrics["summary"]["total_evaluated"] == 5
        assert metrics["summary"]["passed_through"] == 5

    def test_filter_engine_disabled(self):
        disabled_config = FilterConfig(enabled=False)
        engine = FilterEngine(disabled_config)

        record = MagicMock()  # Any record should pass
        result = engine.should_log(record, {})

        assert result.should_log is True
        assert result.reason == "filtering_disabled"


class TestLoggerIntegration:
    def test_logger_with_filtering_config(self):
        filter_config = FilterConfig(
            enabled=True,
            filters=[LevelFilter(min_level=logging.WARNING)],
            collect_metrics=True,
        )

        config = LoggerConfig(
            log_level="DEBUG",  # Logger level is DEBUG
            filter_config=filter_config,  # But filter requires WARNING+
        )

        logger = get_logger("test_filtered", config)

        # INFO should be filtered out
        result = log_with_context(logger, "info", "This should be filtered", config)
        assert result is False  # Filtering prevented log

        # ERROR should pass through
        result = log_with_context(logger, "error", "This should pass", config)
        assert result is True  # Log was successful

    def test_filter_metrics_integration(self):
        filter_config = FilterConfig(
            enabled=True,
            filters=[LevelFilter(min_level=logging.INFO)],
            collect_metrics=True,
        )

        config = LoggerConfig(filter_config=filter_config)
        logger = get_logger("test_metrics", config)

        # Generate some logs
        log_with_context(logger, "debug", "Debug message", config)  # Should be filtered
        log_with_context(logger, "info", "Info message", config)  # Should pass
        log_with_context(logger, "error", "Error message", config)  # Should pass

        metrics = get_filter_metrics(config)
        assert metrics is not None
        assert metrics["summary"]["total_evaluated"] == 3
        assert metrics["summary"]["filtered_out"] == 1
        assert metrics["summary"]["passed_through"] == 2

        # Test metrics reset
        reset_filter_metrics(config)
        metrics_after_reset = get_filter_metrics(config)
        assert metrics_after_reset["summary"]["total_evaluated"] == 0

    def test_no_filtering_when_disabled(self):
        config = LoggerConfig(filter_config=None)  # No filtering

        metrics = get_filter_metrics(config)
        assert metrics is None

        logger = get_logger("test_no_filter", config)
        result = log_with_context(logger, "debug", "Debug message", config)
        assert result is True  # Should always pass without filtering


class TestEnvironmentConfiguration:
    def test_config_from_environment(self):
        with patch.dict(
            os.environ,
            {
                "STRUCTURED_LOG_FILTERING": "true",
                "STRUCTURED_LOG_SAMPLE_RATE": "0.5",
                "STRUCTURED_LOG_MAX_PER_SECOND": "100",
                "STRUCTURED_LOG_SAMPLING_STRATEGY": "hash",
            },
        ):
            config = LoggerConfig.from_env()

            assert config.filter_config is not None
            assert config.filter_config.enabled is True
            assert len(config.filter_config.filters) == 2

            # Check sampling filter configuration
            sampling_filter = None
            for f in config.filter_config.filters:
                if isinstance(f, SamplingFilter):
                    sampling_filter = f
                    break

            assert sampling_filter is not None
            assert sampling_filter.sample_rate == 0.5
            assert sampling_filter.max_per_second == 100
            assert sampling_filter.strategy == "hash"

    def test_config_filtering_disabled_from_env(self):
        with patch.dict(os.environ, {"STRUCTURED_LOG_FILTERING": "false"}):
            config = LoggerConfig.from_env()
            assert config.filter_config is None
