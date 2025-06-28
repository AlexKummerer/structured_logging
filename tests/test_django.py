"""
Tests for Django integration
"""

import json
import time
from unittest.mock import MagicMock, Mock, patch, call

import pytest

# Skip all tests if Django is not installed
try:
    import django
    from django.conf import settings
    from django.test import RequestFactory
    from django.http import HttpResponse
    from django.contrib.auth.models import User, AnonymousUser

    # Configure Django settings for tests
    if not settings.configured:
        settings.configure(
            DEBUG=True,
            SECRET_KEY="test-secret-key",
            INSTALLED_APPS=[
                "django.contrib.auth",
                "django.contrib.contenttypes",
                "django.contrib.sessions",
            ],
            MIDDLEWARE=[],
            DATABASES={
                "default": {
                    "ENGINE": "django.db.backends.sqlite3",
                    "NAME": ":memory:",
                }
            },
            USE_TZ=True,
        )
        django.setup()

    HAS_DJANGO = True
except ImportError:
    RequestFactory = None
    HttpResponse = None
    User = None
    AnonymousUser = None
    HAS_DJANGO = False

# Import Django integration conditionally
if HAS_DJANGO:
    from structured_logging.integrations.django import (
        DjangoLoggingConfig,
        DjangoStructuredLoggingMiddleware,
        DjangoLoggingCommand,
        setup_django_logging,
        get_django_logger,
    )
else:
    DjangoLoggingConfig = None
    DjangoStructuredLoggingMiddleware = None


@pytest.mark.skipif(not HAS_DJANGO, reason="Django not installed")
class TestDjangoLoggingConfig:
    """Test Django logging configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = DjangoLoggingConfig()

        assert config.log_request_body is False
        assert config.log_response_body is False
        assert config.log_request_headers is True
        assert config.log_response_headers is False
        assert config.request_body_max_size == 10000
        assert config.response_body_max_size == 10000

        assert "authorization" in config.sensitive_headers
        assert "cookie" in config.sensitive_headers
        assert config.header_prefix == "header_"

        assert config.include_user_info is True
        assert "id" in config.user_fields
        assert "username" in config.user_fields

        assert config.log_database_queries is True
        assert config.slow_query_threshold == 1.0
        assert config.log_exceptions is True

        assert "/health" in config.excluded_paths
        assert "/static/" in config.excluded_paths

    def test_custom_config(self):
        """Test custom configuration"""
        config = DjangoLoggingConfig(
            log_request_body=True,
            log_response_body=True,
            request_body_max_size=5000,
            sensitive_headers={"custom-secret"},
            header_prefix="hdr_",
            user_fields=["id", "email"],
            slow_query_threshold=0.5,
            excluded_paths={"/api/health"},
            only_errors=True,
        )

        assert config.log_request_body is True
        assert config.log_response_body is True
        assert config.request_body_max_size == 5000
        assert config.sensitive_headers == {"custom-secret"}
        assert config.header_prefix == "hdr_"
        assert config.user_fields == ["id", "email"]
        assert config.slow_query_threshold == 0.5
        assert config.excluded_paths == {"/api/health"}
        assert config.only_errors is True


@pytest.mark.skipif(not HAS_DJANGO, reason="Django not installed")
class TestDjangoStructuredLoggingMiddleware:
    """Test Django middleware functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.factory = RequestFactory()
        self.middleware = DjangoStructuredLoggingMiddleware()

        # Mock logger
        self.middleware.logger = Mock()

    def test_middleware_creation(self):
        """Test middleware creation and initialization"""
        middleware = DjangoStructuredLoggingMiddleware()

        assert middleware.config is not None
        assert middleware.logger is not None
        assert isinstance(middleware.config, DjangoLoggingConfig)

    def test_basic_request_response_logging(self):
        """Test basic request and response logging"""
        # Create request
        request = self.factory.get("/api/users", HTTP_USER_AGENT="test-agent")
        request._start_time = time.time()

        # Process request
        self.middleware.process_request(request)

        # Create response
        response = HttpResponse(status=200)

        # Process response
        self.middleware.process_response(request, response)

        # Check logging calls
        assert self.middleware.logger.info.call_count >= 1
        log_call = self.middleware.logger.info.call_args

        assert "Request completed" in log_call[0][0]
        extra = log_call[1]["extra"]
        assert extra["status_code"] == 200
        assert "response_time_seconds" in extra

    def test_excluded_paths(self):
        """Test path exclusion"""
        # Request to excluded path
        request = self.factory.get("/health")
        self.middleware.process_request(request)

        assert hasattr(request, "_skip_logging")
        assert request._skip_logging is True

        # Process response - should not log
        response = HttpResponse(status=200)
        self.middleware.process_response(request, response)

        # No logging should occur
        assert self.middleware.logger.info.call_count == 0

    def test_user_context_logging(self):
        """Test user information logging"""
        # Create request with user
        request = self.factory.get("/api/profile")
        request._start_time = time.time()

        # Mock user
        request.user = Mock()
        request.user.id = 123
        request.user.username = "testuser"
        request.user.email = "test@example.com"
        request.user.is_staff = False
        request.user.is_superuser = False

        # Process request
        self.middleware.process_request(request)

        # Process response
        response = HttpResponse(status=200)
        self.middleware.process_response(request, response)

        # Check user data was logged
        log_call = self.middleware.logger.info.call_args
        extra = log_call[1]["extra"]

        assert extra.get("user.id") == 123
        assert extra.get("user.username") == "testuser"
        assert extra.get("user.email") == "test@example.com"

    def test_anonymous_user_logging(self):
        """Test anonymous user logging"""
        request = self.factory.get("/api/public")
        request._start_time = time.time()
        request.user = AnonymousUser()

        self.middleware.process_request(request)

        response = HttpResponse(status=200)
        self.middleware.process_response(request, response)

        log_call = self.middleware.logger.info.call_args
        extra = log_call[1]["extra"]

        assert extra.get("user_id") is None
        assert extra.get("user_anonymous") is True

    def test_request_headers_logging(self):
        """Test request headers logging"""
        request = self.factory.get(
            "/api/data",
            HTTP_AUTHORIZATION="Bearer token123",
            HTTP_X_CUSTOM_HEADER="custom-value",
            HTTP_COOKIE="session=abc123",
        )
        request._start_time = time.time()

        self.middleware.process_request(request)

        response = HttpResponse(status=200)
        self.middleware.process_response(request, response)

        log_call = self.middleware.logger.info.call_args
        extra = log_call[1]["extra"]

        # Sensitive headers should be redacted
        assert extra.get("header_authorization") == "[REDACTED]"
        assert extra.get("header_cookie") == "[REDACTED]"

        # Non-sensitive headers should be included
        assert extra.get("header_x_custom_header") == "custom-value"

    def test_request_body_logging(self):
        """Test request body logging"""
        # Configure to log request body
        self.middleware.config.log_request_body = True

        # JSON request
        request = self.factory.post(
            "/api/users",
            data=json.dumps({"username": "newuser", "email": "new@example.com"}),
            content_type="application/json",
        )
        request._start_time = time.time()

        self.middleware.process_request(request)

        response = HttpResponse(status=201)
        self.middleware.process_response(request, response)

        log_call = self.middleware.logger.info.call_args
        extra = log_call[1]["extra"]

        assert "request_body" in extra
        assert extra["request_body"]["username"] == "newuser"
        assert extra["request_body"]["email"] == "new@example.com"

    def test_response_body_logging(self):
        """Test response body logging"""
        # Configure to log response body
        self.middleware.config.log_response_body = True

        request = self.factory.get("/api/users")
        request._start_time = time.time()

        self.middleware.process_request(request)

        # JSON response
        response_data = {"users": [{"id": 1, "name": "User 1"}]}
        response = HttpResponse(
            json.dumps(response_data), content_type="application/json", status=200
        )

        self.middleware.process_response(request, response)

        log_call = self.middleware.logger.info.call_args
        extra = log_call[1]["extra"]

        assert "response_body" in extra
        assert extra["response_body"]["users"][0]["id"] == 1

    def test_error_response_logging(self):
        """Test error response logging with appropriate log level"""
        request = self.factory.get("/api/not-found")
        request._start_time = time.time()

        self.middleware.process_request(request)

        # 404 response
        response = HttpResponse(status=404)
        self.middleware.process_response(request, response)

        # Should use warning level for 4xx
        assert self.middleware.logger.warning.called

        # 500 response
        request = self.factory.get("/api/error")
        request._start_time = time.time()

        self.middleware.process_request(request)

        response = HttpResponse(status=500)
        self.middleware.process_response(request, response)

        # Should use error level for 5xx
        assert self.middleware.logger.error.called

    def test_exception_handling(self):
        """Test exception logging"""
        request = self.factory.get("/api/crash")
        request._start_time = time.time()

        self.middleware.process_request(request)

        # Process exception
        exception = ValueError("Something went wrong")
        self.middleware.process_exception(request, exception)

        # Should log error with exception info
        assert self.middleware.logger.error.called
        log_call = self.middleware.logger.error.call_args

        assert "Unhandled exception" in log_call[0][0]
        extra = log_call[1]["extra"]
        assert extra["exception_type"] == "ValueError"
        assert extra["exception_message"] == "Something went wrong"
        assert log_call[1]["exc_info"] is True

    @patch("django.db.connection")
    def test_database_query_tracking(self, mock_connection):
        """Test database query tracking"""
        # Setup mock connection
        mock_connection.execute = Mock(return_value=None)
        mock_connection.vendor = "sqlite"
        mock_connection.alias = "default"

        request = self.factory.get("/api/data")
        request._start_time = time.time()

        # Reset query tracking
        self.middleware._reset_query_tracking(request)

        # Simulate queries
        request._queries_count = 3
        request._queries_time = 0.15
        request._slow_queries = []

        self.middleware.process_request(request)

        response = HttpResponse(status=200)
        self.middleware.process_response(request, response)

        log_call = self.middleware.logger.info.call_args
        extra = log_call[1]["extra"]

        assert extra.get("db_query_count") == 3
        assert extra.get("db_query_time_seconds") == 0.15

    def test_slow_query_detection(self):
        """Test slow query detection and logging"""
        request = self.factory.get("/api/slow")
        request._start_time = time.time()

        # Add slow query
        request._queries_count = 1
        request._queries_time = 2.5
        request._slow_queries = [{"sql": "SELECT * FROM large_table", "time": 2.5}]

        self.middleware.process_request(request)

        response = HttpResponse(status=200)
        self.middleware.process_response(request, response)

        log_call = self.middleware.logger.info.call_args
        extra = log_call[1]["extra"]

        assert "slow_queries" in extra
        assert extra["slow_query_count"] == 1
        assert extra["slow_queries"][0]["time"] == 2.5

    def test_only_errors_mode(self):
        """Test only_errors configuration"""
        self.middleware.config.only_errors = True

        # Successful request - should not log
        request = self.factory.get("/api/success")
        request._start_time = time.time()

        self.middleware.process_request(request)

        response = HttpResponse(status=200)
        self.middleware.process_response(request, response)

        # Should not log successful requests
        assert self.middleware.logger.info.call_count == 0

        # Error request - should log
        request = self.factory.get("/api/error")
        request._start_time = time.time()

        self.middleware.process_request(request)

        response = HttpResponse(status=500)
        self.middleware.process_response(request, response)

        # Should log error requests
        assert self.middleware.logger.error.call_count == 1

    def test_client_ip_extraction(self):
        """Test client IP extraction from various headers"""
        # X-Forwarded-For
        request = self.factory.get("/", HTTP_X_FORWARDED_FOR="192.168.1.1, 10.0.0.1")
        ip = self.middleware._get_client_ip(request)
        assert ip == "192.168.1.1"

        # REMOTE_ADDR fallback
        request = self.factory.get("/", REMOTE_ADDR="172.16.0.1")
        ip = self.middleware._get_client_ip(request)
        assert ip == "172.16.0.1"

    def test_request_id_generation(self):
        """Test unique request ID generation"""
        request = self.factory.get("/api/test")
        request._start_time = time.time()

        self.middleware.process_request(request)

        # Request should have context with request_id
        assert hasattr(request, "_logging_context")

        # Generate multiple IDs and ensure uniqueness
        ids = set()
        for _ in range(10):
            request_id = self.middleware._generate_request_id()
            assert request_id not in ids
            ids.add(request_id)


@pytest.mark.skipif(not HAS_DJANGO, reason="Django not installed")
class TestDjangoSignalHandlers:
    """Test Django signal integration"""

    def setup_method(self):
        """Setup test fixtures"""
        self.middleware = DjangoStructuredLoggingMiddleware()
        self.middleware.logger = Mock()

        # Configure to log auth events
        self.middleware.config.log_auth_events = True

    @patch("django.contrib.auth.signals.user_logged_in.connect")
    def test_signal_registration(self, mock_connect):
        """Test signal handler registration"""
        middleware = DjangoStructuredLoggingMiddleware()

        # Should connect to auth signals
        assert mock_connect.called

    def test_user_login_signal(self):
        """Test user login signal handling"""
        request = Mock()
        request.META = {"REMOTE_ADDR": "127.0.0.1"}

        user = Mock()
        user.id = 123
        user.username = "testuser"

        self.middleware._handle_user_login(None, request=request, user=user)

        assert self.middleware.logger.info.called
        log_call = self.middleware.logger.info.call_args

        assert "User logged in" in log_call[0][0]
        extra = log_call[1]["extra"]
        assert extra["event"] == "user_login"
        assert extra["user_id"] == 123
        assert extra["username"] == "testuser"

    def test_user_logout_signal(self):
        """Test user logout signal handling"""
        request = Mock()
        request.META = {"REMOTE_ADDR": "127.0.0.1"}

        user = Mock()
        user.id = 123
        user.username = "testuser"

        self.middleware._handle_user_logout(None, request=request, user=user)

        assert self.middleware.logger.info.called
        log_call = self.middleware.logger.info.call_args

        assert "User logged out" in log_call[0][0]
        extra = log_call[1]["extra"]
        assert extra["event"] == "user_logout"

    def test_login_failed_signal(self):
        """Test failed login signal handling"""
        request = Mock()
        request.META = {"REMOTE_ADDR": "127.0.0.1"}

        credentials = {"username": "baduser", "password": "wrongpass"}

        self.middleware._handle_login_failed(
            None, credentials=credentials, request=request
        )

        assert self.middleware.logger.warning.called
        log_call = self.middleware.logger.warning.call_args

        assert "Failed login attempt" in log_call[0][0]
        extra = log_call[1]["extra"]
        assert extra["event"] == "login_failed"
        assert extra["username"] == "baduser"


@pytest.mark.skipif(not HAS_DJANGO, reason="Django not installed")
class TestDjangoHelperFunctions:
    """Test Django helper functions"""

    def test_setup_django_logging(self):
        """Test Django logging setup function"""
        config = DjangoLoggingConfig(log_request_body=True)

        with patch("django.conf.settings") as mock_settings:
            setup_django_logging(logger_name="test_app", config=config)

            assert mock_settings.STRUCTURED_LOGGING_LOGGER_NAME == "test_app"
            assert mock_settings.STRUCTURED_LOGGING_CONFIG == config

    def test_get_django_logger(self):
        """Test getting Django logger"""
        with patch("django.conf.settings") as mock_settings:
            mock_settings.STRUCTURED_LOGGING_LOGGER_NAME = "custom_logger"

            logger = get_django_logger()
            assert logger is not None

            # Test with explicit name
            logger = get_django_logger("specific_logger")
            assert logger is not None


@pytest.mark.skipif(not HAS_DJANGO, reason="Django not installed")
class TestDjangoLoggingCommand:
    """Test Django management command base class"""

    def test_command_logging_methods(self):
        """Test command logging methods"""
        command = DjangoLoggingCommand()
        command.logger = Mock()
        command.stdout = Mock()
        command.stderr = Mock()
        command.style = Mock()
        command.style.WARNING = lambda x: f"WARNING: {x}"
        command.style.ERROR = lambda x: f"ERROR: {x}"
        command.style.SUCCESS = lambda x: f"SUCCESS: {x}"

        # Test log methods
        command.log_debug("Debug message", extra_field="value")
        assert command.logger.debug.called

        command.log_info("Info message")
        assert command.logger.info.called
        assert command.stdout.write.called

        command.log_warning("Warning message")
        assert command.logger.warning.called
        assert "WARNING:" in command.stdout.write.call_args[0][0]

        command.log_error("Error message")
        assert command.logger.error.called
        assert "ERROR:" in command.stderr.write.call_args[0][0]

        command.log_success("Success message")
        log_call = command.logger.info.call_args
        assert log_call[1]["extra"]["status"] == "success"
        assert "SUCCESS:" in command.stdout.write.call_args[0][0]


@pytest.mark.skipif(not HAS_DJANGO, reason="Django not installed")
class TestDjangoIntegration:
    """Integration tests for Django logging"""

    def test_full_request_cycle(self):
        """Test full request/response cycle with all features"""
        # Create middleware with full configuration
        config = DjangoLoggingConfig(
            log_request_body=True,
            log_response_body=True,
            log_request_headers=True,
            log_database_queries=True,
            include_user_info=True,
        )

        middleware = DjangoStructuredLoggingMiddleware()
        middleware.config = config
        middleware.logger = Mock()

        # Create request
        factory = RequestFactory()
        request = factory.post(
            "/api/users",
            data=json.dumps({"name": "Test User"}),
            content_type="application/json",
            HTTP_AUTHORIZATION="Bearer token123",
            HTTP_USER_AGENT="TestClient/1.0",
        )

        # Add user
        request.user = Mock()
        request.user.id = 1
        request.user.username = "admin"

        # Add session
        request.session = Mock()
        request.session.session_key = "test-session-123"

        # Process request
        middleware.process_request(request)

        # Create response
        response_data = {"id": 1, "name": "Test User", "created": True}
        response = HttpResponse(
            json.dumps(response_data), content_type="application/json", status=201
        )

        # Process response
        middleware.process_response(request, response)

        # Verify comprehensive logging
        assert middleware.logger.info.called
        log_call = middleware.logger.info.call_args
        extra = log_call[1]["extra"]

        # Check all expected fields
        assert extra["method"] == "POST"
        assert extra["path"] == "/api/users"
        assert extra["status_code"] == 201
        assert extra["user.id"] == 1
        assert extra["user.username"] == "admin"
        assert extra["session_id"] == "test-session-123"
        assert extra["header_authorization"] == "[REDACTED]"
        assert extra["header_user_agent"] == "TestClient/1.0"
        assert "request_body" in extra
        assert "response_body" in extra
        assert "response_time_seconds" in extra


class TestDjangoWithoutDeps:
    """Test behavior when Django is not installed"""

    def test_import_without_django(self):
        """Test that modules can be imported without Django"""
        if HAS_DJANGO:
            pytest.skip("Django is installed")

        # Should be able to import the module
        from structured_logging.integrations import django

        assert not django.HAS_DJANGO

    def test_middleware_creation_without_django(self):
        """Test middleware creation fails gracefully without Django"""
        if HAS_DJANGO:
            pytest.skip("Django is installed")

        from structured_logging.integrations.django import (
            DjangoStructuredLoggingMiddleware,
        )

        with pytest.raises(ImportError) as exc_info:
            DjangoStructuredLoggingMiddleware()

        assert "django is required" in str(exc_info.value)
