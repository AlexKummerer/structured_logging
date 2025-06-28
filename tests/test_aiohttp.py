"""
Tests for aiohttp integration
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Skip all tests if aiohttp is not installed
try:
    import aiohttp
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    HAS_AIOHTTP = True
except ImportError:
    aiohttp = None
    web = None
    TestClient = None
    TestServer = None
    HAS_AIOHTTP = False

# Import aiohttp integration conditionally
if HAS_AIOHTTP:
    from structured_logging.integrations.aiohttp import (
        AioHTTPLoggingConfig,
        WebSocketLogger,
        LoggingClientSession,
        aiohttp_structured_logging_middleware,
        setup_aiohttp_logging,
        log_route,
    )
else:
    AioHTTPLoggingConfig = None
    WebSocketLogger = None
    LoggingClientSession = None


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
class TestAioHTTPLoggingConfig:
    """Test aiohttp logging configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = AioHTTPLoggingConfig()

        assert config.log_request_body is False
        assert config.log_response_body is False
        assert config.log_request_headers is True
        assert config.log_response_headers is False
        assert config.request_body_max_size == 10000

        assert "authorization" in config.sensitive_headers
        assert "cookie" in config.sensitive_headers
        assert config.header_prefix == "header_"

        assert config.log_websocket_frames is False
        assert config.include_response_time is True
        assert config.log_exceptions is True

        assert "/health" in config.excluded_paths
        assert config.only_errors is False
        assert config.trace_id_header == "x-trace-id"

    def test_custom_config(self):
        """Test custom configuration"""
        config = AioHTTPLoggingConfig(
            log_request_body=True,
            log_response_body=True,
            sensitive_headers={"x-secret"},
            excluded_paths={"/api/internal"},
            only_errors=True,
            trace_id_header="x-request-id",
        )

        assert config.log_request_body is True
        assert config.log_response_body is True
        assert config.sensitive_headers == {"x-secret"}
        assert config.excluded_paths == {"/api/internal"}
        assert config.only_errors is True
        assert config.trace_id_header == "x-request-id"


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
class TestAioHTTPMiddleware:
    """Test aiohttp middleware functionality"""

    async def test_basic_request_response_logging(self):
        """Test basic request and response logging"""
        # Create app with middleware
        app = web.Application()

        # Mock logger
        mock_logger = AsyncMock()
        app["structured_logger"] = mock_logger
        app["structured_logging_config"] = AioHTTPLoggingConfig()

        # Add middleware
        app.middlewares.append(aiohttp_structured_logging_middleware)

        # Add test route
        async def handler(request):
            return web.json_response({"status": "ok"})

        app.router.add_get("/api/test", handler)

        # Create test client
        async with TestClient(TestServer(app)) as client:
            # Make request
            resp = await client.get("/api/test")
            assert resp.status == 200

            # Check logging
            assert mock_logger.info.called

            # Check request log
            request_call = mock_logger.info.call_args_list[0]
            assert "Request started" in request_call[0][0]

            # Check response log
            response_call = mock_logger.info.call_args_list[1]
            assert "Request completed" in response_call[0][0]
            extra = response_call[1]["extra"]
            assert extra["status_code"] == 200
            assert "response_time_seconds" in extra

    async def test_excluded_paths(self):
        """Test path exclusion"""
        app = web.Application()

        mock_logger = AsyncMock()
        app["structured_logger"] = mock_logger
        app["structured_logging_config"] = AioHTTPLoggingConfig()

        app.middlewares.append(aiohttp_structured_logging_middleware)

        async def handler(request):
            return web.Response(text="OK")

        app.router.add_get("/health", handler)
        app.router.add_get("/api/data", handler)

        async with TestClient(TestServer(app)) as client:
            # Request to excluded path
            resp = await client.get("/health")
            assert resp.status == 200
            assert not mock_logger.info.called

            # Request to non-excluded path
            resp = await client.get("/api/data")
            assert resp.status == 200
            assert mock_logger.info.called

    async def test_error_logging(self):
        """Test error response logging"""
        app = web.Application()

        mock_logger = AsyncMock()
        app["structured_logger"] = mock_logger
        app["structured_logging_config"] = AioHTTPLoggingConfig()

        app.middlewares.append(aiohttp_structured_logging_middleware)

        # Add routes returning different status codes
        async def not_found(request):
            return web.Response(status=404, text="Not Found")

        async def server_error(request):
            return web.Response(status=500, text="Server Error")

        app.router.add_get("/not-found", not_found)
        app.router.add_get("/error", server_error)

        async with TestClient(TestServer(app)) as client:
            # 404 response - should use warning level
            resp = await client.get("/not-found")
            assert resp.status == 404
            assert mock_logger.warning.called

            # 500 response - should use error level
            resp = await client.get("/error")
            assert resp.status == 500
            assert mock_logger.error.called

    async def test_exception_handling(self):
        """Test exception logging"""
        app = web.Application()

        mock_logger = AsyncMock()
        app["structured_logger"] = mock_logger
        config = AioHTTPLoggingConfig(log_exceptions=True, include_traceback=True)
        app["structured_logging_config"] = config

        app.middlewares.append(aiohttp_structured_logging_middleware)

        # Route that raises exception
        async def failing_handler(request):
            raise ValueError("Test exception")

        app.router.add_get("/fail", failing_handler)

        async with TestClient(TestServer(app)) as client:
            # Should raise exception
            with pytest.raises(ValueError):
                await client.get("/fail")

            # Should log error
            assert mock_logger.error.called
            call_args = mock_logger.error.call_args
            assert "Unhandled exception" in call_args[0][0]
            assert call_args[1]["exc_info"] is True

    async def test_http_exception_handling(self):
        """Test aiohttp HTTP exception logging"""
        app = web.Application()

        mock_logger = AsyncMock()
        app["structured_logger"] = mock_logger
        config = AioHTTPLoggingConfig(log_http_exceptions=True)
        app["structured_logging_config"] = config

        app.middlewares.append(aiohttp_structured_logging_middleware)

        # Route that raises HTTP exception
        async def unauthorized_handler(request):
            raise web.HTTPUnauthorized(reason="Invalid credentials")

        app.router.add_get("/unauthorized", unauthorized_handler)

        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/unauthorized")
            assert resp.status == 401

            # Should log warning
            assert mock_logger.warning.called
            call_args = mock_logger.warning.call_args
            assert "HTTP exception: 401" in call_args[0][0]

    async def test_request_body_logging(self):
        """Test request body logging"""
        app = web.Application()

        mock_logger = AsyncMock()
        app["structured_logger"] = mock_logger
        config = AioHTTPLoggingConfig(log_request_body=True)
        app["structured_logging_config"] = config

        app.middlewares.append(aiohttp_structured_logging_middleware)

        # Echo handler
        async def echo_handler(request):
            data = await request.json()
            return web.json_response(data)

        app.router.add_post("/echo", echo_handler)

        async with TestClient(TestServer(app)) as client:
            payload = {"message": "Hello, World!", "count": 42}
            resp = await client.post("/echo", json=payload)
            assert resp.status == 200

            # Check request body was logged
            request_call = mock_logger.info.call_args_list[0]
            extra = request_call[1]["extra"]
            assert "request_body" in extra
            assert extra["request_body"] == payload

    async def test_response_body_logging(self):
        """Test response body logging"""
        app = web.Application()

        mock_logger = AsyncMock()
        app["structured_logger"] = mock_logger
        config = AioHTTPLoggingConfig(log_response_body=True)
        app["structured_logging_config"] = config

        app.middlewares.append(aiohttp_structured_logging_middleware)

        # Handler returning JSON
        async def json_handler(request):
            return web.json_response({"result": "success", "value": 123})

        app.router.add_get("/json", json_handler)

        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/json")
            assert resp.status == 200

            # Check response body was logged
            response_call = mock_logger.info.call_args_list[1]
            extra = response_call[1]["extra"]
            assert "response_body" in extra
            assert extra["response_body"]["result"] == "success"

    async def test_header_logging(self):
        """Test header logging with sensitive header redaction"""
        app = web.Application()

        mock_logger = AsyncMock()
        app["structured_logger"] = mock_logger
        config = AioHTTPLoggingConfig(log_request_headers=True)
        app["structured_logging_config"] = config

        app.middlewares.append(aiohttp_structured_logging_middleware)

        async def handler(request):
            return web.Response(text="OK")

        app.router.add_get("/api/test", handler)

        async with TestClient(TestServer(app)) as client:
            # Request with headers
            headers = {
                "Authorization": "Bearer secret-token",
                "X-Custom-Header": "custom-value",
                "Cookie": "session=abc123",
            }
            resp = await client.get("/api/test", headers=headers)
            assert resp.status == 200

            # Check headers were logged
            request_call = mock_logger.info.call_args_list[0]
            extra = request_call[1]["extra"]
            assert "headers" in extra

            # Sensitive headers should be redacted
            assert extra["headers"]["authorization"] == "[REDACTED]"
            assert extra["headers"]["cookie"] == "[REDACTED]"

            # Non-sensitive headers should be included
            assert extra["headers"]["x_custom_header"] == "custom-value"

    async def test_trace_id_generation(self):
        """Test trace ID generation and propagation"""
        app = web.Application()

        mock_logger = AsyncMock()
        app["structured_logger"] = mock_logger
        config = AioHTTPLoggingConfig(generate_trace_id=True)
        app["structured_logging_config"] = config

        app.middlewares.append(aiohttp_structured_logging_middleware)

        async def handler(request):
            return web.Response(text="OK")

        app.router.add_get("/api/test", handler)

        async with TestClient(TestServer(app)) as client:
            # Request without trace ID
            resp = await client.get("/api/test")
            assert resp.status == 200

            # Should generate trace ID
            # (Check context was created with trace_id)

            # Request with trace ID
            headers = {"x-trace-id": "test-trace-123"}
            resp = await client.get("/api/test", headers=headers)
            assert resp.status == 200


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
class TestWebSocketLogger:
    """Test WebSocket logging functionality"""

    async def test_websocket_lifecycle_logging(self):
        """Test WebSocket connection lifecycle logging"""
        # Mock WebSocket and request
        mock_ws = Mock(spec=web.WebSocketResponse)
        mock_request = Mock()
        mock_request.path = "/ws/test"
        mock_request.remote = "127.0.0.1"
        mock_request.headers = {}

        # Create logger with mock
        logger = WebSocketLogger(mock_ws, mock_request)
        logger.logger = AsyncMock()

        # Test connection log
        await logger.log_connection()
        assert logger.logger.info.called
        call_args = logger.logger.info.call_args
        assert "WebSocket connected" in call_args[0][0]

        # Test message sent
        await logger.log_message_sent("Hello", "text")
        assert logger.frames_sent == 1
        assert logger.bytes_sent == 5

        # Test message received
        await logger.log_message_received("World", "text")
        assert logger.frames_received == 1
        assert logger.bytes_received == 5

        # Test error
        error = ValueError("Test error")
        await logger.log_error(error)
        assert logger.logger.error.called

        # Test close
        await logger.log_close(1000, "Normal closure")
        close_call = logger.logger.info.call_args
        assert "WebSocket closed" in close_call[0][0]
        extra = close_call[1]["extra"]
        assert extra["close_code"] == 1000
        assert extra["frames_sent"] == 1
        assert extra["frames_received"] == 1

    async def test_websocket_frame_logging(self):
        """Test WebSocket frame content logging"""
        mock_ws = Mock(spec=web.WebSocketResponse)
        mock_request = Mock()
        mock_request.path = "/ws/chat"
        mock_request.remote = "127.0.0.1"
        mock_request.headers = {}

        # Enable frame logging
        config = AioHTTPLoggingConfig(
            log_websocket_frames=True, websocket_frame_max_size=100
        )

        logger = WebSocketLogger(mock_ws, mock_request, config)
        logger.logger = AsyncMock()

        # Small message - should include preview
        await logger.log_message_sent("Short message", "text")
        call_args = logger.logger.debug.call_args
        extra = call_args[1]["extra"]
        assert "message_preview" in extra
        assert extra["message_preview"] == "Short message"

        # Large message - should be truncated
        large_message = "x" * 200
        await logger.log_message_sent(large_message, "text")
        call_args = logger.logger.debug.call_args
        extra = call_args[1]["extra"]
        assert len(extra["message_preview"]) <= 100


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
class TestLoggingClientSession:
    """Test client session logging"""

    async def test_client_request_logging(self):
        """Test outgoing client request logging"""
        # Mock session
        mock_session = AsyncMock()
        mock_response = Mock()
        mock_response.status = 200
        mock_response.content_length = 1024
        mock_session.request.return_value = mock_response

        # Create logging session
        logger = AsyncMock()
        client = LoggingClientSession(mock_session)
        client.logger = logger

        # Make request
        response = await client.request("GET", "https://api.example.com/data")

        # Check request was logged
        assert logger.info.call_count == 2  # Request and response

        # Check request log
        request_call = logger.info.call_args_list[0]
        assert "Client request: GET" in request_call[0][0]
        extra = request_call[1]["extra"]
        assert extra["client_method"] == "GET"
        assert extra["client_url"] == "https://api.example.com/data"

        # Check response log
        response_call = logger.info.call_args_list[1]
        assert "Client response: 200" in response_call[0][0]
        extra = response_call[1]["extra"]
        assert extra["client_status"] == 200
        assert "client_response_time_seconds" in extra

    async def test_client_error_logging(self):
        """Test client request error logging"""
        # Mock session that raises timeout
        mock_session = AsyncMock()
        mock_session.request.side_effect = asyncio.TimeoutError()

        logger = AsyncMock()
        client = LoggingClientSession(mock_session)
        client.logger = logger

        # Make request that times out
        with pytest.raises(asyncio.TimeoutError):
            await client.request("GET", "https://slow.example.com")

        # Check timeout was logged
        assert logger.error.called
        call_args = logger.error.call_args
        assert "Client request timeout" in call_args[0][0]


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
class TestRouteDecorator:
    """Test route logging decorator"""

    async def test_log_route_decorator(self):
        """Test route-level logging decorator"""
        mock_logger = AsyncMock()

        with patch(
            "structured_logging.integrations.aiohttp.get_async_logger",
            return_value=mock_logger,
        ):

            @log_route(name="test_operation", operation_type="read")
            async def test_handler(request):
                return web.json_response({"result": "ok"})

            # Create mock request
            mock_request = Mock()
            mock_request.method = "GET"
            mock_request.path = "/test"

            # Call decorated handler
            response = await test_handler(mock_request)

            # Check logging
            assert mock_logger.info.call_count == 2

            # Check start log
            start_call = mock_logger.info.call_args_list[0]
            assert "Route handler started" in start_call[0][0]
            extra = start_call[1]["extra"]
            assert extra["handler"] == "test_handler"
            assert extra["operation_type"] == "read"

            # Check completion log
            complete_call = mock_logger.info.call_args_list[1]
            assert "Route handler completed" in complete_call[0][0]

    async def test_log_route_exception(self):
        """Test route decorator with exception"""
        mock_logger = AsyncMock()

        with patch(
            "structured_logging.integrations.aiohttp.get_async_logger",
            return_value=mock_logger,
        ):

            @log_route(name="failing_operation")
            async def failing_handler(request):
                raise ValueError("Test error")

            mock_request = Mock()
            mock_request.method = "POST"
            mock_request.path = "/fail"

            # Should raise exception
            with pytest.raises(ValueError):
                await failing_handler(mock_request)

            # Should log error
            assert mock_logger.error.called
            call_args = mock_logger.error.call_args
            assert "Route handler failed" in call_args[0][0]
            assert call_args[1]["exc_info"] is True


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
class TestAioHTTPHelpers:
    """Test aiohttp helper functions"""

    async def test_setup_aiohttp_logging(self):
        """Test application setup helper"""
        app = web.Application()
        config = AioHTTPLoggingConfig(log_request_body=True)

        setup_aiohttp_logging(app, logger_name="test_app", config=config)

        # Check configuration was stored
        assert app["structured_logging_config"] == config
        assert app["structured_logger"] is not None

        # Check middleware was added
        assert len(app.middlewares) == 1


class TestAioHTTPWithoutDeps:
    """Test behavior when aiohttp is not installed"""

    def test_import_without_aiohttp(self):
        """Test that modules can be imported without aiohttp"""
        if HAS_AIOHTTP:
            pytest.skip("aiohttp is installed")

        # Should be able to import the module
        from structured_logging.integrations import aiohttp as aiohttp_integration

        assert not aiohttp_integration.HAS_AIOHTTP

    async def test_middleware_without_aiohttp(self):
        """Test middleware fails gracefully without aiohttp"""
        if HAS_AIOHTTP:
            pytest.skip("aiohttp is installed")

        from structured_logging.integrations.aiohttp import (
            aiohttp_structured_logging_middleware,
        )

        # Mock request and handler
        mock_request = Mock()
        mock_handler = AsyncMock()

        with pytest.raises(ImportError) as exc_info:
            await aiohttp_structured_logging_middleware(mock_request, mock_handler)

        assert "aiohttp is required" in str(exc_info.value)
